# scripts/train.py
import os
import sys
import warnings
import pandas as pd
import numpy as np

import wandb
from omegaconf import OmegaConf

# パスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, load_time_series, perform_autoencoder, impute_missing_values
from src.feature_engineering import feature_engineering
from src.models import LightGBMModel, XGBoostModel, CatBoostModel, TabNetModel
from src.utils import set_seeds, quadratic_weighted_kappa, threshold_rounder, evaluate_predictions
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
import joblib

warnings.filterwarnings('ignore')

def main():
    # wandb の設定ファイルを読み込み
    config = OmegaConf.load('config/parameters.yaml')

    # シードの設定
    set_seeds(config.seed)

    # wandb の初期化
    wandb.init(
        project=config.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=config.run_name,
        reinit=True
    )

    data_dir = 'data'
    train, test, sample_submission = load_data(data_dir)

    train_ts = load_time_series(os.path.join(data_dir, 'series_train.parquet'))
    test_ts = load_time_series(os.path.join(data_dir, 'series_test.parquet'))

    # 'id'列を削除
    df_train_ts = train_ts.drop('id', axis=1)
    df_test_ts = test_ts.drop('id', axis=1)

    # オートエンコーダーを適用
    from src.autoencoder import AutoEncoder
    train_ts_encoded = perform_autoencoder(df_train_ts, encoding_dim=60, epochs=100, batch_size=32)
    test_ts_encoded = perform_autoencoder(df_test_ts, encoding_dim=60, epochs=100, batch_size=32)

    # 'id'列を再度追加
    train_ts_encoded['id'] = train_ts['id']
    test_ts_encoded['id'] = test_ts['id']

    # 時系列データをメインデータセットとマージ
    train = pd.merge(train, train_ts_encoded, how="left", on='id')
    test = pd.merge(test, test_ts_encoded, how="left", on='id')

    # 欠損値の補完
    train = impute_missing_values(train)
    #test = impute_missing_values(test)

    # 特徴量エンジニアリング
    train = feature_engineering(train)
    test = feature_engineering(test)

    # 不要な列の削除
    train = train.drop(['id'], axis=1)
    test = test.drop(['id'], axis=1)

    # 列を揃える
    train = train[list(test.columns) + ['sii']]

    # 無限大の値をNaNに置換
    if np.any(np.isinf(train)):
        train = train.replace([np.inf, -np.inf], np.nan)

    # モデルのトレーニングと予測
    submission = train_and_predict(train, test, sample_submission, config)

    # 結果の保存
    submission.to_csv('submission.csv', index=False)
    wandb.save('submission.csv')
    print("トレーニングと予測が完了しました。結果は'submission.csv'に保存されました。")

def train_and_predict(train, test, sample_submission, config):
    # シードの設定
    set_seeds(config.seed)

    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    oof_preds = np.zeros(len(y), dtype=float)
    test_preds = np.zeros((len(test), config.n_splits))

    # モデルのパラメータ設定
    model_params = {
        'LightGBM': dict(config.lgb_params),
        'XGBoost': dict(config.xgb_params),
        'CatBoost': dict(config.cat_params),
        'TabNet': dict(config.tabnet_params)
    }

    # 使用するモデルの設定
    model_list = config.models  # 例: ['LightGBM', 'XGBoost', 'CatBoost', 'TabNet']

    # モデルのインスタンス化
    model_classes = {
        'LightGBM': LightGBMModel,
        'XGBoost': XGBoostModel,
        'CatBoost': CatBoostModel,
        'TabNet': TabNetModel
    }

    trained_models = []

    # 各フォールドでのトレーニング
    for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_oof_preds = np.zeros(len(val_idx), dtype=float)

        # 各モデルのトレーニング
        for model_name in model_list:
            print(f"Training fold {fold+1} with {model_name}")
            model = model_classes[model_name](model_params[model_name])

            # モデルのトレーニング
            model.fit(X_train, y_train, X_val, y_val)

            # 予測
            y_val_pred = model.predict(X_val)
            fold_oof_preds += y_val_pred / len(model_list)

            # 評価
            val_kappa = quadratic_weighted_kappa(y_val, np.round(y_val_pred).astype(int))

            # wandb にログ
            wandb.log({
                f'fold_{fold+1}_{model_name}_val_qwk': val_kappa
            })

            # テストデータの予測
            test_pred = model.predict(test)
            test_preds[:, fold] += test_pred / len(model_list)

            # モデルの保存を削除
            # model_filename = f'models/{model_name}_fold{fold+1}.pkl'
            # joblib.dump(model, model_filename)
            # wandb.save(model_filename)

            trained_models.append((model_name, fold, model))

            # 特徴量重要度のログ（対応するモデルのみ）
            if hasattr(model, 'get_feature_importance'):
                feature_importances = model.get_feature_importance()
                feature_names = X.columns
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importances
                }).sort_values(by='importance', ascending=False)
                fi_table = wandb.Table(dataframe=fi_df)
                wandb.log({f'fold_{fold+1}_{model_name}_feature_importance': fi_table})

            print(f"Fold {fold+1} - {model_name} Validation QWK: {val_kappa:.4f}")

        # フォールド終了時に oof を更新
        oof_preds[val_idx] = fold_oof_preds

    # QWKスコアの最適化
    KappaOptimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_preds),
                              method='Nelder-Mead')
    assert KappaOptimizer.success, "Optimization did not converge."

    oof_tuned = threshold_rounder(oof_preds, KappaOptimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    # 最適化されたスコアをログ
    wandb.log({
        'optimized_qwk': tKappa
    })

    print(f"----> || Optimized QWK SCORE :: {tKappa:.3f}")

    # テストデータの予測
    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_rounder(tpm, KappaOptimizer.x)

    # 提出ファイルの作成
    submission = pd.DataFrame({
        'id': sample_submission['id'],
        'sii': tpTuned.astype(int)
    })

    return submission

if __name__ == "__main__":
    main()
