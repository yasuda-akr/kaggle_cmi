# src/model.py

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
import lightgbm as lgb
from lightgbm import LGBMRegressor
from lightgbm.callback import CallbackEnv
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback
import torch
import joblib

import wandb
from omegaconf import OmegaConf
from src.utils import quadratic_weighted_kappa, threshold_rounder, evaluate_predictions, set_seeds

config = OmegaConf.load('config/parameters.yaml')

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, encoding_dim * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim * 3, encoding_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(encoding_dim * 2, encoding_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, input_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim * 2, input_dim * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim * 3, input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        from sklearn.impute import SimpleImputer
        self.imputer = SimpleImputer(strategy='median')
        self.best_model_path = 'models/best_tabnet_model.pt'

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        y = y.values if hasattr(y, 'values') else y

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imputed, y, test_size=0.2, random_state=config.seed
        )

        self.model.fit(
            X_train=X_train, y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            eval_name=['valid'], eval_metric=['mse'],
            max_epochs=500, patience=50, batch_size=1024,
            virtual_batch_size=128, num_workers=0, drop_last=False,
            callbacks=[TabNetPretrainedModelCheckpoint(
                filepath=self.best_model_path,
                monitor='valid_mse', mode='min',
                save_best_only=True, verbose=True
            )]
        )

        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)
        return self

    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()

class TabNetPretrainedModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_train_begin(self, logs=None):
        pass  # self.model はすでにアクセス可能

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}')
            self.best = current
            if self.save_best_only:
                self.model.save_model(self.filepath)

def lgb_wandb_callback():
    def _callback(env: CallbackEnv):
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            wandb.log({f"{data_name}_{eval_name}": result, 'iteration': env.iteration})
    return _callback

class WandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for metric, value in logs.items():
                wandb.log({f"TabNet_{metric}": value, 'epoch': epoch})

def train_model(train, test, sample_submission, config):
    # シードの設定
    set_seeds(config.seed)

    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    train_S = []
    val_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    test_preds = np.zeros((len(test), config.n_splits))

    # モデルのパラメータ設定
    lgb_params = dict(config.lgb_params)
    xgb_params = dict(config.xgb_params)
    cat_params = dict(config.cat_params)
    tabnet_params = dict(config.tabnet_params)

    # 各モデルのインスタンス化
    lgb_model = LGBMRegressor(**lgb_params)
    xgb_model = XGBRegressor(**xgb_params)
    cat_model = CatBoostRegressor(**cat_params)
    tabnet_model = TabNetWrapper(**tabnet_params)

    # 各モデルをリスト化
    models = [
        ('LightGBM', lgb_model),
        ('XGBoost', xgb_model),
        ('CatBoost', cat_model),
        ('TabNet', tabnet_model)
    ]

    trained_models = []

    # 各フォールドでのトレーニング
    for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_oof_preds = np.zeros(len(val_idx), dtype=float)

        # 各モデルのトレーニング
        for name, model in models:
            # モデルのクローン
            cloned_model = clone(model)
            if name == 'LightGBM':
                # LightGBM のトレーニングと学習経過の記録
                cloned_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=True),
                        lgb_wandb_callback()
                        ]
                )
            elif name == 'XGBoost':
                # XGBoost のトレーニングと学習経過の記録
                cloned_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='rmse',
                    early_stopping_rounds=50,
                    verbose=False
                )
                # 評価結果のログ
                evals_result = cloned_model.evals_result()
                for metric in evals_result['validation_0']:
                    for i, value in enumerate(evals_result['validation_0'][metric]):
                        wandb.log({f"fold_{fold+1}_XGBoost_{metric}": value, 'iteration': i})
            elif name == 'CatBoost':
                # CatBoost のトレーニングと学習経過の記録
                cloned_model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=50,
                    verbose=False
                )
                # 評価結果のログ
                evals_result = cloned_model.get_evals_result()
                for metric in evals_result['validation']:
                    for i, value in enumerate(evals_result['validation'][metric]):
                        wandb.log({f"fold_{fold+1}_CatBoost_{metric}": value, 'iteration': i})
            elif name == 'TabNet':
                # TabNet のトレーニングと学習経過の記録
                cloned_model.fit(
                    X_train.values, y_train.values.reshape(-1, 1),
                    eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
                    eval_metric=['rmse'],
                    max_epochs=1000,
                    patience=50,
                    batch_size=1024,
                    virtual_batch_size=128,
                    num_workers=0,
                    drop_last=False,
                    callbacks=[WandbCallback()]
                )
            else:
                # デフォルトのトレーニング
                cloned_model.fit(X_train, y_train)

            # 予測
            y_train_pred = cloned_model.predict(X_train)
            y_val_pred = cloned_model.predict(X_val)

            fold_oof_preds += y_val_pred / len(models)

            # 評価
            train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
            val_kappa = quadratic_weighted_kappa(y_val, y_val_pred.round(0).astype(int))

            # wandb にログ
            wandb.log({
                f'fold_{fold+1}_{name}_train_qwk': train_kappa,
                f'fold_{fold+1}_{name}_val_qwk': val_kappa
            })

            # フォールドごとのスコアを保存
            train_S.append(train_kappa)
            val_S.append(val_kappa)

            # アンサンブルの予測
            test_preds[:, fold] += cloned_model.predict(test) / len(models)

            # モデルの保存
            model_filename = f'models/{name}_fold{fold+1}.pkl'
            with open(model_filename, 'wb') as f:
                joblib.dump(cloned_model, f)
            wandb.save(model_filename)

            trained_models.append((name, fold, cloned_model))

            # モデルの特徴量重要度をログ（LightGBM、XGBoost、CatBoost）
            if name in ['LightGBM', 'XGBoost', 'CatBoost']:
                if hasattr(cloned_model, 'feature_importances_'):
                    feature_importances = cloned_model.feature_importances_
                    feature_names = X.columns
                    fi_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': feature_importances
                    }).sort_values(by='importance', ascending=False)
                    fi_table = wandb.Table(dataframe=fi_df)
                    wandb.log({f'fold_{fold+1}_{name}_feature_importance': fi_table})
            elif name == 'TabNet':
                # TabNetの特徴量重要度（省略）
                pass

            print(f"Fold {fold+1} - {name} Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")

        # フォールド終了時に oof を更新
        oof_non_rounded[val_idx] = fold_oof_preds

    # 平均スコアの計算
    mean_train_qwk = np.mean(train_S)
    mean_val_qwk = np.mean(val_S)

    # wandb にログ
    wandb.log({
        'mean_train_qwk': mean_train_qwk,
        'mean_val_qwk': mean_val_qwk
    })

    print(f"Mean Train QWK --> {mean_train_qwk:.4f}")
    print(f"Mean Validation QWK ---> {mean_val_qwk:.4f}")

    # QWKスコアの最適化
    KappaOptimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded),
                              method='Nelder-Mead')
    assert KappaOptimizer.success, "Optimization did not converge."

    oof_tuned = threshold_rounder(oof_non_rounded, KappaOptimizer.x)
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
        'sii': tpTuned
    })

    # 提出ファイルをアーティファクトとして保存
    submission_filename = 'submission.csv'
    submission.to_csv(submission_filename, index=False)
    wandb.save(submission_filename)
    return submission
