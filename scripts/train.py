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
from src.model import train_model

warnings.filterwarnings('ignore')


np.random.seed(42)

def main():
    # wandb の設定ファイルを読み込み
    config = OmegaConf.load('config/config.yaml')
    # wandb の初期化
    wandb.init(project=config.project_name, entity=config.entity)

    data_dir = 'data'
    train, test, sample_submission = load_data(data_dir)

    train_ts = load_time_series(os.path.join(data_dir, 'series_train.parquet'))
    test_ts = load_time_series(os.path.join(data_dir, 'series_test.parquet'))

    # 'id'列を削除
    df_train_ts = train_ts.drop('id', axis=1)
    df_test_ts = test_ts.drop('id', axis=1)

    # オートエンコーダーを適用
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

    # 特徴量エンジニアリング
    train = feature_engineering(train)
    test = feature_engineering(test)

    # 不要な列の削除
    train = train.drop(['id'], axis=1)
    test = test.drop(['id'], axis=1)

    train = train[list(test.columns) + ['sii']]

    # 無限大の値をNaNに置換
    if np.any(np.isinf(train)):
        train = train.replace([np.inf, -np.inf], np.nan)

    # モデルのトレーニングと予測
    submission = train_model(train, test, sample_submission)

    # 結果の保存
    submission.to_csv('submission.csv', index=False)
    wandb.save('submission.csv')
    print("トレーニングと予測が完了しました。結果は'submission.csv'に保存されました。")

if __name__ == "__main__":
    main()
