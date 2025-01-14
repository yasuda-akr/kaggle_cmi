# src/data_preprocessing.py
import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
from joblib import Memory
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# キャッシュの設定
cachedir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'data_preprocessing_cache')
memory = Memory(cachedir, verbose=0)


@memory.cache
def load_data(data_dir):
    train = pl.read_csv(os.path.join(data_dir, 'train.csv')).to_pandas()
    test = pl.read_csv(os.path.join(data_dir, 'test.csv')).to_pandas()
    sample_submission = pl.read_csv(os.path.join(data_dir, 'sample_submission.csv')).to_pandas()
    wandb.log({'train_shape': train.shape, 'test_shape': test.shape})
    return train, test, sample_submission



@memory.cache
def process_file(filename, dirname):
    df = pl.read_parquet(os.path.join(dirname, filename, 'part-0.parquet')).to_pandas()
    df.drop('step', axis=1, inplace=True)
    if np.any(np.isinf(df)):
        df = df.replace([np.inf, -np.inf], np.nan)
    return df.describe().values.reshape(-1), filename.split('=')[1]


@memory.cache
def load_time_series(dirname):
    ids = os.listdir(dirname)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    stats, indexes = zip(*results)
    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df

def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.model import AutoEncoder  # モデル定義をインポート

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    data_tensor = torch.FloatTensor(df_scaled)
    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i: i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()
    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])
    return df_encoded

def impute_missing_values(train):
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
    imputed_data = imputer.fit_transform(train[numeric_cols])
    train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
    train_imputed['sii'] = train_imputed['sii'].round().astype(int)
    for col in train.columns:
        if col not in numeric_cols:
            train_imputed[col] = train[col]
    return train_imputed
