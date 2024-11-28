# src/models.py
#!pip -q install /kaggle/child-mind-institute-problematic-internet-use/models/pytorch_tabnet/pytorch_tabnet-4.1.0-py3-none-any.whl

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback
import torch

import wandb
from omegaconf import OmegaConf
from src.utils import quadratic_weighted_kappa, threshold_rounder, evaluate_predictions

# ハイパーパラメータの読み込み
config = OmegaConf.load('config/parameters.yaml')

# モデルのパラメータ設定
lgb_params = dict(config.lgb_params)
xgb_params = dict(config.xgb_params)
cat_params = dict(config.cat_params)
tabnet_params = dict(config.tabnet_params)

# wandb の設定
wandb.init(project='your_project_name', config={
    'lgb_params': lgb_params,
    'xgb_params': xgb_params,
    'cat_params': cat_params,
    'tabnet_params': tabnet_params,
    'n_splits': config.n_splits,
    'seed': config.seed
})

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
        self.model = self.trainer

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

def train_model(train, test, sample_submission):
    from sklearn.model_selection import train_test_split

    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    train_S = []
    test_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test), config.n_splits))


    # モデルのインスタンス化
    lgb_model = LGBMRegressor(**lgb_params)
    xgb_model = XGBRegressor(**xgb_params)
    cat_model = CatBoostRegressor(**cat_params)
    tabnet_model = TabNetWrapper(**tabnet_params)

    from sklearn.ensemble import VotingRegressor
    ensemble_model = VotingRegressor(estimators=[
        ('lightgbm', lgb_model),
        ('xgboost', xgb_model),
        ('catboost', cat_model),
        ('tabnet', tabnet_model)
    ])

    for fold, (train_idx, val_idx) in enumerate(SKF.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(ensemble_model)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[val_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[val_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)
        # 各 fold の結果を wandb に記録
        wandb.log({
            f'fold_{fold+1}_train_qwk': train_kappa,
            f'fold_{fold+1}_val_qwk': val_kappa
        })
        train_S.append(train_kappa)
        test_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test)

        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOptimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded),
                              method='Nelder-Mead')
    assert KappaOptimizer.success, "Optimization did not converge."

    oof_tuned = threshold_rounder(oof_non_rounded, KappaOptimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    # 平均の結果を記録
    wandb.log({
        'mean_train_qwk': np.mean(train_S),
        'mean_val_qwk': np.mean(test_S),
        'optimized_qwk': tKappa
    })

    # モデルの保存
    wandb.save('models/*.pkl')

    print(f"----> || Optimized QWK SCORE :: {tKappa:.3f}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_rounder(tpm, KappaOptimizer.x)

    submission = pd.DataFrame({
        'id': sample_submission['id'],
        'sii': tpTuned
    })

    return submission
