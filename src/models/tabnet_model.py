# src/models/tabnet_model.py
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback
import torch
import wandb
from .base import BaseModel
from src.utils import TabNetWandbCallback
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class TabNetModel(BaseModel):
    def __init__(self, params):
        self.model = TabNetRegressor(**params)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.best_model_path = 'models/best_tabnet_model.zip'

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        X_train = self.preprocess(X_train)
        X_valid = self.preprocess(X_valid, fit=False)
        y_train = y_train.values.reshape(-1, 1)
        y_valid = y_valid.values.reshape(-1, 1)

        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['rmse'],
            max_epochs=1000,
            patience=50,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=[TabNetWandbCallback()]
        )

        return self

    def predict(self, X):
        X = self.preprocess(X, fit=False)
        return self.model.predict(X).flatten()

    def preprocess(self, X, fit=True):
        if fit:
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def get_feature_importance(self):
        return self.model.feature_importances_
