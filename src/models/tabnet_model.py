# src/models/tabnet_model.py
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback
import torch
import wandb
from .base import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class WandbTabNetCallback(Callback):
    """
    カスタムコールバッククラス。各エポック終了時に損失や評価指標をwandbにログします。
    """
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # エポック番号を1から開始
            logs = {k: v for k, v in logs.items()}
            logs['epoch'] = epoch + 1
            wandb.log(logs)

class TabNetModel(BaseModel):
    def __init__(self, params):
        self.params = params.copy()
        self.params['optimizer_fn'] = torch.optim.Adam
        self.params['scheduler_fn'] = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.params['device_name'] =  'cuda' if torch.cuda.is_available() else 'cpu'
     
        self.model = TabNetRegressor(**params)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        X_train = self.preprocess(X_train)
        X_valid = self.preprocess(X_valid, fit=False)
        y_train = y_train.values.reshape(-1, 1)
        y_valid = y_valid.values.reshape(-1, 1)
        eval_set = [(X_valid, y_valid)]


        history =self.model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=eval_set,
                eval_name=['valid'],
                eval_metric=['rmse'],
                max_epochs=1000,
                patience=50,
                batch_size=1024,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False,
                # コールバックでwandbにメトリクスをログ（必要に応じてカスタムコールバックを追加）
            )
        # カスタムコールバックの設定
        callbacks = [WandbTabNetCallback()]


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
