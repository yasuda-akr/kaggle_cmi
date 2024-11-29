# src/models/lightgbm_model.py
import lightgbm as lgb
from lightgbm import LGBMRegressor
import wandb
from .base import BaseModel
from src.utils import lgb_wandb_callback

class LightGBMModel(BaseModel):
    def __init__(self, params):
        self.model = LGBMRegressor(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb_wandb_callback()
            ],
            verbose=False
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.feature_importances_
