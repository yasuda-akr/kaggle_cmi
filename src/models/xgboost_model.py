# src/models/xgboost_model.py
from xgboost import XGBRegressor
from wandb.integration.xgboost import WandbCallback
from .base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params):
        self.params = params.copy()
        self.model = XGBRegressor(**self.params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # トレーニング
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)] if X_valid is not None and y_valid is not None else [],
            eval_metric='rmse',
            callbacks=[WandbCallback(log_model=False)],  # log_model=False に設定
            early_stopping_rounds=50,
            verbose=False
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.feature_importances_
