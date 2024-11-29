# src/models/xgboost_model.py
from xgboost import XGBRegressor
import wandb
from .base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, params):
        self.model = XGBRegressor(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        # wandbに評価結果をログ
        evals_result = self.model.evals_result()
        for metric in evals_result['validation_0']:
            for i, value in enumerate(evals_result['validation_0'][metric]):
                wandb.log({f"XGBoost_{metric}": value, 'iteration': i})
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.feature_importances_
