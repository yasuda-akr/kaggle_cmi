# src/models/catboost_model.py
from catboost import CatBoostRegressor
import wandb
from .base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, params):
        self.model = CatBoostRegressor(**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False
        )
        # wandbに評価結果をログ
        evals_result = self.model.get_evals_result()
        for metric in evals_result['validation']:
            for i, value in enumerate(evals_result['validation'][metric]):
                wandb.log({f"CatBoost_{metric}": value, 'iteration': i})
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.get_feature_importance()
