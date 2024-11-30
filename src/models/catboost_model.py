# src/models/catboost_model.py
from catboost import CatBoostRegressor
from wandb.integration.catboost import WandbCallback
from .base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self, params):
        self.params = params.copy()
        self.model = CatBoostRegressor(**self.params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # トレーニング
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)] if X_valid is not None and y_valid is not None else [],
            use_best_model=True,
            early_stopping_rounds=50,
            verbose=False,
            callbacks=[WandbCallback()]  
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.get_feature_importance()
