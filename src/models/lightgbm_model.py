# src/models/lightgbm_model.py
import lightgbm as lgb
from wandb.integration.lightgbm import wandb_callback, log_summary
from .base import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, params):
        self.params = params.copy()  # パラメータのコピーを作成
        self.num_boost_round = self.params.pop('num_boost_round', 100)  # デフォルト100
        self.valid_sets = []
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # データセットの作成
        lgb_train = lgb.Dataset(X_train, label=y_train)
        if X_valid is not None and y_valid is not None:
            lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
            self.valid_sets = [lgb_train, lgb_valid]
        else:
            self.valid_sets = [lgb_train]

        # トレーニング
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=self.valid_sets,
            callbacks=[
                wandb_callback(),  # wandbのコールバックを追加
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ],
        )

        # log_summaryを呼び出してモデルのサマリーをwandbにログ（モデルの保存を行わない）
        log_summary(self.model, save_model_checkpoint=False)

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self, importance_type='split'):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.feature_importance(importance_type=importance_type)
