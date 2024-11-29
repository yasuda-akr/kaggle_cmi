# src/models/base.py
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin

class BaseModel(ABC, BaseEstimator, RegressorMixin):
    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass
