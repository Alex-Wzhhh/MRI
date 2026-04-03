from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X):
        raise NotImplementedError
