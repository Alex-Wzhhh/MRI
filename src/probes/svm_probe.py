from __future__ import annotations

from sklearn.svm import LinearSVC

from src.probes.base_probe import BaseProbe


class LinearSVMProbe(BaseProbe):
    def __init__(self, **kwargs):
        self.model = LinearSVC(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.decision_function(X)
