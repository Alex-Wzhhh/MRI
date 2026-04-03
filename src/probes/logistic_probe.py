from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.probes.base_probe import BaseProbe


class LogisticProbe(BaseProbe):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        return self.model.predict_proba(X)[:, 1]
