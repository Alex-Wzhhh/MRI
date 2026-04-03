from __future__ import annotations

from sklearn.neural_network import MLPClassifier

from src.probes.base_probe import BaseProbe


class MLPProbe(BaseProbe):
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score_samples(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
