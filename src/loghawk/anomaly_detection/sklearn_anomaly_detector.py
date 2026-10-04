"""scikit-learn CPU implementation of the LogHawk detector contract."""

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector


class SklearnAnomalyDetector(AnomalyDetector):
    backend = "sklearn"

    def __init__(self):
        self.scaler = None
        self.model = None

    def train(self, X: Any) -> None:
        X = np.asarray(X)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

    def detect(self, X: Any):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Train or load the detector before detect().")
        X_scaled = self.scaler.transform(np.asarray(X))
        scores = -self.model.score_samples(X_scaled)
        flags = self.model.predict(X_scaled) == -1
        return np.asarray(scores), np.asarray(flags, dtype=bool)

    def save(self, path: Any) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Cannot save an untrained detector.")
        joblib.dump(
            {
                "format": "loghawk-anomaly-detector",
                "format_version": 1,
                "backend": self.backend,
                "scaler": self.scaler,
                "model": self.model,
            },
            path,
        )

    def load(self, path: Any) -> "SklearnAnomalyDetector":
        artifact = joblib.load(path)
        if (
            not isinstance(artifact, dict)
            or artifact.get("format") != "loghawk-anomaly-detector"
            or artifact.get("backend") != self.backend
        ):
            raise ValueError("Artifact is not a bundled sklearn LogHawk detector.")
        self.scaler = artifact["scaler"]
        self.model = artifact["model"]
        return self

    def load_legacy(self, model: Any, scaler: Any) -> "SklearnAnomalyDetector":
        """Load the pre-abstraction Stage B model/scaler artifact pair."""
        self.model = model
        self.scaler = scaler
        return self
