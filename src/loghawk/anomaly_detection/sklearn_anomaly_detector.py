"""scikit-learn CPU implementation of the LogHawk detector contract."""

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector
from loghawk.anomaly_detection.detector_capabilities import SKLEARN_CAPABILITIES


class SklearnAnomalyDetector(AnomalyDetector):
    backend = "sklearn"
    algorithm = "sklearn-isolationforest"
    capabilities = SKLEARN_CAPABILITIES

    def __init__(self):
        self.scaler = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.model = None

    def train(self, X: Any) -> None:
        X = np.asarray(X, dtype=np.float64)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.scaler_mean = np.asarray(self.scaler.mean_)
        self.scaler_scale = np.asarray(self.scaler.scale_)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

    def detect(self, X: Any):
        if self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Train or load the detector before detect().")
        X_scaled = (np.asarray(X, dtype=np.float64) - self.scaler_mean) / self.scaler_scale
        scores = -self.model.score_samples(X_scaled)
        flags = self.model.predict(X_scaled) == -1
        return np.asarray(scores), np.asarray(flags, dtype=bool)

    def save(self, path: Any) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained detector.")
        joblib.dump(self.model, path)

    def load(self, path: Any) -> "SklearnAnomalyDetector":
        self.model = joblib.load(path)
        return self

    def export_scaler_state(self) -> dict[str, Any]:
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Cannot export an unfitted scaler.")
        return {
            "mean": np.asarray(self.scaler_mean),
            "scale": np.asarray(self.scaler_scale),
        }

    def load_scaler_state(self, state: dict[str, Any]) -> None:
        self.scaler_mean = np.asarray(state["mean"])
        self.scaler_scale = np.asarray(state["scale"])
        if np.any(self.scaler_scale == 0):
            raise ValueError("Scaler state contains a zero scale value.")

    def load_legacy(self, model: Any, scaler: Any) -> "SklearnAnomalyDetector":
        """Load the pre-abstraction Stage B model/scaler artifact pair."""
        self.model = model
        self.scaler = scaler
        self.scaler_mean = np.asarray(scaler.mean_)
        self.scaler_scale = np.asarray(scaler.scale_)
        return self
