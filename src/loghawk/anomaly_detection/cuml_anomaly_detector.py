"""NVIDIA cuML GPU implementation of the LogHawk detector contract."""

from typing import Any

import cupy as cp
import joblib
import numpy as np
from cuml.ensemble import IsolationForest
from cuml.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector


class CuMLAnomalyDetector(AnomalyDetector):
    backend = "cuml"

    def __init__(self):
        self.scaler = None
        self.model = None

    def train(self, X: Any) -> None:
        # cuML accepts host NumPy arrays and transfers them to GPU memory.
        X = np.asarray(X, dtype=np.float32)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            output_type="numpy",
        )
        self.model.fit(X_scaled)

    def detect(self, X: Any):
        if self.model is None or self.scaler is None:
            raise RuntimeError("Train or load the detector before detect().")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float32))
        scores = -self.model.score_samples(X_scaled)
        flags = self.model.predict(X_scaled) == -1
        return (
            cp.asnumpy(scores) if isinstance(scores, cp.ndarray) else np.asarray(scores),
            cp.asnumpy(flags) if isinstance(flags, cp.ndarray) else np.asarray(flags, dtype=bool),
        )

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

    def load(self, path: Any) -> "CuMLAnomalyDetector":
        artifact = joblib.load(path)
        if (
            not isinstance(artifact, dict)
            or artifact.get("format") != "loghawk-anomaly-detector"
            or artifact.get("backend") != self.backend
        ):
            raise ValueError("Artifact is not a bundled cuML LogHawk detector.")
        self.scaler = artifact["scaler"]
        self.model = artifact["model"]
        return self
