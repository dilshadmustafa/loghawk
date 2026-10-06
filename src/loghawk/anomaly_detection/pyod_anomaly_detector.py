"""CPU adapter for selected PyOD tabular outlier detectors."""

from importlib import import_module
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector
from loghawk.anomaly_detection.detector_capabilities import SKLEARN_CAPABILITIES


_PYOD_MODELS = {
    "pyod-isolationforest": ("pyod.models.iforest", "IForest"),
    "pyod-copod": ("pyod.models.copod", "COPOD"),
    "pyod-ecod": ("pyod.models.ecod", "ECOD"),
    "pyod-hbos": ("pyod.models.hbos", "HBOS"),
    "pyod-knn": ("pyod.models.knn", "KNN"),
    "pyod-lof": ("pyod.models.lof", "LOF"),
    "pyod-ocsvm": ("pyod.models.ocsvm", "OCSVM"),
    "pyod-pca": ("pyod.models.pca", "PCA"),
}


class PyODAnomalyDetector(AnomalyDetector):
    """Adapt PyOD's fit/decision_function/predict API to LogHawk's contract."""

    backend = "pyod"
    capabilities = SKLEARN_CAPABILITIES

    def __init__(self, algorithm: str, contamination: float = 0.05):
        if algorithm not in _PYOD_MODELS:
            raise ValueError(f"Unsupported PyOD algorithm: {algorithm!r}")
        self.algorithm = algorithm
        self.contamination = contamination
        self.scaler = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.model = None

    def train(self, X: Any) -> None:
        values = np.asarray(X, dtype=np.float64)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(values)
        self.scaler_mean = np.asarray(self.scaler.mean_)
        self.scaler_scale = np.asarray(self.scaler.scale_)

        module_name, class_name = _PYOD_MODELS[self.algorithm]
        model_class = getattr(import_module(module_name), class_name)
        self.model = model_class(contamination=self.contamination)
        self.model.fit(scaled)

    def detect(self, X: Any):
        if self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Train or load the detector before detect().")
        values = np.asarray(X, dtype=np.float64)
        scaled = (values - self.scaler_mean) / self.scaler_scale
        scores = np.asarray(self.model.decision_function(scaled), dtype=float)
        flags = np.asarray(self.model.predict(scaled)) == 1
        return scores, np.asarray(flags, dtype=bool)

    def save(self, path: Any) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained detector.")
        joblib.dump(self.model, path)

    def load(self, path: Any) -> "PyODAnomalyDetector":
        self.model = joblib.load(path)
        return self

    def export_scaler_state(self) -> dict[str, Any]:
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Cannot export an unfitted scaler.")
        return {"mean": self.scaler_mean, "scale": self.scaler_scale}

    def load_scaler_state(self, state: dict[str, Any]) -> None:
        self.scaler_mean = np.asarray(state["mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(state["scale"], dtype=np.float64)
        if np.any(self.scaler_scale == 0):
            raise ValueError("Scaler state contains a zero scale value.")
