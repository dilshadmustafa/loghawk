"""CPU adapter for selected PyOD tabular outlier detectors."""

from importlib import import_module
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector
from loghawk.anomaly_detection.detector_capabilities import (
    DetectorCapabilities,
    SKLEARN_CAPABILITIES,
)


_PYOD_MODELS = {
    "pyod-isolationforest": ("pyod.models.iforest", "IForest"),
    "pyod-copod": ("pyod.models.copod", "COPOD"),
    "pyod-ecod": ("pyod.models.ecod", "ECOD"),
    "pyod-hbos": ("pyod.models.hbos", "HBOS"),
    "pyod-knn": ("pyod.models.knn", "KNN"),
    "pyod-lof": ("pyod.models.lof", "LOF"),
    "pyod-ocsvm": ("pyod.models.ocsvm", "OCSVM"),
    "pyod-pca": ("pyod.models.pca", "PCA"),
    "pyod-autoencoder": ("pyod.models.auto_encoder", "AutoEncoder"),
    "pyod-autoencoder-gpu": ("pyod.models.auto_encoder", "AutoEncoder"),
}
_AUTOENCODER_ALGORITHMS = {
    "pyod-autoencoder",
    "pyod-autoencoder-gpu",
}


class PyODAnomalyDetector(AnomalyDetector):
    """Adapt PyOD's fit/decision_function/predict API to LogHawk's contract."""

    backend = "pyod"
    capabilities = SKLEARN_CAPABILITIES

    def __init__(
        self,
        algorithm: str,
        contamination: float = 0.05,
        device: str = "cpu",
    ):
        if algorithm not in _PYOD_MODELS:
            raise ValueError(f"Unsupported PyOD algorithm: {algorithm!r}")
        expected_device = "gpu" if algorithm.endswith("-gpu") else "cpu"
        if device != expected_device:
            raise ValueError(
                f"{algorithm} requires device={expected_device!r}; "
                f"got {device!r}."
            )
        if device not in {"cpu", "gpu"}:
            raise ValueError("PyOD device must be 'cpu' or 'gpu'.")
        if device == "gpu":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "pyod-autoencoder-gpu requires PyTorch. Install a "
                    "CUDA-enabled PyTorch build in the worker environment."
                ) from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "pyod-autoencoder-gpu requires torch.cuda.is_available() "
                    "to be True; refusing to fall back to CPU."
                )
        self.algorithm = algorithm
        self.contamination = contamination
        self.device = device
        self.capabilities = DetectorCapabilities(
            fit_devices=(device,),
            score_devices=(device,),
        )
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
        model_options = {"contamination": self.contamination}
        if self.algorithm in _AUTOENCODER_ALGORITHMS:
            model_options["device"] = (
                "cuda" if self.device == "gpu" else "cpu"
            )
            # LogHawk owns the scaler persisted alongside this model.
            model_options["preprocessing"] = False
        self.model = model_class(**model_options)
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
