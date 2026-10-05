"""NVIDIA cuML training and nvForest inference implementation."""

from typing import Any

import cupy as cp
import numpy as np
from cuml.ensemble import IsolationForest
from cuml.preprocessing import StandardScaler

from loghawk.anomaly_detection.anomaly_detector import AnomalyDetector
from loghawk.anomaly_detection.detector_capabilities import CUML_CAPABILITIES


class CuMLAnomalyDetector(AnomalyDetector):
    backend = "cuml"
    algorithm = "isolation_forest"
    capabilities = CUML_CAPABILITIES

    def __init__(self):
        self.scaler = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.model = None
        self.c_normalization = None
        self.score_cutoff = None

    def train(self, X: Any) -> None:
        X = np.asarray(X, dtype=np.float32)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.scaler_mean = self._to_numpy(self.scaler.mean_).astype(np.float32)
        self.scaler_scale = self._to_numpy(self.scaler.scale_).astype(np.float32)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
            output_type="numpy",
        )
        self.model.fit(X_scaled)

        max_samples = int(self.model.max_samples_)
        self.c_normalization = self._average_path_length_constant(max_samples)
        offset = getattr(self.model, "offset_", None)
        if offset is None:
            offset = getattr(self.model, "offset", None)
        if offset is None:
            raise RuntimeError("Trained cuML IsolationForest has no offset.")
        self.score_cutoff = float(-self._to_numpy(offset).reshape(-1)[0])

    def detect(self, X: Any):
        if self.model is None or self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Train or load the detector before detect().")
        if self.c_normalization is None or self.score_cutoff is None:
            raise RuntimeError("Load cuML score calibration before detect().")

        values = np.asarray(X, dtype=np.float32)
        X_scaled = (values - self.scaler_mean) / self.scaler_scale
        average_path_length = self.model.predict(X_scaled)
        average_path_length = self._to_numpy(average_path_length).reshape(-1)
        scores = np.power(2.0, -average_path_length / self.c_normalization)
        flags = scores >= self.score_cutoff
        return np.asarray(scores), np.asarray(flags, dtype=bool)

    def save(self, path: Any) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an untrained detector.")
        path.write(self.model.as_treelite().serialize_bytes())

    def load(self, path: Any) -> "CuMLAnomalyDetector":
        import nvforest
        import treelite

        treelite_model = treelite.Model.deserialize_bytes(path.read())
        self.model = nvforest.load_from_treelite_model(
            treelite_model,
            device="gpu",
        )
        return self

    def export_scaler_state(self) -> dict[str, Any]:
        if self.scaler_mean is None or self.scaler_scale is None:
            raise RuntimeError("Cannot export an unfitted scaler.")
        return {
            "mean": np.asarray(self.scaler_mean, dtype=np.float32),
            "scale": np.asarray(self.scaler_scale, dtype=np.float32),
        }

    def load_scaler_state(self, state: dict[str, Any]) -> None:
        self.scaler_mean = np.asarray(state["mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(state["scale"], dtype=np.float32)
        if np.any(self.scaler_scale == 0):
            raise ValueError("Scaler state contains a zero scale value.")

    def export_backend_metadata(self) -> dict[str, Any]:
        if self.c_normalization is None or self.score_cutoff is None:
            raise RuntimeError("Cannot export missing cuML score calibration.")
        return {
            "c_normalization": float(self.c_normalization),
            "score_cutoff": float(self.score_cutoff),
        }

    def load_backend_metadata(self, metadata: dict[str, Any]) -> None:
        self.c_normalization = float(metadata["c_normalization"])
        self.score_cutoff = float(metadata["score_cutoff"])
        if self.c_normalization <= 0:
            raise ValueError("cuML c_normalization must be positive.")

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        if isinstance(values, cp.ndarray):
            return cp.asnumpy(values)
        return np.asarray(values)

    @staticmethod
    def _average_path_length_constant(sample_count: int) -> float:
        if sample_count <= 1:
            return 0.0
        if sample_count == 2:
            return 1.0
        harmonic = sum(1.0 / value for value in range(1, sample_count))
        return 2.0 * harmonic - 2.0 * (sample_count - 1) / sample_count
