"""LogHawk's small backend-neutral anomaly detector contract."""

from abc import ABC, abstractmethod
from typing import Any

import loghawk.config as config


class AnomalyDetector(ABC):
    """Common interface for LogHawk anomaly detection backends."""

    backend: str

    @abstractmethod
    def train(self, X: Any) -> None:
        """Fit preprocessing and anomaly detection on a feature matrix."""

    @abstractmethod
    def detect(self, X: Any):
        """Return (positive-is-anomalous scores, boolean anomaly flags)."""

    @abstractmethod
    def save(self, path: Any) -> None:
        """Serialize this detector and its preprocessing state."""

    @abstractmethod
    def load(self, path: Any) -> "AnomalyDetector":
        """Load this detector from a serialized artifact."""


def create_anomaly_detector(backend: str | None = None) -> AnomalyDetector:
    """Create only the selected backend, keeping cuML optional on CPU hosts."""
    selected = (backend or config.LH_ANOMALY_BACKEND).strip().lower()
    if selected == "sklearn":
        from loghawk.anomaly_detection.sklearn_anomaly_detector import (
            SklearnAnomalyDetector,
        )

        return SklearnAnomalyDetector()
    if selected == "cuml":
        try:
            from loghawk.anomaly_detection.cuml_anomaly_detector import (
                CuMLAnomalyDetector,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LH_ANOMALY_BACKEND='cuml' requires cuML in the active "
                "RAPIDS/CUDA environment. Install it using the RAPIDS guide."
            ) from exc

        return CuMLAnomalyDetector()
    raise ValueError(
        f"Unsupported anomaly detector backend: {selected!r}. "
        "Choose 'sklearn' or 'cuml'."
    )
