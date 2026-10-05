"""LogHawk's small backend-neutral anomaly detector contract."""

from abc import ABC, abstractmethod
from typing import Any

import loghawk.config as config
from loghawk.anomaly_detection.detector_capabilities import (
    DetectorCapabilities,
)


class AnomalyDetector(ABC):
    """Common interface for LogHawk anomaly detection backends."""

    backend: str
    algorithm: str = "isolation_forest"
    capabilities: DetectorCapabilities

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
        """Load the detector model from a serialized artifact."""

    @abstractmethod
    def export_scaler_state(self) -> dict[str, Any]:
        """Return portable scaler parameters for a separate scaler artifact."""

    @abstractmethod
    def load_scaler_state(self, state: dict[str, Any]) -> None:
        """Restore scaler parameters from a separate scaler artifact."""

    def export_backend_metadata(self) -> dict[str, Any]:
        """Return backend-specific inference parameters for the manifest."""
        return {}

    def load_backend_metadata(self, metadata: dict[str, Any]) -> None:
        """Restore backend-specific inference parameters from the manifest."""
        del metadata


def create_anomaly_detector(
    backend: str | None = None, algorithm: str | None = None
) -> AnomalyDetector:
    """Create only the selected backend, keeping cuML optional on CPU hosts."""
    selected_algorithm = (algorithm or config.LH_ANOMALY_ALGORITHM).strip().lower()
    if selected_algorithm != "isolation_forest":
        raise ValueError(f"Unsupported anomaly algorithm: {selected_algorithm!r}")
    selected = (backend or config.LH_ANOMALY_BACKEND).strip().lower()
    device = config.LH_ANOMALY_DEVICE
    if selected == "sklearn":
        from loghawk.anomaly_detection.sklearn_anomaly_detector import (
            SklearnAnomalyDetector,
        )

        detector = SklearnAnomalyDetector()
        _validate_capabilities(detector, device)
        return detector
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

        detector = CuMLAnomalyDetector()
        _validate_capabilities(detector, device)
        return detector
    raise ValueError(
        f"Unsupported anomaly detector backend: {selected!r}. "
        "Choose 'sklearn' or 'cuml'."
    )


def _validate_capabilities(detector: AnomalyDetector, device: str) -> None:
    if not detector.capabilities.supports("fit", device):
        raise ValueError(
            f"Backend '{detector.backend}' cannot fit on device '{device}'. "
            f"Supported fit devices: {', '.join(detector.capabilities.fit_devices)}"
        )
    if not detector.capabilities.supports("score", device):
        raise ValueError(
            f"Backend '{detector.backend}' cannot score on device '{device}'. "
            f"Supported score devices: {', '.join(detector.capabilities.score_devices)}"
        )
