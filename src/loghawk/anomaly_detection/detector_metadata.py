"""Versioned manifest helpers for Stage B detector artifacts."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any


ARTIFACT_SCHEMA_VERSION = 2


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


@dataclass
class DetectorMetadata:
    backend: str
    group: str
    feature_columns: list[str]
    detector_path: str
    scaler_path: str
    backend_metadata: dict[str, Any]
    versions: dict[str, str | None]
    artifact_schema_version: int = ARTIFACT_SCHEMA_VERSION
    detector_type: str = "isolation_forest"
    trained_at_utc: str = ""
    capabilities: dict[str, Any] | None = None
    algorithm: str = "isolation_forest"

    def __post_init__(self) -> None:
        if not self.trained_at_utc:
            self.trained_at_utc = datetime.now(timezone.utc).isoformat()
        if self.backend not in {"cuml", "sklearn"}:
            raise ValueError(f"Unsupported detector backend: {self.backend!r}")
        if not self.feature_columns:
            raise ValueError("Detector metadata must include feature columns.")

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["n_features"] = self.n_features
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DetectorMetadata":
        if value.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported detector artifact schema version: "
                f"{value.get('artifact_schema_version')!r}; retrain the model."
            )
        metadata = cls(
            backend=value["backend"],
            group=value["group"],
            feature_columns=list(value["feature_columns"]),
            detector_path=value["detector_path"],
            scaler_path=value["scaler_path"],
            backend_metadata=dict(value.get("backend_metadata", {})),
            versions=dict(value.get("versions", {})),
            artifact_schema_version=value["artifact_schema_version"],
            detector_type=value.get("detector_type", "isolation_forest"),
            trained_at_utc=value.get("trained_at_utc", ""),
            capabilities=value.get("capabilities"),
            algorithm=value.get("algorithm", value.get("detector_type", "isolation_forest")),
        )
        if value.get("n_features") != metadata.n_features:
            raise ValueError("Detector metadata feature count does not match its columns.")
        return metadata
