"""S3 model path and metadata registry helpers for Stage B artifacts."""

import json

import loghawk.config as config
from loghawk.anomaly_detection.detector_metadata import DetectorMetadata


class ModelRegistry:
    """Centralize per-group artifact paths and manifest JSON operations."""

    def __init__(self, model_root: str):
        self.model_root = model_root.rstrip("/") + "/"

    def paths_for_group(
        self, group: str, backend: str, algorithm: str = "sklearn-isolationforest"
    ) -> dict[str, str]:
        root = self.model_root + group + "/"
        if backend == "cuml":
            detector_name = "isolation_forest.tl"
        else:
            artifact_name = algorithm if backend == "pyod" else "isolation_forest"
            detector_name = artifact_name + ".joblib"
        return {
            "root": root,
            "detector": root + "detector/" + detector_name,
            "scaler": root + "scaler/scaler.npz",
            "metadata": root + "metadata.json",
        }

    def model_set_path(self, group: str) -> str:
        return self.model_root + group + "/model_set.json"

    def paths_for_algorithm(self, group: str, algorithm: str) -> dict[str, str]:
        backend, _device = config.anomaly_algorithm_backend_device(algorithm)
        root = (
            self.model_root
            + group
            + "/algorithms/"
            + algorithm
            + "/"
        )
        artifact = "model.tl" if backend == "cuml" else "model.joblib"
        return {
            "root": root,
            "detector": root + "detector/" + artifact,
            "scaler": root + "scaler/scaler.npz",
            "calibration": root + "calibration/scores.npz",
            "metadata": root + "metadata.json",
        }

    @staticmethod
    def write_metadata(stream, metadata: DetectorMetadata) -> None:
        json.dump(metadata.to_dict(), stream, indent=2)
        stream.write("\n")

    @staticmethod
    def read_metadata(stream) -> DetectorMetadata:
        return DetectorMetadata.from_dict(json.load(stream))
