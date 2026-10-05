"""Runtime capabilities declared by a detector implementation."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DetectorCapabilities:
    fit_devices: tuple[str, ...]
    score_devices: tuple[str, ...]
    multi_gpu_fit: bool = False
    multi_gpu_score: bool = False
    distributed_fit: bool = False
    distributed_score: bool = False
    batch_score: bool = True

    def supports(self, operation: str, device: str) -> bool:
        if operation not in {"fit", "score"}:
            raise ValueError("operation must be 'fit' or 'score'")
        devices = self.fit_devices if operation == "fit" else self.score_devices
        return device.lower() in devices

    def to_dict(self) -> dict:
        result = asdict(self)
        result["fit_devices"] = list(self.fit_devices)
        result["score_devices"] = list(self.score_devices)
        return result


SKLEARN_CAPABILITIES = DetectorCapabilities(("cpu",), ("cpu",))
CUML_CAPABILITIES = DetectorCapabilities(("gpu",), ("gpu",))
