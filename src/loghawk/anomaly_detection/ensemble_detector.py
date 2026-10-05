"""Capability-aware weighted ensemble for compatible anomaly detectors."""

import numpy as np

from loghawk.anomaly_detection.score_normalizer import ScoreNormalizer


class EnsembleDetector:
    """Combine detectors whose ``detect`` scores increase with anomaly strength."""

    def __init__(self, detectors, weights=None, threshold: float = 0.5):
        if not detectors:
            raise ValueError("An ensemble requires at least one detector")
        self.detectors = list(detectors)
        self.weights = np.asarray(
            weights if weights is not None else [1.0] * len(self.detectors),
            dtype=float,
        )
        if self.weights.shape != (len(self.detectors),) or np.any(self.weights < 0):
            raise ValueError("Weights must be non-negative and match detector count")
        if not np.isfinite(self.weights).all() or self.weights.sum() <= 0:
            raise ValueError("At least one finite positive weight is required")
        self.weights /= self.weights.sum()
        self.threshold = float(threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.normalizers = [ScoreNormalizer() for _ in self.detectors]

    @property
    def capabilities(self):
        capabilities = [detector.capabilities for detector in self.detectors]
        return {
            "fit_devices": sorted(set.intersection(
                *(set(value.fit_devices) for value in capabilities)
            )),
            "score_devices": sorted(set.intersection(
                *(set(value.score_devices) for value in capabilities)
            )),
            "multi_gpu_fit": all(value.multi_gpu_fit for value in capabilities),
            "multi_gpu_score": all(value.multi_gpu_score for value in capabilities),
            "distributed_fit": all(value.distributed_fit for value in capabilities),
            "distributed_score": all(value.distributed_score for value in capabilities),
        }

    def train(self, X):
        """Train members and calibrate score normalizers on their training scores."""
        for detector, normalizer in zip(self.detectors, self.normalizers):
            detector.train(X)
            scores, _ = detector.detect(X)
            normalizer.fit(scores)
        return self

    def detect(self, X):
        """Return weighted normalized anomaly scores and thresholded flags."""
        normalized = []
        for detector, normalizer in zip(self.detectors, self.normalizers):
            scores, _ = detector.detect(X)
            normalized.append(normalizer.transform(scores))
        combined = np.average(np.vstack(normalized), axis=0, weights=self.weights)
        return combined, combined >= self.threshold
