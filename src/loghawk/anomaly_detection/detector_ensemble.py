"""Score calibration and equal-weight aggregation for detector ensembles."""

from collections.abc import Mapping

import numpy as np


def calibration_reference(scores) -> np.ndarray:
    """Return sorted finite training scores for empirical percentile mapping."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot calibrate an empty or non-finite score set")
    return np.sort(values)


def normalize_scores(scores, reference) -> np.ndarray:
    """Map detector scores to [0, 1] percentiles; larger means more anomalous."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    baseline = np.asarray(reference, dtype=np.float64).reshape(-1)
    if baseline.size == 0 or not np.all(np.isfinite(baseline)):
        raise ValueError("Score calibration reference is empty or invalid")
    if not np.all(np.isfinite(values)):
        raise ValueError("Detector produced non-finite anomaly scores")
    return np.searchsorted(baseline, values, side="right") / baseline.size


def combine_normalized_scores(
    scores_by_algorithm: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return an equal-weight mean and normalized score per algorithm."""
    if len(scores_by_algorithm) < 2:
        raise ValueError("Score aggregation requires at least two detectors")
    normalized = {
        algorithm: np.asarray(scores, dtype=np.float64).reshape(-1)
        for algorithm, scores in scores_by_algorithm.items()
    }
    lengths = {scores.size for scores in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("Detector score arrays have different row counts")
    combined = np.mean(np.vstack(list(normalized.values())), axis=0)
    return combined, normalized
