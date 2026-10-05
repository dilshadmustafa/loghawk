"""Map detector scores onto comparable [0, 1] anomaly scales."""

import numpy as np


class ScoreNormalizer:
    """Quantile normalizer where larger output always means more anomalous."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_ = None
        self.upper_ = None

    def fit(self, scores):
        values = np.asarray(scores, dtype=float).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Scores must be a non-empty finite array")
        self.lower_, self.upper_ = np.quantile(
            values, [self.lower_quantile, self.upper_quantile]
        )
        return self

    def transform(self, scores):
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("Fit the score normalizer before transform().")
        values = np.asarray(scores, dtype=float)
        span = self.upper_ - self.lower_
        if span <= 0:
            return np.zeros_like(values, dtype=float)
        return np.clip((values - self.lower_) / span, 0.0, 1.0)
