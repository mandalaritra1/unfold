"""Statistical covariance of a prediction normalized in a fixed bin window."""

import numpy as np


def normalized_prediction_covariance(counts, covariance, widths, normalization_mask):
    """Propagate raw-count covariance through p_i = n_i / (width_i * N).

    N includes only the normalization-mask bins. The returned covariance covers
    all bins, including bins outside that window. A diagonal sumw2 input cannot
    recover any missing event-level correlations between entries.
    """
    counts = np.asarray(counts, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    widths = np.asarray(widths, dtype=float)
    mask = np.asarray(normalization_mask, dtype=bool)
    if (counts.ndim != 1 or widths.shape != counts.shape or mask.shape != counts.shape
            or covariance.shape != (counts.size, counts.size)):
        raise ValueError("Prediction counts, covariance, widths and mask have incompatible shapes")
    if (not np.all(np.isfinite(counts)) or not np.all(np.isfinite(covariance))
            or not np.all(np.isfinite(widths)) or np.any(widths <= 0)):
        raise ValueError("Prediction inputs must be finite and bin widths positive")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Prediction covariance must be symmetric")
    total = counts[mask].sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Prediction normalization must be positive")
    jacobian = (np.eye(counts.size) - np.outer(counts / total, mask)) / (widths[:, None] * total)
    result = jacobian @ covariance @ jacobian.T
    return 0.5 * (result + result.T)
