"""
Internal utility helpers for ebmetrics.

Not part of the public API (names prefixed with _ and module prefixed with _).
These functions support validation, broadcasting, and sample weighting.
"""

from typing import Union, Optional
import numpy as np
from numpy.typing import ArrayLike


def _to_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a 1D float64 numpy array with basic validation."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")
    return arr


def _broadcast_param(
    param: Union[float, ArrayLike],
    length: int,
    name: str,
) -> np.ndarray:
    """Broadcast a scalar or 1D array-like parameter to match a given length."""
    arr = np.asarray(param, dtype=float)

    if arr.ndim == 0:
        arr = np.full(length, float(arr))
    elif arr.ndim == 1 and arr.shape[0] == length:
        pass
    else:
        raise ValueError(
            f"{name} must be a scalar or 1D array of length {length}; got shape {arr.shape}"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")

    return arr


def _handle_sample_weight(
    sample_weight: Optional[ArrayLike],
    length: int,
) -> np.ndarray:
    """Normalize sample_weight to a 1D non-negative array of given length."""
    if sample_weight is None:
        return np.ones(length, dtype=float)

    w = np.asarray(sample_weight, dtype=float)

    if w.ndim == 0:
        w = np.full(length, float(w))
    elif w.ndim == 1 and w.shape[0] == length:
        pass
    else:
        raise ValueError(
            f"sample_weight must be a scalar or 1D array of length {length}; "
            f"got shape {w.shape}"
        )

    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must contain only finite values (no NaN/inf).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")

    return w