"""
Standard regression and forecasting error metrics.

This module provides classical statistical metrics These metrics 
serve as baselines for comparison and model validation across 
symmetric and asymmetric error formulations.
"""

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "wmape",
    "msle",
    "rmsle",
    "medae",
    "smape",
    "mase",
]

import numpy as np
from numpy.typing import ArrayLike


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
def _validate_shapes(y_true: ArrayLike, y_pred: ArrayLike):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shapes; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )
    return y_true_arr, y_pred_arr


# ----------------------------------------------------------------------
# Core metrics
# ----------------------------------------------------------------------
def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    diff = y_true_arr - y_pred_arr
    return float(np.mean(diff**2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    mask = y_true_arr != 0
    if not np.any(mask):
        raise ValueError("MAPE undefined when all y_true values are zero.")
    pct = np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])
    return float(np.mean(pct) * 100)


def wmape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    numerator = np.sum(np.abs(y_true_arr - y_pred_arr))
    denominator = np.sum(np.abs(y_true_arr))
    if denominator == 0:
        raise ValueError("WMAPE undefined when sum(|y_true|) == 0.")
    return float(numerator / denominator * 100)


# ----------------------------------------------------------------------
# Log-based metrics
# ----------------------------------------------------------------------
def msle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Squared Log Error.

    Requires y_true and y_pred >= 0.
    """
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)

    if np.any(y_true_arr < 0) or np.any(y_pred_arr < 0):
        raise ValueError("MSLE requires non-negative y_true and y_pred.")

    log_t = np.log1p(y_true_arr)
    log_p = np.log1p(y_pred_arr)
    return float(np.mean((log_t - log_p) ** 2))


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Log Error."""
    return float(np.sqrt(msle(y_true, y_pred)))


# ----------------------------------------------------------------------
# Robust + forecasting-specific metrics
# ----------------------------------------------------------------------
def medae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median Absolute Error."""
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    return float(np.median(np.abs(y_true_arr - y_pred_arr)))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Symmetric MAPE (M4/M5 competition version).

    SMAPE = 200 * |y - ŷ| / (|y| + |ŷ|)
    """
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(200 * np.abs(y_true_arr[mask] - y_pred_arr[mask]) / denom[mask]))


def mase(y_true: ArrayLike, y_pred: ArrayLike, y_naive: ArrayLike) -> float:
    """
    Mean Absolute Scaled Error.

    y_naive is typically the naive forecast (y[t-1]).

    MASE = MAE / MAE_naive
    """
    y_true_arr, y_pred_arr = _validate_shapes(y_true, y_pred)
    y_true_arr2, y_naive_arr = _validate_shapes(y_true, y_naive)

    mae_model = np.mean(np.abs(y_true_arr - y_pred_arr))
    mae_naive = np.mean(np.abs(y_true_arr2 - y_naive_arr))

    if mae_naive == 0:
        raise ValueError("MASE undefined because naive MAE is zero.")

    return float(mae_model / mae_naive)