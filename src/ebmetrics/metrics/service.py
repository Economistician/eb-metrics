"""
Service-level and readiness metrics built around shortfalls, tolerances,
and cost-weighted loss.

These metrics are designed to complement CWSL by capturing:
- How often we avoid shortfalls (NSL),
- How deep the shortfalls are (UD),
- How often forecasts stay within a tolerance band (HR@tau),
- A composite readiness score (FRS).
"""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from .._utils import (
    _to_1d_array,
    _broadcast_param,
    _handle_sample_weight,
)
from .loss import cwsl


__all__ = ["nsl", "ud", "hr_at_tau", "frs"]


def nsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    No-Shortfall Level (NSL).

    NSL is the proportion of intervals with no shortfall, optionally weighted.

    A "hit" is defined as y_pred >= y_true for that interval.
    With weights, NSL is the weighted fraction of hits.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, the NSL is
        computed as sum(w_i * hit_i) / sum(w_i). If the total weight is zero,
        a ValueError is raised.

    Returns
    -------
    float
        No-Shortfall Level in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if np.any(y_true_arr < 0):
        raise ValueError("y_true must be non-negative (demand cannot be negative).")
    if np.any(y_pred_arr < 0):
        raise ValueError("y_pred must be non-negative (forecast cannot be negative).")

    n = y_true_arr.shape[0]
    w = _handle_sample_weight(sample_weight, n)

    # Hit = no shortfall → y_pred >= y_true
    hits = (y_pred_arr >= y_true_arr).astype(float)

    weighted_hits = w * hits
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "NSL is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_hits.sum() / total_weight)


def ud(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Underbuild Depth (UD).

    UD measures the average shortfall depth per interval, optionally weighted.

    For each interval i:
        shortfall_i = max(0, y_true[i] - y_pred[i])

    Unweighted UD:
        UD = mean(shortfall_i)

    Weighted UD:
        UD_w = sum(w_i * shortfall_i) / sum(w_i)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, UD is
        computed as a weighted average. If the total weight is zero, a
        ValueError is raised.

    Returns
    -------
    float
        Underbuild depth (average shortfall per interval).

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if np.any(y_true_arr < 0):
        raise ValueError("y_true must be non-negative (demand cannot be negative).")
    if np.any(y_pred_arr < 0):
        raise ValueError("y_pred must be non-negative (forecast cannot be negative).")

    n = y_true_arr.shape[0]
    w = _handle_sample_weight(sample_weight, n)

    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)

    weighted_shortfall = w * shortfall
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "UD is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_shortfall.sum() / total_weight)


def hr_at_tau(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tau: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Hit Rate within Tolerance (HR@τ).

    HR@τ is the proportion of intervals where the absolute error
    is less than or equal to a specified tolerance τ, optionally weighted.

        hit_i = 1 if |y_true[i] - y_pred[i]| <= tau_i
                0 otherwise

    Unweighted HR@τ:
        HR = mean(hit_i)

    Weighted HR@τ:
        HR_w = sum(w_i * hit_i) / sum(w_i)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    tau : float or array-like of shape (n_samples,)
        Absolute error tolerance. Must be non-negative. Can be:
        - scalar: same tolerance for all intervals
        - 1D array: per-interval tolerance

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, HR@τ is
        computed as a weighted average. If the total weight is zero, a
        ValueError is raised.

    Returns
    -------
    float
        Hit rate within tolerance in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid or the total weight is zero.
    """
    y_true_arr = _to_1d_array(y_true, "y_true")
    y_pred_arr = _to_1d_array(y_pred, "y_pred")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    if np.any(y_true_arr < 0):
        raise ValueError("y_true must be non-negative (demand cannot be negative).")
    if np.any(y_pred_arr < 0):
        raise ValueError("y_pred must be non-negative (forecast cannot be negative).")

    n = y_true_arr.shape[0]
    w = _handle_sample_weight(sample_weight, n)

    # Broadcast tau
    tau_arr = _broadcast_param(tau, n, "tau")
    if np.any(tau_arr < 0):
        raise ValueError("tau must be non-negative.")

    abs_error = np.abs(y_true_arr - y_pred_arr)

    hits = (abs_error <= tau_arr).astype(float)

    weighted_hits = w * hits
    total_weight = float(w.sum())

    if total_weight <= 0:
        raise ValueError(
            "HR@τ is undefined: total sample_weight is zero. "
            "Check your weighting scheme."
        )

    return float(weighted_hits.sum() / total_weight)


def frs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Forecast Readiness Score (FRS).

    Defined as:

        FRS = NSL - CWSL

    where:
        - NSL is the No-Shortfall Level (fraction of intervals with no shortfall),
        - CWSL is the Cost-Weighted Service Loss, using the same cu/co penalties.

    Higher FRS indicates a forecast that both:
        - avoids shortfalls (high NSL), and
        - avoids costly asymmetric error (low CWSL).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    cu : float or array-like of shape (n_samples,)
        Underbuild (shortfall) cost per unit. Must be strictly positive.
        Must match the cu used for CWSL if you are comparing values.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit. Must be strictly positive.
        Must match the co used for CWSL if you are comparing values.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. Applied consistently to
        both NSL and CWSL.

    Returns
    -------
    float
        Forecast Readiness Score, typically in the range [-inf, 1].
        In practice, values closer to 1 indicate strong readiness.

    Raises
    ------
    ValueError
        If inputs are invalid or CWSL is undefined given the data.
    """
    # We rely on the existing validation in nsl() and cwsl()
    nsl_val = nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    cwsl_val = cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu,
        co=co,
        sample_weight=sample_weight,
    )
    return float(nsl_val - cwsl_val)