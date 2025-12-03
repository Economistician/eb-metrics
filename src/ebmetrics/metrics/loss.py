"""
Loss-like metrics with explicit cost asymmetry (e.g., CWSL).

These metrics are designed to encode business-relevant costs for
under- and over-forecasting rather than treating all errors symmetrically.
"""

__all__ = ["cwsl"]

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from .._utils import (
    _to_1d_array,
    _broadcast_param,
    _handle_sample_weight,
)


def cwsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: Union[float, ArrayLike],
    co: Union[float, ArrayLike],
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Cost-Weighted Service Loss (CWSL).

    CWSL is a demand-normalized, directionally-aware forecast error metric that
    applies asymmetric penalties to shortfalls and overbuilds. It answers:

        "What fraction of total demand was effectively lost due to the
        cost-weighted impact of forecast error?"

    Formal definition
    -----------------
    For each observation (interval, item, etc.) i:

        y_i    : actual demand
        ŷ_i    : forecast
        s_i    : shortfall  = max(0, y_i - ŷ_i)
        o_i    : overbuild  = max(0, ŷ_i - y_i)

    with penalties:

        cu_i   : cost weight for shortfall
        co_i   : cost weight for overbuild

    The cost-weighted loss is:

        cost_i = cu_i * s_i + co_i * o_i

    and the Cost-Weighted Service Loss is:

        CWSL =  sum_i( cost_i * w_i ) / sum_i( y_i * w_i )

    where w_i are optional sample weights (default w_i = 1).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Actual demand. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecasted demand. Must be non-negative.

    cu : float or array-like of shape (n_samples,)
        Shortfall (underbuild) cost per unit. Must be strictly positive.
        A scalar applies globally; a 1D array allows per-observation penalties.

    co : float or array-like of shape (n_samples,)
        Overbuild (excess) cost per unit. Must be strictly positive.
        A scalar applies globally; a 1D array allows per-observation penalties.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If provided, both the
        numerator (cost) and denominator (demand) are weighted. If the
        total weighted demand is zero while total weighted cost is > 0,
        a ValueError is raised (CWSL undefined in that case).

    Returns
    -------
    float
        Cost-weighted service loss, demand-normalized. Values are >= 0, with
        higher values indicating more cost-weighted error relative to total
        demand.

    Raises
    ------
    ValueError
        If inputs are invalid or CWSL is undefined given the data.

    Notes
    -----
    - If cu == co, CWSL behaves like a demand-normalized symmetric error metric.
    - If cu > co, shortfalls are penalized more heavily than overbuilds.
    - Designed for short-horizon, high-frequency operational forecasting
      where being "short" is worse than being "long".
    """
    # Convert y_true and y_pred to validated 1D arrays
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

    # Broadcast cu and co (allow scalar or 1D array of length n)
    cu_arr = _broadcast_param(cu, n, "cu")
    co_arr = _broadcast_param(co, n, "co")

    if np.any(cu_arr <= 0):
        raise ValueError("cu (shortfall cost) must be strictly positive.")
    if np.any(co_arr <= 0):
        raise ValueError("co (overbuild cost) must be strictly positive.")

    # Handle sample_weight
    w = _handle_sample_weight(sample_weight, n)

    # Compute shortfall and overbuild per interval
    shortfall = np.maximum(0.0, y_true_arr - y_pred_arr)
    overbuild = np.maximum(0.0, y_pred_arr - y_true_arr)

    # Interval cost
    cost = cu_arr * shortfall + co_arr * overbuild

    # Apply weights
    weighted_cost = w * cost
    weighted_demand = w * y_true_arr

    total_cost = float(weighted_cost.sum())
    total_demand = float(weighted_demand.sum())

    if total_demand > 0:
        return total_cost / total_demand

    # total_demand == 0
    if total_cost == 0:
        # No demand and no cost → define CWSL as 0.0
        return 0.0

    # Cost but no demand → undefined metric under this formulation
    raise ValueError(
        "CWSL is undefined: total (weighted) demand is zero while total (weighted) "
        "cost is positive. Check your data slice or weighting scheme."
    )