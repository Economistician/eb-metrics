"""
Asymmetric loss metrics for the Electric Barometer ecosystem.

This module contains loss-like metrics that explicitly encode operational
asymmetry between *underbuild* (shortfall; forecasting below realized demand)
and *overbuild* (excess; forecasting above realized demand).

The primary metric implemented here is **Cost-Weighted Service Loss (CWSL)**,
a demand-normalized loss that generalizes weighted MAPE by assigning distinct
per-unit costs to shortfall and overbuild.

Conceptual definitions, motivation, and interpretation are documented in the
companion research repository (`eb-papers`).
"""

__all__ = [
    "PiecewiseStateAsymmetry",
    "cwsl",
    "piecewise_state_asymmetric_squared_error",
]


from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .._utils import (
    _as_finite_scalar,
    _broadcast_param,
    _handle_sample_weight,
    _validated_nonneg_pair,
)


def cwsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: float | ArrayLike,
    co: float | ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Compute Cost-Weighted Service Loss (CWSL).

    CWSL is a demand-normalized, directionally-aware loss that penalizes
    **shortfalls** and **overbuilds** using explicit per-unit costs.

    For each interval $i$:

    $$
    \begin{aligned}
    s_i &= \max(0, y_i - \hat{y}_i) \\
    o_i &= \max(0, \hat{y}_i - y_i) \\
    \text{cost}_i &= c_{u,i} \; s_i + c_{o,i} \; o_i
    \end{aligned}
    $$

    and the aggregated metric is:

    $$
    \mathrm{CWSL} = \frac{\sum_i w_i \; \text{cost}_i}{\sum_i w_i \; y_i}
    $$

    where $w_i$ are optional sample weights (default $w_i = 1$).
    Lower values indicate better performance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Realized demand $y$. Must be non-negative.

    y_pred : array-like of shape (n_samples,)
        Forecast demand $\hat{y}$. Must be non-negative and have the same
        shape as ``y_true``.

    cu : float or array-like of shape (n_samples,)
        Per-unit shortfall cost $c_u$. Can be a scalar (global cost) or a
        1D array specifying per-interval costs. Must be non-negative.

    co : float or array-like of shape (n_samples,)
        Per-unit overbuild cost $c_o$. Can be a scalar (global cost) or a
        1D array specifying per-interval costs. Must be non-negative.

    sample_weight : float or array-like of shape (n_samples,), optional
        Optional non-negative weights per interval. If ``None``, all intervals
        receive weight ``1.0``.

    Returns
    -------
    float
        The CWSL value. Lower is better.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different shapes, if any demand or
        forecast values are negative, if any costs are negative, or if the
        metric is undefined due to zero total (weighted) demand with positive
        total (weighted) cost.

    Notes
    -----
    - When ``cu == co`` (up to a constant scaling), CWSL behaves similarly to a
      demand-normalized absolute error (wMAPE-like), but retains explicit cost
      semantics.
    - If total (weighted) demand is zero and total (weighted) cost is zero,
      this implementation returns ``0.0``.
    - If total (weighted) demand is zero but total (weighted) cost is positive,
      the metric is undefined under this formulation and a ``ValueError`` is
      raised.

    References
    ----------
    Electric Barometer Technical Note: Cost-Weighted Service Loss (CWSL).
    """
    y_true_arr, y_pred_arr = _validated_nonneg_pair(y_true, y_pred)
    return _cwsl_from_validated(y_true_arr, y_pred_arr, cu, co, sample_weight)


def _cwsl_from_validated(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    cu: float | ArrayLike,
    co: float | ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    n = y_true_arr.shape[0]

    delta = y_true_arr - y_pred_arr
    shortfall = np.maximum(delta, 0.0)
    overbuild = shortfall - delta

    cu_scalar = _as_finite_scalar(cu, "cu")
    co_scalar = _as_finite_scalar(co, "co")

    if cu_scalar is not None and cu_scalar < 0:
        raise ValueError("cu must be non-negative.")
    if co_scalar is not None and co_scalar < 0:
        raise ValueError("co must be non-negative.")

    cu_arr = None if cu_scalar is not None else _broadcast_param(cu, (n,), "cu")
    co_arr = None if co_scalar is not None else _broadcast_param(co, (n,), "co")
    if cu_arr is not None and np.any(cu_arr < 0):
        raise ValueError("cu must be non-negative.")
    if co_arr is not None and np.any(co_arr < 0):
        raise ValueError("co must be non-negative.")

    if sample_weight is None:
        if cu_scalar is not None and co_scalar is not None:
            total_cost = float(cu_scalar * shortfall.sum() + co_scalar * overbuild.sum())
        else:
            cu_term = cu_scalar if cu_scalar is not None else cu_arr
            co_term = co_scalar if co_scalar is not None else co_arr
            total_cost = float(np.sum(cu_term * shortfall + co_term * overbuild))
        total_demand = float(y_true_arr.sum())
    else:
        w = _handle_sample_weight(sample_weight, n, dtype=float)
        if cu_scalar is not None and co_scalar is not None:
            total_cost = float(cu_scalar * np.dot(w, shortfall) + co_scalar * np.dot(w, overbuild))
        else:
            cu_term = cu_scalar if cu_scalar is not None else cu_arr
            co_term = co_scalar if co_scalar is not None else co_arr
            total_cost = float(np.dot(w, cu_term * shortfall + co_term * overbuild))
        total_demand = float(np.dot(w, y_true_arr))

    if total_demand > 0:
        return total_cost / total_demand

    if total_cost == 0:
        return 0.0

    raise ValueError(
        "CWSL is undefined: total (weighted) demand is zero while total (weighted) "
        "cost is positive. Check your data slice or weighting scheme."
    )


@dataclass(frozen=True)
class PiecewiseStateAsymmetry:
    """
    State-dependent asymmetric cost profile.

    The "state" is y_true (e.g., utilization). Each observation is assigned a
    state weight based on the band it falls into. Under-forecasting (y_pred < y_true)
    is additionally penalized relative to over-forecasting.

    Example:
        state_upper_bounds=(0.75, 0.85, 1.01)
        state_weights=(1.0, 2.0, 5.0)

    defines three bands:
        y_true <= 0.75 -> weight 1.0
        0.75 < y_true <= 0.85 -> weight 2.0
        0.85 < y_true <= 1.01 -> weight 5.0
    """

    state_upper_bounds: tuple[float, ...]
    state_weights: tuple[float, ...]
    under_mult: float = 3.0
    over_mult: float = 1.0

    def __post_init__(self) -> None:
        if len(self.state_upper_bounds) != len(self.state_weights):
            raise ValueError("state_upper_bounds and state_weights must have the same length.")
        if any(
            b2 <= b1
            for b1, b2 in zip(self.state_upper_bounds, self.state_upper_bounds[1:], strict=False)
        ):
            raise ValueError("state_upper_bounds must be strictly increasing.")
        if self.under_mult <= 0 or self.over_mult <= 0:
            raise ValueError("Multipliers must be positive.")


def piecewise_state_asymmetric_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    profile: PiecewiseStateAsymmetry,
) -> np.ndarray:
    """
    Compute a state-dependent asymmetric squared error.

    - State weights are determined by y_true's band.
    - Under-forecasting (y_pred < y_true) is multiplied by profile.under_mult.
    - Over-forecasting is multiplied by profile.over_mult.

    Returns per-observation costs (broadcasted shape).
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)

    err = yhat - y  # negative => under-forecast

    bounds = np.asarray(profile.state_upper_bounds, dtype=float)
    weights = np.asarray(profile.state_weights, dtype=float)

    # Closed-on-right bands: y == bound stays in that band (see docstring).
    band_idx = np.searchsorted(bounds, y, side="left")

    # Clamp to valid range: [0, len(weights) - 1]
    band_idx = np.clip(band_idx, 0, len(weights) - 1)

    w_state = weights[band_idx]
    w_asym = np.where(err < 0, profile.under_mult, profile.over_mult)

    return w_state * w_asym * (err**2)
