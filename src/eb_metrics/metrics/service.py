"""
Service-level and readiness metrics for the Electric Barometer ecosystem.

This module contains evaluation metrics that complement asymmetric loss (e.g.,
CWSL) by measuring **service behavior** and **readiness characteristics**:

- **NSL**: how often forecasts avoid shortfall (service reliability)
- **UD**: how deep shortfalls are when they occur (service severity)
- **HR@τ**: how often forecasts fall within a tolerance band (accuracy within bounds)
- **FRS**: a composite readiness score built from NSL and CWSL
- **CWSL sensitivity**: how CWSL changes under alternative cost-ratio assumptions

Operational definitions, interpretation, and motivation are documented in the
companion research repository (`eb-papers`).

Design note
-----------
- `cwsl_sensitivity` remains in **eb-metrics** because it is deterministic metric
  evaluation (a convenience wrapper around `cwsl`).
- DataFrame-oriented plumbing and tuning workflows live in **eb-optimization**.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .._utils import (
    _as_finite_scalar,
    _broadcast_param,
    _handle_sample_weight,
    _validated_nonneg_pair,
)
from .loss import cwsl

__all__ = ["cwsl_sensitivity", "frs", "hr_at_tau", "nsl", "ud"]


def nsl(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Compute No-Shortfall Level (NSL).

    NSL is the (optionally weighted) fraction of evaluation intervals in which
    the forecast does **not** underpredict realized demand.

    For each interval $i$, define a hit indicator:

    $$
    h_i = \mathbb{1}[\hat{y}_i \ge y_i]
    $$

    Then:

    $$
    \mathrm{NSL} = \frac{\sum_i w_i \; h_i}{\sum_i w_i}
    $$

    where $w_i$ are optional sample weights (default $w_i = 1$).
    Higher values are better, with $\mathrm{NSL} \in [0, 1]$.
    """
    y_true_arr, y_pred_arr = _validated_nonneg_pair(y_true, y_pred)
    return _nsl_from_validated(y_true_arr, y_pred_arr, sample_weight)


def _nsl_from_validated(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    sample_weight: ArrayLike | None = None,
) -> float:
    n = y_true_arr.shape[0]
    hits = y_pred_arr >= y_true_arr

    if sample_weight is None:
        if n == 0:
            raise ValueError(
                "NSL is undefined: total sample_weight is zero. Check your weighting scheme."
            )
        return float(np.mean(hits))

    w = _handle_sample_weight(sample_weight, n)
    total_weight = float(w.sum())
    if total_weight <= 0:
        raise ValueError(
            "NSL is undefined: total sample_weight is zero. Check your weighting scheme."
        )

    return float(np.dot(w, hits) / total_weight)


def ud(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Compute Underbuild Depth (UD).

    UD is a *conditional* severity metric: it averages shortfall magnitude
    only over shortfall intervals $T^{SF} = \{ i : y_i > \hat{y}_i \}$.
    Intervals with no shortfall do not enter the denominator.

    Define per-interval shortfall:

    $$
    s_i = \max(0, y_i - \hat{y}_i)
    $$

    Then:

    $$
    \mathrm{UD} = \frac{\sum_{i \in T^{SF}} w_i \; s_i}{\sum_{i \in T^{SF}} w_i}
    $$

    If no shortfalls occur (total shortfall weight is zero), this implementation
    returns $0.0$. Higher values indicate deeper average shortfall; **lower is
    better**.
    """
    y_true_arr, y_pred_arr = _validated_nonneg_pair(y_true, y_pred)
    return _ud_from_validated(y_true_arr, y_pred_arr, sample_weight)


def _ud_from_validated(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    sample_weight: ArrayLike | None = None,
) -> float:
    n = y_true_arr.shape[0]
    shortfall = np.maximum(y_true_arr - y_pred_arr, 0.0)
    mask = y_true_arr > y_pred_arr

    if sample_weight is None:
        if n == 0:
            raise ValueError(
                "UD is undefined: total sample_weight is zero. Check your weighting scheme."
            )
        shortfall_weight = float(np.count_nonzero(mask))
        if shortfall_weight <= 0:
            return 0.0
        return float(shortfall.sum() / shortfall_weight)

    w = _handle_sample_weight(sample_weight, n)
    total_weight = float(w.sum())
    if total_weight <= 0:
        raise ValueError(
            "UD is undefined: total sample_weight is zero. Check your weighting scheme."
        )

    shortfall_weight = float(np.dot(w, mask))
    if shortfall_weight <= 0:
        return 0.0
    return float(np.dot(w, shortfall) / shortfall_weight)


def hr_at_tau(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    tau: float | ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Compute Hit Rate within Tolerance (HR@τ).

    HR@τ measures the (optionally weighted) fraction of intervals whose absolute
    error falls within a tolerance band $\tau$.

    Define absolute error and hit indicator:

    $$
    \begin{aligned}
    e_i &= |y_i - \hat{y}_i| \\
    h_i &= \mathbb{1}[e_i \le \tau_i]
    \end{aligned}
    $$

    Then:

    $$
    \mathrm{HR@\tau} = \frac{\sum_i w_i \; h_i}{\sum_i w_i}
    $$
    """
    y_true_arr, y_pred_arr = _validated_nonneg_pair(y_true, y_pred)
    return _hr_at_tau_from_validated(y_true_arr, y_pred_arr, tau, sample_weight)


def _hr_at_tau_from_validated(
    y_true_arr: np.ndarray,
    y_pred_arr: np.ndarray,
    tau: float | ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    n = y_true_arr.shape[0]
    abs_error = np.abs(y_true_arr - y_pred_arr)
    tau_scalar = _as_finite_scalar(tau, "tau")

    if sample_weight is None and tau_scalar is not None:
        if tau_scalar < 0:
            raise ValueError("tau must be non-negative.")
        if n == 0:
            raise ValueError(
                "HR@τ is undefined: total sample_weight is zero. Check your weighting scheme."
            )
        return float(np.mean(abs_error <= tau_scalar))

    tau_arr = _broadcast_param(tau, (n,), "tau") if tau_scalar is None else None
    if tau_scalar is not None and tau_scalar < 0:
        raise ValueError("tau must be non-negative.")
    if tau_arr is not None and np.any(tau_arr < 0):
        raise ValueError("tau must be non-negative.")
    hits = abs_error <= (tau_scalar if tau_arr is None else tau_arr)

    if sample_weight is None:
        if n == 0:
            raise ValueError(
                "HR@τ is undefined: total sample_weight is zero. Check your weighting scheme."
            )
        return float(np.mean(hits))

    w = _handle_sample_weight(sample_weight, n)
    total_weight = float(w.sum())
    if total_weight <= 0:
        raise ValueError(
            "HR@τ is undefined: total sample_weight is zero. Check your weighting scheme."
        )
    return float(np.dot(w, hits) / total_weight)


def cwsl_sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    R_list: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
    co: float | ArrayLike = 1.0,
    sample_weight: ArrayLike | None = None,
) -> dict[float, float]:
    r"""
    Evaluate CWSL across a grid of cost ratios (cost sensitivity analysis).

    For each candidate ratio:

    $$
    R = \frac{c_u}{c_o}, \quad c_u = R \cdot c_o
    $$

    this helper computes:

    - ``cwsl(y_true, y_pred, cu=R*co, co=co, sample_weight=...)``

    Non-positive R values are ignored. Non-finite values (NaN/inf) raise ValueError.
    If no positive values remain, raises ValueError.
    """
    results: dict[float, float] = {}

    # Convert co to a numeric array to satisfy type checkers for the multiplication
    co_arr = np.asanyarray(co)
    if np.any(co_arr < 0):
        raise ValueError("co must be non-negative.")

    for R in R_list:
        if R is None:
            continue
        Rf = float(R)
        if not np.isfinite(Rf):
            raise ValueError("R_list must contain only finite values (no NaN/inf).")
        if Rf <= 0:
            continue

        value = cwsl(
            y_true=y_true,
            y_pred=y_pred,
            cu=Rf * co_arr,
            co=co_arr,
            sample_weight=sample_weight,
        )
        results[Rf] = float(value)

    if not results:
        raise ValueError("No valid R values in R_list (must contain at least one positive value).")

    return results


def frs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    cu: float | ArrayLike,
    co: float | ArrayLike,
    *,
    cwsl_max: float,
    sample_weight: ArrayLike | None = None,
) -> float:
    r"""
    Compute Forecast Readiness Score (FRS).

    Because NSL lies in $[0, 1]$ but raw CWSL may exceed 1, CWSL is first
    scaled to the unit interval using a required application-specific bound
    $\mathrm{CWSL}_{\max} > 0$ (the largest economically meaningful CWSL):

    $$
    \mathrm{CWSL}_{\mathrm{scaled}}
        = \min\!\left(1, \frac{\mathrm{CWSL}}{\mathrm{CWSL}_{\max}}\right)
    $$

    The composite readiness score is then:

    $$
    \mathrm{FRS} = \mathrm{NSL} - \mathrm{CWSL}_{\mathrm{scaled}}
    $$

    so that $\mathrm{FRS} \in [-1, 1]$. FRS is an evaluative readiness signal,
    not a training loss.

    where:
    - NSL measures the frequency of avoiding shortfall (higher is better)
    - CWSL measures asymmetric, demand-normalized cost (lower is better)
    - $\mathrm{CWSL}_{\max}$ is a required, user-chosen bound with no default
    """
    cwsl_max_val = float(cwsl_max)
    if not np.isfinite(cwsl_max_val) or cwsl_max_val <= 0.0:
        raise ValueError("cwsl_max must be finite and strictly greater than 0.")

    nsl_val = nsl(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)
    cwsl_val = cwsl(
        y_true=y_true,
        y_pred=y_pred,
        cu=cu,
        co=co,
        sample_weight=sample_weight,
    )
    cwsl_scaled = min(1.0, cwsl_val / cwsl_max_val)
    return float(nsl_val - cwsl_scaled)
