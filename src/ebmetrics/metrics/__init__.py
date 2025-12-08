"""
ebmetrics.metrics: Common error metrics for forecasting, analytics, and the
Electric Barometer framework.

This module exposes the public API for users, re-exporting metrics
from the submodules in a clean, stable namespace.

Public API includes:
- Cost-weighted loss metrics (CWSL, etc.)
- Classical regression metrics (MAE, MSE, RMSE, MAPE, etc.)
- Service-level and readiness metrics (NSL, UD, HR@τ, FRS)
- Cost-sensitivity utilities for CWSL
"""

# ----------------------------------------------------------------------
# Loss metrics
# ----------------------------------------------------------------------
from .loss import cwsl
from .cost_ratio import estimate_R_cost_balance

# ----------------------------------------------------------------------
# Classical regression metrics
# ----------------------------------------------------------------------
from .regression import (
    mae,
    mse,
    rmse,
    mape,
    wmape,
    msle,
    rmsle,
    medae,
    smape,
    mase,
)

# ----------------------------------------------------------------------
# Service-level + cost-sensitivity metrics
# ----------------------------------------------------------------------
from .service import (
    nsl,
    ud,
    hr_at_tau,
    frs,
    cwsl_sensitivity,
)

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    # Loss & cost-ratio
    "cwsl",
    "estimate_R_cost_balance",

    # Regression
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

    # Service-level + cost-sensitivity
    "nsl",
    "ud",
    "hr_at_tau",
    "frs",
    "cwsl_sensitivity",
]