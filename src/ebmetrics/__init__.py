"""
ebmetrics: error metrics and evaluation utilities for Electric Barometer.

This package exposes a clean, stable public API for:

- Cost-weighted loss metrics (e.g., CWSL)
- Classical regression metrics (MAE, MSE, RMSE, MAPE, etc.)
- Service-level and readiness metrics (NSL, UD, HR@τ, FRS)
"""

# ----------------------------------------------------------------------
# Loss metrics
# ----------------------------------------------------------------------
from .metrics.loss import cwsl

# ----------------------------------------------------------------------
# Classical regression metrics
# ----------------------------------------------------------------------
from .metrics.regression import (
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
# Service-level metrics
# ----------------------------------------------------------------------
from .metrics.service import (
    nsl,
    ud,
    hr_at_tau,
    frs,
)

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    # Loss
    "cwsl",

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

    # Service-level
    "nsl",
    "ud",
    "hr_at_tau",
    "frs",
]