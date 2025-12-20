"""
Public metric API for the Electric Barometer ecosystem.

The `ebmetrics.metrics` package provides a curated, stable import surface for
Electric Barometer evaluation metrics.

This package groups metrics into four main categories:

- **Asymmetric loss metrics** (`loss`)
  Cost-aware losses that encode directional operational asymmetry.

- **Service and readiness diagnostics** (`service`)
  Metrics that quantify shortfall avoidance, tolerance coverage, and readiness.

- **Classical regression metrics** (`regression`)
  Standard symmetric error metrics used for baseline comparison.

- **Cost-ratio utilities** (`cost_ratio`)
  Helpers for selecting and analyzing the asymmetric cost ratio
  :math:`R = c_u / c_o`.

Conceptual definitions and interpretation of Electric Barometer metrics are
documented in the companion research repository (`eb-papers`). This package is
the executable reference implementation.

Notes
-----
Users are encouraged to import from `ebmetrics.metrics` or from the relevant
submodule (e.g., `ebmetrics.metrics.service`) rather than internal helpers.

Examples
--------
Import from the package surface:

>>> from ebmetrics.metrics import cwsl, nsl, frs

Or import from a submodule:

>>> from ebmetrics.metrics.loss import cwsl
>>> from ebmetrics.metrics.service import nsl
"""

from .cost_ratio import estimate_R_cost_balance
from .loss import cwsl
from .regression import (
    mae,
    mase,
    medae,
    mape,
    mse,
    msle,
    rmse,
    rmsle,
    smape,
    wmape,
)
from .service import cwsl_sensitivity, frs, hr_at_tau, nsl, ud

__all__ = [
    # Asymmetric loss
    "cwsl",
    # Service/readiness
    "nsl",
    "ud",
    "hr_at_tau",
    "frs",
    "cwsl_sensitivity",
    # Classical regression
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
    # Cost-ratio utilities
    "estimate_R_cost_balance",
]