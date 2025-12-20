"""
Framework integrations for Electric Barometer metrics.

The `ebmetrics.frameworks` package provides optional adapters that allow
Electric Barometer metrics to plug into common machine-learning and forecasting
workflows.

Currently supported integrations include:

- **scikit-learn** scorers (for model selection utilities such as ``GridSearchCV``)
- **Keras / TensorFlow** loss functions (for training deep learning models)

Notes
-----
- These integrations are thin wrappers around core Electric Barometer metrics.
  Metric definitions live in :mod:`ebmetrics.metrics`.
- Some integrations rely on optional third-party dependencies and may import
  those dependencies lazily (e.g., TensorFlow).

Conceptual definitions and interpretation are documented in the companion
research repository (`eb-papers`).
"""

from .keras_loss import make_cwsl_keras_loss
from .sklearn_scorer import cwsl_loss, cwsl_scorer

__all__ = [
    "make_cwsl_keras_loss",
    "cwsl_loss",
    "cwsl_scorer",
]