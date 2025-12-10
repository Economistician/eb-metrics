"""
Framework adapters and utilities for extending ebmetrics into external ML ecosystems.
"""

from .keras_loss import make_cwsl_keras_loss
from .sklearn_scorer import cwsl_loss, cwsl_scorer

__all__ = [
    "make_cwsl_keras_loss",
    "cwsl_loss",
    "cwsl_scorer",
]