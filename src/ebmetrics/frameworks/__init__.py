"""
Framework adapters and utilities for extending ebmetrics into external ML ecosystems.

Current modules include:

- keras_loss: Keras/TensorFlow-compatible implementation of CWSL loss.
"""

from .keras_loss import make_cwsl_keras_loss

__all__ = [
    "make_cwsl_keras_loss",
]