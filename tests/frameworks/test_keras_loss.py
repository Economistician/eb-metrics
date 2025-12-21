from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from eb_metrics.frameworks import make_cwsl_keras_loss


# ----------------------------------------------------------------------
# Helpers: fake TensorFlow module so tests don't require real tensorflow
# ----------------------------------------------------------------------
class _FakeKerasBackend:
    @staticmethod
    def epsilon() -> float:
        return 1e-7


class _FakeKeras(SimpleNamespace):
    backend = _FakeKerasBackend()


class _FakeTF(SimpleNamespace):
    float32 = np.float32
    keras = _FakeKeras()

    @staticmethod
    def constant(value: Any, dtype=None):
        return np.array(value, dtype=dtype or np.float32)

    @staticmethod
    def cast(x: Any, dtype=None):
        return np.array(x, dtype=dtype or np.float32)

    class nn:
        @staticmethod
        def relu(x):
            return np.maximum(x, 0.0)

    @staticmethod
    def reduce_sum(x, axis=-1):
        return np.sum(x, axis=axis)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_make_cwsl_keras_loss_validates_costs():
    with pytest.raises(ValueError):
        make_cwsl_keras_loss(cu=0.0, co=1.0)

    with pytest.raises(ValueError):
        make_cwsl_keras_loss(cu=2.0, co=0.0)


def test_make_cwsl_keras_loss_uses_tensorflow(monkeypatch):
    """
    Basic behavior test using a fake TensorFlow module.

    Verifies that the per-sample loss matches the numpy CWSL calculation.
    """
    # Inject fake tensorflow module
    fake_tf = _FakeTF()
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    loss_fn = make_cwsl_keras_loss(cu=2.0, co=1.0)

    # Simple batch of 2 samples, horizon=3
    y_true = np.array([[10.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    y_pred = np.array([[8.0, 0.0, 0.0], [7.0, 3.0, 0.0]])

    # Compute loss via the Keras-style function (using fake tf)
    loss_vals = loss_fn(y_true, y_pred)

    # Manual CWSL per sample
    def cwsl_np(y_t, y_p, cu, co):
        shortfall = np.maximum(y_t - y_p, 0.0)
        overbuild = np.maximum(y_p - y_t, 0.0)
        cost = cu * shortfall + co * overbuild
        return cost.sum() / max(y_t.sum(), 1e-7)

    expected = np.array(
        [
            cwsl_np(y_true[0], y_pred[0], cu=2.0, co=1.0),
            cwsl_np(y_true[1], y_pred[1], cu=2.0, co=1.0),
        ]
    )

    np.testing.assert_allclose(loss_vals, expected, rtol=1e-6, atol=1e-6)


def test_make_cwsl_keras_loss_import_error(monkeypatch):
    """
    If tensorflow is not importable, the helper should raise ImportError
    with a clear message.
    """
    # Ensure tensorflow isn't already loaded
    sys.modules.pop("tensorflow", None)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "tensorflow":
            raise ImportError("no tf here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        make_cwsl_keras_loss(cu=2.0, co=1.0)

    assert "TensorFlow is required to use make_cwsl_keras_loss" in str(excinfo.value)
