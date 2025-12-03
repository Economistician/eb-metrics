import numpy as np
import pytest

from ebmetrics.metrics.service import (
    nsl,
    ud,
    hr_at_tau,
    frs,
)


# ----------------------------------------------------------------------
# NSL (No-Shortfall Level)
# ----------------------------------------------------------------------
def test_nsl_basic_unweighted():
    """
    Simple hand-checkable NSL example.

    y_true = [10, 20]
    y_pred = [10, 15]

    hits:
        10 >= 10 → 1
        15 >= 20 → 0

    NSL = (1 + 0) / 2 = 0.5
    """
    y_true = [10, 20]
    y_pred = [10, 15]

    value = nsl(y_true=y_true, y_pred=y_pred)
    assert np.isclose(value, 0.5)


def test_nsl_with_weights():
    """
    Weighted NSL example.

    y_true = [10, 20]
    y_pred = [9, 21]
    weights = [1, 3]

    hits:
        9  >= 10 → 0
        21 >= 20 → 1

    weighted_hits = [1*0, 3*1] = [0, 3]
    total_weight  = 1 + 3 = 4

    NSL_w = 3 / 4 = 0.75
    """
    y_true = [10, 20]
    y_pred = [9, 21]
    w = [1.0, 3.0]

    value = nsl(y_true=y_true, y_pred=y_pred, sample_weight=w)
    assert np.isclose(value, 0.75)


def test_nsl_zero_total_weight_raises():
    """
    If the total sample_weight is zero, NSL should raise ValueError.
    """
    y_true = [10, 20]
    y_pred = [10, 20]
    w = [0.0, 0.0]

    with pytest.raises(ValueError):
        nsl(y_true=y_true, y_pred=y_pred, sample_weight=w)


# ----------------------------------------------------------------------
# UD (Underbuild Depth)
# ----------------------------------------------------------------------
def test_ud_basic_unweighted():
    """
    Simple UD example.

    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]

    shortfalls = max(0, y - ŷ):
        [1, 0, 5]

    UD = mean([1, 0, 5]) = 2.0
    """
    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]

    value = ud(y_true=y_true, y_pred=y_pred)
    assert np.isclose(value, 2.0)


def test_ud_with_weights():
    """
    Weighted UD example.

    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]
    w      = [1, 1, 2]

    shortfalls = [1, 0, 5]

    weighted_shortfall = [1*1, 1*0, 2*5] = [1, 0, 10] → sum = 11
    total_weight = 1 + 1 + 2 = 4

    UD_w = 11 / 4 = 2.75
    """
    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]
    w = [1.0, 1.0, 2.0]

    value = ud(y_true=y_true, y_pred=y_pred, sample_weight=w)
    assert np.isclose(value, 2.75)


def test_ud_zero_total_weight_raises():
    """
    If the total sample_weight is zero, UD should raise ValueError.
    """
    y_true = [10, 20]
    y_pred = [5, 15]
    w = [0.0, 0.0]

    with pytest.raises(ValueError):
        ud(y_true=y_true, y_pred=y_pred, sample_weight=w)


# ----------------------------------------------------------------------
# HR@tau (Hit Rate within Tolerance)
# ----------------------------------------------------------------------
def test_hr_at_tau_scalar_tau():
    """
    HR@tau with a scalar tolerance.

    y_true = [10, 20, 30]
    y_pred = [9, 22, 27]
    tau    = 2

    abs errors = [1, 2, 3]
    hits       = [1, 1, 0]  (<= 2)

    HR = 2 / 3
    """
    y_true = [10, 20, 30]
    y_pred = [9, 22, 27]
    tau = 2.0

    value = hr_at_tau(y_true=y_true, y_pred=y_pred, tau=tau)
    assert np.isclose(value, 2.0 / 3.0)


def test_hr_at_tau_array_tau():
    """
    HR@tau with a per-interval tolerance.

    y_true = [10, 20, 30]
    y_pred = [9, 22, 27]
    tau    = [0.5, 1.5, 5.0]

    abs errors = [1, 2, 3]

    hits:
      1 <= 0.5  → 0
      2 <= 1.5  → 0
      3 <= 5.0  → 1

    HR = 1 / 3
    """
    y_true = [10, 20, 30]
    y_pred = [9, 22, 27]
    tau = [0.5, 1.5, 5.0]

    value = hr_at_tau(y_true=y_true, y_pred=y_pred, tau=tau)
    assert np.isclose(value, 1.0 / 3.0)


def test_hr_at_tau_zero_total_weight_raises():
    """
    If the total sample_weight is zero, HR@tau should raise ValueError.
    """
    y_true = [10, 20]
    y_pred = [11, 19]
    tau = 2.0
    w = [0.0, 0.0]

    with pytest.raises(ValueError):
        hr_at_tau(y_true=y_true, y_pred=y_pred, tau=tau, sample_weight=w)


# ----------------------------------------------------------------------
# FRS (Forecast Readiness Score)
# ----------------------------------------------------------------------
def test_frs_basic_consistency():
    """
    Basic FRS example using NSL and CWSL.

    y_true = [10, 20]
    y_pred = [10, 15]
    cu = 2, co = 1

    NSL:
        hits = [1, 0] → NSL = 0.5

    CWSL:
        shortfall = [0, 5]
        overbuild = [0, 0]
        cost      = 2*5 = 10
        demand    = 10 + 20 = 30
        CWSL      = 10 / 30 = 1/3

    FRS = NSL - CWSL = 0.5 - 1/3 = 1/6
    """
    y_true = [10, 20]
    y_pred = [10, 15]
    cu = 2.0
    co = 1.0

    value = frs(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
    assert np.isclose(value, 1.0 / 6.0)