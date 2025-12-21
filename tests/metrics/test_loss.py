import numpy as np
import pytest

from eb_metrics.metrics import cwsl


def test_cwsl_two_interval_example():
    """
    Hand-checkable two-interval example.

    Interval 1: y=10, ŷ=8  → shortfall=2, overbuild=0, cost = 2*2 = 4
    Interval 2: y=12, ŷ=15 → shortfall=0, overbuild=3, cost = 1*3 = 3

    total_cost   = 4 + 3 = 7
    total_demand = 10 + 12 = 22

    CWSL = 7 / 22
    """
    y_true = [10, 12]
    y_pred = [8, 15]
    cu = 2.0
    co = 1.0

    value = cwsl(y_true, y_pred, cu=cu, co=co)
    assert np.isclose(value, 7.0 / 22.0)


def test_cwsl_simple_shortfall():
    """
    Simple one-interval shortfall case.

    y_true = 100
    y_pred = 90
    cu = 3, co = 1

    shortfall = 10, overbuild = 0
    cost   = 3 * 10 + 1 * 0 = 30
    demand = 100
    CWSL   = 30 / 100 = 0.30
    """
    value = cwsl(y_true=[100], y_pred=[90], cu=3.0, co=1.0)
    assert np.isclose(value, 0.30)


def test_cwsl_simple_overbuild():
    """
    Simple one-interval overbuild case.

    y_true = 100
    y_pred = 110
    cu = 3, co = 1

    shortfall = 0, overbuild = 10
    cost   = 3 * 0 + 1 * 10 = 10
    demand = 100
    CWSL   = 10 / 100 = 0.10
    """
    value = cwsl(y_true=[100], y_pred=[110], cu=3.0, co=1.0)
    assert np.isclose(value, 0.10)


def test_cwsl_broadcasts_scalar_cu_co():
    """
    Scalar cu/co should broadcast correctly across multiple observations.

    obs 1: y=100, ŷ=90 → shortfall=10, overbuild=0, cost=2*10=20
    obs 2: y=50,  ŷ=60 → shortfall=0,  overbuild=10, cost=1*10=10

    total_cost   = 30
    total_demand = 150
    CWSL         = 30 / 150 = 0.2
    """
    y_true = [100, 50]
    y_pred = [90, 60]

    cu = 2.0
    co = 1.0

    value = cwsl(y_true=y_true, y_pred=y_pred, cu=cu, co=co)
    assert np.isclose(value, 0.20)


def test_cwsl_zero_demand_zero_cost_returns_zero():
    """
    If total demand and total cost are both zero, CWSL is defined as 0.0.

    Example: no demand and perfect zero forecast.
    """
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    value = cwsl(y_true=y_true, y_pred=y_pred, cu=2.0, co=1.0)
    assert np.isclose(value, 0.0)


def test_cwsl_zero_demand_positive_cost_raises():
    """
    If total demand is zero but cost is positive, CWSL is undefined and
    should raise ValueError under this formulation.

    Example: y_true all zero, y_pred positive.
    """
    y_true = [0, 0]
    y_pred = [5, 10]

    with pytest.raises(ValueError):
        cwsl(y_true=y_true, y_pred=y_pred, cu=2.0, co=1.0)
