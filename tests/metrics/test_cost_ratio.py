import numpy as np
import pytest

from eb_metrics.metrics import estimate_R_cost_balance


def test_estimate_R_cost_balance_perfect_forecast_prefers_R_near_1():
    """
    If there is no error at all, the estimator should return an R
    close to 1.0 (specifically the grid value closest to 1.0).
    """
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]

    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert R == 1.0


def test_estimate_R_cost_balance_balanced_example_has_known_optimum():
    """
    Construct a toy example where the cost-balance solution is known.

    shortfall_sum = 10
    overbuild_sum = 20

    Under-cost(R) = R * 10
    Over-cost     = 20

    |Under - Over| is minimized at R* = 2.0, which is in the grid.
    """
    # Interval 1: purely shortfall of 10
    # Interval 2: purely overbuild of 20
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([0.0, 20.0])

    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=(0.5, 1.0, 2.0, 3.0),
        co=1.0,
    )

    assert np.isclose(R, 2.0)


def test_estimate_R_cost_balance_respects_sample_weights():
    """
    Check that sample_weight is accepted and doesn't crash.
    We don't assert a specific R here, just that it runs and returns
    a valid scalar from the grid.
    """
    y_true = [10, 0, 5]
    y_pred = [0, 20, 5]
    w = [1.0, 2.0, 0.5]

    R_grid = (0.5, 1.0, 2.0, 3.0)
    R = estimate_R_cost_balance(
        y_true=y_true,
        y_pred=y_pred,
        R_grid=R_grid,
        co=1.0,
        sample_weight=w,
    )

    assert R in R_grid


def test_estimate_R_cost_balance_raises_on_invalid_R_grid():
    """
    R_grid with no positive entries should raise ValueError.
    """
    y_true = [10, 20]
    y_pred = [8, 22]

    with pytest.raises(ValueError):
        estimate_R_cost_balance(
            y_true=y_true,
            y_pred=y_pred,
            R_grid=(0.0, -1.0),
            co=1.0,
        )
