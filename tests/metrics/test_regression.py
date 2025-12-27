import numpy as np
import pytest

from eb_metrics.metrics.regression import (
    mae,
    mape,
    mase,
    medae,
    mse,
    msle,
    rmse,
    rmsle,
    smape,
    wmape,
)


# ----------------------------------------------------------------------
# Basic sanity checks
# ----------------------------------------------------------------------
def test_mae_simple():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # abs errors = [2, 2, 3] → mean = 7/3
    assert np.isclose(mae(y_true, y_pred), 7 / 3)


def test_mse_rmse_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 1, 1]
    # squared errors: [0,1,4] → mean = 5/3 → rmse = sqrt(5/3)
    assert np.isclose(mse(y_true, y_pred), 5 / 3)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(5 / 3))


def test_mape_basic():
    y_true = np.array([100, 200])
    y_pred = np.array([90, 220])
    # pct errors: [0.10, 0.10] → mape = 10%
    assert np.isclose(mape(y_true, y_pred), 10.0)


def test_wmape_basic():
    y_true = [100, 50]
    y_pred = [110, 40]
    # abs errors: 10 + 10 = 20
    # denom: 150
    assert np.isclose(wmape(y_true, y_pred), (20 / 150) * 100)


# ----------------------------------------------------------------------
# Log metrics
# ----------------------------------------------------------------------
def test_msle_rmsle_basic():
    y_true = [10, 20]
    y_pred = [12, 18]
    msle_val = msle(y_true, y_pred)
    rmsle_val = rmsle(y_true, y_pred)
    assert msle_val >= 0
    assert np.isclose(rmsle_val, np.sqrt(msle_val))


def test_msle_rejects_negative():
    with pytest.raises(ValueError):
        msle([10, -1], [10, 10])


# ----------------------------------------------------------------------
# Robust & forecasting metrics
# ----------------------------------------------------------------------
def test_medae_basic():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 29]
    # absolute errors = [2,2,1] → median = 2
    assert np.isclose(medae(y_true, y_pred), 2.0)


def test_smape_basic():
    y_true = [100, 200]
    y_pred = [110, 190]
    val = smape(y_true, y_pred)
    assert 0 <= val <= 200  # general range check


def test_mase_basic():
    y_true = [10, 12, 15]
    y_pred = [11, 13, 15]
    y_naive = [10, 10, 12]  # lag-1 naive
    val = mase(y_true, y_pred, y_naive)
    assert val >= 0


def test_mase_raises_on_zero_naive_mae():
    y_true = [10, 10, 10]
    y_pred = [10, 10, 10]
    y_naive = [10, 10, 10]

    # naive MAE = 0 → MASE undefined
    with pytest.raises(ValueError):
        mase(y_true, y_pred, y_naive)


# ----------------------------------------------------------------------
# Shape mismatch tests
# ----------------------------------------------------------------------
def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
