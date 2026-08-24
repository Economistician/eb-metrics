from __future__ import annotations

import numpy as np
import pytest

from eb_metrics.metrics.service import (
    cwsl_sensitivity,
    frs,
    hr_at_tau,
    nsl,
    ud,
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


def test_nsl_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        nsl([], [])


def test_nsl_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        nsl([1.0, float("nan")], [1.0, 1.0])


# ----------------------------------------------------------------------
# UD (Underbuild Depth)
# ----------------------------------------------------------------------
def test_ud_basic_unweighted():
    """
    Conditional UD over shortfall intervals only.

    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]

    shortfalls = max(0, y - ŷ):
        [1, 0, 5]

    T^SF = {0, 2} with depths [1, 5]
    UD = (1 + 5) / 2 = 3.0
    """
    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]

    value = ud(y_true=y_true, y_pred=y_pred)
    assert np.isclose(value, 3.0)


def test_ud_technical_note_twelve_interval_example():
    """
    Worked example from the Underbuild Depth technical note.

    Shortfalls in intervals 2, 3, 5, 7, and 9 with depths 3, 3, 3, 4, 4.
    UD = (3 + 3 + 3 + 4 + 4) / 5 = 17 / 5 = 3.4
    """
    y_true = [20, 28, 32, 35, 40, 42, 38, 30, 26, 24, 22, 18]
    y_pred = [22, 25, 29, 36, 37, 45, 34, 31, 22, 27, 23, 19]

    value = ud(y_true=y_true, y_pred=y_pred)
    assert np.isclose(value, 17.0 / 5.0)


def test_ud_with_weights():
    """
    Weighted UD, still conditional on shortfall intervals.

    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]
    w      = [1, 1, 2]

    shortfalls = [1, 0, 5]
    T^SF weights = [1, 2]

    weighted_shortfall = 1*1 + 2*5 = 11
    shortfall_weight   = 1 + 2 = 3

    UD_w = 11 / 3
    """
    y_true = [10, 20, 30]
    y_pred = [9, 25, 25]
    w = [1.0, 1.0, 2.0]

    value = ud(y_true=y_true, y_pred=y_pred, sample_weight=w)
    assert np.isclose(value, 11.0 / 3.0)


def test_ud_no_shortfalls_returns_zero():
    """If every interval meets or exceeds demand, UD is 0.0."""
    y_true = [10, 20, 30]
    y_pred = [10, 25, 31]

    value = ud(y_true=y_true, y_pred=y_pred)
    assert np.isclose(value, 0.0)


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
def _frs_example_inputs() -> tuple[list[int], list[int], float, float]:
    """Shared hand-checkable FRS example: NSL = 0.5, CWSL = 1/3."""
    return [10, 20], [10, 15], 2.0, 1.0


def test_frs_scales_cwsl_by_cwsl_max():
    """
    FRS scales CWSL by a required CWSL_max bound.

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

    CWSL_max = 0.5
    CWSL_scaled = min(1, (1/3) / 0.5) = 2/3
    FRS = NSL - CWSL_scaled = 0.5 - 2/3 = -1/6
    """
    y_true, y_pred, cu, co = _frs_example_inputs()

    value = frs(y_true=y_true, y_pred=y_pred, cu=cu, co=co, cwsl_max=0.5)
    assert np.isclose(value, -1.0 / 6.0)


def test_frs_clips_scaled_cwsl_at_one():
    """
    If CWSL / CWSL_max exceeds 1, the scaled term is clipped at 1.

    Using the shared example (NSL = 0.5, CWSL = 1/3) with CWSL_max = 0.2:

    CWSL_scaled = min(1, (1/3) / 0.2) = 1
    FRS = 0.5 - 1 = -0.5
    """
    y_true, y_pred, cu, co = _frs_example_inputs()

    value = frs(y_true=y_true, y_pred=y_pred, cu=cu, co=co, cwsl_max=0.2)
    assert np.isclose(value, -0.5)


@pytest.mark.parametrize("bad_R", [float("nan"), float("inf"), float("-inf")])
def test_cwsl_sensitivity_rejects_nonfinite_R(bad_R: float):
    """NaN and inf in R_list are invalid grid points, not silently skipped."""
    with pytest.raises(ValueError, match="finite"):
        cwsl_sensitivity(
            y_true=[10.0, 20.0],
            y_pred=[9.0, 21.0],
            R_list=[1.0, bad_R],
            co=1.0,
        )


@pytest.mark.parametrize("cwsl_max", [0.0, -0.3])
def test_frs_rejects_nonpositive_cwsl_max(cwsl_max: float):
    """cwsl_max must be finite and strictly greater than 0."""
    y_true, y_pred, cu, co = _frs_example_inputs()

    with pytest.raises(ValueError):
        frs(y_true=y_true, y_pred=y_pred, cu=cu, co=co, cwsl_max=cwsl_max)
