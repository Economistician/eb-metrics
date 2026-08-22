"""
Property-based mathematical stress tests for core metric primitives.

These tests evaluate theoretical properties of ``cwsl``, ``nsl``, ``ud``,
``hr_at_tau``, and ``frs`` — not wrapper plumbing. Random draws use a fixed
NumPy Generator so failures are reproducible without adding Hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest

from eb_metrics.metrics import cwsl, frs, hr_at_tau, nsl, ud

N_TRIALS = 80
CWSL_MAX = 0.30
RTOL = 1e-12
ATOL = 1e-12


def _rng(seed: int = 20260822) -> np.random.Generator:
    return np.random.default_rng(seed)


def _assert_py_float(value: object, *, name: str) -> float:
    assert type(value) is float, f"{name} must return a Python float; got {type(value)!r}"
    assert np.isfinite(value), f"{name} must be finite; got {value!r}"
    return value


def _random_positive_series(
    rng: np.random.Generator,
    n: int,
    *,
    lo: float = 0.05,
    hi: float = 50.0,
) -> np.ndarray:
    return rng.uniform(lo, hi, size=n)


def _random_costs(rng: np.random.Generator) -> tuple[float, float]:
    cu = float(rng.uniform(0.25, 5.0))
    co = float(rng.uniform(0.25, 5.0))
    return cu, co


def _all_core(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    cu: float = 2.0,
    co: float = 1.0,
    tau: float = 1.0,
    cwsl_max: float = CWSL_MAX,
) -> dict[str, float]:
    return {
        "cwsl": cwsl(y_true, y_pred, cu=cu, co=co),
        "nsl": nsl(y_true, y_pred),
        "ud": ud(y_true, y_pred),
        "hr_at_tau": hr_at_tau(y_true, y_pred, tau=tau),
        "frs": frs(y_true, y_pred, cu=cu, co=co, cwsl_max=cwsl_max),
    }


# ----------------------------------------------------------------------
# 1. Monotonicity and asymmetry
# ----------------------------------------------------------------------
def test_cwsl_increases_strictly_in_cu_on_shortfall_series():
    """Raising cu must raise CWSL whenever every interval is a shortfall."""
    rng = _rng(1)
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 8))
        y_true = _random_positive_series(rng, n)
        gap = rng.uniform(0.05, 4.0, size=n)
        y_pred = np.maximum(y_true - gap, 0.0)
        # Keep a strict shortfall on at least one interval after the clip.
        if not np.any(y_true > y_pred):
            y_true = y_true + 1.0
        cu, co = _random_costs(rng)
        cu_hi = cu + float(rng.uniform(0.1, 3.0))

        low = _assert_py_float(cwsl(y_true, y_pred, cu=cu, co=co), name="cwsl")
        high = _assert_py_float(cwsl(y_true, y_pred, cu=cu_hi, co=co), name="cwsl")
        assert high > low + 1e-15


def test_cwsl_invariant_to_cu_when_no_shortfall():
    """cu cannot appear in CWSL when y_pred >= y_true everywhere."""
    rng = _rng(2)
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 8))
        y_true = _random_positive_series(rng, n)
        y_pred = y_true + rng.uniform(0.0, 4.0, size=n)
        cu, co = _random_costs(rng)
        cu_hi = cu + float(rng.uniform(0.1, 3.0))

        a = cwsl(y_true, y_pred, cu=cu, co=co)
        b = cwsl(y_true, y_pred, cu=cu_hi, co=co)
        assert np.isclose(a, b, rtol=RTOL, atol=ATOL)


def test_cwsl_increases_strictly_in_co_on_overbuild_series():
    """Raising co must raise CWSL whenever every interval is an overbuild."""
    rng = _rng(3)
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 8))
        y_true = _random_positive_series(rng, n)
        y_pred = y_true + rng.uniform(0.05, 4.0, size=n)
        cu, co = _random_costs(rng)
        co_hi = co + float(rng.uniform(0.1, 3.0))

        low = _assert_py_float(cwsl(y_true, y_pred, cu=cu, co=co), name="cwsl")
        high = _assert_py_float(cwsl(y_true, y_pred, cu=cu, co=co_hi), name="cwsl")
        assert high > low + 1e-15


def test_cwsl_invariant_to_co_when_no_overbuild():
    """co cannot appear in CWSL when y_pred <= y_true everywhere."""
    rng = _rng(4)
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 8))
        y_true = _random_positive_series(rng, n)
        y_pred = np.maximum(y_true - rng.uniform(0.0, 4.0, size=n), 0.0)
        cu, co = _random_costs(rng)
        co_hi = co + float(rng.uniform(0.1, 3.0))

        a = cwsl(y_true, y_pred, cu=cu, co=co)
        b = cwsl(y_true, y_pred, cu=cu, co=co_hi)
        assert np.isclose(a, b, rtol=RTOL, atol=ATOL)


def test_nsl_unchanged_until_forecast_crosses_demand():
    """Raising ŷ while it remains below y does not change NSL."""
    rng = _rng(5)
    for _ in range(N_TRIALS):
        y_true = float(rng.uniform(2.0, 20.0))
        y_pred = float(rng.uniform(0.0, y_true * 0.6))
        y_mid = y_pred + 0.4 * (y_true - y_pred)
        assert y_mid < y_true

        a = nsl([y_true], [y_pred])
        b = nsl([y_true], [y_mid])
        assert np.isclose(a, 0.0, atol=ATOL)
        assert np.isclose(b, 0.0, atol=ATOL)


def test_nsl_jumps_when_forecast_crosses_demand():
    """Crossing ŷ from below y to at/above y strictly raises NSL."""
    rng = _rng(6)
    for _ in range(N_TRIALS):
        n = int(rng.integers(2, 6))
        y_true = _random_positive_series(rng, n, lo=1.0, hi=20.0)
        y_pred = y_true.copy()
        i = int(rng.integers(0, n))
        y_pred[i] = y_true[i] * 0.5
        y_crossed = y_pred.copy()
        y_crossed[i] = y_true[i]

        before = nsl(y_true, y_pred)
        after = nsl(y_true, y_crossed)
        assert after > before + 1e-15
        assert np.isclose(after - before, 1.0 / n, rtol=RTOL, atol=ATOL)


def test_nsl_and_ud_unchanged_once_forecast_covers_demand():
    """Once ŷ >= y, further increases leave NSL = 1 and UD = 0."""
    rng = _rng(7)
    for _ in range(N_TRIALS):
        y_true = float(rng.uniform(0.5, 20.0))
        y_pred = y_true + float(rng.uniform(0.0, 5.0))
        y_hi = y_pred + float(rng.uniform(0.1, 5.0))

        assert np.isclose(nsl([y_true], [y_pred]), 1.0)
        assert np.isclose(nsl([y_true], [y_hi]), 1.0)
        assert np.isclose(ud([y_true], [y_pred]), 0.0)
        assert np.isclose(ud([y_true], [y_hi]), 0.0)


def test_ud_decreases_when_shortfall_forecast_rises_but_stays_below():
    """On a single shortfall interval, UD = y - ŷ, so raising ŷ lowers UD."""
    rng = _rng(8)
    for _ in range(N_TRIALS):
        y_true = float(rng.uniform(2.0, 25.0))
        y_lo = float(rng.uniform(0.0, y_true * 0.4))
        y_hi = y_lo + 0.4 * (y_true - y_lo)
        assert y_hi < y_true

        ud_lo = _assert_py_float(ud([y_true], [y_lo]), name="ud")
        ud_hi = _assert_py_float(ud([y_true], [y_hi]), name="ud")
        assert ud_hi < ud_lo - 1e-15
        assert np.isclose(ud_lo, y_true - y_lo, rtol=RTOL, atol=ATOL)
        assert np.isclose(ud_hi, y_true - y_hi, rtol=RTOL, atol=ATOL)


def test_ud_can_increase_when_a_small_shortfall_exits_the_conditioning_set():
    """
    Conditional UD is an average over T^SF only.

    Removing a *small* shortfall (by lifting that ŷ across y) can raise the
    remaining average. This is required by the definition, not a defect.
    """
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([9.0, 1.0])  # depths 1 and 9; UD = 5
    y_after = np.array([10.0, 1.0])  # only the deep shortfall remains; UD = 9

    assert np.isclose(ud(y_true, y_pred), 5.0)
    assert np.isclose(ud(y_true, y_after), 9.0)
    assert ud(y_true, y_after) > ud(y_true, y_pred)


# ----------------------------------------------------------------------
# 2. Boundary and scale invariance
# ----------------------------------------------------------------------
def test_perfect_forecast_canonical_values():
    """ŷ = y implies CWSL=0, NSL=1, UD=0, HR@τ=1, FRS=1."""
    rng = _rng(9)
    taus = [0.0, 1e-12, 0.5, 10.0, 1e6]
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 10))
        y = _random_positive_series(rng, n)
        cu, co = _random_costs(rng)
        tau = float(rng.choice(taus))
        values = _all_core(y, y.copy(), cu=cu, co=co, tau=tau)
        for name, value in values.items():
            _assert_py_float(value, name=name)
        assert values["cwsl"] == 0.0
        assert values["nsl"] == 1.0
        assert values["ud"] == 0.0
        assert values["hr_at_tau"] == 1.0
        assert values["frs"] == 1.0


def test_perfect_zero_forecast_is_well_defined():
    """All-zero demand and forecast: CWSL is defined as 0; service metrics hit."""
    y = np.zeros(4)
    values = _all_core(y, y.copy(), cu=3.0, co=1.0, tau=0.0)
    for name, value in values.items():
        _assert_py_float(value, name=name)
    assert values["cwsl"] == 0.0
    assert values["nsl"] == 1.0
    assert values["ud"] == 0.0
    assert values["hr_at_tau"] == 1.0
    assert values["frs"] == 1.0


def test_multiplicative_scale_invariance_nsl_cwsl_frs():
    """
    Scaling (y, ŷ) by c > 0 leaves NSL, CWSL, and FRS unchanged.

    CWSL costs and demand both scale by c; the ratio is invariant.
    NSL is a comparison of orderings. FRS is a function of those two.
    """
    rng = _rng(10)
    scales = [1e-6, 0.01, 0.5, 2.0, 7.5, 1e3]
    for _ in range(N_TRIALS):
        n = int(rng.integers(2, 8))
        y_true = _random_positive_series(rng, n)
        y_pred = _random_positive_series(rng, n)
        cu, co = _random_costs(rng)
        c = float(rng.choice(scales))

        base = {
            "nsl": nsl(y_true, y_pred),
            "cwsl": cwsl(y_true, y_pred, cu=cu, co=co),
            "frs": frs(y_true, y_pred, cu=cu, co=co, cwsl_max=CWSL_MAX),
        }
        scaled = {
            "nsl": nsl(c * y_true, c * y_pred),
            "cwsl": cwsl(c * y_true, c * y_pred, cu=cu, co=co),
            "frs": frs(c * y_true, c * y_pred, cu=cu, co=co, cwsl_max=CWSL_MAX),
        }
        for name in base:
            _assert_py_float(scaled[name], name=name)
            assert np.isclose(base[name], scaled[name], rtol=1e-10, atol=1e-12), name


def test_hr_is_scale_invariant_only_when_tau_is_scaled():
    """
    |c y - c ŷ| = c |y - ŷ|, so HR@τ is invariant iff τ also scales by c.

    A fixed absolute band is *not* a scale-free property.
    """
    rng = _rng(11)
    for _ in range(N_TRIALS):
        n = int(rng.integers(2, 8))
        y_true = _random_positive_series(rng, n)
        y_pred = _random_positive_series(rng, n)
        tau = float(rng.uniform(0.1, 5.0))
        c = float(rng.choice([0.25, 2.0, 10.0]))

        base = hr_at_tau(y_true, y_pred, tau=tau)
        scaled_series_only = hr_at_tau(c * y_true, c * y_pred, tau=tau)
        scaled_with_tau = hr_at_tau(c * y_true, c * y_pred, tau=c * tau)

        _assert_py_float(scaled_with_tau, name="hr_at_tau")
        assert np.isclose(base, scaled_with_tau, rtol=1e-12, atol=1e-12)
        # Fixed-τ HR must move whenever some errors straddle τ vs cτ.
        errors = np.abs(y_true - y_pred)
        straddles = np.any((errors <= tau) != (c * errors <= tau))
        if straddles:
            assert not np.isclose(base, scaled_series_only, rtol=1e-12, atol=1e-12)


def test_ud_scales_with_demand_units():
    """UD is a magnitude in units of y, so it must scale by c."""
    rng = _rng(12)
    for _ in range(N_TRIALS):
        n = int(rng.integers(2, 8))
        y_true = _random_positive_series(rng, n, lo=1.0)
        y_pred = np.maximum(y_true - rng.uniform(0.2, 3.0, size=n), 0.0)
        if not np.any(y_true > y_pred):
            continue
        c = float(rng.choice([0.5, 2.0, 8.0]))
        base = ud(y_true, y_pred)
        scaled = ud(c * y_true, c * y_pred)
        assert np.isclose(scaled, c * base, rtol=1e-10, atol=1e-12)


# ----------------------------------------------------------------------
# 3. Numerical extremes and pathologies
# ----------------------------------------------------------------------
@pytest.mark.parametrize("y_true", [1e-12, 1e-15, 1e-8])
def test_near_zero_demand_with_matching_forecast_is_finite(y_true: float):
    values = _all_core(np.array([y_true]), np.array([y_true]), cu=2.0, co=1.0, tau=0.0)
    for name, value in values.items():
        _assert_py_float(value, name=name)
    assert values["cwsl"] == 0.0
    assert values["nsl"] == 1.0
    assert values["ud"] == 0.0
    assert values["frs"] == 1.0


def test_near_zero_demand_with_zero_forecast_recovers_cu():
    """CWSL = cu * (y - 0) / y = cu when ŷ = 0 and y is tiny but positive."""
    y_true = np.array([1e-12])
    y_pred = np.array([0.0])
    cu, co = 2.5, 1.0
    value = _assert_py_float(cwsl(y_true, y_pred, cu=cu, co=co), name="cwsl")
    assert np.isclose(value, cu, rtol=1e-9, atol=1e-12)
    assert np.isclose(nsl(y_true, y_pred), 0.0)
    assert np.isclose(ud(y_true, y_pred), 1e-12, rtol=1e-9, atol=0.0)


def test_near_zero_demand_with_unit_overbuild_is_large_but_finite():
    """CWSL ~ co * (1 - ε) / ε is huge, but must stay a finite Python float."""
    y_true = np.array([1e-12])
    y_pred = np.array([1.0])
    value = _assert_py_float(cwsl(y_true, y_pred, cu=2.0, co=1.0), name="cwsl")
    assert value > 1e11
    frs_val = _assert_py_float(
        frs(y_true, y_pred, cu=2.0, co=1.0, cwsl_max=CWSL_MAX), name="frs"
    )
    # NSL = 1 (overbuild covers demand); CWSL clips to 1 → FRS = 0.
    assert np.isclose(nsl(y_true, y_pred), 1.0)
    assert np.isclose(frs_val, 0.0)


def test_zero_demand_positive_forecast_cwsl_and_frs_are_undefined():
    """Documented pathology: zero demand and positive cost makes CWSL undefined."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="undefined"):
        cwsl(y_true, y_pred, cu=2.0, co=1.0)
    with pytest.raises(ValueError, match="undefined"):
        frs(y_true, y_pred, cu=2.0, co=1.0, cwsl_max=CWSL_MAX)

    # Service metrics remain defined: no shortfall, zero depth, miss unless τ is large.
    assert nsl(y_true, y_pred) == 1.0
    assert ud(y_true, y_pred) == 0.0
    assert hr_at_tau(y_true, y_pred, tau=0.0) == 0.0
    assert hr_at_tau(y_true, y_pred, tau=2.0) == 1.0


def test_outputs_are_python_floats_on_random_in_domain_draws():
    rng = _rng(13)
    for _ in range(N_TRIALS):
        n = int(rng.integers(1, 12))
        y_true = _random_positive_series(rng, n, lo=1e-6, hi=1e3)
        y_pred = _random_positive_series(rng, n, lo=0.0, hi=1e3)
        cu, co = _random_costs(rng)
        tau = float(rng.uniform(0.0, 25.0))
        values = _all_core(y_true, y_pred, cu=cu, co=co, tau=tau)
        for name, value in values.items():
            _assert_py_float(value, name=name)
        assert 0.0 <= values["nsl"] <= 1.0
        assert 0.0 <= values["hr_at_tau"] <= 1.0
        assert values["ud"] >= 0.0
        assert values["cwsl"] >= 0.0
        assert -1.0 <= values["frs"] <= 1.0


def test_frs_bounds_hold_under_extreme_cost_ratios():
    """FRS must stay in [-1, 1] even when raw CWSL is far above CWSL_max."""
    rng = _rng(14)
    for _ in range(N_TRIALS):
        y_true = _random_positive_series(rng, 5)
        y_pred = _random_positive_series(rng, 5)
        cu = float(rng.choice([1e-3, 1.0, 50.0, 1e3]))
        co = float(rng.choice([1e-3, 1.0, 50.0, 1e3]))
        value = _assert_py_float(
            frs(y_true, y_pred, cu=cu, co=co, cwsl_max=CWSL_MAX), name="frs"
        )
        assert -1.0 <= value <= 1.0


def test_random_shortfall_and_overbuild_cu_co_partial_derivatives_match_definition():
    """
    On mixed series, ΔCWSL / Δcu equals total shortfall / total demand,
    and ΔCWSL / Δco equals total overbuild / total demand.
    """
    rng = _rng(15)
    for _ in range(N_TRIALS):
        n = int(rng.integers(4, 10))
        y_true = _random_positive_series(rng, n)
        y_pred = _random_positive_series(rng, n)
        shortfall = np.maximum(0.0, y_true - y_pred)
        overbuild = np.maximum(0.0, y_pred - y_true)
        demand = float(y_true.sum())
        cu, co = 1.0, 1.0
        dcu, dco = 0.37, 0.41

        if shortfall.sum() > 0:
            delta = cwsl(y_true, y_pred, cu=cu + dcu, co=co) - cwsl(
                y_true, y_pred, cu=cu, co=co
            )
            expected = dcu * float(shortfall.sum()) / demand
            assert np.isclose(delta, expected, rtol=1e-10, atol=1e-12)

        if overbuild.sum() > 0:
            delta = cwsl(y_true, y_pred, cu=cu, co=co + dco) - cwsl(
                y_true, y_pred, cu=cu, co=co
            )
            expected = dco * float(overbuild.sum()) / demand
            assert np.isclose(delta, expected, rtol=1e-10, atol=1e-12)
