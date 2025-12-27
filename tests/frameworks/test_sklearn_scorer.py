# tests/frameworks/test_sklearn_scorer.py

from __future__ import annotations

import sys

import numpy as np
import pytest

from eb_metrics.frameworks.sklearn_scorer import cwsl_loss, cwsl_scorer
from eb_metrics.metrics import cwsl


def test_cwsl_loss_matches_core_metric():
    y_true = np.array([10.0, 12.0, 8.0, 15.0])
    y_pred = np.array([9.0, 13.0, 7.0, 14.0])
    cu, co = 2.0, 1.0

    v1 = cwsl_loss(y_true, y_pred, cu=cu, co=co)
    v2 = cwsl(y_true, y_pred, cu=cu, co=co)

    assert np.isclose(v1, v2)


def test_cwsl_scorer_returns_negative_loss_and_respects_sample_weight():
    from sklearn.dummy import DummyRegressor

    X = np.arange(6).reshape(-1, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sw = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    est = DummyRegressor(strategy="constant", constant=3.0)
    est.fit(X, y)

    cu, co = 2.0, 1.0
    scorer = cwsl_scorer(cu=cu, co=co)

    score = scorer(est, X, y, sample_weight=sw)
    expected_loss = cwsl_loss(
        y_true=y,
        y_pred=est.predict(X),
        cu=cu,
        co=co,
        sample_weight=sw,
    )

    # scorer should return NEGATIVE CWSL (higher is better)
    assert np.isclose(score, -expected_loss)


def test_cwsl_scorer_works_in_grid_search():
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GridSearchCV

    # Simple setup where predicting the true constant is clearly best.
    X = np.arange(20).reshape(-1, 1)
    y = np.full(20, 10.0)

    scorer = cwsl_scorer(cu=2.0, co=1.0)

    base = DummyRegressor(strategy="constant")
    grid = GridSearchCV(
        estimator=base,
        param_grid={"constant": [0.0, 10.0]},
        scoring=scorer,
        cv=3,
    )

    grid.fit(X, y)

    # Constant 10.0 matches y exactly → CWSL ≈ 0 → best.
    assert grid.best_params_["constant"] == 10.0


def test_cwsl_scorer_raises_import_error_without_sklearn(monkeypatch):
    # Simulate environment without scikit-learn
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", None)

    from eb_metrics.frameworks import sklearn_scorer as mod

    with pytest.raises(ImportError):
        mod.cwsl_scorer(cu=2.0, co=1.0)
