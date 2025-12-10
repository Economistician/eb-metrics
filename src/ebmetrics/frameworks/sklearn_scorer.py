# src/ebmetrics/frameworks/sklearn_scorer.py
from __future__ import annotations

from typing import Callable

import numpy as np

from ebmetrics.metrics import cwsl


def cwsl_loss(
    y_true,
    y_pred,
    *,
    cu: float,
    co: float,
    sample_weight=None,
) -> float:
    """
    Raw CWSL loss function, suitable for use with sklearn-style scorers.

    This returns a *positive* cost (loss). When wrapped with
    sklearn.metrics.make_scorer(greater_is_better=False), the resulting
    scorer will return the *negative* CWSL so that higher scores are
    better, per scikit-learn conventions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth targets.

    y_pred : array-like
        Predicted values from the model.

    cu : float
        Underbuild (shortfall) cost per unit. Must be strictly positive.

    co : float
        Overbuild (excess) cost per unit. Must be strictly positive.

    sample_weight : array-like, optional
        Optional per-sample weights.

    Returns
    -------
    float
        Positive CWSL value (lower is better).
    """
    if cu <= 0.0:
        raise ValueError("cu must be strictly positive.")
    if co <= 0.0:
        raise ValueError("co must be strictly positive.")

    return cwsl(
        y_true=np.asarray(y_true, dtype=float),
        y_pred=np.asarray(y_pred, dtype=float),
        cu=cu,
        co=co,
        sample_weight=sample_weight,
    )


def cwsl_scorer(cu: float, co: float) -> Callable:
    """
    Build a scikit-learn-compatible scorer based on Cost-Weighted Service Loss (CWSL).

    The returned object can be passed anywhere scikit-learn expects a ``scorer``,
    such as GridSearchCV, RandomizedSearchCV, or cross_val_score.

    Notes
    -----
    - The underlying CWSL is a *loss* (lower is better).
    - The scorer returned by this function obeys scikit-learn conventions:
        * it returns the *negative* CWSL (so higher is better),
        * it can be used directly as ``scoring=...`` in model selection tools.

    Parameters
    ----------
    cu : float
        Underbuild (shortfall) cost per unit. Must be strictly positive.

    co : float
        Overbuild (excess) cost per unit. Must be strictly positive.

    Returns
    -------
    Callable
        A scikit-learn scorer object (as returned by ``sklearn.metrics.make_scorer``).

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if cu <= 0.0:
        raise ValueError("cu must be strictly positive.")
    if co <= 0.0:
        raise ValueError("co must be strictly positive.")

    try:
        from sklearn.metrics import make_scorer
    except ImportError as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "cwsl_scorer requires scikit-learn to be installed. "
            "Install it with `pip install scikit-learn`."
        ) from e

    def _loss(y_true, y_pred, sample_weight=None, **kwargs):
        # Accept **kwargs so sklearn can pass needs_proba / needs_threshold, etc.
        return cwsl_loss(
            y_true=y_true,
            y_pred=y_pred,
            cu=cu,
            co=co,
            sample_weight=sample_weight,
        )

    # greater_is_better=False → sklearn negates the loss internally,
    # so the scorer returns -CWSL and can be maximized.
    return make_scorer(
        _loss,
        greater_is_better=False,
        needs_proba=False,
        needs_threshold=False,
    )