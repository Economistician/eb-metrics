from __future__ import annotations

from typing import Callable


def make_cwsl_keras_loss(cu: float, co: float) -> Callable:
    """
    Create a Keras-compatible loss function implementing Cost-Weighted Service Loss (CWSL).

    This helper mirrors the behavior of the numpy-based CWSL metric but is shaped to work
    as a per-sample loss in Keras / TensorFlow models.

    The returned loss has the signature:

        loss(y_true, y_pred) -> tensor of shape (batch_size,)

    For each sample in the batch:

        shortfall = max(0, y_true - y_pred)
        overbuild = max(0, y_pred - y_true)
        cost = cu * shortfall + co * overbuild

        CWSL_sample = sum(cost_t) / sum(y_true_t)

    Notes
    -----
    - TensorFlow is imported lazily; it is *not* a hard dependency of ebmetrics.
      You only need it installed if you call this helper.
    - y_true and y_pred are expected to be non-negative (demand-like).
    - The reduction is over the last axis for each sample. For a typical shape
      (batch_size, horizon), this matches applying the CWSL metric per series.

    Parameters
    ----------
    cu : float
        Underbuild (shortfall) cost per unit. Must be strictly positive.

    co : float
        Overbuild (excess) cost per unit. Must be strictly positive.

    Returns
    -------
    Callable
        A Keras loss function that can be passed to `model.compile(loss=...)`.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    ValueError
        If cu or co are not strictly positive.
    """
    if cu <= 0.0:
        raise ValueError("cu must be strictly positive.")
    if co <= 0.0:
        raise ValueError("co must be strictly positive.")

    try:
        import tensorflow as tf  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "TensorFlow is required to use make_cwsl_keras_loss. "
            "Install it via `pip install tensorflow`."
        ) from e

    cu_t = tf.constant(float(cu), dtype=tf.float32)
    co_t = tf.constant(float(co), dtype=tf.float32)

    def cwsl_loss(y_true, y_pred):
        """
        Per-sample CWSL loss.

        Expects y_true and y_pred with shape (..., T), where the last axis
        indexes time / intervals. The loss is computed per sample by
        reducing over the last axis.
        """
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)

        # shortfall = max(0, y_true - y_pred)
        shortfall = tf.nn.relu(y_true_f - y_pred_f)
        # overbuild = max(0, y_pred - y_true)
        overbuild = tf.nn.relu(y_pred_f - y_true_f)

        cost = cu_t * shortfall + co_t * overbuild

        # Reduce over the last axis (time / horizon)
        total_cost = tf.reduce_sum(cost, axis=-1)
        total_demand = tf.reduce_sum(y_true_f, axis=-1)

        # Avoid division by zero by clamping with epsilon
        eps = tf.keras.backend.epsilon()
        total_demand_safe = tf.maximum(total_demand, eps)

        return total_cost / total_demand_safe

    return cwsl_loss