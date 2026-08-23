# Electric Barometer · Metrics (`eb-metrics`)

[![CI](https://github.com/Economistician/eb-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-metrics/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-metrics)
![PyPI](https://img.shields.io/pypi/v/eb-metrics)

Asymmetric, readiness-oriented forecast evaluation metrics for operational decision systems.

---

## Overview

`eb-metrics` defines asymmetric, readiness-oriented forecast evaluation primitives (CWSL, NSL, UD, HR@τ, FRS, and related measures). It owns metric math only; evaluation orchestration, policies, and adapters live in sibling packages.

---

## Installation

`eb-metrics` is distributed as a standard Python package.

```bash
pip install eb-metrics
```

The package supports Python 3.11 and later.

---

## Core Concepts

- **Asymmetric error** — Overforecasting and underforecasting can have different operational consequences, so evaluation should reflect directional cost differences.
- **Interval reliability** — In readiness-oriented systems, it matters how often forecasts meet demand within each interval, not just average error over time.
- **Shortfall behavior** — Underbuilding events are operationally distinct; evaluation should capture both their frequency and their severity.
- **Tolerance-based adequacy** — Many systems can absorb small deviations; reliability can be expressed as the frequency of “accurate enough” intervals.
- **Readiness-oriented evaluation** — Forecast quality is assessed by execution feasibility and risk, not solely statistical deviation.

---

## Minimal Example

The following example computes Cost-Weighted Service Loss (CWSL) for a single demand series using asymmetric penalties for underbuild and overbuild:

```python
import numpy as np
from eb_metrics import cwsl

# Realized demand and corresponding forecast
y_true = np.array([20, 28, 32, 35, 40, 42])
y_pred = np.array([22, 25, 29, 36, 37, 45])

# Compute cost-weighted service loss
loss = cwsl(
    y_true=y_true,
    y_pred=y_pred,
    cu=2.0,
    co=1.0,
)

print(loss)
```

---

## License

BSD 3-Clause License.
© 2026 Kyle Corrie.
