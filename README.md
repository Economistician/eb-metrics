# EB Metrics

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%2B-blue)
![Project Status](https://img.shields.io/badge/Status-Alpha-yellow)

**Error metrics and evaluation utilities for the Electric Barometer ecosystem**

`eb-metrics` is the core metric library of the Electric Barometer (EB) framework — a forecasting evaluation system built for operational, high-frequency forecasting domains such as QSR (quick-service restaurants), retail, logistics, energy, and any environment where forecast *direction*, *magnitude*, and *operational cost* matter.

This library provides:

- **Cost-weighted loss metrics** (e.g., CWSL) that explicitly model asymmetric operational costs  
- **Classical regression metrics** (MAE, RMSE, MAPE, sMAPE, MASE, etc.)  
- **Service-level metrics** (NSL, UD, HR@τ, FRS) that measure readiness and real-world performance  
- A consistent, well-tested API designed for easy integration into model training pipelines or the Electric Barometer framework

`eb-metrics` is lightweight, dependency-minimal, and fully unit-tested.

---

## Installation

Until the first public release, `eb-metrics` is installed directly from source.

### **Prerequisites**
- Python **3.10+**
- `git`
- `pip` and `venv` (standard with Python)

### **Install from Source (Recommended)**

Clone the repository:

```bash
git clone https://github.com/Economistician/eb-metrics.git
cd eb-metrics
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

Install the package in editable mode (for development):

```bash
pip install -e ".[dev]"
```

This installs:

- The ebmetrics library
- All built-in metric modules
- Developer tools for testing and validation

---

## Quick Start

The examples below demonstrate the core functionality of `eb-metrics`, including
cost-weighted loss metrics, classical regression metrics, and service-level metrics.

### **Basic Usage**

```python
from ebmetrics.metrics import cwsl, mae, mape, nsl, frs

y_true = [10, 12, 8, 15]
y_pred = [8, 14, 7, 15]

# Cost weights for underbuild (cu) and overbuild (co)
cu = 2.0   # shortfall cost
co = 1.0   # overbuild cost

# Cost-Weighted Service Loss
value = cwsl(y_true, y_pred, cu=cu, co=co)
print("CWSL:", value)

# Classical metrics
print("MAE:", mae(y_true, y_pred))
print("MAPE:", mape(y_true, y_pred))

# Service-level metrics
print("NSL:", nsl(y_true, y_pred))   # No-Shortfall Level
print("FRS:", frs(y_true, y_pred, cu=cu, co=co))   # Forecast Readiness Score
```

### **Weighted Example**

```python
import numpy as np
from ebmetrics.metrics import cwsl

y_true = np.array([100, 200, 50])
y_pred = np.array([90, 210, 40])
weights = np.array([1.0, 2.0, 0.5])

cu, co = 2.0, 1.0

value = cwsl(y_true, y_pred, cu=cu, co=co, sample_weight=weights)
print("Weighted CWSL:", value)
```

### **Why CWSL?**

CWSL generalizes wMAPE by introducing explicit **asymmetric operational costs**:

- Shortfalls can be penalized more heavily (e.g., lost sales, service failures)
- Overbuilds can be penalized differently (e.g., waste, prep labor)
- When cu == co, CWSL ≈ wMAPE
- When cu > co, the metric favors forecasts that avoid shortfalls

This makes CWSL ideal for **operational forecasting domains** such as QSR, retail, logistics, and inventory-sensitive environments.

---

## Available Metrics

`eb-metrics` provides three categories of evaluation metrics designed for
operational forecasting, model comparison, and real-world readiness analysis.

---

### **1. Cost-Weighted Loss Metrics (Asymmetric Error Metrics)**

These metrics explicitly encode operational costs for underforecasting (shortfalls)
and overforecasting (excess). Ideal for QSR, retail, logistics, and any domain where
being “short” is more expensive than being “long.”

| Metric | Description |
|--------|-------------|
| **`cwsl`** | *Cost-Weighted Service Loss.* Directionally-aware, demand-normalized error with asymmetric penalties. Generalizes wMAPE. |

**Usage example:**

```python
from ebmetrics.metrics import cwsl
cwsl(y_true, y_pred, cu=2.0, co=1.0)
```

### **2. Classical Regression Metrics**

Industry-standard error metrics for baseline comparison, diagnostic checks,
or compatibility with established ML workflows.

| Metric  | Description                                      |
| ------- | ------------------------------------------------ |
| `mae`   | Mean Absolute Error                              |
| `mse`   | Mean Squared Error                               |
| `rmse`  | Root Mean Squared Error                          |
| `mape`  | Mean Absolute Percentage Error                   |
| `wmape` | Weighted MAPE (demand-normalized absolute error) |
| `smape` | Symmetric MAPE                                   |
| `msle`  | Mean Squared Log Error                           |
| `rmsle` | Root MSLE                                        |
| `medae` | Median Absolute Error                            |
| `mase`  | Mean Absolute Scaled Error                       |

**Usage**:

```python
from ebmetrics.metrics import rmse, wmape
rmse(y_true, y_pred)
wmape(y_true, y_pred)
```

### **3. Service-Level Metrics (Operational Performance Metrics)**

Metrics specifically designed for operational readiness, service consistency, and
forecast quality in high-frequency environments.

| Metric      | Description                                                     |
| ----------- | --------------------------------------------------------------- |
| `nsl`       | *No Shortfall Level.* Fraction of intervals with no underbuild. |
| `ud`        | *Underbuild Depth.* Average shortfall amount.                   |
| `hr_at_tau` | Hit Rate within a tolerance `τ`.                                |
| `frs`       | *Forecast Readiness Score.* Combines NSL and CWSL.              |

**Example**:

```python
from ebmetrics.metrics import nsl, frs
nsl(y_true, y_pred)
frs(y_true, y_pred, cu=2.0, co=1.0)
```

### **4. Cost Ratio Estimation (Choosing R = cu/co)

`eb-metrics` includes a data-driven helper for estimating the **optimal cost ratio**

\[R = \frac{c_u}{c_o}\]

based on historical forecast performance.  
This allows users to ground asymmetric cost assumptions in empirical behavior rather than arbitrary choices.

#### **Why estimate R?**

- Many operational domains (QSR, retail, logistics) do not know their exact shortfall-vs-overbuild cost asymmetry.
- Forecast error patterns often reveal an implicit operating preference.
- The estimator finds the value of \(R\) where:

\[\text{total underbuild cost} \approx \text{total overbuild cost}\]

producing a stable, interpretable baseline cost ratio.

#### **Global Cost-Ratio Estimation**

```python
from ebmetrics.metrics import estimate_R_cost_balance

R_hat = estimate_R_cost_balance(
    y_true,
    y_pred,
    R_grid=[0.5, 1.0, 2.0, 3.0],
    co=1.0,
)
```

`R_hat` is the value of R that minimizes: The Sum of shortfall less the sum of overbuild

You can use `R_hat` directly, or as the center of a sensitivity sweep:

```python
[R_hat / 2, R_hat, R_hat * 2]
```

**Interpretation**

- If the forecast historically underpredicts, the optimizer tends to push 𝑅↑.
- If it overpredicts, the optimizer tends to push 𝑅↓.
- If errors are balanced, 𝑅≈1.

This provides a principled starting point for CWSL-based optimization, model selection, and operational tuning.

### **5. Cost Sensitivity Utilities for CWSL**

Utilities for probing how sensitive model performance is to different
assumptions about the shortfall vs. overbuild cost ratio R=cu/co.

| Metric             | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| `cwsl_sensitivity` | Evaluate CWSL over a set of candidate cost ratios 𝑅, holding `co` fixed and using `cu = R * co`. |

Each metric has been:

- designed for real-world forecasting use cases
- validated with unit tests
- implemented with consistent input checking
- optimized for numpy vector operations

### **6. Framework Integrations (TensorFlow / Keras / scikit-learn)** 

`eb-metrics` includes an optional integration for deep learning workflows: a **Keras-compatible CWSL loss function**.

#### **Keras / TensorFlow CWSL Loss**

This utility creates a **per-sample asymmetric loss**, allowing neural networks to train directly using operational cost weights.

```python
from ebmetrics.frameworks import make_cwsl_keras_loss

loss_fn = make_cwsl_keras_loss(cu=2.0, co=1.0)

model.compile(optimizer="adam", loss=loss_fn)
```

#### **Features**
- Fully differentiable and TensorFlow-native  
- Computes shortfall and overbuild costs per sample  
- Normalizes cost by demand (matching the CWSL metric definition)  
- Supports arbitrary forecast horizons (reduces over time axis)  
- TensorFlow is imported lazily and **not** required unless used 

#### **Why this matters**
This enables:
- Cost-aware neural forecasting  
- Asymmetric training objectives  
- Deep learning compatibility with Electric Barometer metrics  

#### **scikit-learn CWSL Scorer**

For tree-based models and other sklearn estimators, `eb-metrics` provides a **CWSL-based scorer** that plugs directly into hyperparameter search utilities like `GridSearchCV` and `RandomizedSearchCV`.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from ebmetrics.frameworks import cwsl_scorer

scorer = cwsl_scorer(cu=2.0, co=1.0)

model = RandomForestRegressor(random_state=0)

grid = GridSearchCV(
    estimator=model,
    param_grid={"n_estimators": [50, 100, 200]},
    scoring=scorer,  # maximizes negative CWSL
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

Key properties:

- Uses **CWSL as the underlying loss**, but exposed as a sklearn scorer (higher is better)
- Fully compatible with `sample_weight`
- Works with any regressor that follows the standard sklearn API
- Lets you tune models directly on **operational cost** instead of symmetric error

---

## API Summary

The `ebmetrics` namespace provides a clean, stable interface to all core metrics.
You may import metrics directly from `ebmetrics.metrics` or from the specific
submodule (e.g., `.loss`, `.regression`, `.service`).

### **Cost-Weighted Loss Metrics**

```python
from ebmetrics.metrics import cwsl
```

| Function                                               | Description                                                                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`cwsl(y_true, y_pred, cu, co, sample_weight=None)`** | Cost-Weighted Service Loss. Asymmetric, demand-normalized forecast error metric that generalizes wMAPE by assigning explicit penalties to shortfall and overbuild. |

### **Regression Metrics**

```python
from ebmetrics.metrics import (
    mae, mse, rmse, mape, wmape,
    msle, rmsle, medae, smape, mase
)
```

| Function | Description                    |
| -------- | ------------------------------ |
| `mae`    | Mean Absolute Error            |
| `mse`    | Mean Squared Error             |
| `rmse`   | Root Mean Squared Error        |
| `mape`   | Mean Absolute Percentage Error |
| `wmape`  | Weighted MAPE                  |
| `smape`  | Symmetric MAPE                 |
| `msle`   | Mean Squared Log Error         |
| `rmsle`  | Root MSLE                      |
| `medae`  | Median Absolute Error          |
| `mase`   | Mean Absolute Scaled Error     |

### **Service-Level Metrics**

```python
from ebmetrics.metrics import nsl, ud, hr_at_tau, frs
```

| Function        | Description                                                   |
| --------------- | ------------------------------------------------------------- |
| **`nsl`**       | No-Shortfall Level — fraction of intervals with no shortfall. |
| **`ud`**        | Underbuild Depth — average shortfall amount.                  |
| **`hr_at_tau`** | Hit rate within a tolerance τ.                                |
| **`frs`**       | Forecast Readiness Score — combines NSL and CWSL.             |

### **Cost Sensitivity Utilities**

```python
from ebmetrics.metrics import cwsl_sensitivity
```

| Function           | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| `cwsl_sensitivity` | Evaluate CWSL over a grid of cost rations R=cu/co for robustness and tuning. |


### **Importing from Submodules**

All metrics can also be imported from their respective submodules:

```python
from ebmetrics.metrics.loss import cwsl
from ebmetrics.metrics.regression import rmse, wmape
from ebmetrics.metrics.service import nsl, frs, cwsl_sensitivity
```

All public functions are listed in __all__ for clean autocompletion and stable API expectations.

---

## Project Structure

The repository follows a clean, modern Python package layout designed for
readability, testability, and long-term maintainability.

```bash
eb-metrics/
│
├── src/ebmetrics/
│ ├── init.py
│ ├── _utils.py # Internal helper utilities
│ └── metrics/
│ ├── init.py # Public API re-export
│ ├── loss.py # Cost-weighted metrics (CWSL)
│ ├── regression.py # Classical regression metrics
│ └── service.py # Service-level operational metrics
│
├── tests/
│ ├── metrics/
│ │ ├── test_loss.py # Unit tests for CWSL and cost-weighted metrics
│ │ ├── test_regression.py # Unit tests for classical regression metrics
│ │ └── test_service.py # Unit tests for NSL, UD, HR@τ, FRS
│
├── pyproject.toml # Build config, dependencies, metadata
├── README.md # Project documentation
└── LICENSE # BSD-3-Clause license
```

This layout follows Python packaging best practices:

- `src/` layout prevents accidental imports
- Tests mirror the metric categories
- `_utils.py` centralizes consistent array validation and broadcasting
- Each metric family lives in its own submodule
- The public API is exposed through `ebmetrics.metrics`

---

## Running Tests

`eb-metrics` includes a comprehensive unit test suite covering all metric
families (loss, regression, and service-level metrics).

To run the tests locally:

### **1. Activate your virtual environment**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### **2. Run the full test suite**

```bash
pytest
```

To include coverage information:

```bash
pytest --cov=ebmetrics --cov-report=term-missing
```

All tests are located under:

tests/
└── metrics/
    ├── test_loss.py
    ├── test_regression.py
    └── test_service.py

The test suite validates:

- Input validation (shape, non-negativity, broadcasting)
- Correct numerical behavior
- Weighted vs. unweighted variants
- Error handling for edge cases
- Consistency across metrics

---

## License

`eb-metrics` is released under the **BSD-3-Clause License**, a permissive open-source
license that allows commercial and private use while providing attribution.

A full copy of the license is included in the repository.

This license aligns with the broader Electric Barometer ecosystem and ensures that
the metric implementations remain open, extensible, and usable in both research and
production environments.