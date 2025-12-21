# eb-metrics

**eb-metrics** is the core metric library of the **Electric Barometer** ecosystem.

It provides a principled set of error metrics and evaluation utilities designed
for **operational forecasting environments**—contexts where **directional error**,
**asymmetric cost**, and **service reliability** matter more than symmetric
accuracy alone.

Unlike generic regression metrics, `eb-metrics` is built to evaluate forecasts
in settings where underprediction and overprediction have **materially different
consequences**, such as quick-service restaurants (QSR), retail operations,
logistics, inventory planning, and other service-constrained systems.

---

## Naming convention

Electric Barometer packages follow a consistent naming convention:

- **Distribution names** (used with `pip install`) use hyphens  
  e.g. `pip install eb-metrics`
- **Python import paths** use underscores  
  e.g. `import eb_metrics`

This follows standard Python packaging practices and avoids ambiguity between
package names and module imports.

---

## What this package provides

### Asymmetric, cost-aware loss metrics
Metrics that explicitly encode operational cost asymmetry between shortfall
and overbuild.

- **Cost-Weighted Service Loss (CWSL)**  
  A demand-normalized, directionally-aware loss that generalizes weighted MAPE
  by assigning explicit costs to underbuild and overbuild.

---

### Service-level and readiness diagnostics
Metrics that evaluate forecast behavior from a *service reliability* and
*operational readiness* perspective.

- **No Shortfall Level (NSL)** — frequency of avoiding shortfall  
- **Underbuild Depth (UD)** — severity of shortfalls when they occur  
- **Hit Rate within Tolerance (HR@τ)** — accuracy within operational bounds  
- **Forecast Readiness Score (FRS)** — composite readiness metric combining NSL and CWSL  

---

### Classical regression metrics
Standard symmetric error metrics included for baseline comparison and diagnostic
validation.

- MAE, MSE, RMSE  
- MAPE, WMAPE, sMAPE  
- MedAE, MASE, MSLE, RMSLE  

---

### Framework integrations
Adapters that allow Electric Barometer metrics to integrate cleanly into
common machine-learning workflows.

- **scikit-learn** scorers (e.g., for `GridSearchCV`, `cross_val_score`)
- **Keras / TensorFlow** loss functions for cost-aware model training

---

## Documentation structure

- **API Reference**  
  All metric and framework documentation is generated automatically from
  NumPy-style docstrings in the source code using `mkdocstrings`.

Conceptual motivation, formal definitions, and interpretive guidance for these
metrics are documented in the companion research repository **eb-papers**.

---

## Intended audience

This documentation is intended for:

- data scientists and applied ML practitioners
- forecasting and demand-planning teams
- operations and service analytics leaders
- researchers working in cost-sensitive or service-constrained environments

The focus throughout is on **decision-relevant evaluation**, not abstract
statistical accuracy.

---

## Relationship to the Electric Barometer framework

`eb-metrics` provides the **metric layer** of the Electric Barometer ecosystem.
It is designed to be used alongside:

- **eb-evaluation** — structured forecast evaluation workflows
- **eb-adapters** — integrations with external forecasting systems
- **eb-papers** — formal definitions, theory, and technical notes

Together, these components support a unified approach to measuring
*forecast readiness*, not just forecast error.