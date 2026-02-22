# eb-metrics

`eb-metrics` provides the core metric implementations used across the Electric Barometer ecosystem.

The package focuses on **cost-aware**, **asymmetric**, and **readiness-oriented** evaluation of forecasts and operational decisions. It defines metric semantics and reference implementations that can be reused consistently across modeling, optimization, evaluation, and governance workflows.

## Scope

This package is responsible for:

- Implementing metric definitions and reference behavior
- Encoding asymmetric cost structures and service risk
- Providing framework-agnostic metric primitives
- Supplying adapters for common machine learning frameworks

It intentionally avoids model training logic, optimization policy, and workflow orchestration.

## Contents

- **Metrics**
  Core loss, service, and readiness metrics

- **Framework integrations**
  Adapters for using metrics within external ML frameworks

## API reference

- [Metrics](api/metrics.md)
- [Framework integrations](api/frameworks.md)
