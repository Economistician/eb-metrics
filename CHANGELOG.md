# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Classical regression metrics (`rmse`, `mape`, `wmape`, `mase`, and the shared `_validate_shapes` helper) reject non-1D, empty, and non-finite inputs with explicit `ValueError`s, matching service-kernel domain checks. Valid finite 1D series are unchanged.

### Performance

- Public NSL, UD, HR@τ, and CWSL wrappers share internal validated-array kernels so callers can reuse a single `_validated_nonneg_pair` result.
- `_to_1d_array` returns a 1D C-contiguous `float64` ndarray as a view (no `asarray` copy) and still rejects empty and non-finite values.
- `_validate_shapes` reuses C-contiguous `float64` arrays; `rmse` derives from a single validated MSE; `mase` reuses the validated `y_true` array.

## [0.2.8] - 2026-08-23

### Changed

- Tightened README Overview; removed cloned Role section.
- Applied repo-wide Ruff formatting and optional-import typing guards so `python tooling/check.py` passes cleanly.
- Changelog version header now matches `pyproject.toml` (`0.2.8`).
- Development Status classifier is `5 - Production/Stable`.

### Added

- Added zero-allocation scalar fast paths for unweighted cwsl, nsl, ud, and hr_at_tau.
- Added 67 property-based stress tests verifying scale invariance, monotonicity, and exact zero limits.
- Exported py.typed marker for PEP 561 compliance.
