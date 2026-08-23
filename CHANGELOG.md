# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.x] - 2026-08-22

### Added

- Added zero-allocation scalar fast paths for unweighted cwsl, nsl, ud, and hr_at_tau.
- Added 67 property-based stress tests verifying scale invariance, monotonicity, and exact zero limits.
- Exported py.typed marker for PEP 561 compliance.
