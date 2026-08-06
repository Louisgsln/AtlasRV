# Changelog

All notable changes to AtlasRV are documented here.

## 0.2.0 — 2026-08-06

### Added

- Causal expanding OLS and rolling OLS alongside the Kalman hedge model.
- Like-for-like model-comparison command.
- Execution-cost attribution for commission, spread, slippage, impact, borrow,
  financing, and the backwards-compatible all-in cost.
- Benjamini-Hochberg false-discovery control across the research universe.
- Causal market-regime classification and conditional performance.
- Correlation-adjusted sleeve allocation, optional portfolio volatility target,
  effective-bet metrics, and asset-class allocation attribution.
- Canonical CSV data snapshots with SHA-256 integrity manifests.
- Unified synthetic, CSV, and Yahoo research command.
- Self-contained HTML research report and expanded Streamlit dashboard.
- Broader exploratory market universe covering equity, rates, credit,
  commodities, crypto, FX, and volatility.
- Docker image, pre-commit hooks, package-build checks, and Python 3.13 CI.
- Regression tests for the new statistical, execution, portfolio, and integrity layers.

### Changed

- Package maturity moved from alpha to beta.
- Research reports no longer assume that every input dataset is synthetic.
- Portfolio allocation now penalises highly correlated sleeves.

### Compatibility

The original **cost_bps** configuration field remains supported. Set it to zero
when using the explicit execution-cost components to avoid double counting.

## 0.1.0 — 2026-08-06

- Initial typed Python research platform.
- Dynamic Kalman hedge ratio, causal z-score, next-bar execution, two-leg costs.
- Purged walk-forward selection, inverse-volatility portfolio, static reports,
  optional Streamlit viewer, deterministic synthetic universe, and CI.
