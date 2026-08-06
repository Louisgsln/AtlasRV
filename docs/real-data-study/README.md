# Public FRED multi-asset study

This directory is the tracked, auditable output of the publication workflow.
The generated tables and charts appear here after the first successful manual run.

## Scope

The study uses public observed FRED series across equity, rates, credit,
volatility, FX, crypto, and commodities from 2021 onward. It runs the same causal
walk-forward engine used by the deterministic benchmark.

The publication command intentionally uses **--include-review**. This preserves
failed diagnostics and stress-tests portfolio plumbing; it does not waive the
statistical decisions shown in diagnostics.csv.

Several inputs are indices, yields, spreads, or reference levels rather than
executable total-return instruments. Costs are illustrative, calendars are mixed,
and the output must not be presented as a live trading record.

## Reproduce

~~~bash
python -m pip install -e ".[dev]"
atlas-rv research \
  --provider fred \
  --config configs/fred_market_study.yml \
  --start 2021-01-01 \
  --include-review \
  --output reports/fred_market_study
~~~

The workflow publishes the HTML dashboard to
https://louisgsln.github.io/AtlasRV/ and attaches a ZIP of this study to release
v0.2.0 after lint, typing, tests, and package checks pass.
