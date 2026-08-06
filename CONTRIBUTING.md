# Contributing

AtlasRV treats research correctness as a software requirement. A contribution
that changes a strategy must explain its information timing, economic rationale,
transaction-cost assumptions, and expected failure modes.

## Local workflow

```bash
python -m pip install -e ".[dev]"
ruff check src tests
mypy src
pytest
atlas-rv demo --output reports/demo
```

## Pull requests

- Keep notebooks exploratory; reusable logic belongs under `src/atlas_rv`.
- Add a regression test for every correctness fix.
- Never introduce a feature computed with future observations.
- Do not claim a live edge from synthetic or revised historical data.
- Document new market-data assumptions and licensing constraints.

