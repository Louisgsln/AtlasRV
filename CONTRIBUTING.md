# Contributing

AtlasRV treats research correctness as a software requirement. A change to a
strategy must explain its information timing, economic rationale, cost assumptions,
and expected failure modes.

## Local workflow

~~~bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check src tests
mypy src
pytest
python -m build
twine check dist/*
~~~

## Pull requests

- Keep notebooks exploratory; reusable logic belongs under src/atlas_rv.
- Add a regression test for every correctness fix.
- Never introduce a feature computed with future observations.
- Separate gross return from every cost component and net return.
- Document new market-data assumptions, timestamps, and licensing constraints.
- Do not claim a live edge from synthetic, revised, or proxy data.
- Update the changelog for user-visible changes.

## Adding a relationship

Each pair requires:

1. an economic transmission mechanism;
2. asset-class metadata;
3. compatible timestamps and trading calendars;
4. a plausible execution and financing model;
5. a stability failure mode;
6. out-of-sample validation after universe-level multiple-testing control.
