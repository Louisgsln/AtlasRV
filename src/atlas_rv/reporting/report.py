"""Persist a complete, inspectable research run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from atlas_rv.backtest.engine import PairBacktestResult
from atlas_rv.backtest.walk_forward import WalkForwardResult
from atlas_rv.reporting.charts import plot_pair_zscores, plot_portfolio
from atlas_rv.research.diagnostics import PairDiagnostics
from atlas_rv.risk.portfolio import PortfolioResult


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _serializable_metrics(metrics: Mapping[str, float]) -> dict[str, float | None]:
    return {key: _finite(float(value)) for key, value in metrics.items()}


def _serializable_diagnostics(item: PairDiagnostics) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in item.to_dict().items():
        if isinstance(value, float):
            result[key] = _finite(value)
        else:
            result[key] = value
    return result


def _percent(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2f}"


def _days(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.1f}d"


def write_research_bundle(
    output_directory: str | Path,
    *,
    prices: pd.DataFrame,
    pair_results: Mapping[str, PairBacktestResult],
    diagnostics: Mapping[str, PairDiagnostics],
    portfolio: PortfolioResult,
    walk_forward_results: Mapping[str, WalkForwardResult] | None = None,
    docs_assets_directory: str | Path | None = None,
) -> Path:
    """Write tables, charts, machine-readable results, and a Markdown report."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pair_directory = output / "pairs"
    pair_directory.mkdir(exist_ok=True)

    prices.to_csv(output / "prices.csv", index_label="date")
    for name, pair_result in pair_results.items():
        pair_result.frame.to_csv(pair_directory / f"{name}_backtest.csv.gz", compression="gzip")
    portfolio.frame.to_csv(output / "portfolio.csv", index_label="date")
    portfolio.weights.to_csv(output / "portfolio_weights.csv", index_label="date")

    diagnostic_table = pd.DataFrame(
        {name: item.to_dict() for name, item in diagnostics.items()}
    ).T
    diagnostic_table.to_csv(output / "diagnostics.csv", index_label="pair")
    metric_table = pd.DataFrame(
        {name: pair_result.metrics for name, pair_result in pair_results.items()}
    ).T
    metric_table.to_csv(output / "pair_metrics.csv", index_label="pair")

    if walk_forward_results:
        folds_directory = output / "walk_forward_folds"
        folds_directory.mkdir(exist_ok=True)
        for name, walk_forward_result in walk_forward_results.items():
            walk_forward_result.folds.to_csv(folds_directory / f"{name}.csv")

    plot_portfolio(portfolio, output / "portfolio_equity.png")
    plot_pair_zscores(pair_results, output / "pair_zscores.png")
    if docs_assets_directory is not None:
        assets = Path(docs_assets_directory)
        plot_portfolio(portfolio, assets / "demo_portfolio.png")
        plot_pair_zscores(pair_results, assets / "demo_zscores.png")

    payload = {
        "portfolio": _serializable_metrics(portfolio.metrics),
        "diagnostics": {
            name: _serializable_diagnostics(item) for name, item in diagnostics.items()
        },
        "pairs": {
            name: {
                "metrics": _serializable_metrics(pair_result.metrics),
                "diagnostics": _serializable_diagnostics(diagnostics[name]),
            }
            for name, pair_result in pair_results.items()
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    report_path = output / "research_report.md"
    portfolio_metrics = _serializable_metrics(portfolio.metrics)
    lines = [
        "# AtlasRV research report",
        "",
        "> Reproducible demonstration on deterministic synthetic data. These results are a",
        "> software and methodology test, not evidence of a live trading edge.",
        "",
        "## Executive summary",
        "",
        f"- Out-of-sample Sharpe: **{_number(portfolio_metrics.get('sharpe'))}**",
        f"- Annualized return: **{_percent(portfolio_metrics.get('annualized_return'))}**",
        f"- Annualized volatility: **{_percent(portfolio_metrics.get('annualized_volatility'))}**",
        f"- Maximum drawdown: **{_percent(portfolio_metrics.get('max_drawdown'))}**",
        f"- Research gate: **{len(pair_results)}/{len(diagnostics)} relationships traded**",
        "",
        "![Portfolio equity and drawdown](portfolio_equity.png)",
        "",
        "## Relationship diagnostics",
        "",
        "| Pair | Coint p-value | Half-life | Hurst | Beta instability | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name, item in diagnostics.items():
        lines.append(
            f"| {name} | {item.coint_pvalue:.4f} | {_days(item.half_life_days)} | "
            f"{item.hurst_exponent:.2f} | {item.beta_instability:.1%} | "
            f"{'PASS' if item.passes_research_gate else 'REVIEW'} |"
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "1. Form an economic hypothesis before running a statistical test.",
            "2. Estimate a time-varying log-price hedge ratio with a causal Kalman filter.",
            "3. Standardize one-step pricing errors against a trailing window ending yesterday.",
            "4. Select parameters on rolling training windows separated from each test window by a purge gap.",
            "5. Apply weights one bar after signal formation and deduct two-leg turnover costs.",
            "6. Allocate independent sleeves with lagged inverse volatility and capped weights.",
            "",
            "## Known limitations",
            "",
            "- Synthetic relationships are deliberately cleaner than real cross-asset markets.",
            "- ETF and futures proxies introduce basis, roll, financing, and trading-hours differences.",
            "- Linear transaction costs omit spread convexity, market impact, and capacity constraints.",
            "- Cointegration tests have finite-sample and multiple-testing risk.",
            "- A research gate is a diagnostic, never a guarantee of future stability.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python -m pip install -e '.[dev]'",
            "atlas-rv demo --output reports/demo",
            "```",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
