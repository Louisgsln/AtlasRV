"""Persist a complete, inspectable research run."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from html import escape
from pathlib import Path

import pandas as pd

from atlas_rv.backtest.engine import PairBacktestResult
from atlas_rv.backtest.walk_forward import WalkForwardResult
from atlas_rv.data.snapshot import write_snapshot
from atlas_rv.reporting.charts import plot_pair_zscores, plot_portfolio
from atlas_rv.research.diagnostics import PairDiagnostics
from atlas_rv.research.regimes import RegimeAnalysis, analyze_regimes
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


def _write_html_report(
    path: Path,
    *,
    dataset_label: str,
    portfolio: PortfolioResult,
    diagnostics: Mapping[str, PairDiagnostics],
    regime_analysis: RegimeAnalysis,
) -> None:
    metrics = _serializable_metrics(portfolio.metrics)
    cards = [
        ("Sharpe", _number(metrics.get("sharpe"))),
        ("Annual return", _percent(metrics.get("annualized_return"))),
        ("Annual volatility", _percent(metrics.get("annualized_volatility"))),
        ("Maximum drawdown", _percent(metrics.get("max_drawdown"))),
        ("Effective bets", _number(metrics.get("average_effective_bets"))),
    ]
    card_html = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in cards
    )
    diagnostic_rows = "".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{item.coint_pvalue:.4f}</td>"
        f"<td>{item.coint_qvalue:.4f}</td>"
        f"<td>{_days(item.half_life_days)}</td>"
        f"<td>{item.beta_instability:.1%}</td>"
        f"<td class={'pass' if item.passes_research_gate else 'review'}>"
        f"{'PASS' if item.passes_research_gate else 'REVIEW'}</td>"
        "</tr>"
        for name, item in diagnostics.items()
    )
    regime_html = (
        "<p>Insufficient history for regime attribution.</p>"
        if regime_analysis.metrics.empty
        else regime_analysis.metrics.to_html(
            border=0,
            classes="dataframe",
            float_format=lambda value: f"{value:.3f}",
        )
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AtlasRV research dashboard</title>
<style>
:root {{ --navy:#102a43; --blue:#2f80ed; --ink:#243b53; --muted:#627d98; --bg:#f5f8fb; }}
body {{ margin:0; font:15px/1.55 Inter,Arial,sans-serif; color:var(--ink); background:var(--bg); }}
main {{ max-width:1120px; margin:0 auto; padding:40px 24px 72px; }}
h1,h2 {{ color:var(--navy); }} h1 {{ margin-bottom:4px; }}
.subtitle {{ color:var(--muted); margin-top:0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:24px 0; }}
.card {{ background:white; border:1px solid #d9e2ec; border-radius:10px; padding:16px; }}
.card span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; }}
.card strong {{ display:block; font-size:24px; color:var(--navy); margin-top:4px; }}
.panel {{ background:white; border:1px solid #d9e2ec; border-radius:12px; padding:22px; margin:18px 0; overflow:auto; }}
img {{ width:100%; height:auto; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px; border-bottom:1px solid #e8eef4; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }} .pass {{ color:#16803c; font-weight:700; }}
.review {{ color:#b54708; font-weight:700; }} code {{ background:#eef2f6; padding:2px 5px; }}
</style>
</head>
<body><main>
<h1>AtlasRV research dashboard</h1>
<p class="subtitle">{escape(dataset_label)} · causal cross-asset relative value</p>
<div class="cards">{card_html}</div>
<section class="panel"><h2>Causal out-of-sample portfolio</h2><img src="portfolio_equity.png" alt="Portfolio equity and drawdown"></section>
<section class="panel"><h2>Research gate</h2>
<table><thead><tr><th>Relationship</th><th>p-value</th><th>FDR q-value</th><th>Half-life</th><th>Beta instability</th><th>Decision</th></tr></thead>
<tbody>{diagnostic_rows}</tbody></table></section>
<section class="panel"><h2>Performance by ex-ante regime</h2>{regime_html}</section>
<section class="panel"><h2>Deterministic benchmark</h2><code>atlas-rv research --provider synthetic --config configs/universe.yml --output reports/research</code></section>
<p class="subtitle">Research software only. FRED levels are observed public series, not necessarily executable total-return instruments.</p>
</main></body></html>
"""
    path.write_text(html, encoding="utf-8")


def write_research_bundle(
    output_directory: str | Path,
    *,
    prices: pd.DataFrame,
    pair_results: Mapping[str, PairBacktestResult],
    diagnostics: Mapping[str, PairDiagnostics],
    portfolio: PortfolioResult,
    walk_forward_results: Mapping[str, WalkForwardResult] | None = None,
    docs_assets_directory: str | Path | None = None,
    dataset_label: str = "research dataset",
    annualization: int = 252,
) -> Path:
    """Write tables, charts, checksums, JSON, Markdown, and HTML."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    pair_directory = output / "pairs"
    pair_directory.mkdir(exist_ok=True)

    prices.to_csv(output / "prices.csv", index_label="date")
    snapshot_manifest = write_snapshot(output / "data_snapshot", prices)
    for name, pair_result in pair_results.items():
        pair_result.frame.to_csv(pair_directory / f"{name}_backtest.csv.gz", compression="gzip")
    portfolio.frame.to_csv(output / "portfolio.csv", index_label="date")
    portfolio.weights.to_csv(output / "portfolio_weights.csv", index_label="date")
    portfolio.class_allocations.to_csv(
        output / "portfolio_class_allocations.csv",
        index_label="date",
    )

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

    benchmark_returns = (
        prices.pct_change(fill_method=None)
        .replace([float("inf"), -float("inf")], float("nan"))
        .mean(axis=1)
        .fillna(0.0)
    )
    regime_analysis = analyze_regimes(
        portfolio.frame["portfolio_return"],
        benchmark_returns,
        annualization=annualization,
    )
    regime_analysis.labels.to_csv(output / "regimes.csv", index_label="date")
    regime_analysis.metrics.to_csv(output / "regime_metrics.csv", index_label="regime")

    plot_portfolio(portfolio, output / "portfolio_equity.png")
    plot_pair_zscores(pair_results, output / "pair_zscores.png")
    if docs_assets_directory is not None:
        assets = Path(docs_assets_directory)
        plot_portfolio(portfolio, assets / "demo_portfolio.png")
        plot_pair_zscores(pair_results, assets / "demo_zscores.png")

    manifest_payload = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
    payload = {
        "dataset": {"label": dataset_label, "snapshot": manifest_payload},
        "portfolio": _serializable_metrics(portfolio.metrics),
        "diagnostics": {
            name: _serializable_diagnostics(item) for name, item in diagnostics.items()
        },
        "regimes": {
            str(index): _serializable_metrics(
                {str(key): float(value) for key, value in row.items()}
            )
            for index, row in regime_analysis.metrics.iterrows()
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
        f"> Dataset: **{dataset_label}**. Historical and synthetic results are not a live",
        "> trading claim or investment advice.",
        "",
        "## Executive summary",
        "",
        f"- Out-of-sample Sharpe: **{_number(portfolio_metrics.get('sharpe'))}**",
        f"- Annualized return: **{_percent(portfolio_metrics.get('annualized_return'))}**",
        f"- Annualized volatility: **{_percent(portfolio_metrics.get('annualized_volatility'))}**",
        f"- Maximum drawdown: **{_percent(portfolio_metrics.get('max_drawdown'))}**",
        f"- Average effective bets: **{_number(portfolio_metrics.get('average_effective_bets'))}**",
        f"- Research gate: **{len(pair_results)}/{len(diagnostics)} relationships traded**",
        "",
        "![Portfolio equity and drawdown](portfolio_equity.png)",
        "",
        "## Relationship diagnostics",
        "",
        "| Pair | Coint p | FDR q | Half-life | Hurst | Beta instability | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, item in diagnostics.items():
        lines.append(
            f"| {name} | {item.coint_pvalue:.4f} | {item.coint_qvalue:.4f} | "
            f"{_days(item.half_life_days)} | {item.hurst_exponent:.2f} | "
            f"{item.beta_instability:.1%} | "
            f"{'PASS' if item.passes_research_gate else 'REVIEW'} |"
        )
    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "1. Register an economic thesis before testing the relationship.",
            "2. Control the universe-level false-discovery rate.",
            "3. Estimate a causal Kalman, rolling OLS, or expanding OLS hedge ratio.",
            "4. Standardize one-step errors against history ending yesterday.",
            "5. Select parameters on rolling train/purge/test folds.",
            "6. Apply next-bar weights and explicit spread, impact, borrow, and funding costs.",
            "7. Allocate sleeves with lagged volatility and correlation penalties.",
            "8. Attribute performance to regimes classified only with prior information.",
            "",
            "## Reproduction and integrity",
            "",
            f"- Data checksum: {manifest_payload['sha256']}",
            "- Machine-readable results: summary.json",
            "- Interactive report: research_report.html",
            "",
            "Reproduce with:",
            "",
            "    python -m pip install -e '.[dev]'",
            "    atlas-rv research --provider synthetic --config configs/universe.yml --output reports/research",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_html_report(
        output / "research_report.html",
        dataset_label=dataset_label,
        portfolio=portfolio,
        diagnostics=diagnostics,
        regime_analysis=regime_analysis,
    )
    return report_path
