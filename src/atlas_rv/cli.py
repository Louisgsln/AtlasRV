"""Command-line interface for reproducible cross-asset research."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from atlas_rv.backtest.engine import PairBacktester, PairBacktestResult
from atlas_rv.backtest.walk_forward import WalkForwardResult, run_walk_forward
from atlas_rv.config import (
    PairConfig,
    PortfolioConfig,
    StrategyConfig,
    WalkForwardConfig,
    load_config,
)
from atlas_rv.data.sources import CsvSource, FredSource, YahooFinanceSource
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.data.validation import clean_prices, validate_prices
from atlas_rv.reporting.report import write_research_bundle
from atlas_rv.research.diagnostics import diagnose_pair
from atlas_rv.research.model_comparison import compare_hedge_models
from atlas_rv.research.multiple_testing import apply_false_discovery_control
from atlas_rv.risk.portfolio import PortfolioResult, combine_pair_results


def _combine(
    results: dict[str, PairBacktestResult],
    config: PortfolioConfig,
    *,
    annualization: int,
) -> PortfolioResult:
    return combine_pair_results(
        results,
        annualization=annualization,
        volatility_lookback=config.volatility_lookback,
        correlation_lookback=config.correlation_lookback,
        correlation_penalty=config.correlation_penalty,
        rebalance_frequency=config.rebalance_frequency,
        max_weight=config.max_weight,
        allocation_cost_bps=config.allocation_cost_bps,
        target_volatility=config.target_volatility,
        max_leverage=config.max_leverage,
    )


def _run_study(
    prices: pd.DataFrame,
    pairs: tuple[PairConfig, ...],
    strategy: StrategyConfig,
    walk_forward: WalkForwardConfig,
    portfolio_config: PortfolioConfig,
    *,
    annualization: int,
    fdr_alpha: float,
    output: str | Path,
    full_sample: bool,
    include_review: bool,
    docs_assets: str | Path | None,
    dataset_label: str,
) -> int:
    cleaned = clean_prices(prices)
    quality = validate_prices(cleaned)
    diagnostics = {pair.name: diagnose_pair(cleaned, pair) for pair in pairs}
    diagnostics = apply_false_discovery_control(diagnostics, alpha=fdr_alpha)
    eligible_pairs = tuple(
        pair
        for pair in pairs
        if include_review or diagnostics[pair.name].passes_research_gate
    )
    if len(eligible_pairs) < 2:
        raise RuntimeError(
            "Research gate approved fewer than two relationships; inspect diagnostics "
            "or rerun with --include-review"
        )

    walk_forward_results: dict[str, WalkForwardResult] = {}
    pair_results: dict[str, PairBacktestResult] = {}
    if full_sample:
        backtester = PairBacktester(annualization=annualization)
        pair_results = {
            pair.name: backtester.run(cleaned, pair, strategy)
            for pair in eligible_pairs
        }
    else:
        for pair in eligible_pairs:
            result = run_walk_forward(
                cleaned,
                pair,
                strategy,
                walk_forward,
                annualization=annualization,
            )
            walk_forward_results[pair.name] = result
            pair_results[pair.name] = PairBacktestResult(
                pair=pair,
                config=strategy,
                frame=result.frame,
                metrics=result.metrics,
            )

    portfolio = _combine(pair_results, portfolio_config, annualization=annualization)
    report_path = write_research_bundle(
        output,
        prices=cleaned,
        pair_results=pair_results,
        diagnostics=diagnostics,
        portfolio=portfolio,
        walk_forward_results=walk_forward_results,
        docs_assets_directory=docs_assets,
        dataset_label=dataset_label,
        annualization=annualization,
    )
    print(
        f"AtlasRV complete: {quality.rows} rows, {len(eligible_pairs)}/{len(pairs)} "
        f"relationships traded, Sharpe={portfolio.metrics.get('sharpe', float('nan')):.2f}"
    )
    print(f"Research report: {report_path}")
    print(f"Interactive HTML: {Path(output) / 'research_report.html'}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    universe = generate_cross_asset_universe(
        observations=args.observations,
        seed=args.seed,
    )
    return _run_study(
        universe.prices,
        universe.pairs,
        StrategyConfig(),
        WalkForwardConfig(),
        PortfolioConfig(),
        annualization=252,
        fdr_alpha=0.05,
        output=args.output,
        full_sample=args.full_sample,
        include_review=False,
        docs_assets=args.docs_assets,
        dataset_label=f"deterministic synthetic universe (seed={args.seed})",
    )


def _run_csv(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    symbols = sorted({symbol for pair in config.pairs for symbol in (pair.x, pair.y)})
    prices = CsvSource(args.prices).load(symbols)
    return _run_study(
        prices,
        config.pairs,
        config.strategy,
        config.walk_forward,
        config.portfolio,
        annualization=config.annualization,
        fdr_alpha=config.fdr_alpha,
        output=args.output,
        full_sample=not args.walk_forward,
        include_review=args.include_review,
        docs_assets=None,
        dataset_label=f"CSV snapshot: {args.prices}",
    )


def _load_research_prices(
    args: argparse.Namespace,
    symbols: list[str],
) -> tuple[pd.DataFrame, str]:
    if args.provider == "synthetic":
        universe = generate_cross_asset_universe(
            observations=args.observations,
            seed=args.seed,
        )
        return universe.prices, f"deterministic synthetic universe (seed={args.seed})"
    if args.provider == "csv":
        if not args.prices:
            raise ValueError("--prices is required when --provider=csv")
        return CsvSource(args.prices).load(symbols), f"CSV snapshot: {args.prices}"
    if args.provider == "fred":
        gate_policy = (
            "research-gate override; REVIEW relationships included"
            if getattr(args, "include_review", False)
            else "research gate enforced"
        )
        prices = FredSource().load(
            symbols,
            start=args.start,
            end=args.end,
        )
        return prices, f"FRED public observed series (real-data stress study; {gate_policy})"
    prices = YahooFinanceSource().load(
        symbols,
        start=args.start,
        end=args.end,
    )
    return prices, "Yahoo Finance adjusted proxies (exploratory)"


def _research(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    symbols = sorted({symbol for pair in config.pairs for symbol in (pair.x, pair.y)})
    prices, label = _load_research_prices(args, symbols)
    return _run_study(
        prices,
        config.pairs,
        config.strategy,
        config.walk_forward,
        config.portfolio,
        annualization=config.annualization,
        fdr_alpha=config.fdr_alpha,
        output=args.output,
        full_sample=args.full_sample,
        include_review=args.include_review,
        docs_assets=args.docs_assets,
        dataset_label=label,
    )


def _compare_models(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pair = next((item for item in config.pairs if item.name == args.pair), None)
    if pair is None:
        choices = ", ".join(item.name for item in config.pairs)
        raise ValueError(f"Unknown pair {args.pair!r}. Choose one of: {choices}")
    symbols = sorted({pair.x, pair.y})
    prices, _ = _load_research_prices(args, symbols)
    cleaned = clean_prices(prices)
    validate_prices(cleaned)
    comparison = compare_hedge_models(
        cleaned,
        pair,
        config.strategy,
        annualization=config.annualization,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.metrics.to_csv(output)
    print(
        comparison.metrics[
            ["sharpe", "annualized_return", "max_drawdown", "sharpe_rank"]
        ]
    )
    print(f"Model comparison: {output}")
    return 0


def _download(args: argparse.Namespace) -> int:
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required")
    prices = YahooFinanceSource().load(symbols, start=args.start, end=args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output, index_label="date")
    print(f"Saved {len(prices)} rows to {output}")
    return 0


def _add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=("synthetic", "csv", "fred", "yahoo"),
        default="synthetic",
    )
    parser.add_argument("--prices", default=None, help="Wide CSV for the csv provider")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--observations", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-rv",
        description="Regime-aware cross-asset relative-value research lab.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the deterministic synthetic study")
    demo.add_argument("--output", default="reports/demo")
    demo.add_argument("--observations", type=int, default=1_500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--full-sample", action="store_true")
    demo.add_argument("--docs-assets", default=None)
    demo.set_defaults(handler=_demo)

    research = subparsers.add_parser(
        "research",
        help="Run one config with synthetic, CSV, FRED, or Yahoo data",
    )
    research.add_argument("--config", default="configs/universe.yml")
    research.add_argument("--output", default="reports/research")
    research.add_argument("--full-sample", action="store_true")
    research.add_argument("--include-review", action="store_true")
    research.add_argument("--docs-assets", default=None)
    _add_provider_arguments(research)
    research.set_defaults(handler=_research)

    run_csv = subparsers.add_parser("run-csv", help="Run configured pairs against a wide CSV")
    run_csv.add_argument("--prices", required=True)
    run_csv.add_argument("--config", default="configs/universe.yml")
    run_csv.add_argument("--output", default="reports/csv_run")
    run_csv.add_argument("--include-review", action="store_true")
    run_csv.add_argument("--walk-forward", action="store_true")
    run_csv.set_defaults(handler=_run_csv)

    compare = subparsers.add_parser(
        "compare-models",
        help="Compare expanding OLS, rolling OLS, and Kalman",
    )
    compare.add_argument("--config", default="configs/universe.yml")
    compare.add_argument("--pair", required=True)
    compare.add_argument("--output", default="reports/model_comparison.csv")
    _add_provider_arguments(compare)
    compare.set_defaults(handler=_compare_models)

    download = subparsers.add_parser("download", help="Download optional Yahoo Finance data")
    download.add_argument("--symbols", required=True, help="Comma-separated tickers")
    download.add_argument("--start", default="2015-01-01")
    download.add_argument("--end", default=None)
    download.add_argument("--output", default="data/cache/yahoo_prices.csv")
    download.set_defaults(handler=_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
