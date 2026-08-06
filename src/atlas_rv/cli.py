"""Command-line interface for reproducible research runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from atlas_rv.backtest.engine import PairBacktester, PairBacktestResult
from atlas_rv.backtest.walk_forward import run_walk_forward
from atlas_rv.config import StrategyConfig, WalkForwardConfig, load_config
from atlas_rv.data.sources import CsvSource, YahooFinanceSource
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.data.validation import clean_prices, validate_prices
from atlas_rv.reporting.report import write_research_bundle
from atlas_rv.research.diagnostics import diagnose_pair
from atlas_rv.risk.portfolio import combine_pair_results


def _demo(args: argparse.Namespace) -> int:
    universe = generate_cross_asset_universe(
        observations=args.observations,
        seed=args.seed,
    )
    prices = clean_prices(universe.prices)
    report = validate_prices(prices)
    strategy = StrategyConfig()
    walk_forward = WalkForwardConfig()
    diagnostics = {pair.name: diagnose_pair(prices, pair) for pair in universe.pairs}
    eligible_pairs = tuple(
        pair for pair in universe.pairs if diagnostics[pair.name].passes_research_gate
    )
    if len(eligible_pairs) < 2:
        raise RuntimeError("Research gate approved fewer than two relationships")

    walk_forward_results = {}
    pair_results: dict[str, PairBacktestResult] = {}
    if args.full_sample:
        backtester = PairBacktester()
        pair_results = {
            pair.name: backtester.run(prices, pair, strategy) for pair in eligible_pairs
        }
    else:
        for pair in eligible_pairs:
            result = run_walk_forward(prices, pair, strategy, walk_forward)
            walk_forward_results[pair.name] = result
            pair_results[pair.name] = PairBacktestResult(
                pair=pair,
                config=strategy,
                frame=result.frame,
                metrics=result.metrics,
            )

    portfolio = combine_pair_results(pair_results)
    report_path = write_research_bundle(
        args.output,
        prices=prices,
        pair_results=pair_results,
        diagnostics=diagnostics,
        portfolio=portfolio,
        walk_forward_results=walk_forward_results,
        docs_assets_directory=args.docs_assets,
    )
    print(
        f"AtlasRV demo complete: {report.rows} rows, {len(eligible_pairs)}/{len(universe.pairs)} "
        "relationships approved, "
        f"Sharpe={portfolio.metrics.get('sharpe', float('nan')):.2f}"
    )
    print(f"Research report: {report_path}")
    return 0


def _run_csv(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    symbols = sorted({symbol for pair in config.pairs for symbol in (pair.x, pair.y)})
    prices = clean_prices(CsvSource(args.prices).load(symbols))
    validate_prices(prices)
    backtester = PairBacktester(annualization=config.annualization)
    diagnostics = {pair.name: diagnose_pair(prices, pair) for pair in config.pairs}
    eligible_pairs = tuple(
        pair
        for pair in config.pairs
        if args.include_review or diagnostics[pair.name].passes_research_gate
    )
    if len(eligible_pairs) < 2:
        raise RuntimeError(
            "Research gate approved fewer than two relationships; inspect diagnostics or "
            "rerun with --include-review"
        )
    results = {
        pair.name: backtester.run(prices, pair, config.strategy) for pair in eligible_pairs
    }
    portfolio = combine_pair_results(results, annualization=config.annualization)
    report_path = write_research_bundle(
        args.output,
        prices=prices,
        pair_results=results,
        diagnostics=diagnostics,
        portfolio=portfolio,
    )
    print(f"Research report: {report_path}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas-rv",
        description="Regime-aware cross-asset relative-value research lab.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a fully reproducible synthetic study")
    demo.add_argument("--output", default="reports/demo", help="Research output directory")
    demo.add_argument("--observations", type=int, default=1_500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument(
        "--full-sample",
        action="store_true",
        help="Skip walk-forward selection (faster, but not the default research claim)",
    )
    demo.add_argument(
        "--docs-assets",
        default=None,
        help="Optional directory receiving README-ready charts",
    )
    demo.set_defaults(handler=_demo)

    run_csv = subparsers.add_parser("run-csv", help="Run configured pairs against a wide CSV")
    run_csv.add_argument("--prices", required=True)
    run_csv.add_argument("--config", default="configs/universe.yml")
    run_csv.add_argument("--output", default="reports/csv_run")
    run_csv.add_argument(
        "--include-review",
        action="store_true",
        help="Backtest relationships that fail the research gate",
    )
    run_csv.set_defaults(handler=_run_csv)

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
