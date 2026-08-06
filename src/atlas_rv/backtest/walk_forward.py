"""Rolling train/purge/test parameter selection."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from atlas_rv.backtest.engine import PairBacktester
from atlas_rv.config import PairConfig, StrategyConfig, WalkForwardConfig
from atlas_rv.risk.metrics import performance_metrics


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardResult:
    pair: PairConfig
    frame: pd.DataFrame
    folds: pd.DataFrame
    metrics: dict[str, float]


def build_walk_forward_folds(
    index: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> tuple[WalkForwardFold, ...]:
    """Build disjoint out-of-sample folds with a purged gap."""

    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Walk-forward index must be sorted and unique")
    folds: list[WalkForwardFold] = []
    train_start = 0
    fold_number = 0
    while True:
        train_end = train_start + config.train_size
        test_start = train_end + config.purge_size
        test_end = test_start + config.test_size
        if test_end > len(index):
            break
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_start_date=pd.Timestamp(index[train_start]),
                train_end_date=pd.Timestamp(index[train_end - 1]),
                test_start_date=pd.Timestamp(index[test_start]),
                test_end_date=pd.Timestamp(index[test_end - 1]),
            )
        )
        train_start += config.test_size
        fold_number += 1
    return tuple(folds)


def _selection_score(metrics: dict[str, float], turnover_penalty: float) -> float:
    sharpe = metrics.get("sharpe", float("nan"))
    trades = metrics.get("trades", 0.0)
    annualized_turnover = metrics.get("annualized_turnover", 0.0)
    if not np.isfinite(sharpe) or trades < 3:
        return -float("inf")
    return float(sharpe - turnover_penalty * annualized_turnover)


def run_walk_forward(
    prices: pd.DataFrame,
    pair: PairConfig,
    base_config: StrategyConfig,
    walk_forward_config: WalkForwardConfig,
    *,
    annualization: int = 252,
    turnover_penalty: float = 0.002,
) -> WalkForwardResult:
    """Select parameters on training data and report only held-out returns."""

    aligned = prices[[pair.y, pair.x]].dropna().sort_index()
    folds = build_walk_forward_folds(pd.DatetimeIndex(aligned.index), walk_forward_config)
    if not folds:
        raise ValueError("Dataset is too short for the requested walk-forward schedule")

    backtester = PairBacktester(annualization=annualization)
    test_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, float | int | str]] = []

    for fold in folds:
        training = aligned.iloc[fold.train_start : fold.train_end]
        best_score = -float("inf")
        best_config: StrategyConfig | None = None
        best_metrics: dict[str, float] = {}

        for lookback in walk_forward_config.zscore_lookbacks:
            for entry in walk_forward_config.entry_thresholds:
                if entry >= base_config.stop_z:
                    continue
                candidate = replace(
                    base_config,
                    zscore_lookback=lookback,
                    entry_z=entry,
                )
                result = backtester.run(training, pair, candidate)
                score = _selection_score(result.metrics, turnover_penalty)
                if score > best_score:
                    best_score = score
                    best_config = candidate
                    best_metrics = result.metrics

        if best_config is None:
            raise RuntimeError(f"No viable parameter set for fold {fold.fold}")

        history_and_test = aligned.iloc[fold.train_start : fold.test_end]
        complete_result = backtester.run(history_and_test, pair, best_config)
        test_frame = complete_result.frame.loc[fold.test_start_date : fold.test_end_date].copy()
        test_frame["fold"] = fold.fold
        test_frame["selected_lookback"] = best_config.zscore_lookback
        test_frame["selected_entry_z"] = best_config.entry_z
        test_frames.append(test_frame)
        fold_records.append(
            {
                "fold": fold.fold,
                "train_start": str(fold.train_start_date.date()),
                "train_end": str(fold.train_end_date.date()),
                "test_start": str(fold.test_start_date.date()),
                "test_end": str(fold.test_end_date.date()),
                "lookback": best_config.zscore_lookback,
                "entry_z": best_config.entry_z,
                "selection_score": best_score,
                "train_sharpe": best_metrics.get("sharpe", float("nan")),
                "train_trades": best_metrics.get("trades", 0.0),
            }
        )

    combined = pd.concat(test_frames).sort_index()
    if combined.index.has_duplicates:
        raise RuntimeError("Out-of-sample folds unexpectedly overlap")
    combined["equity"] = (1.0 + combined["net_return"].clip(lower=-0.999999)).cumprod()
    running_peak = combined["equity"].cummax()
    combined["drawdown"] = combined["equity"] / running_peak - 1.0
    metrics = performance_metrics(
        combined["net_return"],
        annualization=annualization,
        turnover=combined["turnover"],
        positions=combined["position"],
    )
    return WalkForwardResult(
        pair=pair,
        frame=combined,
        folds=pd.DataFrame.from_records(fold_records).set_index("fold"),
        metrics=metrics,
    )
