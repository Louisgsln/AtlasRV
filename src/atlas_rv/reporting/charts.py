"""Deterministic static visualisations suitable for GitHub."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from atlas_rv.backtest.engine import PairBacktestResult
from atlas_rv.risk.portfolio import PortfolioResult

plt.switch_backend("Agg")


_NAVY = "#102A43"
_BLUE = "#2F80ED"
_RED = "#D64545"
_GREY = "#829AB1"


def _style_axes(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9E2EC", linewidth=0.7, alpha=0.7)
    axis.tick_params(colors="#486581")


def plot_portfolio(portfolio: PortfolioResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    top.plot(portfolio.frame.index, portfolio.frame["equity"], color=_BLUE, linewidth=1.8)
    top.set_title("AtlasRV out-of-sample portfolio", loc="left", color=_NAVY, weight="bold")
    top.set_ylabel("Growth of $1")
    _style_axes(top)

    bottom.fill_between(
        portfolio.frame.index,
        portfolio.frame["drawdown"],
        0.0,
        color=_RED,
        alpha=0.35,
        linewidth=0.0,
    )
    bottom.set_ylabel("Drawdown")
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(bottom)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_pair_zscores(
    results: Mapping[str, PairBacktestResult],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(results), 1, figsize=(11, 2.1 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for axis, (name, result) in zip(axes, results.items(), strict=True):
        axis.plot(result.frame.index, result.frame["zscore"], color=_NAVY, linewidth=0.8)
        axis.axhline(result.config.entry_z, color=_RED, linewidth=0.7, linestyle="--")
        axis.axhline(-result.config.entry_z, color=_RED, linewidth=0.7, linestyle="--")
        axis.axhline(0.0, color=_GREY, linewidth=0.6)
        axis.set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center")
        _style_axes(axis)
    axes[0].set_title("One-step pricing errors (causal z-scores)", loc="left", color=_NAVY, weight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output
