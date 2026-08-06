"""Transparent two-leg execution, financing, and borrow costs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionCostBreakdown:
    """Daily cost components expressed as portfolio-return deductions."""

    frame: pd.DataFrame

    @property
    def total(self) -> pd.Series:
        return self.frame["transaction_cost"]


@dataclass(frozen=True)
class ExecutionCostModel:
    """Deterministic daily execution and holding-cost model."""

    legacy_cost_bps: float = 0.0
    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_coefficient_bps: float = 0.0
    borrow_rate_bps_annual: float = 0.0
    financing_rate_bps_annual: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.legacy_cost_bps,
            self.commission_bps,
            self.half_spread_bps,
            self.slippage_bps,
            self.impact_coefficient_bps,
            self.borrow_rate_bps_annual,
            self.financing_rate_bps_annual,
        )
        if any(value < 0 for value in values):
            raise ValueError("Execution-cost inputs cannot be negative")

    def calculate(
        self,
        target_weights: pd.DataFrame,
        held_weights: pd.DataFrame,
        *,
        annualization: int,
    ) -> ExecutionCostBreakdown:
        """Calculate causal costs for weights decided one bar earlier."""

        if annualization <= 0:
            raise ValueError("annualization must be strictly positive")
        if not target_weights.index.equals(held_weights.index):
            raise ValueError("Target and held weights must share the same index")
        if list(target_weights.columns) != list(held_weights.columns):
            raise ValueError("Target and held weights must share the same columns")

        turnover_by_leg = target_weights.diff().abs().shift(1).fillna(0.0)
        turnover = turnover_by_leg.sum(axis=1).rename("turnover")
        bps = 10_000.0

        frame = pd.DataFrame(index=target_weights.index)
        frame["turnover"] = turnover
        frame["legacy_cost"] = turnover * self.legacy_cost_bps / bps
        frame["commission_cost"] = turnover * self.commission_bps / bps
        frame["spread_cost"] = turnover * self.half_spread_bps / bps
        frame["slippage_cost"] = turnover * self.slippage_bps / bps
        frame["impact_cost"] = (
            turnover_by_leg.pow(2).sum(axis=1) * self.impact_coefficient_bps / bps
        )

        short_notional = (-held_weights).clip(lower=0.0).sum(axis=1)
        gross_notional = held_weights.abs().sum(axis=1)
        frame["borrow_cost"] = (
            short_notional * self.borrow_rate_bps_annual / bps / annualization
        )
        frame["financing_cost"] = (
            gross_notional * self.financing_rate_bps_annual / bps / annualization
        )
        components = [
            "legacy_cost",
            "commission_cost",
            "spread_cost",
            "slippage_cost",
            "impact_cost",
            "borrow_cost",
            "financing_cost",
        ]
        frame["transaction_cost"] = frame[components].sum(axis=1)
        return ExecutionCostBreakdown(frame=frame)
