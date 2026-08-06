"""False-discovery control across a researched universe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from atlas_rv.research.diagnostics import PairDiagnostics


def benjamini_hochberg(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Return Benjamini-Hochberg adjusted p-values keyed like the input."""

    if not pvalues:
        return {}
    names = list(pvalues)
    values = np.asarray([pvalues[name] for name in names], dtype=float)
    if (~np.isfinite(values)).any() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must be finite and lie in [0, 1]")

    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    count = len(values)
    for reverse_position in range(count - 1, -1, -1):
        rank = reverse_position + 1
        running = min(running, float(ranked[reverse_position] * count / rank))
        adjusted[order[reverse_position]] = min(running, 1.0)
    return {name: float(adjusted[position]) for position, name in enumerate(names)}


def apply_false_discovery_control(
    diagnostics: Mapping[str, PairDiagnostics],
    *,
    alpha: float = 0.05,
) -> dict[str, PairDiagnostics]:
    """Apply universe-level FDR control without weakening pair-level gates."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    qvalues = benjamini_hochberg(
        {name: item.coint_pvalue for name, item in diagnostics.items()}
    )
    controlled: dict[str, PairDiagnostics] = {}
    for name, item in diagnostics.items():
        reasons = list(item.gate_reasons)
        if qvalues[name] > alpha and "cointegration_fdr" not in reasons:
            reasons.append("cointegration_fdr")
        controlled[name] = replace(
            item,
            coint_qvalue=qvalues[name],
            passes_research_gate=item.passes_research_gate and qvalues[name] <= alpha,
            gate_reasons=tuple(reasons),
        )
    return controlled
