"""Statistical diagnostics, model comparison, regimes, and multiple testing."""

from atlas_rv.research.diagnostics import PairDiagnostics, diagnose_pair
from atlas_rv.research.model_comparison import (
    ModelComparisonResult,
    compare_hedge_models,
)
from atlas_rv.research.multiple_testing import (
    apply_false_discovery_control,
    benjamini_hochberg,
)
from atlas_rv.research.regimes import (
    RegimeAnalysis,
    analyze_regimes,
    classify_market_regimes,
)

__all__ = [
    "ModelComparisonResult",
    "PairDiagnostics",
    "RegimeAnalysis",
    "analyze_regimes",
    "apply_false_discovery_control",
    "benjamini_hochberg",
    "classify_market_regimes",
    "compare_hedge_models",
    "diagnose_pair",
]
