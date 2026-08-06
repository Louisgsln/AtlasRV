"""Interactive Streamlit explorer for an AtlasRV research bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="AtlasRV", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {background:#f5f8fb; border:1px solid #d9e2ec;
      padding:14px; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("AtlasRV · Cross-Asset Relative Value Lab")
st.caption("Causal research, transparent costs, portfolio risk, and reproducible artefacts")

bundle = Path(st.sidebar.text_input("Research bundle", "reports/research"))
st.sidebar.markdown("Generate one with: atlas-rv research --provider synthetic")

if not (bundle / "summary.json").exists():
    st.info("No research bundle found at this path.")
    st.stop()


@st.cache_data
def load_bundle(path: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    directory = Path(path)
    summary_data = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    portfolio_data = pd.read_csv(
        directory / "portfolio.csv",
        index_col="date",
        parse_dates=True,
    )
    diagnostics_data = pd.read_csv(directory / "diagnostics.csv", index_col="pair")
    return summary_data, portfolio_data, diagnostics_data


summary, portfolio, diagnostics = load_bundle(str(bundle))
metrics = summary["portfolio"]
dataset = summary.get("dataset", {})
if isinstance(dataset, dict):
    st.sidebar.caption(str(dataset.get("label", "research dataset")))


def metric_text(key: str, *, percent: bool = False) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.1%}" if percent else f"{float(value):.2f}"


columns = st.columns(5)
columns[0].metric("Sharpe", metric_text("sharpe"))
columns[1].metric("Annual return", metric_text("annualized_return", percent=True))
columns[2].metric("Annual volatility", metric_text("annualized_volatility", percent=True))
columns[3].metric("Max drawdown", metric_text("max_drawdown", percent=True))
columns[4].metric("Effective bets", metric_text("average_effective_bets"))

overview, allocation, research, pair_tab, integrity = st.tabs(
    ["Overview", "Allocation & risk", "Research gate", "Pair explorer", "Integrity"]
)

with overview:
    equity_figure = go.Figure()
    equity_figure.add_trace(
        go.Scatter(
            x=portfolio.index,
            y=portfolio["equity"],
            name="Growth of $1",
            line={"color": "#2F80ED", "width": 2},
        )
    )
    equity_figure.update_layout(
        title="Portfolio equity",
        template="plotly_white",
        height=430,
    )
    st.plotly_chart(equity_figure, use_container_width=True)

    left, right = st.columns(2)
    with left:
        drawdown_figure = px.area(
            portfolio,
            y="drawdown",
            title="Drawdown",
            color_discrete_sequence=["#D64545"],
        )
        drawdown_figure.update_layout(template="plotly_white", height=300)
        st.plotly_chart(drawdown_figure, use_container_width=True)
    with right:
        regime_path = bundle / "regime_metrics.csv"
        if regime_path.exists():
            regimes = pd.read_csv(regime_path, index_col="regime")
            st.subheader("Conditional performance")
            st.dataframe(regimes, use_container_width=True)
        else:
            st.info("No regime attribution in this bundle.")

with allocation:
    weights_path = bundle / "portfolio_weights.csv"
    classes_path = bundle / "portfolio_class_allocations.csv"
    if weights_path.exists():
        weights = pd.read_csv(weights_path, index_col="date", parse_dates=True)
        st.plotly_chart(
            px.area(weights, title="Sleeve allocations"),
            use_container_width=True,
        )
        correlation = portfolio.filter(like="sleeve_").corr()
        st.plotly_chart(
            px.imshow(
                correlation,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Realised sleeve correlation",
            ),
            use_container_width=True,
        )
    if classes_path.exists():
        classes = pd.read_csv(classes_path, index_col="date", parse_dates=True)
        st.plotly_chart(
            px.area(classes, title="Allocation mix by asset class"),
            use_container_width=True,
        )

with research:
    display_columns = [
        "coint_pvalue",
        "coint_qvalue",
        "adf_pvalue",
        "half_life_days",
        "hurst_exponent",
        "beta_instability",
        "passes_research_gate",
        "gate_reasons",
    ]
    available = [column for column in display_columns if column in diagnostics]
    st.dataframe(diagnostics[available], use_container_width=True)
    st.caption(
        "The q-value controls false discoveries across the universe; a PASS also "
        "requires stationarity, plausible half-life, Hurst, and beta stability."
    )

with pair_tab:
    pair_files = sorted((bundle / "pairs").glob("*_backtest.csv.gz"))
    if not pair_files:
        st.info("No pair-level files in this bundle.")
    else:
        labels = {path.name.replace("_backtest.csv.gz", ""): path for path in pair_files}
        selected = st.selectbox("Relationship", list(labels))
        pair_frame = pd.read_csv(labels[selected], index_col=0, parse_dates=True)
        top, bottom = st.columns(2)
        with top:
            st.plotly_chart(
                px.line(pair_frame, y="zscore", title=f"{selected}: causal z-score"),
                use_container_width=True,
            )
        with bottom:
            cost_columns = [
                column
                for column in (
                    "commission_cost",
                    "spread_cost",
                    "slippage_cost",
                    "impact_cost",
                    "borrow_cost",
                    "financing_cost",
                    "legacy_cost",
                )
                if column in pair_frame
            ]
            if cost_columns:
                cumulative_costs = pair_frame[cost_columns].cumsum()
                st.plotly_chart(
                    px.line(cumulative_costs, title="Cumulative cost attribution"),
                    use_container_width=True,
                )
        st.dataframe(pair_frame.tail(100), use_container_width=True)

with integrity:
    snapshot = dataset.get("snapshot", {}) if isinstance(dataset, dict) else {}
    if isinstance(snapshot, dict) and snapshot:
        st.json(snapshot)
    st.markdown(
        """
        Every bundle contains canonical price bytes and a SHA-256 manifest.
        Re-reading a modified snapshot fails closed, making research inputs auditable.
        """
    )
