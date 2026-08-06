"""Optional Streamlit viewer for a generated AtlasRV research bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="AtlasRV", layout="wide")
st.title("AtlasRV — Cross-Asset Relative Value Lab")
bundle = Path(st.sidebar.text_input("Research bundle", "reports/demo"))

if not (bundle / "summary.json").exists():
    st.info("Generate a bundle first: `atlas-rv demo --output reports/demo`")
    st.stop()

summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
portfolio = pd.read_csv(bundle / "portfolio.csv", index_col="date", parse_dates=True)
diagnostics = pd.read_csv(bundle / "diagnostics.csv", index_col="pair")

metrics = summary["portfolio"]
columns = st.columns(4)
columns[0].metric("Sharpe", f"{metrics['sharpe']:.2f}")
columns[1].metric("Annual return", f"{metrics['annualized_return']:.1%}")
columns[2].metric("Annual volatility", f"{metrics['annualized_volatility']:.1%}")
columns[3].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")

st.plotly_chart(
    px.line(portfolio, y="equity", title="Out-of-sample portfolio equity"),
    use_container_width=True,
)
st.subheader("Relationship diagnostics")
st.dataframe(
    diagnostics[
        [
            "coint_pvalue",
            "half_life_days",
            "hurst_exponent",
            "beta_instability",
            "passes_research_gate",
        ]
    ],
    use_container_width=True,
)
