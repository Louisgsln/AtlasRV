# AtlasRV interview guide

## Thirty-second version

> AtlasRV is a Python platform for cross-asset relative-value research. It compares
> three causal hedge-ratio models, controls multiple testing, selects signal parameters
> through purged walk-forward folds, and uses next-bar execution with separately
> attributed spread, slippage, impact, borrow, and funding costs. Approved sleeves
> are combined with lagged volatility and correlation controls. Every run produces
> auditable data, risk, regime, and reporting artefacts, and the tests prove that
> changing future prices cannot alter the past.

## Two-minute walkthrough

1. **Problem:** economically linked assets can adjust at different speeds, while
   static relationships and naive backtests frequently break.
2. **Data:** all sources implement one price-frame contract; validation and a
   SHA-256 snapshot make the exact research input reproducible.
3. **Gate:** cointegration, ADF, half-life, Hurst, and beta stability are combined
   with Benjamini-Hochberg false-discovery control.
4. **Model:** expanding OLS, rolling OLS, and Kalman all emit causal one-step errors.
5. **Signal:** the current error is standardised with history ending yesterday and
   passed through an entry, exit, stop, and cooldown state machine.
6. **Execution:** a target formed at close t earns only the return from t to t+1.
   Gross return reconciles to net through seven named cost components.
7. **Validation:** parameters are selected inside rolling training windows, separated
   from disjoint test windows by a purge.
8. **Portfolio:** ex-ante inverse volatility is adjusted for cross-sleeve correlation,
   capped, and reported with effective bets and asset-class mix.
9. **Interpretation:** performance is attributed to regimes labelled with prior data.
10. **Engineering:** typed package, CLI, tests, CI, Docker, deterministic fixture,
    integrity manifest, HTML report, and Streamlit dashboard.

## Technical questions to expect

### Why relative value rather than outright direction?

The hypothesis concerns a temporary deviation from a shared economic driver. Hedging
the common component aims to isolate convergence, although residual factor exposure
always remains and must be measured.

### Why compare three hedge estimators?

A dynamic model can look superior because other assumptions changed. AtlasRV holds
the signal, execution, costs, and risk logic constant. Expanding OLS is the slow
causal benchmark, rolling OLS adapts through a hard window, and Kalman adapts
continuously through process noise.

### Why not use full-sample static OLS?

Its coefficient at the beginning of the backtest uses prices from the end. That is
look-ahead leakage. Expanding OLS estimates the slow benchmark using observations
strictly before each prediction.

### What does the Kalman delta control?

It governs process variance and therefore how quickly alpha and beta may change.
Too small a value under-reacts to a new relationship; too large a value can absorb
noise and erase the mean-reverting error. It must be stress-tested out of sample.

### Is cointegration enough?

No. It is a noisy statistical diagnostic. The economic link, stationarity, plausible
half-life, Hurst, beta stability, costs, borrow, liquidity, and out-of-sample behaviour
all matter. A low p-value does not guarantee tradability or permanence.

### Why introduce q-values?

With many candidates, some p-values fall below 5% by chance. Benjamini-Hochberg
controls the expected false-discovery proportion across the registered universe.
It is less conservative than family-wise error control but still does not eliminate
human specification mining.

### Where can look-ahead enter?

- full-sample regression or normalisation;
- a z-score window including the current or future error incorrectly;
- same-bar signal and return;
- volatility or covariance using the return being sized;
- revised macro data;
- adjusted futures rolls built with future information;
- universe selection after seeing the entire history;
- regime labels computed without a shift.

AtlasRV has future-perturbation tests for model and backtest invariance.

### How do costs work?

Trading costs scale with per-leg turnover. Market impact is quadratic. Borrow applies
to held short notional, and funding applies to gross held notional. Each component
is stored separately, so gross minus costs must exactly equal net.

### Why penalise correlation after inverse volatility?

Inverse volatility alone can allocate to several names that are all the same risk-off
trade. A trailing absolute-correlation penalty reduces redundant sleeves. The
effective number of bets shows whether nominal diversification is real.

### What does the asset-class allocation mean?

A sleeve weight is split across its registered asset classes for an allocation-mix
view. It is not a directional sensitivity. A production factor model would estimate
equity beta, duration, curve, commodity delta, FX, vega, and liquidity exposures.

### Are the regime results predictive?

No. The labels are causal but descriptive. They show where the strategy historically
worked or failed. Trading the labels would require a separate forecast and its own
walk-forward validation.

### Why include synthetic data?

It provides known ground truth, deterministic CI, no credentials, and a deliberate
structural break. It proves software behaviour and failure handling. It cannot prove
an economic edge.

## Design trade-offs to volunteer

- Log prices give an elasticity-like beta but exclude non-positive levels.
- Daily closes simplify timing while hiding asynchronous sessions and intraday fills.
- The cost model is transparent but not a limit-order-book simulator.
- FDR controls registered p-values, not repeated human redesign of the universe.
- Correlation-adjusted inverse volatility is interpretable but not a full optimiser.
- ETF and continuous-futures proxies hide basis and roll construction choices.

## Next institutional extensions

1. exchange-calendar-aware timestamp alignment;
2. explicit futures-chain rolls and contract-level P&L;
3. point-in-time corporate actions and constituent histories;
4. ADV-based nonlinear capacity and partial-fill simulation;
5. factor constraints for beta, duration, curve, vega, currency, and liquidity;
6. nested cross-validation or deflated Sharpe for broader model searches;
7. immutable vendor-data versions and experiment registry;
8. paper-trading integration with pre-trade risk limits.

## Good closing line

> The point of AtlasRV is not to present a perfect Sharpe ratio. It is to show
> that I know where quantitative research lies to you, and that I can encode
> safeguards against those failure modes in a maintainable system.
