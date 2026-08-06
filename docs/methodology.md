# Methodology and information timing

AtlasRV is built around one invariant:

> A return may only be earned with information that existed before that return began.

## 1. Economic relationship before statistics

Each candidate contains two instruments, asset classes, and an economic thesis.
Statistical similarity is not sufficient. The transmission mechanism might be
cash-flow exposure, a shared discount-rate factor, balance-sheet seniority,
duration, futures basis, funding, or a common risk-premium shock.

The config is part of the research record. Repeatedly changing a universe after
seeing results is a form of selection bias even when every backtest is coded correctly.

## 2. Data contract and immutable snapshot

Every provider returns a wide price frame indexed by timestamps. Validation checks:

- sorted and unique timestamps;
- numeric, finite observations;
- minimum history;
- missing values and conservative forward fills;
- stale-price fractions;
- strictly positive levels for log-price models.

Each run writes canonical CSV bytes plus a SHA-256 manifest containing the symbols,
dates, dimensions, and checksum. This proves which exact input bytes produced a report.

Yahoo-adjusted values are exploratory proxies. Institutional research requires
point-in-time licensed data, exchange calendars, explicit futures rolls, corporate
actions, and timestamps representing when observations became knowable.

## 3. Research gate and false discoveries

For each relationship AtlasRV reports:

- Engle-Granger cointegration p-value;
- ADF p-value of the static residual;
- mean-reversion half-life;
- Hurst exponent;
- return correlation;
- first-half versus second-half beta instability.

Testing many candidates at 5% creates false positives. The Benjamini-Hochberg
procedure sorts the m p-values and computes monotone adjusted q-values. A pair
must pass both its individual stability checks and the universe-level q-value
threshold. The correction can reject an additional pair but never reverse an
existing rejection.

FDR control does not remove all data snooping. The candidate universe and every
manual research iteration still need governance.

## 4. Three causal hedge models

Positive price levels are transformed to logs:

[
log Y_t = alpha_t + eta_t log X_t + arepsilon_t.
]

AtlasRV supports:

1. **Expanding OLS** as a slow causal benchmark. The estimate for t uses rows before t.
2. **Rolling OLS** using a fixed trailing window ending at t-1.
3. **Kalman regression** with random-walk alpha and beta states.

A full-sample static OLS coefficient is intentionally not a tradable benchmark,
because its early predictions contain late observations.

Every estimator returns the same contract: prior-state alpha, prior-state beta,
one-step innovation, and innovation variance. Model comparison holds the signal,
execution, costs, and risk settings constant.

For the Kalman model, the tradable error is the prior prediction innovation:

[
e_t = log Y_t - hat{alpha}_{t|t-1}
      - hat{eta}_{t|t-1}log X_t.
]

## 5. Causal signal state machine

The z-score compares the current innovation with a distribution ending yesterday:

[
z_t = rac{e_t - mu(e_{t-L:t-1})}
           {sigma(e_{t-L:t-1})}.
]

Long- and short-spread trades enter beyond symmetric thresholds, exit inside a
smaller band, and stop beyond an outer band. After a stop, the relationship must
return inside the entry band before it can re-enter.

## 6. Hedge weights and next-bar execution

For spread direction s and hedge ratio beta, target weights are normalised to
unit gross exposure before sleeve volatility scaling:

[
w^Y_t = rac{s_t}{1+|eta_t|},
qquad
w^X_t = -rac{s_teta_t}{1+|eta_t|}.
]

Targets formed using close t become held weights for the close-to-close return
from t to t+1. Same-bar signal and P&L are prohibited.

## 7. Execution and holding costs

Per-leg turnover is based on the target-weight change that becomes active for the
current return. The cost model exposes:

- backwards-compatible all-in linear cost;
- commission;
- half bid-ask spread;
- slippage;
- quadratic impact in per-leg turnover;
- annualised borrow on short held notional;
- annualised financing on gross held notional.

For every day:

[
r^{net}_t = r^{gross}_t
- c^{legacy}_t - c^{commission}_t - c^{spread}_t
- c^{slippage}_t - c^{impact}_t - c^{borrow}_t - c^{funding}_t.
]

These are transparent stress assumptions, not an order-book or capacity model.

## 8. Purged walk-forward selection

Each fold has a rolling training interval, purge gap, and disjoint test interval.
Lookback and entry threshold are selected only on training returns after costs,
with a turnover penalty. Headline returns concatenate test windows only.

The purge reduces temporal leakage but cannot correct a universe chosen with
full-history hindsight.

## 9. Correlation-aware portfolio construction

Approved sleeve returns are combined using ex-ante volatility estimates shifted
by one bar. At each rebalance, inverse-volatility scores are penalised when a sleeve
has high average absolute trailing correlation with the others. Weights are capped,
and optional portfolio-level volatility scaling uses a trailing covariance matrix.

Reports include:

- sleeve weights and allocation turnover;
- allocation costs;
- gross allocation;
- effective number of bets;
- allocation mix by asset class.

The asset-class mix is not a directional factor model. A production system still
needs beta, duration, convexity, vega, currency, liquidity, and concentration constraints.

## 10. Causal regime attribution

A benchmark proxy is classified by trailing volatility and trend statistics shifted
by one bar. Strategy performance is grouped by the regime known before each return.
This describes conditional behaviour; it does not predict the next regime.

## 11. Known limitations

- Cointegration is not causality and can break permanently.
- Proxy instruments introduce basis, roll, financing, and trading-hours differences.
- Daily bars omit latency, partial fills, queueing, and intraday capacity.
- Cost coefficients require venue- and size-specific calibration.
- FDR does not remove discretionary universe or specification mining.
- Regime labels are sensitive to their chosen windows.
- A relative-value portfolio can retain hidden common factors.
- Synthetic data validates software behaviour, not profitability.
