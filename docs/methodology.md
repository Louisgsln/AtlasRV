# Methodology and information timing

AtlasRV is designed around a simple rule: a return can only be earned with
information that existed before that return began.

## 1. Economic relationship first

A candidate pair is registered with two instruments, their asset classes, and
an economic thesis. Statistical similarity alone is not sufficient. Examples
include crude oil versus energy equities, high-yield credit versus equities,
and gold versus inflation-protected bonds.

## 2. Dynamic fair value

Positive price levels are transformed to logs and modelled as

\[
\log Y_t = \alpha_t + \beta_t \log X_t + \varepsilon_t.
\]

The state \((\alpha_t, \beta_t)\) follows a random walk and is estimated with a
Kalman filter. The spread used by the strategy is the one-step innovation

\[
e_t = \log Y_t - \hat{\alpha}_{t|t-1}
      - \hat{\beta}_{t|t-1}\log X_t.
\]

Because the prediction is formed before observing \(Y_t\), this is not a fitted
in-sample residual.

## 3. Causal standardisation

The z-score compares today's innovation with a trailing distribution ending at
\(t-1\):

\[
z_t = \frac{e_t - \mu(e_{t-L:t-1})}{\sigma(e_{t-L:t-1})}.
\]

Long-spread and short-spread trades enter beyond symmetric thresholds, exit
inside a smaller band, and stop beyond an outer band. A stopped relationship
must return inside the entry band before it can trade again.

## 4. Hedge weights and execution

For spread direction \(s_t \in \{-1,0,1\}\), weights are normalised to unit gross
exposure before volatility scaling:

\[
w^Y_t = \frac{s_t}{1 + |\beta_t|}, \qquad
w^X_t = -\frac{s_t\beta_t}{1 + |\beta_t|}.
\]

Targets formed at close \(t\) are applied to the close-to-close return
\(t \rightarrow t+1\). Two-leg turnover is

\[
\tau_{t+1} = |w^Y_t-w^Y_{t-1}| + |w^X_t-w^X_{t-1}|,
\]

and linear trading cost is \(\tau_{t+1}c\), where \(c\) is expressed in basis
points. The implementation exposes gross return, cost, net return, and turnover
as separate columns.

## 5. Volatility control

An optional multiplier targets annualised portfolio volatility using realised
strategy returns available at the decision date. It is capped to prevent a calm
window from producing unbounded leverage.

## 6. Walk-forward selection

Each fold contains:

1. a rolling training window;
2. a purge interval;
3. a disjoint test window.

Lookback and entry threshold are selected only on the training window. The
selection objective is training Sharpe after costs minus a turnover penalty.
All headline strategy returns from the default demo are concatenated test
windows, not training returns.

## 7. Research gate

AtlasRV reports, but does not blindly trust:

- Engle-Granger cointegration p-value;
- ADF p-value of the static residual;
- estimated mean-reversion half-life;
- Hurst exponent;
- return correlation;
- first-half versus second-half beta instability.

The synthetic universe deliberately includes a structural beta change so the
gate has something it should reject.

## 8. Cross-sleeve allocation

Held-out pair returns are combined with lagged inverse-volatility weights. The
weights rebalance periodically and are concentration-capped; when too few
sleeves have estimable volatility, unused risk remains in cash.

## Limitations

- Cointegration does not imply a causal or permanent relationship.
- A universe selected after observing history creates selection bias.
- Yahoo-adjusted proxies do not model futures rolls, financing, bid/ask spreads,
  asynchronous closing times, corporate actions, or point-in-time constituents.
- Linear costs omit market impact and capacity.
- Hyperparameter searches create multiple-testing risk even with walk-forward
  evaluation.
- Synthetic results validate software behaviour, not profitability.

