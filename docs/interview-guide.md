# AtlasRV interview guide

## Thirty-second version

> AtlasRV is a Python research platform for cross-asset relative-value trades.
> It starts from economically linked instruments, estimates a dynamic hedge
> ratio with a Kalman filter, and evaluates mean-reversion signals with purged
> walk-forward testing, next-bar execution, two-leg costs, and portfolio risk
> controls. I also built a deterministic synthetic universe containing a known
> structural break, so the tests prove the system can reject an unstable pair
> instead of making every backtest look successful.

## Two-minute version

1. **Problem:** static pairs models break when sensitivities and regimes change.
2. **Data contract:** every provider produces the same aligned wide price frame.
3. **Model:** sequential log-price regression estimates alpha and beta without
   peeking beyond the current close.
4. **Signal:** today's forecast error is standardised against history ending
   yesterday, then passed through an entry/exit/stop state machine.
5. **Execution:** positions are lagged by one bar and costs depend on both legs.
6. **Validation:** candidate parameters are chosen in rolling train windows,
   separated from disjoint test windows by a purge.
7. **Portfolio:** test returns are allocated with lagged inverse volatility and
   concentration limits.
8. **Engineering:** typed package, CLI, tests, CI, machine-readable artefacts,
   static report, and an optional dashboard.

## Questions to expect

### Why use a Kalman filter rather than rolling OLS?

Rolling OLS imposes a hard cutoff where the oldest observation suddenly loses
all weight. The state-space model updates continuously and explicitly controls
how quickly alpha and beta may drift. The trade-off is sensitivity to process
and observation variance, which must be stress-tested.

### Is the innovation tradable?

Not by itself. It becomes a candidate signal only after economic justification,
stability checks, realistic alignment, execution lag, and costs. A statistically
stationary residual can still be untradeable because of borrow, roll, liquidity,
or asynchronous market hours.

### Why regress log prices?

It makes beta an elasticity-like exposure and preserves positivity. It is not
appropriate for negative rates or raw spreads; those require a level or return
model with a different data contract.

### Where could look-ahead bias enter?

- z-score windows that include future observations;
- same-bar signal and P&L;
- volatility scaling using the return being sized;
- revised macro data;
- full-sample universe selection;
- adjusted futures series built with future roll information.

AtlasRV has an explicit regression test that changes future prices and asserts
that every prior model state, position, and return remains identical.

### Why include synthetic data?

It makes the repository executable without credentials or licensed market data,
and gives the model a known ground truth. It is a software test fixture. Claims
about economic performance must come from point-in-time real data.

### What would you build next?

1. exchange-calendar-aware alignment and timestamped intraday data;
2. futures-chain construction with explicit roll rules;
3. nonlinear and state-dependent transaction costs;
4. a multiple-testing correction and deflated Sharpe ratio;
5. portfolio factor constraints for equity beta, duration, commodity delta, and
   liquidity;
6. experiment tracking with immutable data snapshots.

## Honest weaknesses to volunteer

- The default cost model is linear.
- Kalman hyperparameters are fixed rather than estimated in each fold.
- The research gate uses conventional thresholds with no family-wise correction.
- ETF proxies mix trading hours and may hide basis risk.
- A relative-value portfolio can concentrate in a common risk-off factor even
  when its sleeves look different by name.

