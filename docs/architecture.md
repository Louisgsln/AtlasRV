# Architecture

AtlasRV separates market assumptions from reusable research infrastructure.

~~~mermaid
flowchart TD
    A["Sources"] --> B["Validation + snapshot"]
    B --> C["Diagnostics + FDR"]
    C --> D["Hedge model"]
    D --> E["Signal + state machine"]
    E --> F["Walk-forward backtest"]
    F --> G["Execution costs"]
    G --> H["Portfolio allocation"]
    H --> I["Regimes + reporting"]
~~~

## Module boundaries

| Package | Responsibility | Key invariant |
|---|---|---|
| data | Providers, cleaning, validation, immutable snapshots | Inputs are aligned and auditable |
| research | Diagnostics, FDR, models comparison, regimes | Selection never silently becomes a performance claim |
| models | Kalman, rolling OLS, expanding OLS | State at t cannot use y after t |
| signals | Causal z-score and trade state machine | Standardisation history ends at t-1 |
| backtest | Position timing and purged walk-forward folds | Target at t earns return t to t+1 |
| execution | Trading and holding costs | Gross minus named costs equals net |
| risk | Performance and sleeve allocation | Risk estimates and correlations are lagged |
| reporting | CSV, JSON, Markdown, HTML, charts | Every headline result has inspectable inputs |

## Dependency direction

The backtester depends on configuration, models, signals, execution, and metrics.
Research comparison may orchestrate backtests, but core models never import the
research runner. Data providers do not know about strategies. Reporting consumes
result objects and does not recompute trading decisions.

## Extension points

- Implement MarketDataSource for a licensed provider.
- Add a causal estimator returning DynamicRegressionResult.
- Add cost components without changing gross returns.
- Add portfolio constraints before the held-weight multiplication.
- Add report consumers using the existing CSV and JSON artefacts.
