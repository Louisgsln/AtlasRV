# Reproducible demo results

These numbers come from the deterministic synthetic universe generated with
`seed=7`. They validate information timing, model behaviour, reporting, and
failure handling. They are **not** evidence of expected returns in real markets.

## Out-of-sample portfolio

| Metric | Result |
|---|---:|
| Held-out observations | 882 |
| Annualized return | 8.52% |
| Annualized volatility | 5.41% |
| Sharpe ratio | 1.54 |
| Maximum drawdown | -3.99% |
| Relationships traded | 4 / 5 |

![Portfolio equity and drawdown](assets/demo_portfolio.png)

## Research gate

| Relationship | Coint p-value | Half-life | Hurst | Beta instability | Decision |
|---|---:|---:|---:|---:|:---:|
| Oil / energy equities | <0.0001 | 9.3d | 0.16 | 5.4% | PASS |
| Copper / miners | <0.0001 | 13.1d | 0.26 | 8.6% | PASS |
| Gold / TIPS | <0.0001 | 13.4d | 0.22 | 32.2% | PASS |
| Credit / equity | <0.0001 | 7.0d | 0.12 | 12.3% | PASS |
| Banks / curve proxy | 0.9871 | n/a | 0.64 | 335.5% | REVIEW |

The rejected relationship contains a deliberate structural beta change. It is
diagnosed and reported but excluded from the portfolio.

## Sleeve-level held-out results

| Sleeve | Sharpe | Annualized return | Maximum drawdown | Trades |
|---|---:|---:|---:|---:|
| Oil / energy equities | 0.85 | 7.75% | -7.64% | 18 |
| Copper / miners | 1.14 | 11.56% | -10.77% | 20 |
| Gold / TIPS | 0.90 | 7.62% | -8.81% | 37 |
| Credit / equity | 0.64 | 6.22% | -12.72% | 55 |

Reproduce the run with:

```bash
atlas-rv demo --observations 1500 --seed 7 --output reports/demo
```

