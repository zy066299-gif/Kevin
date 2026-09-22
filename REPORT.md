# Phase 1 Backtest Results

## TL;DR

- The highest annualized return among the 36 strategy/ticker tests was **Buy & Hold on AAPL** at **27.6%**.
- The highest median annualized return across the nine-instrument universe was **Buy & Hold** at **14.2%**.
- Data were loaded from the local **cache** path. Quality status required review for: **none**.
- These are frictionless, in-sample, long/cash results on an ex-post survivor universe. Treat them as a learning exercise, not an investment recommendation.

## Strategy-level medians

| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Win rate | Exposure | Beat B&H |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buy & Hold | 14.2% | 22.0% | 0.72 | 38.5% | 53.1% | 100.0% | benchmark |
| SMA 50/200 | 7.1% | 18.1% | 0.48 | 41.9% | 52.9% | 74.6% | 0/9 |
| Momentum 12M (absolute) | 7.7% | 18.0% | 0.54 | 31.4% | 52.7% | 76.6% | 0/9 |
| RSI 14 | 2.9% | 11.9% | 0.28 | 28.0% | 53.1% | 9.3% | 0/9 |

## Interpretation guardrails

- All signals are lagged one session before returns are applied.
- Adjusted close is used for indicators and returns; dividend cash flows are not added again.
- Buy-and-hold, SMA, momentum, and RSI are evaluated over identical dates for each ticker.
- Cash earns 0%; this understates long/cash strategies during high-rate periods.
- Costs, slippage, taxes, next-open gaps, shorting, and capacity are omitted from headline metrics.
- Nine assets and four approaches create multiple-comparison risk; isolated winners may be luck.
- The selected stocks all survived the full period, so the stock universe is survivorship-biased.

See `results/performance_metrics.csv`, `results/strategy_summary.csv`, and the executed notebook for the auditable detail.
