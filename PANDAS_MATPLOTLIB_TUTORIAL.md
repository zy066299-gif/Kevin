# Short pandas + Matplotlib Tutorial

This 15-minute walkthrough uses the project's cleaned snapshot. Run it in a notebook after activating the project environment.

## 1. Import the libraries and load data

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

project_root = Path.cwd()
prices = pd.read_parquet(project_root / "data" / "processed" / "clean_prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])

print(prices.shape)
prices.head()
```

A pandas `DataFrame` is a table. Here, one row represents one ticker on one trading date.

## 2. Select columns and filter rows

```python
spy = (
    prices.loc[prices["ticker"].eq("SPY"), ["date", "adj_close", "volume"]]
    .sort_values("date")
    .set_index("date")
)
spy.loc["2020-02":"2020-04"].head()
```

`.loc[...]` selects rows and columns. The chained operations keep the data ordered and make dates the index.

## 3. Calculate a return and moving average

```python
spy["daily_return"] = spy["adj_close"].pct_change(fill_method=None)
spy["sma_50"] = spy["adj_close"].rolling(50, min_periods=50).mean()
spy["sma_200"] = spy["adj_close"].rolling(200, min_periods=200).mean()
spy.tail()
```

`pct_change` calculates fractional change; `0.01` means 1%. `rolling(...).mean()` calculates a moving average without using future rows.

## 4. Summarize by ticker

```python
summary = (
    prices.sort_values(["ticker", "date"])
    .groupby("ticker")
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        observations=("date", "size"),
        average_volume=("volume", "mean"),
    )
)
summary
```

`groupby` splits the table by ticker; `agg` calculates one or more summaries for each group.

## 5. Draw a clear chart

```python
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(spy.index, spy["adj_close"], label="Adjusted close", color="#2457A6", linewidth=1.8)
ax.plot(spy.index, spy["sma_50"], label="50-session SMA", color="#D46A1F", linewidth=1.3)
ax.plot(spy.index, spy["sma_200"], label="200-session SMA", color="#D19A00", linewidth=1.3)
ax.set(
    title="SPY adjusted close and moving averages",
    xlabel="Date",
    ylabel="Adjusted price (USD)",
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#D9DDE3", linewidth=0.7)
ax.legend(frameon=False)
plt.show()
```

Every analytical chart should have a descriptive title, labeled units, readable contrast, and a legend only when it adds information.

## 6. Apply the anti-look-ahead pattern

```python
spy["signal"] = (spy["sma_50"] > spy["sma_200"]).astype(float)
spy["position"] = spy["signal"].shift(1).fillna(0.0)
spy["strategy_return"] = spy["position"] * spy["daily_return"]
```

The `shift(1)` is essential: today's closing-price signal controls the next session's return. Without the shift, the backtest would let a signal earn the same return used to create it.

## 7. Continue into the full project

Open `notebooks/phase1_backtesting.ipynb` for the complete data-quality checks, four strategies, performance table, equity curves, and cost sensitivity. The reusable implementation lives in `src/phase1_backtest/`.

