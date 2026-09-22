import pandas as pd

from phase1_backtest.config import BacktestConfig
from phase1_backtest.data import canonical_sha256, clean_and_profile


def test_cleaner_drops_invalid_adjusted_prices_without_forward_fill():
    config = BacktestConfig(
        tickers=("AAA", "BBB"),
        download_start="2020-01-01",
        evaluation_start="2020-01-01",
        end_exclusive="2020-02-01",
    )
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    rows = []
    for ticker in config.tickers:
        for date, price in zip(dates, [100.0, 101.0]):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "adj_close": price,
                    "volume": 1000,
                    "dividends": 0,
                    "stock_splits": 0,
                }
            )
    rows[-1]["adj_close"] = None
    cleaned, quality = clean_and_profile(pd.DataFrame(rows), config)
    assert len(cleaned.loc[cleaned["ticker"] == "BBB"]) == 1
    assert quality.set_index("ticker").loc["BBB", "invalid_or_missing_adj_close_rows"] == 1


def test_canonical_hash_is_stable_for_same_rows():
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ticker": ["AAA"], "value": [1.0]})
    assert canonical_sha256(frame) == canonical_sha256(frame.copy())

