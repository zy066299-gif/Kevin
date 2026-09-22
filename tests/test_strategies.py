from dataclasses import replace

import numpy as np
import pandas as pd

from phase1_backtest.config import BacktestConfig
from phase1_backtest.strategies import generate_signals, rsi_regime_signal, wilder_rsi


BASE = BacktestConfig(
    tickers=("TEST",),
    download_start="2020-01-01",
    evaluation_start="2020-01-02",
    end_exclusive="2025-01-01",
)


def test_rising_and_falling_series_have_expected_regimes():
    index = pd.bdate_range("2020-01-01", periods=320)
    rising = pd.Series(np.linspace(100, 200, len(index)), index=index)
    falling = pd.Series(np.linspace(200, 100, len(index)), index=index)
    rising_signals, rising_indicators = generate_signals(rising, BASE)
    falling_signals, falling_indicators = generate_signals(falling, BASE)
    assert rising_signals["SMA 50/200"].iloc[-1] == 1
    assert rising_signals["Momentum 12M (absolute)"].iloc[-1] == 1
    assert rising_indicators["rsi_14"].iloc[-1] == 100
    assert falling_signals["SMA 50/200"].iloc[-1] == 0
    assert falling_signals["Momentum 12M (absolute)"].iloc[-1] == 0
    assert falling_signals["RSI 14"].iloc[-1] == 1


def test_flat_rsi_is_neutral_and_regime_has_explicit_exit():
    prices = pd.Series([100.0] * 40)
    rsi = wilder_rsi(prices, period=14)
    assert rsi.dropna().eq(50.0).all()
    sample = pd.Series([50.0, 29.0, 25.0, 49.9, 50.0, 20.0])
    assert rsi_regime_signal(sample, entry=30, exit=50).tolist() == [0, 1, 1, 1, 0, 1]


def test_future_price_change_cannot_change_earlier_signals():
    index = pd.bdate_range("2020-01-01", periods=320)
    prices = pd.Series(100 + np.arange(len(index)) * 0.1, index=index)
    original, _ = generate_signals(prices, BASE)
    altered = prices.copy()
    altered.iloc[-1] *= 10
    changed, _ = generate_signals(altered, BASE)
    pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


def test_configurable_short_windows_do_not_change_strategy_labels():
    config = replace(BASE, sma_fast=2, sma_slow=3, momentum_lookback=2, rsi_period=2)
    prices = pd.Series([1.0, 2.0, 3.0, 4.0])
    signals, _ = generate_signals(prices, config)
    assert list(signals.columns) == ["Buy & Hold", "SMA 50/200", "Momentum 12M (absolute)", "RSI 14"]
