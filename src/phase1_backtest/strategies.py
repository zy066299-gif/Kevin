from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig


STRATEGY_ORDER = ["Buy & Hold", "SMA 50/200", "Momentum 12M (absolute)", "RSI 14"]


def wilder_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with an SMA seed and recursive 1/period smoothing."""
    values = pd.to_numeric(prices, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= period:
        return output

    deltas = values.diff().to_numpy(dtype=float)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    if np.isnan(deltas[1 : period + 1]).any():
        return output

    average_gain = float(np.mean(gains[1 : period + 1]))
    average_loss = float(np.mean(losses[1 : period + 1]))

    def to_rsi(avg_gain: float, avg_loss: float) -> float:
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
        relative_strength = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + relative_strength)

    output.iloc[period] = to_rsi(average_gain, average_loss)
    for position in range(period + 1, len(values)):
        if not np.isfinite(deltas[position]):
            average_gain = np.nan
            average_loss = np.nan
            continue
        if not np.isfinite(average_gain) or not np.isfinite(average_loss):
            window = deltas[position - period + 1 : position + 1]
            if np.isnan(window).any():
                continue
            average_gain = float(np.mean(np.where(window > 0, window, 0.0)))
            average_loss = float(np.mean(np.where(window < 0, -window, 0.0)))
        else:
            average_gain = ((period - 1) * average_gain + gains[position]) / period
            average_loss = ((period - 1) * average_loss + losses[position]) / period
        output.iloc[position] = to_rsi(average_gain, average_loss)
    return output


def rsi_regime_signal(rsi: pd.Series, entry: float = 30.0, exit: float = 50.0) -> pd.Series:
    if entry >= exit:
        raise ValueError("RSI entry must be below the exit threshold")
    in_market = False
    signal = pd.Series(0.0, index=rsi.index, dtype=float)
    for index, value in rsi.items():
        if pd.isna(value):
            signal.loc[index] = 0.0
            continue
        if not in_market and value < entry:
            in_market = True
        elif in_market and value >= exit:
            in_market = False
        signal.loc[index] = float(in_market)
    return signal


def generate_signals(prices: pd.Series, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = pd.to_numeric(prices, errors="coerce").astype(float)
    fast = adjusted.rolling(config.sma_fast, min_periods=config.sma_fast).mean()
    slow = adjusted.rolling(config.sma_slow, min_periods=config.sma_slow).mean()
    momentum = adjusted.pct_change(config.momentum_lookback, fill_method=None)
    rsi = wilder_rsi(adjusted, config.rsi_period)

    signals = pd.DataFrame(index=adjusted.index)
    signals["Buy & Hold"] = adjusted.notna().astype(float)
    signals["SMA 50/200"] = (fast > slow).where(slow.notna(), 0.0).astype(float)
    signals["Momentum 12M (absolute)"] = (momentum > 0).where(momentum.notna(), 0.0).astype(float)
    signals["RSI 14"] = rsi_regime_signal(rsi, config.rsi_entry, config.rsi_exit)

    indicators = pd.DataFrame(
        {
            "adj_close": adjusted,
            "sma_fast": fast,
            "sma_slow": slow,
            "momentum_12m": momentum,
            "rsi_14": rsi,
        },
        index=adjusted.index,
    )
    return signals[STRATEGY_ORDER], indicators
