from __future__ import annotations

import math

import numpy as np
import pandas as pd


def performance_metrics(
    detail: pd.DataFrame,
    trading_days_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, float | int | str]:
    ordered = detail.sort_values("date").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("At least two dated observations are required")
    realized = ordered.iloc[1:].copy()
    returns = realized["strategy_return"].astype(float).fillna(0.0)
    positions = realized["position"].astype(float).fillna(0.0)
    equity = (1.0 + ordered["strategy_return"].astype(float).fillna(0.0)).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    observations = int(len(returns))
    if equity.iloc[-1] <= 0:
        annualized_return = np.nan
    else:
        annualized_return = float(equity.iloc[-1] ** (trading_days_per_year / observations) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(trading_days_per_year))
    daily_risk_free = (1.0 + annual_risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    daily_std = float(returns.std(ddof=1))
    sharpe = (
        float((returns.mean() - daily_risk_free) / daily_std * math.sqrt(trading_days_per_year))
        if daily_std > 0
        else np.nan
    )
    equity_with_anchor = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)], ignore_index=True)
    drawdown = equity_with_anchor / equity_with_anchor.cummax() - 1.0
    max_drawdown = float(-drawdown.min())
    active_mask = positions > 0
    win_rate = float((returns.loc[active_mask] > 0).mean()) if active_mask.any() else np.nan
    exposure = float(active_mask.mean())
    entry_flags = realized["position"].diff().gt(0)
    entry_flags.iloc[0] = bool(realized["position"].iloc[0] > 0)
    entries = int(entry_flags.sum())
    turnover = float(ordered["turnover"].fillna(0.0).sum())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "exposure": exposure,
        "entries": entries,
        "turnover": turnover,
        "observations": observations,
        "start_date": ordered["date"].iloc[0].date().isoformat(),
        "end_date": ordered["date"].iloc[-1].date().isoformat(),
    }
