from __future__ import annotations

import pandas as pd

from .config import BacktestConfig
from .strategies import generate_signals


def backtest_one_ticker(
    frame: pd.DataFrame,
    config: BacktestConfig,
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("date").set_index("date")
    adjusted = ordered["adj_close"].astype(float)
    asset_return = adjusted.pct_change(fill_method=None)
    signals, indicators = generate_signals(adjusted, config)

    # A signal observed at close t becomes the position for the following return.
    positions = signals.shift(1).fillna(0.0)
    if not positions.isin([0.0, 1.0]).all().all():
        raise ValueError("Only long/cash positions are supported")

    evaluation_mask = (positions.index >= pd.Timestamp(config.evaluation_start)) & (
        positions.index < pd.Timestamp(config.end_exclusive)
    )
    positions = positions.loc[evaluation_mask].copy()
    signals = signals.loc[evaluation_mask].copy()
    asset_return = asset_return.loc[evaluation_mask].copy()
    indicators = indicators.loc[evaluation_mask].copy()
    if len(positions) < 2:
        raise ValueError("Evaluation window has fewer than two observations")

    turnover = positions.diff().abs()
    # The first row is an equity anchor at the evaluation-start close. Establish
    # the position for the first measured return on the following row, rather
    # than charging for a pre-window position that earns no in-window return.
    turnover.iloc[0] = 0.0
    turnover.iloc[1] = positions.iloc[1].abs()
    cost_rate = float(transaction_cost_bps) / 10_000.0
    strategy_returns = positions.mul(asset_return, axis=0) - turnover * cost_rate

    # The first evaluation row is the normalized equity anchor; returns begin next session.
    strategy_returns.iloc[0] = 0.0
    equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()

    details: list[pd.DataFrame] = []
    for strategy in positions.columns:
        detail = pd.DataFrame(
            {
                "asset_return": asset_return,
                "signal": signals[strategy],
                "position": positions[strategy],
                "turnover": turnover[strategy],
                "strategy_return": strategy_returns[strategy],
                "equity": equity[strategy],
            }
        )
        detail["strategy"] = strategy
        details.append(detail.reset_index())
    detail_frame = pd.concat(details, ignore_index=True)
    indicator_frame = indicators.reset_index()
    return detail_frame, indicator_frame
