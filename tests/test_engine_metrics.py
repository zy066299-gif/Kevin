from dataclasses import replace

import numpy as np
import pandas as pd

from phase1_backtest.config import BacktestConfig
from phase1_backtest.engine import backtest_one_ticker
from phase1_backtest.metrics import performance_metrics


BASE = BacktestConfig(
    tickers=("TEST",),
    download_start="2020-01-01",
    evaluation_start="2020-01-01",
    end_exclusive="2022-01-01",
    sma_fast=2,
    sma_slow=3,
    momentum_lookback=2,
    rsi_period=2,
)


def make_frame(prices):
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame({"date": dates, "adj_close": prices})


def test_signal_is_lagged_before_return_application():
    frame = make_frame([100.0, 110.0, 121.0, 108.9, 119.79])
    detail, _ = backtest_one_ticker(frame, BASE)
    buy_hold = detail.loc[detail["strategy"] == "Buy & Hold"].reset_index(drop=True)
    expected = buy_hold["position"] * buy_hold["asset_return"]
    expected.iloc[0] = 0.0
    np.testing.assert_allclose(buy_hold["strategy_return"], expected, rtol=0, atol=1e-12)


def test_sma_transition_uses_prior_close_signal():
    frame = make_frame([3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    detail, _ = backtest_one_ticker(frame, BASE)
    sma = detail.loc[detail["strategy"] == "SMA 50/200"].reset_index(drop=True)
    assert sma["signal"].nunique() == 2
    np.testing.assert_array_equal(sma["position"].iloc[1:].to_numpy(), sma["signal"].iloc[:-1].to_numpy())
    np.testing.assert_allclose(
        sma["strategy_return"].iloc[1:],
        (sma["position"] * sma["asset_return"]).iloc[1:],
        rtol=0,
        atol=1e-12,
    )


def test_buy_and_hold_reconciles_to_first_and_last_evaluation_price():
    prices = [100.0, 110.0, 99.0, 118.8]
    detail, _ = backtest_one_ticker(make_frame(prices), BASE)
    buy_hold = detail.loc[detail["strategy"] == "Buy & Hold"]
    metrics = performance_metrics(buy_hold)
    assert np.isclose(metrics["total_return"], prices[-1] / prices[0] - 1.0)
    assert metrics["entries"] == 1


def test_max_drawdown_known_path_is_twenty_five_percent():
    detail = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=3),
            "strategy_return": [0.0, 0.2, -0.25],
            "position": [1.0, 1.0, 1.0],
            "turnover": [1.0, 0.0, 0.0],
        }
    )
    metrics = performance_metrics(detail)
    assert np.isclose(metrics["max_drawdown"], 0.25)


def test_flat_path_returns_nan_sharpe_not_infinity():
    detail = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=4),
            "strategy_return": [0.0, 0.0, 0.0, 0.0],
            "position": [1.0, 1.0, 1.0, 1.0],
            "turnover": [1.0, 0.0, 0.0, 0.0],
        }
    )
    metrics = performance_metrics(detail)
    assert np.isnan(metrics["sharpe_ratio"])
    assert metrics["annualized_volatility"] == 0.0


def test_transaction_cost_is_charged_on_position_change():
    config = replace(BASE, evaluation_start="2020-01-01")
    detail, _ = backtest_one_ticker(make_frame([100.0, 101.0, 102.0, 103.0]), config, transaction_cost_bps=10)
    buy_hold = detail.loc[detail["strategy"] == "Buy & Hold"].reset_index(drop=True)
    assert buy_hold.loc[0, "strategy_return"] == 0.0
    expected_second_return = buy_hold.loc[1, "asset_return"] - 0.001
    assert np.isclose(buy_hold.loc[1, "strategy_return"], expected_second_return)


def test_cost_boundary_ignores_pre_window_position_and_buys_starting_allocation_once():
    config = replace(BASE, evaluation_start="2020-01-03")
    detail, _ = backtest_one_ticker(make_frame([100.0, 101.0, 102.0, 103.0, 104.0]), config, transaction_cost_bps=10)
    buy_hold = detail.loc[detail["strategy"] == "Buy & Hold"].reset_index(drop=True)
    assert buy_hold.loc[0, "position"] == 1.0
    assert buy_hold.loc[0, "turnover"] == 0.0
    assert buy_hold.loc[0, "strategy_return"] == 0.0
    assert buy_hold.loc[1, "turnover"] == 1.0
    assert np.isclose(buy_hold.loc[1, "strategy_return"], buy_hold.loc[1, "asset_return"] - 0.001)
