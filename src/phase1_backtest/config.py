from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    tickers: tuple[str, ...]
    download_start: str
    evaluation_start: str
    end_exclusive: str
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0
    sma_fast: int = 50
    sma_slow: int = 200
    momentum_lookback: int = 252
    rsi_period: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 50.0
    cost_sensitivity_bps: float = 10.0

    @property
    def evaluation_end(self) -> str:
        """Last inclusive calendar date implied by the exclusive download end."""
        import pandas as pd

        return str((pd.Timestamp(self.end_exclusive) - pd.Timedelta(days=1)).date())


def load_config(path: str | Path) -> BacktestConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["tickers"] = tuple(payload["tickers"])
    return BacktestConfig(**payload)

