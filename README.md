# Phase 1: Backtesting Strategy

A reproducible, GitHub-ready and Google Colab-compatible starter project for the Phase 1 brief. It downloads and audits daily Yahoo Finance data, implements four long/cash strategies, applies signals to next-session returns, evaluates each strategy/ticker pair, and saves an executed notebook plus reusable outputs.

## What is included

- Universe: `SPY`, `QQQ`, `IWM`, `AAPL`, `JPM`, `JNJ`, `XOM`, `WMT`, `CAT`
- Warm-up data: 2013-01-01 through 2013-12-31
- Fixed evaluation period: 2014-01-02 through 2024-12-31
- Strategies:
  - Buy and hold
  - 50/200-day simple moving-average regime
  - 252-session time-series momentum
  - Wilder RSI(14) mean reversion: enter below 30, exit at or above 50
- Metrics: total return, CAGR, annualized volatility, Sharpe ratio, maximum drawdown, and active-day win rate
- Diagnostics: exposure, entries, turnover, data quality, corporate actions, and 10 bp one-way cost sensitivity

The primary output is [`notebooks/phase1_backtesting.ipynb`](notebooks/phase1_backtesting.ipynb). A rendered HTML copy is generated beside it.

New to the tools? Start with [`PANDAS_MATPLOTLIB_TUTORIAL.md`](PANDAS_MATPLOTLIB_TUTORIAL.md), a short walkthrough using the cleaned snapshot.

## Key assumptions

- Signals use the adjusted close available at the end of session `t`.
- `position[t] = signal[t-1]`, so a same-day signal cannot earn the return that created it.
- Strategies are long or cash only, with fractional shares and no leverage.
- Base results use zero transaction costs, slippage, taxes, and cash yield; a 10 bp one-way sensitivity is also reported.
- Sharpe uses a 0% annual risk-free rate and 252 trading sessions per year.
- Win rate is positive-return active days divided by all active days; cash days are excluded.
- Maximum drawdown is reported as a positive loss magnitude.
- The fixed, ex-post universe has survivorship and selection bias. Results are descriptive, not evidence of deployable alpha.

## Recommended setup: Anaconda

```bash
conda env create -f environment.yml
conda activate phase1-backtesting
python run_backtest.py --refresh-data
pytest
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/phase1_backtesting.ipynb
python scripts/export_notebook_html.py
```

## Alternative setup: standard Python

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python run_backtest.py --refresh-data
pytest
```

Subsequent runs can omit `--refresh-data`; the pipeline will use the audited local snapshot and work offline.

## Google Colab handoff

After this folder is pushed to GitHub, open the notebook in Colab and run:

```python
!git clone YOUR_REPOSITORY_URL
%cd YOUR_REPOSITORY_FOLDER
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
```

The checked-in processed snapshot lets the notebook rerun without a live Yahoo request. Use `python run_backtest.py --refresh-data` only when you intentionally want a new snapshot.

## Suggested GitHub setup

```bash
git init
git add .
git commit -m "Add Phase 1 backtesting project"
# Then create an empty shared GitHub repository and add its URL:
git remote add origin YOUR_REPOSITORY_URL
git branch -M main
git push -u origin main
```

Raw Yahoo downloads are ignored by Git. The cleaned educational snapshot and its provenance manifest are organized under `data/processed/`. Review Yahoo's terms before redistributing market data.

## Project layout

```text
phase1_backtesting/
├── config/phase1.json
├── data/
│   ├── raw/                  # ignored live download
│   └── processed/            # cleaned snapshot + provenance
├── notebooks/                # executed notebook and HTML view
├── results/                  # tidy metrics, curves, and quality checks
├── figures/                  # publication-ready charts
├── src/phase1_backtest/      # reusable implementation
├── tests/                    # offline synthetic tests
├── run_backtest.py
├── environment.yml
└── requirements.txt
```

## Data notes

The downloader explicitly uses `auto_adjust=False` and `actions=True`. That preserves raw OHLC, a separate adjusted close, dividends, and splits. Adjusted close is used for signals and returns; dividends must not be added again. Missing observations are flagged rather than forward-filled.

Source: Yahoo Finance via `yfinance`. The configured `end` date is exclusive, so `2025-01-01` includes observations through 2024-12-31.
