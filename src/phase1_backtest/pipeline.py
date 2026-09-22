from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import canonical_sha256, load_or_download_snapshot
from .engine import backtest_one_ticker
from .metrics import performance_metrics
from .plots import plot_median_performance, plot_return_heatmap, plot_spy_equity
from .strategies import STRATEGY_ORDER


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics.groupby("strategy", sort=False)
        .agg(
            median_total_return=("total_return", "median"),
            median_annualized_return=("annualized_return", "median"),
            median_annualized_volatility=("annualized_volatility", "median"),
            median_sharpe_ratio=("sharpe_ratio", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            median_win_rate=("win_rate", "median"),
            median_exposure=("exposure", "median"),
        )
        .reindex(STRATEGY_ORDER)
        .reset_index()
    )
    benchmark = metrics.loc[metrics["strategy"] == "Buy & Hold", ["ticker", "annualized_return"]].rename(
        columns={"annualized_return": "benchmark_annualized_return"}
    )
    comparison = metrics.merge(benchmark, on="ticker", how="left")
    beat_counts = (
        comparison.assign(beat_buy_hold=comparison["annualized_return"] > comparison["benchmark_annualized_return"])
        .groupby("strategy")["beat_buy_hold"]
        .agg(["sum", "count"])
    )
    summary["tickers_beating_buy_hold"] = summary["strategy"].map(beat_counts["sum"]).astype(int)
    summary["ticker_count"] = summary["strategy"].map(beat_counts["count"]).astype(int)
    return summary


def _write_report(
    project_root: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    quality: pd.DataFrame,
    source_mode: str,
) -> None:
    best_combo = metrics.sort_values("annualized_return", ascending=False).iloc[0]
    best_median = summary.sort_values("median_annualized_return", ascending=False).iloc[0]
    review_tickers = quality.loc[quality["status"] != "PASS", "ticker"].tolist()

    def pct(value: float) -> str:
        return "n/a" if pd.isna(value) else f"{value:.1%}"

    lines = [
        "# Phase 1 Backtest Results",
        "",
        "## TL;DR",
        "",
        f"- The highest annualized return among the 36 strategy/ticker tests was **{best_combo['strategy']} on {best_combo['ticker']}** at **{pct(best_combo['annualized_return'])}**.",
        f"- The highest median annualized return across the nine-instrument universe was **{best_median['strategy']}** at **{pct(best_median['median_annualized_return'])}**.",
        f"- Data were loaded from the local **{source_mode}** path. Quality status required review for: **{', '.join(review_tickers) if review_tickers else 'none'}**.",
        "- These are frictionless, in-sample, long/cash results on an ex-post survivor universe. Treat them as a learning exercise, not an investment recommendation.",
        "",
        "## Strategy-level medians",
        "",
        "| Strategy | Ann. return | Ann. vol. | Sharpe | Max drawdown | Win rate | Exposure | Beat B&H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        beat_label = "benchmark" if row.strategy == "Buy & Hold" else f"{row.tickers_beating_buy_hold}/{row.ticker_count}"
        lines.append(
            f"| {row.strategy} | {pct(row.median_annualized_return)} | {pct(row.median_annualized_volatility)} | "
            f"{row.median_sharpe_ratio:.2f} | {pct(row.median_max_drawdown)} | {pct(row.median_win_rate)} | "
            f"{pct(row.median_exposure)} | {beat_label} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- All signals are lagged one session before returns are applied.",
            "- Adjusted close is used for indicators and returns; dividend cash flows are not added again.",
            "- Buy-and-hold, SMA, momentum, and RSI are evaluated over identical dates for each ticker.",
            "- Cash earns 0%; this understates long/cash strategies during high-rate periods.",
            "- Costs, slippage, taxes, next-open gaps, shorting, and capacity are omitted from headline metrics.",
            "- Nine assets and four approaches create multiple-comparison risk; isolated winners may be luck.",
            "- The selected stocks all survived the full period, so the stock universe is survivorship-biased.",
            "",
            "See `results/performance_metrics.csv`, `results/strategy_summary.csv`, and the executed notebook for the auditable detail.",
            "",
        ]
    )
    (project_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(project_root: str | Path, refresh_data: bool = False) -> dict[str, object]:
    root = Path(project_root).resolve()
    config = load_config(root / "config" / "phase1.json")
    for directory in ["results", "figures", "data/processed", "data/raw"]:
        (root / directory).mkdir(parents=True, exist_ok=True)

    cleaned, quality, source_metadata, source_mode = load_or_download_snapshot(root, config, refresh=refresh_data)
    base_details: list[pd.DataFrame] = []
    base_indicators: list[pd.DataFrame] = []
    base_metrics: list[dict] = []
    cost_metrics: list[dict] = []

    for ticker in config.tickers:
        ticker_frame = cleaned.loc[cleaned["ticker"] == ticker].copy()
        if ticker_frame.empty:
            raise ValueError(f"No cleaned rows for {ticker}")

        details, indicators = backtest_one_ticker(ticker_frame, config, transaction_cost_bps=0.0)
        details["ticker"] = ticker
        indicators["ticker"] = ticker
        base_details.append(details)
        base_indicators.append(indicators)
        for strategy in STRATEGY_ORDER:
            subset = details.loc[details["strategy"] == strategy]
            row = performance_metrics(subset, config.trading_days_per_year, config.risk_free_rate)
            row.update({"ticker": ticker, "strategy": strategy, "transaction_cost_bps": 0.0})
            base_metrics.append(row)

        cost_details, _ = backtest_one_ticker(
            ticker_frame,
            config,
            transaction_cost_bps=config.cost_sensitivity_bps,
        )
        for strategy in STRATEGY_ORDER:
            subset = cost_details.loc[cost_details["strategy"] == strategy]
            row = performance_metrics(subset, config.trading_days_per_year, config.risk_free_rate)
            row.update(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "transaction_cost_bps": config.cost_sensitivity_bps,
                }
            )
            cost_metrics.append(row)

    details_frame = pd.concat(base_details, ignore_index=True)
    indicators_frame = pd.concat(base_indicators, ignore_index=True)
    metrics_frame = pd.DataFrame(base_metrics)
    metrics_frame["strategy"] = pd.Categorical(metrics_frame["strategy"], STRATEGY_ORDER, ordered=True)
    metrics_frame = metrics_frame.sort_values(["ticker", "strategy"]).reset_index(drop=True)
    metrics_frame["strategy"] = metrics_frame["strategy"].astype(str)
    cost_frame = pd.DataFrame(cost_metrics)
    summary_frame = _summarize(metrics_frame)

    results_dir = root / "results"
    details_frame.to_csv(results_dir / "daily_backtest.csv.gz", index=False, compression="gzip")
    indicators_frame.to_csv(results_dir / "indicators.csv.gz", index=False, compression="gzip")
    metrics_frame.to_csv(results_dir / "performance_metrics.csv", index=False)
    cost_label = f"{config.cost_sensitivity_bps:g}bps".replace(".", "p")
    cost_frame.to_csv(results_dir / f"cost_sensitivity_{cost_label}.csv", index=False)
    summary_frame.to_csv(results_dir / "strategy_summary.csv", index=False)
    quality.to_csv(results_dir / "data_quality.csv", index=False)

    figures_dir = root / "figures"
    plot_return_heatmap(metrics_frame, figures_dir / "annualized_return_heatmap.png", list(config.tickers))
    plot_spy_equity(details_frame, figures_dir / "spy_equity_curves.png")
    plot_median_performance(summary_frame, figures_dir / "median_performance.png")

    cost_comparison = metrics_frame[["ticker", "strategy", "annualized_return"]].merge(
        cost_frame[["ticker", "strategy", "annualized_return"]],
        on=["ticker", "strategy"],
        suffixes=("_base", "_with_cost"),
    )
    cost_comparison["annualized_return_drag"] = (
        cost_comparison["annualized_return_base"] - cost_comparison["annualized_return_with_cost"]
    )
    cost_comparison.to_csv(results_dir / "cost_sensitivity_comparison.csv", index=False)

    run_manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": source_mode,
        "config": json.loads((root / "config" / "phase1.json").read_text(encoding="utf-8")),
        "data_sha256": canonical_sha256(cleaned),
        "metrics_sha256": canonical_sha256(metrics_frame),
        "git_commit": _git_commit(root),
        "source_metadata": source_metadata,
        "validation": {
            "same_evaluation_dates_per_ticker": bool(
                details_frame.groupby(["ticker", "strategy"])["date"].agg(["min", "max", "count"])
                .groupby(level=0)
                .nunique()
                .eq(1)
                .all()
                .all()
            ),
            "finite_reported_metrics": bool(
                np.isfinite(
                    metrics_frame[
                        [
                            "total_return",
                            "annualized_return",
                            "annualized_volatility",
                            "max_drawdown",
                            "win_rate",
                            "exposure",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            ),
            "positions_binary": bool(details_frame["position"].isin([0.0, 1.0]).all()),
        },
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    _write_report(root, metrics_frame, summary_frame, quality, source_mode)

    return {
        "config": config,
        "cleaned": cleaned,
        "quality": quality,
        "details": details_frame,
        "indicators": indicators_frame,
        "metrics": metrics_frame,
        "summary": summary_frame,
        "cost_comparison": cost_comparison,
        "manifest": run_manifest,
        "source_mode": source_mode,
    }
