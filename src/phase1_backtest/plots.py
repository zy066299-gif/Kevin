from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .strategies import STRATEGY_ORDER


COLORS = {
    "Buy & Hold": "#2457A6",
    "SMA 50/200": "#D19A00",
    "Momentum 12M (absolute)": "#D46A1F",
    "RSI 14": "#7A8F2A",
}

LINE_STYLES = {
    "Buy & Hold": "-",
    "SMA 50/200": "--",
    "Momentum 12M (absolute)": "-.",
    "RSI 14": ":",
}


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_return_heatmap(metrics: pd.DataFrame, path: Path, ticker_order: list[str]) -> None:
    pivot = (
        metrics.pivot(index="strategy", columns="ticker", values="annualized_return")
        .reindex(index=STRATEGY_ORDER, columns=ticker_order)
    )
    values = pivot.to_numpy(dtype=float)
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 0.01)
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    image = axis.imshow(values, cmap="RdYlBu", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(column, row, f"{value:.1%}", ha="center", va="center", fontsize=9, color="#1F2933")
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_title("Annualized return by strategy and ticker (2014–2024)", loc="left", weight="bold")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.ax.set_ylabel("Annualized return", rotation=270, labelpad=18)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_spy_equity(equity: pd.DataFrame, path: Path) -> None:
    frame = equity.loc[equity["ticker"] == "SPY"].copy()
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for strategy in STRATEGY_ORDER:
        subset = frame.loc[frame["strategy"] == strategy]
        axis.plot(
            subset["date"],
            subset["equity"],
            label=strategy,
            color=COLORS[strategy],
            linestyle=LINE_STYLES[strategy],
            linewidth=2,
        )
    _style_axes(axis)
    axis.set_title("SPY equity curves: lagged long/cash strategies", loc="left", weight="bold")
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Date")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_median_performance(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.set_index("strategy").reindex(STRATEGY_ORDER).reset_index()
    x = np.arange(len(ordered))
    colors = [COLORS[strategy] for strategy in ordered["strategy"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(x, ordered["median_annualized_return"], color=colors)
    axes[0].set_xticks(x, ordered["strategy"], rotation=20, ha="right")
    axes[0].set_title("Median annualized return", loc="left", weight="bold")
    axes[0].set_ylabel("Annualized return")
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _style_axes(axes[0])
    axes[1].bar(x, ordered["median_max_drawdown"], color=colors)
    axes[1].set_xticks(x, ordered["strategy"], rotation=20, ha="right")
    axes[1].set_title("Median maximum drawdown", loc="left", weight="bold")
    axes[1].set_ylabel("Loss from prior peak")
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _style_axes(axes[1])
    figure.suptitle("Cross-universe strategy comparison (nine instruments)", x=0.01, ha="left", weight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
