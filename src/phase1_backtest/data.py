from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BacktestConfig


CANONICAL_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
]


def _canonical_field(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    aliases = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "stock_splits": "stock_splits",
        "stocksplits": "stock_splits",
    }
    return aliases.get(text, text)


def _normalize_download(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Normalize yfinance's version-dependent wide output to one row/date/ticker."""
    if raw.empty:
        raise ValueError("yfinance returned no rows")

    requested = tuple(tickers)
    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(map(str, raw.columns.get_level_values(0)))
        level_one = set(map(str, raw.columns.get_level_values(1)))
        for ticker in requested:
            if ticker in level_zero:
                frame = raw.xs(ticker, axis=1, level=0, drop_level=True).copy()
            elif ticker in level_one:
                frame = raw.xs(ticker, axis=1, level=1, drop_level=True).copy()
            else:
                raise ValueError(f"Ticker {ticker} is absent from the yfinance response")
            frame.columns = [_canonical_field(column) for column in frame.columns]
            frame["ticker"] = ticker
            frames.append(frame.reset_index())
    elif len(requested) == 1:
        frame = raw.copy()
        frame.columns = [_canonical_field(column) for column in frame.columns]
        frame["ticker"] = requested[0]
        frames.append(frame.reset_index())
    else:
        raise ValueError("Expected MultiIndex columns for a multi-ticker response")

    tidy = pd.concat(frames, ignore_index=True)
    date_column = next(
        (column for column in tidy.columns if str(column).lower() in {"date", "datetime"}),
        None,
    )
    if date_column is None:
        raise ValueError("The yfinance response has no date column")
    tidy = tidy.rename(columns={date_column: "date"})
    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce", utc=True).dt.tz_localize(None)

    for column in CANONICAL_COLUMNS:
        if column not in tidy:
            tidy[column] = 0.0 if column in {"dividends", "stock_splits"} else np.nan

    return tidy[CANONICAL_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_market_data(config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    options = {
        "start": config.download_start,
        "end": config.end_exclusive,
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "repair": False,
        "keepna": True,
        "group_by": "ticker",
        "multi_level_index": True,
        "progress": False,
        "threads": True,
        "timeout": 30,
    }
    raw = yf.download(tickers=list(config.tickers), **options)
    tidy = _normalize_download(raw, config.tickers)
    metadata = {
        "schema_version": 1,
        "provider": "Yahoo Finance via yfinance",
        "source_url": "https://finance.yahoo.com/",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "yfinance_version": yf.__version__,
        "tickers": list(config.tickers),
        "request": options,
        "end_date_semantics": "exclusive",
    }
    return raw, tidy, metadata


def clean_and_profile(tidy: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = tidy.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    numeric_columns = [column for column in CANONICAL_COLUMNS if column not in {"date", "ticker"}]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    duplicate_mask = data.duplicated(["ticker", "date"], keep=False)
    duplicate_counts = duplicate_mask.groupby(data["ticker"]).sum().to_dict()
    data = data.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    invalid_adj = data["adj_close"].isna() | ~np.isfinite(data["adj_close"]) | (data["adj_close"] <= 0)
    invalid_by_ticker = invalid_adj.groupby(data["ticker"]).sum().to_dict()
    cleaned = data.loc[~invalid_adj].copy()
    cleaned["volume"] = cleaned["volume"].fillna(0.0)
    cleaned[["dividends", "stock_splits"]] = cleaned[["dividends", "stock_splits"]].fillna(0.0)
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)

    union_dates = set(cleaned["date"].dropna().unique())
    profiles: list[dict] = []
    for ticker in config.tickers:
        original = data.loc[data["ticker"] == ticker].copy()
        frame = cleaned.loc[cleaned["ticker"] == ticker].copy()
        ticker_dates = set(frame["date"].dropna().unique())
        high_floor = frame[["open", "close", "low"]].max(axis=1, skipna=True)
        low_ceiling = frame[["open", "close", "high"]].min(axis=1, skipna=True)
        ohlc_violations = int(((frame["high"] < high_floor) | (frame["low"] > low_ceiling)).fillna(False).sum())
        adjusted_returns = frame["adj_close"].pct_change(fill_method=None)
        raw_returns = frame["close"].pct_change(fill_method=None)
        return_adjustment_gap = (adjusted_returns - raw_returns).abs()
        adjustment_factor = frame["adj_close"] / frame["close"]
        adjusted_price_differs = (adjustment_factor - 1.0).abs().gt(1e-10)
        missing_from_union = len(union_dates - ticker_dates)
        negative_volume_rows = int((frame["volume"] < 0).sum())
        critical_count = (
            int(duplicate_counts.get(ticker, 0))
            + int(invalid_by_ticker.get(ticker, 0))
            + ohlc_violations
            + missing_from_union
            + negative_volume_rows
        )
        profiles.append(
            {
                "ticker": ticker,
                "rows_downloaded": int(len(original)),
                "rows_clean": int(len(frame)),
                "first_date": frame["date"].min().date().isoformat() if len(frame) else None,
                "last_date": frame["date"].max().date().isoformat() if len(frame) else None,
                "duplicate_key_rows": int(duplicate_counts.get(ticker, 0)),
                "invalid_or_missing_adj_close_rows": int(invalid_by_ticker.get(ticker, 0)),
                "missing_dates_vs_universe_calendar": int(missing_from_union),
                "ohlc_rule_violations": ohlc_violations,
                "negative_volume_rows": negative_volume_rows,
                "dividend_events": int((frame["dividends"] != 0).sum()),
                "split_events": int((frame["stock_splits"] != 0).sum()),
                "rows_with_adjusted_vs_raw_price_difference": int(adjusted_price_differs.fillna(False).sum()),
                "largest_abs_adjusted_vs_raw_return_gap": float(return_adjustment_gap.max()),
                "largest_abs_adjusted_return": float(adjusted_returns.abs().max()),
                "largest_abs_raw_close_return": float(raw_returns.abs().max()),
                "status": "PASS" if critical_count == 0 else "REVIEW",
            }
        )

    profile = pd.DataFrame(profiles)
    return cleaned, profile


def canonical_sha256(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    if "date" in canonical:
        canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_gzip_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    os.replace(temporary, path)


def save_snapshot(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    quality: pd.DataFrame,
    metadata: dict,
    project_root: Path,
) -> dict[str, Path]:
    raw_path = project_root / "data" / "raw" / "yfinance_download.csv.gz"
    clean_csv = project_root / "data" / "processed" / "clean_prices.csv.gz"
    clean_parquet = project_root / "data" / "processed" / "clean_prices.parquet"
    quality_path = project_root / "results" / "data_quality.csv"
    manifest_path = project_root / "data" / "processed" / "manifest.json"
    for parent in {raw_path.parent, clean_csv.parent, quality_path.parent}:
        parent.mkdir(parents=True, exist_ok=True)

    raw.to_csv(raw_path, compression="gzip")
    _write_gzip_csv_atomic(cleaned, clean_csv)
    cleaned.to_parquet(clean_parquet, index=False)
    quality.to_csv(quality_path, index=False)

    metadata = dict(metadata)
    metadata["clean_data_sha256"] = canonical_sha256(cleaned)
    metadata["clean_rows"] = int(len(cleaned))
    metadata["actual_first_date"] = cleaned["date"].min().date().isoformat()
    metadata["actual_last_date"] = cleaned["date"].max().date().isoformat()
    metadata["rows_by_ticker"] = cleaned.groupby("ticker").size().astype(int).to_dict()
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return {
        "clean_csv": clean_csv,
        "clean_parquet": clean_parquet,
        "quality": quality_path,
        "manifest": manifest_path,
    }


def load_or_download_snapshot(
    project_root: Path,
    config: BacktestConfig,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    clean_parquet = project_root / "data" / "processed" / "clean_prices.parquet"
    quality_path = project_root / "results" / "data_quality.csv"
    manifest_path = project_root / "data" / "processed" / "manifest.json"

    if not refresh and clean_parquet.exists() and quality_path.exists() and manifest_path.exists():
        cleaned = pd.read_parquet(clean_parquet)
        cleaned["date"] = pd.to_datetime(cleaned["date"])
        quality = pd.read_csv(quality_path)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_hash = canonical_sha256(cleaned)
        if observed_hash != metadata.get("clean_data_sha256"):
            raise ValueError("Cached clean data failed its SHA-256 check; refresh the snapshot")
        if tuple(metadata.get("tickers", [])) != config.tickers:
            raise ValueError("Cached ticker universe does not match config; refresh the snapshot")
        cached_request = metadata.get("request", {})
        if cached_request.get("start") != config.download_start or cached_request.get("end") != config.end_exclusive:
            raise ValueError("Cached date range does not match config; refresh the snapshot")
        return cleaned, quality, metadata, "cache"

    raw, tidy, metadata = download_market_data(config)
    cleaned, quality = clean_and_profile(tidy, config)
    save_snapshot(raw, cleaned, quality, metadata, project_root)
    return cleaned, quality, metadata, "download"
