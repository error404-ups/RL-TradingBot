"""
Data Pipeline — EUR/USD Hourly (10 Years)
==========================================
Downloads via yfinance, cleans, splits into train/val/test.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_eurusd(period: str = "10y", interval: str = "1h", force: bool = False) -> pd.DataFrame:
    """
    Download EUR/USD hourly data for the past 10 years.
    Note: yfinance limits hourly data to ~2 years per call,
    so we stitch multiple requests together.
    """
    cache_path = os.path.join(DATA_DIR, "eurusd_1h_10y.parquet")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path) and not force:
        print(f"[Data] Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"[Data] Loaded {len(df):,} rows | {df.index[0]} → {df.index[-1]}")
        return df

    print("[Data] Downloading EUR/USD hourly data (stitching 5×2y chunks)...")

    # yfinance hourly max window is ~730 days; stitch 5 calls
    periods = [
        ("2014-01-01", "2016-01-01"),
        ("2016-01-01", "2018-01-01"),
        ("2018-01-01", "2020-01-01"),
        ("2020-01-01", "2022-01-01"),
        ("2022-01-01", "2024-12-31"),
    ]

    frames = []
    for start, end in periods:
        chunk = yf.download(
            tickers="EURUSD=X",
            start=start,
            end=end,
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if not chunk.empty:
            frames.append(chunk)
            print(f"  ✓ {start} → {end}: {len(chunk):,} rows")

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep OHLCV only
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    df.to_parquet(cache_path)
    print(f"[Data] Saved {len(df):,} rows → {cache_path}")
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split."""
    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:i_train].reset_index(drop=True)
    val   = df.iloc[i_train:i_val].reset_index(drop=True)
    test  = df.iloc[i_val:].reset_index(drop=True)

    print(f"[Data] Split → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


if __name__ == "__main__":
    df = download_eurusd()
    train, val, test = split_data(df)
    print(df.tail())