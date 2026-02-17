"""
Data Pipeline — EUR/USD Hybrid Dataset (10 Years)
==================================================
Strategy:
  - yfinance hourly data is limited to the last 730 days max.
  - For older history we download DAILY data (10y) and upsample
    it to synthetic hourly bars so the environment has enough data.
  - The two datasets are concatenated: daily-upsampled (older) +
    real hourly (recent 2y), giving ~10 years of hourly-resolution data.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by newer yfinance versions."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _keep_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].copy()


def _upsample_daily_to_hourly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily OHLCV bars to synthetic hourly bars (24 per day).
    Each day is split into 24 hours with minor Gaussian noise so the
    environment sees realistic intra-day variation without look-ahead.
    """
    rows = []
    rng  = np.random.default_rng(42)

    for _, row in df_daily.iterrows():
        date  = row.name if hasattr(row.name, "date") else pd.Timestamp(row.name)
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        vol   = row.get("Volume", 0) / 24

        # Build intra-day price path: open → random walk capped to [L, H] → close
        prices = np.linspace(o, c, 24) + rng.normal(0, (h - l) * 0.05, 24)
        prices = np.clip(prices, l, h)
        prices[0]  = o
        prices[-1] = c

        for hour in range(24):
            ts = pd.Timestamp(date).replace(hour=hour, minute=0, second=0, microsecond=0)
            p  = prices[hour]
            noise = abs(rng.normal(0, (h - l) * 0.02))
            rows.append({
                "Open":   p,
                "High":   p + noise,
                "Low":    max(l, p - noise),
                "Close":  p,
                "Volume": vol,
            })

    result = pd.DataFrame(rows)
    result.index = pd.DatetimeIndex([
        pd.Timestamp(row.name).replace(hour=h, minute=0, second=0, microsecond=0)
        for _, row in df_daily.iterrows()
        for h in range(24)
    ])
    return result


# -----------------------------------------------------------------------
# Main download function
# -----------------------------------------------------------------------

def download_eurusd(force: bool = False) -> pd.DataFrame:
    """
    Build a ~10-year EUR/USD hourly dataset:
      1. Try to get real hourly data for the last 2 years (yfinance limit).
      2. Download daily data for the full 10-year window.
      3. Upsample daily → synthetic hourly for the period NOT covered by real hourly.
      4. Concatenate and cache.
    """
    cache_path = os.path.join(DATA_DIR, "eurusd_1h_10y.parquet")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path) and not force:
        print(f"[Data] Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"[Data] Loaded {len(df):,} rows | {df.index[0]} → {df.index[-1]}")
        return df

    today     = datetime.today()
    start_10y = (today - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    start_2y  = (today - timedelta(days=729)).strftime("%Y-%m-%d")   # safe margin

    # ── 1. Real hourly data (last ~2 years) ──────────────────────────
    print("[Data] Downloading real hourly data (last 2 years)...")
    hourly = yf.download(
        tickers="EURUSD=X",
        start=start_2y,
        interval="1h",
        progress=False,
        auto_adjust=True,
    )
    hourly = _flatten_columns(hourly)
    hourly = _keep_ohlcv(hourly)
    if hourly.index.tz is not None:
        hourly.index = hourly.index.tz_localize(None)
    hourly.dropna(inplace=True)
    print(f"  ✓ Real hourly: {len(hourly):,} rows | {hourly.index[0]} → {hourly.index[-1]}")

    # ── 2. Daily data (full 10 years) ────────────────────────────────
    print("[Data] Downloading daily data (10 years)...")
    daily = yf.download(
        tickers="EURUSD=X",
        start=start_10y,
        interval="1d",
        progress=False,
        auto_adjust=True,
    )
    daily = _flatten_columns(daily)
    daily = _keep_ohlcv(daily)
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    daily.dropna(inplace=True)
    print(f"  ✓ Daily: {len(daily):,} rows | {daily.index[0]} → {daily.index[-1]}")

    # ── 3. Upsample old daily portion to synthetic hourly ────────────
    cutoff    = hourly.index[0].normalize()          # midnight of first real hourly bar
    daily_old = daily[daily.index < cutoff]
    print(f"[Data] Upsampling {len(daily_old):,} daily bars → synthetic hourly...")
    synthetic = _upsample_daily_to_hourly(daily_old)
    print(f"  ✓ Synthetic hourly: {len(synthetic):,} rows")

    # ── 4. Concatenate ───────────────────────────────────────────────
    df = pd.concat([synthetic, hourly])
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    df.to_parquet(cache_path)
    print(f"[Data] Total dataset: {len(df):,} hourly rows | {df.index[0]} → {df.index[-1]}")
    print(f"[Data] Saved → {cache_path}")
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