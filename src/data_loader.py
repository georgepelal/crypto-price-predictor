"""Data loader module for fetching and caching cryptocurrency OHLCV and Derivatives data."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests


def get_safe_symbol_filename(symbol: str, timeframe: str, prefix: str = "") -> str:
    """Format symbol for safe file storage, e.g. BTC_USDT_1h.csv"""
    clean_sym = symbol.replace("/", "_").replace(":", "_")
    tag = f"{prefix}_" if prefix else ""
    return f"{tag}{clean_sym}_{timeframe}.csv"


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    total_bars: int = 2000,
    exchange_name: str = "binance",
    cache_dir: str = "data/raw",
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV candles from CCXT with pagination and caching."""
    os.makedirs(cache_dir, exist_ok=True)
    file_name = get_safe_symbol_filename(symbol, timeframe)
    file_path = Path(cache_dir) / file_name

    # Check cache if not forced
    if not force_download and file_path.exists():
        print(f"[*] Loading cached OHLCV data from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if len(df) >= total_bars:
            return df.tail(total_bars)

    print(f"[*] Fetching {total_bars} bars for {symbol} ({timeframe}) from {exchange_name}...")

    exchange_class = getattr(ccxt, exchange_name, None)
    if exchange_class is None:
        raise ValueError(f"Exchange '{exchange_name}' is not supported by ccxt.")

    exchange = exchange_class({"enableRateLimit": True, "timeout": 30000})

    tf_seconds = exchange.parse_timeframe(timeframe)
    now_ms = exchange.milliseconds()
    approx_span_ms = total_bars * tf_seconds * 1000
    since_ms = now_ms - approx_span_ms

    all_candles = []
    limit_per_call = 1000

    try:
        while len(all_candles) < total_bars:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit_per_call)
            if not batch:
                break

            all_candles.extend(batch)
            last_ts = batch[-1][0]

            if last_ts == since_ms or last_ts >= now_ms - (tf_seconds * 1000):
                break

            since_ms = last_ts + (tf_seconds * 1000)
            time.sleep(exchange.rateLimit / 1000.0)

    except Exception as e:
        print(f"[!] Warning: CCXT fetch encountered an error: {e}")
        if file_path.exists():
            print(f"[*] Falling back to cached data at {file_path}")
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            print("[!] No cached data found. Generating synthetic OHLCV data for offline demonstration...")
            return generate_synthetic_crypto_data(symbol, timeframe, bars=total_bars)

    if not all_candles:
        if file_path.exists():
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        print("[!] No candles returned. Generating synthetic data...")
        return generate_synthetic_crypto_data(symbol, timeframe, bars=total_bars)

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.to_csv(file_path)
    print(f"[+] Download complete: {len(df)} bars saved to {file_path}")
    return df.tail(total_bars)


def fetch_derivatives_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_ms: Optional[int] = None,
    cache_dir: str = "data/raw",
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch historical Funding Rates and Open Interest from Binance Perpetual Futures.

    Args:
        symbol: Spot or futures symbol, e.g. 'BTC/USDT'
        timeframe: Candle interval ('1h', '15m')
        since_ms: Unix timestamp start in milliseconds
        cache_dir: Directory for storing raw data
        force_download: Bypass cache if True

    Returns:
        pd.DataFrame with 'funding_rate' and 'oi_val' indexed by UTC timestamp.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_name = get_safe_symbol_filename(symbol, timeframe, prefix="derivatives")
    file_path = Path(cache_dir) / file_name

    if not force_download and file_path.exists():
        print(f"[*] Loading cached derivatives data from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    base_coin = symbol.split("/")[0]
    swap_symbol = f"{base_coin}/USDT:USDT" if ":" not in symbol else symbol

    print(f"[*] Fetching derivatives data (Funding Rate & Open Interest) for {swap_symbol}...")

    try:
        ex = ccxt.binance({"options": {"defaultType": "swap"}, "enableRateLimit": True, "timeout": 30000})
        now_ms = ex.milliseconds()

        # Binance openInterestHist supports maximum 30 days back (~720 hours)
        max_oi_lookback_ms = 28 * 24 * 3600 * 1000
        oi_start_ms = max(since_ms, now_ms - max_oi_lookback_ms) if since_ms else (now_ms - max_oi_lookback_ms)

        # 1. Fetch Open Interest History
        oi_records = []
        try:
            curr_since = oi_start_ms
            for _ in range(4):
                batch = ex.fetch_open_interest_history(swap_symbol, timeframe=timeframe, since=curr_since, limit=500)
                if not batch:
                    break
                oi_records.extend(batch)
                last_ts = batch[-1]["timestamp"]
                if last_ts <= curr_since or last_ts >= now_ms - (3600 * 1000):
                    break
                curr_since = last_ts + (3600 * 1000)
                time.sleep(ex.rateLimit / 1000.0)
        except Exception as e_oi:
            print(f"[!] Warning: Could not fetch OI history ({e_oi}).")

        # 2. Fetch Funding Rate History (supports up to 120+ days)
        fr_start_ms = since_ms or (now_ms - (90 * 24 * 3600 * 1000))
        fr_records = []
        try:
            fr_batch = ex.fetch_funding_rate_history(swap_symbol, since=fr_start_ms, limit=500)
            if fr_batch:
                fr_records.extend(fr_batch)
        except Exception as e_fr:
            print(f"[!] Warning: Could not fetch Funding Rate history ({e_fr}).")

        # Process OI
        if oi_records:
            oi_df = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime(item["timestamp"], unit="ms", utc=True),
                    "oi_val": float(item.get("openInterestValue") or item.get("info", {}).get("sumOpenInterestValue", 0.0)),
                }
                for item in oi_records
            ]).drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        else:
            oi_df = pd.DataFrame(columns=["oi_val"])

        # Process FR
        if fr_records:
            fr_df = pd.DataFrame([
                {
                    "timestamp": pd.to_datetime(item["timestamp"], unit="ms", utc=True),
                    "funding_rate": float(item["fundingRate"]),
                }
                for item in fr_records
            ]).drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        else:
            fr_df = pd.DataFrame(columns=["funding_rate"])

        # Merge derivatives
        if not oi_df.empty and not fr_df.empty:
            deriv_df = oi_df.join(fr_df, how="outer").sort_index()
        elif not oi_df.empty:
            deriv_df = oi_df
            deriv_df["funding_rate"] = 0.0001  # baseline neutral 0.01%
        elif not fr_df.empty:
            deriv_df = fr_df
            deriv_df["oi_val"] = 8e9
        else:
            print("[!] Derivatives API empty. Generating synthetic derivatives...")
            return generate_synthetic_derivatives_data(timeframe=timeframe, bars=700)

        deriv_df["funding_rate"] = deriv_df["funding_rate"].ffill().bfill().fillna(0.0001)
        deriv_df["oi_val"] = deriv_df["oi_val"].ffill().bfill().fillna(8e9)

        deriv_df.to_csv(file_path)
        print(f"[+] Saved derivatives data to {file_path} ({len(deriv_df)} rows)")
        return deriv_df

    except Exception as e:
        print(f"[!] Derivatives fetch error: {e}. Generating synthetic fallback...")
        return generate_synthetic_derivatives_data(timeframe=timeframe, bars=700)


def generate_synthetic_derivatives_data(timeframe: str = "1h", bars: int = 700) -> pd.DataFrame:
    """Generate synthetic funding rates and open interest for offline testing."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    freq = "1h" if timeframe == "1h" else "15min"
    dates = pd.date_range(end=now, periods=bars, freq=freq, tz=timezone.utc)

    # Mean-reverting funding rate around +0.0001 (0.01%)
    fr = np.random.normal(0.0001, 0.0002, bars)
    # Open interest around $8B with random walk
    oi_walk = np.cumsum(np.random.normal(0, 5e7, bars))
    oi_val = np.maximum(5e9, 8e9 + oi_walk)

    df = pd.DataFrame({"funding_rate": fr, "oi_val": oi_val}, index=dates)
    df.index.name = "timestamp"
    return df


def generate_synthetic_crypto_data(
    symbol: str = "BTC/USDT", timeframe: str = "1h", bars: int = 2000
) -> pd.DataFrame:
    """Generate realistic synthetic crypto OHLCV data using Geometric Brownian Motion with jumps."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    freq = freq_map.get(timeframe, "1h")
    dates = pd.date_range(end=now, periods=bars, freq=freq, tz=timezone.utc)

    mu = 0.0001
    sigma = 0.008
    returns = np.random.normal(mu, sigma, bars)
    jumps = np.random.choice([0, 1], size=bars, p=[0.98, 0.02]) * np.random.normal(0, 0.03, bars)
    total_returns = returns + jumps

    price_0 = 65000.0 if "BTC" in symbol else 3500.0
    close_prices = price_0 * np.exp(np.cumsum(total_returns))

    high_noise = np.random.uniform(0.001, 0.006, bars)
    low_noise = np.random.uniform(0.001, 0.006, bars)

    high_prices = close_prices * (1 + high_noise)
    low_prices = close_prices * (1 - low_noise)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = price_0

    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    base_volume = 1200.0 if "BTC" in symbol else 15000.0
    volume = base_volume * np.exp(np.random.normal(0, 0.5, bars)) * (1 + np.abs(total_returns) * 20)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def fetch_fear_and_greed(
    limit: int = 730,
    cache_dir: str = "data/raw",
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch historical Crypto Fear & Greed Index from Alternative.me."""
    os.makedirs(cache_dir, exist_ok=True)
    file_path = Path(cache_dir) / "fear_and_greed.csv"

    if not force_download and file_path.exists():
        print(f"[*] Loading cached Fear & Greed data from: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    print(f"[*] Fetching Crypto Fear & Greed Index (last {limit} days)...")
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json().get("data", [])
        if not payload:
            raise ValueError("Empty response from Fear & Greed API")

        df = pd.DataFrame(payload)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["fng_value"] = df["value"].astype(float)
        df = df[["timestamp", "fng_value"]].drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()

        df.to_csv(file_path)
        print(f"[+] Saved Fear & Greed data to {file_path} ({len(df)} days)")
        return df
    except Exception as e:
        print(f"[!] Warning: Fear & Greed API error ({e}). Generating synthetic fallback...")
        return generate_synthetic_fear_and_greed(days=limit)


def generate_synthetic_fear_and_greed(days: int = 730) -> pd.DataFrame:
    """Generate synthetic Fear & Greed index values for offline testing."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    dates = pd.date_range(end=now, periods=days, freq="1D", tz=timezone.utc)
    # Mean-reverting random walk between 10 and 90
    values = np.clip(50.0 + np.cumsum(np.random.normal(0, 3, days)), 10.0, 90.0)
    df = pd.DataFrame({"fng_value": values}, index=dates)
    df.index.name = "timestamp"
    return df


def fetch_multitimeframe_ohlcv(
    symbol: str = "BTC/USDT",
    primary_timeframe: str = "1h",
    primary_bars: int = 700,
    exchange_name: str = "binance",
    cache_dir: str = "data/raw",
    force_download: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch primary OHLCV alongside higher timeframes (4h and 1d) for multi-scale analysis.

    Returns:
        Tuple of (df_primary, df_4h, df_1d).
    """
    df_primary = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe=primary_timeframe,
        total_bars=primary_bars,
        exchange_name=exchange_name,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    bars_4h_needed = max(300, primary_bars // 3)
    df_4h = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="4h",
        total_bars=bars_4h_needed,
        exchange_name=exchange_name,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    bars_1d_needed = max(150, primary_bars // 12)
    df_1d = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe="1d",
        total_bars=bars_1d_needed,
        exchange_name=exchange_name,
        cache_dir=cache_dir,
        force_download=force_download,
    )

    return df_primary, df_4h, df_1d


