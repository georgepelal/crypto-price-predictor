"""Historical market data with an append-only local cache.

Sources (all public, no API key):
  * Binance spot klines  (api.binance.com /api/v3/klines)  incl. taker-buy volume
  * Binance USDT-M perpetual funding rates (fapi.binance.com /fapi/v1/fundingRate)
  * Crypto Fear & Greed index (api.alternative.me)

Rules:
  * Caches live in data/history/ and are only ever *extended*: new rows are merged in,
    existing rows are never thrown away. Live trading and research share the same cache.
  * The still-open (incomplete) candle is never stored.
  * Network failures raise. There is NO synthetic-data fallback here.
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

HISTORY_DIR = Path("data/history")
SPOT_URL = "https://api.binance.com/api/v3/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FNG_URL = "https://api.alternative.me/fng/"

TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OHLCV_KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]


class DataFetchError(RuntimeError):
    pass


def _binance_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT'."""
    return symbol.replace("/", "").replace(":USDT", "").upper()


def _get(url: str, params: dict, retries: int = 4) -> list:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429 or r.status_code >= 500:
                raise DataFetchError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise DataFetchError(f"GET {url} {params} failed after {retries} attempts: {last_err}")


def _cache_path(name: str, cache_dir: Optional[Path] = None) -> Path:
    d = Path(cache_dir) if cache_dir is not None else HISTORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp)
    os.replace(tmp, path)


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        out = new
    elif new.empty:
        out = old
    else:
        out = pd.concat([old, new])
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# --------------------------------------------------------------------------- OHLCV

def _fetch_klines(symbol: str, timeframe: str, start_ms: int, end_ms: Optional[int] = None) -> pd.DataFrame:
    step = TF_MS[timeframe]
    now_ms = int(time.time() * 1000)
    end_ms = end_ms or now_ms
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _get(SPOT_URL, {
            "symbol": _binance_symbol(symbol), "interval": timeframe,
            "startTime": cursor, "limit": 1000,
        })
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + step
        if next_cursor <= cursor or len(batch) < 1000:
            break
        cursor = next_cursor
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=OHLCV_KEEP)
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    # Drop the candle that is still open (close_time in the future).
    df = df[df["close_time"].astype(np.int64) < now_ms]
    df.index = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df.index.name = "timestamp"
    return df[OHLCV_KEEP].astype(float)


def update_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    start: str = "2020-01-01",
    cache_dir: Optional[Path] = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Return full cached OHLCV history, fetching only bars newer than the cache (and older
    than the cache if ``start`` is earlier than what is stored)."""
    path = _cache_path(f"{_binance_symbol(symbol)}_{timeframe}.csv", cache_dir)
    cached = _read_cache(path)
    if offline:
        if cached.empty:
            raise DataFetchError(f"No cached data at {path} and offline=True")
        return cached

    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    pieces = []
    if cached.empty:
        pieces.append(_fetch_klines(symbol, timeframe, start_ms))
    else:
        first_ms = int(cached.index[0].value // 1_000_000)
        if start_ms < first_ms - TF_MS[timeframe]:
            pieces.append(_fetch_klines(symbol, timeframe, start_ms, first_ms))
        last_ms = int(cached.index[-1].value // 1_000_000)
        pieces.append(_fetch_klines(symbol, timeframe, last_ms + TF_MS[timeframe]))
    merged = cached
    for p in pieces:
        merged = _merge(merged, p)
    if merged.empty:
        raise DataFetchError(f"No OHLCV data returned for {symbol} {timeframe}")
    if len(merged) != len(cached):
        _write_cache(merged, path)
    return merged


# --------------------------------------------------------------------------- funding

def update_funding(
    symbol: str,
    start: str = "2020-01-01",
    cache_dir: Optional[Path] = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Perpetual funding-rate history (one row per 8h settlement, index = settlement time)."""
    path = _cache_path(f"funding_{_binance_symbol(symbol)}.csv", cache_dir)
    cached = _read_cache(path)
    if offline:
        if cached.empty:
            raise DataFetchError(f"No cached funding at {path} and offline=True")
        return cached
    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    cursor = start_ms if cached.empty else int(cached.index[-1].value // 1_000_000) + 1
    rows = []
    while True:
        batch = _get(FUNDING_URL, {"symbol": _binance_symbol(symbol), "startTime": cursor, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    if rows:
        new = pd.DataFrame(rows)
        new.index = pd.to_datetime(new["fundingTime"].astype(np.int64), unit="ms", utc=True)
        # Settlement times carry a few ms of jitter; snap to the minute.
        new.index = new.index.floor("min")
        new.index.name = "timestamp"
        new = new[["fundingRate"]].astype(float).rename(columns={"fundingRate": "funding_rate"})
        merged = _merge(cached, new)
        _write_cache(merged, path)
        return merged
    if cached.empty:
        raise DataFetchError(f"No funding data returned for {symbol}")
    return cached


# --------------------------------------------------------------------------- fear & greed

def update_fear_greed(cache_dir: Optional[Path] = None, offline: bool = False, max_age_hours: float = 6.0) -> pd.DataFrame:
    """Daily Fear & Greed index. Index = UTC day the value refers to (00:00)."""
    path = _cache_path("fear_greed.csv", cache_dir)
    cached = _read_cache(path)
    if offline or (not cached.empty and path.exists()
                   and (time.time() - path.stat().st_mtime) < max_age_hours * 3600):
        if cached.empty:
            raise DataFetchError(f"No cached Fear & Greed at {path} and offline=True")
        return cached
    payload = _get(FNG_URL, {"limit": 0, "format": "json"})
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not data:
        if cached.empty:
            raise DataFetchError("Empty Fear & Greed response")
        return cached
    new = pd.DataFrame(data)
    new.index = pd.to_datetime(new["timestamp"].astype(np.int64), unit="s", utc=True)
    new.index.name = "timestamp"
    new = new[["value"]].astype(float).rename(columns={"value": "fng"})
    merged = _merge(cached, new)
    _write_cache(merged, path)
    return merged


def update_all(symbols, timeframe: str = "1h", start: str = "2020-01-01", offline: bool = False,
               cache_dir: Optional[Path] = None, verbose: bool = True) -> dict:
    """Refresh every source for every symbol. Returns {"ohlcv": {sym: df}, "funding": {...}, "fng": df}."""
    out = {"ohlcv": {}, "funding": {}, "fng": None}
    for sym in symbols:
        out["ohlcv"][sym] = update_ohlcv(sym, timeframe, start, cache_dir, offline)
        try:
            out["funding"][sym] = update_funding(sym, start, cache_dir, offline)
        except DataFetchError as e:
            if verbose:
                print(f"[!] funding unavailable for {sym}: {e}")
            out["funding"][sym] = None
        if verbose:
            o = out["ohlcv"][sym]
            print(f"[data] {sym} {timeframe}: {len(o):,} bars {o.index[0]:%Y-%m-%d} -> {o.index[-1]:%Y-%m-%d %H:%M}")
    try:
        out["fng"] = update_fear_greed(cache_dir, offline)
    except DataFetchError as e:
        if verbose:
            print(f"[!] fear & greed unavailable: {e}")
    return out
