"""Assemble the pooled (multi-symbol) modelling table: features + forward-return targets."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.features import WARMUP_BARS, build_features, feature_columns

MARKET_SYMBOL = "BTC/USDT"


def add_targets(feats: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """fwd_ret = log(close[t+H] / close[t]); target = 1 if fwd_ret > 0 else 0.
    ret_next = log(close[t+1]/close[t]) is what a position held from close[t] earns over the next bar.
    The last H rows have no label (NaN)."""
    out = feats.copy()
    c = out["_close"]
    out["fwd_ret"] = np.log(c.shift(-horizon) / c)
    out["ret_next"] = np.log(c.shift(-1) / c)
    out["target"] = (out["fwd_ret"] > 0).astype(float).where(out["fwd_ret"].notna())
    return out


def build_symbol_features(data: dict, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    market = data["ohlcv"].get(MARKET_SYMBOL)
    return build_features(
        data["ohlcv"][symbol],
        funding=data["funding"].get(symbol),
        fng=data.get("fng"),
        market=market,
        timeframe=timeframe,
    )


def build_dataset(data: dict, symbols: List[str], horizon: int, timeframe: str = "1h",
                  start: Optional[str] = None, float32: bool = False) -> pd.DataFrame:
    """Long table with columns [timestamp, symbol, <features>, _close, fwd_ret, ret_next, target]."""
    frames = []
    for sym in symbols:
        f = build_symbol_features(data, sym, timeframe)
        f = f.iloc[WARMUP_BARS:]
        f = add_targets(f, horizon)
        if start is not None:
            f = f[f.index >= pd.Timestamp(start, tz="UTC")]
        if float32:
            num = [c for c in f.columns if c not in ("fwd_ret", "ret_next", "_close")]
            f[num] = f[num].astype("float32")
        f["symbol"] = sym
        frames.append(f)
    ds = pd.concat(frames).rename_axis("timestamp").reset_index()
    ds = ds.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return ds


def model_features(ds: pd.DataFrame) -> List[str]:
    return [c for c in feature_columns(ds) if c not in ("timestamp", "ret_next")]
