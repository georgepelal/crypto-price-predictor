"""Single feature builder used by research, final training and live trading.

Timing convention
-----------------
Rows are indexed by the candle OPEN time ``t`` (Binance convention). A row's features use only
information available when that candle CLOSES (t + 1 bar). The trading decision for row ``t`` is
made at that close and the position earns returns from close[t] onward.

All features are scale-free (returns, ratios, z-scores) so one model can be pooled across assets.
Higher-timeframe context is built from rolling windows over the base bars, which avoids
resample/merge alignment bugs entirely.

``WARMUP_BARS`` is how much history a row needs before all features are populated. Live inference
must load at least that many bars (it does: it reads the shared on-disk cache).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

WARMUP_BARS = 24 * 31  # longest window below is 720 bars (30 days on 1h)
EPS = 1e-12


def regularize(ohlcv: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Put bars on a complete time grid. Missing bars (exchange outages) become flat bars at the
    previous close with zero volume, so that shift(k) always means k bars of wall-clock time."""
    df = ohlcv.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    grid = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    out = df.reindex(grid)
    missing = out["close"].isna()
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    for c in ("volume", "quote_volume", "trades", "taker_buy_base"):
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    out["is_filled"] = missing.astype(float)
    out.index.name = "timestamp"
    return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / (dn + EPS))


def _zscore(s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    mp = min_periods or window // 2
    m = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std()
    return (s - m) / (sd + EPS)


def price_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Features from a single asset's OHLCV (+ taker buy volume if present)."""
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame(index=df.index)

    # Momentum over many horizons (1h .. 7d) and their volatility-scaled versions
    vol24 = lr.rolling(24, min_periods=12).std()
    vol168 = lr.rolling(168, min_periods=84).std()
    for k in (1, 2, 4, 8, 12, 24, 48, 72, 168, 336, 720):
        r = np.log(c / c.shift(k))
        f[f"ret_{k}"] = r
        if k >= 4:
            f[f"ret_{k}_z"] = r / (vol168 * np.sqrt(k) + EPS)

    # Volatility regime
    f["vol_24"] = vol24
    f["vol_168"] = vol168
    f["vol_ratio_24_168"] = vol24 / (vol168 + EPS)
    f["vol_ratio_168_720"] = vol168 / (lr.rolling(720, min_periods=360).std() + EPS)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / c
    park = np.log(h / l) ** 2 / (4 * np.log(2))
    f["park_vol_24"] = np.sqrt(park.rolling(24, min_periods=12).mean())

    # Candle shape
    rng = (h - l).replace(0, np.nan)
    f["bar_range"] = (h - l) / c
    f["close_loc"] = ((c - l) / rng).fillna(0.5)            # 0 = closed at low, 1 = at high
    f["body"] = (c - o) / c

    # Oscillators / trend distance
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_56"] = _rsi(c, 56)
    for span in (24, 72, 168, 720):
        ema = c.ewm(span=span, min_periods=span // 2, adjust=False).mean()
        f[f"dist_ema_{span}"] = np.log(c / ema) / (vol168 * np.sqrt(span) + EPS)
    for w in (24, 168, 720):
        hh = h.rolling(w, min_periods=w // 2).max()
        ll = l.rolling(w, min_periods=w // 2).min()
        f[f"range_pos_{w}"] = (c - ll) / (hh - ll + EPS)
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    f["bb_z_20"] = (c - bb_m) / (bb_s + EPS)

    # Volume & order flow
    lv = np.log1p(v)
    f["vol_z_24"] = _zscore(lv, 24)
    f["vol_z_168"] = _zscore(lv, 168)
    if "taker_buy_base" in df.columns:
        buy_ratio = (df["taker_buy_base"] / (v + EPS)).where(v > 0, 0.5)
        f["taker_buy_ratio"] = buy_ratio
        f["taker_buy_ratio_8"] = buy_ratio.rolling(8, min_periods=4).mean()
        f["taker_buy_ratio_24"] = buy_ratio.rolling(24, min_periods=12).mean()
        imb = (2 * buy_ratio - 1) * v
        f["flow_imb_24"] = imb.rolling(24, min_periods=12).sum() / (v.rolling(24, min_periods=12).sum() + EPS)
    if "trades" in df.columns:
        f["trades_z_168"] = _zscore(np.log1p(df["trades"]), 168)
    if "is_filled" in df.columns:
        f["filled_bar"] = df["is_filled"]

    if prefix:
        f = f.add_prefix(prefix)
    return f


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    # Decision time is the bar close, so use close-time hour.
    t = index + pd.Timedelta(hours=1)
    f = pd.DataFrame(index=index)
    f["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * t.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * t.dayofweek / 7)
    return f


def funding_features(index: pd.DatetimeIndex, funding: Optional[pd.DataFrame], bar: pd.Timedelta) -> pd.DataFrame:
    """Latest funding settled strictly before the decision time (bar close)."""
    f = pd.DataFrame(index=index)
    if funding is None or funding.empty:
        return f
    fr = funding["funding_rate"].sort_index()
    ff = pd.DataFrame({
        "fund_last": fr,
        "fund_avg_3": fr.rolling(3, min_periods=1).mean(),      # ~1 day
        "fund_avg_21": fr.rolling(21, min_periods=7).mean(),    # ~1 week
        "fund_z_90": _zscore(fr, 90, 30),                       # ~1 month
    })
    decision = pd.DataFrame({"decision_time": index + bar}, index=index).rename_axis("timestamp")
    merged = pd.merge_asof(
        decision.reset_index().sort_values("decision_time"),
        ff.rename_axis("settle_time").reset_index(),
        left_on="decision_time", right_on="settle_time",
        direction="backward", allow_exact_matches=False,
    ).set_index("timestamp")
    stale = (merged["decision_time"] - merged["settle_time"]) > pd.Timedelta(hours=24)
    for col in ff.columns:
        f[col] = merged[col].where(~stale)
    return f


def fear_greed_features(index: pd.DatetimeIndex, fng: Optional[pd.DataFrame], bar: pd.Timedelta) -> pd.DataFrame:
    """Daily index. Treated as known only from the START of the following UTC day (conservative)."""
    f = pd.DataFrame(index=index)
    if fng is None or fng.empty:
        return f
    s = fng["fng"].sort_index()
    daily = pd.DataFrame({
        "fng": s,
        "fng_chg_7d": s - s.shift(7),
        "fng_z_90": _zscore(s, 90, 30),
    })
    daily.index = daily.index + pd.Timedelta(days=1)  # available from next day 00:00 UTC
    decision = pd.DataFrame({"decision_time": index + bar}, index=index).rename_axis("timestamp")
    merged = pd.merge_asof(
        decision.reset_index().sort_values("decision_time"),
        daily.rename_axis("avail_time").reset_index(),
        left_on="decision_time", right_on="avail_time", direction="backward",
    ).set_index("timestamp")
    stale = (merged["decision_time"] - merged["avail_time"]) > pd.Timedelta(days=3)
    for col in daily.columns:
        f[col] = merged[col].where(~stale)
    return f


def build_features(
    ohlcv: pd.DataFrame,
    funding: Optional[pd.DataFrame] = None,
    fng: Optional[pd.DataFrame] = None,
    market: Optional[pd.DataFrame] = None,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Full feature matrix for one symbol.

    ``market`` is the (regularized) OHLCV of the market leader (BTC); its momentum/vol is added as
    ``mkt_*`` features so altcoin models see the market factor. Pass None for BTC itself.
    """
    bar = pd.Timedelta(timeframe)
    df = regularize(ohlcv, freq=timeframe)
    parts: List[pd.DataFrame] = [price_features(df), calendar_features(df.index),
                                 funding_features(df.index, funding, bar),
                                 fear_greed_features(df.index, fng, bar)]
    if market is not None:
        m = regularize(market, freq=timeframe).reindex(df.index)
        mc = m["close"].ffill()
        mlr = np.log(mc / mc.shift(1))
        mf = pd.DataFrame(index=df.index)
        for k in (1, 4, 24, 168):
            mf[f"mkt_ret_{k}"] = np.log(mc / mc.shift(k))
        mf["mkt_vol_24"] = mlr.rolling(24, min_periods=12).std()
        own = np.log(df["close"] / df["close"].shift(1))
        cov = own.rolling(720, min_periods=360).cov(mlr)
        mf["beta_720"] = cov / (mlr.rolling(720, min_periods=360).var() + EPS)
        mf["resid_ret_24"] = np.log(df["close"] / df["close"].shift(24)) - mf["beta_720"] * mf["mkt_ret_24"]
        parts.append(mf)
    feats = pd.concat(parts, axis=1)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    # Keep raw prices alongside for targets/backtests (not model inputs).
    feats["_close"] = df["close"]
    return feats


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not c.startswith("_") and c not in ("target", "fwd_ret", "symbol")]
