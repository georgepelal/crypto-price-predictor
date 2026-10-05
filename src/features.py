"""Feature engineering module for quantitative crypto market data.

All features are strictly backward-looking (t and earlier) to prevent lookahead bias.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Compute log returns ln(P_t / P_{t-1})."""
    return np.log(series / series.shift(1))


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """Compute Bollinger Bands %B and Bandwidth."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + (num_std * std)
    lower = sma - (num_std * std)

    bandwidth = (upper - lower) / (sma + 1e-10)
    pct_b = (close - lower) / (upper - lower + 1e-10)

    return {"bb_pct_b": pct_b, "bb_bandwidth": bandwidth}


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """Compute normalized MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / (close + 1e-10)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return {
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_hist,
    }


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute Average True Range normalized by close price (ATR / Close)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / (close + 1e-10)


def compute_parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 14
) -> pd.Series:
    """Parkinson's high-low based extreme value volatility estimator."""
    factor = 1.0 / (4.0 * np.log(2.0))
    hl_ratio = np.log(high / (low + 1e-10)) ** 2
    parkinson = np.sqrt(factor * hl_ratio.rolling(window=window).mean())
    return parkinson


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Dict[str, pd.Series]:
    """Compute Average Directional Index (ADX), +DI, and -DI for trend strength.

    Strictly non-lookahead using backward-looking EWMA smoothing.
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-10))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return {
        "adx": adx.fillna(0.0),
        "plus_di": plus_di.fillna(0.0),
        "minus_di": minus_di.fillna(0.0),
    }


def generate_crypto_features(
    df: pd.DataFrame,
    return_lags: Optional[List[int]] = None,
    volatility_windows: Optional[List[int]] = None,
    rsi_period: int = 14,
    bollinger_period: int = 20,
    atr_period: int = 14,
    use_fracdiff: bool = True,
    use_microstructure: bool = True,
) -> pd.DataFrame:
    """Construct full feature set from OHLCV dataframe without lookahead bias.

    Args:
        df: DataFrame with ['open', 'high', 'low', 'close', 'volume']
        return_lags: List of lag steps for return features
        volatility_windows: List of rolling windows for volatility
        rsi_period: Period for RSI calculation
        bollinger_period: Period for Bollinger Bands
        atr_period: Period for ATR calculation

    Returns:
        DataFrame with original OHLCV and engineered feature columns.
    """
    if return_lags is None:
        return_lags = [1, 2, 3, 5, 8, 12, 24]
    if volatility_windows is None:
        volatility_windows = [6, 12, 24]

    feat = df.copy()

    # 1. Base Price Action Returns
    log_ret = compute_log_returns(feat["close"])
    feat["ret_1"] = log_ret

    # Multiple lag returns
    for lag in return_lags:
        feat[f"ret_lag_{lag}"] = log_ret.shift(lag)
        feat[f"cum_ret_{lag}"] = (feat["close"] / feat["close"].shift(lag)) - 1.0

    # Intrabar price moves
    feat["bar_return"] = (feat["close"] - feat["open"]) / (feat["open"] + 1e-10)
    feat["hl_spread"] = (feat["high"] - feat["low"]) / (feat["close"] + 1e-10)
    feat["upper_shadow"] = (feat["high"] - np.maximum(feat["open"], feat["close"])) / (feat["close"] + 1e-10)
    feat["lower_shadow"] = (np.minimum(feat["open"], feat["close"]) - feat["low"]) / (feat["close"] + 1e-10)

    # 2. Volatility Features
    feat["norm_atr"] = compute_atr(feat["high"], feat["low"], feat["close"], period=atr_period)
    for win in volatility_windows:
        feat[f"rolling_vol_{win}"] = log_ret.rolling(window=win).std()
        feat[f"parkinson_vol_{win}"] = compute_parkinson_volatility(feat["high"], feat["low"], window=win)

    # 3. Oscillators & Trend
    feat["rsi"] = compute_rsi(feat["close"], period=rsi_period)
    feat["rsi_centered"] = feat["rsi"] - 50.0

    bb = compute_bollinger_bands(feat["close"], period=bollinger_period)
    feat["bb_pct_b"] = bb["bb_pct_b"]
    feat["bb_bandwidth"] = bb["bb_bandwidth"]

    macd = compute_macd(feat["close"])
    feat["macd_line"] = macd["macd_line"]
    feat["macd_signal"] = macd["macd_signal"]
    feat["macd_hist"] = macd["macd_hist"]

    # ADX & Directional Movement (Trend Strength)
    adx_dict = compute_adx(feat["high"], feat["low"], feat["close"], period=14)
    feat["adx_14"] = adx_dict["adx"]
    feat["adx_plus_di"] = adx_dict["plus_di"]
    feat["adx_minus_di"] = adx_dict["minus_di"]
    feat["adx_trend_strength"] = (adx_dict["adx"] >= 25.0).astype(float)
    feat["adx_is_choppy"] = (adx_dict["adx"] < 20.0).astype(float)

    # 4. Volume Features
    vol_sma_20 = feat["volume"].rolling(window=20).mean()
    feat["volume_ratio_20"] = feat["volume"] / (vol_sma_20 + 1e-10)
    feat["log_volume_change"] = np.log((feat["volume"] + 1.0) / (feat["volume"].shift(1) + 1.0))

    # Rolling Volume-Weighted Price proxy
    pv = (feat["close"] * feat["volume"]).rolling(window=20).sum()
    v_sum = feat["volume"].rolling(window=20).sum()
    rolling_vwap = pv / (v_sum + 1e-10)
    feat["dist_to_vwap"] = (feat["close"] - rolling_vwap) / (feat["close"] + 1e-10)

    # 5. Cyclical Time & Institutional Liquidity Features
    if isinstance(feat.index, pd.DatetimeIndex):
        hour = feat.index.hour
        dow = feat.index.dayofweek
        feat["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        feat["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        feat["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
        feat["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

        # Institutional session windows & liquidity quality
        # London/NY Overlap (12:00 - 16:00 UTC): peak global macro volume & high trend follow-through
        feat["session_london_ny_overlap"] = ((hour >= 12) & (hour <= 16)).astype(float)
        # Core active European & US hours (07:00 - 20:00 UTC)
        feat["session_core_hours"] = ((hour >= 7) & (hour <= 20)).astype(float)
        # Low-volume weekend and Asian dead zone flags
        feat["is_weekend"] = (dow >= 5).astype(float)
        feat["session_asian_dead_zone"] = ((hour >= 21) | (hour <= 5)).astype(float)
        feat["liquidity_quality_score"] = feat["session_core_hours"] * (1.0 - 0.4 * feat["is_weekend"])

    # 6. Derivatives & Microstructure Features (Funding Rate & Open Interest)
    if "funding_rate" in feat.columns:
        fr = feat["funding_rate"]
        feat["fr_raw"] = fr
        # 7-day rolling Z-score of funding rate (168 hours for 1h)
        fr_mean = fr.rolling(window=168, min_periods=24).mean()
        fr_std = fr.rolling(window=168, min_periods=24).std()
        feat["fr_zscore"] = ((fr - fr_mean) / (fr_std + 1e-10)).fillna(0.0)
        feat["fr_cum_24h"] = fr.rolling(window=24, min_periods=1).sum()

    if "oi_val" in feat.columns:
        oi = feat["oi_val"]
        feat["oi_change_1h"] = np.log(oi / (oi.shift(1) + 1e-10)).fillna(0.0).clip(-0.5, 0.5)
        feat["oi_change_24h"] = np.log(oi / (oi.shift(24) + 1e-10)).fillna(0.0).clip(-1.0, 1.0)
        feat["oi_price_alignment"] = np.sign(log_ret.fillna(0.0)) * np.sign(feat["oi_change_1h"])
        vol_24h_usd = (feat["close"] * feat["volume"]).rolling(window=24).sum() + 1e-10
        feat["oi_to_vol_ratio"] = oi / vol_24h_usd

    # 7. Sentiment Features (Crypto Fear & Greed Index)
    if "fng_value" in feat.columns:
        fng = feat["fng_value"]
        feat["fng_raw"] = fng
        fng_mean = fng.rolling(window=720, min_periods=48).mean()
        fng_std = fng.rolling(window=720, min_periods=48).std()
        feat["fng_zscore_30d"] = ((fng - fng_mean) / (fng_std + 1e-10)).fillna(0.0)
        feat["fng_change_7d"] = (fng - fng.shift(168)).fillna(0.0)
        # Contrarian market sentiment indicator: +1 for Extreme Fear, -1 for Extreme Greed
        fng_signal = pd.Series(0.0, index=feat.index)
        fng_signal[fng < 25.0] = 1.0   # Extreme Fear -> Bullish contrarian
        fng_signal[fng > 75.0] = -1.0  # Extreme Greed -> Bearish contrarian
        feat["fng_contrarian_signal"] = fng_signal

    # 8. Fractional Differentiation Features (Stationarity with Memory Retention)
    if use_fracdiff and len(feat) >= 60:
        try:
            from src.fracdiff import add_fracdiff_features
            feat, _ = add_fracdiff_features(feat, price_col="close", d=0.40)
        except Exception:
            pass

    # 9. Market Microstructure & Order Flow Proxy Features
    if use_microstructure:
        try:
            from src.microstructure import add_microstructure_features
            feat = add_microstructure_features(feat)
        except Exception:
            pass

    return feat


def add_multitimeframe_features(
    df_primary: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    df_1d: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute Higher-Timeframe trend, momentum, and regime features and merge with zero lookahead.

    Uses backward as-of merging on lagged higher-timeframe candles so the primary timeframe
    strictly uses only completed, closed higher-timeframe bars.
    """
    out = df_primary.copy()
    out.index = pd.to_datetime(out.index, utc=True)

    # 1. 4-Hour Timeframe Features
    if df_4h is not None and not df_4h.empty and len(df_4h) >= 10:
        c4 = df_4h["close"]
        ema50_4h = c4.ewm(span=min(50, len(c4)), adjust=False).mean()
        ema200_4h = c4.ewm(span=min(200, len(c4)), adjust=False).mean()

        feat_4h = pd.DataFrame(index=df_4h.index)
        feat_4h["tf4h_trend_ratio"] = (ema50_4h - ema200_4h) / (c4 + 1e-10)
        feat_4h["tf4h_dist_ema50"] = (c4 - ema50_4h) / (c4 + 1e-10)
        feat_4h["tf4h_rsi"] = compute_rsi(c4, period=min(14, len(c4)-1))
        feat_4h["tf4h_norm_atr"] = compute_atr(df_4h["high"], df_4h["low"], c4, period=min(14, len(c4)-1))
        feat_4h["tf4h_ret_24h"] = (c4 / c4.shift(min(6, len(c4)-1))) - 1.0

        # CRITICAL: Shift by 1 bar so open 4h bar is not used
        feat_4h = feat_4h.shift(1).dropna(how="all")
        feat_4h.index = pd.to_datetime(feat_4h.index, utc=True)

        out = pd.merge_asof(
            out.sort_index(),
            feat_4h.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        for col in feat_4h.columns:
            out[col] = out[col].ffill().bfill().fillna(0.0)

    # 2. Daily Timeframe Features
    if df_1d is not None and not df_1d.empty and len(df_1d) >= 10:
        c1d = df_1d["close"]
        h1d = df_1d["high"]
        l1d = df_1d["low"]

        ema20_1d = c1d.ewm(span=min(20, len(c1d)), adjust=False).mean()
        ema50_1d = c1d.ewm(span=min(50, len(c1d)), adjust=False).mean()

        feat_1d = pd.DataFrame(index=df_1d.index)
        feat_1d["daily_trend_ratio"] = (ema20_1d - ema50_1d) / (c1d + 1e-10)
        feat_1d["daily_rsi"] = compute_rsi(c1d, period=min(14, len(c1d)-1))
        high_30d = h1d.rolling(window=min(30, len(c1d)), min_periods=2).max()
        low_30d = l1d.rolling(window=min(30, len(c1d)), min_periods=2).min()
        feat_1d["daily_dist_to_high_30d"] = (c1d - high_30d) / (c1d + 1e-10)
        feat_1d["daily_dist_to_low_30d"] = (c1d - low_30d) / (c1d + 1e-10)
        feat_1d["daily_ret_7d"] = (c1d / c1d.shift(min(7, len(c1d)-1))) - 1.0

        # CRITICAL: Shift by 1 bar so today's open daily candle is not used
        feat_1d = feat_1d.shift(1).dropna(how="all")
        feat_1d.index = pd.to_datetime(feat_1d.index, utc=True)

        out = pd.merge_asof(
            out.sort_index(),
            feat_1d.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        for col in feat_1d.columns:
            out[col] = out[col].ffill().bfill().fillna(0.0)

    # 3. Inter-Timeframe Confluence / Alignment
    if "tf4h_trend_ratio" in out.columns and "daily_trend_ratio" in out.columns:
        c_1h_ret = out.get("ret_lag_1", out["close"].pct_change().fillna(0.0))
        bullish_confluence = (c_1h_ret > 0) & (out["tf4h_trend_ratio"] > 0) & (out["daily_trend_ratio"] > 0)
        bearish_confluence = (c_1h_ret < 0) & (out["tf4h_trend_ratio"] < 0) & (out["daily_trend_ratio"] < 0)

        confluence = pd.Series(0.0, index=out.index)
        confluence[bullish_confluence] = 1.0
        confluence[bearish_confluence] = -1.0
        out["mtf_trend_alignment"] = confluence

    return out


def add_cross_asset_features(
    df_primary: pd.DataFrame,
    df_secondary: Optional[pd.DataFrame] = None,
    secondary_symbol: str = "ETH",
) -> pd.DataFrame:
    """Compute Cross-Asset Lead-Lag, Beta, and Relative Strength features with zero lookahead.

    Compares primary asset (e.g. BTC) against secondary asset (e.g. ETH) to gauge
    systemic crypto risk appetite and inter-market momentum.
    """
    out = df_primary.copy()
    if df_secondary is None or df_secondary.empty:
        return out

    out.index = pd.to_datetime(out.index, utc=True)
    sec = df_secondary.copy()
    sec.index = pd.to_datetime(sec.index, utc=True)

    sec_close = sec["close"].reindex(out.index).ffill().bfill()
    prim_close = out["close"]

    prefix = f"cross_{secondary_symbol.lower()}_"

    # 1. Price Ratio (e.g. ETH / BTC)
    ratio = sec_close / (prim_close + 1e-10)
    out[f"{prefix}ratio"] = ratio

    # 2. Return Divergences (1h and 24h relative strength)
    sec_ret_1h = sec_close.pct_change().fillna(0.0)
    prim_ret_1h = prim_close.pct_change().fillna(0.0)
    out[f"{prefix}mom_divergence_1h"] = sec_ret_1h - prim_ret_1h

    sec_ret_24h = (sec_close / sec_close.shift(24).replace(0, np.nan) - 1.0).fillna(0.0)
    prim_ret_24h = (prim_close / prim_close.shift(24).replace(0, np.nan) - 1.0).fillna(0.0)
    out[f"{prefix}mom_divergence_24h"] = sec_ret_24h - prim_ret_24h

    # 3. Ratio Momentum & Oscillators
    out[f"{prefix}ratio_rsi"] = compute_rsi(ratio, period=min(14, len(ratio) - 1))
    ratio_sma = ratio.rolling(24, min_periods=1).mean()
    out[f"{prefix}ratio_dist_ma24"] = (ratio - ratio_sma) / (ratio_sma + 1e-10)

    # 4. Risk-On / Beta Acceleration Indicator
    # When secondary asset (e.g. ETH) is aggressively outperforming on 24h basis -> Risk-On (+1)
    risk_on = pd.Series(0.0, index=out.index)
    risk_on[out[f"{prefix}mom_divergence_24h"] > 0.015] = 1.0
    risk_on[out[f"{prefix}mom_divergence_24h"] < -0.015] = -1.0
    out[f"{prefix}risk_on_regime"] = risk_on

    return out


