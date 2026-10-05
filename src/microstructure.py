"""Market Microstructure & Order Flow Proxy Features.

Implements institutional quantitative microstructure indicators from academic finance:
1. Amihud Illiquidity Ratio (Amihud, 2002)
2. Corwin-Schultz High-Low Bid-Ask Spread Estimator (Corwin & Schultz, 2012)
3. Roll's Effective Spread Estimator (Roll, 1984)
4. Order Flow Imbalance (OFI) via Tick Rule (Lee & Ready, 1991)
5. Kyle's Lambda Price Impact Proxy (Kyle, 1985)

All features are backward-looking (t and earlier) to guarantee zero lookahead bias.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window: int = 24,
) -> pd.Series:
    """Compute Amihud (2002) illiquidity ratio z-score: |R_t| / (P_t * Vol_t).

    Measures price impact per dollar traded. High values indicate fragile liquidity.
    """
    ret = (close / close.shift(1) - 1.0).abs()
    dollar_vol = (close * volume) + 1e-10
    raw_amihud = ret / dollar_vol

    # Rolling standardized z-score
    roll_mean = raw_amihud.rolling(window=window, min_periods=window // 2).mean()
    roll_std = raw_amihud.rolling(window=window, min_periods=window // 2).std()
    zscore = ((raw_amihud - roll_mean) / (roll_std + 1e-12)).fillna(0.0).clip(-3.0, 3.0)
    return zscore


def compute_kyle_lambda(
    close: pd.Series,
    volume: pd.Series,
    window: int = 24,
) -> pd.Series:
    """Compute Kyle's Lambda (1985) price impact proxy: |Delta P| / DollarVolume."""
    delta_p = close.diff().abs()
    dollar_vol = (close * volume) + 1e-10
    raw_lambda = delta_p / dollar_vol

    roll_mean = raw_lambda.rolling(window=window, min_periods=window // 2).mean()
    roll_std = raw_lambda.rolling(window=window, min_periods=window // 2).std()
    zscore = ((raw_lambda - roll_mean) / (roll_std + 1e-12)).fillna(0.0).clip(-3.0, 3.0)
    return zscore


def compute_corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Estimate bid-ask spread from high-low price ranges (Corwin & Schultz, 2012).

    Uses the variance of 2-day high-low log ratios to extract effective spread.
    """
    log_hl = np.log(high / (low + 1e-10))
    beta = (log_hl.shift(1) ** 2) + (log_hl ** 2)

    h2 = np.maximum(high, high.shift(1))
    l2 = np.minimum(low, low.shift(1))
    gamma = np.log(h2 / (l2 + 1e-10)) ** 2

    denom = 3.0 - (2.0 * np.sqrt(2.0))
    alpha = ((np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom) - np.sqrt(gamma / denom)
    alpha = np.maximum(0.0, alpha.fillna(0.0))

    # Spread formula: 2 * (e^alpha - 1) / (1 + e^alpha)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.fillna(0.0).clip(0.0, 0.05)


def compute_roll_effective_spread(
    close: pd.Series,
    window: int = 24,
) -> pd.Series:
    """Estimate effective bid-ask spread from serial autocovariance (Roll, 1984).

    Spread = 2 * sqrt(max(0, -Cov(Delta P_t, Delta P_{t-1}))) / P_t
    """
    dp = close.diff()
    cov = dp.rolling(window=window, min_periods=window // 2).cov(dp.shift(1))
    spread = 2.0 * np.sqrt(np.maximum(0.0, -cov)) / (close + 1e-10)
    return spread.fillna(0.0).clip(0.0, 0.05)


def compute_order_flow_imbalance(
    close: pd.Series,
    volume: pd.Series,
    windows: Tuple[int, ...] = (6, 12, 24),
) -> Dict[str, pd.Series]:
    """Estimate aggressive buyer vs seller volume imbalance using the Tick Rule (Lee & Ready, 1991).

    Sign_t = +1 if P_t > P_{t-1}, -1 if P_t < P_{t-1}, else Sign_{t-1}.
    """
    diff = close.diff().fillna(0.0)
    n = len(close)
    sign = np.zeros(n, dtype=float)
    curr_sign = 0.0

    for i in range(n):
        d = diff.iloc[i]
        if d > 0:
            curr_sign = 1.0
        elif d < 0:
            curr_sign = -1.0
        sign[i] = curr_sign

    vol = volume.values
    buy_vol = vol * (1.0 + sign) / 2.0
    sell_vol = vol * (1.0 - sign) / 2.0

    s_buy = pd.Series(buy_vol, index=close.index)
    s_sell = pd.Series(sell_vol, index=close.index)

    # Instantaneous 1-bar Order Flow Imbalance (-1 to +1)
    ofi_1h = (s_buy - s_sell) / (s_buy + s_sell + 1e-10)

    results = {"micro_ofi_1h": ofi_1h}

    # Cumulative moving average imbalance across multiple windows
    for win in windows:
        roll_buy = s_buy.rolling(window=win, min_periods=1).sum()
        roll_sell = s_sell.rolling(window=win, min_periods=1).sum()
        results[f"micro_ofi_ma_{win}"] = (roll_buy - roll_sell) / (roll_buy + roll_sell + 1e-10)

    # Imbalance acceleration: short-term pressure vs medium-term pressure
    if 6 in windows and 24 in windows:
        results["micro_ofi_acceleration"] = results["micro_ofi_ma_6"] - results["micro_ofi_ma_24"]

    return results


def compute_cvd_divergence(
    close: pd.Series,
    volume: pd.Series,
    window: int = 24,
) -> Dict[str, pd.Series]:
    """Compute Cumulative Volume Delta (CVD) and divergence / absorption signals.

    Bullish Divergence (Absorption): Price makes a lower low while aggressive buyers load (CVD higher low).
    Bearish Divergence (Exhaustion): Price makes a higher high while aggressive buying fails (CVD lower high).
    """
    diff = close.diff().fillna(0.0)
    sign = np.where(diff > 0, 1.0, np.where(diff < 0, -1.0, 0.0))
    s_series = pd.Series(sign, index=close.index).replace(0.0, np.nan).ffill().fillna(1.0)
    delta_vol = volume * s_series
    cvd = delta_vol.cumsum()

    # Rolling standardized z-score of 24h CVD trend
    cvd_mean = cvd.rolling(window=window, min_periods=window // 2).mean()
    cvd_std = cvd.rolling(window=window, min_periods=window // 2).std()
    cvd_z = ((cvd - cvd_mean) / (cvd_std + 1e-8)).fillna(0.0).clip(-3.0, 3.0)

    # Relative rank comparison over rolling window
    cvd_roll_min = cvd.rolling(window, min_periods=1).min()
    cvd_roll_max = cvd.rolling(window, min_periods=1).max()
    p_roll_min = close.rolling(window, min_periods=1).min()
    p_roll_max = close.rolling(window, min_periods=1).max()

    p_rank = (close - p_roll_min) / (p_roll_max - p_roll_min + 1e-8)
    cvd_rank = (cvd - cvd_roll_min) / (cvd_roll_max - cvd_roll_min + 1e-8)
    cvd_div = (cvd_rank - p_rank).fillna(0.0).clip(-1.0, 1.0)

    return {
        "micro_cvd_zscore": cvd_z,
        "micro_cvd_divergence": cvd_div,
    }


def compute_liquidation_flush_proxy(
    df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Detect forced liquidation cascades followed by absorption wicks (Snapback Reversals)."""
    close = df["close"]
    open_p = df.get("open", close)
    high = df["high"]
    low = df["low"]
    atr = (high - low).rolling(14, min_periods=1).mean()

    # Lower and upper shadow lengths in dollars
    lower_shadow = np.maximum(0.0, np.minimum(open_p, close) - low)
    upper_shadow = np.maximum(0.0, high - np.maximum(open_p, close))

    # Detect sudden Open Interest collapse if available
    if "oi_change_1h" in df.columns:
        oi_chg = df["oi_change_1h"]
    elif "oi_val" in df.columns:
        oi_chg = df["oi_val"].pct_change().fillna(0.0)
    else:
        oi_chg = pd.Series(0.0, index=df.index)

    # Long Liquidation Flush: OI drops sharply (longs liquidated) with long lower wick
    flush_long = (oi_chg <= -0.010) & (lower_shadow >= (0.7 * atr))
    # Short Liquidation Flush: OI drops sharply (shorts liquidated) with long upper wick
    flush_short = (oi_chg <= -0.010) & (upper_shadow >= (0.7 * atr))

    return {
        "micro_liq_flush_bullish": flush_long.astype(float),
        "micro_liq_flush_bearish": flush_short.astype(float),
    }


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all microstructure, CVD divergence, and liquidation features to a DataFrame.

    Requires columns: 'high', 'low', 'close', 'volume'.
    """
    out = df.copy()

    # 1. Amihud Illiquidity & Kyle's Lambda
    out["micro_amihud_zscore"] = compute_amihud_illiquidity(out["close"], out["volume"], window=24)
    out["micro_kyle_lambda_zscore"] = compute_kyle_lambda(out["close"], out["volume"], window=24)

    # 2. Effective Spread Estimators
    cs_spread = compute_corwin_schultz_spread(out["high"], out["low"])
    out["micro_corwin_schultz"] = cs_spread
    out["micro_cs_spread_ma12"] = cs_spread.rolling(window=12, min_periods=1).mean()
    out["micro_roll_spread"] = compute_roll_effective_spread(out["close"], window=24)

    # 3. Order Flow Imbalance via Tick Rule
    ofi_dict = compute_order_flow_imbalance(out["close"], out["volume"], windows=(6, 12, 24))
    for col, s in ofi_dict.items():
        out[col] = s

    # 4. Cumulative Volume Delta (CVD) Divergence & Absorption
    cvd_dict = compute_cvd_divergence(out["close"], out["volume"], window=24)
    for col, s in cvd_dict.items():
        out[col] = s

    # 5. Liquidation Cascade Flush Detection
    liq_dict = compute_liquidation_flush_proxy(out)
    for col, s in liq_dict.items():
        out[col] = s

    return out
