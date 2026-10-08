"""Target labeling module for supervised machine learning in financial time series.

Provides fixed-horizon return labeling, volatility-buffered classification,
and triple-barrier labeling (Marcos Lopez de Prado).
"""

from typing import Tuple
import numpy as np
import pandas as pd


def make_directional_labels(
    df: pd.DataFrame,
    forward_bars: int = 1,
    atr_multiplier: float = 0.35,
    fee_buffer: float = 0.0005,
) -> pd.DataFrame:
    """Create volatility-buffered directional labels.

    Target classes:
      2 = Up (Bullish, return > fee + multiplier * ATR)
      1 = Neutral (Noise / Range-bound)
      0 = Down (Bearish, return < -(fee + multiplier * ATR))

    Args:
        df: DataFrame with 'close' and optionally 'norm_atr'
        forward_bars: How many bars forward to evaluate return (e.g. 1 bar = next 1h)
        atr_multiplier: Multiplier for dynamic volatility threshold
        fee_buffer: Constant hurdle to account for round-trip fees (e.g. 0.05%)

    Returns:
        DataFrame with added 'fwd_return' and 'target' columns.
    """
    out = df.copy()

    # Forward return: ln(Close_{t+k} / Close_t)
    fwd_return = np.log(out["close"].shift(-forward_bars) / out["close"])
    out["fwd_return"] = fwd_return

    # Dynamic volatility hurdle
    if "norm_atr" in out.columns:
        threshold = fee_buffer + (atr_multiplier * out["norm_atr"])
    else:
        # Fallback to rolling std if norm_atr not present
        ret = np.log(out["close"] / out["close"].shift(1))
        rolling_std = ret.rolling(24).std()
        threshold = fee_buffer + (atr_multiplier * rolling_std)

    out["threshold"] = threshold

    # Class assignment: 0 = Down, 1 = Neutral, 2 = Up
    target = np.ones(len(out), dtype=int)  # default 1 (Neutral)
    target[fwd_return > threshold] = 2    # Up
    target[fwd_return < -threshold] = 0   # Down

    # Invalidate the last forward_bars rows (since future is unknown)
    target[-forward_bars:] = -1
    out["target"] = target

    return out


def apply_triple_barrier_labeling(
    df: pd.DataFrame,
    horizon_bars: int = 6,
    profit_mult: float = 1.5,
    stop_mult: float = 1.0,
) -> pd.DataFrame:
    """Triple Barrier Method labeling (Marcos Lopez de Prado):
    1. Upper barrier (Take Profit): Close_t * (1 + profit_mult * ATR)
    2. Lower barrier (Stop Loss): Close_t * (1 - stop_mult * ATR)
    3. Vertical barrier: Horizon of H bars.

    Returns:
        DataFrame with 'target', 'fwd_return', 't1', 'tp_dist', and 'sl_dist' columns.
    """
    out = df.copy()
    close = out["close"].values
    high = out["high"].values
    low = out["low"].values
    atr = out["norm_atr"].values if "norm_atr" in out.columns else np.full(len(out), 0.01)
    timestamps = out.index

    n = len(out)
    targets = np.full(n, 1, dtype=int)
    returns = np.zeros(n)
    t1_list = [None] * n
    tp_dists = np.zeros(n)
    sl_dists = np.zeros(n)

    for i in range(n - horizon_bars):
        c_i = close[i]
        curr_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 0.01

        upper_barrier = c_i * (1.0 + (profit_mult * curr_atr))
        lower_barrier = c_i * (1.0 - (stop_mult * curr_atr))
        tp_dists[i] = profit_mult * curr_atr
        sl_dists[i] = stop_mult * curr_atr

        hit = 1
        ret_val = 0.0
        hit_ts = timestamps[i + horizon_bars]

        for step in range(1, horizon_bars + 1):
            idx = i + step
            h_val = high[idx]
            l_val = low[idx]

            # In case of dual touch in same bar, conservative assumption: stop loss hit first
            if l_val <= lower_barrier and h_val >= upper_barrier:
                hit = 0
                ret_val = np.log(lower_barrier / c_i)
                hit_ts = timestamps[idx]
                break
            elif h_val >= upper_barrier:
                hit = 2  # profit barrier hit
                ret_val = np.log(upper_barrier / c_i)
                hit_ts = timestamps[idx]
                break
            elif l_val <= lower_barrier:
                hit = 0  # stop loss barrier hit
                ret_val = np.log(lower_barrier / c_i)
                hit_ts = timestamps[idx]
                break

        if hit == 1:
            # End of horizon (timeout / vertical barrier)
            ret_val = np.log(close[i + horizon_bars] / c_i)
            hit_ts = timestamps[i + horizon_bars]

        targets[i] = hit
        returns[i] = ret_val
        t1_list[i] = hit_ts

    # Mask trailing horizon bars
    targets[-horizon_bars:] = -1
    out["target"] = targets
    out["fwd_return"] = returns
    out["t1"] = pd.Series(t1_list, index=out.index)
    out["tp_dist"] = tp_dists
    out["sl_dist"] = sl_dists

    return out
