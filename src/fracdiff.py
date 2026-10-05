"""Fractional Differentiation (FracDiff) for Financial Machine Learning.

Based on Marcos Lopez de Prado's 'Advances in Financial Machine Learning' (Chapter 5).
Achieves stationarity (passes Augmented Dickey-Fuller test) while preserving
maximum long-term price memory, resolving the classic dilemma between
non-stationary raw prices and memory-wiped integer returns (d=1.0).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def get_weights_ffd(d: float, threshold: float = 1e-3, max_len: int = 120) -> np.ndarray:
    """Generate memory weights for Fixed-Width Window Fractional Differentiation (FFD).

    Args:
        d: Differentiation degree (0 < d < 1).
        threshold: Weight cutoff threshold (default 1e-3).
        max_len: Maximum window length to avoid excessive lookback loss.

    Returns:
        np.ndarray: Array of weights [w_0, w_1, ..., w_{L-1}].
    """
    w = [1.0]
    k = 1
    while k < max_len:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w, dtype=float)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-3,
    max_len: int = 120,
) -> pd.Series:
    """Apply Fixed-Width Window Fractional Differentiation to a price series.

    Ensures zero lookahead bias by convolving only with past prices.

    Args:
        series: Pandas Series of prices (e.g. Close).
        d: Fractional order of differentiation.
        threshold: Weight threshold for window cutoff.
        max_len: Maximum window width.

    Returns:
        pd.Series: Fractionally differentiated series aligned with input index.
    """
    w = get_weights_ffd(d, threshold=threshold, max_len=max_len)
    width = len(w)

    vals = series.values
    if len(vals) < width:
        return pd.Series(np.nan, index=series.index)

    # Vectorized 1D convolution
    convolved = np.convolve(vals, w, mode="valid")

    # Prepend NaNs for warmup period to preserve alignment
    full_vals = np.empty(len(vals), dtype=float)
    full_vals[: width - 1] = np.nan
    full_vals[width - 1 :] = convolved

    return pd.Series(full_vals, index=series.index, name=f"fracdiff_d{d:.2f}")


def find_optimal_d(
    series: pd.Series,
    threshold: float = 1e-3,
    max_len: int = 120,
    p_val_target: float = 0.05,
    d_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Find the minimum differentiation order d* that achieves stationarity.

    Uses Augmented Dickey-Fuller (ADF) test to identify the lowest d
    such that p-value <= p_val_target, preserving maximum correlation with original prices.

    Args:
        series: Price series (e.g. Close).
        threshold: Weight cutoff threshold.
        max_len: Maximum window width.
        p_val_target: Target ADF p-value for stationarity (default 0.05).
        d_grid: Custom grid of d candidates (default np.linspace(0.1, 0.9, 9)).

    Returns:
        Dict with optimal 'd', 'adf_pvalue', 'corr_with_raw', 'window_width', and 'tested_grid'.
    """
    if d_grid is None:
        d_grid = np.linspace(0.1, 0.9, 9)

    clean_series = series.dropna()
    tested_grid = []
    best_d = 0.5  # fallback
    found_stationary = False

    for d in d_grid:
        diff_s = frac_diff_ffd(clean_series, d=float(d), threshold=threshold, max_len=max_len)
        valid = diff_s.dropna()

        if len(valid) < 30:
            continue

        try:
            adf_stat, p_val, _, _, _, _ = adfuller(valid.values, maxlag=1, regression="c")
            corr = float(np.corrcoef(clean_series.iloc[-len(valid) :].values, valid.values)[0, 1])
        except Exception:
            continue

        is_stat = p_val <= p_val_target
        tested_grid.append({
            "d": round(float(d), 2),
            "p_value": round(float(p_val), 4),
            "correlation": round(corr, 4),
            "is_stationary": is_stat,
            "window": len(get_weights_ffd(d, threshold, max_len)),
        })

        if is_stat and not found_stationary:
            best_d = float(d)
            found_stationary = True

    # If no d achieved p_val <= 0.05, pick highest d in grid
    if not found_stationary and tested_grid:
        best_d = min(tested_grid, key=lambda x: x["p_value"])["d"]

    selected_info = next((x for x in tested_grid if abs(x["d"] - best_d) < 1e-4), {})

    return {
        "best_d": round(best_d, 2),
        "is_stationary": selected_info.get("is_stationary", False),
        "adf_pvalue": selected_info.get("p_value", 1.0),
        "correlation": selected_info.get("correlation", 0.0),
        "window_width": selected_info.get("window", 0),
        "tested_grid": tested_grid,
    }


def add_fracdiff_features(
    df: pd.DataFrame,
    price_col: str = "close",
    d: Optional[float] = None,
    threshold: float = 1e-3,
    max_len: int = 120,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute and append stationary Fractional Differentiation features to a DataFrame.

    Generates:
        - `fracdiff_close`: FFD price with memory retention
        - `fracdiff_zscore_24h`: Rolling 24h z-score of fracdiff
        - `fracdiff_momentum_3h`: 3h rate of change of fracdiff
        - `fracdiff_dev_ema`: Deviation of fracdiff from its 12h EMA

    Args:
        df: DataFrame containing the price column.
        price_col: Column name to differentiate (default 'close').
        d: Fixed d order. If None, automatically searches for optimal d*.
        threshold: Weight threshold.
        max_len: Max lookback window.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Augmented DataFrame and metadata dictionary.
    """
    df_out = df.copy()

    if d is None:
        opt = find_optimal_d(df_out[price_col], threshold=threshold, max_len=max_len)
        d_val = opt["best_d"]
        meta = opt
    else:
        d_val = float(d)
        w = get_weights_ffd(d_val, threshold, max_len)
        meta = {"best_d": d_val, "window_width": len(w)}

    fd = frac_diff_ffd(df_out[price_col], d=d_val, threshold=threshold, max_len=max_len)
    df_out["fracdiff_close"] = fd

    # Standardized stationary oscillators
    roll_mean = fd.rolling(window=24, min_periods=12).mean()
    roll_std = fd.rolling(window=24, min_periods=12).std()
    df_out["fracdiff_zscore_24h"] = ((fd - roll_mean) / (roll_std + 1e-8)).fillna(0.0)

    # FracDiff momentum (3-bar difference)
    df_out["fracdiff_momentum_3h"] = (fd - fd.shift(3)).fillna(0.0)

    # FracDiff deviation from rolling EMA
    fd_ema = fd.ewm(span=12, adjust=False).mean()
    df_out["fracdiff_dev_ema"] = ((fd - fd_ema) / (fd_ema.abs() + 1e-8)).fillna(0.0)

    return df_out, meta
