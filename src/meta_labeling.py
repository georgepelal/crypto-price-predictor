"""Meta-Labeling & Dynamic Bet Sizing Engine (Marcos Lopez de Prado).

Separates the side decision (Long / Short) from the size decision (Probability of Profit).
Trains a secondary meta-model to predict trade success and dynamically size positions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

from src.model import train_walk_forward
from src.backtest import run_backtest


def create_meta_labels(
    sub_df: pd.DataFrame,
    primary_signals: pd.Series,
    horizon_bars: int = 6,
    profit_mult: float = 1.5,
    stop_mult: float = 1.0,
    taker_fee: float = 0.0005,
    slippage: float = 0.0002,
) -> pd.Series:
    """Generate binary meta-labels (1 = profitable trade net of fees, 0 = loss).

    Simulates the actual trade path using dynamic ATR profit-taking and stop-loss barriers.
    """
    common_idx = sub_df.index.intersection(primary_signals.index)
    aligned_df = sub_df.loc[common_idx]
    aligned_sig = primary_signals.loc[common_idx]

    close = aligned_df["close"].values
    high = aligned_df["high"].values
    low = aligned_df["low"].values
    atr = aligned_df["norm_atr"].values if "norm_atr" in aligned_df.columns else np.full(len(aligned_df), 0.01)
    round_trip_cost = (taker_fee + slippage) * 2.0

    n = len(aligned_df)
    meta_labels = pd.Series(-1, index=aligned_df.index, dtype=int)

    for i in range(n - horizon_bars):
        sig = aligned_sig.iloc[i]
        if sig == 0.0 or np.isnan(sig):
            continue

        c_i = close[i]
        curr_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 0.01

        tp_target = c_i * (1.0 + (profit_mult * curr_atr)) if sig > 0 else c_i * (1.0 - (profit_mult * curr_atr))
        sl_target = c_i * (1.0 - (stop_mult * curr_atr)) if sig > 0 else c_i * (1.0 + (stop_mult * curr_atr))

        is_success = 0
        for step in range(1, horizon_bars + 1):
            idx = i + step
            h_val = high[idx]
            l_val = low[idx]

            if sig > 0:  # Long
                if l_val <= sl_target:
                    is_success = 0
                    break
                elif h_val >= tp_target:
                    is_success = 1
                    break
            else:  # Short
                if h_val >= sl_target:
                    is_success = 0
                    break
                elif l_val <= tp_target:
                    is_success = 1
                    break

        # If neither barrier was touched before timeout, evaluate net exit return
        if step == horizon_bars and is_success == 0:
            c_exit = close[i + horizon_bars]
            exit_ret = ((c_exit / c_i) - 1.0) if sig > 0 else (1.0 - (c_exit / c_i))
            if exit_ret - round_trip_cost > 0.0:
                is_success = 1

        meta_labels.iloc[i] = is_success

    return meta_labels


def compute_expected_value(
    p_up: float,
    p_down: float,
    atr_val: float,
    profit_mult: float = 1.5,
    stop_mult: float = 1.0,
    round_trip_friction: float = 0.0014,
) -> Dict[str, float]:
    """Calculate mathematical Expected Value for Long and Short trading setups."""
    atr_clean = max(atr_val, 0.005)
    tp_distance = profit_mult * atr_clean
    sl_distance = stop_mult * atr_clean

    ev_long = (p_up * tp_distance) - (p_down * sl_distance) - round_trip_friction
    ev_short = (p_down * tp_distance) - (p_up * sl_distance) - round_trip_friction

    return {
        "ev_long": float(ev_long),
        "ev_short": float(ev_short),
        "best_ev": float(max(ev_long, ev_short)),
        "tp_distance": float(tp_distance),
        "sl_distance": float(sl_distance),
    }


def train_meta_model(
    X: pd.DataFrame,
    meta_labels: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 50,
) -> Dict[str, Any]:
    """Train Walk-Forward binary meta-model on trade instances."""
    common_idx = X.index.intersection(meta_labels.index)
    X_aligned = X.loc[common_idx]
    labels_aligned = meta_labels.loc[common_idx]

    trade_mask = labels_aligned != -1
    X_trades = X_aligned.loc[trade_mask]
    y_trades = labels_aligned.loc[trade_mask]

    n_samples = len(X_trades)
    if n_samples < 10:
        return {
            "error": "Insufficient trade samples to train meta-model.",
            "meta_probs": pd.Series(0.5, index=common_idx),
        }

    full_meta_probs = pd.Series(0.5, index=common_idx, dtype=float)

    # Expanding window
    step = max(5, (n_samples - min_train_size) // n_splits)
    if step <= 0:
        step = 1

    oof_meta_probs = pd.Series(index=X_trades.index, dtype=float)

    last_meta_model = None
    for fold in range(n_splits):
        train_end = min(min_train_size + (fold * step), n_samples - 1)
        test_end = min(train_end + step, n_samples) if fold < n_splits - 1 else n_samples

        if train_end >= test_end:
            break

        X_tr = X_trades.iloc[:train_end]
        y_tr = y_trades.iloc[:train_end]
        X_te = X_trades.iloc[train_end:test_end]
        y_te = y_trades.iloc[train_end:test_end]

        # Check at least 2 classes in train
        if len(np.unique(y_tr)) < 2:
            oof_meta_probs.iloc[train_end:test_end] = 0.5
            continue

        meta_clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.03, max_depth=3, random_state=42
        )
        meta_clf.fit(X_tr, y_tr)

        te_probs = meta_clf.predict_proba(X_te)[:, 1]
        oof_meta_probs.iloc[train_end:test_end] = te_probs
        last_meta_model = meta_clf

    valid_trades = ~oof_meta_probs.isna()
    full_meta_probs.loc[oof_meta_probs[valid_trades].index] = oof_meta_probs[valid_trades]

    # Train final model on entire trade dataset for live production inference
    final_meta_model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.03, max_depth=3, random_state=42
    )
    if len(np.unique(y_trades)) >= 2:
        final_meta_model.fit(X_trades, y_trades)
    else:
        final_meta_model = last_meta_model

    # Out-of-fold diagnostic metrics
    precision, acc = 0.5, 0.5
    if valid_trades.sum() > 5:
        y_eval = y_trades.loc[valid_trades]
        p_eval = oof_meta_probs.loc[valid_trades]
        pred_eval = (p_eval >= 0.50).astype(int)
        precision = float(precision_score(y_eval, pred_eval, zero_division=0))
        acc = float(accuracy_score(y_eval, pred_eval))

    return {
        "meta_model": final_meta_model,
        "meta_probs": full_meta_probs,
        "n_trade_samples": n_samples,
        "precision": precision,
        "accuracy": acc,
    }


def compute_kelly_positions(
    primary_signals: pd.Series,
    meta_probabilities: pd.Series,
    min_prob_threshold: float = 0.50,
) -> pd.Series:
    """Calculate fractional bet sizing using Meta-Model probability.

    If P(Win) <= min_prob_threshold: Bet Size = 0 (Trade Vetoed)
    If P(Win) > min_prob_threshold: Scale bet smoothly up to 100% position.
    """
    positions = pd.Series(0.0, index=primary_signals.index)

    for i in range(len(primary_signals)):
        sig = primary_signals.iloc[i]
        if sig == 0.0:
            positions.iloc[i] = 0.0
            continue

        p_win = meta_probabilities.iloc[i]
        if p_win <= min_prob_threshold:
            positions.iloc[i] = 0.0  # Vetoed trade
        else:
            # Linear scaling from threshold to 0.85
            size_fraction = min(1.0, (p_win - min_prob_threshold) / 0.35)
            positions.iloc[i] = sig * size_fraction

    return positions
