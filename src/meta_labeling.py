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
    taker_fee: float = 0.0005,
    slippage: float = 0.0002,
) -> pd.Series:
    """Generate binary meta-labels (1 = profitable trade net of fees, 0 = loss)."""
    common_idx = sub_df.index.intersection(primary_signals.index)
    aligned_df = sub_df.loc[common_idx]
    aligned_sig = primary_signals.loc[common_idx]

    next_returns = aligned_df["close"].pct_change().shift(-1).fillna(0.0)
    round_trip_cost = (taker_fee + slippage) * 2.0

    meta_labels = pd.Series(-1, index=aligned_df.index, dtype=int)

    for i in range(len(aligned_df) - 1):
        sig = aligned_sig.iloc[i]
        if sig != 0.0:
            ret = next_returns.iloc[i]
            trade_pnl = (sig * ret) - round_trip_cost
            meta_labels.iloc[i] = 1 if trade_pnl > 0.0 else 0

    return meta_labels


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

    return {
        "meta_model": last_meta_model,
        "meta_probs": full_meta_probs,
        "n_trade_samples": n_samples,
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
