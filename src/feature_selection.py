"""Feature Selection & Redundancy Pruning Module for Financial Machine Learning.

Implements two-stage institutional feature selection:
1. Collinearity & Redundancy Filtering (Correlation Clustering):
   Removes redundant features with correlation |r| > threshold, keeping the one with higher predictive signal.
2. Permutation Importance & Gradient Boosting Ranking:
   Evaluates true out-of-sample importance, pruning noisy and overfitted features.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def remove_collinear_features(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_threshold: float = 0.80,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated redundant features while preserving the one with higher signal.

    Args:
        X: Feature matrix.
        y: Target series.
        correlation_threshold: Absolute correlation threshold (default 0.80).

    Returns:
        Tuple of (Filtered DataFrame, list of dropped redundant features).
    """
    corr_matrix = X.corr(method="spearman").abs()

    # Pre-calculate simple correlation of each feature with target y
    target_corrs = {col: abs(X[col].corr(y.astype(float), method="spearman")) for col in X.columns}

    dropped_features = set()
    cols = list(X.columns)

    for i in range(len(cols)):
        col_i = cols[i]
        if col_i in dropped_features:
            continue

        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            if col_j in dropped_features:
                continue

            if corr_matrix.loc[col_i, col_j] >= correlation_threshold:
                # Keep the one with higher correlation to target
                if target_corrs.get(col_i, 0.0) >= target_corrs.get(col_j, 0.0):
                    dropped_features.add(col_j)
                else:
                    dropped_features.add(col_i)
                    break

    selected_cols = [c for c in X.columns if c not in dropped_features]
    return X[selected_cols], sorted(list(dropped_features))


def rank_features_by_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
    random_state: int = 42,
) -> Tuple[List[str], Dict[str, float]]:
    """Train a gradient booster to compute permutation feature importance scores.

    Args:
        X: Feature matrix.
        y: Target series.
        top_k: Number of top features to retain.
        random_state: Random seed.

    Returns:
        Tuple of (Top-K feature names, dict of importance scores).
    """
    from sklearn.inspection import permutation_importance

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state,
        early_stopping=True,
    )

    # Train on training partition (first 80% to preserve time structure)
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model.fit(X_train, y_train)

    # Evaluate permutation importance on validation partition
    perm = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )

    importances = dict(zip(X.columns, perm.importances_mean))

    # Sort descending
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:top_k]]

    return top_features, dict(sorted_features)


def select_optimal_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 20,
    correlation_threshold: float = 0.80,
) -> Tuple[List[str], Dict[str, Any]]:
    """Complete two-stage feature selection pipeline: De-correlation + Permutation Ranking.

    Args:
        X: Full feature matrix.
        y: Target labels.
        top_k: Desired number of top features.
        correlation_threshold: Redundancy cutoff.

    Returns:
        Tuple of (List of selected feature column names, metadata report dict).
    """
    initial_count = X.shape[1]

    # Stage 1: Remove redundant collinear columns
    X_clean, dropped_collinear = remove_collinear_features(
        X, y, correlation_threshold=correlation_threshold
    )
    after_decorr_count = X_clean.shape[1]

    # Stage 2: Rank remaining features by predictive value
    effective_k = min(top_k, after_decorr_count)
    selected_cols, all_importances = rank_features_by_importance(
        X_clean, y, top_k=effective_k
    )

    report = {
        "initial_features": initial_count,
        "dropped_collinear_count": len(dropped_collinear),
        "dropped_collinear_list": dropped_collinear,
        "selected_features_count": len(selected_cols),
        "selected_features": selected_cols,
        "feature_importances": {k: round(float(v), 5) for k, v in all_importances.items() if k in selected_cols},
    }

    return selected_cols, report
