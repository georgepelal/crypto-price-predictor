"""Model training and Walk-Forward cross-validation engine.

Supports LightGBM and scikit-learn HistGradientBoosting with automatic fallback.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib


def get_model_instance(model_type: str = "auto", random_state: int = 42, **kwargs) -> Tuple[Any, str]:
    """Instantiate classifier with automatic fallback between LightGBM, HistGradientBoosting, and Ensemble."""
    if model_type == "ensemble":
        m1 = HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 180),
            learning_rate=kwargs.get("learning_rate", 0.03),
            max_depth=kwargs.get("max_depth", 5),
            random_state=random_state,
        )
        m2 = RandomForestClassifier(
            n_estimators=kwargs.get("rf_estimators", 80),
            max_depth=kwargs.get("max_depth", 6),
            random_state=random_state,
            n_jobs=-1,
        )
        m3 = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=400, C=1.0, random_state=random_state),
        )
        ens = VotingClassifier(
            estimators=[("hgb", m1), ("rf", m2), ("lr", m3)],
            voting="soft",
            weights=[0.50, 0.30, 0.20],
        )
        return ens, "Heterogeneous Ensemble (HGB+RF+LR)"

    if model_type in ["lightgbm"]:
        try:
            import lightgbm as lgb
            params = {
                "n_estimators": kwargs.get("max_iter", 200),
                "learning_rate": kwargs.get("learning_rate", 0.03),
                "max_depth": kwargs.get("max_depth", 5),
                "random_state": random_state,
                "verbosity": -1,
            }
            test_model = lgb.LGBMClassifier(**params)
            test_model.fit([[0, 1], [1, 0]], [0, 1])
            return lgb.LGBMClassifier(**params), "LightGBM"
        except Exception as e:
            print(f"[!] LightGBM failed to load ({e}). Falling back to HistGradientBoosting...")

    # Native sklearn HistGradientBoosting (zero C++ dylib dependencies, same histogram tree algorithm)
    params = {
        "max_iter": kwargs.get("max_iter", 200),
        "learning_rate": kwargs.get("learning_rate", 0.03),
        "max_depth": kwargs.get("max_depth", 5),
        "random_state": random_state,
    }
    return HistGradientBoostingClassifier(**params), "HistGradientBoosting"


class WalkForwardValidator:
    """Expanding-window Walk-Forward validator for time series."""

    def __init__(self, n_splits: int = 5, min_train_size: int = 500):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(self, n_samples: int):
        """Yield (train_indices, test_indices) tuples."""
        if n_samples <= self.min_train_size:
            raise ValueError(f"Sample size ({n_samples}) must exceed min_train_size ({self.min_train_size})")

        available_for_test = n_samples - self.min_train_size
        test_size = available_for_test // self.n_splits

        for fold in range(self.n_splits):
            train_end = self.min_train_size + (fold * test_size)
            test_end = train_end + test_size if fold < self.n_splits - 1 else n_samples

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            yield train_idx, test_idx


def train_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 500,
    model_type: str = "auto",
    calibrate_probabilities: bool = True,
    **model_params,
) -> Dict[str, Any]:
    """Train and evaluate model strictly using Walk-Forward Cross Validation.

    Returns:
        dict containing out-of-sample predictions, probabilities, metrics, and trained final model.
    """
    validator = WalkForwardValidator(n_splits=n_splits, min_train_size=min_train_size)

    n_samples = len(X)
    classes = np.unique(y)
    n_classes = len(classes)

    # Containers for out-of-sample predictions
    oof_preds = pd.Series(index=X.index, dtype=float)
    oof_probs = pd.DataFrame(index=X.index, columns=[f"prob_class_{c}" for c in classes], dtype=float)
    fold_metrics = []

    last_model = None
    model_name = ""

    for fold, (train_idx, test_idx) in enumerate(validator.split(n_samples)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        base_model, raw_model_name = get_model_instance(model_type, **model_params)
        if calibrate_probabilities and len(np.unique(y_train)) >= 2:
            from sklearn.calibration import CalibratedClassifierCV
            cv_folds = min(3, max(2, len(X_train) // 80))
            model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=cv_folds)
            model_name = f"{raw_model_name} (Calibrated Sigmoid)"
        else:
            model = base_model
            model_name = raw_model_name

        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)

        oof_preds.iloc[test_idx] = test_preds
        oof_probs.iloc[test_idx, :] = test_probs

        acc = accuracy_score(y_test, test_preds)
        fold_metrics.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": acc,
        })
        last_model = model

    # Evaluate overall out-of-sample performance
    valid_mask = ~oof_preds.isna()
    valid_y = y[valid_mask]
    valid_preds = oof_preds[valid_mask].astype(int)

    overall_acc = accuracy_score(valid_y, valid_preds)
    report = classification_report(valid_y, valid_preds, output_dict=True, zero_division=0)

    # Feature Importance
    feature_importances = {}
    if hasattr(last_model, "feature_importances_"):
        for col, imp in zip(X.columns, last_model.feature_importances_):
            feature_importances[col] = float(imp)
        feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    else:
        try:
            from sklearn.inspection import permutation_importance
            sample_size = min(len(X_test), 200)
            perm = permutation_importance(
                last_model,
                X_test.iloc[-sample_size:],
                y_test.iloc[-sample_size:],
                n_repeats=3,
                random_state=42,
                n_jobs=1,
            )
            for col, imp in zip(X.columns, perm.importances_mean):
                feature_importances[col] = max(0.0, float(imp))
            feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            print(f"[!] Warning: Could not compute permutation importance: {e}")

    return {
        "model_name": model_name,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "fold_metrics": fold_metrics,
        "overall_accuracy": overall_acc,
        "classification_report": report,
        "feature_importances": feature_importances,
        "final_model": last_model,
        "valid_mask": valid_mask,
    }
