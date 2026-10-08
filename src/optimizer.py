"""Hyperparameter and Trading Strategy Optimization Engine using Optuna.

Optimizes both model learning parameters and strategy execution thresholds
directly toward maximizing Out-of-Sample Sharpe Ratio and Profit Factor net of fees.
"""

from typing import Dict, Any, Tuple
import optuna
import pandas as pd
import numpy as np

from src.features import generate_crypto_features
from src.labeling import make_directional_labels
from src.model import train_walk_forward
from src.backtest import run_backtest

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_pipeline(
    raw_df: pd.DataFrame,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    n_trials: int = 30,
    n_splits: int = 5,
    model_type: str = "auto",
) -> Dict[str, Any]:
    """Run Bayesian optimization via Optuna to find best parameters for risk-adjusted returns."""
    print(f"[*] Starting Optuna Bayesian Optimization ({n_trials} trials, model: {model_type}) for {symbol} ({timeframe})...")

    # Pre-generate features once to maximize optimization speed
    feat_df = generate_crypto_features(raw_df)

    def objective(trial: optuna.Trial) -> float:
        # 1. Hyperparameters to search
        atr_mult = trial.suggest_float("atr_multiplier", 0.25, 0.55, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.07, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 6)
        max_iter = trial.suggest_int("max_iter", 60, 180, step=20)
        confidence = trial.suggest_float("confidence_threshold", 0.33, 0.45, step=0.02)

        extra_kwargs = {}
        if model_type == "ensemble":
            w_hgb = trial.suggest_float("w_hgb", 0.3, 0.7)
            w_rf = trial.suggest_float("w_rf", 0.1, 0.5)
            w_lr = trial.suggest_float("w_lr", 0.1, 0.4)
            w_sum = w_hgb + w_rf + w_lr
            weights = [w_hgb / w_sum, w_rf / w_sum, w_lr / w_sum]
            rf_est = trial.suggest_int("rf_estimators", 30, 80, step=10)
            lr_c = trial.suggest_float("lr_c", 0.2, 5.0, log=True)
            extra_kwargs["ensemble_weights"] = weights
            extra_kwargs["rf_estimators"] = rf_est
            extra_kwargs["lr_c"] = lr_c

        # 2. Dynamic labeling with trial ATR hurdle
        labeled_df = make_directional_labels(feat_df, forward_bars=1, atr_multiplier=atr_mult, fee_buffer=0.0005)

        feature_cols = [
            c for c in labeled_df.columns
            if c not in ["open", "high", "low", "close", "volume", "target", "fwd_return", "threshold"]
        ]
        clean_df = labeled_df[labeled_df["target"] != -1].dropna(subset=feature_cols + ["target"])

        if len(clean_df) < 150:
            return -10.0

        X = clean_df[feature_cols]
        y = clean_df["target"].astype(int)

        effective_min_train = max(80, int(len(X) * 0.4))

        # 3. Fast Walk-Forward CV
        try:
            train_results = train_walk_forward(
                X,
                y,
                n_splits=n_splits,
                min_train_size=effective_min_train,
                model_type=model_type,
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_iter=max_iter,
                **extra_kwargs,
            )
        except Exception:
            return -10.0

        # 4. Realistic Backtest
        bt = run_backtest(
            df=clean_df,
            predictions=train_results["oof_preds"],
            probabilities=train_results["oof_probs"],
            confidence_threshold=confidence,
            bars_per_year=8760 if timeframe == "1h" else 35040,
        )

        metrics = bt.get("metrics", {})
        trades = metrics.get("total_trades", 0)
        sharpe = metrics.get("sharpe_ratio", 0.0)
        ret_pct = metrics.get("total_return_pct", 0.0)
        mdd_pct = metrics.get("max_drawdown_pct", -10.0)
        acc = train_results.get("overall_accuracy", 0.33)

        if trades < 2:
            return -5.0 + (trades * 0.5)

        score = sharpe + (ret_pct / 5.0) - (abs(mdd_pct) * 0.1) + ((acc - 0.50) * 10.0)
        return float(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    print(f"[+] Optimization finished! Best Trial Score: {best_trial.value:.3f}")
    print(f"[*] Optimal Parameters: {best_trial.params}")

    return {
        "best_score": best_trial.value,
        "best_params": best_trial.params,
        "study": study,
    }
