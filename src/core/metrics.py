"""Evaluation metrics: classification quality, information coefficient, trading performance and
block-bootstrap confidence intervals."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def classification_metrics(oos: pd.DataFrame) -> Dict[str, float]:
    d = oos.dropna(subset=["target"])
    y = d["target"].astype(int).values
    p = np.clip(d["prob_up"].values, 1e-6, 1 - 1e-6)
    prior = np.clip(d["train_base_rate"].values, 1e-6, 1 - 1e-6)  # what you'd know without a model
    ll_model = log_loss(y, p, labels=[0, 1])
    ll_prior = log_loss(y, prior, labels=[0, 1])
    return {
        "n": int(len(d)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(((p > 0.5) == y).mean()),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
        "logloss": float(ll_model),
        "logloss_prior": float(ll_prior),
        "logloss_skill_pct": float(100 * (1 - ll_model / ll_prior)),   # >0 means better than prior
        "brier": float(brier_score_loss(y, p)),
        "up_rate": float(y.mean()),
    }


def information_coefficient(oos: pd.DataFrame, freq: str = "MS") -> Dict[str, float]:
    """Monthly Spearman rank correlation between predicted edge and realised forward return.
    Mean IC with a t-stat across months (months are ~independent)."""
    d = oos.dropna(subset=["fwd_ret"]).copy()
    d["month"] = d["timestamp"].dt.tz_localize(None).dt.to_period("M")
    ics = d.groupby("month")[["prob_up", "fwd_ret"]].apply(lambda g: stats.spearmanr(g["prob_up"], g["fwd_ret"])[0] if len(g) > 50 else np.nan)
    ics = ics.dropna()
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if len(ics) > 2 else np.nan
    return {"ic_mean": float(ics.mean()), "ic_t": float(t), "ic_pos_months_pct": float(100 * (ics > 0).mean()),
            "months": int(len(ics))}


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def trading_metrics(hourly: pd.Series, per_symbol: Optional[Dict[str, pd.DataFrame]] = None,
                    bootstrap: int = 2000, block_days: int = 20, seed: int = 0) -> Dict[str, float]:
    daily = (1 + hourly).groupby(hourly.index.floor("D")).prod() - 1
    eq = (1 + daily).cumprod()
    years = len(daily) / 365.0
    sharpe = daily.mean() / (daily.std(ddof=1) + 1e-12) * np.sqrt(365)
    out = {
        "total_return_pct": float(100 * (eq.iloc[-1] - 1)),
        "cagr_pct": float(100 * (eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1)),
        "ann_vol_pct": float(100 * daily.std(ddof=1) * np.sqrt(365)),
        "sharpe": float(sharpe),
        "max_dd_pct": float(100 * _max_drawdown(eq)),
        "days": int(len(daily)),
    }
    if per_symbol:
        out["turnover_per_year"] = float(np.mean([d["turnover"].sum() for d in per_symbol.values()]) / max(years, 1e-9))
        out["time_in_market_pct"] = float(100 * np.mean([(d["pos"].abs() > 1e-9).mean() for d in per_symbol.values()]))
        out["cost_drag_pct_per_year"] = float(100 * np.mean([d["cost"].sum() for d in per_symbol.values()]) / max(years, 1e-9))
        out["funding_pct_per_year"] = float(100 * np.mean([d["funding"].sum() for d in per_symbol.values()]) / max(years, 1e-9))
        gross_h = pd.DataFrame({s: d["gross"] for s, d in per_symbol.items()}).fillna(0.0).mean(axis=1)
        gross_d = (1 + gross_h).groupby(gross_h.index.floor("D")).prod() - 1
        out["gross_sharpe"] = float(gross_d.mean() / (gross_d.std(ddof=1) + 1e-12) * np.sqrt(365))
        out["gross_return_pct_per_year"] = float(100 * gross_d.mean() * 365)
    if bootstrap and len(daily) > 3 * block_days:
        rng = np.random.default_rng(seed)
        x = daily.values
        n = len(x)
        nblocks = int(np.ceil(n / block_days))
        sh = np.empty(bootstrap)
        for b in range(bootstrap):
            starts = rng.integers(0, n - block_days, nblocks)
            sample = np.concatenate([x[s:s + block_days] for s in starts])[:n]
            sh[b] = sample.mean() / (sample.std(ddof=1) + 1e-12) * np.sqrt(365)
        out["sharpe_ci_low"] = float(np.percentile(sh, 2.5))
        out["sharpe_ci_high"] = float(np.percentile(sh, 97.5))
        out["p_sharpe_le_0"] = float((sh <= 0).mean())
    return out


def yearly_returns(hourly: pd.Series) -> pd.Series:
    yr = (1 + hourly).groupby(hourly.index.year).prod() - 1
    return 100 * yr
