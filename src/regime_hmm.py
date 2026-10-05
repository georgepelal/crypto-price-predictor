"""Market Regime Detection & Chop Suppression via Causal Gaussian Mixture Filtering.

Classifies financial market structure into 3 distinct regimes:
- Bull Trending Regime (Positive drift, moderate volatility)
- Bear Trending Regime (Negative drift, moderate/high volatility)
- High-Volatility Chop Regime (Mean-reverting wicks, extreme volatility, false breakout risk)

Applies causal forward estimation without backward smoothing to eliminate lookahead bias.
"""

from typing import Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


class MarketRegimeFilter:
    """3-State Regime Detector (0=Bear Trend, 1=High-Vol Chop, 2=Bull Trend)."""

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="diag",
            reg_covar=1e-2,
            random_state=random_state,
            max_iter=150,
        )
        self.regime_map: Dict[int, str] = {}
        self.is_fitted = False

    def fit(self, feat_df: pd.DataFrame) -> "MarketRegimeFilter":
        """Fit Gaussian Mixture on returns, Parkinson volatility, and OI change."""
        X = self._prepare_features(feat_df)
        X_scaled = self.scaler.fit_transform(X)
        self.gmm.fit(X_scaled)

        # Semantics extraction from cluster centers
        means = self.gmm.means_
        ret_idx = 0
        vol_idx = 1

        # The cluster with highest volatility is CHOP / CRISIS
        chop_cluster = int(np.argmax(means[:, vol_idx]))
        remaining = [i for i in range(self.n_regimes) if i != chop_cluster]

        # Of the remaining two, highest return is BULL, lowest is BEAR
        bull_cluster = remaining[int(np.argmax(means[remaining, ret_idx]))]
        bear_cluster = [i for i in remaining if i != bull_cluster][0]

        self.regime_map = {
            bull_cluster: "bull_trend",
            bear_cluster: "bear_trend",
            chop_cluster: "high_vol_chop",
        }
        self.is_fitted = True
        return self

    def predict_regimes(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate causal posterior probabilities P(Regime | X_t)."""
        if not self.is_fitted:
            self.fit(feat_df)

        X = self._prepare_features(feat_df)
        X_scaled = self.scaler.transform(X)
        probs = self.gmm.predict_proba(X_scaled)

        out = pd.DataFrame(index=feat_df.index)
        for cluster_idx, name in self.regime_map.items():
            out[f"prob_{name}"] = probs[:, cluster_idx]

        # Predicted hard regime label
        hard_clusters = self.gmm.predict(X_scaled)
        out["dominant_regime"] = [self.regime_map.get(c, "unknown") for c in hard_clusters]
        return out

    def evaluate_live_bar(self, feat_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate regime probabilities for the latest single live candle."""
        regimes_df = self.predict_regimes(feat_df)
        latest = regimes_df.iloc[-1]
        return {
            "dominant_regime": latest["dominant_regime"],
            "prob_bull_trend": round(float(latest.get("prob_bull_trend", 0.0)), 4),
            "prob_bear_trend": round(float(latest.get("prob_bear_trend", 0.0)), 4),
            "prob_high_vol_chop": round(float(latest.get("prob_high_vol_chop", 0.0)), 4),
            "is_chop_veto": float(latest.get("prob_high_vol_chop", 0.0)) >= 0.40,
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract standardized inputs for regime clustering."""
        ret = df["close"].pct_change().fillna(0.0).clip(-0.20, 0.20).values

        # Use parkinson volatility if present, else rolling std
        if "parkinson_vol_12" in df.columns:
            vol = df["parkinson_vol_12"].fillna(0.01).clip(1e-4, 0.20).values
        elif "norm_atr" in df.columns:
            vol = df["norm_atr"].fillna(0.01).clip(1e-4, 0.20).values
        else:
            vol = df["close"].pct_change().rolling(12).std().fillna(0.01).clip(1e-4, 0.20).values

        # Open interest change or volume change
        if "oi_change_1h" in df.columns:
            flow = df["oi_change_1h"].fillna(0.0).clip(-0.50, 0.50).values
        elif "log_volume_change" in df.columns:
            flow = df["log_volume_change"].fillna(0.0).clip(-2.0, 2.0).values
        else:
            flow = np.zeros(len(df))

        X = np.column_stack([ret, vol, flow])
        return np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
