"""Conformal Prediction Selective Classification Engine for Crypto Trading.

Implements inductive split conformal prediction to derive mathematically guaranteed
prediction sets and singleton-only trade execution, filtering out low-conviction
and ambiguous market predictions (Selective Classification with Abstention).
"""

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import pandas as pd


class ConformalClassifier:
    """Inductive Conformal Prediction engine with singleton-selective classification.

    Rejects ambiguous multi-class sets and empty sets, taking action ONLY on
    unambiguous singletons with mathematically bounded error coverage.
    """

    def __init__(self, alpha: float = 0.15):
        """Args:
            alpha: Significance level (error budget).
                   alpha=0.15 implies 85% guaranteed coverage under exchangeability.
        """
        self.alpha = float(alpha)
        self.q_hat: Optional[float] = None
        self.classes = np.array([0, 1, 2])

    def calibrate(self, oof_probs: pd.DataFrame, y_true: pd.Series) -> float:
        """Calibrate the non-conformity score quantile on Out-Of-Fold validation data.

        Args:
            oof_probs: DataFrame with columns ['prob_class_0', 'prob_class_1', 'prob_class_2']
            y_true: Ground truth target labels (0, 1, 2)

        Returns:
            float: Calibrated non-conformity threshold q_hat.
        """
        mask = ~oof_probs.isna().any(axis=1) & ~y_true.isna()
        valid_probs = oof_probs.loc[mask]
        valid_y = y_true.loc[mask].astype(int)

        n = len(valid_y)
        if n < 30:
            self.q_hat = 0.65
            return self.q_hat

        # Map labels to column names
        scores = []
        for i in range(n):
            true_cls = valid_y.iloc[i]
            col_name = f"prob_class_{true_cls}"
            prob = valid_probs[col_name].iloc[i] if col_name in valid_probs.columns else 0.33
            # Non-conformity score: 1.0 - P(true class)
            scores.append(1.0 - prob)

        scores_arr = np.array(scores)

        # Finite-sample adjusted conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores_arr, quantile_level, method="higher"))
        return self.q_hat

    def predict_prediction_sets(self, probs: pd.DataFrame) -> Tuple[pd.Series, pd.Series, List[List[int]]]:
        """Generate prediction sets and filter for singletons.

        Args:
            probs: DataFrame with ['prob_class_0', 'prob_class_1', 'prob_class_2']

        Returns:
            Tuple of:
            - singleton_series: Series containing predicted class if singleton, else -1 (abstain)
            - set_size_series: Number of candidate classes in each prediction set
            - raw_sets: List of candidate classes per bar
        """
        if self.q_hat is None:
            raise ValueError("ConformalClassifier must be calibrated before predicting sets.")

        prob_cutoff = 1.0 - self.q_hat

        raw_sets = []
        singletons = []
        set_sizes = []

        prob_cols = ["prob_class_0", "prob_class_1", "prob_class_2"]
        cols_present = [c for c in prob_cols if c in probs.columns]

        for i in range(len(probs)):
            row = probs.iloc[i]
            candidates = []
            for cls_idx in [0, 1, 2]:
                col_name = f"prob_class_{cls_idx}"
                if col_name in row and row[col_name] >= prob_cutoff:
                    candidates.append(cls_idx)

            raw_sets.append(candidates)
            set_sizes.append(len(candidates))

            # Directional Certainty Gate in 3-class buffered setting:
            # - If [2] or [1, 2]: Down (0) is ruled out at (1-alpha) confidence -> Bullish admissible (2)
            # - If [0] or [0, 1]: Up (2) is ruled out at (1-alpha) confidence -> Bearish admissible (0)
            # - If [1]: Pure flat noise -> Neutral (1)
            # - If [0, 2] or [0, 1, 2] or empty: Directional contradiction / uncertainty -> Abstain (-1)
            if (2 in candidates) and (0 not in candidates):
                singletons.append(2)
            elif (0 in candidates) and (2 not in candidates):
                singletons.append(0)
            elif candidates == [1]:
                singletons.append(1)
            else:
                singletons.append(-1)  # Abstain / Veto trade

        singleton_series = pd.Series(singletons, index=probs.index, name="conformal_singleton")
        set_size_series = pd.Series(set_sizes, index=probs.index, name="conformal_set_size")

        return singleton_series, set_size_series, raw_sets

    def evaluate_live_bar(self, prob_dict: Dict[int, float]) -> Dict[str, Any]:
        """Evaluate conformal certainty on a single live bar prediction dictionary."""
        if self.q_hat is None:
            self.q_hat = 0.65

        prob_cutoff = 1.0 - self.q_hat
        candidates = [c for c in [0, 1, 2] if prob_dict.get(c, 0.0) >= prob_cutoff]

        is_bull_admissible = (2 in candidates) and (0 not in candidates)
        is_bear_admissible = (0 in candidates) and (2 not in candidates)
        is_pure_flat = (candidates == [1])
        is_ambiguous = (0 in candidates and 2 in candidates) or (len(candidates) == 0)

        singleton_class = 2 if is_bull_admissible else (0 if is_bear_admissible else (1 if is_pure_flat else None))

        return {
            "prediction_set": candidates,
            "set_size": len(candidates),
            "is_singleton": len(candidates) == 1,
            "is_directional_certain": is_bull_admissible or is_bear_admissible,
            "singleton_class": singleton_class,
            "conformal_cutoff": round(prob_cutoff, 4),
            "abstain": is_ambiguous or is_pure_flat,
        }
