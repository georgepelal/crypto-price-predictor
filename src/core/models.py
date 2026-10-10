"""Model factories. Deliberately conservative hyper-parameters, fixed in advance (no tuning on
out-of-sample data)."""

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, random_state: int = 42) -> Any:
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=1000,   # large leaves: financial labels are very noisy
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=10.0,
            n_jobs=4,
            random_state=random_state,
            verbose=-1,
        )
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median", copy=False),
            StandardScaler(copy=False),
            LogisticRegression(C=0.01, max_iter=2000),
        )
    raise ValueError(f"unknown model {name!r}; choose lgbm or logreg")
