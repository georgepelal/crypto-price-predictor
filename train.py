"""Train the production model on all available history.

    .venv/bin/python train.py            # refresh data, train, save models/core_lgbm_1h_h1.joblib
    .venv/bin/python train.py --offline  # use cached data only
"""
import argparse

from src.core.deploy import DEFAULT_SPEC, train_final

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--model", default=DEFAULT_SPEC["model"])
    ap.add_argument("--horizon", type=int, default=DEFAULT_SPEC["horizon"])
    a = ap.parse_args()
    train_final({"model": a.model, "horizon": a.horizon}, offline=a.offline)
