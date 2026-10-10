"""Train / load the production model and produce live decisions.

The live path calls exactly the same functions as research: history -> build_dataset ->
model_features -> make_model -> expected_return -> step_position.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.core import history
from src.core.dataset import build_dataset, build_symbol_features, model_features
from src.core.models import make_model
from src.core.strategy import expected_return
from src.core.universe import U20, tradable

# Deployed since 2026-10-10 (experiment 2): train on all 20 coins of U20 (incl. delisted history),
# trade the ones still listed. Previous deployment (2026-10-08..10): universe u3, BTC/ETH/SOL.
DEFAULT_SPEC = {
    "universe": "u20",
    "train_symbols": U20,
    "symbols": tradable(U20),
    "timeframe": "1h",
    "data_start": "2020-01-01",
    "model": "lgbm",
    "horizon": 1,
    "policy": "cost_lf",          # cost-aware long/flat (see docs/RESEARCH_LOG.md for why)
    "cost_per_side": 0.0007,      # 5 bps perp taker + 2 bps slippage
    "retrain_days": 30,
}
MODEL_DIR = Path("models")


def model_path(spec: dict) -> Path:
    return MODEL_DIR / f"core_{spec['model']}_{spec['timeframe']}_h{spec['horizon']}_{spec.get('universe', 'u3')}.joblib"


def train_final(spec: Optional[dict] = None, offline: bool = False, verbose: bool = True) -> Path:
    """Fit on ALL labelled history (same recipe as each walk-forward fold) and save with metadata."""
    spec = {**DEFAULT_SPEC, **(spec or {})}
    train_syms = spec.get("train_symbols", spec["symbols"])
    data = history.update_all(train_syms, spec["timeframe"], spec["data_start"], offline=offline, verbose=verbose)
    # same dtype choice as research.py so the deployed model matches the evaluated one
    ds = build_dataset(data, train_syms, spec["horizon"], spec["timeframe"], float32=len(train_syms) > 5)
    feats = model_features(ds)
    tr = ds[ds["target"].notna()]
    model = make_model(spec["model"])
    model.fit(tr[feats], tr["target"].astype(int))
    meta = {
        **spec,
        "features": feats,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(tr)),
        "train_start": str(tr["timestamp"].min()),
        "train_end": str(tr["timestamp"].max()),
        "train_up_rate": float(tr["target"].mean()),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    path = model_path(spec)
    joblib.dump({"model": model, "meta": meta}, path)
    with open(path.with_suffix(".json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if verbose:
        print(f"[train] saved {path}  rows={len(tr):,}  data to {meta['train_end']}")
    return path


def load_model(spec: Optional[dict] = None) -> dict:
    spec = {**DEFAULT_SPEC, **(spec or {})}
    path = model_path(spec)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python train.py")
    return joblib.load(path)


def model_age_days(bundle: dict) -> float:
    t = pd.Timestamp(bundle["meta"]["trained_at"])
    return (pd.Timestamp.now(tz="UTC") - t).total_seconds() / 86400


def latest_signals(bundle: dict, data: dict, symbols: Optional[List[str]] = None) -> List[Dict]:
    """Prediction for the most recent CLOSED bar of every symbol."""
    meta = bundle["meta"]
    out = []
    for sym in symbols or meta["symbols"]:
        f = build_symbol_features(data, sym, meta["timeframe"])
        row = f.iloc[[-1]]
        missing = [c for c in meta["features"] if c not in row.columns]
        if missing:
            raise ValueError(f"live features missing {missing} - train/serve mismatch")
        p = float(bundle["model"].predict_proba(row[meta["features"]])[:, 1][0])
        vol = float(row["vol_168"].iloc[0])
        mu = float(expected_return(pd.Series([p]), pd.Series([vol]), meta["horizon"]).iloc[0])
        out.append({
            "symbol": sym,
            "bar_open": row.index[-1],
            "bar_close_price": float(row["_close"].iloc[0]),
            "prob_up": p,
            "vol_168": vol,
            "exp_ret": mu,
            "hurdle": 2 * meta["cost_per_side"],
        })
    return out
