"""Rolling-origin (walk-forward) evaluation.

For every test block [T_k, T_k + retrain_every):
  * train on rows whose label is fully known before T_k  (timestamp <= T_k - horizon bars),
    optionally limited to a trailing window,
  * predict the block. Nothing from the test block (or later) touches the model, the features'
    normalisation, or any threshold.
"""

import time
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.models import make_model


def walk_forward(
    ds: pd.DataFrame,
    features: List[str],
    horizon: int,
    model_name: str = "lgbm",
    test_start: str = "2023-01-01",
    retrain_every: str = "30D",
    train_window: Optional[str] = None,
    bar: str = "1h",
    verbose: bool = True,
) -> pd.DataFrame:
    ts = ds["timestamp"]
    t0 = pd.Timestamp(test_start, tz="UTC")
    t_end = ts.max() + pd.Timedelta(bar)
    step = pd.Timedelta(retrain_every)
    purge = pd.Timedelta(bar) * horizon
    blocks = []
    k = t0
    while k < t_end:
        blocks.append((k, min(k + step, t_end)))
        k += step

    preds = []
    started = time.time()
    for i, (b0, b1) in enumerate(blocks):
        train_mask = (ts <= b0 - purge - pd.Timedelta(bar)) & ds["target"].notna()
        if train_window:
            train_mask &= ts >= b0 - pd.Timedelta(train_window)
        test_mask = (ts >= b0) & (ts < b1)
        if test_mask.sum() == 0:
            continue
        tr = ds.loc[train_mask]
        model = make_model(model_name)
        model.fit(tr[features], tr["target"].astype(int))
        te = ds.loc[test_mask]
        p = model.predict_proba(te[features])[:, 1]
        out = te[["timestamp", "symbol", "fwd_ret", "ret_next", "target"]].copy()
        out["prob_up"] = p
        out["train_base_rate"] = float(tr["target"].mean())
        out["fold"] = i
        preds.append(out)
        if verbose:
            print(f"  [{model_name} h={horizon}] fold {i + 1}/{len(blocks)} "
                  f"train={len(tr):,} test={len(te):,} {b0:%Y-%m-%d} ({time.time() - started:.0f}s)",
                  flush=True)
    return pd.concat(preds, ignore_index=True)
