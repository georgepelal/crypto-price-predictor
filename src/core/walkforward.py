"""Rolling-origin (walk-forward) evaluation.

For every test block [T_k, T_k + retrain_every):
  * train on rows whose label is fully known before T_k  (timestamp <= T_k - horizon bars),
    optionally limited to a trailing window,
  * predict the block. Nothing from the test block (or later) touches the model, the features'
    normalisation, or any threshold.
"""

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.models import make_model


class TimeBudgetExceeded(RuntimeError):
    pass


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
    fold_cache: Optional[Path] = None,
    max_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """fold_cache: directory where each fold's predictions are saved; existing folds are reused, so a
    long run can be split across several invocations. max_seconds: stop starting new folds after this
    long and raise TimeBudgetExceeded (re-run to continue)."""
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
    if fold_cache is not None:
        Path(fold_cache).mkdir(parents=True, exist_ok=True)
    for i, (b0, b1) in enumerate(blocks):
        cache_file = Path(fold_cache) / f"fold_{i:03d}_{b0:%Y%m%d}.pkl" if fold_cache is not None else None
        if cache_file is not None and cache_file.exists():
            preds.append(pd.read_pickle(cache_file))
            continue
        if max_seconds is not None and time.time() - started > max_seconds:
            raise TimeBudgetExceeded(f"{i}/{len(blocks)} folds done; re-run to continue")
        train_mask = (ts <= b0 - purge - pd.Timedelta(bar)) & ds["target"].notna()
        if train_window:
            train_mask &= ts >= b0 - pd.Timedelta(train_window)
        test_mask = (ts >= b0) & (ts < b1)
        if test_mask.sum() == 0:
            continue
        # Slice only the needed columns (one copy) to keep memory low on large universes.
        X_tr = ds.loc[train_mask, features]
        y_tr = ds.loc[train_mask, "target"].astype(int)
        model = make_model(model_name)
        model.fit(X_tr, y_tr)
        n_train, base_rate = len(X_tr), float(y_tr.mean())
        del X_tr, y_tr
        te = ds.loc[test_mask]
        p = model.predict_proba(te[features])[:, 1]
        out = te[["timestamp", "symbol", "fwd_ret", "ret_next", "target"]].copy()
        out["prob_up"] = p
        out["train_base_rate"] = base_rate
        out["fold"] = i
        if cache_file is not None:
            out.to_pickle(cache_file)
        preds.append(out)
        if verbose:
            print(f"  [{model_name} h={horizon}] fold {i + 1}/{len(blocks)} "
                  f"train={n_train:,} test={len(te):,} {b0:%Y-%m-%d} ({time.time() - started:.0f}s)",
                  flush=True)
    return pd.concat(preds, ignore_index=True)
