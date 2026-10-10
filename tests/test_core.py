"""Tests for the core pipeline: no lookahead, correct purging, correct P&L timing and costs."""

import unittest

import numpy as np
import pandas as pd

from src.core import walkforward as wf_mod
from src.core.dataset import add_targets, build_dataset, model_features
from src.core.features import build_features, funding_features
from src.core.history import _merge
from src.core.strategy import cost_aware_positions, simulate_symbol, step_position


def synthetic_ohlcv(n=2500, seed=0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "quote_volume": vol * close, "trades": rng.integers(1000, 2000, n).astype(float),
                         "taker_buy_base": vol * rng.uniform(0.4, 0.6, n)}, index=idx)


def synthetic_funding(index):
    st = pd.date_range(index[0].ceil("8h"), index[-1], freq="8h")
    return pd.DataFrame({"funding_rate": np.linspace(0.0001, 0.0003, len(st))}, index=st)


def synthetic_fng(index):
    days = pd.date_range(index[0].floor("D"), index[-1].floor("D"), freq="D")
    return pd.DataFrame({"fng": np.linspace(20, 80, len(days))}, index=days)


class TestCoreNoLookahead(unittest.TestCase):
    def setUp(self):
        self.o = synthetic_ohlcv()
        self.m = synthetic_ohlcv(seed=1)
        self.fu = synthetic_funding(self.o.index)
        self.fg = synthetic_fng(self.o.index)

    def test_features_do_not_change_when_future_is_removed(self):
        full = build_features(self.o, self.fu, self.fg, self.m)
        for cut in (self.o.index[900], self.o.index[1777], self.o.index[-2]):
            trunc = build_features(self.o[self.o.index <= cut], self.fu[self.fu.index <= cut + pd.Timedelta("1h")],
                                   self.fg[self.fg.index <= cut], self.m[self.m.index <= cut])
            a, b = full.loc[cut], trunc.loc[cut]
            both_nan = a.isna() & b.isna()
            diff = (a - b).abs()[~both_nan]
            self.assertTrue((diff <= 1e-9 * (1 + a.abs()[~both_nan])).all(),
                            f"lookahead in: {list(diff[diff > 1e-9].index)}")

    def test_funding_settled_at_decision_time_is_not_used(self):
        idx = pd.date_range("2024-01-01 06:00", periods=3, freq="1h", tz="UTC")  # 06,07,08 opens
        fu = pd.DataFrame({"funding_rate": [0.01, 0.02]},
                          index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True))
        f = funding_features(idx, fu, pd.Timedelta("1h"))
        # bar opened 07:00 closes 08:00 == settlement time -> must still see the 00:00 value
        self.assertAlmostEqual(f.loc[idx[1], "fund_last"], 0.01)
        self.assertAlmostEqual(f.loc[idx[2], "fund_last"], 0.02)

    def test_targets(self):
        f = build_features(self.o, None, None, self.m)
        t = add_targets(f, horizon=4)
        c = t["_close"]
        self.assertAlmostEqual(t["fwd_ret"].iloc[10], np.log(c.iloc[14] / c.iloc[10]))
        self.assertTrue(t["target"].iloc[-4:].isna().all())
        self.assertAlmostEqual(t["ret_next"].iloc[10], np.log(c.iloc[11] / c.iloc[10]))


class TestWalkForwardPurge(unittest.TestCase):
    def test_training_labels_end_before_test_block(self):
        o, m = synthetic_ohlcv(3000), synthetic_ohlcv(3000, seed=1)
        data = {"ohlcv": {"BTC/USDT": m, "ETH/USDT": o}, "funding": {}, "fng": None}
        h = 6
        ds = build_dataset(data, ["BTC/USDT", "ETH/USDT"], horizon=h)
        feats = model_features(ds)
        seen = []
        real = wf_mod.make_model

        class Spy:
            def __init__(self):
                self.m = real("logreg")

            def fit(self, X, y):
                seen.append(ds.loc[X.index, "timestamp"].max())
                return self.m.fit(X, y)

            def predict_proba(self, X):
                return self.m.predict_proba(X)

        wf_mod.make_model = lambda name: Spy()
        try:
            oos = wf_mod.walk_forward(ds, feats, h, "logreg", test_start=str(ds.timestamp.iloc[len(ds) // 2].date()),
                                      retrain_every="10D", verbose=False)
        finally:
            wf_mod.make_model = real
        starts = list(oos.groupby("fold")["timestamp"].min())
        for train_max, test_start in zip(seen, starts):
            # label of the last training row (t + h bars) must be known before the first test decision
            self.assertLessEqual(train_max + pd.Timedelta(hours=h + 1), test_start + pd.Timedelta(hours=1))


class TestWalkForwardResume(unittest.TestCase):
    def test_fold_cache_resume_gives_identical_predictions(self):
        import tempfile
        o, m = synthetic_ohlcv(3000), synthetic_ohlcv(3000, seed=1)
        data = {"ohlcv": {"BTC/USDT": m, "ETH/USDT": o}, "funding": {}, "fng": None}
        ds = build_dataset(data, ["BTC/USDT", "ETH/USDT"], horizon=1, float32=True)
        feats = model_features(ds)
        start = str(ds.timestamp.iloc[len(ds) // 2].date())
        full = wf_mod.walk_forward(ds, feats, 1, "logreg", test_start=start, retrain_every="10D", verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(wf_mod.TimeBudgetExceeded):
                wf_mod.walk_forward(ds, feats, 1, "logreg", test_start=start, retrain_every="10D",
                                    verbose=False, fold_cache=tmp, max_seconds=0.0)
            resumed = wf_mod.walk_forward(ds, feats, 1, "logreg", test_start=start, retrain_every="10D",
                                          verbose=False, fold_cache=tmp)
        pd.testing.assert_frame_equal(full.reset_index(drop=True), resumed.reset_index(drop=True))


class TestStrategy(unittest.TestCase):
    def test_position_earns_next_bar_and_pays_costs(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        frame = pd.DataFrame({"ret_next": np.log([1.01, 1.02, 0.99, 1.0])}, index=idx)
        pos = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
        r = simulate_symbol(frame, pos, cost_per_side=0.001)
        self.assertAlmostEqual(r["gross"].iloc[0], 0.0)
        self.assertAlmostEqual(r["gross"].iloc[1], 0.02, places=10)    # held from close[1] to close[2]
        self.assertAlmostEqual(r["cost"].iloc[1], 0.001)                # entry
        self.assertAlmostEqual(r["cost"].iloc[3], 0.001)                # exit
        self.assertAlmostEqual(r["cost"].iloc[2], 0.0)

    def test_cost_aware_rule_has_hysteresis(self):
        c = 0.0007  # hurdle = 14 bps
        self.assertEqual(step_position(0.0, 0.0010, c), 0.0)     # 10 bps < hurdle: stay flat
        self.assertEqual(step_position(0.0, 0.0020, c), 1.0)     # enter
        self.assertEqual(step_position(1.0, 0.0005, c), 1.0)     # still positive: hold
        self.assertEqual(step_position(1.0, -0.0001, c), 0.0)    # turns negative: exit
        self.assertEqual(step_position(0.0, -0.0020, c, allow_short=False), 0.0)
        mu = pd.Series([0.002, 0.001, -0.001, -0.003])
        self.assertEqual(list(cost_aware_positions(mu, c)), [1.0, 1.0, 0.0, -1.0])


class TestHistoryCache(unittest.TestCase):
    def test_merge_never_drops_old_rows(self):
        old = synthetic_ohlcv(500)
        new = old.iloc[-100:].copy()
        new.iloc[-1, new.columns.get_loc("close")] = 1.0  # revised last row wins
        merged = _merge(old, new)
        self.assertEqual(len(merged), 500)
        self.assertEqual(merged["close"].iloc[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
