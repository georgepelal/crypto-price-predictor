"""Unit and smoke tests for the Crypto ML Price Predictor pipeline."""

import unittest
import numpy as np
import pandas as pd

from src.data_loader import generate_synthetic_crypto_data
from src.features import generate_crypto_features
from src.labeling import make_directional_labels, apply_triple_barrier_labeling
from src.model import train_walk_forward
from src.backtest import run_backtest


class TestCryptoMLPipeline(unittest.TestCase):
    def setUp(self):
        # 300 synthetic hourly candles for testing
        self.df = generate_synthetic_crypto_data(symbol="BTC/USDT", timeframe="1h", bars=300)

    def test_synthetic_data_structure(self):
        self.assertEqual(len(self.df), 300)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, self.df.columns)
        # Check no negative prices
        self.assertTrue((self.df["close"] > 0).all())
        # Check high >= low
        self.assertTrue((self.df["high"] >= self.df["low"]).all())

    def test_feature_engineering_no_nans_on_clean_tail(self):
        feat = generate_crypto_features(self.df)
        self.assertIn("rsi", feat.columns)
        self.assertIn("norm_atr", feat.columns)
        self.assertIn("bb_pct_b", feat.columns)
        self.assertIn("ret_lag_1", feat.columns)
        # Tail should not contain NaNs after rolling warmup
        clean_tail = feat.iloc[35:]
        self.assertFalse(clean_tail["rsi"].isna().any())
        self.assertFalse(clean_tail["norm_atr"].isna().any())

    def test_directional_labeling(self):
        feat = generate_crypto_features(self.df)
        labeled = make_directional_labels(feat, forward_bars=1)
        self.assertIn("target", labeled.columns)
        unique_targets = set(labeled["target"].unique())
        # Targets must be subset of {-1, 0, 1, 2}
        self.assertTrue(unique_targets.issubset({-1, 0, 1, 2}))

    def test_backtest_execution(self):
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        dummy_df = pd.DataFrame(
            {"close": np.linspace(100, 110, n)},
            index=dates,
        )
        # Alternate between Up (2) and Neutral (1)
        preds = pd.Series([2 if i % 2 == 0 else 1 for i in range(n)], index=dates)
        bt = run_backtest(dummy_df, preds, initial_capital=10000.0)
        self.assertIn("metrics", bt)
        self.assertIn("equity", bt["results_df"].columns)
        self.assertGreater(bt["metrics"]["final_capital"], 0)

    def test_derivatives_features(self):
        from src.data_loader import generate_synthetic_derivatives_data
        deriv = generate_synthetic_derivatives_data(timeframe="1h", bars=300)
        merged = self.df.join(deriv)
        feat = generate_crypto_features(merged)
        self.assertIn("fr_raw", feat.columns)
        self.assertIn("fr_zscore", feat.columns)
        self.assertIn("oi_change_1h", feat.columns)
        self.assertIn("oi_price_alignment", feat.columns)

    def test_optimizer(self):
        from src.optimizer import optimize_pipeline
        # Quick 2-trial run on synthetic data
        opt = optimize_pipeline(self.df, symbol="BTC/USDT", timeframe="1h", n_trials=2, n_splits=2)
        self.assertIn("best_params", opt)
        self.assertIn("atr_multiplier", opt["best_params"])

    def test_multitimeframe_features(self):
        from src.features import add_multitimeframe_features
        df_4h = self.df.iloc[::4].copy()
        df_1d = self.df.iloc[::24].copy()
        mtf_feat = add_multitimeframe_features(self.df, df_4h=df_4h, df_1d=df_1d)
        self.assertIn("tf4h_trend_ratio", mtf_feat.columns)
        self.assertIn("daily_trend_ratio", mtf_feat.columns)
        self.assertIn("mtf_trend_alignment", mtf_feat.columns)


    def test_directional_edge_backtest(self):
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        dummy_df = pd.DataFrame(
            {"close": np.linspace(100, 110, n)},
            index=dates,
        )
        preds = pd.Series([2] * n, index=dates)
        probs = pd.DataFrame(
            {
                "prob_class_0": [0.10] * n,
                "prob_class_1": [0.60] * n,
                "prob_class_2": [0.30] * n,
            },
            index=dates,
        )
        # Edge = 0.30 - 0.10 = +0.20 >= 0.06 -> Should enter Long
        bt = run_backtest(
            dummy_df,
            predictions=preds,
            probabilities=probs,
            signal_mode="edge",
            edge_threshold=0.06,
        )
        self.assertIn("metrics", bt)
        self.assertGreater(bt["metrics"]["total_trades"], 0)
        self.assertGreater(bt["metrics"]["final_capital"], 0)


    def test_fractional_differentiation(self):
        from src.fracdiff import get_weights_ffd, frac_diff_ffd, find_optimal_d, add_fracdiff_features
        # 1. Weights
        w = get_weights_ffd(d=0.40, threshold=1e-3)
        self.assertEqual(w[0], 1.0)
        self.assertLess(w[1], 0.0)  # First lag is negative (-d)

        # 2. FFD execution on close
        fd = frac_diff_ffd(self.df["close"], d=0.40)
        self.assertEqual(len(fd), len(self.df))
        self.assertFalse(fd.dropna().empty)

        # 3. Optimal d search
        opt = find_optimal_d(self.df["close"])
        self.assertIn("best_d", opt)
        self.assertGreaterEqual(opt["best_d"], 0.1)
        self.assertLessEqual(opt["best_d"], 0.9)

        # 4. Feature addition
        df_fd, meta = add_fracdiff_features(self.df, price_col="close", d=0.40)
        self.assertIn("fracdiff_close", df_fd.columns)
        self.assertIn("fracdiff_zscore_24h", df_fd.columns)
        self.assertIn("fracdiff_momentum_3h", df_fd.columns)
        self.assertIn("fracdiff_dev_ema", df_fd.columns)

    def test_market_microstructure(self):
        from src.microstructure import (
            compute_amihud_illiquidity,
            compute_corwin_schultz_spread,
            compute_roll_effective_spread,
            compute_order_flow_imbalance,
            add_microstructure_features,
        )
        # 1. Amihud
        amihud = compute_amihud_illiquidity(self.df["close"], self.df["volume"])
        self.assertEqual(len(amihud), len(self.df))

        # 2. Corwin-Schultz
        cs = compute_corwin_schultz_spread(self.df["high"], self.df["low"])
        self.assertEqual(len(cs), len(self.df))
        self.assertTrue((cs >= 0.0).all())

        # 3. Roll
        roll = compute_roll_effective_spread(self.df["close"])
        self.assertEqual(len(roll), len(self.df))
        self.assertTrue((roll >= 0.0).all())

        # 4. OFI
        ofi_dict = compute_order_flow_imbalance(self.df["close"], self.df["volume"])
        self.assertIn("micro_ofi_1h", ofi_dict)
        self.assertIn("micro_ofi_ma_12", ofi_dict)

        # 5. Full microstructure integration
        df_micro = add_microstructure_features(self.df)
        expected_cols = [
            "micro_amihud_zscore",
            "micro_kyle_lambda_zscore",
            "micro_corwin_schultz",
            "micro_roll_spread",
            "micro_ofi_1h",
            "micro_ofi_ma_12",
        ]
        for col in expected_cols:
            self.assertIn(col, df_micro.columns)

    def test_feature_selection(self):
        from src.feature_selection import select_optimal_features
        feat = generate_crypto_features(self.df)
        labeled = make_directional_labels(feat)
        cols = [c for c in labeled.columns if c not in ["open", "high", "low", "close", "volume", "target", "fwd_return", "threshold"]]
        clean = labeled[labeled["target"] != -1].dropna(subset=cols + ["target"])
        selected, report = select_optimal_features(clean[cols], clean["target"].astype(int), top_k=5)
        self.assertEqual(len(selected), 5)
        self.assertIn("dropped_collinear_count", report)

    def test_paper_trading_tracker(self):
        import os
        from datetime import datetime, timezone
        from src.paper_tracker import execute_paper_candle, get_paper_trading_summary
        test_db = "/tmp/test_paper_tracker.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        now = datetime.now(timezone.utc)
        # 1. Open Long
        res1 = execute_paper_candle("BTC/USDT", now, 100, 105, 95, 100, "LONG / BUY", sl_price=90, tp_price=120, edge=0.20, db_path=test_db)
        self.assertEqual(len(res1["actions"]), 1)

        # 2. Trigger Take-Profit
        res2 = execute_paper_candle("BTC/USDT", now, 115, 125, 110, 122, "NEUTRAL", sl_price=None, tp_price=None, edge=0.0, db_path=test_db)
        self.assertEqual(len(res2["actions"]), 1)

        summary = get_paper_trading_summary(db_path=test_db)
        self.assertEqual(summary["winning_trades"], 1)
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(summary["win_rate_pct"], 100.0)

        if os.path.exists(test_db):
            os.remove(test_db)

    def test_conformal_prediction(self):
        from src.conformal import ConformalClassifier
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        probs = pd.DataFrame(
            {
                "prob_class_0": [0.10] * n,
                "prob_class_1": [0.20] * n,
                "prob_class_2": [0.70] * n,
            },
            index=dates,
        )
        y_true = pd.Series([2] * n, index=dates)

        cc = ConformalClassifier(alpha=0.15)
        q_hat = cc.calibrate(probs, y_true)
        self.assertGreater(q_hat, 0.0)

        singletons, set_sizes, raw_sets = cc.predict_prediction_sets(probs)
        self.assertEqual(len(singletons), n)
        # Class 2 has 70% probability, should be recognized
        self.assertIn(singletons.iloc[0], [2, -1])

        live_eval = cc.evaluate_live_bar({0: 0.05, 1: 0.10, 2: 0.85})
        self.assertIn("is_singleton", live_eval)

    def test_market_regime_filter(self):
        from src.regime_hmm import MarketRegimeFilter
        rf = MarketRegimeFilter(n_regimes=3)
        regimes = rf.predict_regimes(self.df)
        self.assertEqual(len(regimes), len(self.df))
        self.assertIn("dominant_regime", regimes.columns)
        self.assertIn("prob_high_vol_chop", regimes.columns)
        self.assertIn("prob_bull_trend", regimes.columns)

        live_bar = rf.evaluate_live_bar(self.df)
        self.assertIn("is_chop_veto", live_bar)

    def test_adx_indicator(self):
        from src.features import compute_adx
        adx_dict = compute_adx(self.df["high"], self.df["low"], self.df["close"], period=14)
        self.assertIn("adx", adx_dict)
        self.assertIn("plus_di", adx_dict)
        self.assertIn("minus_di", adx_dict)
        self.assertEqual(len(adx_dict["adx"]), len(self.df))
        self.assertTrue((adx_dict["adx"] >= 0.0).all())
        self.assertTrue((adx_dict["adx"] <= 100.0).all())

        feat = generate_crypto_features(self.df)
        self.assertIn("adx_14", feat.columns)
        self.assertIn("adx_trend_strength", feat.columns)
        self.assertIn("adx_is_choppy", feat.columns)

    def test_cvd_divergence(self):
        from src.microstructure import compute_cvd_divergence
        cvd_res = compute_cvd_divergence(self.df["close"], self.df["volume"], window=24)
        self.assertIn("micro_cvd_zscore", cvd_res)
        self.assertIn("micro_cvd_divergence", cvd_res)
        self.assertEqual(len(cvd_res["micro_cvd_zscore"]), len(self.df))
        self.assertTrue((cvd_res["micro_cvd_zscore"] >= -3.01).all())
        self.assertTrue((cvd_res["micro_cvd_zscore"] <= 3.01).all())
        self.assertTrue((cvd_res["micro_cvd_divergence"] >= -1.01).all())
        self.assertTrue((cvd_res["micro_cvd_divergence"] <= 1.01).all())

    def test_liquidation_flush_proxy(self):
        from src.microstructure import compute_liquidation_flush_proxy
        liq_res = compute_liquidation_flush_proxy(self.df)
        self.assertIn("micro_liq_flush_bullish", liq_res)
        self.assertIn("micro_liq_flush_bearish", liq_res)
        self.assertEqual(len(liq_res["micro_liq_flush_bullish"]), len(self.df))
        self.assertTrue(set(liq_res["micro_liq_flush_bullish"].unique()).issubset({0.0, 1.0}))
        self.assertTrue(set(liq_res["micro_liq_flush_bearish"].unique()).issubset({0.0, 1.0}))

    def test_session_liquidity_features(self):
        feat = generate_crypto_features(self.df)
        self.assertIn("session_london_ny_overlap", feat.columns)
        self.assertIn("session_core_hours", feat.columns)
        self.assertIn("is_weekend", feat.columns)
        self.assertIn("session_asian_dead_zone", feat.columns)
        self.assertIn("liquidity_quality_score", feat.columns)

    def test_profit_lock_execution(self):
        import os
        from datetime import datetime, timezone
        from src.paper_tracker import execute_paper_candle, get_paper_trading_summary
        test_db = "/tmp/test_profit_lock.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        now = datetime.now(timezone.utc)
        # 1. Open Long at 100 with SL 90 and TP 120 (Target gain: +20)
        res1 = execute_paper_candle("BTC/USDT", now, 100, 102, 98, 100, "LONG / BUY", sl_price=90, tp_price=120, edge=0.20, db_path=test_db)
        self.assertEqual(len(res1["actions"]), 1)

        # 2. Candle surges to 115 (115 >= 100 + 0.65*20 = 113) -> Profit Lock triggers!
        # SL elevated to 100 + 0.25*20 = 105
        res2 = execute_paper_candle("BTC/USDT", now, 108, 115, 107, 114, "NEUTRAL", sl_price=None, tp_price=None, edge=0.0, db_path=test_db)
        self.assertTrue(any("Profit-Lock Activated" in a for a in res2["actions"]))

        # 3. Next candle market reverses violently down to 103 (below the 105 locked SL)
        res3 = execute_paper_candle("BTC/USDT", now, 110, 111, 103, 104, "NEUTRAL", sl_price=None, tp_price=None, edge=0.0, db_path=test_db)
        self.assertTrue(any("PROFIT_LOCK" in a for a in res3["actions"]))

        summary = get_paper_trading_summary(db_path=test_db)
        self.assertEqual(summary["closed_trades"], 1)
        self.assertEqual(summary["winning_trades"], 1)
        self.assertGreater(summary["realized_pnl_usd"], 0)

        if os.path.exists(test_db):
            os.remove(test_db)

    def test_calibrated_probabilities(self):
        feat = generate_crypto_features(self.df)
        labeled = make_directional_labels(feat)
        cols = ["rsi", "norm_atr", "bb_bandwidth"]
        clean = labeled[labeled["target"] != -1].dropna(subset=cols + ["target"])
        X = clean[cols]
        y = clean["target"].astype(int)

        train_res = train_walk_forward(X, y, n_splits=2, min_train_size=100, calibrate_probabilities=True, max_iter=20)
        self.assertIn("Calibrated Sigmoid", train_res["model_name"])
        self.assertGreater(train_res["overall_accuracy"], 0.0)
        # Check probs sum to 1
        probs = train_res["oof_probs"].dropna()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_cross_asset_features(self):
        from src.features import add_cross_asset_features
        df_eth = self.df.copy()
        df_eth["close"] = df_eth["close"] * 0.05
        df_cross = add_cross_asset_features(self.df, df_secondary=df_eth, secondary_symbol="ETH")
        self.assertIn("cross_eth_ratio", df_cross.columns)
        self.assertIn("cross_eth_mom_divergence_1h", df_cross.columns)
        self.assertIn("cross_eth_mom_divergence_24h", df_cross.columns)
        self.assertIn("cross_eth_risk_on_regime", df_cross.columns)

    def test_heterogeneous_ensemble(self):
        from src.model import get_model_instance
        model, name = get_model_instance(model_type="ensemble", max_iter=20, rf_estimators=10)
        self.assertIn("Heterogeneous Ensemble", name)
        X = np.random.randn(50, 4)
        y = np.random.choice([0, 1, 2], 50)
        model.fit(X, y)
        probs = model.predict_proba(X)
        self.assertEqual(probs.shape, (50, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_fractional_kelly_sizing(self):
        from src.paper_tracker import compute_fractional_kelly_risk
        r_high = compute_fractional_kelly_risk(p_win=0.75, sl_distance=1.8, tp_distance=1.2)
        self.assertGreater(r_high, 0.015)
        self.assertLessEqual(r_high, 0.035)

    def test_optuna_ensemble_tuning(self):
        from src.optimizer import optimize_pipeline
        opt = optimize_pipeline(self.df, symbol="BTC/USDT", timeframe="1h", n_trials=2, n_splits=2, model_type="ensemble")
        self.assertIn("best_params", opt)
        bp = opt["best_params"]
        # Ensemble weights or params should be present if sampled
        self.assertTrue(any(k in bp for k in ["w_hgb", "rf_estimators", "lr_c"]))

    def test_multi_asset_scanner_ranking(self):
        from multi_asset_scanner import scan_and_allocate_capital
        from unittest.mock import patch

        mock_btc = {
            "symbol": "BTC/USDT", "timestamp": pd.Timestamp("2026-10-06"), "price": 65000.0,
            "atr_pct": 1.2, "p_down": 0.20, "p_neutral": 0.30, "p_up": 0.50, "edge": 0.30,
            "direction": "BUY / LONG", "regime": "BULL_TREND", "adx": 32.0, "is_singleton": True,
            "pred_set": [2], "veto_reason": None, "conviction_score": 0.48, "sl_price": 63500.0,
            "tp_price": 67000.0, "prob_dict": {0: 0.2, 1: 0.3, 2: 0.5}, "open_bar": 64800.0,
            "high_bar": 65200.0, "low_bar": 64700.0, "status": "OK",
        }
        mock_eth = {
            "symbol": "ETH/USDT", "timestamp": pd.Timestamp("2026-10-06"), "price": 3500.0,
            "atr_pct": 1.8, "p_down": 0.35, "p_neutral": 0.40, "p_up": 0.25, "edge": -0.10,
            "direction": "SELL / SHORT", "regime": "BEAR_TREND", "adx": 22.0, "is_singleton": False,
            "pred_set": [0, 1], "veto_reason": None, "conviction_score": 0.088, "sl_price": 3610.0,
            "tp_price": 3420.0, "prob_dict": {0: 0.35, 1: 0.4, 2: 0.25}, "open_bar": 3510.0,
            "high_bar": 3520.0, "low_bar": 3490.0, "status": "OK",
        }

        with patch("multi_asset_scanner.scan_single_asset") as mock_scan:
            mock_scan.side_effect = lambda symbol, **kwargs: mock_btc if "BTC" in symbol else mock_eth
            result = scan_and_allocate_capital(symbols=["BTC/USDT", "ETH/USDT"], paper_trade=False)
            self.assertEqual(len(result["ranked_assets"]), 2)
            # Top candidate should be BTC due to conviction_score 0.48 > 0.088
            self.assertEqual(result["best_candidate"]["symbol"], "BTC/USDT")
            self.assertEqual(result["best_candidate"]["direction"], "BUY / LONG")

    def test_triple_barrier_with_t1(self):
        from src.labeling import apply_triple_barrier_labeling
        tb = apply_triple_barrier_labeling(self.df, horizon_bars=6, profit_mult=1.5, stop_mult=1.0)
        self.assertIn("target", tb.columns)
        self.assertIn("t1", tb.columns)
        self.assertIn("tp_dist", tb.columns)
        self.assertIn("sl_dist", tb.columns)
        unique_targets = set(tb["target"].unique())
        self.assertTrue(unique_targets.issubset({-1, 0, 1, 2}))
        # Last 6 bars must be masked with -1
        self.assertTrue((tb["target"].iloc[-6:] == -1).all())

    def test_purged_walk_forward_validator(self):
        from src.model import PurgedWalkForwardValidator
        n = 300
        purge_bars = 6
        validator = PurgedWalkForwardValidator(n_splits=3, min_train_size=100, purge_bars=purge_bars, embargo_pct=0.01)
        splits = list(validator.split(n))
        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            # Guarantee zero overlap and purge buffer between train and test
            self.assertLessEqual(train_idx[-1], test_idx[0] - purge_bars)
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)

    def test_meta_labeling_pipeline(self):
        from src.meta_labeling import create_meta_labels, train_meta_model
        # Generate primary signals alternating long/short
        sigs = pd.Series([1 if i % 4 == 0 else (-1 if i % 4 == 2 else 0) for i in range(len(self.df))], index=self.df.index)
        meta_labels = create_meta_labels(self.df, sigs, horizon_bars=4, profit_mult=1.5, stop_mult=1.0)
        self.assertIn(1, meta_labels.values)
        
        # Prepare feature matrix
        feat = generate_crypto_features(self.df)
        cols = ["rsi", "norm_atr", "bb_bandwidth"]
        meta_res = train_meta_model(feat[cols], meta_labels, n_splits=2, min_train_size=50)
        self.assertIn("meta_probs", meta_res)
        self.assertIn("meta_model", meta_res)
        self.assertGreater(meta_res["n_trade_samples"], 5)
        # Probabilities must be valid in [0, 1]
        valid_p = meta_res["meta_probs"].dropna()
        self.assertTrue((valid_p >= 0.0).all() and (valid_p <= 1.0).all())

    def test_expected_value_calculation(self):
        from src.meta_labeling import compute_expected_value
        ev_bull = compute_expected_value(p_up=0.60, p_down=0.20, atr_val=0.02, profit_mult=1.5, stop_mult=1.0)
        self.assertGreater(ev_bull["ev_long"], 0.0)
        self.assertLess(ev_bull["ev_short"], 0.0)
        self.assertEqual(ev_bull["best_ev"], ev_bull["ev_long"])

        ev_bear = compute_expected_value(p_up=0.20, p_down=0.60, atr_val=0.02, profit_mult=1.5, stop_mult=1.0)
        self.assertGreater(ev_bear["ev_short"], 0.0)
        self.assertLess(ev_bear["ev_long"], 0.0)


if __name__ == "__main__":
    unittest.main()





