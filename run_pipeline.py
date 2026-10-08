"""End-to-end pipeline runner for Crypto Price Predictor.

Executes data collection, feature engineering, labeling, Walk-Forward model training,
and realistic backtesting.
"""

import argparse
import os
from pathlib import Path
import yaml
import joblib
import pandas as pd

from src.data_loader import fetch_ohlcv_ccxt, fetch_derivatives_data, fetch_multitimeframe_ohlcv
from src.features import generate_crypto_features, add_multitimeframe_features
from src.labeling import make_directional_labels
from src.model import train_walk_forward
from src.backtest import run_backtest
from src.utils import print_backtest_report, plot_backtest_results


def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Crypto ML Price Predictor Pipeline")
    parser.add_argument("--symbol", type=str, default=None, help="Trading pair (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Bar timeframe (e.g. 1h, 15m)")
    parser.add_argument("--bars", type=int, default=None, help="Number of candles to fetch")
    parser.add_argument("--exchange", type=str, default=None, help="Exchange name (e.g. binance)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config yaml")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of data")
    parser.add_argument("--confidence", type=float, default=None, help="Probability threshold for entering trade")
    parser.add_argument("--use-derivatives", dest="use_derivatives", action="store_true", default=None, help="Enable Funding Rate & OI features")
    parser.add_argument("--no-derivatives", dest="use_derivatives", action="store_false", help="Disable derivatives features")
    parser.add_argument("--use-mtf", dest="use_mtf", action="store_true", default=None, help="Enable Multi-Timeframe features (4h & 1d context)")
    parser.add_argument("--no-mtf", dest="use_mtf", action="store_false", help="Disable Multi-Timeframe features")
    parser.add_argument("--signal-mode", type=str, default="edge", choices=["edge", "absolute"], help="Entry logic: edge (Directional Asymmetry) or absolute")
    parser.add_argument("--edge-threshold", type=float, default=0.06, help="Directional edge threshold P(Up) - P(Down)")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna Bayesian hyperparameter & threshold optimization")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna optimization trials")
    parser.add_argument("--select-features", action="store_true", help="Run automated two-stage feature selection (de-correlation + ranking)")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to retain with --select-features")
    parser.add_argument("--use-conformal", action="store_true", help="Enable Conformal Prediction singleton gate (selective abstention)")
    parser.add_argument("--conformal-alpha", type=float, default=0.20, help="Conformal significance level error budget (default 0.20)")
    parser.add_argument("--use-regime-filter", action="store_true", help="Enable Market Regime Filter (veto high-vol chop)")
    parser.add_argument("--use-mtf-gate", action="store_true", help="Enforce hard structural Multi-Timeframe trend gating")
    parser.add_argument("--use-cross-asset", action="store_true", help="Enable Cross-Asset Lead-Lag features (ETH/BTC ratio & beta)")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "ensemble", "lightgbm", "hist"], help="Classifier architecture (auto, ensemble, lightgbm, hist)")
    parser.add_argument("--use-triple-barrier", action="store_true", help="Enable Dynamic Triple Barrier Method labeling (take profit, stop loss, time horizon)")
    parser.add_argument("--tb-horizon", type=int, default=6, help="Triple barrier max holding horizon in bars (default: 6)")
    parser.add_argument("--tb-profit-mult", type=float, default=1.5, help="Triple barrier profit-take ATR multiplier (default: 1.5)")
    parser.add_argument("--tb-stop-mult", type=float, default=1.0, help="Triple barrier stop-loss ATR multiplier (default: 1.0)")
    parser.add_argument("--use-purged-cv", action="store_true", help="Enable Purged & Embargoed Walk-Forward CV (eliminates label overlap leakage)")
    parser.add_argument("--use-meta-labeling", action="store_true", help="Enable Secondary Meta-Model (predicts trade success and vetoes false breakouts)")
    parser.add_argument("--meta-threshold", type=float, default=0.55, help="Minimum probability of trade success required by Meta-Model (default: 0.55)")
    parser.add_argument("--use-ev-filter", action="store_true", help="Filter trades using mathematical Expected Value (EV)")
    parser.add_argument("--ev-hurdle", type=float, default=0.002, help="Minimum Expected Value per trade (default: 0.002 = +0.20% net)")
    args = parser.parse_args()

    # Load default configuration
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    label_cfg = cfg.get("labeling", {})
    model_cfg = cfg.get("model", {})
    bt_cfg = cfg.get("backtest", {})

    # CLI arguments override config
    symbol = args.symbol or data_cfg.get("symbol", "BTC/USDT")
    timeframe = args.timeframe or data_cfg.get("timeframe", "1h")
    total_bars = args.bars or data_cfg.get("bars_to_fetch", 700)
    exchange = args.exchange or data_cfg.get("exchange", "binance")
    confidence = args.confidence or bt_cfg.get("confidence_threshold", 0.45)
    use_derivatives = args.use_derivatives if args.use_derivatives is not None else data_cfg.get("use_derivatives", True)
    use_mtf = args.use_mtf if args.use_mtf is not None else data_cfg.get("use_mtf", True)

    print("=" * 65)
    print("   STARTING CRYPTO PRICE PREDICTOR PIPELINE")
    print(f"   Pair: {symbol} | Timeframe: {timeframe} | Target Bars: {total_bars}")
    print(f"   Derivatives Features (Funding & OI): {'ENABLED' if use_derivatives else 'DISABLED'}")
    print(f"   Multi-Timeframe Context (4h & 1d):   {'ENABLED' if use_mtf else 'DISABLED'}")
    print("=" * 65)

    # 1. Ingest OHLCV Data (Single or Multi-Timeframe)
    df_4h, df_1d = None, None
    if use_mtf and timeframe == "1h":
        print("[*] Ingesting Multi-Timeframe OHLCV data (1h, 4h, 1d)...")
        raw_df, df_4h, df_1d = fetch_multitimeframe_ohlcv(
            symbol=symbol,
            primary_timeframe=timeframe,
            primary_bars=total_bars,
            exchange_name=exchange,
            cache_dir=data_cfg.get("raw_dir", "data/raw"),
            force_download=args.force_download,
        )
    else:
        raw_df = fetch_ohlcv_ccxt(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=total_bars,
            exchange_name=exchange,
            cache_dir=data_cfg.get("raw_dir", "data/raw"),
            force_download=args.force_download,
        )

    if raw_df is None or len(raw_df) < 150:
        print("[!] Error: Insufficient data rows retrieved.")
        return

    # Ingest and merge Derivatives Data if enabled
    if use_derivatives:
        start_ts_ms = int(raw_df.index[0].timestamp() * 1000)
        deriv_df = fetch_derivatives_data(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=start_ts_ms,
            cache_dir=data_cfg.get("raw_dir", "data/raw"),
            force_download=args.force_download,
        )
        if not deriv_df.empty:
            raw_df = raw_df.join(deriv_df, how="left")
            raw_df["funding_rate"] = raw_df["funding_rate"].ffill().bfill().fillna(0.0001)
            raw_df["oi_val"] = raw_df["oi_val"].ffill().bfill().fillna(8e9)
            print(f"[+] Successfully merged Derivatives data ({len(deriv_df)} records).")

    print(f"[+] Loaded dataset: {len(raw_df)} bars. Span: {raw_df.index[0]} to {raw_df.index[-1]}")

    # 2. Engineer Features
    print("[*] Generating non-lookahead technical, momentum, and volatility features...")
    feat_df = generate_crypto_features(
        raw_df,
        return_lags=feat_cfg.get("return_lags"),
        volatility_windows=feat_cfg.get("volatility_windows"),
        rsi_period=feat_cfg.get("rsi_period", 14),
        bollinger_period=feat_cfg.get("bollinger_period", 20),
        atr_period=feat_cfg.get("atr_period", 14),
    )

    if use_mtf and df_4h is not None and df_1d is not None:
        print("[*] Computing and merging Multi-Timeframe features (4h & 1d context)...")
        feat_df = add_multitimeframe_features(feat_df, df_4h=df_4h, df_1d=df_1d)
        print("[+] Merged Multi-Timeframe signals (4h EMA trend, 4h RSI, Daily range & Confluence).")

    # Ingest and merge Cross-Asset Lead-Lag features if enabled
    if args.use_cross_asset:
        sec_sym = "ETH/USDT" if "BTC" in symbol else "BTC/USDT"
        sec_name = "ETH" if "ETH" in sec_sym else "BTC"
        print(f"[*] Ingesting Cross-Asset Benchmark data ({sec_sym})...")
        df_sec = fetch_ohlcv_ccxt(
            symbol=sec_sym,
            timeframe=timeframe,
            total_bars=total_bars,
            exchange_name=exchange,
            cache_dir=data_cfg.get("raw_dir", "data/raw"),
            force_download=args.force_download,
        )
        if df_sec is not None and not df_sec.empty:
            from src.features import add_cross_asset_features
            feat_df = add_cross_asset_features(feat_df, df_secondary=df_sec, secondary_symbol=sec_name)
            print(f"[+] Merged Cross-Asset signals ({sec_name} ratio, beta momentum, and risk-on regime).")

    atr_mult = label_cfg.get("atr_multiplier", 0.35)
    max_depth = model_cfg.get("max_depth", 5)
    learning_rate = model_cfg.get("learning_rate", 0.03)
    max_iter = model_cfg.get("max_iter", 200)
    extra_model_params = {}

    if args.optimize:
        from src.optimizer import optimize_pipeline
        opt_results = optimize_pipeline(
            raw_df,
            symbol=symbol,
            timeframe=timeframe,
            n_trials=args.trials,
            n_splits=model_cfg.get("n_splits", 5),
            model_type=args.model_type if args.model_type != "auto" else model_cfg.get("type", "auto"),
        )
        best = opt_results["best_params"]
        atr_mult = best.get("atr_multiplier", atr_mult)
        max_depth = best.get("max_depth", max_depth)
        learning_rate = best.get("learning_rate", learning_rate)
        max_iter = best.get("max_iter", max_iter)
        confidence = best.get("confidence_threshold", confidence)
        if "w_hgb" in best:
            w_sum = best["w_hgb"] + best["w_rf"] + best["w_lr"]
            extra_model_params["ensemble_weights"] = [best["w_hgb"] / w_sum, best["w_rf"] / w_sum, best["w_lr"] / w_sum]
        if "rf_estimators" in best:
            extra_model_params["rf_estimators"] = best["rf_estimators"]
        if "lr_c" in best:
            extra_model_params["lr_c"] = best["lr_c"]
        print(f"[*] Training pipeline with optimized settings (ATR={atr_mult:.2f}, Confidence={confidence:.2f}, Depth={max_depth}, LR={learning_rate:.4f})")


    # 3. Target Labeling
    if args.use_triple_barrier:
        from src.labeling import apply_triple_barrier_labeling
        print(f"[*] Generating Dynamic Triple Barrier labels (Horizon={args.tb_horizon}h, Profit={args.tb_profit_mult}x ATR, Stop={args.tb_stop_mult}x ATR)...")
        labeled_df = apply_triple_barrier_labeling(
            feat_df,
            horizon_bars=args.tb_horizon,
            profit_mult=args.tb_profit_mult,
            stop_mult=args.tb_stop_mult,
        )
    else:
        print("[*] Generating volatility-buffered directional labels...")
        labeled_df = make_directional_labels(
            feat_df,
            forward_bars=label_cfg.get("forward_bars", 1),
            atr_multiplier=atr_mult,
            fee_buffer=label_cfg.get("fee_buffer", 0.0005),
        )

    # Clean missing values resulting from rolling indicators and target shifting
    feature_cols = [
        c for c in labeled_df.columns
        if c not in ["open", "high", "low", "close", "volume", "target", "fwd_return", "threshold", "t1", "tp_dist", "sl_dist"]
    ]

    # Filter out unlabelled trailing rows (-1) and drop NaNs
    valid_data = labeled_df[labeled_df["target"] != -1].dropna(subset=feature_cols + ["target"])

    X = valid_data[feature_cols]
    y = valid_data["target"].astype(int)

    if args.select_features:
        from src.feature_selection import select_optimal_features
        print(f"[*] Running Two-Stage Feature Selection (Target: Top-{args.top_k} features)...")
        selected_cols, fs_report = select_optimal_features(X, y, top_k=args.top_k)
        print(f"[+] Pruned {fs_report['dropped_collinear_count']} collinear features. Retained Top-{len(selected_cols)} features.")
        print(f"[*] Selected Features: {selected_cols}")
        feature_cols = selected_cols
        X = X[feature_cols]

    print(f"[+] Clean dataset: {len(X)} samples with {len(feature_cols)} features.")
    class_dist = y.value_counts().to_dict()
    print(f"[*] Class Distribution: Down (0): {class_dist.get(0, 0)}, Neutral (1): {class_dist.get(1, 0)}, Up (2): {class_dist.get(2, 0)}")

    # 4. Train Model with Walk-Forward Validation
    print(f"[*] Running Walk-Forward Cross Validation ({model_cfg.get('n_splits', 5)} folds)...")
    configured_min_train = model_cfg.get("min_train_size", 500)
    effective_min_train = min(configured_min_train, max(100, int(len(X) * 0.4)))

    use_purged = args.use_purged_cv or args.use_triple_barrier
    purge_bars = args.tb_horizon if args.use_triple_barrier else 6
    if use_purged:
        print(f"[*] Activating Purged & Embargoed Cross-Validation (Purge Window: {purge_bars} bars, Embargo: 1%)...")

    train_results = train_walk_forward(
        X,
        y,
        n_splits=model_cfg.get("n_splits", 5),
        min_train_size=effective_min_train,
        model_type=args.model_type if args.model_type != "auto" else model_cfg.get("type", "auto"),
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        use_purged_cv=use_purged,
        purge_bars=purge_bars,
        **extra_model_params,
    )

    print(f"[+] Validation complete using {train_results['model_name']}.")
    print(f"[+] Out-of-Sample Accuracy: {train_results['overall_accuracy']:.2%}")

    # Meta-Model Training
    meta_model = None
    meta_probs = None
    if args.use_meta_labeling:
        from src.meta_labeling import create_meta_labels, train_meta_model
        print(f"[*] Training Secondary Meta-Model (Trade Success Filter, Threshold >= {args.meta_threshold:.2%})...")
        primary_sigs = pd.Series(0.0, index=valid_data.index)
        p_up = train_results["oof_probs"]["prob_class_2"].fillna(0.0)
        p_down = train_results["oof_probs"]["prob_class_0"].fillna(0.0)
        edge_series = p_up - p_down
        primary_sigs[edge_series >= args.edge_threshold] = 1.0
        primary_sigs[edge_series <= -args.edge_threshold] = -1.0

        meta_labels = create_meta_labels(
            valid_data,
            primary_sigs,
            horizon_bars=args.tb_horizon if args.use_triple_barrier else 6,
            profit_mult=args.tb_profit_mult if args.use_triple_barrier else 1.5,
            stop_mult=args.tb_stop_mult if args.use_triple_barrier else 1.0,
        )
        meta_results = train_meta_model(X, meta_labels, n_splits=model_cfg.get("n_splits", 5))
        meta_probs = meta_results["meta_probs"]
        meta_model = meta_results.get("meta_model")
        print(f"[+] Meta-Model active: {meta_results['n_trade_samples']} sample trades. OOF Win Precision: {meta_results['precision']:.2%}")

    # 5. Backtest Strategy with Precision Filters
    conformal_singletons = None
    if args.use_conformal:
        from src.conformal import ConformalClassifier
        print(f"[*] Calibrating Conformal Prediction Gate (Alpha={args.conformal_alpha:.2f})...")
        cc = ConformalClassifier(alpha=args.conformal_alpha)
        q_hat = cc.calibrate(train_results["oof_probs"], valid_data["target"])
        conformal_singletons, set_sizes, _ = cc.predict_prediction_sets(train_results["oof_probs"])
        singleton_count = int((conformal_singletons != -1).sum())
        print(f"[+] Conformal Gate active: {singleton_count}/{len(conformal_singletons)} trades retained as unambiguous singletons.")

    regime_probs = None
    if args.use_regime_filter:
        from src.regime_hmm import MarketRegimeFilter
        print("[*] Computing Causal Market Regimes (Bull Trend, Bear Trend, High-Vol Chop)...")
        rf = MarketRegimeFilter()
        regime_probs = rf.predict_regimes(feat_df)
        chop_count = int((regime_probs["prob_high_vol_chop"] >= 0.40).sum())
        print(f"[+] Regime Filter active: {chop_count} bars identified as high-vol chop (trading suppressed).")

    if args.signal_mode == "edge":
        print(f"[*] Running realistic backtest (Mode: Directional Edge, Threshold: P(Up) - P(Down) >= {args.edge_threshold:.2f})...")
    else:
        print(f"[*] Running realistic backtest (Mode: Absolute Confidence, Threshold >= {confidence:.2f})...")

    bt_results = run_backtest(
        df=valid_data,
        predictions=train_results["oof_preds"],
        probabilities=train_results["oof_probs"],
        initial_capital=bt_cfg.get("initial_capital", 10000.0),
        taker_fee=bt_cfg.get("taker_fee", 0.0005),
        slippage=bt_cfg.get("slippage", 0.0002),
        confidence_threshold=confidence,
        signal_mode=args.signal_mode,
        edge_threshold=args.edge_threshold,
        conformal_singletons=conformal_singletons,
        regime_probs=regime_probs,
        use_mtf_gate=args.use_mtf_gate,
        use_ev_gate=args.use_ev_filter,
        ev_hurdle=args.ev_hurdle,
        meta_probabilities=meta_probs,
        meta_threshold=args.meta_threshold,
        profit_mult=args.tb_profit_mult if args.use_triple_barrier else 1.5,
        stop_mult=args.tb_stop_mult if args.use_triple_barrier else 1.0,
        use_barrier_holding=args.use_triple_barrier,
        max_holding_bars=args.tb_horizon,
        bars_per_year=8760 if timeframe == "1h" else 35040,
    )

    # 6. Report and Visualization
    print_backtest_report(
        metrics=bt_results["metrics"],
        model_name=train_results["model_name"],
        symbol=symbol,
        timeframe=timeframe,
    )

    chart_path = plot_backtest_results(
        results_df=bt_results["results_df"],
        feature_importances=train_results["feature_importances"],
        symbol=symbol,
        timeframe=timeframe,
    )

    # 7. Save Checkpoint
    os.makedirs("models", exist_ok=True)
    clean_sym = symbol.replace("/", "_").replace(":", "_")
    model_save_path = f"models/{clean_sym}_{timeframe}_model.joblib"
    joblib.dump(
        {
            "model": train_results["final_model"],
            "meta_model": meta_model,
            "model_name": train_results["model_name"],
            "feature_cols": feature_cols,
            "metrics": bt_results["metrics"],
            "tb_params": {
                "use_triple_barrier": args.use_triple_barrier,
                "horizon_bars": args.tb_horizon,
                "profit_mult": args.tb_profit_mult,
                "stop_mult": args.tb_stop_mult,
            },
            "meta_params": {
                "use_meta_labeling": args.use_meta_labeling,
                "meta_threshold": args.meta_threshold,
            },
            "ev_params": {
                "use_ev_filter": args.use_ev_filter,
                "ev_hurdle": args.ev_hurdle,
            },
        },
        model_save_path,
    )
    print(f"[+] Saved model checkpoint and metadata to: {model_save_path}")
    print("[*] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
