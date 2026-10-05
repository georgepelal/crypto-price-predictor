"""Advanced Quantitative Alpha Pipeline: Meta-Labeling & Sentiment Integration.

Combines Technical Indicators, Perpetual Derivatives, and Crypto Fear & Greed Index
with Marcos Lopez de Prado's Meta-Labeling and dynamic Kelly bet sizing.
"""

import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data_loader import fetch_ohlcv_ccxt, fetch_derivatives_data, fetch_fear_and_greed
from src.features import generate_crypto_features
from src.labeling import make_directional_labels
from src.model import train_walk_forward
from src.backtest import run_backtest
from src.meta_labeling import create_meta_labels, train_meta_model, compute_kelly_positions


def main():
    parser = argparse.ArgumentParser(description="Advanced Meta-Labeling & Sentiment Pipeline")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (1h, 15m)")
    parser.add_argument("--bars", type=int, default=700, help="Number of candles")
    parser.add_argument("--confidence", type=float, default=0.33, help="Primary model threshold")
    parser.add_argument("--force-download", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    print("=" * 70)
    print("   ADVANCED QUANT PIPELINE: META-LABELING & FEAR & GREED ALPHA")
    print(f"   Asset: {args.symbol} | Timeframe: {args.timeframe} | Bars: {args.bars}")
    print("=" * 70)

    # 1. Ingest OHLCV
    raw_df = fetch_ohlcv_ccxt(symbol=args.symbol, timeframe=args.timeframe, total_bars=args.bars, force_download=args.force_download)

    # 2. Ingest Derivatives (Funding Rate & OI)
    start_ms = int(raw_df.index[0].timestamp() * 1000)
    deriv_df = fetch_derivatives_data(symbol=args.symbol, timeframe=args.timeframe, since_ms=start_ms, force_download=args.force_download)
    if not deriv_df.empty:
        raw_df = raw_df.join(deriv_df, how="left")
        raw_df["funding_rate"] = raw_df["funding_rate"].ffill().bfill().fillna(0.0001)
        raw_df["oi_val"] = raw_df["oi_val"].ffill().bfill().fillna(8e9)

    # 3. Ingest Crypto Fear & Greed Index
    fng_df = fetch_fear_and_greed(limit=365, force_download=args.force_download)
    if not fng_df.empty:
        # Reindex / as-of forward-fill onto hourly candles (published at 00:00 UTC)
        raw_df = raw_df.join(fng_df, how="left")
        raw_df["fng_value"] = raw_df["fng_value"].ffill().bfill().fillna(50.0)
        print(f"[+] Successfully merged Fear & Greed Index (Current: {raw_df['fng_value'].iloc[-1]:.0f}/100)")

    # 4. Generate All Features (Technical + Derivatives + Sentiment)
    print("[*] Engineering full multi-factor feature set (50+ indicators)...")
    feat_df = generate_crypto_features(raw_df)
    labeled_df = make_directional_labels(feat_df, forward_bars=1, atr_multiplier=0.30)

    feature_cols = [
        c for c in labeled_df.columns
        if c not in ["open", "high", "low", "close", "volume", "target", "fwd_return", "threshold"]
    ]
    valid_data = labeled_df[labeled_df["target"] != -1].dropna(subset=feature_cols + ["target"])

    X = valid_data[feature_cols]
    y = valid_data["target"].astype(int)

    effective_min_train = max(100, int(len(X) * 0.4))

    # 5. Train Primary Model (Walk-Forward CV)
    print(f"[*] Training Primary Directional Model (Walk-Forward CV)...")
    primary_results = train_walk_forward(
        X,
        y,
        n_splits=5,
        min_train_size=effective_min_train,
        model_type="auto",
    )
    print(f"[+] Primary Model Out-of-Sample Accuracy: {primary_results['overall_accuracy']:.2%}")

    # 6. Baseline Backtest (Fixed 100% Sizing)
    print("[*] Running Baseline Backtest (Fixed Position Sizing)...")
    base_bt = run_backtest(
        df=valid_data,
        predictions=primary_results["oof_preds"],
        probabilities=primary_results["oof_probs"],
        confidence_threshold=args.confidence,
    )
    base_metrics = base_bt["metrics"]

    # Derive raw primary signals
    raw_signals = base_bt["results_df"]["signal"]

    # 7. Generate Meta-Labels and Train Secondary Meta-Model
    print("[*] Generating Meta-Labels & Training Secondary Meta-Model (Trade Success Predictor)...")
    meta_labels = create_meta_labels(valid_data, raw_signals)
    meta_results = train_meta_model(X, meta_labels, n_splits=5, min_train_size=effective_min_train)

    # 8. Fractional Kelly Bet Sizing using Meta-Probabilities
    print("[*] Computing Dynamic Kelly Bet Sizing based on Meta-Model conviction...")
    kelly_positions = compute_kelly_positions(
        primary_signals=raw_signals,
        meta_probabilities=meta_results["meta_probs"],
        min_prob_threshold=0.50,
    )

    # 9. Meta-Labeled Backtest (Dynamic Sizing)
    meta_bt = run_backtest(
        df=valid_data,
        predictions=primary_results["oof_preds"],
        custom_positions=kelly_positions,
    )
    meta_metrics = meta_bt["metrics"]

    # 10. Print Side-by-Side Institutional Comparison
    print("\n" + "=" * 75)
    print("        INSTITUTIONAL STRATEGY COMPARISON: BASELINE vs META-LABELING")
    print(f"        Asset: {args.symbol} ({args.timeframe})")
    print("=" * 75)
    print(f" {'Metric':<28} | {'Baseline (Fixed Size)':<20} | {'Meta-Labeled (Kelly Size)':<20}")
    print("-" * 75)
    print(f" {'Total Out-of-Sample Bars':<28} | {base_metrics['total_test_bars']:<20} | {meta_metrics['total_test_bars']:<20}")
    print(f" {'Position Transitions':<28} | {base_metrics['total_trades']:<20} | {meta_metrics['total_trades']:<20}")
    print(f" {'Active Exposure Bars':<28} | {base_metrics['active_bars']:<20} | {meta_metrics['active_bars']:<20}")
    print(f" {'Trade Win Rate':<28} | {base_metrics['win_rate_pct']:<19.1f}% | {meta_metrics['win_rate_pct']:<19.1f}%")
    print(f" {'Net Strategy Return':<28} | {base_metrics['total_return_pct']:<19.2f}% | {meta_metrics['total_return_pct']:<19.2f}%")
    print(f" {'Annualized Sharpe Ratio':<28} | {base_metrics['sharpe_ratio']:<20.2f} | {meta_metrics['sharpe_ratio']:<20.2f}")
    print(f" {'Annualized Sortino Ratio':<28} | {base_metrics['sortino_ratio']:<20.2f} | {meta_metrics['sortino_ratio']:<20.2f}")
    print(f" {'Maximum Drawdown (MDD)':<28} | {base_metrics['max_drawdown_pct']:<19.2f}% | {meta_metrics['max_drawdown_pct']:<19.2f}%")
    print(f" {'Profit Factor':<28} | {base_metrics['profit_factor']:<20.2f} | {meta_metrics['profit_factor']:<20.2f}")
    print(f" {'Benchmark Buy & Hold':<28} | {base_metrics['benchmark_return_pct']:<19.2f}% | {meta_metrics['benchmark_return_pct']:<19.2f}%")
    print("=" * 75 + "\n")

    # 11. Plot Comparative Equity Curves
    os.makedirs("reports", exist_ok=True)
    clean_sym = args.symbol.replace("/", "_").replace(":", "_")
    chart_path = f"reports/meta_comparison_{clean_sym}_{args.timeframe}.png"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Equity Curves
    ax1.plot(base_bt["results_df"].index, base_bt["results_df"]["equity"], label="Baseline Strategy (Fixed 100%)", color="#ff9800", linewidth=1.8, linestyle="--")
    ax1.plot(meta_bt["results_df"].index, meta_bt["results_df"]["equity"], label="Meta-Labeled Strategy (Dynamic Kelly Sizing)", color="#00e676", linewidth=2.2)
    ax1.plot(base_bt["results_df"].index, base_bt["results_df"]["benchmark_equity"], label=f"Buy & Hold {args.symbol}", color="#90a4ae", alpha=0.7)
    ax1.set_title(f"Meta-Labeling Equity Curve: {args.symbol} ({args.timeframe})", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Capital ($)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Dynamic Position Sizing Over Time
    ax2.plot(meta_bt["results_df"].index, meta_bt["results_df"]["position"], color="#29b6f6", label="Kelly Bet Size [-1.0, +1.0]", linewidth=1.2)
    ax2.set_title("Dynamic Bet Sizing Over Time (Kelly Allocation)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Allocation", fontsize=10)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[+] Saved comparative equity chart to {chart_path}")


if __name__ == "__main__":
    main()
