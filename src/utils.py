import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


def print_backtest_report(metrics: Dict[str, Any], model_name: str, symbol: str, timeframe: str):
    """Print clean terminal summary table of model performance."""
    print("\n" + "=" * 65)
    print(f"       CRYPTO ML BACKTEST REPORT: {symbol} ({timeframe})")
    print(f"       Model: {model_name}")
    print("=" * 65)
    print(f" Total Out-of-Sample Bars : {metrics['total_test_bars']}")
    print(f" Active Trading Bars      : {metrics['active_bars']} ({metrics['active_bars']/metrics['total_test_bars']*100:.1f}% exposure)")
    print(f" Total Position Changes   : {metrics['total_trades']}")
    print("-" * 65)
    print(f" Initial Capital          : ${metrics['initial_capital']:,.2f}")
    print(f" Strategy Final Capital   : ${metrics['final_capital']:,.2f}")
    print(f" Strategy Total Return    : {metrics['total_return_pct']:+.2f}%")
    print(f" Benchmark (Buy & Hold)   : {metrics['benchmark_return_pct']:+.2f}%")
    print(f" Strategy CAGR            : {metrics['cagr_pct']:+.2f}%")
    print("-" * 65)
    print(f" Annualized Sharpe Ratio  : {metrics['sharpe_ratio']:.2f}")
    print(f" Annualized Sortino Ratio : {metrics['sortino_ratio']:.2f}")
    print(f" Strategy Max Drawdown    : {metrics['max_drawdown_pct']:.2f}%")
    print(f" Benchmark Max Drawdown   : {metrics['benchmark_max_drawdown_pct']:.2f}%")
    print(f" Bar Win Rate             : {metrics['win_rate_pct']:.2f}%")
    print(f" Profit Factor            : {metrics['profit_factor']:.2f}")
    print("=" * 65 + "\n")


def plot_backtest_results(
    results_df: pd.DataFrame,
    feature_importances: Dict[str, float],
    symbol: str,
    timeframe: str,
    output_dir: str = "reports",
) -> str:
    """Generate and save visual performance report."""
    os.makedirs(output_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "_").replace(":", "_")
    output_path = Path(output_dir) / f"backtest_{clean_sym}_{timeframe}.png"

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.5, 1.5]})

    # 1. Equity Curve
    ax_equity = axes[0]
    ax_equity.plot(results_df.index, results_df["equity"], label="ML Strategy (Net of Fees)", color="#00bcd4", linewidth=2)
    ax_equity.plot(results_df.index, results_df["benchmark_equity"], label=f"Buy & Hold {symbol}", color="#ff9800", linestyle="--", alpha=0.8)
    ax_equity.set_title(f"Out-of-Sample Equity Curve: {symbol} ({timeframe})", fontsize=14, fontweight="bold")
    ax_equity.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax_equity.grid(True, alpha=0.3)
    ax_equity.legend(loc="upper left")

    # 2. Top 10 Feature Importances
    ax_feat = axes[1]
    if feature_importances:
        top_10 = dict(list(feature_importances.items())[:10])
        names = list(top_10.keys())[::-1]
        values = list(top_10.values())[::-1]
        ax_feat.barh(names, values, color="#4caf50")
        ax_feat.set_title("Top 10 Feature Importances", fontsize=12, fontweight="bold")
        ax_feat.set_xlabel("Relative Importance", fontsize=10)
        ax_feat.grid(True, alpha=0.3, axis="x")
    else:
        ax_feat.text(0.5, 0.5, "Feature importance not available for this model", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[+] Performance chart saved to: {output_path}")
    return str(output_path)
