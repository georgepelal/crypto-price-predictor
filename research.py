"""Honest out-of-sample research run.

Walk-forward evaluation of every (model, horizon) on years of real data, compared against simple
baselines, with costs + funding, confidence intervals, and an explicit multiple-testing warning.

    .venv/bin/python research.py                       # default grid, writes reports/research/<run>/
    .venv/bin/python research.py --offline             # use cached data only
    .venv/bin/python research.py --horizons 1 --models lgbm --policies cost_lf

The PRIMARY configuration (below) was fixed before looking at any results. Treat it as the
headline number; everything else in the grid is sensitivity analysis and is subject to
selection bias if you pick the best one.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import history
from src.core.dataset import build_dataset, model_features
from src.core.metrics import classification_metrics, information_coefficient, trading_metrics, yearly_returns
from src.core.strategy import baseline_positions, policy_positions, run_portfolio
from src.core.walkforward import walk_forward

# v1 primary (fixed before the first run, 2026-10-08): naive overlapping rule -> failed after costs.
# v2 primary (fixed before running the cost-aware rule): same model & horizon, cost-aware decision rule.
PRIMARY = {"model": "lgbm", "horizon": 24, "policy": "cost_ls"}
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--data-start", default="2020-01-01")
    ap.add_argument("--test-start", default="2023-01-01")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 12, 24])
    ap.add_argument("--models", nargs="+", default=["lgbm", "logreg"])
    ap.add_argument("--policies", nargs="+", default=["overlap:0.02", "cost_ls", "cost_lf"],
                    help="overlap:<margin> (naive), cost_ls (cost-aware long/short), cost_lf (cost-aware long/flat)")
    ap.add_argument("--retrain-every", default="30D")
    ap.add_argument("--train-window", default=None, help="e.g. 1095D for a trailing 3-year window; default expanding")
    ap.add_argument("--cost-bps", type=float, default=7.0, help="fee+slippage per side, basis points (perp taker 5 + 2 slip)")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--offline", action="store_true", help="don't hit the network, use data/history cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse oos_<model>_h<H>.csv.gz already in --out (lets a long grid run in chunks)")
    return ap.parse_args()


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    a = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out = Path(a.out or f"reports/research/{run_id}")
    out.mkdir(parents=True, exist_ok=True)
    cost = a.cost_bps / 1e4

    data = history.update_all(a.symbols, a.timeframe, start=a.data_start, offline=a.offline)
    results, oos_store, ds_store = [], {}, {}

    for h in a.horizons:
        ds = build_dataset(data, a.symbols, horizon=h, timeframe=a.timeframe)
        feats = model_features(ds)
        ds_store[h] = ds
        for m in a.models:
            print(f"[wf] model={m} horizon={h} features={len(feats)}", flush=True)
            cache = out / f"oos_{m}_h{h}.csv.gz"
            if a.reuse and cache.exists():
                oos = pd.read_csv(cache, parse_dates=["timestamp"])
                print("     (reused cached out-of-sample predictions)")
            else:
                oos = walk_forward(ds, feats, h, m, test_start=a.test_start, retrain_every=a.retrain_every,
                                   train_window=a.train_window, bar=a.timeframe, verbose=False)
                oos.to_csv(cache, index=False)
            oos = oos.merge(ds[["timestamp", "symbol", "vol_168"]], on=["timestamp", "symbol"], how="left")
            oos_store[(m, h)] = oos
            cls = classification_metrics(oos)
            ic = information_coefficient(oos)
            for mg in a.policies:
                if a.long_only and mg == "cost_ls":
                    continue
                pos = policy_positions(oos, h, mg, cost)
                pf = run_portfolio(oos, pos, cost, data["funding"])
                tm = trading_metrics(pf["portfolio"], pf["per_symbol"])
                row = {"model": m, "horizon": h, "policy": mg, **cls, **ic, **tm,
                       "primary": (m == PRIMARY["model"] and h == PRIMARY["horizon"] and mg == PRIMARY["policy"])}
                results.append(row)
                print(f"     policy={mg:<12} AUC={cls['auc']:.4f} IC={ic['ic_mean']:+.4f} (t={ic['ic_t']:.2f}) "
                      f"gross={tm['gross_sharpe']:+.2f} Sharpe={tm['sharpe']:+.2f} [{tm.get('sharpe_ci_low', np.nan):+.2f},{tm.get('sharpe_ci_high', np.nan):+.2f}] "
                      f"Ret={tm['total_return_pct']:+.1f}% MDD={tm['max_dd_pct']:.1f}%", flush=True)

    # Baselines on the same out-of-sample window, same costs/funding
    any_oos = next(iter(oos_store.values()))
    base_rows, base_curves = [], {}
    for kind in ("buy_hold", "momentum_24h", "momentum_168h"):
        h = PRIMARY["horizon"] if PRIMARY["horizon"] in ds_store else a.horizons[-1]
        pos = baseline_positions(any_oos, ds_store[h], h, kind)
        pf = run_portfolio(any_oos, pos, cost, data["funding"])
        tm = trading_metrics(pf["portfolio"], pf["per_symbol"])
        base_rows.append({"baseline": kind, **tm})
        base_curves[kind] = pf["portfolio"]

    res = pd.DataFrame(results)
    res.to_csv(out / "grid_results.csv", index=False)
    pd.DataFrame(base_rows).to_csv(out / "baselines.csv", index=False)

    # Primary config details
    prim = res[res["primary"]]
    prim_curve = None
    if len(prim):
        oos = oos_store[(PRIMARY["model"], PRIMARY["horizon"])]
        pos = policy_positions(oos, PRIMARY["horizon"], PRIMARY["policy"], cost)
        pf = run_portfolio(oos, pos, cost, data["funding"])
        prim_curve = pf["portfolio"]
        per_sym = {s: trading_metrics(d["net"], {s: d}, bootstrap=0) for s, d in pf["per_symbol"].items()}
        yearly = pd.DataFrame({"model": yearly_returns(prim_curve),
                               **{k: yearly_returns(v) for k, v in base_curves.items()}})
    curves = []
    if prim_curve is not None:
        curves.append((f"Primary ({PRIMARY['model']} h{PRIMARY['horizon']} {PRIMARY['policy']})", prim_curve))
    ck = (CANDIDATE["model"], CANDIDATE["horizon"])
    if ck in oos_store and CANDIDATE != PRIMARY:
        oc = oos_store[ck]
        pc = run_portfolio(oc, policy_positions(oc, CANDIDATE["horizon"], CANDIDATE["policy"], cost), cost, data["funding"])
        curves.append((f"Paper-traded ({CANDIDATE['model']} h{CANDIDATE['horizon']} {CANDIDATE['policy']})", pc["portfolio"]))
    curves += [("Buy & hold", base_curves["buy_hold"]), ("Momentum 24h", base_curves["momentum_24h"])]
    write_plot(out, curves)
    write_summary(out, a, res, pd.DataFrame(base_rows), prim, per_sym if len(prim) else {},
                  yearly if len(prim) else None, len(model_features(next(iter(ds_store.values())))))
    with open(out / "run_config.json", "w") as fh:
        json.dump({**vars(a), "primary": PRIMARY, "run_id": run_id}, fh, indent=2, default=str)
    print(f"\n[done] report: {out / 'summary.md'}")


CANDIDATE = {"model": "lgbm", "horizon": 1, "policy": "cost_lf"}  # what paper_trader.py runs


def write_plot(out: Path, curves):
    """curves: list of (label, hourly_return_series). Single log axis, direct labels, no overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ends = []
    for (name, s), col in zip(curves, colors):
        eq = (1 + s).cumprod()
        ax.plot(eq.index, eq.values, color=col, lw=1.4, label=name)
        ends.append([np.log(eq.iloc[-1]), name, eq.iloc[-1], eq.index[-1]])
    ends.sort()
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    min_gap = 0.06 * (np.log(hi) - np.log(lo))
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    for ly, name, val, t in ends:
        ax.annotate(f"{name}  {val:.2f}x", (t, np.exp(ly)), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#52514e", annotation_clip=False)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(1.0, color="#b9b8b1", lw=0.8)
    ax.grid(True, axis="y", color="#e4e3dd", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b9b8b1")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.set_ylabel("Growth of 1 (log scale, after costs & funding)", color="#52514e", fontsize=9)
    ax.set_title("Out-of-sample equity, equal weight BTC/ETH/SOL", color="#0b0b0b", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(out / "equity.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(out, a, res, base, prim, per_sym, yearly, n_features):
    L = []
    L.append(f"# Research run {out.name}\n")
    L.append(f"- Symbols: {', '.join(a.symbols)} ({a.timeframe}), data from {a.data_start}, "
             f"**out-of-sample from {a.test_start}** (walk-forward, retrain every {a.retrain_every}, "
             f"train window {'expanding' if not a.train_window else a.train_window}, purge = horizon).")
    L.append(f"- {n_features} features, one model pooled across symbols. Costs {a.cost_bps:g} bps/side + perp funding. "
             f"{'Long-only' if a.long_only else 'Long/short'}.")
    L.append(f"- Primary config (fixed in advance): `{PRIMARY}`.")
    L.append(f"- Grid size: {len(res)} configs. **Picking the best of {len(res)} inflates results**; "
             "a config should only be trusted if it is robust across neighbouring settings and years.\n")

    if len(prim):
        p = prim.iloc[0]
        L.append("## Primary config\n")
        L.append("| AUC | Log-loss skill vs prior | Monthly IC (t) | Gross Sharpe | Net Sharpe [95% CI] | P(Sharpe≤0) | Return | Max DD | Turnover/yr | Costs/yr | Funding/yr |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        L.append(f"| {p.auc:.4f} | {p.logloss_skill_pct:+.3f}% | {p.ic_mean:+.4f} ({p.ic_t:.2f}) | {p.gross_sharpe:+.2f} | "
                 f"{p.sharpe:+.2f} [{p.sharpe_ci_low:+.2f}, {p.sharpe_ci_high:+.2f}] | {p.p_sharpe_le_0:.2f} | "
                 f"{p.total_return_pct:+.1f}% | {p.max_dd_pct:.1f}% | {p.turnover_per_year:.0f}x | "
                 f"{p.cost_drag_pct_per_year:.1f}% | {p.funding_pct_per_year:+.1f}% |\n")
        L.append("Per symbol:\n")
        L.append("| Symbol | Sharpe | Return | Max DD |")
        L.append("|---|---|---|---|")
        for s, m in per_sym.items():
            L.append(f"| {s} | {m['sharpe']:+.2f} | {m['total_return_pct']:+.1f}% | {m['max_dd_pct']:.1f}% |")
        L.append("")
        if yearly is not None:
            L.append("Calendar-year returns (%):\n")
            L.append("| Year | " + " | ".join(yearly.columns) + " |")
            L.append("|---|" + "---|" * len(yearly.columns))
            for y, r in yearly.iterrows():
                L.append(f"| {y} | " + " | ".join(f"{v:+.1f}" for v in r.values) + " |")
            L.append("")

    L.append("## Baselines (same window, costs, funding)\n")
    L.append("| Baseline | Sharpe [95% CI] | Return | Max DD | Turnover/yr |")
    L.append("|---|---|---|---|---|")
    for _, b in base.iterrows():
        L.append(f"| {b.baseline} | {b.sharpe:+.2f} [{b.sharpe_ci_low:+.2f}, {b.sharpe_ci_high:+.2f}] | "
                 f"{b.total_return_pct:+.1f}% | {b.max_dd_pct:.1f}% | {b.get('turnover_per_year', 0):.0f}x |")
    L.append("")

    L.append("## Full grid\n")
    L.append("| Model | H | Policy | AUC | LL skill % | IC | IC t | Gross Sharpe | Net Sharpe | CI low | CI high | Return % | MDD % | Turnover/yr |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in res.sort_values(["model", "horizon", "policy"]).iterrows():
        L.append(f"| {r.model} | {r.horizon} | {r.policy} | {r.auc:.4f} | {r.logloss_skill_pct:+.3f} | "
                 f"{r.ic_mean:+.4f} | {r.ic_t:.2f} | {r.gross_sharpe:+.2f} | {r.sharpe:+.2f} | {r.sharpe_ci_low:+.2f} | {r.sharpe_ci_high:+.2f} | "
                 f"{r.total_return_pct:+.1f} | {r.max_dd_pct:.1f} | {r.turnover_per_year:.0f} |")
    L.append("\n## How to read this\n")
    L.append("- **AUC 0.50 / IC 0 = no skill.** In liquid crypto, real hourly-to-daily edges are small "
             "(AUC 0.51–0.53, IC 0.01–0.05). Much larger numbers usually mean a bug or leakage.")
    L.append("- **Log-loss skill > 0** means the probabilities beat the naive 'up-rate from training data' forecast.")
    L.append("- **Sharpe CI** comes from a 20-day block bootstrap of daily returns. If it includes 0, the "
             "strategy is not distinguishable from luck over this window.")
    L.append("- Compare against **buy & hold** and **momentum**: a model that cannot beat a one-line rule is not adding value.")
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
