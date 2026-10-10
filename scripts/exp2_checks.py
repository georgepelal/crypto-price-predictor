"""Robustness checks for experiment 2 (run after research.py --universe u20 ...)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from research import UNIVERSES
from src.core import history
from src.core.strategy import policy_positions, run_portfolio
from src.core.metrics import trading_metrics, yearly_returns

OUT = "reports/research/exp2-u20"
COST = 7e-4
d = history.update_all(UNIVERSES["u20"], offline=True, verbose=False)
oos = pd.read_csv(f"{OUT}/oos_lgbm_h1.csv.gz", parse_dates=["timestamp"])
vol = []
for s in UNIVERSES["u20"]:
    o = d["ohlcv"][s]
    lr = np.log(o["close"]).diff()
    v = lr.rolling(168, min_periods=84).std()
    vol.append(pd.DataFrame({"timestamp": v.index, "symbol": s, "vol_168": v.values}))
oos = oos.merge(pd.concat(vol), on=["timestamp", "symbol"], how="left")

pos = policy_positions(oos, 1, "cost_lf", COST)
pf = run_portfolio(oos, pos, COST, d["funding"])

# 1) trade-level stats
rows = []
for s, f in pf["per_symbol"].items():
    p = f["pos"]
    run_id = (p != p.shift()).cumsum()
    for _, g in f[p > 0].groupby(run_id[p > 0]):
        rows.append({"symbol": s, "entry": g.index[0], "bars": len(g),
                     "gross": (1 + g["gross"]).prod() - 1,
                     # entry cost is inside the run; the exit cost is charged on the next bar
                     "net": (1 + g["net"]).prod() * (1 - COST) - 1})
tr = pd.DataFrame(rows)
tr["year"] = tr.entry.dt.year
print("TRADES: n=%d  avg bars held=%.1f  avg gross=%.1f bps  avg net=%.1f bps  win rate=%.1f%%" % (
    len(tr), tr.bars.mean(), 1e4 * tr.gross.mean(), 1e4 * tr.net.mean(), 100 * (tr.net > 0).mean()))
print(tr.groupby("year").agg(n=("net", "size"), gross_bps=("gross", lambda x: 1e4 * x.mean()),
                             net_bps=("net", lambda x: 1e4 * x.mean()), win=("net", lambda x: 100 * (x > 0).mean())).round(1))
inm = pd.DataFrame({s: (f["pos"] > 0).astype(float) for s, f in pf["per_symbol"].items()})
print("time in market by year (%):", (100 * inm.groupby(inm.index.year).mean().mean(axis=1)).round(1).to_dict())

# 2) top-trade concentration
srt = tr.sort_values("net", ascending=False)
tot = srt.net.sum()
print("share of summed trade P&L from top 1%% of trades: %.0f%%" % (100 * srt.net.head(max(1, len(srt) // 100)).sum() / tot))

# 3) one-bar execution delay
delayed = {s: p.shift(1).fillna(0.0) for s, p in pos.items()}
pfd = run_portfolio(oos, delayed, COST, d["funding"])
tmd = trading_metrics(pfd["portfolio"], pfd["per_symbol"], bootstrap=0)
print("1-BAR DELAY: Sharpe=%+.2f ret=%+.1f%%" % (tmd["sharpe"], tmd["total_return_pct"]))

# 4) BTC/ETH/SOL only, using the U20-trained predictions
sub = oos[oos.symbol.isin(["BTC/USDT", "ETH/USDT", "SOL/USDT"])]
pfs = run_portfolio(sub, policy_positions(sub, 1, "cost_lf", COST), COST, d["funding"])
tms = trading_metrics(pfs["portfolio"], pfs["per_symbol"])
print("U20-trained model on BTC/ETH/SOL: Sharpe=%+.2f [%+.2f,%+.2f] ret=%+.1f%% mdd=%.1f%%  yearly=%s" % (
    tms["sharpe"], tms["sharpe_ci_low"], tms["sharpe_ci_high"], tms["total_return_pct"], tms["max_dd_pct"],
    yearly_returns(pfs["portfolio"]).round(1).to_dict()))
tr.to_csv(f"{OUT}/trades_lgbm_h1_cost_lf.csv", index=False)
