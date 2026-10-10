"""How much does the price move against us in the first minute after the signal bar closes?
Backtest assumes we buy exactly at the 1h close. Live, the bot acts ~15-45 s later.
Uses Binance 1-minute klines for a random sample of entries."""
import sys
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from src.core.history import _get, SPOT_URL

OUT = "reports/research/exp2-u20"
tr = pd.read_csv(f"{OUT}/trades_lgbm_h1_cost_lf.csv", parse_dates=["entry"])
sample = tr.sample(n=min(int(sys.argv[1]) if len(sys.argv) > 1 else 400, len(tr)), random_state=0)
closes = {}
for s in sample.symbol.unique():
    o = pd.read_csv(f"data/history/{s.replace('/', '')}_1h.csv", index_col=0)
    o.index = pd.to_datetime(o.index, utc=True)
    closes[s] = o["close"]

def job(row):
    t_close = row.entry + pd.Timedelta(hours=1)
    k = _get(SPOT_URL, {"symbol": row.symbol.replace("/", ""), "interval": "1m",
                        "startTime": int(t_close.value // 1_000_000), "limit": 2})
    c1h = closes[row.symbol].loc[row.entry]
    return {"symbol": row.symbol, "entry": row.entry, "gross_bps": 1e4 * row.gross,
            "move_at_open_bps": 1e4 * np.log(float(k[0][1]) / c1h),
            "move_after_1m_bps": 1e4 * np.log(float(k[0][4]) / c1h),
            "move_after_2m_bps": 1e4 * np.log(float(k[1][4]) / c1h)}

with ThreadPoolExecutor(8) as ex:
    res = pd.DataFrame(list(ex.map(job, [r for _, r in sample.iterrows()])))
res.to_csv(f"{OUT}/latency_sample.csv", index=False)
print(f"n={len(res)} long entries; positive = price already moved UP (against a buyer) after the close")
for c in ["move_at_open_bps", "move_after_1m_bps", "move_after_2m_bps"]:
    x = res[c]
    print(f"  {c:20s} mean={x.mean():+.1f}  median={x.median():+.1f}  se={x.std()/np.sqrt(len(x)):.1f}")
print(f"  backtest gross per trade in this sample: {res.gross_bps.mean():+.1f} bps")
