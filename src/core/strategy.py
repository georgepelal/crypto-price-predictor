"""Turn out-of-sample probabilities into positions and net returns.

Timing: position decided at the close of row t is held over the next bar and earns ret_next[t].
Costs: ``cost_per_side`` (fee + slippage) charged on every unit of position change.
Funding: positions are assumed to be USDT-M perpetuals, so longs pay / shorts receive the funding
rate (spread evenly over the 8h between settlements).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def raw_signal(prob_up: pd.Series, margin: float, allow_short: bool = True) -> pd.Series:
    sig = pd.Series(0.0, index=prob_up.index)
    sig[prob_up > 0.5 + margin] = 1.0
    if allow_short:
        sig[prob_up < 0.5 - margin] = -1.0
    return sig


def positions_from_signal(sig: pd.Series, horizon: int) -> pd.Series:
    """Overlapping-portfolio construction: each hourly signal is held for ``horizon`` bars with
    weight 1/horizon, i.e. position = mean of the last ``horizon`` signals. Matches the label
    horizon and keeps turnover low."""
    if horizon <= 1:
        return sig
    return sig.rolling(horizon, min_periods=1).mean()


def simulate_symbol(
    frame: pd.DataFrame,
    position: pd.Series,
    cost_per_side: float,
    funding_hourly: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """frame: one symbol, indexed by timestamp, needs ret_next. Returns per-bar P&L components."""
    pos = position.reindex(frame.index).fillna(0.0)
    gross = pos * (np.exp(frame["ret_next"].fillna(0.0)) - 1.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = turnover * cost_per_side
    fund = pd.Series(0.0, index=frame.index)
    if funding_hourly is not None:
        fund = -pos * funding_hourly.reindex(frame.index).fillna(0.0)
    net = gross - costs + fund
    return pd.DataFrame({"pos": pos, "gross": gross, "cost": costs, "funding": fund,
                         "net": net, "turnover": turnover})


def hourly_funding(funding: Optional[pd.DataFrame], index: pd.DatetimeIndex) -> Optional[pd.Series]:
    """Funding cost per unit position for each bar. Bar t covers (t, t+1h]; a settlement at S
    covers (S-8h, S], so bar t belongs to settlement ceil(t+1h, 8h). Each bar pays rate/8."""
    if funding is None or funding.empty:
        return None
    fr = funding["funding_rate"].sort_index()
    fr = fr[~fr.index.duplicated(keep="last")]
    settle = (index + pd.Timedelta(hours=1)).ceil("8h")
    vals = fr.reindex(settle).values / 8.0
    return pd.Series(vals, index=index).fillna(0.0)


def run_portfolio(
    oos: pd.DataFrame,
    positions: Dict[str, pd.Series],
    cost_per_side: float,
    funding: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, object]:
    """Equal capital per symbol. Returns per-symbol P&L frames and the portfolio hourly return."""
    per_symbol = {}
    for sym, g in oos.groupby("symbol"):
        frame = g.set_index("timestamp").sort_index()
        fh = hourly_funding((funding or {}).get(sym), frame.index)
        per_symbol[sym] = simulate_symbol(frame, positions[sym], cost_per_side, fh)
    nets = pd.DataFrame({s: d["net"] for s, d in per_symbol.items()}).fillna(0.0)
    port = nets.mean(axis=1)
    return {"per_symbol": per_symbol, "portfolio": port}


def model_positions(oos: pd.DataFrame, horizon: int, margin: float, allow_short: bool = True) -> Dict[str, pd.Series]:
    out = {}
    for sym, g in oos.groupby("symbol"):
        g = g.set_index("timestamp").sort_index()
        out[sym] = positions_from_signal(raw_signal(g["prob_up"], margin, allow_short), horizon)
    return out


def baseline_positions(oos: pd.DataFrame, ds: pd.DataFrame, horizon: int, kind: str) -> Dict[str, pd.Series]:
    """buy_hold: always long. momentum: sign of trailing 24h return (time-series momentum),
    held with the same overlapping construction and costs as the model."""
    out = {}
    for sym, g in oos.groupby("symbol"):
        idx = g.set_index("timestamp").sort_index().index
        if kind == "buy_hold":
            out[sym] = pd.Series(1.0, index=idx)
        elif kind == "momentum_24h":
            r = ds[ds["symbol"] == sym].set_index("timestamp")["ret_24"].reindex(idx)
            out[sym] = positions_from_signal(np.sign(r).fillna(0.0), horizon)
        elif kind == "momentum_168h":
            r = ds[ds["symbol"] == sym].set_index("timestamp")["ret_168"].reindex(idx)
            out[sym] = positions_from_signal(np.sign(r).fillna(0.0), horizon)
        else:
            raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- cost-aware policy

def expected_return(prob_up: pd.Series, vol_hourly: pd.Series, horizon: int) -> pd.Series:
    """Expected log return over the horizon implied by P(up), assuming a symmetric return
    distribution with stdev sigma_H:  E[r] = (2p - 1) * E|r| = (2p - 1) * sigma_H * sqrt(2/pi)."""
    sigma_h = vol_hourly * np.sqrt(horizon)
    return (2.0 * prob_up - 1.0) * sigma_h * np.sqrt(2.0 / np.pi)


def step_position(cur: float, mu: float, cost_per_side: float, allow_short: bool = True,
                  entry_mult: float = 1.0) -> float:
    """One step of the cost-aware rule (used by the backtest AND the live trader, so they agree):
    * enter (or flip) only when the expected move pays for a round trip: |mu| > 2 * cost,
    * exit to flat when the expected move turns against the position (mu crosses 0),
    * otherwise keep the current position (avoids paying costs on noise)."""
    enter = 2.0 * cost_per_side * entry_mult
    if not np.isfinite(mu):
        return cur
    if mu > enter:
        return 1.0
    if allow_short and mu < -enter:
        return -1.0
    if cur > 0 and mu < 0:
        return 0.0
    if cur < 0 and mu > 0:
        return 0.0
    return cur


def cost_aware_positions(mu: pd.Series, cost_per_side: float, allow_short: bool = True,
                         entry_mult: float = 1.0) -> pd.Series:
    """No-trade band with hysteresis, no tuned parameters (see step_position)."""
    pos = np.zeros(len(mu))
    cur = 0.0
    for i, m in enumerate(mu.values):
        cur = step_position(cur, float(m), cost_per_side, allow_short, entry_mult)
        pos[i] = cur
    return pd.Series(pos, index=mu.index)


def policy_positions(oos: pd.DataFrame, horizon: int, policy: str, cost_per_side: float) -> Dict[str, pd.Series]:
    """policy: 'overlap:<margin>'  -> naive overlapping-portfolio rule with a probability margin
               'cost_ls' / 'cost_lf' -> cost-aware no-trade band, long/short or long/flat.
    ``oos`` must contain a vol_168 column for the cost-aware policies."""
    if policy.startswith("overlap:"):
        return model_positions(oos, horizon, float(policy.split(":")[1]))
    allow_short = policy == "cost_ls"
    out = {}
    for sym, g in oos.groupby("symbol"):
        g = g.set_index("timestamp").sort_index()
        mu = expected_return(g["prob_up"], g["vol_168"], horizon)
        out[sym] = cost_aware_positions(mu, cost_per_side, allow_short=allow_short)
    return out
