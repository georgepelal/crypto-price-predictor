# Research log

Append new entries at the top. Record what was decided **before** looking at results, then what came out.
Numbers are out-of-sample, after costs (7 bps/side) and perp funding, equal weight BTC/ETH/SOL, 1h bars.
Reproduce with `.venv/bin/python research.py` (full report lands in `reports/research/<run>/summary.md`).

---

## 2026-10-10 — Experiment 2: wider universe (PRE-DECLARED, written before running)

**Question.** Does the same model/rule hold up on many more coins? 3 coins over ~4 years gives too few
independent bets to tell skill from luck; 20 coins gives ~7x more.

**Universe U20 (fixed by a rule, not by performance).** The 20 largest non-stablecoin cryptos by market
cap on **2023-01-01** (start of the test period) that traded on Binance spot against USDT:
BTC ETH BNB XRP ADA DOGE MATIC DOT LTC SOL TRX SHIB AVAX UNI LINK ATOM ETC XMR XLM BCH.
MATIC (spot data ends 2024-09, renamed POL) and XMR (delisted 2024-02) are **kept** and simply drop out
when their data ends — dropping them would be survivorship bias. Portfolio = equal weight across the
coins that exist at each hour. SHIB funding comes from the 1000SHIBUSDT perp.

**Everything else unchanged:** same 65 features, lgbm with the same fixed hyper-parameters, walk-forward
(retrain every 30 days, expanding window, purge = horizon), horizon 1h, cost-aware long/flat rule,
7 bps/side + funding, test 2023-01-01 → 2026-10.

**Primary:** lgbm · h=1 · cost_lf on U20. **Pass only if all three hold:**
1. Net Sharpe 95% bootstrap CI lower bound > 0.
2. Positive net return in at least 3 of the 4 calendar years (2023, 2024, 2025, 2026 YTD).
3. Net Sharpe still > 0.5 at 10 bps/side cost (alts are less liquid than BTC).

**Secondary (reported, not used for the decision):** (a) the U20-trained model's P&L on BTC/ETH/SOL only,
vs v1's +1.46 (does training on more coins help?); (b) logistic regression on U20.

**Decision rule.** Pass → propose moving the paper trader to U20. Fail → keep the 3-coin paper trader
as is, record the failure here, and do not try variations of this universe until a new hypothesis is written down.

---

## 2026-10-08 — Core pipeline v1 (Claude)

**Setup.** 6.8 years of Binance 1h data (2020-01 → 2026-10), 65 scale-free features (momentum,
volatility, candle shape, taker-buy flow, funding, Fear & Greed, BTC market factor), one model pooled
across the 3 coins. Walk-forward: retrain every 30 days on all prior data, purge = label horizon.
Out-of-sample test 2023-01 → 2026-10 (≈3.8 years, ~80k predictions per config).
Grid: models {LightGBM, logistic regression} × horizons {1, 4, 12, 24 h} × decision rules
{naive overlapping, cost-aware long/short, cost-aware long/flat} = 24 configs.

**Pre-declared primary.** v1: lgbm, 24h, naive overlapping rule. After v1 failed on costs, v2 was
declared *before* running it: same model/horizon, cost-aware rule.

### Finding 1 — the model has real, repeatable predictive skill
| Horizon | AUC (lgbm) | Monthly IC (t-stat) |
|---|---|---|
| 1h | 0.557 | +0.074 (15.0) |
| 4h | 0.548 | +0.066 (9.6) |
| 12h | 0.532 | +0.051 (3.8) |
| 24h | 0.524 | +0.039 (2.3) |

Same picture with logistic regression, and on a period never used for any decision
(2021-07 → 2022-12): 1h AUC 0.556, IC t = 12. Truncation tests confirm no feature uses future data.
Skill is strongest at short horizons and is driven by short-term reversal / candle-location / order-flow
features (`ret_1`, `ret_2`, `close_loc`, `taker_buy_ratio`, `bb_z_20`).

### Finding 2 — almost all of it is eaten by trading costs
* Average |1h move| ≈ 44 bps; the model's average edge per hour is 1–4 bps; a round trip costs 14 bps.
* Naive rules (trade whenever P(up) ≠ 0.5) are destroyed: 1h → −100%, 4h → −90%, 24h → −53%.
  Gross (pre-cost) Sharpe was +1.3 to +2.2 at 1–4h.

### Finding 3 — a cost-aware rule keeps a small, episodic edge
Rule (no tuned parameters, `src/core/strategy.step_position`): expected move
E[r] = (2p − 1)·σ_H·√(2/π); enter only if |E[r]| > round-trip cost; exit when E[r] changes sign.

| Config | Net Sharpe [95% CI] | Return | Max DD | Time in market |
|---|---|---|---|---|
| lgbm 1h long/flat | +1.46 [+0.40, +2.59] | +86% | −13% | 3% |
| lgbm 1h long/short | +0.97 [+0.09, +2.36] | +68% | −28% | 6% |
| v2 primary: lgbm 24h long/short | −0.49 [−1.33, +0.46] | −69% | −75% | — |
| Buy & hold | +0.99 [−0.12, +1.91] | +373% | −64% | 100% |

**Caveats (important):**
* The 1h configs were *not* the pre-declared primary; they were picked from a 24-config grid → selection bias.
* Profits are concentrated in high-volatility episodes: lgbm 1h long/flat made +78% in 2024 versus
  +5% (2023), −2% (2025) and +2% (2026 YTD). On 2021-07 → 2022-12 it made +79% in 2021 H2 and −10% in 2022
  (Sharpe +1.0, CI includes 0). Logistic regression with the same rule made ~0 there.
* Fragile to execution: acting one bar late → Sharpe +0.43; costs of 10 bps/side → +1.3;
  15 bps/side → +0.2. The live bot must act within seconds of the candle close and pay taker ≤ 5 bps.
* It does not beat buy & hold on return; its value (if real) is low drawdown and low market exposure.

### Decision
Paper-trade **lgbm, 1h, cost-aware long/flat** (`paper_trader.py`, `src/core/deploy.DEFAULT_SPEC`)
to measure live slippage/latency against the backtest. Do not risk real money on it: the honest summary
is "statistically real skill, economically marginal edge".

### Ideas not yet tested (each needs its own pre-declared test)
* Maker (limit) execution to cut costs — needs fill-probability modelling.
* Wider universe (top-20 coins) for more independent bets.
* Predicting volatility / sizing instead of direction (vol is far more predictable than sign).
* Regime gating: only trade when σ is high enough that expected moves clear costs.
