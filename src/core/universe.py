"""Named symbol universes (single source of truth for research and live trading)."""

U3 = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Top-20 non-stablecoins by market cap on 2023-01-01 that traded on Binance spot (experiment 2,
# docs/RESEARCH_LOG.md). Includes coins delisted later, to avoid survivorship bias in research.
U20 = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "MATIC/USDT",
       "DOT/USDT", "LTC/USDT", "SOL/USDT", "TRX/USDT", "SHIB/USDT", "AVAX/USDT", "UNI/USDT",
       "LINK/USDT", "ATOM/USDT", "ETC/USDT", "XMR/USDT", "XLM/USDT", "BCH/USDT"]

DELISTED = {"MATIC/USDT": "2024-09-10", "XMR/USDT": "2024-02-20"}

UNIVERSES = {"u3": U3, "u20": U20}


def tradable(symbols):
    return [s for s in symbols if s not in DELISTED]
