#!/bin/bash
# ==============================================================================
# Institutional Crypto Bot - 2-Week Live Paper Trading Daemon
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/data/trader.pid"
LOG_FILE="$SCRIPT_DIR/logs/live_trader.log"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"

mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/logs"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[!] Trading bot is ALREADY running with PID: $PID"
        echo "    Check logs: tail -f $LOG_FILE"
        echo "    To stop it: ./stop_live_2weeks.sh"
        exit 0
    else
        echo "[*] Stale PID file found. Cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

echo "=================================================================="
echo " Starting 2-Week Autonomous Multi-Asset Paper Trading Bot"
echo " Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Logging to: $LOG_FILE"
echo "=================================================================="

# Run caffeinate to prevent Mac from sleeping system while process runs, in background
nohup caffeinate -s "$PYTHON_BIN" multi_asset_scanner.py --watch --interval 60 >> "$LOG_FILE" 2>&1 &
NEW_PID=$!

echo "$NEW_PID" > "$PID_FILE"
echo "[+] Bot started successfully in background!"
echo "    PID: $NEW_PID"
echo "    Monitor live logs: tail -f logs/live_trader.log"
echo "    View Dashboard   : .venv/bin/python3 report_dashboard.py"
echo "    Stop the bot     : ./stop_live_2weeks.sh"
echo "=================================================================="
