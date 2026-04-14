#!/bin/bash
# =============================================================================
# start.sh — Railway entry point
# Starts the trading bot in the background and Streamlit in the foreground.
# Railway keeps the service alive as long as the foreground process runs.
# =============================================================================

set -e

echo "[RAILWAY] Starting MNQ/QQQ Trading Bot..."

# Launch the trading bot in the background
python main.py &
BOT_PID=$!
echo "[RAILWAY] Bot started (PID $BOT_PID)"

# Launch the Streamlit dashboard in the foreground on Railway's assigned port.
# PORT is injected automatically by Railway; default 8501 for local testing.
echo "[RAILWAY] Starting dashboard on port ${PORT:-8501}..."
exec streamlit run dashboard.py \
  --server.port "${PORT:-8501}" \
  --server.address "0.0.0.0" \
  --browser.gatherUsageStats false
