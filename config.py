# =============================================================================
# config.py — API credentials, strategy settings, and risk parameters
# All bot-wide constants live here so every module can import them cleanly.
#
# Secrets (API keys, passwords, tokens) are loaded from a .env file so they
# are never hardcoded in source code.  Create your .env by copying .env.example
# and filling in the real values.  The .env file is listed in .gitignore.
# =============================================================================

import os
from pathlib import Path

# Load .env file if python-dotenv is installed (pip install python-dotenv)
# Falls back gracefully to plain environment variables if dotenv is absent.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass   # dotenv not installed — rely on os.environ set externally

def _env(key: str, default: str = "") -> str:
    """Read a string from environment, falling back to default."""
    return os.environ.get(key, default)

def _env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from environment ('true'/'false', case-insensitive)."""
    val = os.environ.get(key, "").strip().lower()
    if val == "true":  return True
    if val == "false": return False
    return default

def _env_int(key: str, default: int = 0) -> int:
    """Read an integer from environment."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default

# ─── Alpaca API Credentials ───────────────────────────────────────────────────
# Set these in your .env file — never hardcode secrets here.
ALPACA_API_KEY    = _env("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _env("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = _env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# ─── Supabase Credentials ─────────────────────────────────────────────────────
# Optional — if not set, Supabase logging is silently skipped.
SUPABASE_URL = _env("SUPABASE_URL")
SUPABASE_KEY = _env("SUPABASE_KEY")

# ─── Environment Flags ────────────────────────────────────────────────────────
PAPER_TRADING = _env_bool("PAPER_TRADING", default=True)

# ─── Instrument ───────────────────────────────────────────────────────────────
# QQQ = Invesco QQQ Trust ETF — mirrors NASDAQ-100 (close proxy to MNQ)
SYMBOL = _env("SYMBOL", "QQQ")

# ─── Tick / Contract Specification ───────────────────────────────────────────
# For stocks/ETFs there are no "ticks" per se.  We model each $0.01 move
# as one "tick" so that the existing risk math works unchanged:
#
#   offset (points) = STOP_LOSS_TICKS × TICK_SIZE
#   P&L (dollars)   = (points_pnl / TICK_SIZE) × TICK_VALUE_DOLLARS × ORDER_QTY
#                   = points_pnl × ORDER_QTY   (because TICK_VALUE = TICK_SIZE)
#
# Example:  STOP_LOSS_TICKS=100 → 100 × $0.01 = $1.00/share SL offset
TICK_SIZE          = 0.01    # $0.01 per tick for stocks/ETFs
TICK_VALUE_DOLLARS = 0.01    # $0.01 per share per tick

# ─── Strategy Parameters ──────────────────────────────────────────────────────
EMA_FAST   = 9    # fast EMA period
EMA_SLOW   = 21   # slow EMA period used as trend filter
RSI_PERIOD = 14   # RSI lookback

RSI_BUY_THRESHOLD  = 45   # go LONG only when RSI < this (not overbought; QQQ-tuned)
RSI_SELL_THRESHOLD = 55   # go SHORT only when RSI > this (not oversold; QQQ-tuned)

BAR_TIMEFRAME = 1    # minutes per bar (1-minute bars)
BARS_TO_FETCH = 100  # number of historical bars to pull for indicator warmup

# ─── Order Size ───────────────────────────────────────────────────────────────
ORDER_QTY = 10   # shares per trade (10 shares × ~$450/share ≈ $4,500 notional)

# ─── Stop-Loss / Take-Profit ──────────────────────────────────────────────────
# Expressed in "ticks" (cents for a stock).  Converted to price offset at runtime.
#   100 ticks × $0.01 = $1.00 per share stop distance
#   200 ticks × $0.01 = $2.00 per share profit target
STOP_LOSS_TICKS   = 100   # $1.00 / share stop distance
TAKE_PROFIT_TICKS = 300   # $3.00 / share profit target  ← 1:3 R:R (backtested: +$64 vs -$13)

# Convenience dollar references (used for display / logging only)
# These show per-share dollar amounts; multiply by ORDER_QTY for total
STOP_LOSS_DOLLARS   = STOP_LOSS_TICKS   * TICK_VALUE_DOLLARS   # $1.00 / share
TAKE_PROFIT_DOLLARS = TAKE_PROFIT_TICKS * TICK_VALUE_DOLLARS   # $3.00 / share

# ─── Daily Risk Rules ─────────────────────────────────────────────────────────
MAX_TRADES_PER_DAY = 1       # 1 trade per day (conservative paper-test limit)
DAILY_LOSS_LIMIT   = -800.0  # stop trading when daily realized P&L hits this
                              # 10 shares × $1 SL = $10/trade max loss;
                              # $100 limit gives 10 losing trades before halt

# ─── Trading Hours (US Eastern Time) ─────────────────────────────────────────
# Session: 9:30 AM – 11:00 AM ET  (first 90 min — highest QQQ liquidity)
TRADING_START = (9, 30)    # (hour, minute) in US/Eastern
TRADING_END   = (11, 30)   # (hour, minute) in US/Eastern  ← extended: more time to hit TP
TIMEZONE      = "US/Eastern"

# ─── News Blackout Dates ──────────────────────────────────────────────────────
# No trading on days with high-impact macro events: NFP, FOMC, CPI
NEWS_BLACKOUT_DATES = [
    # NFP (Non-Farm Payrolls) — first Friday of each month
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-11", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    # FOMC Decision Days
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # CPI Release Days
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-10", "2025-08-12",
    "2025-09-10", "2025-10-08", "2025-11-13", "2025-12-10",
    # 2026 dates
    "2026-01-09", "2026-01-14", "2026-01-28",
    "2026-02-06", "2026-02-11", "2026-03-06",
    "2026-03-11", "2026-03-18", "2026-04-03",
    "2026-04-10", "2026-04-28",
    "2026-05-08", "2026-05-12",
]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE       = "trades.csv"   # CSV trade journal location
LOG_TO_CONSOLE = True           # print trade alerts to terminal

# ─── Loop Settings ────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 30   # how often the main loop checks for signals (seconds)

# ─── Telegram Alerts ──────────────────────────────────────────────────────────
TELEGRAM_ENABLED   = _env_bool("TELEGRAM_ENABLED", default=False)
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = _env("TELEGRAM_CHAT_ID")
TELEGRAM_TIMEOUT   = 10

# ─── Dashboard / State File ───────────────────────────────────────────────────
STATE_FILE                = "bot_state.json"
DASHBOARD_REFRESH_SECONDS = 10

# ─── Backtester Settings ──────────────────────────────────────────────────────
BACKTEST_SYMBOL   = "QQQ"         # yfinance ticker for backtesting
BACKTEST_INTERVAL = "1h"          # "1m" (7d max), "1h" (730d), "1d" (years)
BACKTEST_PERIOD   = "2y"          # yfinance period string
BACKTEST_START    = ""            # "YYYY-MM-DD" — leave empty to use PERIOD
BACKTEST_END      = ""            # "YYYY-MM-DD" — leave empty to use PERIOD
BACKTEST_RESULTS_FILE = "backtest_results.csv"
