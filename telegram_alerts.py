# =============================================================================
# telegram_alerts.py — Telegram Bot API integration
#
# Sends formatted alerts to your Telegram chat whenever:
#   • A trade is opened
#   • A trade is closed (WIN or LOSS)
#   • The daily loss limit is hit
#   • The end-of-day summary is ready
#
# Setup (2 minutes):
#   1. Open Telegram → search @BotFather → send /newbot → follow prompts
#   2. Copy the BOT_TOKEN you receive
#   3. Start a chat with your new bot, send any message
#   4. Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
#      Find "chat" → "id" in the JSON — that is your CHAT_ID
#   5. Fill in TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in config.py
#   6. Set TELEGRAM_ENABLED = True in config.py
#
# All send functions silently no-op when TELEGRAM_ENABLED = False,
# so you can deploy with alerts off and enable them later without code changes.
# =============================================================================

from __future__ import annotations

import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any

from config import (
    TELEGRAM_ENABLED,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TIMEOUT,
    TIMEZONE,
    SYMBOL,
    ORDER_QTY,
    PAPER_TRADING,
    DAILY_LOSS_LIMIT,
)

# Telegram Bot API base URL (uses HTTPS, no extra library needed)
_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

_TZ = ZoneInfo(TIMEZONE)


# ─── Core sender ─────────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """
    Send a plain text (or HTML-formatted) message to your configured chat.

    Args:
        text:       Message content.  HTML tags like <b>, <i>, <code> work.
        parse_mode: "HTML" (default) or "Markdown"

    Returns:
        True on success, False on any error.
        Errors are printed to console but never raise exceptions so they
        cannot crash the trading loop.
    """
    # Hard no-op if Telegram is disabled or credentials are placeholders
    if not TELEGRAM_ENABLED:
        return False
    if "YOUR_" in TELEGRAM_BOT_TOKEN or "YOUR_" in str(TELEGRAM_CHAT_ID):
        print("[TELEGRAM] Credentials not set — skipping send.")
        return False

    url     = _API_BASE.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=TELEGRAM_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.exceptions.Timeout:
        print("[TELEGRAM] Send timed out — continuing.")
        return False
    except Exception as exc:
        print(f"[TELEGRAM] Send failed: {exc}")
        return False


# ─── Formatted alert helpers ─────────────────────────────────────────────────

def alert_trade_opened(
    trade_id:    int,
    side:        str,
    entry_price: float,
    stop_price:  float,
    target_price: float,
    indicators:  Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Send a trade-opened alert with entry details and SL/TP levels.

    Example message:
        🟢 TRADE OPENED — LONG
        Symbol:  MNQM6  ×1
        Entry:   19,200.00
        Stop:    19,197.50  (-2.50 pts)
        Target:  19,205.00  (+5.00 pts)
        EMA9: 19,198  EMA21: 19,185  RSI: 42.3
        Time: 09:45:12 ET  [PAPER]
    """
    indicators = indicators or {}
    now        = datetime.now(tz=_TZ).strftime("%H:%M:%S ET")
    mode_tag   = "📄 PAPER" if PAPER_TRADING else "💰 LIVE"
    direction  = "🟢" if side == "Long" else "🔴"

    sl_pts = abs(entry_price - stop_price)
    tp_pts = abs(target_price - entry_price)

    ema_fast = indicators.get("ema_fast", 0.0)
    ema_slow = indicators.get("ema_slow", 0.0)
    rsi      = indicators.get("rsi", 0.0)

    text = (
        f"{direction} <b>TRADE OPENED — {side.upper()}</b>  [{mode_tag}]\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"<b>Symbol:</b>  {SYMBOL}  ×{ORDER_QTY}\n"
        f"<b>Entry:</b>   {entry_price:,.2f}\n"
        f"<b>Stop:</b>    {stop_price:,.2f}  <i>(-{sl_pts:.2f} pts)</i>\n"
        f"<b>Target:</b>  {target_price:,.2f}  <i>(+{tp_pts:.2f} pts)</i>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"EMA9: {ema_fast:.2f}  EMA21: {ema_slow:.2f}  RSI: {rsi:.1f}\n"
        f"<i>#{trade_id}  @  {now}</i>"
    )
    return send_message(text)


def alert_trade_closed(
    trade_id:    int,
    side:        str,
    entry_price: float,
    exit_price:  float,
    exit_reason: str,
    pnl_dollars: float,
    daily_pnl:   float,
) -> bool:
    """
    Send a trade-closed alert with result and running daily P&L.

    Example message:
        ✅ TRADE CLOSED — WIN
        LONG MNQM6  #1
        Entry → Exit:  19,200.00 → 19,205.00
        Reason:  TARGET
        P&L:    +$2.50  |  Daily: +$2.50
        Time: 09:52:00 ET
    """
    now     = datetime.now(tz=_TZ).strftime("%H:%M:%S ET")
    outcome = "WIN ✅" if pnl_dollars >= 0 else "LOSS ❌"
    emoji   = "✅" if pnl_dollars >= 0 else "❌"
    pnl_str = f"+${pnl_dollars:.2f}" if pnl_dollars >= 0 else f"-${abs(pnl_dollars):.2f}"
    dpnl    = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"

    reason_labels = {
        "TARGET":      "🎯 Take Profit",
        "STOP":        "🛑 Stop Loss",
        "SESSION_END": "⏱ Session End",
        "MANUAL":      "✋ Manual",
    }
    reason_label = reason_labels.get(exit_reason, exit_reason)

    text = (
        f"{emoji} <b>TRADE CLOSED — {outcome}</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"<b>{side} {SYMBOL}</b>  ×{ORDER_QTY}  <i>(#{trade_id})</i>\n"
        f"<b>Entry → Exit:</b>  {entry_price:,.2f} → {exit_price:,.2f}\n"
        f"<b>Reason:</b>  {reason_label}\n"
        f"<b>P&L:</b>     <b>{pnl_str}</b>  |  Daily: {dpnl}\n"
        f"<i>@ {now}</i>"
    )
    return send_message(text)


def alert_daily_loss_limit(daily_pnl: float) -> bool:
    """Send an urgent alert when the daily loss limit is breached."""
    text = (
        f"🚨 <b>DAILY LOSS LIMIT HIT</b> 🚨\n"
        f"Daily P&L has reached <b>${daily_pnl:.2f}</b>\n"
        f"Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
        f"<b>Trading halted for today.</b>\n"
        f"<i>{datetime.now(tz=_TZ).strftime('%Y-%m-%d %H:%M:%S ET')}</i>"
    )
    return send_message(text)


def alert_daily_summary(
    date_str:    str,
    num_trades:  int,
    num_wins:    int,
    total_pnl:   float,
) -> bool:
    """
    Send end-of-day summary after the session closes or bot shuts down.
    Called automatically by main.py on shutdown.
    """
    win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0
    pnl_str  = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"
    emoji    = "📈" if total_pnl >= 0 else "📉"

    text = (
        f"{emoji} <b>Daily Summary — {date_str}</b>\n"
        f"┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        f"Trades:    {num_trades}  (Wins: {num_wins}  Losses: {num_trades - num_wins})\n"
        f"Win Rate:  {win_rate:.0f}%\n"
        f"Total P&L: <b>{pnl_str}</b>\n"
        f"Symbol:    {SYMBOL}"
    )
    return send_message(text)


def alert_bot_started() -> bool:
    """Notify on startup so you know the bot is live."""
    mode = "📄 PAPER TRADING" if PAPER_TRADING else "💰 LIVE TRADING"
    text = (
        f"🤖 <b>MNQ Bot Started</b>\n"
        f"Mode:    {mode}\n"
        f"Symbol:  {SYMBOL}\n"
        f"Session: 09:30–11:30 ET\n"
        f"<i>{datetime.now(tz=_TZ).strftime('%Y-%m-%d %H:%M:%S ET')}</i>"
    )
    return send_message(text)


def alert_bot_stopped() -> bool:
    """Notify on clean shutdown."""
    text = (
        f"🛑 <b>MNQ Bot Stopped</b>\n"
        f"<i>{datetime.now(tz=_TZ).strftime('%Y-%m-%d %H:%M:%S ET')}</i>"
    )
    return send_message(text)


# ─── Quick test (run: python telegram_alerts.py) ─────────────────────────────
if __name__ == "__main__":
    print("Testing Telegram connection…")
    ok = send_message("✅ MNQ Bot Telegram test — connection working!")
    if ok:
        print("Message sent successfully. Check your Telegram chat.")
    else:
        print("Failed. Check TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, and TELEGRAM_ENABLED in config.py.")
