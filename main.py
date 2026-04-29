#!/usr/bin/env python3
# =============================================================================
# main.py — Main loop that orchestrates the entire trading bot
#
# Startup sequence:
#   1. Connect to Tradovate (or skip if PAPER_TRADING)
#   2. Begin polling loop every POLL_INTERVAL_SECONDS
#   3. Each iteration:
#       a. Check RiskManager.can_trade()  → skip if blocked
#       b. Fetch latest 1-min OHLCV bars
#       c. Evaluate strategy signals
#       d. If signal and no open trade → enter position, place SL/TP orders
#       e. If in trade → check whether SL or TP has been hit
#       f. At session end → force-close any open position
#   4. Ctrl-C exits cleanly, printing the daily summary
#
# Run:
#   python main.py
# =============================================================================

import sys
import time
import signal
import datetime
import json
import threading
from zoneinfo import ZoneInfo

from config import (
    PAPER_TRADING,
    POLL_INTERVAL_SECONDS,
    TRADING_END,
    TRADING_START,
    TIMEZONE,
    SYMBOL,
    ORDER_QTY,
    STATE_FILE,
    DAILY_LOSS_LIMIT,
)
from broker       import AlpacaBroker
from strategy     import Strategy
from risk_manager import RiskManager
from logger       import TradeLogger

# Telegram — optional; silently skipped if unconfigured
try:
    import telegram_alerts as _tg
    _TG = True
except Exception:
    _TG = False


# ─── In-memory log ring buffer (shared with dashboard via STATE_FILE) ─────────
_log_buffer: list = []
_LOG_MAX = 60

# ─── Resilience settings ──────────────────────────────────────────────────────
_HEARTBEAT_INTERVAL = 300    # seconds between heartbeat log lines (5 min)
_WATCHDOG_THRESHOLD = 1800   # seconds of silence during session before alert (30 min)

_last_heartbeat: float = 0.0  # epoch-seconds of last heartbeat emission
_last_activity:  float = 0.0  # epoch-seconds of last main-loop iteration

def _log(msg: str):
    """Print to console and append to the rolling log buffer."""
    print(msg)
    now_str = datetime.datetime.now(tz=_tz).strftime("%H:%M:%S")
    _log_buffer.append({"time": now_str, "msg": msg})
    if len(_log_buffer) > _LOG_MAX:
        _log_buffer.pop(0)


# ─── Active trade tracking (module-level so the signal handler can access it)─
_active_trade: dict = {
    "open":                False,
    "trade_id":            None,
    "side":                None,    # "Long" or "Short"
    "entry_price":         None,
    "stop_price":          None,
    "target_price":        None,
    "breakeven_activated": False,   # True once SL is moved to entry price
}

# ─── Shared singleton objects ─────────────────────────────────────────────────
broker   = AlpacaBroker()
strategy = Strategy()
risk     = RiskManager()
log      = TradeLogger()
_tz      = ZoneInfo(TIMEZONE)


# ─────────────────────────────────────────────────────────────────────────────
# Heartbeat — emitted outside session so Railway logs show the bot is alive
# ─────────────────────────────────────────────────────────────────────────────

def _emit_heartbeat_if_due():
    """Log a keep-alive line every 5 minutes when outside session hours."""
    global _last_heartbeat
    now = datetime.datetime.now(tz=_tz)
    ss  = now.replace(hour=TRADING_START[0], minute=TRADING_START[1], second=0, microsecond=0)
    se  = now.replace(hour=TRADING_END[0],   minute=TRADING_END[1],   second=0, microsecond=0)
    if ss <= now <= se:
        return   # inside session — heartbeat not needed
    if time.time() - _last_heartbeat >= _HEARTBEAT_INTERVAL:
        _log(f"[HEARTBEAT] Bot alive — waiting for session  ({now.strftime('%H:%M ET')})")
        _last_heartbeat = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog — background thread that alerts if loop stops during session hours
# ─────────────────────────────────────────────────────────────────────────────

def _watchdog_thread():
    """
    Daemon thread: if the main loop stops updating _last_activity for
    _WATCHDOG_THRESHOLD seconds while the session is open, send a Telegram
    alert and print to console so Railway flags the issue.
    """
    while True:
        time.sleep(60)   # check every minute
        now = datetime.datetime.now(tz=_tz)
        ss  = now.replace(hour=TRADING_START[0], minute=TRADING_START[1], second=0, microsecond=0)
        se  = now.replace(hour=TRADING_END[0],   minute=TRADING_END[1],   second=0, microsecond=0)
        if not (ss <= now <= se):
            continue   # only watch during session hours
        elapsed = time.time() - _last_activity
        if elapsed >= _WATCHDOG_THRESHOLD:
            mins = int(elapsed // 60)
            msg  = (f"[WATCHDOG] No loop activity for {mins} min during session — "
                    "bot may be frozen!")
            print(msg)
            if _TG:
                try:
                    _tg.send_message(
                        f"⚠️ <b>WATCHDOG ALERT</b>\n"
                        f"No bot loop activity for <b>{mins} minutes</b> during session!\n"
                        f"<i>{now.strftime('%H:%M:%S ET')}</i>"
                    )
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Startup reconciliation — restore state after a crash / redeploy
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_position():
    """
    On startup, check Alpaca for an existing open position and restore
    _active_trade + RiskManager state so the bot doesn't ignore or
    double-up on a position that survived a restart.
    """
    pos = broker.get_open_position()
    if pos is None:
        _log("[MAIN] Reconcile: no open position found — starting fresh.")
        return

    side        = pos["side"]
    entry_price = pos["entry_price"]
    stop_price  = risk.get_stop_price(entry_price, side)
    target_price = risk.get_target_price(entry_price, side)

    _log(f"[MAIN] *** Reconcile: found existing {side} position @ {entry_price:.2f} ***")
    _log(f"[MAIN]     Restoring SL={stop_price:.2f}  TP={target_price:.2f}")

    # Restore risk manager state
    risk.in_trade    = True
    risk.entry_price = entry_price
    risk.trade_side  = side
    risk.trades_today += 1   # count it against today's limit

    # Register trade in the logger so close_trade() can write to CSV on exit
    trade_id = log.open_trade(
        side         = side,
        entry_price  = entry_price,
        indicators   = {},
        paper        = PAPER_TRADING,
        stop_price   = stop_price,
        target_price = target_price,
    )
    _log(f"[MAIN]     Reconciled trade logged as id #{trade_id}")

    # Restore active trade tracking dict
    _active_trade.update({
        "open":                True,
        "trade_id":            trade_id,
        "side":                side,
        "entry_price":         entry_price,
        "stop_price":          stop_price,
        "target_price":        target_price,
        "breakeven_activated": False,   # unknown on restart; start conservative
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_position(signal: str, current_price: float):
    """
    Execute a new position entry.

    Steps:
        1. Place the entry market order (or simulate in paper mode)
        2. Compute SL and TP price levels via RiskManager
        3. Place protective stop and limit orders
        4. Record the trade in RiskManager and Logger
        5. Update the _active_trade tracking dict
    """
    # Determine action strings for entry and the bracket orders
    if signal == "LONG":
        entry_action  = "Buy"
        stop_action   = "Sell"   # stop-loss exits a long
        target_action = "Sell"   # take-profit exits a long
        side          = "Long"
    else:  # SHORT
        entry_action  = "Sell"
        stop_action   = "Buy"    # stop-loss covers a short
        target_action = "Buy"    # take-profit covers a short
        side          = "Short"

    # ── Step 1: Compute approx SL / TP from current price ────────────────────
    # These are used as bracket legs on the entry order.  After fill we
    # recompute from the actual fill price for software-level monitoring.
    approx_stop   = risk.get_stop_price(current_price, side)
    approx_target = risk.get_target_price(current_price, side)

    # ── Step 2: Entry market order (with bracket SL/TP legs) ─────────────────
    # Submitting SL and TP as bracket legs in a single order prevents
    # Alpaca's "insufficient qty" error that occurs when two independent
    # closing orders compete for the same position shares.
    entry_order = broker.place_market_order(
        entry_action, qty=ORDER_QTY,
        stop_price=approx_stop, target_price=approx_target,
    )
    if entry_order is None:
        _log("[MAIN] Entry order failed — skipping this bar.")
        return

    # Use the actual fill price returned by the broker
    fill_price = entry_order.get("fillPrice", current_price)

    # ── Step 3: Recompute exact SL / TP from actual fill price ───────────────
    # Used for software-level exit monitoring in run_iteration().
    stop_price   = risk.get_stop_price(fill_price, side)
    target_price = risk.get_target_price(fill_price, side)

    _log(f"[MAIN] Entry: {side} @ {fill_price:.2f}  |  "
         f"SL: {stop_price:.2f}  |  TP: {target_price:.2f}")

    # ── Step 4: Record in RiskManager and Logger ──────────────────────────────
    risk.record_trade_open(fill_price, side)

    indicator_snapshot = strategy.last_values
    trade_id = log.open_trade(
        side         = side,
        entry_price  = fill_price,
        indicators   = {
            "ema_fast": indicator_snapshot.get("ema_fast", 0.0),
            "ema_slow": indicator_snapshot.get("ema_slow", 0.0),
            "rsi":      indicator_snapshot.get("rsi", 0.0),
        },
        paper        = PAPER_TRADING,
        stop_price   = stop_price,
        target_price = target_price,
    )

    # ── Step 5: Update tracking state ────────────────────────────────────────
    _active_trade.update({
        "open":                True,
        "trade_id":            trade_id,
        "side":                side,
        "entry_price":         fill_price,
        "stop_price":          stop_price,
        "target_price":        target_price,
        "breakeven_activated": False,
    })

    # ── Step 6: Telegram alert — full detail including SL/TP ─────────────────
    if _TG:
        try:
            indicator_snap = strategy.last_values
            _tg.alert_trade_opened(
                trade_id     = trade_id,
                side         = side,
                entry_price  = fill_price,
                stop_price   = stop_price,
                target_price = target_price,
                indicators   = {
                    "ema_fast": indicator_snap.get("ema_fast", 0.0),
                    "ema_slow": indicator_snap.get("ema_slow", 0.0),
                    "rsi":      indicator_snap.get("rsi", 0.0),
                },
            )
        except Exception:
            pass


def close_position(reason: str, exit_price: float):
    """
    Close the open position and record the result.

    reason — "STOP" | "TARGET" | "SESSION_END" | "MANUAL"
    """
    if not _active_trade["open"]:
        return

    # Cancel any remaining SL/TP bracket orders
    broker.cancel_all_orders()

    # Market-close the position
    broker.close_position()

    # Record in RiskManager
    risk.record_trade_close(_active_trade["exit_price"]
                             if "exit_price" in _active_trade
                             else exit_price)

    # Flush to CSV log
    log.close_trade(
        trade_id             = _active_trade["trade_id"],
        exit_price           = exit_price,
        exit_reason          = reason,
        daily_pnl            = risk.daily_pnl,
        breakeven_activated  = _active_trade.get("breakeven_activated", False),
    )

    # Reset tracking state
    _active_trade.update({
        "open":                False,
        "trade_id":            None,
        "side":                None,
        "entry_price":         None,
        "stop_price":          None,
        "target_price":        None,
        "breakeven_activated": False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main loop iteration
# ─────────────────────────────────────────────────────────────────────────────

def run_iteration():
    """
    Execute one full cycle of the trading loop:
        • If in a trade: check for SL/TP hit or session-end exit
        • If flat: check for new entry signal
    """
    now = datetime.datetime.now(tz=_tz)

    # ── News calendar — always fetch once per day regardless of session state ──
    # Ensures the dashboard shows today's news events even outside session hours
    # (e.g. pre-market Railway restarts).  No-op after the first call each day.
    risk.check_news_calendar()

    # ── Session bounds ────────────────────────────────────────────────────────
    session_start = now.replace(hour=TRADING_START[0], minute=TRADING_START[1],
                                second=0, microsecond=0)
    session_end   = now.replace(hour=TRADING_END[0],   minute=TRADING_END[1],
                                second=0, microsecond=0)
    in_session    = session_start <= now <= session_end

    # ── Outside session: close any stale position, then wait ─────────────────
    # Handles both post-session restarts (after 11:30 ET) AND pre-session
    # restarts (e.g. 4 AM Railway redeploy) so we never trade outside hours.
    if not in_session:
        if _active_trade["open"]:
            boundary = ("Session end" if now > session_end
                        else f"Pre-session restart (session opens {TRADING_START[0]:02d}:{TRADING_START[1]:02d} ET)")
            _log(f"[MAIN] {boundary} — force-closing open position.")
            current_price = broker.get_current_price() or _active_trade["entry_price"]
            close_position("SESSION_END", current_price)
        return   # nothing to do outside session hours

    # ── If currently in a trade: monitor for SL/TP ───────────────────────────
    if _active_trade["open"]:
        current_price = broker.get_current_price()
        if current_price is None:
            _log("[MAIN] Could not get current price — skipping exit check.")
            return

        # ── Breakeven stop-loss activation ────────────────────────────────────
        # When unrealized profit reaches +$15 (≈15 ticks × 10 shares), move
        # the stop-loss to entry price so the trade cannot turn into a loss.
        entry_price = _active_trade["entry_price"]
        side        = _active_trade["side"]
        if side == "Long":
            unrealized_pnl = (current_price - entry_price) * ORDER_QTY
        else:  # Short
            unrealized_pnl = (entry_price - current_price) * ORDER_QTY

        be_status = "ACTIVE ✅" if _active_trade["breakeven_activated"] else "pending"
        _log(f"[MAIN] Position monitor: unrealized=${unrealized_pnl:+.2f}  breakeven={be_status}")

        if unrealized_pnl >= 15.0 and not _active_trade["breakeven_activated"]:
            # Compute breakeven price: 1 cent beyond entry so a flat exit is
            # still a microscopic win rather than a dead-flat fill.
            new_sl = round(entry_price + 0.01, 2) if side == "Long" \
                     else round(entry_price - 0.01, 2)

            # Cancel existing bracket orders and replace SL at breakeven.
            # The TP level is unchanged — software monitoring continues to enforce it.
            broker.cancel_all_orders()
            stop_action = "Sell" if side == "Long" else "Buy"
            broker.place_stop_order(stop_action, new_sl)

            _active_trade["stop_price"]          = new_sl
            _active_trade["breakeven_activated"] = True

            _log(
                f"[MAIN] ✅ Breakeven activated — SL moved to {new_sl:.2f}  "
                f"(entry {entry_price:.2f}), risk now $0"
            )

        exit_reason = strategy.check_exit(
            current_price = current_price,
            stop_price    = _active_trade["stop_price"],
            target_price  = _active_trade["target_price"],
            side          = _active_trade["side"],
        )

        if exit_reason:
            close_position(exit_reason, current_price)
        return   # don't look for new entries while in a trade

    # ── Flat: check whether we're allowed to look for a signal ───────────────
    if not risk.can_trade():
        return   # RiskManager printed the reason if relevant

    # ── Fetch bars and evaluate strategy ─────────────────────────────────────
    bars = broker.fetch_bars()
    if bars.empty:
        _log("[MAIN] No bar data returned — skipping this iteration.")
        return

    signal = strategy.evaluate(bars)

    if signal in ("LONG", "SHORT"):
        # Use the last close as the approximate market price
        current_price = bars["close"].iloc[-1]
        open_position(signal, current_price)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

def write_state():
    """
    Write a lightweight JSON snapshot of the current bot state to STATE_FILE.
    The Streamlit dashboard reads this file each refresh cycle.
    Written after every run_iteration() call — failures are silently ignored.
    """
    now           = datetime.datetime.now(tz=_tz)
    # Use effective_start (may be delayed by news calendar) for session gate
    eff_start     = risk.effective_start
    session_start = now.replace(hour=eff_start[0], minute=eff_start[1],
                                second=0, microsecond=0)
    session_end   = now.replace(hour=TRADING_END[0], minute=TRADING_END[1],
                                second=0, microsecond=0)
    in_session    = session_start <= now <= session_end

    # Only fetch live price during session hours — avoids triggering Alpaca
    # WebSocket session collisions every 30 s while the market is closed.
    last_price = broker.get_current_price() if in_session else None

    state = {
        "timestamp":      now.isoformat(),
        "last_price":     last_price,
        "in_trade":       _active_trade["open"],
        "position_side":  _active_trade.get("side"),
        "entry_price":    _active_trade.get("entry_price"),
        "stop_price":     _active_trade.get("stop_price"),
        "target_price":   _active_trade.get("target_price"),
        "daily_pnl":      risk.daily_pnl,
        "trades_today":   risk.trades_today,
        "trading_halted": risk.trading_halted,
        "session_active": in_session,
        "indicators":     {
            k: (float(v) if hasattr(v, "__float__") else v)
            for k, v in strategy.last_values.items()
            if k in ("ema_fast", "ema_slow", "rsi", "close")
        },
        "market_regime":  strategy.last_regime,
        "recent_logs":    list(_log_buffer[-25:]),
        # ── News calendar snapshot (populated once per day by RiskManager) ──
        "news_events":          risk.news_events,
        "news_day_blocked":     risk.day_blocked,
        "news_session_delayed": risk.session_delayed,
        "news_effective_start": f"{eff_start[0]:02d}:{eff_start[1]:02d}",
    }

    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except OSError:
        pass   # disk write errors must never halt trading


def shutdown(signum=None, frame=None):
    """Handle Ctrl-C or SIGTERM: close any open position and print summary."""
    print("\n[MAIN] Shutdown signal received — cleaning up…")

    if _active_trade["open"]:
        print("[MAIN] Open position detected — closing before exit.")
        current_price = broker.get_current_price() or _active_trade["entry_price"]
        close_position("MANUAL", current_price)

    # Send Telegram shutdown + daily summary alerts
    if _TG:
        try:
            _tg.alert_bot_stopped()
            # Build summary from risk manager state
            _tg.alert_daily_summary(
                date_str   = datetime.datetime.now(tz=_tz).date().isoformat(),
                num_trades = risk.trades_today,
                num_wins   = 0,   # approximate — full win count needs CSV parse
                total_pnl  = risk.daily_pnl,
            )
        except Exception:
            pass

    log.print_daily_summary()
    print("[MAIN] Bot stopped.")
    sys.exit(0)


# Register signal handlers so Ctrl-C exits cleanly
signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Startup banner ────────────────────────────────────────────────────────
    mode_label = "PAPER TRADING" if PAPER_TRADING else "*** LIVE TRADING ***"
    print(
        f"\n{'#' * 60}\n"
        f"  QQQ / MNQ-Proxy Trading Bot (Alpaca)\n"
        f"  Symbol:  {SYMBOL}\n"
        f"  Mode:    {mode_label}\n"
        f"  Poll:    every {POLL_INTERVAL_SECONDS}s\n"
        f"  Session: {TRADING_END[0] - (TRADING_END[0] - 9):02d}:30 – "
        f"{TRADING_END[0]:02d}:{TRADING_END[1]:02d} ET\n"
        f"{'#' * 60}\n"
    )

    # ── Connect to broker ─────────────────────────────────────────────────────
    if not broker.connect():
        print("[MAIN] Failed to connect to Alpaca. Check .env credentials.")
        if not PAPER_TRADING:
            sys.exit(1)
        # In paper mode we can continue without a live connection

    # ── Reconcile any position that survived a restart / redeploy ────────────
    reconcile_position()

    print(f"[MAIN] Risk status: {risk.status_summary()}")

    # ── Log current session status so Railway shows why we may be waiting ────
    _now = datetime.datetime.now(tz=_tz)
    _ss  = _now.replace(hour=TRADING_START[0], minute=TRADING_START[1], second=0, microsecond=0)
    _se  = _now.replace(hour=TRADING_END[0],   minute=TRADING_END[1],   second=0, microsecond=0)
    if _ss <= _now <= _se:
        print(f"[MAIN] Session is OPEN — trading until {TRADING_END[0]:02d}:{TRADING_END[1]:02d} ET.")
    elif _now < _ss:
        print(f"[MAIN] Pre-session — waiting for {TRADING_START[0]:02d}:{TRADING_START[1]:02d} ET open.")
    else:
        print(f"[MAIN] Session CLOSED for today — waiting for next trading day.")

    print(f"[MAIN] Starting main loop.  Press Ctrl-C to stop.\n")

    # ── Regime backtest against recent trade history ───────────────────────────
    strategy.run_regime_backtest(n=8)

    # Notify Telegram that the bot is live
    if _TG:
        try:
            _tg.alert_bot_started()
        except Exception:
            pass

    # ── Start watchdog daemon thread ──────────────────────────────────────────
    _wdog = threading.Thread(target=_watchdog_thread, name="watchdog", daemon=True)
    _wdog.start()

    # ── Polling loop ──────────────────────────────────────────────────────────
    global _last_activity
    while True:
        _last_activity = time.time()   # watchdog resets on every iteration
        try:
            run_iteration()
        except KeyboardInterrupt:
            shutdown()
        except Exception as exc:
            # Log unexpected errors but keep running — resilience is critical
            _log(f"[MAIN] Unhandled error in run_iteration: {exc}")
            import traceback
            traceback.print_exc()

        # Write state snapshot for the dashboard after every iteration
        write_state()

        # Emit heartbeat log line if outside session and 5 min have elapsed
        _emit_heartbeat_if_due()

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
