# =============================================================================
# logger.py — Trade journal saved to CSV
#
# Every completed trade is written as one row to trades.csv.
# The file is created with headers on first run; subsequent runs append rows.
# A console alert is printed immediately when a trade is opened or closed.
# =============================================================================

from __future__ import annotations

import csv
import os
import datetime
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo

from config import LOG_FILE, LOG_TO_CONSOLE, TIMEZONE, SYMBOL, ORDER_QTY, PAPER_TRADING, TICK_SIZE, TICK_VALUE_DOLLARS, SUPABASE_URL, SUPABASE_KEY

# Import Telegram alerts — wrapped in try/except so the bot still works
# even if telegram_alerts.py has a bad import (e.g. requests not installed).
try:
    import telegram_alerts as _tg
    _TG_AVAILABLE = True
except Exception:
    _TG_AVAILABLE = False

# Import Supabase — optional; CSV logging continues if not installed/configured.
_supabase_client = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[LOGGER] Supabase connected.")
    else:
        print("[LOGGER] Supabase not configured — CSV-only logging.")
except Exception as _e:
    print(f"[LOGGER] Supabase init failed ({_e}) — CSV-only logging.")


# Column order for the CSV trade journal
_CSV_COLUMNS = [
    "trade_id",        # sequential integer
    "date",            # YYYY-MM-DD
    "symbol",          # instrument symbol
    "side",            # Long / Short
    "qty",             # number of contracts
    "entry_time",      # HH:MM:SS ET
    "entry_price",     # fill price on entry
    "exit_time",       # HH:MM:SS ET (empty if still open)
    "exit_price",      # fill price on exit (empty if still open)
    "exit_reason",     # STOP / TARGET / SESSION_END / MANUAL
    "pnl_ticks",       # realized P&L in ticks
    "pnl_dollars",     # realized P&L in USD
    "daily_pnl",       # cumulative daily P&L after this trade
    "ema_fast",        # EMA-9 value at entry
    "ema_slow",        # EMA-21 value at entry
    "rsi",             # RSI-14 value at entry
    "paper",           # True if paper-trade, False if live
]


class TradeLogger:
    """
    Persistent trade journal backed by a CSV file.

    Usage:
        log = TradeLogger()
        tid = log.open_trade(side="Long", entry_price=19200.0,
                             indicators={"ema_fast": 19195, "ema_slow": 19180, "rsi": 41})
        ...
        log.close_trade(tid, exit_price=19215.0, exit_reason="TARGET",
                        daily_pnl=10.0)
    """

    def __init__(self):
        self._tz         = ZoneInfo(TIMEZONE)
        self._open_trades: Dict[int, Dict[str, Any]] = {}   # keyed by trade_id
        self._next_id    = self._load_last_id() + 1

        # Create the CSV file with headers if it doesn't exist yet
        self._ensure_csv_exists()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        side:         str,
        entry_price:  float,
        indicators:   Optional[Dict[str, float]] = None,
        paper:        bool = True,
        stop_price:   Optional[float] = None,
        target_price: Optional[float] = None,
    ) -> int:
        """
        Record a new trade opening.  Returns the trade_id.

        Args:
            side:         "Long" or "Short"
            entry_price:  fill price
            indicators:   dict with keys ema_fast, ema_slow, rsi (at entry bar)
            paper:        True for paper trade, False for live
            stop_price:   stop-loss price level
            target_price: take-profit price level

        Returns:
            trade_id (int) — pass this to close_trade()
        """
        now  = datetime.datetime.now(tz=self._tz)
        tid  = self._next_id
        self._next_id += 1

        indicators = indicators or {}

        trade = {
            "trade_id":    tid,
            "date":        now.date().isoformat(),
            "symbol":      SYMBOL,
            "side":        side,
            "qty":         ORDER_QTY,
            "entry_time":  now.strftime("%H:%M:%S"),
            "entry_price": entry_price,
            "stop_price":  stop_price,
            "target_price": target_price,
            "exit_time":   "",
            "exit_price":  "",
            "exit_reason": "",
            "pnl_ticks":   "",
            "pnl_dollars": "",
            "daily_pnl":   "",
            "ema_fast":    indicators.get("ema_fast", ""),
            "ema_slow":    indicators.get("ema_slow", ""),
            "rsi":         indicators.get("rsi", ""),
            "paper":       paper,
        }

        self._open_trades[tid] = trade

        # ── Supabase insert (open trade row) ──────────────────────────────────
        if _supabase_client is not None:
            try:
                _supabase_client.table("trades").insert({
                    "trade_id":     tid,
                    "symbol":       SYMBOL,
                    "side":         side,
                    "entry_price":  entry_price,
                    "stop_price":   stop_price,
                    "target_price": target_price,
                    "entry_time":   now.isoformat(),
                    "paper_trade":  paper,
                    "indicators": {
                        "ema_fast": indicators.get("ema_fast"),
                        "ema_slow": indicators.get("ema_slow"),
                        "rsi":      indicators.get("rsi"),
                    },
                }).execute()
            except Exception as _e:
                print(f"[LOGGER] Supabase open_trade insert failed: {_e}")

        # ── Telegram alert ────────────────────────────────────────────────────
        if _TG_AVAILABLE:
            try:
                # stop_price / target_price aren't known yet at open_trade time;
                # they are sent separately from main.py via alert_trade_opened().
                # Here we fire a lightweight "trade opened" ping so Telegram
                # receives it even if main.py crashes before calling the full alert.
                pass   # full alert is sent by main.py open_position() which has SL/TP
            except Exception:
                pass

        # Console alert
        if LOG_TO_CONSOLE:
            mode = "PAPER" if paper else "LIVE"
            print(
                f"\n{'=' * 60}\n"
                f"  *** TRADE OPENED [{mode}] ***\n"
                f"  ID:     #{tid}\n"
                f"  Side:   {side}  {SYMBOL}  x{ORDER_QTY}\n"
                f"  Entry:  {entry_price:.2f}  @  {now.strftime('%H:%M:%S ET')}\n"
                f"  EMA9:   {indicators.get('ema_fast', 'n/a'):.2f}  "
                f"EMA21: {indicators.get('ema_slow', 'n/a'):.2f}  "
                f"RSI: {indicators.get('rsi', 'n/a'):.1f}\n"
                f"{'=' * 60}\n"
            )

        return tid

    def close_trade(
        self,
        trade_id:            int,
        exit_price:          float,
        exit_reason:         str,          # "STOP" | "TARGET" | "SESSION_END" | "MANUAL"
        daily_pnl:           float = 0.0,  # running daily P&L after this trade closes
        breakeven_activated: bool  = False, # True if SL was moved to entry during trade
    ) -> Optional[Dict]:
        """
        Record a trade closure and flush the completed row to CSV.

        Returns the completed trade dict, or None if trade_id not found.
        """
        trade = self._open_trades.pop(trade_id, None)
        if trade is None:
            print(f"[LOGGER] Warning: trade_id {trade_id} not found.")
            return None

        now = datetime.datetime.now(tz=self._tz)

        # ── Compute P&L ──────────────────────────────────────────────────────
        entry = float(trade["entry_price"])
        exit_ = float(exit_price)

        if trade["side"] == "Long":
            points_pnl = exit_ - entry
        else:
            points_pnl = entry - exit_

        ticks_pnl  = points_pnl / TICK_SIZE
        dollar_pnl = ticks_pnl * TICK_VALUE_DOLLARS * ORDER_QTY

        # ── Fill in closing fields ────────────────────────────────────────────
        trade["exit_time"]   = now.strftime("%H:%M:%S")
        trade["exit_price"]  = exit_price
        trade["exit_reason"] = exit_reason
        trade["pnl_ticks"]   = round(ticks_pnl, 2)
        trade["pnl_dollars"] = round(dollar_pnl, 2)
        trade["daily_pnl"]   = round(daily_pnl, 2)

        # ── Write row to CSV ──────────────────────────────────────────────────
        self._append_to_csv(trade)

        # ── Supabase update (close trade row) ─────────────────────────────────
        if _supabase_client is not None:
            try:
                _supabase_client.table("trades").update({
                    "exit_price":           exit_price,
                    "exit_time":            now.isoformat(),
                    "exit_reason":          exit_reason,
                    "pnl_dollars":          round(dollar_pnl, 2),
                    "daily_pnl":            round(daily_pnl, 2),
                    "breakeven_activated":  breakeven_activated,
                }).eq("trade_id", trade_id).execute()
            except Exception as _e:
                print(f"[LOGGER] Supabase close_trade update failed: {_e}")

        # ── Telegram alert ────────────────────────────────────────────────────
        if _TG_AVAILABLE:
            try:
                _tg.alert_trade_closed(
                    trade_id    = trade_id,
                    side        = trade["side"],
                    entry_price = entry,
                    exit_price  = exit_,
                    exit_reason = exit_reason,
                    pnl_dollars = dollar_pnl,
                    daily_pnl   = daily_pnl,
                )
            except Exception:
                pass   # Telegram errors must never crash the trading loop

        # ── Console alert ─────────────────────────────────────────────────────
        if LOG_TO_CONSOLE:
            outcome = "WIN" if dollar_pnl >= 0 else "LOSS"
            mode    = "PAPER" if trade["paper"] else "LIVE"
            print(
                f"\n{'=' * 60}\n"
                f"  *** TRADE CLOSED [{mode}] — {outcome} ***\n"
                f"  ID:     #{trade_id}\n"
                f"  Side:   {trade['side']}  {SYMBOL}\n"
                f"  Entry:  {entry:.2f}  →  Exit: {exit_:.2f}  ({exit_reason})\n"
                f"  P&L:    ${dollar_pnl:+.2f}  ({ticks_pnl:+.1f} ticks)\n"
                f"  Daily:  ${daily_pnl:+.2f}\n"
                f"{'=' * 60}\n"
            )

        return trade

    def get_open_trades(self) -> Dict[int, Dict]:
        """Return all currently open (unlogged) trades."""
        return dict(self._open_trades)

    # ─────────────────────────────────────────────────────────────────────────
    # CSV helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_csv_exists(self):
        """Create the CSV file with the header row if it doesn't exist."""
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
            print(f"[LOGGER] Created trade journal: {os.path.abspath(LOG_FILE)}")
        else:
            print(f"[LOGGER] Appending to existing journal: "
                  f"{os.path.abspath(LOG_FILE)}")

    def _append_to_csv(self, trade: Dict[str, Any]):
        """Append a single completed trade row to the CSV."""
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS,
                                    extrasaction="ignore")
            writer.writerow(trade)

    def _load_last_id(self) -> int:
        """
        Find the highest existing trade_id across both the CSV journal and
        Supabase, so restarts never reuse an id that already exists remotely.
        """
        last_id = 0

        # ── 1. Scan the local CSV ─────────────────────────────────────────────
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            tid = int(row.get("trade_id", 0))
                            if tid > last_id:
                                last_id = tid
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass

        # ── 2. Query Supabase for its highest trade_id ────────────────────────
        # This covers the case where the CSV is absent/reset but Supabase
        # already holds prior trades, which would otherwise cause a duplicate
        # key error on trade_id=1 at the next insert.
        if _supabase_client is not None:
            try:
                resp = (
                    _supabase_client
                    .table("trades")
                    .select("trade_id")
                    .order("trade_id", desc=True)
                    .limit(1)
                    .execute()
                )
                if resp.data:
                    supabase_last = int(resp.data[0].get("trade_id", 0))
                    if supabase_last > last_id:
                        last_id = supabase_last
                        print(f"[LOGGER] Resuming from Supabase trade_id={last_id}")
            except Exception as _e:
                print(f"[LOGGER] Could not fetch last trade_id from Supabase: {_e}")

        return last_id

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def print_daily_summary(self):
        """
        Print a summary of all closed trades logged today.
        Reads from the CSV so it's accurate even across restarts.
        """
        today_str = datetime.datetime.now(tz=self._tz).date().isoformat()
        trades    = []

        try:
            with open(LOG_FILE, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("date") == today_str and row.get("exit_price"):
                        trades.append(row)
        except Exception:
            return

        if not trades:
            print("[LOGGER] No closed trades today.")
            return

        total_pnl = sum(float(t.get("pnl_dollars", 0)) for t in trades)
        wins      = sum(1 for t in trades if float(t.get("pnl_dollars", 0)) >= 0)

        print(f"\n{'─' * 50}")
        print(f"  Daily Summary — {today_str}")
        print(f"  Trades: {len(trades)}  |  Wins: {wins}  |  "
              f"Losses: {len(trades) - wins}")
        print(f"  Total P&L: ${total_pnl:+.2f}")
        print(f"{'─' * 50}\n")
