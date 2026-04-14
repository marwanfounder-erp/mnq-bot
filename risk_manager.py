# =============================================================================
# risk_manager.py — Topstep rule enforcement and daily risk tracking
#
# Responsibilities:
#   • Track trades taken today (enforce MAX_TRADES_PER_DAY)
#   • Track realized daily P&L (enforce DAILY_LOSS_LIMIT)
#   • Block trading on news-blackout dates (NFP / FOMC / CPI)
#   • Block trading outside the allowed session window
#   • Provide a single can_trade() gate that the main loop checks each bar
# =============================================================================

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo          # stdlib in Python 3.9+; no extra install
from typing import Optional

from config import (
    DAILY_LOSS_LIMIT,
    MAX_TRADES_PER_DAY,
    NEWS_BLACKOUT_DATES,
    TRADING_START,
    TRADING_END,
    TIMEZONE,
    STOP_LOSS_DOLLARS,
    TAKE_PROFIT_DOLLARS,
    TICK_SIZE,
    TICK_VALUE_DOLLARS,
    STOP_LOSS_TICKS,
    TAKE_PROFIT_TICKS,
    ORDER_QTY,
)


class RiskManager:
    """
    Enforces all Topstep and strategy-level risk rules.

    Usage:
        rm = RiskManager()
        if rm.can_trade():
            # generate signal and place order
            rm.record_trade_open(price=15200.0, side="Long")
            ...
            rm.record_trade_close(pnl=10.0)
    """

    def __init__(self):
        # ── Daily state (reset each calendar day) ──
        self._today: Optional[datetime.date] = None
        self.trades_today: int = 0          # number of trades taken today
        self.daily_pnl: float = 0.0         # cumulative realized P&L for today
        self.trading_halted: bool = False   # set True when daily loss limit hit

        # ── Active trade tracking ──
        self.in_trade: bool = False
        self.entry_price: Optional[float] = None
        self.trade_side: Optional[str] = None   # "Long" or "Short"

        # ── Timezone for session checks ──
        self._tz = ZoneInfo(TIMEZONE)

    # ─────────────────────────────────────────────────────────────────────────
    # Day-reset logic
    # ─────────────────────────────────────────────────────────────────────────

    def _check_new_day(self):
        """Reset daily counters if the calendar date has rolled over."""
        today = datetime.datetime.now(tz=self._tz).date()
        if self._today != today:
            self._today = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.trading_halted = False
            print(f"[RISK] New trading day: {today}  —  daily counters reset.")

    # ─────────────────────────────────────────────────────────────────────────
    # Individual rule checks
    # ─────────────────────────────────────────────────────────────────────────

    def is_within_session(self) -> bool:
        """
        Return True if current ET time falls inside the allowed window.
        Window: TRADING_START to TRADING_END (both inclusive of their minute).
        """
        now = datetime.datetime.now(tz=self._tz)
        start = now.replace(hour=TRADING_START[0], minute=TRADING_START[1],
                             second=0, microsecond=0)
        end   = now.replace(hour=TRADING_END[0],   minute=TRADING_END[1],
                             second=0, microsecond=0)
        return start <= now <= end

    def is_news_blackout(self) -> bool:
        """
        Return True if today is a high-impact news day (NFP / FOMC / CPI).
        Dates are listed in config.NEWS_BLACKOUT_DATES as 'YYYY-MM-DD' strings.
        """
        today_str = datetime.datetime.now(tz=self._tz).date().isoformat()
        return today_str in NEWS_BLACKOUT_DATES

    def under_trade_limit(self) -> bool:
        """Return True if we have not yet hit the max-trades-per-day cap."""
        return self.trades_today < MAX_TRADES_PER_DAY

    def under_loss_limit(self) -> bool:
        """Return True if daily P&L is still above the daily loss floor."""
        return self.daily_pnl > DAILY_LOSS_LIMIT

    def not_in_trade(self) -> bool:
        """Return True when no position is currently open."""
        return not self.in_trade

    # ─────────────────────────────────────────────────────────────────────────
    # Main gate — call this before every signal evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def can_trade(self) -> bool:
        """
        Master permission check.  Returns True only when ALL conditions pass:
            1. Within the allowed session window
            2. Not a news-blackout day
            3. Daily trade count < MAX_TRADES_PER_DAY
            4. Daily P&L > DAILY_LOSS_LIMIT
            5. Not already in an open position
            6. Trading not manually halted

        Prints the reason if blocked.
        """
        self._check_new_day()

        if self.trading_halted:
            print("[RISK] Trading halted — daily loss limit was hit.")
            return False

        if not self.is_within_session():
            # Silently skip — this fires every polling interval outside hours
            return False

        if self.is_news_blackout():
            print("[RISK] BLOCKED — News blackout date.  No trading today.")
            return False

        if not self.under_trade_limit():
            print(f"[RISK] BLOCKED — Already took {self.trades_today} trade(s) today "
                  f"(limit: {MAX_TRADES_PER_DAY}).")
            return False

        if not self.under_loss_limit():
            print(f"[RISK] BLOCKED — Daily P&L ${self.daily_pnl:.2f} "
                  f"has hit loss limit ${DAILY_LOSS_LIMIT:.2f}.")
            self.trading_halted = True
            return False

        if not self.not_in_trade():
            # Already in a trade — normal state, don't spam the console
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Trade lifecycle recording
    # ─────────────────────────────────────────────────────────────────────────

    def record_trade_open(self, price: float, side: str):
        """
        Call this immediately after an entry order is confirmed.

        Args:
            price: fill price of the entry
            side:  "Long" or "Short"
        """
        self.in_trade    = True
        self.entry_price = price
        self.trade_side  = side
        self.trades_today += 1
        print(f"[RISK] Trade opened: {side} @ {price:.2f}  "
              f"(trade #{self.trades_today} today)")

    def record_trade_close(self, exit_price: float):
        """
        Call this when the position is closed (SL hit, TP hit, or manual exit).
        Calculates realized P&L in dollars and updates daily_pnl.

        Args:
            exit_price: fill price of the exit
        """
        if not self.in_trade or self.entry_price is None:
            return

        # Calculate P&L in index points, then convert to dollars
        if self.trade_side == "Long":
            points_pnl = exit_price - self.entry_price
        else:  # Short
            points_pnl = self.entry_price - exit_price

        # Ticks = points / tick_size; dollars = ticks * tick_value * qty
        ticks_pnl    = points_pnl / TICK_SIZE
        dollar_pnl   = ticks_pnl * TICK_VALUE_DOLLARS * ORDER_QTY

        self.daily_pnl += dollar_pnl
        self.in_trade   = False

        outcome = "WIN" if dollar_pnl >= 0 else "LOSS"
        print(f"[RISK] Trade closed ({outcome}): {self.trade_side} "
              f"entry={self.entry_price:.2f} exit={exit_price:.2f} "
              f"P&L=${dollar_pnl:+.2f}  |  Daily P&L: ${self.daily_pnl:+.2f}")

        # Check if the daily loss limit is now breached after this trade
        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            self.trading_halted = True
            print(f"[RISK] *** DAILY LOSS LIMIT HIT (${self.daily_pnl:.2f}) — "
                  "Trading halted for the remainder of the day. ***")

        self.entry_price = None
        self.trade_side  = None

        return dollar_pnl

    # ─────────────────────────────────────────────────────────────────────────
    # Computed SL / TP price levels
    # ─────────────────────────────────────────────────────────────────────────

    def get_stop_price(self, entry_price: float, side: str) -> float:
        """
        Compute the absolute stop-loss price given the entry and direction.

        MNQ: STOP_LOSS_TICKS × TICK_SIZE = number of index points to subtract/add
        """
        offset = STOP_LOSS_TICKS * TICK_SIZE   # price points
        if side == "Long":
            return round(entry_price - offset, 2)
        else:
            return round(entry_price + offset, 2)

    def get_target_price(self, entry_price: float, side: str) -> float:
        """
        Compute the absolute take-profit price given the entry and direction.
        """
        offset = TAKE_PROFIT_TICKS * TICK_SIZE   # price points
        if side == "Long":
            return round(entry_price + offset, 2)
        else:
            return round(entry_price - offset, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Status summary
    # ─────────────────────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        """Return a human-readable summary of the current risk state."""
        self._check_new_day()
        return (
            f"Date: {self._today} | "
            f"Trades today: {self.trades_today}/{MAX_TRADES_PER_DAY} | "
            f"Daily P&L: ${self.daily_pnl:+.2f} (limit ${DAILY_LOSS_LIMIT}) | "
            f"In trade: {self.in_trade} | "
            f"Halted: {self.trading_halted}"
        )
