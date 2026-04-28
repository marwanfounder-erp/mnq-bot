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
import json as _json
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo          # stdlib in Python 3.9+; no extra install
from typing import Optional

from config import (
    DAILY_LOSS_LIMIT,
    MAX_TRADES_PER_DAY,
    NEWS_BLACKOUT_DATES,
    TRADING_START,
    TRADING_END,
    TRADING_CUTOFF,
    TIMEZONE,
    STOP_LOSS_DOLLARS,
    TAKE_PROFIT_DOLLARS,
    TICK_SIZE,
    TICK_VALUE_DOLLARS,
    STOP_LOSS_TICKS,
    TAKE_PROFIT_TICKS,
    ORDER_QTY,
    FINNHUB_API_KEY,
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

        # ── News calendar cache (refreshed once per trading day) ──
        self._news_cache_date: Optional[datetime.date] = None
        self._news_events: list = []               # list of event dicts for dashboard
        self._day_blocked: bool = False            # True = FOMC / full-day block
        self._session_delayed: bool = False        # True = start pushed to 10:00 ET
        self._effective_start: tuple = TRADING_START  # may be overridden by news

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
        Uses _effective_start (may be delayed by news calendar) instead of
        the raw TRADING_START constant.
        """
        now   = datetime.datetime.now(tz=self._tz)
        start = now.replace(hour=self._effective_start[0],
                            minute=self._effective_start[1],
                            second=0, microsecond=0)
        end   = now.replace(hour=TRADING_END[0], minute=TRADING_END[1],
                            second=0, microsecond=0)
        return start <= now <= end

    def is_news_blackout(self) -> bool:
        """
        Return True if today is a high-impact news day.
        Checks both the dynamic FOMC flag set by check_news_calendar()
        and the static NEWS_BLACKOUT_DATES list in config.
        """
        if self._day_blocked:
            return True
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

    def is_past_entry_cutoff(self) -> bool:
        """
        Return True if the current ET time is past TRADING_CUTOFF (10:45 AM).
        Blocks new entries too close to session end, ensuring at least
        45 minutes remain for take-profit to be reached before 11:30 AM.
        Only affects new entries — existing open trades continue to be managed.
        """
        now    = datetime.datetime.now(tz=self._tz)
        cutoff = now.replace(hour=TRADING_CUTOFF[0], minute=TRADING_CUTOFF[1],
                             second=0, microsecond=0)
        return now > cutoff

    # ─────────────────────────────────────────────────────────────────────────
    # News calendar (Finnhub) — fetched once per day, cached in memory
    # ─────────────────────────────────────────────────────────────────────────

    # Keywords in event names that indicate an FOMC / Fed rate decision
    _FOMC_KEYWORDS = frozenset([
        "fomc", "fed funds", "rate decision", "federal reserve",
        "interest rate decision", "monetary policy",
    ])

    def check_news_calendar(self) -> None:
        """
        Fetch today's high-impact US economic events from Finnhub once per day.

        Side-effects:
          • _news_events  — list of dicts consumed by the dashboard
          • _day_blocked  — True if an FOMC event is detected (entire day halted)
          • _session_delayed — True if a pre-10AM event delays session start
          • _effective_start — overridden to (10, 0) when session is delayed
        Falls back silently to the static NEWS_BLACKOUT_DATES if the API call fails.
        """
        today = datetime.datetime.now(tz=self._tz).date()
        if self._news_cache_date == today:
            return   # already fetched today — nothing to do

        # Reset for the new day before we know the result
        self._news_cache_date  = today
        self._news_events      = []
        self._day_blocked      = False
        self._session_delayed  = False
        self._effective_start  = TRADING_START

        try:
            raw_events = self._fetch_finnhub_events(today)
            self._process_news_events(raw_events)
        except Exception as exc:
            print(f"[NEWS] API unavailable ({exc}) — using static blackout dates only.")

    def _fetch_finnhub_events(self, date: datetime.date) -> list:
        """
        Call the Finnhub economic calendar endpoint and return the raw event list.
        Raises on network / key errors so check_news_calendar() can log and fall back.
        """
        if not FINNHUB_API_KEY or FINNHUB_API_KEY == "your_finnhub_key_here":
            raise ValueError("FINNHUB_API_KEY not configured")

        date_str = date.isoformat()
        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={date_str}&to={date_str}&token={FINNHUB_API_KEY}"
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = _json.loads(resp.read().decode())

        return payload.get("economicCalendar", [])

    def _process_news_events(self, raw_events: list) -> None:
        """
        Filter for high-impact US events, classify each one, and populate
        _news_events / _day_blocked / _session_delayed / _effective_start.
        """
        _UTC = ZoneInfo("UTC")
        high_us = [
            e for e in raw_events
            if (e.get("country") or "").upper() == "US"
            and (e.get("impact") or "").lower() == "high"
        ]

        for ev in high_us:
            event_name = ev.get("event") or "Unknown"
            time_raw   = ev.get("time") or ""
            impact_str = (ev.get("impact") or "high").capitalize()

            # Parse UTC timestamp → ET
            et_hour, et_minute, time_et_str = self._parse_event_time(time_raw, _UTC)

            # Classify the event
            is_fomc = any(kw in event_name.lower() for kw in self._FOMC_KEYWORDS)

            if is_fomc:
                status = "BLOCKED"
                self._day_blocked = True
            elif et_hour < 10:
                # Event before 10:00 AM ET — delay session start to 10:00
                status = "DELAYED"
                self._session_delayed = True
                self._effective_start = (10, 0)
            elif (et_hour > TRADING_END[0]
                  or (et_hour == TRADING_END[0] and et_minute > TRADING_END[1])):
                # Event after session end — no impact on trading
                status = "CLEAR"
            else:
                # Event falls inside the 10:00–11:30 window
                status = "CLEAR"

            self._news_events.append({
                "time_et": time_et_str,
                "event":   event_name,
                "impact":  impact_str,
                "status":  status,
            })

        # ── Logging ──────────────────────────────────────────────────────────
        if not high_us:
            print("[NEWS] No high-impact US events today — trading normally.")
            return

        event_summary = ", ".join(
            f"{e['event']} @ {e['time_et']} ET" for e in self._news_events
        )
        print(f"[NEWS] Today's high-impact US events: {event_summary}")

        if self._day_blocked:
            print("[NEWS] FOMC day detected — entire session BLOCKED.")
        elif self._session_delayed:
            start = self._effective_start
            print(f"[NEWS] Pre-10AM event found — delaying session start to "
                  f"{start[0]:02d}:{start[1]:02d} ET.")
        else:
            print("[NEWS] Events outside session window — no trading impact.")

    @staticmethod
    def _parse_event_time(
        time_raw: str, utc_zone: ZoneInfo
    ) -> tuple[int, int, str]:
        """
        Parse a Finnhub time string (UTC) and return (hour_et, minute_et, 'HH:MM').
        Returns (0, 0, '??:??') on parse failure.
        """
        _tz_et = ZoneInfo(TIMEZONE)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                utc_dt = datetime.datetime.strptime(time_raw, fmt).replace(tzinfo=utc_zone)
                et_dt  = utc_dt.astimezone(_tz_et)
                return et_dt.hour, et_dt.minute, et_dt.strftime("%H:%M")
            except ValueError:
                continue
        return 0, 0, "??:??"

    # ── Read-only properties for main.py / dashboard access ─────────────────

    @property
    def news_events(self) -> list:
        """Cached list of today's high-impact event dicts (for dashboard)."""
        return list(self._news_events)

    @property
    def day_blocked(self) -> bool:
        """True if the full day is blocked due to FOMC."""
        return self._day_blocked

    @property
    def session_delayed(self) -> bool:
        """True if session start has been pushed to 10:00 ET."""
        return self._session_delayed

    @property
    def effective_start(self) -> tuple:
        """The actual session start tuple — may differ from TRADING_START."""
        return self._effective_start

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
        self.check_news_calendar()   # no-op after first call each day

        if self.trading_halted:
            print("[RISK] Trading halted — daily loss limit was hit.")
            return False

        if not self.is_within_session():
            # Silently skip — this fires every polling interval outside hours
            return False

        if self.is_past_entry_cutoff():
            print(f"[RISK] BLOCKED — Too late to enter new trade "
                  f"(after {TRADING_CUTOFF[0]:02d}:{TRADING_CUTOFF[1]:02d} ET)")
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
