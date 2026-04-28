# =============================================================================
# strategy.py — Entry and exit signal logic
#
# Strategy rules (all must be true simultaneously):
#
#   LONG entry:
#       • Close price > EMA 21  (uptrend filter)
#       • EMA 9 trending upward  (price accelerating)
#       • BULL regime (close > EMA21 for ≥3 consecutive bars) [Regime Filter]
#       • RSI between 35 and 50  (oversold-recovery sweet spot) [RSI Range Filter]
#       • Price above EMA21 for ≥3 consecutive bars  [Trend Strength Filter]
#       • RSI rising for last 2 bars                 [RSI Direction Filter]
#
#   SHORT entry:
#       • Close price < EMA 21  (downtrend filter)
#       • EMA 9 trending downward
#       • BEAR regime (close < EMA21 for ≥3 consecutive bars) [Regime Filter]
#       • RSI between 50 and 65  (overbought-pullback sweet spot) [RSI Range Filter]
#       • No bar in last 5 had RSI > 65              [RSI Overbought Memory Filter]
#       • Price below EMA21 for ≥3 consecutive bars  [Trend Strength Filter]
#       • RSI falling for last 2 bars                [RSI Direction Filter]
#
#   Regime rules:
#       • BULL regime  → LONG only  (SHORT signals blocked)
#       • BEAR regime  → SHORT only (LONG signals blocked)
#       • UNCLEAR      → both LONG and SHORT blocked
#
#   Exit signals (position management):
#       • Price hits stop-loss  (risk_manager provides the level)
#       • Price hits take-profit (risk_manager provides the level)
#       • Session end (11:00 AM ET) — force-close any open position
#
# =============================================================================

from __future__ import annotations

import csv as _csv
import pandas as pd
from typing import Optional, Literal

from config import (
    RSI_LONG_MIN,
    RSI_LONG_MAX,
    RSI_SHORT_MIN,
    RSI_SHORT_MAX,
    EMA_FAST,
    EMA_SLOW,
    LOG_FILE,
)
from indicators import calculate_all_indicators, get_latest_values


# ─── Signal type alias ────────────────────────────────────────────────────────
Signal = Literal["LONG", "SHORT", "HOLD"]

# ─── Regime type alias ────────────────────────────────────────────────────────
Regime = Literal["BULL", "BEAR", "UNCLEAR"]


class Strategy:
    """
    Stateless signal generator.
    Call evaluate() with the latest OHLCV DataFrame to get a signal.
    """

    def __init__(self):
        self.last_signal: Signal = "HOLD"
        self.last_values: dict   = {}
        self.last_regime: Regime = "UNCLEAR"

    # ─────────────────────────────────────────────────────────────────────────
    # Core signal evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> Signal:
        """
        Compute indicators on the latest bars and return a trading signal.

        Args:
            df: DataFrame with columns [time, open, high, low, close, volume]
                — must contain at least EMA_SLOW (21) bars.

        Returns:
            "LONG"  → enter a long position now
            "SHORT" → enter a short position now
            "HOLD"  → no trade; wait for next bar
        """
        if df.empty or len(df) < EMA_SLOW:
            print(f"[STRATEGY] Not enough bars ({len(df)} of {EMA_SLOW} needed) — HOLD")
            return "HOLD"

        # ── Step 1: Calculate all indicators ──────────────────────────────────
        try:
            df_ind = calculate_all_indicators(df)
        except ValueError as e:
            print(f"[STRATEGY] Indicator error: {e} — HOLD")
            return "HOLD"

        # ── Step 2: Extract latest values ─────────────────────────────────────
        vals = get_latest_values(df_ind)
        self.last_values = vals

        close    = vals["close"]
        ema_fast = vals["ema_fast"]
        ema_slow = vals["ema_slow"]
        rsi      = vals["rsi"]

        # Also look at the previous bar's EMA-fast to judge slope
        prev_ema_fast = df_ind["ema_fast"].iloc[-2] if len(df_ind) > 1 else ema_fast

        ema_fast_rising  = ema_fast > prev_ema_fast
        ema_fast_falling = ema_fast < prev_ema_fast

        # ── Step 3: Apply base rules ──────────────────────────────────────────
        #
        # LONG signal:  price > EMA21  AND  EMA9 rising
        # (RSI range is now enforced as a dedicated filter in _apply_filters)
        long_base = (
            close > ema_slow    # uptrend filter
            and ema_fast_rising # short-term momentum up
        )

        # SHORT signal: price < EMA21  AND  EMA9 falling
        short_base = (
            close < ema_slow     # downtrend filter
            and ema_fast_falling # short-term momentum down
        )

        # ── Step 4: Apply additional filters ─────────────────────────────────
        candidate = "LONG" if long_base else ("SHORT" if short_base else "HOLD")

        # Detect regime once — used both for filtering and state export
        regime = self._detect_regime(df_ind)
        self.last_regime = regime

        if candidate != "HOLD":
            signal = self._apply_filters(df_ind, candidate, regime)
        else:
            signal = "HOLD"
            # Still print filter summary so every bar is auditable
            self._print_filter_summary(df_ind, candidate, regime)

        self.last_signal = signal

        # ── Step 5: Debug print ───────────────────────────────────────────────
        self._print_evaluation(close, ema_fast, ema_slow, rsi,
                               ema_fast_rising, signal)
        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Exit signal check (called each bar while in a position)
    # ─────────────────────────────────────────────────────────────────────────

    def check_exit(
        self,
        current_price: float,
        stop_price: float,
        target_price: float,
        side: str,
    ) -> Optional[str]:
        """
        Check whether an open position should be closed.

        Args:
            current_price: latest market price
            stop_price:    stop-loss level (from RiskManager)
            target_price:  take-profit level (from RiskManager)
            side:          "Long" or "Short"

        Returns:
            "STOP"   → close position at stop-loss
            "TARGET" → close position at take-profit
            None     → hold the position
        """
        if side == "Long":
            if current_price <= stop_price:
                print(f"[STRATEGY] STOP hit — Long exit @ {current_price:.2f} "
                      f"(stop was {stop_price:.2f})")
                return "STOP"
            if current_price >= target_price:
                print(f"[STRATEGY] TARGET hit — Long exit @ {current_price:.2f} "
                      f"(target was {target_price:.2f})")
                return "TARGET"

        elif side == "Short":
            if current_price >= stop_price:
                print(f"[STRATEGY] STOP hit — Short exit @ {current_price:.2f} "
                      f"(stop was {stop_price:.2f})")
                return "STOP"
            if current_price <= target_price:
                print(f"[STRATEGY] TARGET hit — Short exit @ {current_price:.2f} "
                      f"(target was {target_price:.2f})")
                return "TARGET"

        return None  # continue holding

    # ─────────────────────────────────────────────────────────────────────────
    # Market regime detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_regime(self, df_ind: pd.DataFrame) -> Regime:
        """
        Classify market regime using EMA21 vs close price for last 3 bars.

        BULL   — close > EMA21 for all 3 consecutive bars
        BEAR   — close < EMA21 for all 3 consecutive bars
        UNCLEAR — price crossing EMA21; not consistently on one side
        """
        if len(df_ind) < 3:
            return "UNCLEAR"

        last3_close = df_ind["close"].iloc[-3:].values
        last3_ema   = df_ind["ema_slow"].iloc[-3:].values

        all_above = all(c > e for c, e in zip(last3_close, last3_ema))
        all_below = all(c < e for c, e in zip(last3_close, last3_ema))

        if all_above:
            return "BULL"
        elif all_below:
            return "BEAR"
        return "UNCLEAR"

    # ─────────────────────────────────────────────────────────────────────────
    # Additional signal filters
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_filters(
        self, df_ind: pd.DataFrame, candidate: Signal, regime: Regime
    ) -> Signal:
        """
        Run all filters against a LONG or SHORT candidate.
        Returns the original candidate if all pass, or "HOLD" if any fail.

        Filter order (regime is checked FIRST):
          0. Market Regime         — BULL→LONG only, BEAR→SHORT only, UNCLEAR→block all
          1. RSI Range             — LONG: 35<RSI<50, SHORT: 50<RSI<65
          2. RSI Overbought Memory — SHORT only: no bar in last 5 had RSI > 65
          3. Trend Strength        — price on same side of EMA21 for ≥3 bars
          4. RSI Direction         — RSI rising (LONG) or falling (SHORT) last 2 bars
        """
        rsi_series   = df_ind["rsi"]
        close_series = df_ind["close"]
        ema_slow_ser = df_ind["ema_slow"]
        rsi          = rsi_series.iloc[-1]

        # ── Filter 0: Market Regime (checked FIRST) ───────────────────────────
        regime_pass = True
        if regime == "UNCLEAR":
            regime_pass = False
            print(
                f"[STRATEGY] UNCLEAR REGIME — "
                f"{candidate} signal ignored, waiting for trend ✅"
            )
        elif regime == "BULL" and candidate == "SHORT":
            regime_pass = False
            print("[STRATEGY] BULL REGIME — SHORT signal ignored ✅")
        elif regime == "BEAR" and candidate == "LONG":
            regime_pass = False
            print("[STRATEGY] BEAR REGIME — LONG signal ignored ✅")

        # Early exit: no point running other filters if regime blocks the trade
        if not regime_pass:
            print(
                f"[STRATEGY] Filter check:\n"
                f"  Regime         → {regime} ❌\n"
                f"  RSI range      → N/A\n"
                f"  RSI memory     → N/A\n"
                f"  Trend strength → N/A\n"
                f"  RSI direction  → N/A\n"
                f"  Final signal   → HOLD"
            )
            return "HOLD"

        # ── Filter 1: RSI Range ───────────────────────────────────────────────
        rsi_range_pass = True
        if candidate == "LONG":
            if not (RSI_LONG_MIN < rsi < RSI_LONG_MAX):
                rsi_range_pass = False
                print(
                    f"[STRATEGY] BLOCKED — RSI {rsi:.1f} outside LONG range "
                    f"({RSI_LONG_MIN}-{RSI_LONG_MAX}) ❌"
                )
        else:  # SHORT
            if not (RSI_SHORT_MIN < rsi < RSI_SHORT_MAX):
                rsi_range_pass = False
                print(
                    f"[STRATEGY] BLOCKED — RSI {rsi:.1f} outside SHORT range "
                    f"({RSI_SHORT_MIN}-{RSI_SHORT_MAX}) ❌"
                )

        # ── Filter 2: RSI Overbought Memory (SHORT only) ──────────────────────
        rsi_memory_pass = True
        if candidate == "SHORT":
            last5_rsi = rsi_series.iloc[-5:] if len(rsi_series) >= 5 else rsi_series
            max_rsi_recent = last5_rsi.max()
            if max_rsi_recent > 65:
                rsi_memory_pass = False
                print(
                    f"[STRATEGY] BLOCKED SHORT — RSI was overbought recently "
                    f"(max RSI last 5 bars: {max_rsi_recent:.1f})"
                )

        # ── Filter 3: Trend Strength (≥3 consecutive bars on correct side) ────
        trend_strength_pass = True
        if len(df_ind) >= 3:
            last3_close    = close_series.iloc[-3:].values
            last3_ema_slow = ema_slow_ser.iloc[-3:].values
            if candidate == "LONG":
                above = all(c > e for c, e in zip(last3_close, last3_ema_slow))
                if not above:
                    trend_strength_pass = False
                    print("[STRATEGY] BLOCKED LONG — Price not above EMA21 for 3 bars")
            else:  # SHORT
                below = all(c < e for c, e in zip(last3_close, last3_ema_slow))
                if not below:
                    trend_strength_pass = False
                    print("[STRATEGY] BLOCKED SHORT — Price not below EMA21 for 3 bars")
        else:
            trend_strength_pass = False
            print(f"[STRATEGY] BLOCKED {candidate} — Not enough bars for trend strength check")

        # ── Filter 4: RSI Direction (consistent over last 2 bars) ─────────────
        rsi_direction_pass = True
        if len(rsi_series) >= 2:
            rsi_curr = rsi_series.iloc[-1]
            rsi_prev = rsi_series.iloc[-2]
            if candidate == "LONG":
                if not (rsi_curr > rsi_prev):
                    rsi_direction_pass = False
                    print("[STRATEGY] BLOCKED LONG — RSI not rising consistently")
            else:  # SHORT
                if not (rsi_curr < rsi_prev):
                    rsi_direction_pass = False
                    print("[STRATEGY] BLOCKED SHORT — RSI not falling consistently")
        else:
            rsi_direction_pass = False
            print(f"[STRATEGY] BLOCKED {candidate} — Not enough bars for RSI direction check")

        # ── Filter summary ────────────────────────────────────────────────────
        all_pass = (rsi_range_pass and rsi_memory_pass
                    and trend_strength_pass and rsi_direction_pass)
        final    = candidate if all_pass else "HOLD"

        regime_mark    = "✅"
        rsi_range_mark = "✅" if rsi_range_pass       else "❌"
        mem_mark       = "✅" if rsi_memory_pass       else "❌"
        trend_mark     = "✅" if trend_strength_pass   else "❌"
        dir_mark       = "✅" if rsi_direction_pass    else "❌"

        # RSI memory filter only applies to SHORT; mark N/A for LONG
        mem_display = mem_mark if candidate == "SHORT" else "N/A"

        # RSI range bounds depend on direction
        rsi_bounds = (f"{RSI_LONG_MIN}-{RSI_LONG_MAX}"
                      if candidate == "LONG"
                      else f"{RSI_SHORT_MIN}-{RSI_SHORT_MAX}")

        print(
            f"[STRATEGY] Filter check:\n"
            f"  Regime         → {regime} {regime_mark}\n"
            f"  RSI range      → {rsi_range_mark} (value: {rsi:.1f}, range: {rsi_bounds})\n"
            f"  RSI memory     → {mem_display}\n"
            f"  Trend strength → {trend_mark}\n"
            f"  RSI direction  → {dir_mark}\n"
            f"  Final signal   → {final}"
        )
        return final

    def _print_filter_summary(
        self, df_ind: pd.DataFrame, candidate: Signal, regime: Regime
    ) -> None:
        """
        Emit the filter-summary line even when the base condition is HOLD
        so every bar produces a consistent audit trail.
        """
        print(
            f"[STRATEGY] Filter check:\n"
            f"  Regime         → {regime} (base condition not met)\n"
            f"  RSI range      → N/A\n"
            f"  RSI memory     → N/A\n"
            f"  Trend strength → N/A\n"
            f"  RSI direction  → N/A\n"
            f"  Final signal   → HOLD"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Regime backtest (printed once at startup)
    # ─────────────────────────────────────────────────────────────────────────

    def run_regime_backtest(self, n: int = 8) -> None:
        """
        Read the last N closed trades from trades.csv and show what the
        regime filter would have done to each one.

        Uses entry_price vs ema_slow at entry as a single-bar regime proxy
        (full 3-bar history is not stored in the CSV, so this is an
        approximation — directionally accurate for audit purposes).
        """
        try:
            with open(LOG_FILE, newline="") as f:
                rows = list(_csv.DictReader(f))
        except (FileNotFoundError, OSError):
            return   # no trade history yet — skip silently

        closed = [r for r in rows if r.get("exit_price")]
        if not closed:
            return

        recent = closed[-n:]
        total_saved   = 0.0
        total_missed  = 0.0
        blocked_losses = 0

        print(
            f"[REGIME BACKTEST] ─── Last {len(recent)} closed trades vs regime filter ───"
        )
        for row in recent:
            try:
                tid   = row.get("trade_id", "?")
                side  = row.get("side", "")
                entry = float(row.get("entry_price") or 0)
                ema_s = float(row.get("ema_slow")    or 0)
                pnl   = float(row.get("pnl_dollars") or 0)
            except (ValueError, TypeError):
                continue

            if ema_s == 0:
                continue

            # Single-bar regime proxy at entry
            regime = "BULL" if entry > ema_s else "BEAR"

            blocked = (
                (side == "Long"  and regime == "BEAR") or
                (side == "Short" and regime == "BULL")
            )

            if blocked:
                if pnl < 0:
                    total_saved  += abs(pnl)
                    blocked_losses += 1
                    tag = f"BLOCKED ✅  saved ${abs(pnl):.2f}"
                else:
                    total_missed += pnl
                    tag = f"BLOCKED ⚠️  (missed +${pnl:.2f})"
            else:
                result = "WIN " if pnl > 0 else "LOSS"
                tag = f"ALLOWED ✅  {result} ${pnl:+.2f}"

            print(
                f"[REGIME BACKTEST]  Trade {tid:>3} {side.upper():<5} "
                f"→ {regime} regime → {tag}"
            )

        print(
            f"[REGIME BACKTEST]  Losses avoided   → ${total_saved:.2f}  "
            f"({blocked_losses} trade{'s' if blocked_losses != 1 else ''})"
        )
        if total_missed > 0:
            print(
                f"[REGIME BACKTEST]  Wins also blocked → ${total_missed:.2f}  "
                f"(regime is a two-edged filter)"
            )
        print("[REGIME BACKTEST] ──────────────────────────────────────────────")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _print_evaluation(
        self,
        close: float,
        ema_fast: float,
        ema_slow: float,
        rsi: float,
        ema_rising: bool,
        signal: Signal,
    ):
        """Print a concise one-line bar evaluation summary."""
        arrow   = "↑" if ema_rising else "↓"
        regime  = self.last_regime

        print(
            f"[STRATEGY] close={close:.2f}  "
            f"EMA9={ema_fast:.2f}{arrow}  EMA21={ema_slow:.2f}  "
            f"RSI={rsi:.1f}  regime={regime}  →  {signal}"
        )

    def summary(self) -> dict:
        """Return the last-evaluated indicator snapshot and signal."""
        return {**self.last_values, "signal": self.last_signal, "regime": self.last_regime}
