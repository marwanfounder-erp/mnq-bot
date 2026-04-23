# =============================================================================
# strategy.py — Entry and exit signal logic
#
# Strategy rules (all must be true simultaneously):
#
#   LONG entry:
#       • Close price > EMA 21  (uptrend filter)
#       • RSI(14) < 45           (momentum dip — not overbought)
#       • EMA 9 trending upward  (price accelerating)
#       • Price above EMA21 for ≥3 consecutive bars  [Trend Strength Filter]
#       • RSI rising for last 2 bars                 [RSI Direction Filter]
#
#   SHORT entry:
#       • Close price < EMA 21  (downtrend filter)
#       • RSI(14) > 55           (momentum surge — not oversold)
#       • EMA 9 trending downward
#       • No bar in last 5 had RSI > 65              [RSI Overbought Memory Filter]
#       • Price below EMA21 for ≥3 consecutive bars  [Trend Strength Filter]
#       • RSI falling for last 2 bars                [RSI Direction Filter]
#
#   Exit signals (position management):
#       • Price hits stop-loss  (risk_manager provides the level)
#       • Price hits take-profit (risk_manager provides the level)
#       • Session end (11:00 AM ET) — force-close any open position
#
# =============================================================================

from __future__ import annotations

import pandas as pd
from typing import Optional, Literal

from config import (
    RSI_BUY_THRESHOLD,
    RSI_SELL_THRESHOLD,
    EMA_FAST,
    EMA_SLOW,
)
from indicators import calculate_all_indicators, get_latest_values


# ─── Signal type alias ────────────────────────────────────────────────────────
Signal = Literal["LONG", "SHORT", "HOLD"]


class Strategy:
    """
    Stateless signal generator.
    Call evaluate() with the latest OHLCV DataFrame to get a signal.
    """

    def __init__(self):
        self.last_signal: Signal = "HOLD"
        self.last_values: dict   = {}

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
        # LONG signal:  price > EMA21  AND  RSI < 45  AND  EMA9 rising
        long_base = (
            close > ema_slow              # uptrend filter
            and rsi < RSI_BUY_THRESHOLD   # pullback / dip
            and ema_fast_rising           # short-term momentum up
        )

        # SHORT signal: price < EMA21  AND  RSI > 55  AND  EMA9 falling
        short_base = (
            close < ema_slow              # downtrend filter
            and rsi > RSI_SELL_THRESHOLD  # elevated / overbought
            and ema_fast_falling          # short-term momentum down
        )

        # ── Step 4: Apply additional filters ─────────────────────────────────
        candidate = "LONG" if long_base else ("SHORT" if short_base else "HOLD")

        if candidate != "HOLD":
            signal = self._apply_filters(df_ind, candidate)
        else:
            signal = "HOLD"
            # Still print filter summary so every bar is auditable
            self._print_filter_summary(df_ind, candidate)

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
    # Additional signal filters
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_filters(self, df_ind: pd.DataFrame, candidate: Signal) -> Signal:
        """
        Run the three additional filters against a LONG or SHORT candidate.
        Returns the original candidate if all pass, or "HOLD" if any fail.

        Filters applied:
          1. RSI Overbought Memory  — SHORT only: no bar in last 5 had RSI > 65
          2. Trend Strength         — price on same side of EMA21 for ≥3 bars
          3. RSI Direction          — RSI rising (LONG) or falling (SHORT) last 2 bars
        """
        rsi_series   = df_ind["rsi"]
        close_series = df_ind["close"]
        ema_slow_ser = df_ind["ema_slow"]

        # ── Filter 1: RSI Overbought Memory (SHORT only) ──────────────────────
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

        # ── Filter 2: Trend Strength (≥3 consecutive bars on correct side) ────
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

        # ── Filter 3: RSI Direction (consistent over last 2 bars) ─────────────
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
        all_pass = rsi_memory_pass and trend_strength_pass and rsi_direction_pass
        final    = candidate if all_pass else "HOLD"

        mem_mark   = "✅" if rsi_memory_pass   else "❌"
        trend_mark = "✅" if trend_strength_pass else "❌"
        dir_mark   = "✅" if rsi_direction_pass  else "❌"

        # RSI memory filter only applies to SHORT; mark N/A for LONG
        mem_display = mem_mark if candidate == "SHORT" else "N/A"

        print(
            f"[STRATEGY] Filter check:  "
            f"RSI memory {mem_display}  "
            f"Trend strength {trend_mark}  "
            f"RSI direction {dir_mark}  "
            f"→  Final signal: {final}"
        )
        return final

    def _print_filter_summary(self, df_ind: pd.DataFrame, candidate: Signal) -> None:
        """
        Emit the filter-summary line even when the base condition is HOLD
        so every bar produces a consistent audit trail.
        """
        print(
            "[STRATEGY] Filter check:  "
            "RSI memory N/A  Trend strength N/A  RSI direction N/A  "
            "→  Final signal: HOLD  (base condition not met)"
        )

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
        trend   = "BULL" if close > ema_slow else "BEAR"
        rsi_lbl = f"RSI={rsi:.1f}"
        arrow   = "↑" if ema_rising else "↓"

        print(
            f"[STRATEGY] close={close:.2f}  "
            f"EMA9={ema_fast:.2f}{arrow}  EMA21={ema_slow:.2f}  "
            f"{rsi_lbl}  trend={trend}  →  {signal}"
        )

    def summary(self) -> dict:
        """Return the last-evaluated indicator snapshot and signal."""
        return {**self.last_values, "signal": self.last_signal}
