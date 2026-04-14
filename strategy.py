# =============================================================================
# strategy.py — Entry and exit signal logic
#
# Strategy rules (all must be true simultaneously):
#
#   LONG entry:
#       • Close price > EMA 21  (uptrend filter)
#       • RSI(14) < 45           (momentum dip — not overbought)
#       • EMA 9 trending upward  (price accelerating)
#
#   SHORT entry:
#       • Close price < EMA 21  (downtrend filter)
#       • RSI(14) > 55           (momentum surge — not oversold)
#       • EMA 9 trending downward
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

        # ── Step 3: Apply rules ───────────────────────────────────────────────
        #
        # LONG signal:  price > EMA21  AND  RSI < 45  AND  EMA9 rising
        long_condition = (
            close > ema_slow         # uptrend filter
            and rsi < RSI_BUY_THRESHOLD   # pullback / dip
            and ema_fast_rising      # short-term momentum up
        )

        # SHORT signal: price < EMA21  AND  RSI > 55  AND  EMA9 falling
        short_condition = (
            close < ema_slow         # downtrend filter
            and rsi > RSI_SELL_THRESHOLD  # elevated / overbought
            and ema_fast_falling     # short-term momentum down
        )

        # ── Step 4: Choose signal (priority: LONG > SHORT > HOLD) ─────────────
        if long_condition:
            signal = "LONG"
        elif short_condition:
            signal = "SHORT"
        else:
            signal = "HOLD"

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
