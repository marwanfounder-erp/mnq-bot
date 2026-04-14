#!/usr/bin/env python3
# =============================================================================
# backtester.py — Historical strategy backtesting on MNQ/NQ data
#
# Data sources (choose one):
#   A. yfinance   → downloads NQ=F or MNQ=F directly (recommended for quick tests)
#   B. CSV file   → pass a path to --csv; any OHLCV CSV with a datetime index
#
# Usage:
#   python backtester.py                          # uses config defaults
#   python backtester.py --interval 1h --period 2y
#   python backtester.py --csv /path/to/data.csv
#   python backtester.py --start 2024-01-01 --end 2024-12-31 --interval 1h
#
# yfinance interval limits (as of 2026):
#   1m  → max 7 days of history
#   2m  → max 60 days
#   5m  → max 60 days
#   15m → max 60 days
#   1h  → max 730 days   ← best balance of granularity and history
#   1d  → years of history (SL/TP too tight for daily bars, but trend visible)
#
# Strategy replayed:
#   • EMA 9 / 21 trend filter
#   • RSI 14 confirmation
#   • Session filter: 9:30–11:00 AM ET
#   • Max 1 trade per day
#   • SL = STOP_LOSS_TICKS ticks | TP = TAKE_PROFIT_TICKS ticks
#   • News blackout dates from config
# =============================================================================

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD,
    RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD,
    STOP_LOSS_TICKS, TAKE_PROFIT_TICKS,
    TICK_SIZE, TICK_VALUE_DOLLARS, ORDER_QTY,
    TRADING_START, TRADING_END, TIMEZONE,
    NEWS_BLACKOUT_DATES,
    BACKTEST_SYMBOL, BACKTEST_INTERVAL,
    BACKTEST_PERIOD, BACKTEST_START, BACKTEST_END,
    BACKTEST_RESULTS_FILE,
)
from indicators import calculate_all_indicators

_TZ = ZoneInfo(TIMEZONE)


# =============================================================================
# Data loading
# =============================================================================

def load_from_yfinance(
    symbol:   str = BACKTEST_SYMBOL,
    interval: str = BACKTEST_INTERVAL,
    period:   str = BACKTEST_PERIOD,
    start:    str = BACKTEST_START,
    end:      str = BACKTEST_END,
) -> pd.DataFrame:
    """
    Download OHLCV bars from Yahoo Finance via yfinance.

    Returns a DataFrame with columns: [time, open, high, low, close, volume]
    where `time` is timezone-aware (US/Eastern).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[BACKTEST] yfinance not installed.  Run:  pip install yfinance")
        sys.exit(1)

    print(f"[BACKTEST] Downloading {symbol} at {interval} interval…")

    kwargs: Dict = {"interval": interval, "auto_adjust": True}
    if start and end:
        kwargs["start"] = start
        kwargs["end"]   = end
    else:
        kwargs["period"] = period

    ticker = yf.Ticker(symbol)
    raw    = ticker.history(**kwargs)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r} "
                         f"with interval={interval!r}. "
                         f"Try a longer period or a different interval.")

    # Normalise column names to lowercase
    raw.columns = [c.lower() for c in raw.columns]

    # yfinance 1.2+ already returns America/New_York-aware timestamps.
    # Convert to a consistent US/Eastern representation using pytz so the
    # session time comparisons work correctly against naive time objects.
    import pytz
    _et_pytz = pytz.timezone("US/Eastern")
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(_et_pytz)

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "Datetime"
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "time"})

    print(f"[BACKTEST] Loaded {len(df):,} bars  "
          f"({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")
    return df


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected columns (case-insensitive):
        datetime / time / date, open, high, low, close, volume

    The datetime column can be in any format pandas can parse.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    print(f"[BACKTEST] Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find the datetime column
    time_col = next(
        (c for c in df.columns if c in ("datetime", "time", "date", "timestamp")),
        None,
    )
    if time_col is None:
        raise ValueError(
            "CSV must have a column named 'datetime', 'time', 'date', or 'timestamp'."
        )

    df["time"] = pd.to_datetime(df[time_col])

    # Localise to ET if the timestamps are naive
    import pytz
    _et_pytz = pytz.timezone("US/Eastern")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(_et_pytz, ambiguous="infer",
                                                nonexistent="shift_forward")
    else:
        df["time"] = df["time"].dt.tz_convert(_et_pytz)

    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = (df[["time", "open", "high", "low", "close", "volume"]]
            .sort_values("time")
            .reset_index(drop=True))

    print(f"[BACKTEST] Loaded {len(df):,} bars  "
          f"({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")
    return df


# =============================================================================
# Session & news filters (mirrors risk_manager.py logic for backtesting)
# =============================================================================

def _within_session(ts: pd.Timestamp) -> bool:
    """Return True if `ts` (ET) falls inside the configured session window."""
    t = ts.time()
    start = datetime.min.replace(
        hour=TRADING_START[0], minute=TRADING_START[1]).time()
    end   = datetime.min.replace(
        hour=TRADING_END[0],   minute=TRADING_END[1]).time()
    return start <= t <= end


def _is_news_date(ts: pd.Timestamp) -> bool:
    """Return True if the bar's date is a news blackout date."""
    return ts.date().isoformat() in NEWS_BLACKOUT_DATES


# =============================================================================
# Core backtester
# =============================================================================

class Backtester:
    """
    Event-driven bar-by-bar strategy simulator.

    After __init__, call:
        bt.load(df)                   # provide OHLCV data
        results = bt.run()            # run the simulation
        bt.report()                   # print performance metrics
        bt.save_results()             # write trades to CSV
    """

    def __init__(self):
        self.df:      Optional[pd.DataFrame] = None   # full OHLCV dataset
        self.trades:  List[Dict]             = []     # completed trades

        # Derived from config
        self._sl_pts = STOP_LOSS_TICKS  * TICK_SIZE   # points
        self._tp_pts = TAKE_PROFIT_TICKS * TICK_SIZE  # points

    def load(self, df: pd.DataFrame):
        """Attach an OHLCV DataFrame (from yfinance or CSV)."""
        self.df = df.copy()
        self.trades = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main simulation loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """
        Replay the strategy bar-by-bar over the loaded data.

        KEY DESIGN CHOICE: indicators are computed on SESSION BARS ONLY.
        NQ/MNQ futures trade 24/7 on Globex. Computing RSI on the full
        24-hour bar series keeps RSI artificially elevated during the day
        session (overnight moves absorb intraday pullbacks), making the
        RSI < 45 condition nearly impossible to trigger.
        Session-only indicator computation matches what any standard
        intraday chart on a trading platform shows.

        Simulation rules:
          - Warmup: skip the first EMA_SLOW session bars
          - Entry:  signal on bar N -> execute at bar N+1 OPEN (no look-ahead)
          - Exit:   scan full-df bars from entry for SL/TP or session end
          - One trade per calendar day maximum

        Returns a list of completed trade dicts.
        """
        if self.df is None or self.df.empty:
            raise RuntimeError("No data loaded. Call load() first.")

        full_df   = self.df
        sess_mask = full_df["time"].apply(_within_session)
        sess_df   = full_df[sess_mask].reset_index(drop=True)

        if len(sess_df) < EMA_SLOW + 2:
            print(f"[BACKTEST] Not enough session bars ({len(sess_df)}) for "
                  f"warm-up (need {EMA_SLOW + 2}).")
            return []

        print(f"[BACKTEST] Running simulation on {len(sess_df):,} session bars "
              f"(filtered from {len(full_df):,} total bars)...")

        # Map session-bar position -> original full_df index for exit scanning
        sess_full_idx = full_df[sess_mask].index.tolist()

        in_trade      = False
        entry_price   = 0.0
        stop_price    = 0.0
        target_price  = 0.0
        trade_side    = ""
        entry_bar_ts  = None
        entry_full_i  = 0
        daily_traded: Dict[date, int] = {}

        for si in range(EMA_SLOW, len(sess_df)):
            bar   = sess_df.iloc[si]
            ts    = bar["time"]
            date_ = ts.date()

            # ── If in a trade: scan full_df bars forward for SL/TP/session-end
            if in_trade:
                exited          = False
                exit_price_val  = 0.0
                exit_reason_val = ""
                exit_ts         = ts

                for fi in range(entry_full_i, len(full_df)):
                    fbar = full_df.iloc[fi]
                    fts  = fbar["time"]

                    # If bar is outside session, force-close at its open
                    if not _within_session(fts):
                        if fi > entry_full_i:
                            exit_price_val  = fbar["open"]
                            exit_reason_val = "SESSION_END"
                            exit_ts         = fts
                            exited = True
                            break
                        continue

                    hit, ep, er = self._check_exit_on_bar(
                        fbar, trade_side, stop_price, target_price
                    )
                    if hit:
                        exit_price_val  = ep
                        exit_reason_val = er
                        exit_ts         = fts
                        exited = True
                        break

                if not exited:
                    continue   # still holding; check again next session bar

                self._record_trade(
                    side         = trade_side,
                    entry_price  = entry_price,
                    exit_price   = exit_price_val,
                    exit_reason  = exit_reason_val,
                    entry_bar_ts = entry_bar_ts,
                    exit_bar_ts  = exit_ts,
                )
                in_trade = False
                continue

            # ── Pre-entry filters
            if _is_news_date(ts):
                continue
            if daily_traded.get(date_, 0) >= 1:
                continue
            if si + 1 >= len(sess_df):
                continue

            # ── Compute indicators on SESSION bars only (up to and including si)
            window = sess_df.iloc[max(0, si - 200): si + 1].copy()
            try:
                ind_df = calculate_all_indicators(window)
            except ValueError:
                continue

            last     = ind_df.iloc[-1]
            prev_ema = ind_df["ema_fast"].iloc[-2] if len(ind_df) > 1 else last["ema_fast"]

            close    = last["close"]
            ema_fast = last["ema_fast"]
            ema_slow = last["ema_slow"]
            rsi      = last["rsi"]
            ema_up   = ema_fast > prev_ema
            ema_down = ema_fast < prev_ema

            # ── Entry signal (mirrors strategy.py exactly)
            long_signal  = (close > ema_slow
                            and rsi < RSI_BUY_THRESHOLD
                            and ema_up)
            short_signal = (close < ema_slow
                            and rsi > RSI_SELL_THRESHOLD
                            and ema_down)

            if not long_signal and not short_signal:
                continue

            signal = "Long" if long_signal else "Short"

            # ── Enter at next SESSION bar OPEN (no look-ahead bias)
            next_sess_bar = sess_df.iloc[si + 1]
            next_ts       = next_sess_bar["time"]

            if not _within_session(next_ts):
                continue

            entry_price  = next_sess_bar["open"]
            stop_price   = (entry_price - self._sl_pts if signal == "Long"
                            else entry_price + self._sl_pts)
            target_price = (entry_price + self._tp_pts if signal == "Long"
                            else entry_price - self._tp_pts)

            in_trade     = True
            trade_side   = signal
            entry_bar_ts = next_ts
            entry_full_i = sess_full_idx[si + 1]
            daily_traded[next_ts.date()] = daily_traded.get(next_ts.date(), 0) + 1

        # ── Close any trade still open at end of data
        if in_trade:
            last_bar = full_df.iloc[-1]
            self._record_trade(
                side         = trade_side,
                entry_price  = entry_price,
                exit_price   = last_bar["close"],
                exit_reason  = "END_OF_DATA",
                entry_bar_ts = entry_bar_ts,
                exit_bar_ts  = last_bar["time"],
            )

        print(f"[BACKTEST] Simulation complete --- {len(self.trades)} trades found.")
        return self.trades

    # Exit logic
    # ─────────────────────────────────────────────────────────────────────────

    def _check_exit_on_bar(
        self,
        bar:          pd.Series,
        side:         str,
        stop_price:   float,
        target_price: float,
    ) -> Tuple[bool, float, str]:
        """
        Determine whether SL or TP was hit within `bar`.

        When both SL and TP are within the same bar's range (rare but possible
        on wide bars), we use the "closer to open" heuristic:
            • The level nearer to the bar's open was likely hit first.
            • This is conservative and avoids artificially inflating win rate.

        Returns: (exited, exit_price, exit_reason)
        """
        high = bar["high"]
        low  = bar["low"]
        open_ = bar["open"]

        if side == "Long":
            sl_hit = low  <= stop_price
            tp_hit = high >= target_price

            if sl_hit and tp_hit:
                # Both within bar — check which is closer to open
                if (open_ - stop_price) <= (target_price - open_):
                    return True, stop_price,   "STOP"
                else:
                    return True, target_price, "TARGET"
            elif sl_hit:
                return True, stop_price,   "STOP"
            elif tp_hit:
                return True, target_price, "TARGET"

        else:  # Short
            sl_hit = high >= stop_price
            tp_hit = low  <= target_price

            if sl_hit and tp_hit:
                if (stop_price - open_) <= (open_ - target_price):
                    return True, stop_price,   "STOP"
                else:
                    return True, target_price, "TARGET"
            elif sl_hit:
                return True, stop_price,   "STOP"
            elif tp_hit:
                return True, target_price, "TARGET"

        return False, 0.0, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Trade recording
    # ─────────────────────────────────────────────────────────────────────────

    def _record_trade(
        self,
        side:         str,
        entry_price:  float,
        exit_price:   float,
        exit_reason:  str,
        entry_bar_ts: pd.Timestamp,
        exit_bar_ts:  pd.Timestamp,
    ):
        """Compute P&L and append a completed trade to self.trades."""
        points_pnl = (exit_price - entry_price if side == "Long"
                      else entry_price - exit_price)

        ticks_pnl  = points_pnl / TICK_SIZE
        dollar_pnl = ticks_pnl * TICK_VALUE_DOLLARS * ORDER_QTY

        self.trades.append({
            "side":        side,
            "entry_date":  entry_bar_ts.date().isoformat(),
            "entry_time":  entry_bar_ts.strftime("%H:%M"),
            "exit_time":   exit_bar_ts.strftime("%H:%M"),
            "entry_price": round(entry_price, 2),
            "exit_price":  round(exit_price, 2),
            "exit_reason": exit_reason,
            "points_pnl":  round(points_pnl, 2),
            "ticks_pnl":   round(ticks_pnl, 2),
            "dollar_pnl":  round(dollar_pnl, 2),
            "win":         dollar_pnl > 0,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Performance report
    # ─────────────────────────────────────────────────────────────────────────

    def report(self) -> Dict:
        """
        Compute and print a full performance summary.
        Returns a dict of all metrics so callers can use them programmatically.
        """
        if not self.trades:
            print("[BACKTEST] No trades to report.")
            return {}

        trades_df   = pd.DataFrame(self.trades)
        total       = len(trades_df)
        wins        = trades_df["win"].sum()
        losses      = total - wins
        win_rate    = wins / total * 100

        pnl         = trades_df["dollar_pnl"]
        total_pnl   = pnl.sum()
        avg_win     = pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0
        avg_loss    = pnl[pnl < 0].mean() if (pnl < 0).any() else 0.0
        profit_factor = (pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())
                         if (pnl < 0).any() and (pnl > 0).any() else float("inf"))
        rr_ratio    = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Equity curve and drawdown
        equity      = pnl.cumsum()
        peak        = equity.cummax()
        drawdown    = equity - peak                 # all ≤ 0
        max_dd      = drawdown.min()
        max_dd_pct  = (max_dd / (peak.max() + 1e-9)) * 100  # avoid /0

        # Sharpe ratio (annualised, assuming 252 trading days)
        if pnl.std() != 0:
            sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Best / worst single trade
        best_trade  = pnl.max()
        worst_trade = pnl.min()

        # Date range
        date_range  = (f"{trades_df['entry_date'].iloc[0]}  →  "
                       f"{trades_df['entry_date'].iloc[-1]}")

        # Exit breakdown
        exit_counts = trades_df["exit_reason"].value_counts().to_dict()

        metrics = {
            "total_trades":   total,
            "wins":           int(wins),
            "losses":         int(losses),
            "win_rate_%":     round(win_rate, 1),
            "total_pnl_$":    round(total_pnl, 2),
            "avg_win_$":      round(avg_win, 2),
            "avg_loss_$":     round(avg_loss, 2),
            "profit_factor":  round(profit_factor, 2),
            "rr_ratio":       round(rr_ratio, 2),
            "max_drawdown_$": round(max_dd, 2),
            "max_drawdown_%": round(max_dd_pct, 1),
            "sharpe_ratio":   round(sharpe, 2),
            "best_trade_$":   round(best_trade, 2),
            "worst_trade_$":  round(worst_trade, 2),
            "date_range":     date_range,
            "exit_breakdown": exit_counts,
        }

        # ── Console output ────────────────────────────────────────────────────
        sep = "─" * 50
        print(f"\n{sep}")
        print(f"  BACKTEST RESULTS — {BACKTEST_SYMBOL}  [{BACKTEST_INTERVAL}]")
        print(f"  {date_range}")
        print(sep)
        print(f"  Total Trades:    {total}")
        print(f"  Wins / Losses:   {wins} / {losses}  ({win_rate:.1f}% win rate)")
        print(f"  Total P&L:       ${total_pnl:+.2f}")
        print(f"  Avg Win:         ${avg_win:+.2f}")
        print(f"  Avg Loss:        ${avg_loss:+.2f}")
        print(f"  Profit Factor:   {profit_factor:.2f}")
        print(f"  R:R Ratio:       {rr_ratio:.2f}")
        print(f"  Max Drawdown:    ${max_dd:.2f}  ({max_dd_pct:.1f}%)")
        print(f"  Sharpe Ratio:    {sharpe:.2f}")
        print(f"  Best Trade:      ${best_trade:+.2f}")
        print(f"  Worst Trade:     ${worst_trade:+.2f}")
        print(f"  Exit Breakdown:  {exit_counts}")
        print(sep)

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────────────────

    def save_results(self, filepath: str = BACKTEST_RESULTS_FILE):
        """Write every simulated trade to a CSV file."""
        if not self.trades:
            print("[BACKTEST] No trades to save.")
            return

        fieldnames = list(self.trades[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.trades)

        print(f"[BACKTEST] Results saved → {filepath}  ({len(self.trades)} rows)")

    def equity_curve(self) -> pd.Series:
        """Return the cumulative P&L Series (equity curve) for plotting."""
        if not self.trades:
            return pd.Series(dtype=float)
        pnl = pd.DataFrame(self.trades)["dollar_pnl"]
        return pnl.cumsum().reset_index(drop=True)


# =============================================================================
# CLI entry point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the MNQ EMA/RSI strategy on historical data."
    )
    p.add_argument("--symbol",   default=BACKTEST_SYMBOL,
                   help=f"yfinance ticker (default: {BACKTEST_SYMBOL})")
    p.add_argument("--interval", default=BACKTEST_INTERVAL,
                   choices=["1m","2m","5m","15m","30m","1h","1d"],
                   help=f"Bar size (default: {BACKTEST_INTERVAL})")
    p.add_argument("--period",   default=BACKTEST_PERIOD,
                   help=f"yfinance period, e.g. '2y' (default: {BACKTEST_PERIOD})")
    p.add_argument("--start",    default=BACKTEST_START,
                   help="Start date YYYY-MM-DD (overrides --period)")
    p.add_argument("--end",      default=BACKTEST_END,
                   help="End date YYYY-MM-DD (overrides --period)")
    p.add_argument("--csv",      default="",
                   help="Path to a local OHLCV CSV file (skips yfinance download)")
    p.add_argument("--save",     action="store_true",
                   help="Save trade results to backtest_results.csv")
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        df = load_from_csv(args.csv)
    else:
        df = load_from_yfinance(
            symbol   = args.symbol,
            interval = args.interval,
            period   = args.period,
            start    = args.start,
            end      = args.end,
        )

    # ── Run backtest ──────────────────────────────────────────────────────────
    bt = Backtester()
    bt.load(df)
    bt.run()
    bt.report()

    if args.save:
        bt.save_results()


if __name__ == "__main__":
    main()
