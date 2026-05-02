#!/usr/bin/env python3
"""
backtest.py — Historical strategy backtester for MNQ-BOT / QQQ

Downloads 5-minute QQQ bars (60 days) and simulates the live strategy
with the EXACT same rules: EMA9/21 regime filter, RSI range, RSI overbought
memory, trend strength, RSI direction, session hours, entry cutoff,
1 trade/day, SL = $1.00/share, TP = $3.00/share, 10 shares.

Also tests 4 RSI_LONG_MAX variants and prints a comparison table.
Results saved to backtest_results.csv.

Interval / lookback can be changed via BAR_INTERVAL / LOOKBACK_DAYS below.
  "1m"  → max ~30 days   (highest resolution, least history)
  "5m"  → max 60 days    (good balance — default)
  "1h"  → max 730 days   (most history, coarser entries)
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timedelta, time as dtime, date
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency: pip install yfinance")

try:
    import pytz
    EASTERN = pytz.timezone("US/Eastern")
except ImportError:
    sys.exit("Missing dependency: pip install pytz")


# ─── Strategy constants (mirrored from config.py) ─────────────────────────────
SYMBOL          = "QQQ"
ORDER_QTY       = 10
STOP_LOSS       = 1.00     # $ per share
TAKE_PROFIT     = 3.00     # $ per share

SESSION_START   = dtime(9, 30)
SESSION_END     = dtime(11, 30)
ENTRY_CUTOFF    = dtime(11, 0)

EMA_FAST_P      = 9
EMA_SLOW_P      = 21
RSI_P           = 14

RSI_LONG_MIN    = 35
RSI_SHORT_MIN   = 50
RSI_SHORT_MAX   = 65

# ─── RSI_LONG_MAX variants ────────────────────────────────────────────────────
VARIANTS: dict[str, float] = {
    "A  RSI_LONG_MAX=50 (current)": 50,
    "B  RSI_LONG_MAX=55":           55,
    "C  RSI_LONG_MAX=58":           58,
    "D  RSI_LONG_MAX=62":           62,
}

RESULTS_FILE  = "backtest_results.csv"
MIN_BARS      = EMA_SLOW_P + 5   # warmup bars before evaluating signals

# ─── Data settings ────────────────────────────────────────────────────────────
BAR_INTERVAL  = "5m"   # yfinance interval: "1m" / "5m" / "1h"
LOOKBACK_DAYS = 60     # how many calendar days to fetch


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data download
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance column names to lowercase flat strings."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename = {}
    for col in df.columns:
        for base in ("open", "high", "low", "close", "volume"):
            if col == base or col.startswith(base + "_"):
                rename[col] = base
    df = df.rename(columns=rename)

    needed = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[needed]


def download_data(
    symbol: str = SYMBOL,
    interval: str = BAR_INTERVAL,
    days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Download historical bars from Yahoo Finance.

    5m / 1h intervals: single call (Yahoo Finance supports up to 60 / 730 days).
    1m interval: chunked 6-day calls (capped at ~30 days by Yahoo Finance).
    """
    end_dt   = datetime.now(EASTERN)
    start_dt = end_dt - timedelta(days=days)

    print(f"\nDownloading {symbol} {interval} bars …")
    print(f"  Requested range : {start_dt.date()} → {end_dt.date()}  ({days} days)")

    def _fetch_period(period_str: str) -> pd.DataFrame:
        """Download using yfinance period string — avoids boundary edge cases."""
        raw = yf.download(
            symbol,
            period=period_str,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        return _flatten_columns(raw)

    def _fetch_range(start: datetime, end: datetime) -> pd.DataFrame:
        """Download using explicit start/end dates."""
        raw = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return pd.DataFrame()
        return _flatten_columns(raw)

    if interval == "1m":
        # Yahoo Finance caps 1m to ~30 days; fetch in 6-day chunks
        print(f"  (1m interval — chunking in 6-day windows)\n")
        chunks: list[pd.DataFrame] = []
        chunk_end = end_dt
        while chunk_end > start_dt:
            chunk_start = max(chunk_end - timedelta(days=6), start_dt)
            try:
                chunk = _fetch_range(chunk_start, chunk_end)
                if not chunk.empty:
                    chunks.append(chunk)
                    print(
                        f"  ✓ {chunk_start.date()} – {chunk_end.date()} ({len(chunk):,} bars)",
                        end="\r",
                    )
            except Exception as exc:
                print(f"\n  ✗ {chunk_start.date()}–{chunk_end.date()} skipped: {exc}")
            chunk_end = chunk_start
        print()
        if not chunks:
            sys.exit("No data downloaded.")
        df = pd.concat(chunks[::-1])
    else:
        # Single period call for 5m / 1h / 1d — yfinance handles boundary internally
        # Use (days-1)d to stay safely within Yahoo Finance's rolling window limit
        period_str = f"{days - 1}d"
        print(f"  (fetching via period='{period_str}' — avoids boundary edge cases)\n")
        df = _fetch_period(period_str)
        if df.empty:
            sys.exit("No data downloaded. Check internet connection and ticker symbol.")

    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    # Ensure US/Eastern timezone
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC").tz_convert(EASTERN)
    else:
        df.index = df.index.tz_convert(EASTERN)

    # Keep only regular session bars
    df = df[
        (df.index.time >= SESSION_START) &
        (df.index.time <= SESSION_END)
    ]

    print(f"\n  Bars loaded  : {len(df):,}")
    print(f"  Actual range : {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Indicators
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ema_fast, ema_slow, rsi columns to a copy of df."""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_P, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_P, adjust=False).mean()

    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=RSI_P - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_P - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(100)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Signal evaluation  (exact mirror of strategy.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _regime(close_slice: np.ndarray, ema_slow_slice: np.ndarray) -> str:
    """BULL / BEAR / UNCLEAR based on last 3 bars above/below EMA21."""
    above = all(c > e for c, e in zip(close_slice, ema_slow_slice))
    below = all(c < e for c, e in zip(close_slice, ema_slow_slice))
    if above:
        return "BULL"
    if below:
        return "BEAR"
    return "UNCLEAR"


def eval_signal(
    i: int,
    close: np.ndarray,
    ema_fast: np.ndarray,
    ema_slow: np.ndarray,
    rsi: np.ndarray,
    rsi_long_max: float,
) -> str:
    """
    Replicate every filter from strategy.py._apply_filters() for bar i.
    Returns 'LONG', 'SHORT', or 'HOLD'.
    """
    c   = close[i]
    ef  = ema_fast[i]
    es  = ema_slow[i]
    r   = rsi[i]
    ef0 = ema_fast[i - 1]
    r0  = rsi[i - 1]

    ema_rising  = ef > ef0
    ema_falling = ef < ef0

    # ── Base conditions ───────────────────────────────────────────────────────
    long_base  = c > es and ema_rising
    short_base = c < es and ema_falling

    if not long_base and not short_base:
        return "HOLD"

    candidate = "LONG" if long_base else "SHORT"

    # ── Filter 0: Regime (BULL→LONG only, BEAR→SHORT only, UNCLEAR→block) ────
    reg = _regime(close[i - 2: i + 1], ema_slow[i - 2: i + 1])
    if reg == "UNCLEAR":
        return "HOLD"
    if reg == "BULL" and candidate == "SHORT":
        return "HOLD"
    if reg == "BEAR" and candidate == "LONG":
        return "HOLD"

    # ── Filter 1: RSI range ───────────────────────────────────────────────────
    if candidate == "LONG":
        if not (RSI_LONG_MIN < r < rsi_long_max):
            return "HOLD"
    else:
        if not (RSI_SHORT_MIN < r < RSI_SHORT_MAX):
            return "HOLD"

    # ── Filter 2: RSI overbought memory (SHORT only) ──────────────────────────
    if candidate == "SHORT":
        start = max(0, i - 4)
        if rsi[start: i + 1].max() > 65:
            return "HOLD"

    # ── Filter 3: Trend strength — price on correct side of EMA21 ≥3 bars ────
    last3_c = close[i - 2: i + 1]
    last3_e = ema_slow[i - 2: i + 1]
    if candidate == "LONG":
        if not all(c_ > e_ for c_, e_ in zip(last3_c, last3_e)):
            return "HOLD"
    else:
        if not all(c_ < e_ for c_, e_ in zip(last3_c, last3_e)):
            return "HOLD"

    # ── Filter 4: RSI direction — rising (LONG) or falling (SHORT) ───────────
    if candidate == "LONG" and not (r > r0):
        return "HOLD"
    if candidate == "SHORT" and not (r < r0):
        return "HOLD"

    return candidate


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Simulation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, rsi_long_max: float) -> list[dict]:
    """
    Simulate the full strategy on `df` with a given RSI_LONG_MAX.
    Returns a list of trade dicts.
    """
    close_a    = df["close"].values
    high_a     = df["high"].values
    low_a      = df["low"].values
    ema_fast_a = df["ema_fast"].values
    ema_slow_a = df["ema_slow"].values
    rsi_a      = df["rsi"].values
    times      = df.index

    trades: list[dict]        = []
    daily_count: dict[date, int] = {}

    in_trade     = False
    entry_px     = 0.0
    entry_ts     = None
    side         = ""
    stop_px      = 0.0
    target_px    = 0.0
    entry_rsi_v  = 0.0

    for i in range(MIN_BARS, len(df)):
        ts     = times[i]
        bar_t  = ts.time()
        bar_dt = ts.date()

        # ── Manage open position ──────────────────────────────────────────────
        if in_trade:
            hi = high_a[i]
            lo = low_a[i]
            c  = close_a[i]

            exit_reason: str | None = None
            exit_px: float          = 0.0

            if side == "LONG":
                if lo <= stop_px:                  # SL check first (conservative)
                    exit_reason = "STOP"
                    exit_px     = stop_px
                elif hi >= target_px:
                    exit_reason = "TARGET"
                    exit_px     = target_px
            else:  # SHORT
                if hi >= stop_px:
                    exit_reason = "STOP"
                    exit_px     = stop_px
                elif lo <= target_px:
                    exit_reason = "TARGET"
                    exit_px     = target_px

            # Force-close at session end
            if exit_reason is None and bar_t >= SESSION_END:
                exit_reason = "SESSION_END"
                exit_px     = c

            if exit_reason:
                pnl = (
                    (exit_px - entry_px) * ORDER_QTY
                    if side == "LONG"
                    else (entry_px - exit_px) * ORDER_QTY
                )
                trades.append({
                    "date":        bar_dt.isoformat(),
                    "month":       ts.strftime("%Y-%m"),
                    "side":        side,
                    "entry_time":  entry_ts.strftime("%H:%M"),
                    "exit_time":   ts.strftime("%H:%M"),
                    "entry_price": round(entry_px, 4),
                    "exit_price":  round(exit_px, 4),
                    "exit_reason": exit_reason,
                    "pnl":         round(pnl, 2),
                    "entry_rsi":   round(entry_rsi_v, 2),
                })
                in_trade = False
            continue  # don't look for new entries this bar

        # ── Look for new entries ──────────────────────────────────────────────

        # Only during session hours, before entry cutoff
        if bar_t < SESSION_START or bar_t >= ENTRY_CUTOFF:
            continue

        # Max 1 trade per calendar day
        if daily_count.get(bar_dt, 0) >= 1:
            continue

        sig = eval_signal(i, close_a, ema_fast_a, ema_slow_a, rsi_a, rsi_long_max)

        if sig == "HOLD":
            continue

        # Enter at close of signal bar
        entry_px     = close_a[i]
        entry_ts     = ts
        side         = sig
        entry_rsi_v  = rsi_a[i]
        in_trade     = True
        daily_count[bar_dt] = daily_count.get(bar_dt, 0) + 1

        if sig == "LONG":
            stop_px   = entry_px - STOP_LOSS
            target_px = entry_px + TAKE_PROFIT
        else:
            stop_px   = entry_px + STOP_LOSS
            target_px = entry_px - TAKE_PROFIT

    # Close any trade still open at end of data
    if in_trade:
        last_c  = close_a[-1]
        last_ts = times[-1]
        pnl     = (
            (last_c - entry_px) * ORDER_QTY
            if side == "LONG"
            else (entry_px - last_c) * ORDER_QTY
        )
        trades.append({
            "date":        last_ts.date().isoformat(),
            "month":       last_ts.strftime("%Y-%m"),
            "side":        side,
            "entry_time":  entry_ts.strftime("%H:%M"),
            "exit_time":   last_ts.strftime("%H:%M"),
            "entry_price": round(entry_px, 4),
            "exit_price":  round(last_c, 4),
            "exit_reason": "DATA_END",
            "pnl":         round(pnl, 2),
            "entry_rsi":   round(entry_rsi_v, 2),
        })

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: list[dict]) -> dict:
    """Aggregate stats from a list of trade dicts."""
    if not trades:
        return dict(
            total_trades=0, long_trades=0, short_trades=0,
            wins=0, losses=0, win_rate=0.0, total_pnl=0.0,
            profit_factor=0.0, max_drawdown=0.0,
            avg_win=0.0, avg_loss=0.0,
            gross_profit=0.0, gross_loss=0.0,
            best_month="N/A", worst_month="N/A",
        )

    tdf  = pd.DataFrame(trades)
    pnls = tdf["pnl"].astype(float)

    wins_s  = pnls[pnls > 0]
    loss_s  = pnls[pnls <= 0]

    gross_profit  = wins_s.sum()
    gross_loss    = abs(loss_s.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Max drawdown on equity curve
    cum     = pnls.cumsum()
    drawdown = cum - cum.cummax()
    max_dd  = drawdown.min()

    # Monthly P&L for best/worst
    monthly = tdf.groupby("month")["pnl"].sum()
    best_m  = f"{monthly.idxmax()} (${monthly.max():+.2f})" if not monthly.empty else "N/A"
    worst_m = f"{monthly.idxmin()} (${monthly.min():+.2f})" if not monthly.empty else "N/A"

    return dict(
        total_trades  = int(len(pnls)),
        long_trades   = int((tdf["side"] == "LONG").sum()),
        short_trades  = int((tdf["side"] == "SHORT").sum()),
        wins          = int(len(wins_s)),
        losses        = int(len(loss_s)),
        win_rate      = round(len(wins_s) / len(pnls) * 100, 1),
        total_pnl     = round(float(pnls.sum()), 2),
        profit_factor = round(profit_factor, 2) if profit_factor != float("inf") else 9999.0,
        max_drawdown  = round(float(max_dd), 2),
        avg_win       = round(float(wins_s.mean()), 2) if not wins_s.empty  else 0.0,
        avg_loss      = round(float(loss_s.mean()), 2) if not loss_s.empty  else 0.0,
        gross_profit  = round(float(gross_profit), 2),
        gross_loss    = round(float(gross_loss), 2),
        best_month    = best_m,
        worst_month   = worst_m,
    )


def monthly_table(trades: list[dict]) -> pd.DataFrame:
    """Per-month breakdown: trades, wins, win%, total P&L, cumulative P&L."""
    if not trades:
        return pd.DataFrame()
    tdf = pd.DataFrame(trades)
    grp = (
        tdf.groupby("month")
        .agg(
            trades    = ("pnl", "count"),
            wins      = ("pnl", lambda x: (x > 0).sum()),
            month_pnl = ("pnl", "sum"),
        )
        .reset_index()
    )
    grp["win_pct"]  = (grp["wins"] / grp["trades"] * 100).round(1)
    grp["month_pnl"] = grp["month_pnl"].round(2)
    grp["cum_pnl"]  = grp["month_pnl"].cumsum().round(2)
    return grp


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Display helpers
# ═══════════════════════════════════════════════════════════════════════════════

SEP_WIDE  = "═" * 68
SEP_THIN  = "─" * 68


def _pnl_str(v: float) -> str:
    return f"${v:+.2f}"


def print_variant_results(label: str, m: dict, monthly: pd.DataFrame) -> None:
    """Print full stats for one RSI variant."""
    print(f"\n{SEP_WIDE}")
    print(f"  VARIANT {label}")
    print(SEP_THIN)
    if m["total_trades"] == 0:
        print("  No trades generated for this variant.")
        return

    pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] < 9999 else "∞"

    print(f"  Total Trades     :  {m['total_trades']}  "
          f"(Long {m['long_trades']} / Short {m['short_trades']})")
    print(f"  Wins / Losses    :  {m['wins']} / {m['losses']}")
    print(f"  Win Rate         :  {m['win_rate']}%")
    print(f"  Total P&L        :  {_pnl_str(m['total_pnl'])}")
    print(f"  Gross Profit     :  ${m['gross_profit']:.2f}")
    print(f"  Gross Loss       :  ${m['gross_loss']:.2f}")
    print(f"  Profit Factor    :  {pf_str}")
    print(f"  Max Drawdown     :  {_pnl_str(m['max_drawdown'])}")
    print(f"  Avg Win          :  {_pnl_str(m['avg_win'])}")
    print(f"  Avg Loss         :  {_pnl_str(m['avg_loss'])}")
    print(f"  Best Month       :  {m['best_month']}")
    print(f"  Worst Month      :  {m['worst_month']}")

    if not monthly.empty:
        print()
        print("  Monthly Breakdown:")
        print(f"  {'Month':<9} {'Trades':>6} {'Wins':>5} {'Win%':>6} {'P&L':>10} {'Cum P&L':>10}")
        print("  " + "─" * 50)
        for _, row in monthly.iterrows():
            print(
                f"  {row['month']:<9} {int(row['trades']):>6} "
                f"{int(row['wins']):>5} {row['win_pct']:>5.1f}% "
                f"{_pnl_str(row['month_pnl']):>10} "
                f"{_pnl_str(row['cum_pnl']):>10}"
            )


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print side-by-side comparison of all RSI variants."""
    print(f"\n{SEP_WIDE}")
    print("  RSI VARIANT COMPARISON")
    print(SEP_THIN)
    hdr = f"  {'Variant':<32} {'Trades':>6} {'Win%':>6} {'Total P&L':>11} {'Prof.F':>7} {'MaxDD':>10}"
    print(hdr)
    print("  " + "─" * 66)

    best_label  = ""
    best_pnl    = float("-inf")

    for label, m in results.items():
        if m["total_trades"] == 0:
            print(f"  {label:<32} {'—':>6} {'—':>6} {'—':>11} {'—':>7} {'—':>10}")
            continue

        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] < 9999 else "∞"
        row = (
            f"  {label:<32} "
            f"{m['total_trades']:>6} "
            f"{m['win_rate']:>5.1f}% "
            f"{_pnl_str(m['total_pnl']):>11} "
            f"{pf_str:>7} "
            f"{_pnl_str(m['max_drawdown']):>10}"
        )
        print(row)

        if m["total_pnl"] > best_pnl:
            best_pnl   = m["total_pnl"]
            best_label = label

    print("  " + "─" * 66)
    if best_label:
        print(f"\n  ★ Best total P&L  →  {best_label}  ({_pnl_str(best_pnl)})")

    # Also show best by profit factor
    best_pf_label = max(
        (l for l, m in results.items() if m["total_trades"] > 0),
        key=lambda l: results[l]["profit_factor"],
        default="",
    )
    if best_pf_label:
        pf_val = results[best_pf_label]["profit_factor"]
        pf_str = f"{pf_val:.2f}" if pf_val < 9999 else "∞"
        print(f"  ★ Best profit factor  →  {best_pf_label}  (PF = {pf_str})")

    print(f"\n{SEP_WIDE}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CSV export
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(all_trades: dict[str, list[dict]], all_metrics: dict[str, dict]) -> None:
    """
    Save two sheets to backtest_results.csv:
      • Summary stats per variant
      • All individual trades with variant label
    """
    # ── Summary sheet ─────────────────────────────────────────────────────────
    summary_rows = []
    for label, m in all_metrics.items():
        row = {"variant": label}
        row.update(m)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ── Trades sheet ──────────────────────────────────────────────────────────
    trade_rows = []
    for label, trades in all_trades.items():
        for t in trades:
            row = {"variant": label}
            row.update(t)
            trade_rows.append(row)
    trades_df = pd.DataFrame(trade_rows)

    with open(RESULTS_FILE, "w", newline="") as fh:
        fh.write("# SUMMARY\n")
        summary_df.to_csv(fh, index=False)
        fh.write("\n# TRADES\n")
        trades_df.to_csv(fh, index=False)

    print(f"\n  Results saved → {RESULTS_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(SEP_WIDE)
    print(f"  MNQ-BOT BACKTEST — {SYMBOL} {BAR_INTERVAL.upper()} BARS  |  "
          f"SL=${STOP_LOSS}  TP=${TAKE_PROFIT}  Qty={ORDER_QTY}  Lookback={LOOKBACK_DAYS}d")
    print(f"  Session: 09:30–11:30 ET  |  Entry cutoff: 11:00 ET  |  Max 1 trade/day")
    print(SEP_WIDE)

    # ── Download & prepare data ───────────────────────────────────────────────
    raw   = download_data(SYMBOL, BAR_INTERVAL, LOOKBACK_DAYS)
    df    = compute_indicators(raw)

    print(f"\n  Trading days in dataset : {df.index.normalize().nunique()}")
    print(f"  Indicators computed     : EMA{EMA_FAST_P}, EMA{EMA_SLOW_P}, RSI{RSI_P}")

    # ── Run all variants ──────────────────────────────────────────────────────
    all_trades  : dict[str, list[dict]] = {}
    all_metrics : dict[str, dict]       = {}

    for label, rsi_long_max in VARIANTS.items():
        print(f"\n  Running {label} …", end=" ", flush=True)
        trades  = run_backtest(df, rsi_long_max)
        metrics = compute_metrics(trades)
        all_trades[label]  = trades
        all_metrics[label] = metrics
        print(f"{metrics['total_trades']} trades  |  P&L: {_pnl_str(metrics['total_pnl'])}")

    # ── Print detailed results per variant ────────────────────────────────────
    for label, metrics in all_metrics.items():
        monthly = monthly_table(all_trades[label])
        print_variant_results(label, metrics, monthly)

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(all_metrics)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_results(all_trades, all_metrics)

    # ── Final summary ─────────────────────────────────────────────────────────
    best = max(
        ((l, m) for l, m in all_metrics.items() if m["total_trades"] > 0),
        key=lambda x: x[1]["total_pnl"],
        default=("N/A", {}),
    )
    print(f"\n  RECOMMENDATION: {best[0]}")
    if best[1]:
        print(
            f"  → {best[1]['total_trades']} trades, "
            f"{best[1]['win_rate']}% win rate, "
            f"P&L {_pnl_str(best[1]['total_pnl'])}, "
            f"MaxDD {_pnl_str(best[1]['max_drawdown'])}"
        )
    print(f"\n{SEP_WIDE}\n")


if __name__ == "__main__":
    main()
