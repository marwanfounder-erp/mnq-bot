#!/usr/bin/env python3
# =============================================================================
# dashboard.py — Streamlit live monitoring dashboard
#
# Run:
#   streamlit run dashboard.py
#
# The dashboard reads two files written by main.py:
#   • trades.csv     — completed trade log (always up-to-date)
#   • bot_state.json — lightweight live state snapshot (written each poll cycle)
#
# Auto-refreshes every DASHBOARD_REFRESH_SECONDS (default 10 s).
# Works even when main.py is not running — shows "Bot offline" badge.
# =============================================================================

import json
import os
import csv
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import (
    LOG_FILE,
    STATE_FILE,
    DASHBOARD_REFRESH_SECONDS,
    TIMEZONE,
    SYMBOL,
    DAILY_LOSS_LIMIT,
    PAPER_TRADING,
    TRADING_START,
    TRADING_END,
    SUPABASE_URL,
    SUPABASE_KEY,
)

# ─── Supabase client (optional) ───────────────────────────────────────────────
_sb = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        from supabase import create_client as _create_client
        _sb = _create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    pass

_TZ = ZoneInfo(TIMEZONE)

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="MNQ Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Minimal custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighter metric cards */
    [data-testid="metric-container"] { background:#1e1e2e; border-radius:8px;
        padding:12px 16px; }
    /* Table font size */
    .stDataFrame { font-size: 13px; }
    /* Status badges */
    .badge-online  { background:#22c55e; color:#fff; padding:2px 10px;
        border-radius:12px; font-size:12px; font-weight:600; }
    .badge-offline { background:#ef4444; color:#fff; padding:2px 10px;
        border-radius:12px; font-size:12px; font-weight:600; }
    .badge-paper   { background:#f59e0b; color:#fff; padding:2px 10px;
        border-radius:12px; font-size:12px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data loading helpers
# =============================================================================

@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_bot_state() -> dict:
    """
    Read bot_state.json written by main.py each poll cycle.
    Returns an empty dict if the file doesn't exist (bot offline).
    """
    if not Path(STATE_FILE).exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_trades() -> pd.DataFrame:
    """
    Read the complete trades.csv journal.
    Returns an empty DataFrame if the file doesn't exist.
    """
    if not Path(LOG_FILE).exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        # Cast numeric columns safely
        for col in ("entry_price", "exit_price", "pnl_dollars",
                    "pnl_ticks", "daily_pnl", "ema_fast", "ema_slow", "rsi"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def load_trades_from_supabase() -> pd.DataFrame:
    """
    Fetch all closed trades from Supabase.
    Falls back to empty DataFrame if Supabase is unavailable.
    """
    if _sb is None:
        return pd.DataFrame()
    try:
        response = _sb.table("trades").select(
            "trade_id,symbol,side,entry_price,exit_price,"
            "entry_time,exit_time,exit_reason,pnl_dollars,daily_pnl,paper_trade"
        ).not_.is_("exit_price", "null").execute()

        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        for col in ("entry_price", "exit_price", "pnl_dollars", "daily_pnl"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def today_str() -> str:
    return datetime.datetime.now(tz=_TZ).date().isoformat()


def filter_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    return df[df["date"] == today_str()].copy()


# =============================================================================
# Layout helpers
# =============================================================================

def _pnl_delta_str(val: float) -> str:
    return f"+${val:.2f}" if val >= 0 else f"-${abs(val):.2f}"


def render_header(state: dict):
    """Top bar: title, bot status badge, last-updated time."""
    col_title, col_status, col_time = st.columns([3, 1, 1])

    with col_title:
        st.markdown("## 📈 MNQ Trading Bot Dashboard")

    with col_status:
        if not state:
            st.markdown('<span class="badge-offline">OFFLINE</span>',
                        unsafe_allow_html=True)
        elif PAPER_TRADING:
            st.markdown('<span class="badge-paper">PAPER TRADING</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-online">LIVE</span>',
                        unsafe_allow_html=True)

    with col_time:
        now_et = datetime.datetime.now(tz=_TZ).strftime("%H:%M:%S ET")
        st.caption(f"Last refresh: {now_et}")

    st.divider()


def render_kpi_row(state: dict, today_trades: pd.DataFrame, all_trades: pd.DataFrame):
    """
    Five top-level KPI cards:
        Daily P&L  |  Drawdown %  |  Session  |  Trades taken  |  All-Time P&L
    """
    # Compute values
    daily_pnl    = state.get("daily_pnl", 0.0)
    in_trade     = state.get("in_trade", False)
    halted       = state.get("trading_halted", False)
    session_on   = state.get("session_active", False)
    _closed_today = (today_trades[today_trades["exit_price"].notna()]
                     if not today_trades.empty and "exit_price" in today_trades.columns
                     else pd.DataFrame())
    trades_count = state.get("trades_today", len(_closed_today))

    # Drawdown as % of the absolute daily loss limit
    drawdown_pct = (abs(min(daily_pnl, 0)) / abs(DAILY_LOSS_LIMIT)) * 100
    drawdown_pct = min(drawdown_pct, 100.0)

    # All-time P&L — prefer Supabase (full history), fall back to CSV
    sb_trades = load_trades_from_supabase()
    if not sb_trades.empty and "pnl_dollars" in sb_trades.columns:
        alltime_pnl    = sb_trades["pnl_dollars"].sum()
        alltime_trades = len(sb_trades)
    elif not all_trades.empty and "pnl_dollars" in all_trades.columns:
        closed_all     = all_trades[all_trades["pnl_dollars"].notna()]
        alltime_pnl    = closed_all["pnl_dollars"].sum()
        alltime_trades = len(closed_all)
    else:
        alltime_pnl    = 0.0
        alltime_trades = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Daily P&L",
            value=_pnl_delta_str(daily_pnl),
            delta=f"{daily_pnl / abs(DAILY_LOSS_LIMIT) * 100:.1f}% of limit",
        )

    with col2:
        st.metric(
            label="Drawdown vs Limit",
            value=f"{drawdown_pct:.1f}%",
            delta=f"Limit: ${abs(DAILY_LOSS_LIMIT):.0f}",
            delta_color="off",
        )

    with col3:
        session_label = "ACTIVE" if session_on else "CLOSED"
        session_color = "normal" if session_on else "off"
        window = (f"{TRADING_START[0]:02d}:{TRADING_START[1]:02d}–"
                  f"{TRADING_END[0]:02d}:{TRADING_END[1]:02d} ET")
        st.metric(label="Session", value=session_label, delta=window,
                  delta_color=session_color)

    with col4:
        halt_note = " ⛔ HALTED" if halted else ""
        st.metric(label="Trades Today",
                  value=f"{trades_count} / 1{halt_note}",
                  delta="In position" if in_trade else "Flat")

    with col5:
        st.metric(
            label="All-Time P&L",
            value=_pnl_delta_str(alltime_pnl),
            delta=f"{alltime_trades} closed trade{'s' if alltime_trades != 1 else ''}",
            delta_color="off",
        )


def render_position_card(state: dict):
    """Show current open position details (or 'Flat' if none)."""
    st.subheader("Current Position")

    in_trade = state.get("in_trade", False)

    if not state:
        st.info("Bot is offline — start main.py to see live data.")
        return

    if not in_trade:
        st.success("✅  Flat — no open position.")
        return

    side         = state.get("position_side", "—")
    entry_price  = state.get("entry_price", 0.0)
    stop_price   = state.get("stop_price", 0.0)
    target_price = state.get("target_price", 0.0)
    last_price   = state.get("last_price", 0.0)

    # Unrealized P&L estimate
    if side == "Long":
        pts_pnl = last_price - entry_price
    else:
        pts_pnl = entry_price - last_price

    ticks_pnl  = pts_pnl / 0.25
    dollar_pnl = ticks_pnl * 0.50

    color      = "🟢" if dollar_pnl >= 0 else "🔴"
    pnl_str    = _pnl_delta_str(dollar_pnl)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Side",       f"{color} {side} {SYMBOL}")
    col2.metric("Entry",      f"{entry_price:,.2f}")
    col3.metric("Stop Loss",  f"{stop_price:,.2f}",
                delta=f"-{abs(entry_price - stop_price):.2f} pts",
                delta_color="inverse")
    col4.metric("Take Profit", f"{target_price:,.2f}",
                delta=f"+{abs(target_price - entry_price):.2f} pts")
    col5.metric("Unrealized P&L", pnl_str,
                delta=f"Last: {last_price:,.2f}")


def render_indicators(state: dict):
    """Latest EMA / RSI snapshot from the most recent bar evaluation."""
    indicators = state.get("indicators", {})
    if not indicators:
        return

    st.subheader("Latest Indicators")

    ema_fast = indicators.get("ema_fast", 0.0)
    ema_slow = indicators.get("ema_slow", 0.0)
    rsi      = indicators.get("rsi", 50.0)
    last_price = state.get("last_price", 0.0)

    trend = "🐂 Bullish" if last_price > ema_slow else "🐻 Bearish"
    rsi_zone = ("Oversold 🔵" if rsi < 30
                else "Overbought 🔴" if rsi > 70
                else "Neutral ⚪")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Price", f"{last_price:,.2f}")
    col2.metric(f"EMA {9}",   f"{ema_fast:,.2f}")
    col3.metric(f"EMA {21}",  f"{ema_slow:,.2f}")
    col4.metric("RSI (14)",   f"{rsi:.1f}", delta=rsi_zone, delta_color="off")
    col5.metric("Trend",      trend)


def render_drawdown_bar(state: dict):
    """Visual progress bar showing how close daily P&L is to the limit."""
    daily_pnl   = state.get("daily_pnl", 0.0)
    used_pct    = min(abs(min(daily_pnl, 0)) / abs(DAILY_LOSS_LIMIT), 1.0)

    st.subheader("Daily Loss Meter")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.progress(used_pct)
    with col2:
        st.caption(f"{used_pct * 100:.1f}% of ${abs(DAILY_LOSS_LIMIT):.0f} limit")

    if used_pct > 0.8:
        st.warning("⚠️ Approaching daily loss limit — trading will halt soon.")
    if state.get("trading_halted"):
        st.error("🚨 Daily loss limit hit — trading halted for today.")


def render_todays_trades(today_trades: pd.DataFrame):
    """Table of all trades taken today."""
    st.subheader("Today's Trades")

    if today_trades.empty or "exit_price" not in today_trades.columns:
        st.info("No completed trades today.")
        return

    closed = today_trades[today_trades["exit_price"].notna()]

    if closed.empty:
        st.info("No completed trades today.")
        return

    display_cols = ["trade_id", "side", "entry_time", "exit_time",
                    "entry_price", "exit_price", "exit_reason",
                    "pnl_ticks", "pnl_dollars"]
    available = [c for c in display_cols if c in closed.columns]

    styled = closed[available].copy()

    # Colour P&L column: green positive, red negative
    def colour_pnl(val):
        try:
            v = float(val)
            return "color: #22c55e" if v >= 0 else "color: #ef4444"
        except Exception:
            return ""

    st.dataframe(
        styled.style.applymap(colour_pnl, subset=["pnl_dollars"]
                              if "pnl_dollars" in styled.columns else []),
        use_container_width=True,
        hide_index=True,
    )


def render_equity_curve(all_trades: pd.DataFrame):
    """Cumulative P&L line chart across all historical trades."""
    st.subheader("Equity Curve (All Time)")

    closed = all_trades[all_trades["pnl_dollars"].notna()].copy()
    if closed.empty:
        st.info("No trade history yet.")
        return

    closed["cumulative_pnl"] = closed["pnl_dollars"].cumsum()
    closed["label"]          = closed["date"] + " " + closed["entry_time"].fillna("")

    # Build a simple line chart with pandas (no extra plotting library needed)
    chart_data = closed[["cumulative_pnl"]].copy()
    chart_data.index = range(len(chart_data))

    st.line_chart(chart_data, y="cumulative_pnl",
                  use_container_width=True)

    # Summary row below chart
    total_pnl  = closed["pnl_dollars"].sum()
    total_wins = (closed["pnl_dollars"] > 0).sum()
    total_all  = len(closed)
    wr = total_wins / total_all * 100 if total_all > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total P&L",   _pnl_delta_str(total_pnl))
    col2.metric("Total Trades", total_all)
    col3.metric("Win Rate",    f"{wr:.1f}%")


def render_win_loss_bars(all_trades: pd.DataFrame):
    """Simple bar chart: wins vs losses by month."""
    closed = all_trades[all_trades["pnl_dollars"].notna()].copy()
    if closed.empty or len(closed) < 3:
        return

    closed["month"] = pd.to_datetime(closed["date"]).dt.to_period("M").astype(str)
    monthly = (closed.groupby("month")
                     .agg(wins=("pnl_dollars", lambda x: (x > 0).sum()),
                          losses=("pnl_dollars", lambda x: (x < 0).sum()),
                          pnl=("pnl_dollars", "sum"))
                     .reset_index())

    if monthly.empty:
        return

    st.subheader("Monthly Breakdown")
    st.bar_chart(monthly.set_index("month")[["wins", "losses"]],
                 use_container_width=True)


# =============================================================================
# Main dashboard render
# =============================================================================

def main():
    # Load data
    state       = load_bot_state()
    all_trades  = load_trades()
    today_trades = filter_today(all_trades)

    # ── Header ────────────────────────────────────────────────────────────────
    render_header(state)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    render_kpi_row(state, today_trades, all_trades)
    st.divider()

    # ── Drawdown meter ────────────────────────────────────────────────────────
    render_drawdown_bar(state)
    st.divider()

    # ── Current position ──────────────────────────────────────────────────────
    render_position_card(state)
    st.divider()

    # ── Indicators ────────────────────────────────────────────────────────────
    if state:
        render_indicators(state)
        st.divider()

    # ── Today's trades ────────────────────────────────────────────────────────
    render_todays_trades(today_trades)
    st.divider()

    # ── Historical equity curve ───────────────────────────────────────────────
    render_equity_curve(all_trades)
    render_win_loss_bars(all_trades)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    # Clears the cache and re-runs the entire script every N seconds.
    import time
    time.sleep(DASHBOARD_REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    main()
