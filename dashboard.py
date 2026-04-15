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
)

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


def today_str() -> str:
    return datetime.datetime.now(tz=_TZ).date().isoformat()


def filter_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    return df[df["date"] == today_str()].copy()


def _closed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pnl_dollars" not in df.columns:
        return pd.DataFrame()
    return df[df["pnl_dollars"].notna()].copy()


def _week_pnl(df: pd.DataFrame) -> float:
    c = _closed(df)
    if c.empty:
        return 0.0
    now = datetime.datetime.now(tz=_TZ).date()
    week_start = now - datetime.timedelta(days=now.weekday())
    dates = pd.to_datetime(c["date"]).dt.date
    return float(c.loc[(dates >= week_start) & (dates <= now), "pnl_dollars"].sum())


def _month_pnl(df: pd.DataFrame) -> float:
    c = _closed(df)
    if c.empty:
        return 0.0
    now = datetime.datetime.now(tz=_TZ)
    d = pd.to_datetime(c["date"])
    return float(c.loc[(d.dt.year == now.year) & (d.dt.month == now.month), "pnl_dollars"].sum())


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


def render_kpi_row(state: dict, today_trades: pd.DataFrame):
    """
    Four top-level KPI cards:
        Daily P&L  |  Drawdown %  |  Session  |  Trades taken
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Daily P&L",
            value=_pnl_delta_str(daily_pnl),
            delta=f"{daily_pnl / abs(DAILY_LOSS_LIMIT) * 100:.1f}% of limit",
        )

    with col2:
        dd_color = "normal" if drawdown_pct < 50 else ("off" if drawdown_pct < 80 else "inverse")
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


def render_period_pnl(all_trades: pd.DataFrame):
    """Week P&L and Month P&L side by side."""
    week_pnl  = _week_pnl(all_trades)
    month_pnl = _month_pnl(all_trades)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("This Week P&L",  _pnl_delta_str(week_pnl))
    with col2:
        st.metric("This Month P&L", _pnl_delta_str(month_pnl))


def render_pnl_calendar(all_trades: pd.DataFrame):
    """Calendar heatmap: daily P&L cells, weekly row totals, monthly total."""
    import calendar as cal_mod
    import streamlit.components.v1 as components

    st.subheader("P&L Calendar")

    c = _closed(all_trades)
    daily_pnl: dict = {}
    if not c.empty and "date" in c.columns:
        c["_d"] = c["date"].astype(str).str[:10]
        daily_pnl = c.groupby("_d")["pnl_dollars"].sum().to_dict()

    now = datetime.datetime.now(tz=_TZ)

    # Month selector: all months with data + current month
    month_opts = sorted(
        {d[:7] for d in daily_pnl} | {now.strftime("%Y-%m")},
        reverse=True,
    )
    col_sel, _ = st.columns([1, 4])
    with col_sel:
        selected = st.selectbox("Month", options=month_opts, index=0,
                                label_visibility="collapsed")

    year      = int(selected[:4])
    month     = int(selected[5:7])
    weeks     = cal_mod.monthcalendar(year, month)
    month_lbl = datetime.date(year, month, 1).strftime("%B %Y")
    today_iso = now.date().isoformat()

    # ── Cells ──────────────────────────────────────────────────────────────
    DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

    header_cells = "".join(
        f'<th style="color:#9ca3af;font-size:12px;font-weight:600;'
        f'padding:8px 4px;text-align:center;letter-spacing:.05em">{d}</th>'
        for d in DAY_NAMES
    )
    header_cells += (
        '<th style="color:#9ca3af;font-size:12px;font-weight:600;'
        'padding:8px 12px;text-align:center;border-left:1px solid #374151;'
        'letter-spacing:.05em">WEEK</th>'
    )

    rows_html   = ""
    month_total = 0.0

    for week in weeks:
        row   = ""
        wk_t  = 0.0
        valid = False

        for day in week:
            if day == 0:
                row += (
                    '<td style="padding:10px 4px;text-align:center;'
                    'border-radius:6px;background:#111827"></td>'
                )
                continue

            valid  = True
            ds     = f"{year:04d}-{month:02d}-{day:02d}"
            pnl    = daily_pnl.get(ds)
            today  = ds == today_iso
            ring   = "box-shadow:0 0 0 2px #3b82f6;" if today else ""

            if pnl is not None:
                wk_t        += pnl
                month_total += pnl
                color = "#4ade80" if pnl >= 0 else "#f87171"
                bg    = "rgba(74,222,128,0.15)" if pnl >= 0 else "rgba(248,113,113,0.15)"
                sign  = "+" if pnl >= 0 else ""
                row += (
                    f'<td style="padding:10px 6px;text-align:center;border-radius:6px;'
                    f'background:{bg};{ring}">'
                    f'<div style="color:#6b7280;font-size:11px;margin-bottom:4px">{day}</div>'
                    f'<div style="color:{color};font-size:15px;font-weight:700;line-height:1">'
                    f'{sign}${pnl:.0f}</div>'
                    f'</td>'
                )
            else:
                row += (
                    f'<td style="padding:10px 6px;text-align:center;border-radius:6px;'
                    f'background:#1f2937;{ring}">'
                    f'<div style="color:#4b5563;font-size:11px;margin-bottom:4px">{day}</div>'
                    f'<div style="color:#374151;font-size:14px">—</div>'
                    f'</td>'
                )

        # Week total
        if valid:
            if wk_t != 0:
                wc    = "#4ade80" if wk_t >= 0 else "#f87171"
                wsign = "+" if wk_t >= 0 else ""
                row += (
                    f'<td style="padding:10px 12px;text-align:center;font-weight:700;'
                    f'font-size:15px;color:{wc};border-left:1px solid #374151">'
                    f'{wsign}${wk_t:.0f}</td>'
                )
            else:
                row += (
                    '<td style="padding:10px 12px;text-align:center;color:#374151;'
                    'border-left:1px solid #374151">—</td>'
                )
        rows_html += f"<tr>{row}</tr>"

    # Month total footer
    mc    = "#4ade80" if month_total >= 0 else "#f87171"
    msign = "+" if month_total >= 0 else ""
    no_data_note = (
        '<tr><td colspan="8" style="padding:16px;text-align:center;'
        'color:#6b7280;font-size:13px;border-top:1px solid #374151">'
        'No trades recorded yet — trades appear here once the bot closes a position.</td></tr>'
        if not daily_pnl else ""
    )
    footer = (
        f'<tr>'
        f'<td colspan="7" style="padding:10px 8px;text-align:right;color:#6b7280;'
        f'font-size:12px;border-top:1px solid #374151">Month Total</td>'
        f'<td style="padding:10px 12px;text-align:center;font-weight:700;font-size:17px;'
        f'color:{mc};border-left:1px solid #374151;border-top:1px solid #374151">'
        f'{msign}${month_total:.0f}</td>'
        f'</tr>'
    )

    n_weeks    = len(weeks)
    cal_height = 46 + n_weeks * 64 + 50 + (40 if not daily_pnl else 0)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    margin: 0; padding: 0;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  .cal-wrap {{
    width: 100%;
    background: #111827;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1f2937;
  }}
  .cal-title {{
    padding: 12px 16px 0;
    color: #e5e7eb;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: .03em;
  }}
  table {{
    border-collapse: separate;
    border-spacing: 4px;
    width: 100%;
    padding: 8px;
    box-sizing: border-box;
  }}
</style>
</head>
<body>
<div class="cal-wrap">
  <div class="cal-title">{month_lbl}</div>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {rows_html}
      {no_data_note}
      {footer}
    </tbody>
  </table>
</div>
</body>
</html>
"""
    components.html(html, height=cal_height, scrolling=False)


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
    render_kpi_row(state, today_trades)
    render_period_pnl(all_trades)
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

    # ── P&L Calendar ──────────────────────────────────────────────────────────
    render_pnl_calendar(all_trades)
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
