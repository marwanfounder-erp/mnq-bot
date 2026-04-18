# =============================================================================
# broker.py — Alpaca paper/live trading broker
#
# Wraps the alpaca-py SDK to provide a clean interface used by main.py:
#
#   connect()            — authenticate and validate account
#   fetch_bars()         — 1-min OHLCV bars for indicator calculation
#   get_current_price()  — latest trade price
#   place_market_order() — entry order
#   place_stop_order()   — stop-loss bracket leg
#   place_limit_order()  — take-profit bracket leg
#   cancel_all_orders()  — wipe open bracket orders before closing
#   close_position()     — market-close the open position
#
# Install:  pip install alpaca-py
# =============================================================================

from __future__ import annotations

import time
import datetime
import pandas as pd
from typing import Optional

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    PAPER_TRADING,
    SYMBOL,
    ORDER_QTY,
    BARS_TO_FETCH,
)


class AlpacaBroker:
    """
    Alpaca trading broker adapter.

    In paper-trading mode (PAPER_TRADING=True) all orders go to the Alpaca
    paper environment — no real money is risked.
    """

    def __init__(self):
        self._trading_client = None
        self._data_client    = None
        self._connected      = False

    # ─────────────────────────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Initialise Alpaca clients and verify the account is reachable.
        Returns True on success, False on failure.
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_SECRET_KEY,
                paper      = PAPER_TRADING,
            )
            self._data_client = StockHistoricalDataClient(
                api_key    = ALPACA_API_KEY,
                secret_key = ALPACA_SECRET_KEY,
            )

            account      = self._trading_client.get_account()
            mode         = "PAPER" if PAPER_TRADING else "LIVE"
            equity       = float(account.equity)
            buying_power = float(account.buying_power)
            print(f"[BROKER] Connected to Alpaca {mode} account")
            print(f"[BROKER] Account ID: {account.id}")
            print(f"[BROKER] Equity: ${equity:,.2f}  |  Buying Power: ${buying_power:,.2f}")

            self._connected = True
            return True

        except ImportError:
            print("[BROKER] alpaca-py not installed. Run: pip install alpaca-py")
            return False
        except Exception as exc:
            print(f"[BROKER] Connection failed: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Market data
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_bars(self) -> pd.DataFrame:
        """
        Fetch the most recent 1-minute OHLCV bars for SYMBOL.

        Returns a DataFrame with columns:
            time (datetime), open, high, low, close, volume
        Returns an empty DataFrame on error.
        """
        if not self._connected:
            return pd.DataFrame()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Use a time-window wide enough to cover BARS_TO_FETCH trading minutes.
            # We look back 4× the bar count to account for overnight/weekend gaps.
            now   = datetime.datetime.now(datetime.timezone.utc)
            start = now - datetime.timedelta(minutes=BARS_TO_FETCH * 4)

            request = StockBarsRequest(
                symbol_or_symbols = [SYMBOL],
                timeframe         = TimeFrame.Minute,
                start             = start,
                end               = now,
                feed              = "iex",   # IEX feed — free for all Alpaca accounts
            )

            bars_resp = self._data_client.get_stock_bars(request)
            raw = bars_resp.df

            # Empty response = market closed or no data in range
            if raw.empty or "close" not in raw.columns:
                print(f"[BROKER] No bar data for {SYMBOL} in the last {BARS_TO_FETCH * 4} min "
                      f"(market likely closed).")
                return pd.DataFrame()

            df = raw.reset_index()

            # After reset_index the MultiIndex (symbol, timestamp) becomes columns
            if "symbol" in df.columns:
                df = df[df["symbol"] == SYMBOL].copy()

            df = df.rename(columns={"timestamp": "time"})

            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["close"])
            df = df.sort_values("time").tail(BARS_TO_FETCH).reset_index(drop=True)

            keep = ["time", "open", "high", "low", "close", "volume"]
            df   = df[[c for c in keep if c in df.columns]]

            print(f"[BROKER] Fetched {len(df)} bars for {SYMBOL}")
            return df

        except Exception as exc:
            print(f"[BROKER] fetch_bars error: {exc}")
            return pd.DataFrame()

    def get_current_price(self) -> Optional[float]:
        """
        Return the latest trade price for SYMBOL.
        Returns None if data is unavailable.
        """
        if not self._connected:
            return None

        try:
            from alpaca.data.requests import StockLatestTradeRequest

            request = StockLatestTradeRequest(symbol_or_symbols=[SYMBOL])
            latest  = self._data_client.get_stock_latest_trade(request)
            return float(latest[SYMBOL].price)

        except Exception as exc:
            print(f"[BROKER] get_current_price error: {exc}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Order management
    # ─────────────────────────────────────────────────────────────────────────

    def place_market_order(self, action: str, qty: int = ORDER_QTY,
                           stop_price: float = None,
                           target_price: float = None) -> Optional[dict]:
        """
        Submit a market order, optionally with bracket (SL + TP) legs.

        When stop_price and target_price are both provided the order is
        submitted as a single bracket order (OrderClass.BRACKET) so both
        legs share the same parent and Alpaca never raises
        "insufficient qty available" on the second leg.

        Args:
            action:       "Buy" or "Sell"
            qty:          number of shares
            stop_price:   stop-loss trigger price  (bracket leg, optional)
            target_price: take-profit limit price  (bracket leg, optional)

        Returns:
            dict {"fillPrice": float, "order_id": str} or None on failure.
        """
        if not self._connected:
            return None

        try:
            from alpaca.trading.requests import (MarketOrderRequest,
                                                  TakeProfitRequest,
                                                  StopLossRequest)
            from alpaca.trading.enums    import OrderClass, OrderSide, TimeInForce

            side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL

            use_bracket = stop_price is not None and target_price is not None

            if use_bracket:
                req = MarketOrderRequest(
                    symbol        = SYMBOL,
                    qty           = qty,
                    side          = side,
                    time_in_force = TimeInForce.DAY,
                    order_class   = OrderClass.BRACKET,
                    take_profit   = TakeProfitRequest(
                                        limit_price=round(target_price, 2)),
                    stop_loss     = StopLossRequest(
                                        stop_price=round(stop_price, 2)),
                )
                print(f"[BROKER] Bracket {action} x{qty} {SYMBOL} — "
                      f"SL:{stop_price:.2f}  TP:{target_price:.2f}")
            else:
                req = MarketOrderRequest(
                    symbol        = SYMBOL,
                    qty           = qty,
                    side          = side,
                    time_in_force = TimeInForce.DAY,
                )
                print(f"[BROKER] Market {action} x{qty} {SYMBOL}")

            order = self._trading_client.submit_order(req)
            print(f"[BROKER] Order submitted — id: {order.id}")

            fill_price = self._await_fill(str(order.id))
            return {"fillPrice": fill_price, "order_id": str(order.id)}

        except Exception as exc:
            print(f"[BROKER] place_market_order error: {exc}")
            return None

    def place_stop_order(self, action: str, stop_price: float,
                         qty: int = ORDER_QTY) -> Optional[dict]:
        """
        Submit a GTC stop-loss order.

        Args:
            action:     "Buy" (covers a short) or "Sell" (exits a long)
            stop_price: trigger price
        """
        if not self._connected:
            return None

        try:
            from alpaca.trading.requests import StopOrderRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce

            side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL

            req   = StopOrderRequest(
                symbol        = SYMBOL,
                qty           = qty,
                side          = side,
                stop_price    = round(stop_price, 2),
                time_in_force = TimeInForce.GTC,
            )
            order = self._trading_client.submit_order(req)
            print(f"[BROKER] Stop {action} @ {stop_price:.2f} — id: {order.id}")
            return {"order_id": str(order.id)}

        except Exception as exc:
            print(f"[BROKER] place_stop_order error: {exc}")
            return None

    def place_limit_order(self, action: str, limit_price: float,
                          qty: int = ORDER_QTY) -> Optional[dict]:
        """
        Submit a GTC take-profit limit order.

        Args:
            action:      "Buy" (covers a short) or "Sell" (exits a long)
            limit_price: target price
        """
        if not self._connected:
            return None

        try:
            from alpaca.trading.requests import LimitOrderRequest
            from alpaca.trading.enums    import OrderSide, TimeInForce

            side = OrderSide.BUY if action.lower() == "buy" else OrderSide.SELL

            req   = LimitOrderRequest(
                symbol        = SYMBOL,
                qty           = qty,
                side          = side,
                limit_price   = round(limit_price, 2),
                time_in_force = TimeInForce.GTC,
            )
            order = self._trading_client.submit_order(req)
            print(f"[BROKER] Limit {action} @ {limit_price:.2f} — id: {order.id}")
            return {"order_id": str(order.id)}

        except Exception as exc:
            print(f"[BROKER] place_limit_order error: {exc}")
            return None

    def cancel_all_orders(self):
        """Cancel every open order for this account."""
        if not self._connected:
            return

        try:
            statuses = self._trading_client.cancel_orders()
            print(f"[BROKER] Cancelled {len(statuses)} open order(s).")
        except Exception as exc:
            print(f"[BROKER] cancel_all_orders error: {exc}")

    def close_position(self):
        """Market-close the entire open position for SYMBOL."""
        if not self._connected:
            return

        try:
            self._trading_client.close_position(SYMBOL)
            print(f"[BROKER] Position closed for {SYMBOL}.")
        except Exception as exc:
            print(f"[BROKER] close_position error: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Account helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_account_equity(self) -> Optional[float]:
        """Return current account equity in dollars."""
        if not self._connected:
            return None
        try:
            return float(self._trading_client.get_account().equity)
        except Exception:
            return None

    def get_open_position(self) -> Optional[dict]:
        """
        Return the current open position for SYMBOL, or None if flat.

        Returns dict with keys:
            side        — "Long" or "Short"
            qty         — number of shares (int)
            entry_price — average fill price (float)
        """
        if not self._connected:
            return None
        try:
            pos = self._trading_client.get_open_position(SYMBOL)
            side = "Long" if pos.side.value == "long" else "Short"
            return {
                "side":        side,
                "qty":         int(float(pos.qty)),
                "entry_price": float(pos.avg_entry_price),
            }
        except Exception:
            return None   # 404 = no position; any other error = treat as flat

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _await_fill(self, order_id: str, max_wait: float = 5.0) -> Optional[float]:
        """
        Poll the order status until filled or max_wait seconds elapses.
        Returns the average fill price, or falls back to the current market price.
        """
        try:
            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline:
                order = self._trading_client.get_order_by_id(order_id)
                if order.filled_avg_price:
                    return float(order.filled_avg_price)
                if str(order.status) in ("filled", "partially_filled"):
                    break
                time.sleep(0.5)
        except Exception:
            pass

        return self.get_current_price()
