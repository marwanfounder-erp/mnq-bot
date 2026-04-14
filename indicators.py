# =============================================================================
# indicators.py — Technical indicator calculations using pandas
# All functions accept a pandas DataFrame with at minimum a 'close' column
# and return a pandas Series (or tuple of Series) aligned to the same index.
# =============================================================================

import pandas as pd
import numpy as np
from config import EMA_FAST, EMA_SLOW, RSI_PERIOD


# ─── Exponential Moving Average ───────────────────────────────────────────────

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA) for a given period.

    Uses pandas' exponentially weighted moving average with adjust=False,
    which matches the standard formula used by most trading platforms:
        EMA_t = close_t * multiplier + EMA_(t-1) * (1 - multiplier)
        where multiplier = 2 / (period + 1)

    Args:
        series: pandas Series of price values (typically 'close')
        period: lookback period (e.g. 9, 21)

    Returns:
        pandas Series of EMA values, same length as input
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EMA_FAST and EMA_SLOW columns to the DataFrame in-place.

    Args:
        df: DataFrame with a 'close' column

    Returns:
        The same DataFrame with 'ema_fast' and 'ema_slow' columns added
    """
    df = df.copy()
    df["ema_fast"] = calculate_ema(df["close"], EMA_FAST)   # EMA 9
    df["ema_slow"] = calculate_ema(df["close"], EMA_SLOW)   # EMA 21
    return df


# ─── Relative Strength Index ──────────────────────────────────────────────────

def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing method (standard RSI).

    Formula:
        delta  = close.diff()
        gain   = delta where delta > 0, else 0
        loss   = abs(delta) where delta < 0, else 0
        avg_gain = Wilder's smoothed average of gain over `period` bars
        avg_loss = Wilder's smoothed average of loss over `period` bars
        RS     = avg_gain / avg_loss
        RSI    = 100 - (100 / (1 + RS))

    Wilder's smoothing = EWM with alpha = 1/period  (com = period - 1)

    Args:
        series: pandas Series of closing prices
        period: RSI lookback period (default 14)

    Returns:
        pandas Series of RSI values in range [0, 100]
    """
    # Price differences bar-over-bar
    delta = series.diff()

    # Separate gains (positive changes) and losses (absolute negative changes)
    gain = delta.clip(lower=0)          # keep positive, zero out negatives
    loss = (-delta).clip(lower=0)       # make losses positive, zero out gains

    # Wilder's smoothed averages — equivalent to EMA with alpha = 1/period
    # adjust=False matches the recursive Wilder formula exactly
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # Avoid division by zero; where avg_loss == 0, RSI = 100 (no losses)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # all-gain periods → RSI = 100

    return rsi


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all required indicators and attach them to the DataFrame.

    Columns added:
        ema_fast  — EMA(9) of close
        ema_slow  — EMA(21) of close
        rsi       — RSI(14) of close

    Args:
        df: DataFrame with at least a 'close' column

    Returns:
        New DataFrame with indicator columns appended
    """
    if df.empty or "close" not in df.columns:
        raise ValueError("DataFrame must have a 'close' column")

    if len(df) < EMA_SLOW:
        raise ValueError(
            f"Need at least {EMA_SLOW} bars for EMA{EMA_SLOW}; "
            f"got {len(df)}"
        )

    df = df.copy()

    # Compute EMAs
    df["ema_fast"] = calculate_ema(df["close"], EMA_FAST)
    df["ema_slow"] = calculate_ema(df["close"], EMA_SLOW)

    # Compute RSI
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)

    return df


# ─── Signal Helpers ───────────────────────────────────────────────────────────

def get_latest_values(df: pd.DataFrame) -> dict:
    """
    Extract the most recent bar's indicator values as a plain dict.

    Returns a dict with keys:
        close, ema_fast, ema_slow, rsi,
        trend_bullish (bool), trend_bearish (bool)
    """
    if df.empty:
        return {}

    last = df.iloc[-1]

    return {
        "close":         last["close"],
        "ema_fast":      last["ema_fast"],
        "ema_slow":      last["ema_slow"],
        "rsi":           last["rsi"],
        # Trend flags: price above/below slow EMA
        "trend_bullish": last["close"] > last["ema_slow"],
        "trend_bearish": last["close"] < last["ema_slow"],
    }


# ─── Quick self-test (run: python indicators.py) ──────────────────────────────
if __name__ == "__main__":
    import math

    # Generate synthetic sine-wave price data for testing
    n = 100
    prices = [15000 + 200 * math.sin(i / 5) for i in range(n)]
    test_df = pd.DataFrame({"close": prices})

    result = calculate_all_indicators(test_df)

    print("Last 5 rows with indicators:")
    print(result[["close", "ema_fast", "ema_slow", "rsi"]].tail(5).to_string())

    vals = get_latest_values(result)
    print(f"\nLatest values: {vals}")
    print("indicators.py self-test passed.")
