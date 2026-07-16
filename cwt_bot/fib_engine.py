from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .structures import swing_points


FIB_RATIOS = {
    "23.6%": 0.236,
    "38.2%": 0.382,
    "50.0%": 0.500,
    "61.8%": 0.618,
    "78.6%": 0.786,
}


def fibonacci_retracement_confluence(df: pd.DataFrame, trend: str, tolerance_atr: float = 0.45) -> Dict[str, Any]:
    """
    Build a recent swing Fibonacci retracement map and determine whether
    current price is near a key retracement level.
    """
    if df is None or df.empty or len(df) < 30:
        return {"status": "INSUFFICIENT_DATA", "levels": [], "nearest": None, "confluence": False}

    swings = swing_points(df, window=3)
    highs = swings.get("highs", [])[-6:]
    lows = swings.get("lows", [])[-6:]
    close = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else abs(close) * 0.02
    tolerance = max(abs(atr) * tolerance_atr, abs(close) * 0.005)

    if trend == "Bullish" and highs and lows:
        swing_high = highs[-1]
        prior_lows = [l for l in lows if l["pos"] < swing_high["pos"]]
        if not prior_lows:
            return {"status": "NO_SWING", "levels": [], "nearest": None, "confluence": False}
        swing_low = prior_lows[-1]
        high_price = float(swing_high["price"])
        low_price = float(swing_low["price"])
        direction = "Bullish Retracement"
        levels = []
        for label, ratio in FIB_RATIOS.items():
            level = high_price - (high_price - low_price) * ratio
            levels.append({"Fib": label, "Price": round(level, 6), "Distance": round(abs(close - level), 6)})
    elif trend == "Bearish" and highs and lows:
        swing_low = lows[-1]
        prior_highs = [h for h in highs if h["pos"] < swing_low["pos"]]
        if not prior_highs:
            return {"status": "NO_SWING", "levels": [], "nearest": None, "confluence": False}
        swing_high = prior_highs[-1]
        high_price = float(swing_high["price"])
        low_price = float(swing_low["price"])
        direction = "Bearish Retracement"
        levels = []
        for label, ratio in FIB_RATIOS.items():
            level = low_price + (high_price - low_price) * ratio
            levels.append({"Fib": label, "Price": round(level, 6), "Distance": round(abs(close - level), 6)})
    else:
        return {"status": "NO_TREND", "levels": [], "nearest": None, "confluence": False}

    nearest = min(levels, key=lambda x: x["Distance"]) if levels else None
    confluence = bool(nearest and nearest["Distance"] <= tolerance)
    preferred = nearest["Fib"] in {"38.2%", "50.0%", "61.8%"} if nearest else False
    return {
        "status": "OK",
        "direction": direction,
        "swing_high": round(high_price, 6),
        "swing_low": round(low_price, 6),
        "levels": levels,
        "nearest": nearest,
        "tolerance": round(tolerance, 6),
        "confluence": confluence,
        "preferred_retracement": preferred,
        "message": (
            f"Price is near {nearest['Fib']} Fibonacci retracement."
            if confluence and nearest
            else "Price is not near a key Fibonacci retracement."
        ),
    }
