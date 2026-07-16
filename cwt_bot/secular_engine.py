from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np


def classify_secular_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Long-term structural bias suitable for Position / Long-Term mode.
    Uses multi-period price structure rather than fragile single-candle signals.
    """
    if df is None or df.empty or len(df) < 52:
        return {
            "status": "INSUFFICIENT_DATA",
            "secular_bias": "Unknown",
            "score": None,
            "message": "Need at least ~52 candles of weekly/monthly style data for secular classification.",
        }

    close = df["close"].astype(float)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    recent = close.iloc[-1]
    peak = float(close.tail(min(len(close), 104)).max())
    trough = float(close.tail(min(len(close), 104)).min())
    drawdown_pct = ((recent - peak) / peak) * 100 if peak else None
    recovery_pct = ((recent - trough) / trough) * 100 if trough else None

    six_period_return = close.pct_change(26).iloc[-1] * 100 if len(close) > 26 else None
    one_year_return = close.pct_change(52).iloc[-1] * 100 if len(close) > 52 else None

    score = 50.0
    notes = []

    if pd.notna(ma20.iloc[-1]) and pd.notna(ma50.iloc[-1]):
        if recent > ma20.iloc[-1] > ma50.iloc[-1]:
            score += 20
            notes.append("Price > MA20 > MA50.")
        elif recent < ma20.iloc[-1] < ma50.iloc[-1]:
            score -= 20
            notes.append("Price < MA20 < MA50.")

    if six_period_return is not None and pd.notna(six_period_return):
        if six_period_return > 10:
            score += 12
            notes.append("26-period return is strongly positive.")
        elif six_period_return < -10:
            score -= 12
            notes.append("26-period return is strongly negative.")

    if one_year_return is not None and pd.notna(one_year_return):
        if one_year_return > 20:
            score += 12
            notes.append("52-period return is strongly positive.")
        elif one_year_return < -20:
            score -= 12
            notes.append("52-period return is strongly negative.")

    if drawdown_pct is not None:
        if drawdown_pct > -15:
            score += 6
            notes.append("Drawdown from recent peak is contained.")
        elif drawdown_pct < -35:
            score -= 8
            notes.append("Large drawdown from recent peak.")

    score = max(0.0, min(100.0, score))
    if score >= 70:
        bias = "Secular Bull / Long Bias"
    elif score <= 30:
        bias = "Secular Bear / Avoid or Defensive"
    else:
        bias = "Neutral / Transitional"

    return {
        "status": "OK",
        "secular_bias": bias,
        "score": round(score, 2),
        "drawdown_pct": None if drawdown_pct is None else round(drawdown_pct, 2),
        "recovery_from_trough_pct": None if recovery_pct is None else round(recovery_pct, 2),
        "period_26_return_pct": None if six_period_return is None or pd.isna(six_period_return) else round(float(six_period_return), 2),
        "period_52_return_pct": None if one_year_return is None or pd.isna(one_year_return) else round(float(one_year_return), 2),
        "notes": " | ".join(notes),
        "message": f"Secular assessment: {bias}.",
    }
