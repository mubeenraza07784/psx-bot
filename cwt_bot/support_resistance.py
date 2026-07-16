from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from .structures import swing_points


def _cluster(levels: List[float], tolerance: float) -> List[float]:
    clusters: List[List[float]] = []
    for level in sorted(levels):
        placed = False
        for cl in clusters:
            if abs(np.mean(cl) - level) <= tolerance:
                cl.append(level)
                placed = True
                break
        if not placed:
            clusters.append([level])
    return [float(np.mean(c)) for c in clusters]


def support_resistance(df: pd.DataFrame) -> Dict[str, Any]:
    swings = swing_points(df, window=3)
    prices = [s['price'] for s in swings['highs'] + swings['lows']]
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else float(df['close'].iloc[-1]) * 0.01
    tolerance = max(atr * 0.35, float(df['close'].iloc[-1]) * 0.002)
    clustered = _cluster(prices, tolerance)
    close = float(df['close'].iloc[-1])
    supports = sorted([v for v in clustered if v < close])
    resistances = sorted([v for v in clustered if v > close])
    return {'supports': supports, 'resistances': resistances, 'nearest_support': supports[-1] if supports else None, 'nearest_resistance': resistances[0] if resistances else None}


def detect_sbr_rbs(df: pd.DataFrame, sr: Dict[str, Any]) -> Dict[str, Any]:
    close = float(df['close'].iloc[-1])
    prior_close = float(df['close'].iloc[-2]) if len(df) >= 2 else close
    tol = float(df['atr'].iloc[-1]) * 0.25 if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else close * 0.002
    result = {'label': 'None', 'bias': 'Neutral', 'score': 0, 'level': None}
    for support in sr['supports'][-5:]:
        if prior_close < support + tol and close < support + tol:
            result = {'label': 'Support Becomes Resistance (SBR)', 'bias': 'Bearish', 'score': 45, 'level': support}
    for resistance in sr['resistances'][:5]:
        if prior_close > resistance - tol and close > resistance - tol:
            result = {'label': 'Resistance Becomes Support (RBS)', 'bias': 'Bullish', 'score': 45, 'level': resistance}
    return result
