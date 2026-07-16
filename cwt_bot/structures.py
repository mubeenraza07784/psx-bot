from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def swing_points(df: pd.DataFrame, window: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    highs: List[Dict[str, Any]] = []
    lows: List[Dict[str, Any]] = []
    if len(df) < window * 2 + 1:
        return {'highs': highs, 'lows': lows}
    h = df['high'].values
    l = df['low'].values
    idx = list(df.index)
    for i in range(window, len(df) - window):
        if h[i] == np.nanmax(h[i - window : i + window + 1]):
            highs.append({'pos': i, 'time': idx[i], 'price': float(h[i])})
        if l[i] == np.nanmin(l[i - window : i + window + 1]):
            lows.append({'pos': i, 'time': idx[i], 'price': float(l[i])})
    return {'highs': highs, 'lows': lows}


def classify_trend(df: pd.DataFrame, swings: Dict[str, List[Dict[str, Any]]] | None = None) -> Dict[str, Any]:
    swings = swings or swing_points(df)
    highs = swings['highs'][-4:]
    lows = swings['lows'][-4:]
    trend = 'Sideways'
    structure = []
    if len(highs) >= 2 and len(lows) >= 2:
        high_up = highs[-1]['price'] > highs[-2]['price']
        high_down = highs[-1]['price'] < highs[-2]['price']
        low_up = lows[-1]['price'] > lows[-2]['price']
        low_down = lows[-1]['price'] < lows[-2]['price']
        if high_up and low_up:
            trend = 'Bullish'
            structure = ['HH', 'HL']
        elif high_down and low_down:
            trend = 'Bearish'
            structure = ['LH', 'LL']
        else:
            trend = 'Sideways'
            structure = ['Mixed']
    return {'trend': trend, 'structure': structure, 'phase': market_phase(df, trend), 'swings': swings}


def market_phase(df: pd.DataFrame, trend: str) -> str:
    if len(df) < 30:
        return 'Unclear'
    recent = df.tail(30)
    prior = df.iloc[-60:-30] if len(df) >= 60 else df.iloc[:0]
    prior_slope = 0.0
    if not prior.empty:
        prior_slope = np.polyfit(np.arange(len(prior)), prior['close'].values, 1)[0]
    current_slope = np.polyfit(np.arange(len(recent)), recent['close'].values, 1)[0]
    if trend == 'Bullish':
        return 'Advancing'
    if trend == 'Bearish':
        return 'Declining'
    if abs(current_slope) < max(recent['close'].mean() * 0.001, 1e-9):
        if prior_slope < 0:
            return 'Accumulation'
        if prior_slope > 0:
            return 'Distribution'
    return 'Sideways / Pause'
