from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from .structures import swing_points


def _tol(df: pd.DataFrame) -> float:
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else float(df['close'].iloc[-1]) * 0.01
    return max(atr * 0.9, float(df['close'].iloc[-1]) * 0.012)


def _line(values: list[float], xs: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, float(values[-1]) if values else 0.0
    slope, intercept = np.polyfit(np.array(xs, dtype=float), np.array(values, dtype=float), 1)
    return float(slope), float(intercept)


def _width_at(slope_h: float, intercept_h: float, slope_l: float, intercept_l: float, x: float) -> float:
    return float((slope_h * x + intercept_h) - (slope_l * x + intercept_l))


def detect_reversal_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    swings = swing_points(df, window=3)
    highs = swings['highs'][-8:]
    lows = swings['lows'][-8:]
    tol = _tol(df)
    patterns: List[Dict[str, Any]] = []

    def add(name: str, bias: str, score: int, detail: str, level: Optional[float] = None):
        if name not in [p['name'] for p in patterns]:
            patterns.append({'name': name, 'bias': bias, 'score': score, 'detail': detail, 'level': level})

    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        between_lows = [l for l in lows if a['pos'] < l['pos'] < b['pos']]
        if abs(a['price'] - b['price']) <= tol and between_lows:
            neckline = min(l['price'] for l in between_lows)
            add('Double Top', 'Bearish', 78, 'Two comparable highs with neckline support.', neckline)
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        between_highs = [h for h in highs if a['pos'] < h['pos'] < b['pos']]
        if abs(a['price'] - b['price']) <= tol and between_highs:
            neckline = max(h['price'] for h in between_highs)
            add('Double Bottom', 'Bullish', 78, 'Two comparable lows with neckline resistance.', neckline)

    if len(highs) >= 3:
        last3 = highs[-3:]
        if max(h['price'] for h in last3) - min(h['price'] for h in last3) <= tol * 1.3:
            neck_lows = [l for l in lows if last3[0]['pos'] < l['pos'] < last3[-1]['pos']]
            neckline = min([l['price'] for l in neck_lows], default=None)
            add('Triple Top', 'Bearish', 80, 'Three comparable highs; support break confirms reversal.', neckline)
        left, head, right = highs[-3], highs[-2], highs[-1]
        if head['price'] > left['price'] and head['price'] > right['price'] and abs(left['price'] - right['price']) <= tol * 1.8:
            neck_lows = [l for l in lows if left['pos'] < l['pos'] < right['pos']]
            neckline = min([l['price'] for l in neck_lows], default=None)
            add('Head and Shoulders', 'Bearish', 84, 'Three-peak bearish reversal structure.', neckline)

    if len(lows) >= 3:
        last3 = lows[-3:]
        if max(l['price'] for l in last3) - min(l['price'] for l in last3) <= tol * 1.3:
            neck_highs = [h for h in highs if last3[0]['pos'] < h['pos'] < last3[-1]['pos']]
            neckline = max([h['price'] for h in neck_highs], default=None)
            add('Triple Bottom', 'Bullish', 80, 'Three comparable lows; resistance break confirms reversal.', neckline)
        left, head, right = lows[-3], lows[-2], lows[-1]
        if head['price'] < left['price'] and head['price'] < right['price'] and abs(left['price'] - right['price']) <= tol * 1.8:
            neck_highs = [h for h in highs if left['pos'] < h['pos'] < right['pos']]
            neckline = max([h['price'] for h in neck_highs], default=None)
            add('Inverse Head and Shoulders', 'Bullish', 84, 'Three-trough bullish reversal structure.', neckline)

    recent_highs = highs[-4:]
    recent_lows = lows[-4:]
    if len(recent_highs) >= 3 and len(recent_lows) >= 3:
        xh = [int(h['pos']) for h in recent_highs]
        yh = [float(h['price']) for h in recent_highs]
        xl = [int(l['pos']) for l in recent_lows]
        yl = [float(l['price']) for l in recent_lows]
        sh, ih = _line(yh, xh)
        sl, il = _line(yl, xl)
        x0 = float(min(xh + xl))
        x1 = float(max(xh + xl))
        width_start = _width_at(sh, ih, sl, il, x0)
        width_end = _width_at(sh, ih, sl, il, x1)
        converging = width_start > 0 and width_end > 0 and width_end < width_start * 0.82
        broadening = width_start > 0 and width_end > width_start * 1.25
        if converging and sh > 0 and sl > 0 and sl > sh:
            add('Rising Wedge', 'Bearish', 72, 'Rising converging structure; bearish reversal/breakdown risk.')
        if converging and sh < 0 and sl < 0 and sh < sl:
            add('Falling Wedge', 'Bullish', 72, 'Falling converging structure; bullish reversal/breakout watch.')
        if broadening and sh > 0 and sl < 0:
            add('Broadening Wedge', 'Neutral', 60, 'Expanding volatility structure; wait for confirmed breakout direction.')

    return patterns
