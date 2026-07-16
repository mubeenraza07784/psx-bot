from __future__ import annotations

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from .structures import swing_points


def _safe_atr(df: pd.DataFrame) -> float:
    if 'atr' in df.columns and len(df) and pd.notna(df['atr'].iloc[-1]):
        return float(df['atr'].iloc[-1])
    return max(float(df['close'].iloc[-1]) * 0.01, 0.01)


def _line(values: list[float], xs: list[int]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, float(values[-1]) if values else 0.0
    slope, intercept = np.polyfit(np.array(xs, dtype=float), np.array(values, dtype=float), 1)
    return float(slope), float(intercept)


def _width_at(slope_h: float, intercept_h: float, slope_l: float, intercept_l: float, x: float) -> float:
    return float((slope_h * x + intercept_h) - (slope_l * x + intercept_l))


def detect_continuation_patterns(df: pd.DataFrame, trend: str) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    if len(df) < 35:
        return patterns

    recent = df.tail(55).copy()
    atr = _safe_atr(df)
    close = float(recent['close'].iloc[-1])
    swings = swing_points(recent, window=2)
    highs = swings['highs']
    lows = swings['lows']

    def add(name: str, bias: str, score: int, detail: str, projection: float | None = None, level: float | None = None):
        if name not in [p['name'] for p in patterns]:
            patterns.append({'name': name, 'bias': bias, 'score': score, 'detail': detail, 'projection': projection, 'level': level})

    # Flag and pennant logic: strong impulse + compact consolidation.
    first = recent.iloc[: max(12, len(recent) // 2)]
    second = recent.iloc[max(12, len(recent) // 2):]
    first_move = float(first['close'].iloc[-1] - first['close'].iloc[0])
    second_move = float(second['close'].iloc[-1] - second['close'].iloc[0])
    second_range = float(second['high'].max() - second['low'].min())
    impulse_ok = abs(first_move) > max(atr * 3, close * 0.035)
    compact = second_range < abs(first_move) * 0.75

    if impulse_ok and compact:
        if first_move > 0 and second_move <= atr * 1.5:
            add('Bullish Flag', 'Bullish', 78, 'Strong bullish impulse followed by controlled sideways/down consolidation.', close + abs(first_move))
        if first_move < 0 and second_move >= -atr * 1.5:
            add('Bearish Flag', 'Bearish', 78, 'Strong bearish impulse followed by controlled sideways/up consolidation.', close - abs(first_move))

    # Swing-line pattern logic: triangles, channels, wedges, pennants.
    if len(highs) >= 3 and len(lows) >= 3:
        h = highs[-4:]
        l = lows[-4:]
        xh = [int(v['pos']) for v in h]
        yh = [float(v['price']) for v in h]
        xl = [int(v['pos']) for v in l]
        yl = [float(v['price']) for v in l]
        sh, ih = _line(yh, xh)
        sl, il = _line(yl, xl)
        x0 = float(min(xh + xl))
        x1 = float(max(xh + xl))
        width_start = _width_at(sh, ih, sl, il, x0)
        width_end = _width_at(sh, ih, sl, il, x1)
        converging = width_start > 0 and width_end > 0 and width_end < width_start * 0.78
        nearly_flat_highs = abs(sh) <= max(atr * 0.05, close * 0.0008)
        nearly_flat_lows = abs(sl) <= max(atr * 0.05, close * 0.0008)
        roughly_parallel = abs(sh - sl) <= max(atr * 0.07, close * 0.001)

        if nearly_flat_highs and sl > 0:
            add('Ascending Triangle', 'Bullish', 73, 'Flat resistance with rising lows; bullish breakout watch.', level=max(yh))
        if nearly_flat_lows and sh < 0:
            add('Descending Triangle', 'Bearish', 73, 'Flat support with falling highs; bearish breakdown watch.', level=min(yl))
        if converging and sh < 0 and sl > 0:
            add('Symmetrical Triangle', trend if trend in {'Bullish', 'Bearish'} else 'Neutral', 68, 'Converging highs and lows; wait for breakout direction.')
            if impulse_ok:
                add('Pennant', 'Bullish' if first_move > 0 else 'Bearish', 70, 'Impulse move followed by tight converging consolidation.')
        if sh > 0 and sl > 0 and roughly_parallel:
            add('Ascending Channel', 'Bullish', 60, 'Parallel upward swing path; trend remains valid above lower channel.')
        if sh < 0 and sl < 0 and roughly_parallel:
            add('Descending Channel', 'Bearish', 60, 'Parallel downward swing path; trend remains weak below upper channel.')
        if converging and sh > 0 and sl > 0 and sl > sh:
            add('Rising Wedge', 'Bearish', 69, 'Rising highs and faster-rising lows show compression; breakdown risk.')
        if converging and sh < 0 and sl < 0 and sh < sl:
            add('Falling Wedge', 'Bullish', 69, 'Falling lows and faster-falling highs show compression; breakout watch.')

    # Rectangles / range consolidation.
    if len(highs) >= 2 and len(lows) >= 2:
        high_band = max(h['price'] for h in highs[-4:]) - min(h['price'] for h in highs[-4:])
        low_band = max(l['price'] for l in lows[-4:]) - min(l['price'] for l in lows[-4:])
        recent_range = float(recent['high'].tail(35).max() - recent['low'].tail(35).min())
        if high_band <= atr * 1.4 and low_band <= atr * 1.4 and recent_range <= close * 0.12:
            add('Rectangle / Range Consolidation', trend if trend in {'Bullish', 'Bearish'} else 'Neutral', 63, 'Repeated support and resistance zones; wait for range breakout.')

    full = recent['close'].values
    mid = len(full) // 2
    left, center, right = full[0], full[mid], full[-1]
    if center < min(left, right) and abs(left - right) <= max(atr * 5, abs(right) * 0.05):
        add('Cup and Handle (approx.)', 'Bullish', 62, 'Rounded recovery structure; verify handle/breakout manually.')
    if center > max(left, right) and abs(left - right) <= max(atr * 5, abs(right) * 0.05):
        add('Inverted Cup and Handle (approx.)', 'Bearish', 62, 'Rounded top structure; verify handle/breakdown manually.')

    return patterns
