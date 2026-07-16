from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def detect_fvgs(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    bullish: List[Dict[str, Any]] = []
    bearish: List[Dict[str, Any]] = []
    work = df.tail(lookback)
    start_pos = len(df) - len(work)
    rows = list(work.itertuples())
    for j in range(2, len(rows)):
        a = rows[j - 2]
        c = rows[j]
        absolute_pos = start_pos + j
        if c.low > a.high:
            bullish.append({'pos': absolute_pos, 'low': float(a.high), 'high': float(c.low)})
        if c.high < a.low:
            bearish.append({'pos': absolute_pos, 'low': float(c.high), 'high': float(a.low)})
    close = float(df['close'].iloc[-1])
    bull_candidates = [z for z in bullish if z['low'] <= close]
    bear_candidates = [z for z in bearish if z['high'] >= close]
    nearest_bull = min(bull_candidates, key=lambda z: abs(close - z['high']), default=None)
    nearest_bear = min(bear_candidates, key=lambda z: abs(close - z['low']), default=None)
    return {
        'bullish': bullish,
        'bearish': bearish,
        'nearest_bullish_zone': None if nearest_bull is None else f"{nearest_bull['low']:.5f}–{nearest_bull['high']:.5f}",
        'nearest_bearish_zone': None if nearest_bear is None else f"{nearest_bear['low']:.5f}–{nearest_bear['high']:.5f}",
        'nearest_bullish': nearest_bull,
        'nearest_bearish': nearest_bear,
    }
