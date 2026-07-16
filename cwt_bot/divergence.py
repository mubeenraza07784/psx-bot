from __future__ import annotations

from typing import Dict, Any
import pandas as pd
from .structures import swing_points


def detect_rsi_divergence(df: pd.DataFrame) -> Dict[str, Any]:
    swings = swing_points(df, window=3)
    highs = swings['highs'][-3:]
    lows = swings['lows'][-3:]
    result = {'label': 'None', 'bias': 'Neutral', 'score': 0, 'detail': ''}
    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        r1 = float(df['rsi'].iloc[h1['pos']])
        r2 = float(df['rsi'].iloc[h2['pos']])
        if h2['price'] > h1['price'] and r2 < r1 and max(r1, r2) >= 70:
            result = {'label': 'Bearish RSI Divergence', 'bias': 'Bearish', 'score': 80, 'detail': 'Price HH with RSI LH above/near 70.'}
    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        r1 = float(df['rsi'].iloc[l1['pos']])
        r2 = float(df['rsi'].iloc[l2['pos']])
        if l2['price'] < l1['price'] and r2 > r1 and min(r1, r2) <= 30:
            result = {'label': 'Bullish RSI Divergence', 'bias': 'Bullish', 'score': 80, 'detail': 'Price LL with RSI HL below/near 30.'}
    return result
