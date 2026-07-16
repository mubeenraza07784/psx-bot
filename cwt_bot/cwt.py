from __future__ import annotations

from typing import Dict, Any
import pandas as pd


def alligator_state(df: pd.DataFrame) -> Dict[str, Any]:
    clean = df.dropna(subset=['jaw', 'teeth', 'lips'])
    if clean.empty:
        return {'state': 'Insufficient Data', 'direction': 'Neutral', 'mouth_open': False}
    row = clean.iloc[-1]
    gap_1 = abs(row['lips'] - row['teeth'])
    gap_2 = abs(row['teeth'] - row['jaw'])
    price = max(abs(float(row['close'])), 1e-9)
    mouth_open = (gap_1 / price > 0.0005) and (gap_2 / price > 0.0005)
    if row['lips'] > row['teeth'] > row['jaw']:
        direction = 'Bullish'
    elif row['lips'] < row['teeth'] < row['jaw']:
        direction = 'Bearish'
    else:
        direction = 'Neutral'
    state = 'Sleeping / Closed' if not mouth_open else f'Open {direction}'
    return {'state': state, 'direction': direction, 'mouth_open': mouth_open}


def cwt_bias(df: pd.DataFrame) -> Dict[str, Any]:
    state = alligator_state(df)
    clean = df.dropna(subset=['jaw', 'teeth', 'lips'])
    if clean.empty:
        return {**state, 'setup': 'Unavailable'}
    row = clean.iloc[-1]
    above_lips = row['ha_close'] > row['lips']
    below_lips = row['ha_close'] < row['lips']
    close_above_jaw = row['close'] > row['jaw']
    close_below_jaw = row['close'] < row['jaw']
    setup = 'Wait'
    if state['mouth_open'] and state['direction'] == 'Bullish' and above_lips:
        setup = 'CWT Trend Buy'
    elif state['mouth_open'] and state['direction'] == 'Bearish' and below_lips:
        setup = 'CWT Trend Sell'
    elif state['mouth_open'] and state['direction'] == 'Bearish' and close_above_jaw:
        setup = 'CWT Reversal Buy'
    elif state['mouth_open'] and state['direction'] == 'Bullish' and close_below_jaw:
        setup = 'CWT Reversal Sell'
    elif not state['mouth_open']:
        setup = 'No Trade / Sleeping Alligator'
    return {**state, 'setup': setup}
