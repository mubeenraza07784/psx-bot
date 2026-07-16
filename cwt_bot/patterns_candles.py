from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd


def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    if len(df) < 4:
        return patterns
    c = df.iloc[-1]
    p = df.iloc[-2]
    p2 = df.iloc[-3]
    eps = 1e-9
    body = max(float(c['body']), eps)
    rng = max(float(c['range']), eps)
    bullish = c['close'] > c['open']
    bearish = c['close'] < c['open']

    def add(name: str, bias: str, score: int, detail: str):
        patterns.append({'name': name, 'bias': bias, 'score': score, 'detail': detail})

    if c['lower_wick'] >= body * 2 and c['upper_wick'] <= body * 0.75:
        add('Hammer' if bullish else 'Hanging Man', 'Bullish' if bullish else 'Bearish', 55, 'Long lower wick')
    if c['upper_wick'] >= body * 2 and c['lower_wick'] <= body * 0.75:
        add('Inverted Hammer' if bullish else 'Shooting Star', 'Bullish' if bullish else 'Bearish', 55, 'Long upper wick')
    if bullish and p['close'] < p['open'] and c['open'] <= p['close'] and c['close'] >= p['open']:
        add('Bullish Engulfing', 'Bullish', 70, 'Bull body engulfs prior bear body')
    if bearish and p['close'] > p['open'] and c['open'] >= p['close'] and c['close'] <= p['open']:
        add('Bearish Engulfing', 'Bearish', 70, 'Bear body engulfs prior bull body')
    prior_mid = (p['open'] + p['close']) / 2
    if bullish and p['close'] < p['open'] and c['open'] < p['low'] and c['close'] > prior_mid and c['close'] < p['open']:
        add('Piercing Pattern', 'Bullish', 65, 'Closes above prior bearish midpoint')
    if bearish and p['close'] > p['open'] and c['open'] > p['high'] and c['close'] < prior_mid and c['close'] > p['open']:
        add('Dark Cloud Cover', 'Bearish', 65, 'Closes below prior bullish midpoint')
    if bullish and p['close'] < p['open'] and c['high'] <= p['high'] and c['low'] >= p['low']:
        add('Bullish Harami / Inside Bar', 'Bullish', 50, 'Small bullish candle inside prior range')
    if bearish and p['close'] > p['open'] and c['high'] <= p['high'] and c['low'] >= p['low']:
        add('Bearish Harami / Inside Bar', 'Bearish', 50, 'Small bearish candle inside prior range')
    tol = max(float(df['atr'].iloc[-1]) * 0.15, abs(float(c['close'])) * 0.001)
    if abs(float(c['low']) - float(p['low'])) <= tol and bullish:
        add('Tweezers Bottom', 'Bullish', 55, 'Two similar lows')
    if abs(float(c['high']) - float(p['high'])) <= tol and bearish:
        add('Tweezers Top', 'Bearish', 55, 'Two similar highs')
    if p2['close'] < p2['open'] and (p['body'] / max(p['range'], eps) <= 0.35) and bullish and c['close'] > ((p2['open'] + p2['close']) / 2):
        add('Morning Doji Star' if (p['body'] / max(p['range'], eps) <= 0.10) else 'Morning Star', 'Bullish', 80, 'Three-candle bullish reversal')
    if p2['close'] > p2['open'] and (p['body'] / max(p['range'], eps) <= 0.35) and bearish and c['close'] < ((p2['open'] + p2['close']) / 2):
        add('Evening Doji Star' if (p['body'] / max(p['range'], eps) <= 0.10) else 'Evening Star', 'Bearish', 80, 'Three-candle bearish reversal')
    trio = df.iloc[-3:]
    if all(trio['close'] > trio['open']) and trio['close'].is_monotonic_increasing:
        add('Three White Soldiers', 'Bullish', 75, 'Three advancing bullish candles')
    if all(trio['close'] < trio['open']) and trio['close'].is_monotonic_decreasing:
        add('Three Black Crows', 'Bearish', 75, 'Three declining bearish candles')
    return patterns
