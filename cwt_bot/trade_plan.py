from __future__ import annotations

from typing import Dict, Any
import pandas as pd


def build_trade_plan(df: pd.DataFrame, bias: str, setup_type: str, scenario_label: str, support_resistance: Dict[str, Any], fvg: Dict[str, Any], pattern_projection: float | None = None) -> Dict[str, Any]:
    close = float(df['close'].iloc[-1])
    atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]) else close * 0.01
    supports = support_resistance.get('supports', [])
    resistances = support_resistance.get('resistances', [])
    nearest_support = supports[-1] if supports else close - atr * 2
    nearest_resistance = resistances[0] if resistances else close + atr * 2
    order_type = 'Wait'
    entry = sl = None
    rationale = 'No aligned trade plan.'

    if bias == 'Bullish':
        if 'Scenario 1' in scenario_label or 'Reversal' in setup_type or 'Continuation' in setup_type:
            order_type = 'Buy Stop'
            entry = max(close + atr * 0.15, nearest_resistance)
            sl = min(nearest_support, close - atr * 1.5)
            rationale = 'Bullish bias: confirmation above local trigger / resistance.'
        else:
            bull_zone = fvg.get('nearest_bullish')
            order_type = 'Buy Limit'
            entry = float(bull_zone['high']) if bull_zone else float(nearest_support)
            sl = min(float(bull_zone['low']) if bull_zone else entry - atr, nearest_support - atr * 0.3)
            rationale = 'Bullish higher timeframe with pullback/sideways execution: limit order near support/FVG.'
    elif bias == 'Bearish':
        if 'Scenario 1' in scenario_label or 'Reversal' in setup_type or 'Continuation' in setup_type:
            order_type = 'Sell Stop'
            entry = min(close - atr * 0.15, nearest_support)
            sl = max(nearest_resistance, close + atr * 1.5)
            rationale = 'Bearish bias: confirmation below local trigger / support.'
        else:
            bear_zone = fvg.get('nearest_bearish')
            order_type = 'Sell Limit'
            entry = float(bear_zone['low']) if bear_zone else float(nearest_resistance)
            sl = max(float(bear_zone['high']) if bear_zone else entry + atr, nearest_resistance + atr * 0.3)
            rationale = 'Bearish higher timeframe with pullback/sideways execution: limit order near resistance/FVG.'
    tp = rr = invalidation = None
    if entry is not None and sl is not None:
        risk = abs(entry - sl)
        if bias == 'Bullish':
            tp = max(entry + 3 * risk, pattern_projection or float('-inf'))
            invalidation = 'Plan invalid below stop-loss / broken bullish structure.'
        elif bias == 'Bearish':
            tp = min(entry - 3 * risk, pattern_projection or float('inf'))
            invalidation = 'Plan invalid above stop-loss / broken bearish structure.'
        rr = '1:3'
    return {'order_type': order_type, 'entry': None if entry is None else round(float(entry), 8), 'stop_loss': None if sl is None else round(float(sl), 8), 'take_profit': None if tp is None else round(float(tp), 8), 'rr': rr, 'invalidation': invalidation, 'rationale': rationale}
