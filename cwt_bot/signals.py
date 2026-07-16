from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
from .indicators import add_indicators
from .structures import classify_trend
from .cwt import cwt_bias
from .patterns_candles import detect_candlestick_patterns
from .patterns_reversal import detect_reversal_patterns
from .patterns_continuation import detect_continuation_patterns
from .divergence import detect_rsi_divergence
from .fvg import detect_fvgs
from .support_resistance import support_resistance, detect_sbr_rbs
from .trade_plan import build_trade_plan


def classify_mtf_trade_scenario(higher_trend: str, lower_trend: str) -> Dict[str, Any]:
    """
    Multi-Timeframe Trade Scenarios from Week 5/6:
    - Scenario 1: HTF and execution TF align in the same direction.
    - Scenario 2: HTF trend exists, execution TF is pulling back in the opposite direction.
    - Scenario 3: HTF trend exists, execution TF is sideways.
    """
    if higher_trend == 'Bullish':
        if lower_trend == 'Bullish':
            return {
                'label': 'Scenario 1 — MTF Buyers: Bullish HTF / Bullish Execution TF',
                'number': 'Scenario 1',
                'side': 'Buyers',
                'system': 'MTF Trade Scenario',
                'quality': 'Preferred',
                'rule': 'Higher timeframe bullish and execution timeframe bullish.',
            }
        if lower_trend == 'Bearish':
            return {
                'label': 'Scenario 2 — MTF Buyers: Bullish HTF / Bearish Pullback Execution TF',
                'number': 'Scenario 2',
                'side': 'Buyers',
                'system': 'MTF Trade Scenario',
                'quality': 'Secondary',
                'rule': 'Higher timeframe bullish and execution timeframe bearish/pullback.',
            }
        return {
            'label': 'Scenario 3 — MTF Buyers: Bullish HTF / Sideways Execution TF',
            'number': 'Scenario 3',
            'side': 'Buyers',
            'system': 'MTF Trade Scenario',
            'quality': 'Watch',
            'rule': 'Higher timeframe bullish and execution timeframe sideways.',
        }

    if higher_trend == 'Bearish':
        if lower_trend == 'Bearish':
            return {
                'label': 'Scenario 1 — MTF Sellers: Bearish HTF / Bearish Execution TF',
                'number': 'Scenario 1',
                'side': 'Sellers',
                'system': 'MTF Trade Scenario',
                'quality': 'Preferred',
                'rule': 'Higher timeframe bearish and execution timeframe bearish.',
            }
        if lower_trend == 'Bullish':
            return {
                'label': 'Scenario 2 — MTF Sellers: Bearish HTF / Bullish Pullback Execution TF',
                'number': 'Scenario 2',
                'side': 'Sellers',
                'system': 'MTF Trade Scenario',
                'quality': 'Secondary',
                'rule': 'Higher timeframe bearish and execution timeframe bullish/pullback.',
            }
        return {
            'label': 'Scenario 3 — MTF Sellers: Bearish HTF / Sideways Execution TF',
            'number': 'Scenario 3',
            'side': 'Sellers',
            'system': 'MTF Trade Scenario',
            'quality': 'Watch',
            'rule': 'Higher timeframe bearish and execution timeframe sideways.',
        }

    return {
        'label': 'No MTF Scenario — Higher Timeframe Trend Not Clear',
        'number': 'Other',
        'side': 'Neutral',
        'system': 'MTF Trade Scenario',
        'quality': 'Low',
        'rule': 'Higher timeframe is not classified as bullish or bearish.',
    }


def classify_cwt_setup_scenario(cwt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single-chart CWT Setup Scenarios from Week 2/3/4:
    - Scenario 1: Alligator trend continuation buy/sell.
    - Scenario 2: CWT jawline reversal buy/sell.
    - Scenario 3: Sleeping Alligator / closed mouth.
    """
    setup = str(cwt.get('setup', 'Wait'))
    state = str(cwt.get('state', ''))
    direction = str(cwt.get('direction', 'Neutral'))
    mouth_open = bool(cwt.get('mouth_open', False))

    if setup == 'CWT Trend Buy':
        return {
            'label': 'Scenario 1 — CWT Buyers: Open Bullish Alligator / Price Above Lips',
            'number': 'Scenario 1',
            'side': 'Buyers',
            'system': 'CWT Setup Scenario',
            'quality': 'Trend Continuation',
            'rule': 'Alligator mouth open, bullish ordering, Heikin candle above Lips.',
        }
    if setup == 'CWT Trend Sell':
        return {
            'label': 'Scenario 1 — CWT Sellers: Open Bearish Alligator / Price Below Lips',
            'number': 'Scenario 1',
            'side': 'Sellers',
            'system': 'CWT Setup Scenario',
            'quality': 'Trend Continuation',
            'rule': 'Alligator mouth open, bearish ordering, Heikin candle below Lips.',
        }
    if setup == 'CWT Reversal Buy':
        return {
            'label': 'Scenario 2 — CWT Buyers: Downward Alligator / Close Above Jawline',
            'number': 'Scenario 2',
            'side': 'Buyers',
            'system': 'CWT Setup Scenario',
            'quality': 'Reversal',
            'rule': 'Alligator mouth open and pointing downwards, candle closes above Jawline.',
        }
    if setup == 'CWT Reversal Sell':
        return {
            'label': 'Scenario 2 — CWT Sellers: Upward Alligator / Close Below Jawline',
            'number': 'Scenario 2',
            'side': 'Sellers',
            'system': 'CWT Setup Scenario',
            'quality': 'Reversal',
            'rule': 'Alligator mouth open and pointing upwards, candle closes below Jawline.',
        }
    if setup == 'No Trade / Sleeping Alligator' or (not mouth_open and 'Sleeping' in state):
        return {
            'label': 'Scenario 3 — CWT Sleeping Alligator / No Trade',
            'number': 'Scenario 3',
            'side': 'Neutral',
            'system': 'CWT Setup Scenario',
            'quality': 'No Trade',
            'rule': 'Alligator mouth is closed / sleeping.',
        }

    return {
        'label': f'No CWT Scenario — {setup}',
        'number': 'Other',
        'side': 'Neutral',
        'system': 'CWT Setup Scenario',
        'quality': 'Unclassified',
        'rule': 'Current CWT state does not fit Scenario 1, 2, or 3 exactly.',
    }


# Backward-compatible alias used by older modules.
def classify_scenario(higher_trend: str, lower_trend: str) -> Dict[str, Any]:
    return classify_mtf_trade_scenario(higher_trend, lower_trend)

def choose_setup(higher_trend: Dict[str, Any], execution_trend: Dict[str, Any], cwt: Dict[str, Any], divergence: Dict[str, Any], reversals: List[Dict[str, Any]], continuations: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_breakdown: List[Dict[str, Any]] = []
    bullish_score = bearish_score = 0
    if higher_trend['trend'] == 'Bullish':
        bullish_score += 25; score_breakdown.append({'Factor': 'Higher TF Trend', 'Bias': 'Bullish', 'Score': 25})
    elif higher_trend['trend'] == 'Bearish':
        bearish_score += 25; score_breakdown.append({'Factor': 'Higher TF Trend', 'Bias': 'Bearish', 'Score': 25})
    if execution_trend['trend'] == 'Bullish':
        bullish_score += 15; score_breakdown.append({'Factor': 'Execution TF Trend', 'Bias': 'Bullish', 'Score': 15})
    elif execution_trend['trend'] == 'Bearish':
        bearish_score += 15; score_breakdown.append({'Factor': 'Execution TF Trend', 'Bias': 'Bearish', 'Score': 15})
    if cwt['direction'] == 'Bullish' and cwt['mouth_open']:
        bullish_score += 20; score_breakdown.append({'Factor': 'Alligator / CWT', 'Bias': 'Bullish', 'Score': 20})
    elif cwt['direction'] == 'Bearish' and cwt['mouth_open']:
        bearish_score += 20; score_breakdown.append({'Factor': 'Alligator / CWT', 'Bias': 'Bearish', 'Score': 20})
    setup_type = 'Continuation'
    if divergence['bias'] == 'Bullish':
        bullish_score += 18; setup_type = 'Reversal'; score_breakdown.append({'Factor': 'RSI Divergence', 'Bias': 'Bullish', 'Score': 18})
    elif divergence['bias'] == 'Bearish':
        bearish_score += 18; setup_type = 'Reversal'; score_breakdown.append({'Factor': 'RSI Divergence', 'Bias': 'Bearish', 'Score': 18})
    pattern_candidates = reversals if setup_type == 'Reversal' else continuations
    for p in pattern_candidates[:2]:
        pts = min(int(p.get('score', 0) / 4), 18)
        if p.get('bias') == 'Bullish':
            bullish_score += pts; score_breakdown.append({'Factor': p['name'], 'Bias': 'Bullish', 'Score': pts})
        elif p.get('bias') == 'Bearish':
            bearish_score += pts; score_breakdown.append({'Factor': p['name'], 'Bias': 'Bearish', 'Score': pts})
    bias = 'Neutral'
    if bullish_score >= bearish_score + 10:
        bias = 'Bullish'
    elif bearish_score >= bullish_score + 10:
        bias = 'Bearish'
    confidence = min(100, max(bullish_score, bearish_score))
    action = 'WAIT' if bias == 'Neutral' else ('BUY PLAN' if bias == 'Bullish' else 'SELL PLAN')
    return {'bias': bias, 'action': action, 'setup_type': setup_type, 'confidence': confidence, 'bullish_score': bullish_score, 'bearish_score': bearish_score, 'score_breakdown': score_breakdown}


def analyze_symbol(symbol: str, higher_df: pd.DataFrame, lower_df: pd.DataFrame, asset_class: str, analysis_tf: str, execution_tf: str, risk_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    risk_context = risk_context or {}
    higher = add_indicators(higher_df)
    lower = add_indicators(lower_df)
    higher_trend = classify_trend(higher)
    execution_trend = classify_trend(lower)
    scenario = classify_mtf_trade_scenario(higher_trend['trend'], execution_trend['trend'])
    cwt = cwt_bias(lower)
    cwt_scenario = classify_cwt_setup_scenario(cwt)
    divergence = detect_rsi_divergence(higher)
    candles = detect_candlestick_patterns(lower)
    reversals = detect_reversal_patterns(higher)
    continuations = detect_continuation_patterns(higher, higher_trend['trend'])
    fvg = detect_fvgs(lower)
    sr = support_resistance(lower)
    sbr_rbs = detect_sbr_rbs(lower, sr)
    signal = choose_setup(higher_trend, execution_trend, cwt, divergence, reversals, continuations)
    for candle in candles[:2]:
        pts = min(int(candle['score'] / 8), 10)
        if candle['bias'] == signal['bias']:
            signal['confidence'] = min(100, signal['confidence'] + pts)
            signal['score_breakdown'].append({'Factor': candle['name'], 'Bias': candle['bias'], 'Score': pts})
    if sbr_rbs['bias'] == signal['bias']:
        signal['confidence'] = min(100, signal['confidence'] + sbr_rbs['score'] // 5)
        signal['score_breakdown'].append({'Factor': sbr_rbs['label'], 'Bias': sbr_rbs['bias'], 'Score': sbr_rbs['score'] // 5})
    projections = [p.get('projection') for p in continuations if p.get('projection') is not None]
    trade_plan = build_trade_plan(lower, signal['bias'], signal['setup_type'], scenario['label'], sr, fvg, projections[0] if projections else None)
    warnings: List[str] = []
    if risk_context.get('high_impact_news'):
        warnings.append('High-impact news/event risk is flagged.')
        signal['confidence'] = max(0, signal['confidence'] - 10)
    if risk_context.get('funding_rate_risk') and asset_class == 'Crypto':
        warnings.append('Funding-rate risk is flagged for crypto/futures.')
        signal['confidence'] = max(0, signal['confidence'] - 5)
    if risk_context.get('benchmark_conflict'):
        warnings.append('Benchmark or macro trend conflicts with the setup.')
        signal['confidence'] = max(0, signal['confidence'] - 10)
    if asset_class == 'Crypto' and not risk_context.get('isolated_margin_preferred', True):
        warnings.append('Uploaded lecture framework preferred isolated margin for futures risk containment.')
    return {'symbol': symbol, 'asset_class': asset_class, 'analysis_tf': analysis_tf, 'execution_tf': execution_tf, 'higher_frame': higher, 'execution_frame': lower, 'higher_trend': higher_trend, 'execution_trend': execution_trend, 'scenario': scenario, 'mtf_scenario': scenario, 'cwt_scenario': cwt_scenario, 'cwt': cwt, 'divergence': divergence, 'patterns': {'candlestick': candles, 'reversal': reversals, 'continuation': continuations, 'sbr_rbs': [sbr_rbs] if sbr_rbs['label'] != 'None' else []}, 'fvg': fvg, 'support_resistance': sr, 'signal': signal, 'trade_plan': trade_plan, 'warnings': warnings}
