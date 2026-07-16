from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
from .data_sources import load_yfinance_ohlcv, load_ccxt_ohlcv
from .signals import analyze_symbol


def scan_symbols(symbols: List[str], source: str, asset_class: str, analysis_tf: str, execution_tf: str, period: str, exchange: str, risk_context: Dict[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        try:
            if source == 'CCXT Crypto':
                higher = load_ccxt_ohlcv(symbol=symbol, timeframe=analysis_tf, exchange_id=exchange)
                lower = load_ccxt_ohlcv(symbol=symbol, timeframe=execution_tf, exchange_id=exchange)
            else:
                higher = load_yfinance_ohlcv(symbol=symbol, interval=analysis_tf, period=period)
                lower = load_yfinance_ohlcv(symbol=symbol, interval=execution_tf, period=period)
            result = analyze_symbol(symbol, higher, lower, asset_class, analysis_tf, execution_tf, risk_context or {})
            rows.append({'Symbol': symbol, 'Bias': result['signal']['bias'], 'Action': result['signal']['action'], 'Setup': result['signal']['setup_type'], 'Scenario': result['scenario']['label'], 'Confidence': result['signal']['confidence'], 'Order': result['trade_plan']['order_type'], 'Entry': result['trade_plan']['entry'], 'SL': result['trade_plan']['stop_loss'], 'TP': result['trade_plan']['take_profit'], 'Divergence': result['divergence']['label'], 'Warnings': ' | '.join(result['warnings'])})
        except Exception as exc:
            rows.append({'Symbol': symbol, 'Bias': 'ERROR', 'Action': 'ERROR', 'Confidence': None, 'Warnings': str(exc)})
    return pd.DataFrame(rows).sort_values(by=['Confidence'], ascending=False, na_position='last') if rows else pd.DataFrame()
