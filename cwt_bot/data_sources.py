from __future__ import annotations

from typing import Any
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance-style MultiIndex columns safely."""
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        # yfinance usually returns columns like ('Close', 'AAPL').
        # We only need the field name for a single-symbol request.
        work.columns = [str(col[0]).strip() for col in work.columns]
    else:
        work.columns = [str(col).strip() for col in work.columns]
    return work


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common OHLCV aliases to canonical names without creating duplicate close columns."""
    work = _flatten_columns(df)
    original_columns = list(work.columns)

    # Prefer unadjusted Close when both Close and Adj Close exist.
    lowered = {str(col).strip().lower().replace(' ', '_'): col for col in original_columns}
    preferred: dict[str, str] = {}

    aliases = {
        'open': ['open', 'o'],
        'high': ['high', 'h'],
        'low': ['low', 'l'],
        'close': ['close', 'c', 'adj_close', 'adjclose'],
        'volume': ['volume', 'v'],
        'datetime': ['date', 'datetime', 'timestamp', 'time'],
    }

    for canonical, candidates in aliases.items():
        # Explicit priority is important for close: close > c > adj_close.
        for alias in candidates:
            if alias in lowered:
                preferred[canonical] = lowered[alias]
                break

    selected = pd.DataFrame(index=work.index)
    for canonical, original in preferred.items():
        # If duplicated labels somehow remain, work.loc[:, original] can be a DataFrame.
        series_or_frame = work.loc[:, original]
        if isinstance(series_or_frame, pd.DataFrame):
            series_or_frame = series_or_frame.iloc[:, 0]
        selected[canonical] = series_or_frame

    return selected


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError('No OHLCV data returned.')

    work = _canonicalize_columns(df)

    if 'datetime' in work.columns:
        work['datetime'] = pd.to_datetime(work['datetime'], errors='coerce', utc=True)
        work = work.dropna(subset=['datetime']).set_index('datetime')
    else:
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index, errors='coerce', utc=True)
        work = work[~work.index.isna()]

    needed = ['open', 'high', 'low', 'close']
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise ValueError(f'Missing OHLC columns: {missing}')

    if 'volume' not in work.columns:
        work['volume'] = 0.0

    for col in ['open', 'high', 'low', 'close', 'volume']:
        # Ensure the input is a 1-D Series before numeric conversion.
        series_or_frame = work.loc[:, col]
        if isinstance(series_or_frame, pd.DataFrame):
            series_or_frame = series_or_frame.iloc[:, 0]
        work[col] = pd.to_numeric(series_or_frame, errors='coerce')

    work = work.dropna(subset=['open', 'high', 'low', 'close']).sort_index()
    if work.empty:
        raise ValueError('OHLCV data became empty after normalization.')

    return work[['open', 'high', 'low', 'close', 'volume']]


def load_yfinance_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_ohlcv(data)


def load_ccxt_ohlcv(symbol: str, timeframe: str, exchange_id: str = 'binance', limit: int = 500) -> pd.DataFrame:
    import ccxt
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f'Unknown CCXT exchange: {exchange_id}')
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({'enableRateLimit': True})
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not bars:
        raise ValueError('CCXT returned no OHLCV candles.')
    df = pd.DataFrame(bars, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
    return normalize_ohlcv(df)


def load_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return normalize_ohlcv(df)
