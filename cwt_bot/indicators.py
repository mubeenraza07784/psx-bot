from __future__ import annotations

import numpy as np
import pandas as pd


def smma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean()


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    ha_open = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i == 0:
            ha_open.append((row['open'] + row['close']) / 2.0)
        else:
            ha_open.append((ha_open[-1] + ha['ha_close'].iloc[i - 1]) / 2.0)
    ha['ha_open'] = ha_open
    ha['ha_high'] = pd.concat([df['high'], ha['ha_open'], ha['ha_close']], axis=1).max(axis=1)
    ha['ha_low'] = pd.concat([df['low'], ha['ha_open'], ha['ha_close']], axis=1).min(axis=1)
    ha['ha_bull'] = ha['ha_close'] >= ha['ha_open']
    return ha


def alligator(df: pd.DataFrame) -> pd.DataFrame:
    median = (df['high'] + df['low']) / 2.0
    out = pd.DataFrame(index=df.index)
    out['jaw'] = smma(median, 13).shift(8)
    out['teeth'] = smma(median, 8).shift(5)
    out['lips'] = smma(median, 5).shift(3)
    return out


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = smma(gain, length)
    avg_loss = smma(loss, length).replace(0, np.nan)
    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))
    return result.fillna(50.0)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df['close'].shift(1)
    ranges = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1)
    return smma(ranges.max(axis=1), length).fillna((df['high'] - df['low']).rolling(length).mean())


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ha = heikin_ashi(out)
    alg = alligator(out)
    out = out.join(ha).join(alg)
    out['rsi'] = rsi(out['close'])
    out['atr'] = atr(out)
    out['body'] = (out['close'] - out['open']).abs()
    out['range'] = (out['high'] - out['low']).replace(0, np.nan)
    out['upper_wick'] = out['high'] - out[['open', 'close']].max(axis=1)
    out['lower_wick'] = out[['open', 'close']].min(axis=1) - out['low']
    return out
