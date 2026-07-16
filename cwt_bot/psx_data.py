from __future__ import annotations

from typing import Any, List, Optional
from io import StringIO
import re

import numpy as np
import pandas as pd
import requests

from .data_sources import normalize_ohlcv, load_yfinance_ohlcv


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]

    def clean(s: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

    cleaned_cols = {clean(c): c for c in df.columns}
    for name in candidates:
        n = clean(name)
        if n in cleaned_cols:
            return cleaned_cols[n]

    for c in df.columns:
        cc = clean(c)
        for name in candidates:
            n = clean(name)
            if n and (n in cc or cc in n):
                return c

    return None


def _smart_datetime(values: pd.Series) -> pd.Series:
    """
    Parse date/time values from PSX payloads.
    Supports text dates plus Unix timestamps in seconds or milliseconds.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(numeric) else 0

    if numeric_ratio >= 0.7:
        med = float(numeric.dropna().abs().median())
        if med > 10_000_000_000:
            return pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        if med > 1_000_000_000:
            return pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)

    return pd.to_datetime(values, errors="coerce", utc=True)


def normalize_price_dataframe(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """
    Adapted from the earlier PSX Advanced Investor Bot data loader.

    It accepts many PSX/DPS field aliases, fills missing open/high/low
    from close when necessary, and returns a CWT-engine-compatible
    OHLCV frame with a DatetimeIndex.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    mapping = {
        "date": ["date", "time", "timestamp", "datetime", "d"],
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "price", "last", "rate", "ltp"],
        "volume": ["volume", "vol", "v"],
    }

    ren = {}
    for target, cands in mapping.items():
        col = find_column(d, cands)
        if col is not None:
            ren[col] = target

    d = d.rename(columns=ren)

    if "date" not in d.columns:
        d["date"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(d), freq="D")

    if "close" not in d.columns:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            if col == "volume":
                d[col] = 0
            elif col in ["open", "high", "low"]:
                d[col] = d["close"]
            else:
                return pd.DataFrame()
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["date"] = _smart_datetime(d["date"])
    if d["date"].isna().all():
        d["date"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(d), freq="D")

    d = d.dropna(subset=["date", "close"]).sort_values("date")
    if d.empty:
        return pd.DataFrame()

    # Return in the exact form required by the CWT bot.
    out = d.rename(columns={"date": "datetime"})[
        ["datetime", "open", "high", "low", "close", "volume"]
    ]
    return normalize_ohlcv(out)


def _parse_psx_json(js: Any, symbol: str) -> pd.DataFrame:
    """
    Data-source parser adapted from the prior PSX Advanced Investor Bot.

    Handles:
    - dict payloads with data/chart/prices/results keys
    - raw list payloads
    - list[dict] rows
    - list[list] rows
    """
    payload = None

    if isinstance(js, dict):
        for k in ["data", "chart", "prices", "results"]:
            if k in js and js[k]:
                payload = js[k]
                break

        if payload is None:
            for v in js.values():
                if isinstance(v, list) and v:
                    payload = v
                    break

    elif isinstance(js, list):
        payload = js

    if payload is None:
        return pd.DataFrame()

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return normalize_price_dataframe(pd.DataFrame(payload), symbol)

    if isinstance(payload, list) and payload and isinstance(payload[0], (list, tuple)):
        rows = []
        for item in payload:
            if len(item) >= 6:
                # Original v11 parsing behavior:
                # date, open, high, low, close, volume are read from the last 5 positions.
                rows.append({
                    "date": item[0],
                    "open": item[-5],
                    "high": item[-4],
                    "low": item[-3],
                    "close": item[-2],
                    "volume": item[-1],
                })
            elif len(item) >= 3:
                # Typical DPS time-series form:
                # [date/timestamp, close_or_price, volume]
                rows.append({
                    "date": item[0],
                    "open": item[1],
                    "high": item[1],
                    "low": item[1],
                    "close": item[1],
                    "volume": item[2],
                })

        return normalize_price_dataframe(pd.DataFrame(rows), symbol)

    return pd.DataFrame()


def load_psx_data(symbol: str, interval: str = "daily", timeout: int = 12) -> pd.DataFrame:
    """
    Core PSX/DPS loader reused from the earlier bot and adapted for the CWT engine.
    """
    symbol = symbol.upper().strip()

    endpoints = [
        f"https://dps.psx.com.pk/timeseries/eod/{symbol}",
        f"https://dps.psx.com.pk/timeseries/int/{symbol}",
    ]
    if interval == "intraday":
        endpoints = list(reversed(endpoints))

    errs = []
    for url in endpoints:
        try:
            r = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json,text/plain,*/*",
                },
                timeout=timeout,
            )
            r.raise_for_status()

            df = _parse_psx_json(r.json(), symbol)
            if not df.empty:
                df.attrs["psx_source"] = url
                if {"high", "low"}.issubset(df.columns):
                    df.attrs["source_warning"] = (
                        "PSX DPS data loaded. If this endpoint only provides price/volume rows, "
                        "high/low may be synthesized from close as in the earlier PSX bot."
                    )
                return df

            errs.append(url + ": parsed empty")

        except Exception as e:
            errs.append(url + ": " + str(e))

    raise RuntimeError("Could not fetch usable PSX data. " + " | ".join(errs))


# Compatibility alias used by the CWT PSX app.
def load_psx_dps_ohlcv(symbol: str, mode: str = "daily") -> pd.DataFrame:
    return load_psx_data(symbol, interval=mode)


def load_psx_yahoo_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    clean = symbol.strip().upper()
    yahoo_symbol = clean if clean.endswith(".KA") else f"{clean}.KA"
    df = load_yfinance_ohlcv(yahoo_symbol, interval=interval, period=period)
    df.attrs["source_symbol"] = yahoo_symbol
    return df


def load_psx_csv(uploaded_file: Any) -> pd.DataFrame:
    return normalize_ohlcv(pd.read_csv(uploaded_file))


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV if source data is finer than requested.
    Daily data cannot create genuine intraday candles.
    """
    rule_map = {
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1wk": "W-FRI",
        "1mo": "ME",
    }
    rule = rule_map.get(timeframe)
    if rule is None:
        return df.copy()

    if df.empty:
        return df.copy()

    resampled = pd.DataFrame({
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
        "volume": df["volume"].resample(rule).sum(),
    }).dropna(subset=["open", "high", "low", "close"])

    resampled.attrs.update(df.attrs)
    return resampled


def _clean_symbol_series(series: pd.Series) -> list[str]:
    symbols: list[str] = []
    for value in series.dropna().astype(str):
        sym = value.strip().upper()
        # Keep standard exchange symbols, rights, ETFs etc.; remove obvious header repeats.
        if not sym or sym in {"SYMBOL", "NAN"}:
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,20}", sym):
            symbols.append(sym)
    # Preserve order while deduplicating.
    return list(dict.fromkeys(symbols))


def fetch_psx_symbol_universe(universe: str = "Eligible Scrips") -> list[str]:
    """
    Fetch a current PSX symbol universe from official PSX Data Portal pages.

    Supported:
    - Eligible Scrips: broad PSX symbol list
    - KSE-100 Constituents: current KSE100 constituent symbols

    If PSX page structure changes, the app lets the user fall back to a custom symbol list.
    """
    normalized = universe.strip().lower()
    if "kse" in normalized:
        url = "https://dps.psx.com.pk/indices/KSE100"
    else:
        url = "https://dps.psx.com.pk/eligible-scrips"

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PSX-CWT-Scenario-Scanner/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    response = requests.get(url, headers=headers, timeout=25)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError(f"No HTML tables found on PSX universe page: {url}")

    # Prefer a table with a Symbol-like column.
    for table in tables:
        if table is None or table.empty:
            continue
        for col in table.columns:
            if "symbol" in str(col).strip().lower():
                symbols = _clean_symbol_series(table[col])
                if symbols:
                    return symbols

    # Fallback to first column of first non-empty table.
    for table in tables:
        if table is not None and not table.empty:
            symbols = _clean_symbol_series(table.iloc[:, 0])
            if symbols:
                return symbols

    raise RuntimeError(f"Could not extract symbols from PSX universe page: {url}")
