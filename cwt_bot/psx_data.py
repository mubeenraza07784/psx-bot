from __future__ import annotations

from typing import Any, List, Optional
from io import StringIO
import re
import time

import pandas as pd
import requests

from .data_sources import normalize_ohlcv, load_yfinance_ohlcv


FALLBACK_KSE100_SYMBOLS = [
    "ABL", "ABOT", "AGP", "AICL", "AIRLINK", "AKBL", "APL", "ATLH", "ATRL", "AVN",
    "BAFL", "BAHL", "BOP", "CHCC", "COLG", "DAWH", "DGKC", "EFERT", "ENGROH", "EPCL",
    "FABL", "FATIMA", "FCEPL", "FFC", "FFBL", "GHGL", "HBL", "HUBC", "ILP", "INDU",
    "ISL", "JDWS", "JVDC", "KOHC", "LUCK", "MARI", "MCB", "MEBL", "MLCF", "MTL",
    "NBP", "NESTLE", "NRL", "OGDC", "PABC", "PAEL", "PIOC", "POL", "PPL", "PSO",
    "PSX", "SAZEW", "SEARL", "SHFA", "SRVI", "SYS", "THALL", "THCCL", "TRG", "UBL",
    "UNITY",
]

FALLBACK_ELIGIBLE_SYMBOLS = list(dict.fromkeys(FALLBACK_KSE100_SYMBOLS + [
    "AGIL", "AHCL", "AHL", "AKDHL", "ATIL", "BIFO", "BNL", "BWHL", "CENI",
    "CNERGY", "DAWN", "DCR", "DYNO", "EFUG", "FCSC", "FECTC", "FCCL", "FCEL",
    "FLYNG", "GAL", "GGL", "GHNI", "GRR", "GWLC", "HPL", "IMAGE", "JLICL", "JSGCL",
    "LSEVL", "MDTL", "MCBIM", "NATF", "NETSOL", "OTSU", "PAKOXY", "PNSC", "POWER",
    "PTC", "PTL", "RCML", "SAPT", "SPEL", "TPLI", "TGL", "ZAL",
]))


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]

    def clean(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

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
    for target_col, candidates in mapping.items():
        col = find_column(d, candidates)
        if col is not None:
            ren[col] = target_col

    d = d.rename(columns=ren)

    if "date" not in d.columns:
        d["date"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(d), freq="D")

    if "close" not in d.columns:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            if col == "volume":
                d[col] = 0
            elif col in {"open", "high", "low"}:
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

    out = d.rename(columns={"date": "datetime"})[
        ["datetime", "open", "high", "low", "close", "volume"]
    ]
    return normalize_ohlcv(out)


def _parse_psx_json(js: Any, symbol: str) -> pd.DataFrame:
    payload = None

    if isinstance(js, dict):
        for key in ["data", "chart", "prices", "results"]:
            if key in js and js[key]:
                payload = js[key]
                break

        if payload is None:
            for value in js.values():
                if isinstance(value, list) and value:
                    payload = value
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
                rows.append({
                    "date": item[0],
                    "open": item[-5],
                    "high": item[-4],
                    "low": item[-3],
                    "close": item[-2],
                    "volume": item[-1],
                })
            elif len(item) >= 3:
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
    symbol = str(symbol or "").upper().strip()
    if not symbol:
        raise RuntimeError("Empty PSX symbol")

    endpoints = [
        f"https://dps.psx.com.pk/timeseries/eod/{symbol}",
        f"https://dps.psx.com.pk/timeseries/int/{symbol}",
    ]
    if interval == "intraday":
        endpoints = list(reversed(endpoints))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Connection": "close",
    }

    errs = []
    for url in endpoints:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            df = _parse_psx_json(response.json(), symbol)
            if not df.empty:
                df.attrs["psx_source"] = url
                return df

            errs.append(f"{url}: parsed empty")
        except Exception as exc:
            errs.append(f"{url}: {exc}")

    raise RuntimeError("Could not fetch usable PSX data. " + " | ".join(errs))


def load_psx_dps_ohlcv(symbol: str, mode: str = "daily") -> pd.DataFrame:
    return load_psx_data(symbol, interval=mode)


def _safe_yahoo_period(interval: str, period: str) -> str:
    tf = str(interval or "1d").lower().strip()
    requested = str(period or "1y").lower().strip()

    if tf == "1m":
        return "7d"
    if tf in {"2m", "5m", "15m", "30m", "60m", "90m"}:
        return "60d"
    if tf in {"1h", "4h"}:
        return "6mo" if requested in {"1y", "2y", "5y", "max"} else requested
    return requested


def load_psx_yahoo_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    clean = str(symbol or "").strip().upper()
    if not clean:
        raise RuntimeError("Empty Yahoo symbol")

    yahoo_symbol = clean if clean.endswith(".KA") else f"{clean}.KA"
    requested_interval = str(interval or "1d").strip().lower()

    download_interval = "1h" if requested_interval == "4h" else requested_interval
    download_period = _safe_yahoo_period(requested_interval, period)

    df = load_yfinance_ohlcv(yahoo_symbol, interval=download_interval, period=download_period)
    if requested_interval == "4h":
        df = resample_ohlcv(df, "4h")

    df.attrs["source_symbol"] = yahoo_symbol
    df.attrs["download_interval"] = download_interval
    df.attrs["download_period"] = download_period
    return df


def load_psx_csv(uploaded_file: Any) -> pd.DataFrame:
    return normalize_ohlcv(pd.read_csv(uploaded_file))


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1wk": "W-FRI",
        "1mo": "ME",
    }
    rule = rule_map.get(str(timeframe or "").strip().lower())
    if rule is None:
        return df.copy()

    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "datetime" in work.columns:
            work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True)
            work = work.dropna(subset=["datetime"]).set_index("datetime")
        else:
            return work

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in work.columns:
            return work
        work[col] = pd.to_numeric(work[col], errors="coerce")

    resampled = pd.DataFrame({
        "open": work["open"].resample(rule).first(),
        "high": work["high"].resample(rule).max(),
        "low": work["low"].resample(rule).min(),
        "close": work["close"].resample(rule).last(),
        "volume": work["volume"].resample(rule).sum(),
    }).dropna(subset=["open", "high", "low", "close"])

    resampled.attrs.update(getattr(df, "attrs", {}))
    return resampled


def _clean_symbol_series(series: pd.Series) -> list[str]:
    symbols: list[str] = []
    for value in series.dropna().astype(str):
        sym = value.strip().upper()
        if not sym or sym in {"SYMBOL", "NAN"}:
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,20}", sym):
            symbols.append(sym)
    return list(dict.fromkeys(symbols))


def fetch_psx_symbol_universe(universe: str = "Eligible Scrips") -> list[str]:
    normalized = str(universe or "").strip().lower()
    if "kse" in normalized:
        url = "https://dps.psx.com.pk/indices/KSE100"
        fallback_symbols = FALLBACK_KSE100_SYMBOLS
    else:
        url = "https://dps.psx.com.pk/eligible-scrips"
        fallback_symbols = FALLBACK_ELIGIBLE_SYMBOLS

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))
            if not tables:
                raise RuntimeError(f"No HTML tables found on PSX universe page: {url}")

            for table in tables:
                if table is None or table.empty:
                    continue
                for col in table.columns:
                    if "symbol" in str(col).strip().lower():
                        symbols = _clean_symbol_series(table[col])
                        if symbols:
                            return symbols

            for table in tables:
                if table is not None and not table.empty:
                    symbols = _clean_symbol_series(table.iloc[:, 0])
                    if symbols:
                        return symbols

            raise RuntimeError(f"Could not extract symbols from PSX universe page: {url}")

        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))

    print(f"PSX universe fetch failed, using fallback list for {universe}: {last_error}")
    return list(dict.fromkeys(fallback_symbols))
