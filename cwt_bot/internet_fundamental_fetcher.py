from __future__ import annotations

import re
from typing import Any, Iterable
from urllib.parse import quote_plus

import pandas as pd
import requests


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PSX-Institutional-Terminal/7.6"

# Columns that the existing fundamental engine can understand. The fetcher tries
# to map common webpage/table labels into these canonical fields.
METRIC_ALIASES: dict[str, list[str]] = {
    "company": ["company", "company name", "name"],
    "sector": ["sector", "industry"],
    "price": ["price", "current price", "last price", "market price", "cmp"],
    "market_cap": ["market cap", "market capitalization"],
    "eps": ["eps", "earning per share", "earnings per share"],
    "pe": ["p/e", "pe", "price earning", "price to earnings"],
    "pb": ["p/b", "pb", "price to book"],
    "ps": ["p/s", "ps", "price to sales"],
    "dividend_yield": ["dividend yield", "yield"],
    "roe": ["roe", "return on equity"],
    "roa": ["roa", "return on asset"],
    "net_margin": ["net margin", "npm"],
    "gross_margin": ["gross margin", "gpm"],
    "debt_to_equity": ["debt to equity", "de ratio", "d/e"],
    "current_ratio": ["current ratio"],
    "interest_coverage": ["interest coverage"],
    "revenue_cagr_3y": ["revenue cagr 3y", "sales cagr 3y", "3y revenue cagr"],
    "profit_cagr_3y": ["profit cagr 3y", "pat cagr 3y", "net profit cagr 3y"],
    "eps_growth_avg": ["eps growth", "earnings growth"],
    "inventory_turnover": ["inventory turnover"],
    "inventory_days": ["inventory days", "dio"],
    "dso": ["dso", "days sales outstanding"],
    "ccc": ["cash conversion cycle", "ccc"],
    "cfo_avg_3y": ["operating cash flow", "cfo", "cash flow from operations"],
    "fcf": ["free cash flow", "fcf"],
    "book_value_per_share": ["book value per share", "bvps"],
    "cash_per_share": ["cash per share", "cps"],
    "fair_value": ["fair value", "intrinsic value", "target price"],
    "margin_of_safety": ["margin of safety", "mos"],
}

SOURCE_URL_TEMPLATES = [
    "https://dps.psx.com.pk/company/{symbol}",
    "https://dps.psx.com.pk/profile/{symbol}",
    "https://www.sarmaaya.pk/psx/company/{symbol}",
    "https://sarmaaya.pk/psx/company/{symbol}",
    "https://www.askanalyst.com.pk/company/{symbol}",
    "https://www.investing.com/search/?q={symbol}",
]


def _clean_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _to_number(value: Any) -> float | None:
    text = str(value or "").replace(",", "").strip()
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    pct = "%" in text
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        number = float(match.group(0))
        return number
    except Exception:
        return None


def _canonical_metric(label: Any) -> str | None:
    cleaned = _clean_key(label)
    if not cleaned:
        return None
    for target, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            a = _clean_key(alias)
            if a and (a == cleaned or a in cleaned or cleaned in a):
                return target
    return None


def _request_text(url: str, timeout: int = 15) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    text = response.text or ""
    if len(text) < 100:
        raise ValueError("empty/too-short response")
    return text


def _extract_from_tables(html: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        tables = pd.read_html(html)
    except Exception:
        tables = []
    for table in tables:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        # Two-column metric/value tables.
        if table.shape[1] >= 2:
            for _, row in table.iterrows():
                label = row.iloc[0]
                metric = _canonical_metric(label)
                if metric and metric not in out:
                    val = row.iloc[1]
                    if metric in {"company", "sector"}:
                        out[metric] = str(val).strip()
                    else:
                        num = _to_number(val)
                        if num is not None:
                            out[metric] = num
        # Wide tables where headers are metrics and first row has values.
        if len(table) >= 1:
            first = table.iloc[0]
            for col in table.columns:
                metric = _canonical_metric(col)
                if metric and metric not in out:
                    val = first[col]
                    if metric in {"company", "sector"}:
                        out[metric] = str(val).strip()
                    else:
                        num = _to_number(val)
                        if num is not None:
                            out[metric] = num
    return out


def _extract_from_text(html: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    for metric, aliases in METRIC_ALIASES.items():
        if metric in {"company", "sector"}:
            continue
        for alias in aliases:
            # Examples: P/E 7.5, Dividend Yield: 8.2%, ROE 18.1%
            pattern = rf"{re.escape(alias)}\s*[:\-]?\s*([-+]?\d[\d,]*(?:\.\d+)?\s*%?)"
            match = re.search(pattern, text, flags=re.I)
            if match:
                num = _to_number(match.group(1))
                if num is not None:
                    out[metric] = num
                    break
    return out


def fetch_symbol_fundamentals(symbol: str) -> tuple[dict[str, Any], str, str]:
    """Best-effort online fundamental fetch.

    Returns (row, status, source_note). It never raises for normal missing/blocked
    website cases, because the Decision Center should continue on technical basis.
    """
    s = str(symbol or "").upper().strip()
    if not s:
        return {}, "FAILED", "Blank symbol"
    errors: list[str] = []
    for template in SOURCE_URL_TEMPLATES:
        url = template.format(symbol=quote_plus(s))
        try:
            html = _request_text(url)
            row = {"symbol": s}
            row.update(_extract_from_tables(html))
            row.update({k: v for k, v in _extract_from_text(html).items() if k not in row})
            usable = [k for k in row.keys() if k not in {"symbol", "company", "sector"}]
            if usable:
                row.setdefault("company", s)
                row.setdefault("sector", "Unknown")
                return row, "FOUND", f"Fetched from {url}; fields: {', '.join(sorted(usable)[:12])}"
            errors.append(f"{url}: no recognized metric fields")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return {"symbol": s, "company": s, "sector": "Unknown"}, "NOT FOUND", "; ".join(errors[-3:])


def fetch_internet_fundamentals(symbols: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for symbol in symbols:
        row, status, note = fetch_symbol_fundamentals(str(symbol).upper().strip())
        logs.append({"Symbol": str(symbol).upper().strip(), "Internet Fundamental Status": status, "Internet Fundamental Note": note})
        if status == "FOUND":
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(logs)
