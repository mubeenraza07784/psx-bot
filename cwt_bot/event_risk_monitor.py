from __future__ import annotations

from datetime import datetime, timezone, date
from io import StringIO
from typing import Dict, Any, List
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PSX-CWT-Watchtower/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/plain,*/*",
}


def _fetch_html(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _tables_from_html(html: str) -> List[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except Exception:
        return []


def _first_symbol_match(text: str, symbols: List[str]) -> str | None:
    upper = text.upper()
    for sym in symbols:
        s = str(sym).upper().strip()
        if s and re.search(rf"\b{re.escape(s)}\b", upper):
            return s
    return None


def fetch_psx_company_announcements(symbols: List[str] | None = None, max_rows: int = 80) -> pd.DataFrame:
    url = "https://dps.psx.com.pk/announcements/companies"
    html = _fetch_html(url)
    tables = _tables_from_html(html)
    rows = []
    for table in tables:
        if table is None or table.empty:
            continue
        table = table.copy()
        for _, row in table.iterrows():
            txt = " | ".join(str(v) for v in row.tolist())
            sym = _first_symbol_match(txt, symbols or []) if symbols else None
            if symbols and not sym:
                continue
            rows.append({
                "Source": "PSX Company Announcements",
                "Symbol": sym,
                "Event": txt[:500],
                "URL": url,
            })
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def fetch_psx_notices(max_rows: int = 80) -> pd.DataFrame:
    url = "https://dps.psx.com.pk/announcements/psx"
    html = _fetch_html(url)
    tables = _tables_from_html(html)
    rows = []
    for table in tables:
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            txt = " | ".join(str(v) for v in row.tolist())
            rows.append({"Source": "PSX Notices", "Symbol": None, "Event": txt[:500], "URL": url})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def fetch_psx_financial_announcements(symbols: List[str] | None = None, max_rows: int = 80) -> pd.DataFrame:
    url = "https://www.psx.com.pk/psx/announcement/financial-announcements"
    html = _fetch_html(url)
    tables = _tables_from_html(html)
    rows = []
    for table in tables:
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            txt = " | ".join(str(v) for v in row.tolist())
            sym = _first_symbol_match(txt, symbols or []) if symbols else None
            if symbols and not sym:
                continue
            rows.append({"Source": "PSX Financial Announcements", "Symbol": sym, "Event": txt[:500], "URL": url})
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def fetch_sbp_monetary_policy_calendar() -> pd.DataFrame:
    url = "https://www.sbp.org.pk/m_policy/mp-calendar.asp"
    html = _fetch_html(url)
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    pattern = re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{2}-[A-Za-z]{3}-\d{2})")
    rows = []
    today = datetime.now(timezone.utc).date()
    for day_name, date_text in pattern.findall(text):
        try:
            event_date = datetime.strptime(date_text, "%d-%b-%y").date()
        except Exception:
            continue
        days = (event_date - today).days
        rows.append({
            "Source": "SBP Monetary Policy Calendar",
            "Event": "Monetary Policy Committee meeting",
            "Date": event_date.isoformat(),
            "Days Away": days,
            "Risk Window": "UPCOMING" if 0 <= days <= 14 else ("TODAY" if days == 0 else ("PAST" if days < 0 else "FUTURE")),
            "URL": url,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Date") if not df.empty else df


def fetch_pbs_press_releases(max_rows: int = 30) -> pd.DataFrame:
    url = "https://www.pbs.gov.pk/press-release/"
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for a in soup.find_all("a"):
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")
        if not title:
            continue
        title_up = title.upper()
        if "CPI" in title_up or "CONSUMER PRICE INDEX" in title_up or "NATIONAL ACCOUNTS" in title_up:
            rows.append({
                "Source": "PBS Press Releases",
                "Event": title[:500],
                "URL": href or url,
            })
            if len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)


def build_news_event_risk_snapshot(symbols: List[str] | None = None) -> Dict[str, Any]:
    frames = []
    errors = []

    funcs = [
        ("PSX Company Announcements", lambda: fetch_psx_company_announcements(symbols=symbols)),
        ("PSX Financial Announcements", lambda: fetch_psx_financial_announcements(symbols=symbols)),
        ("PSX Notices", fetch_psx_notices),
        ("SBP Monetary Policy", fetch_sbp_monetary_policy_calendar),
        ("PBS Press Releases", fetch_pbs_press_releases),
    ]
    for name, fn in funcs:
        try:
            df = fn()
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    risk_notes: List[str] = []
    if not combined.empty:
        if "Source" in combined.columns and (combined["Source"] == "PSX Company Announcements").any():
            risk_notes.append("Company announcements were found for the selected symbol/watchlist.")
        if "Source" in combined.columns and (combined["Source"] == "PSX Financial Announcements").any():
            risk_notes.append("Financial announcement records are present; review result/dividend/meeting context.")
        if "Days Away" in combined.columns:
            upcoming = combined[pd.to_numeric(combined["Days Away"], errors="coerce").between(0, 14)]
            if not upcoming.empty:
                risk_notes.append("An SBP monetary policy meeting is within the next 14 days.")
        if "Source" in combined.columns and (combined["Source"] == "PBS Press Releases").any():
            risk_notes.append("Recent PBS macro releases are available; CPI/news shock review is recommended.")

    if any("SBP" in note for note in risk_notes):
        level = "HIGH"
    elif risk_notes:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "status": "OK" if not combined.empty else "NO_DATA",
        "risk_level": level,
        "risk_notes": risk_notes,
        "events": combined,
        "errors": errors,
    }
