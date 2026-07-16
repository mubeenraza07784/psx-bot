from __future__ import annotations

from typing import Any, Dict, List
import re
import pandas as pd


CATALYST_RULES = {
    "Share Buyback": ["buyback", "share buy back", "repurchase of shares", "treasury shares"],
    "Capacity Expansion": ["capacity increase", "plant capacity", "expansion", "debottlenecking", "new plant", "production increase"],
    "Solar / Energy Efficiency": ["solar", "renewable energy", "energy saving", "power project"],
    "Merger / Acquisition": ["merger", "acquisition", "amalgamation", "scheme of arrangement"],
    "Product / Market Expansion": ["new product", "new market", "export order", "commercial production", "launch"],
    "Sales / Production Outperformance": ["sales increase", "production increase", "higher dispatches", "volume growth"],
    "Debt Reduction / Refinancing": ["debt reduction", "loan repayment", "refinancing", "lower finance cost"],
}


def scan_catalyst_text(text: str | None, symbol: str | None = None) -> Dict[str, Any]:
    if not text:
        return {
            "symbol": symbol,
            "catalyst_score": 0,
            "catalyst_grade": "None",
            "catalysts": pd.DataFrame(),
            "explosive_stock_flag": "No",
        }

    lower = str(text).lower()
    rows: List[Dict[str, Any]] = []
    score = 0

    for category, phrases in CATALYST_RULES.items():
        matches = [p for p in phrases if p in lower]
        if matches:
            if category in {"Share Buyback", "Capacity Expansion", "Solar / Energy Efficiency"}:
                points = 3
            elif category in {"Merger / Acquisition", "Product / Market Expansion"}:
                points = 2
            else:
                points = 1
            score += points
            rows.append({
                "Catalyst": category,
                "Points": points,
                "Matched Terms": ", ".join(matches[:5]),
            })

    if score >= 7:
        grade = "High"
        explosive = "Yes"
    elif score >= 4:
        grade = "Medium"
        explosive = "Watchlist"
    elif score > 0:
        grade = "Low"
        explosive = "No"
    else:
        grade = "None"
        explosive = "No"

    return {
        "symbol": symbol,
        "catalyst_score": score,
        "catalyst_grade": grade,
        "catalysts": pd.DataFrame(rows),
        "explosive_stock_flag": explosive,
    }


def catalyst_from_events(events_df: pd.DataFrame, symbols: List[str] | None = None) -> pd.DataFrame:
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in events_df.iterrows():
        event_text = " | ".join(str(v) for v in row.tolist() if pd.notna(v))
        symbol = row.get("Symbol") if "Symbol" in events_df.columns else None
        if symbols and symbol not in symbols:
            continue
        scan = scan_catalyst_text(event_text, symbol=symbol)
        rows.append({
            "Symbol": symbol,
            "Catalyst Score": scan["catalyst_score"],
            "Catalyst Grade": scan["catalyst_grade"],
            "Explosive Flag": scan["explosive_stock_flag"],
            "Event": event_text[:450],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Catalyst Score", "Symbol"], ascending=[False, True])
    return out
