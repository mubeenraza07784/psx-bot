from __future__ import annotations

from typing import Any, Dict, List
import math
import re
import pandas as pd


KEYWORD_FLAGS = {
    "Going Concern": [
        "going concern",
        "material uncertainty",
        "substantial doubt",
    ],
    "Qualified / Adverse Audit": [
        "qualified opinion",
        "adverse opinion",
        "disclaimer of opinion",
        "emphasis of matter",
    ],
    "Auditor or CFO Exit": [
        "auditor resignation",
        "resignation of auditor",
        "chief financial officer resigned",
        "cfo resigned",
        "change of auditor",
    ],
    "Default / Non-compliance": [
        "default",
        "non-compliance",
        "breach of covenant",
        "overdue payable",
    ],
    "Fraud / Investigation": [
        "fraud",
        "investigation",
        "forensic",
        "misstatement",
    ],
    "One-off Charges": [
        "impairment",
        "restructuring",
        "exceptional loss",
        "one-off charge",
        "non-recurring charge",
    ],
    "Serial Acquisition Risk": [
        "acquisition",
        "merger",
        "goodwill",
        "business combination",
    ],
}


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _flag(metric: str, status: str, severity: str, message: str, value: Any = None) -> Dict[str, Any]:
    return {
        "Metric": metric,
        "Status": status,
        "Severity": severity,
        "Value": value,
        "Message": message,
    }


def scan_text_red_flags(text: str | None) -> List[Dict[str, Any]]:
    if not text:
        return []
    text_l = str(text).lower()
    flags: List[Dict[str, Any]] = []
    for category, phrases in KEYWORD_FLAGS.items():
        matches = [p for p in phrases if p in text_l]
        if matches:
            severity = "CRITICAL" if category in {"Going Concern", "Qualified / Adverse Audit", "Fraud / Investigation"} else "HIGH"
            flags.append(_flag(
                metric=category,
                status="Detected",
                severity=severity,
                message=f"Detected keywords: {', '.join(matches[:4])}",
                value=len(matches),
            ))
    return flags


def score_financial_fakery(row: pd.Series, report_text: str | None = None) -> Dict[str, Any]:
    """
    Financial fakery risk model derived from the uploaded Week 10 framework:
    - declining cash flows
    - serial charges / acquirers
    - bills not being paid
    - changes in credit terms and receivables
    - CFO / auditor exits
    - auditor report / going-concern keywords
    """
    flags: List[Dict[str, Any]] = []
    points = 0.0

    cfo3 = _num(row.get("cfo_avg_3y"))
    cfo5 = _num(row.get("cfo_avg_5y"))
    if cfo3 is not None and cfo5 is not None:
        if cfo3 < cfo5:
            flags.append(_flag("CFO Trend", "Warning", "HIGH", "3Y operating cash flow average is below 5Y average.", round(cfo3 - cfo5, 4)))
            points += 2.5
        else:
            flags.append(_flag("CFO Trend", "Pass", "LOW", "3Y operating cash flow average is not below 5Y average.", round(cfo3 - cfo5, 4)))

    ccfo = _num(row.get("ccfo"))
    cpat = _num(row.get("cpat"))
    if ccfo is not None and cpat is not None:
        if ccfo < cpat:
            flags.append(_flag("cCFO vs cPAT", "Warning", "HIGH", "Cumulative CFO is below cumulative PAT; profits may be weakly cash-backed.", round(ccfo - cpat, 4)))
            points += 2.5
        else:
            flags.append(_flag("cCFO vs cPAT", "Pass", "LOW", "Cumulative CFO supports cumulative PAT.", round(ccfo - cpat, 4)))

    cash3 = _num(row.get("cash_per_share_3y"))
    cash5 = _num(row.get("cash_per_share_5y"))
    if cash3 is not None and cash5 is not None:
        if cash3 < cash5:
            flags.append(_flag("Cash per Share Trend", "Warning", "MODERATE", "3Y cash/share average is below 5Y average.", round(cash3 - cash5, 4)))
            points += 1.5
        else:
            flags.append(_flag("Cash per Share Trend", "Pass", "LOW", "Cash/share trend is stable or improving.", round(cash3 - cash5, 4)))

    receivables_growth = _num(row.get("receivables_growth"))
    sales_growth = _num(row.get("sales_growth"))
    if receivables_growth is not None and sales_growth is not None:
        spread = receivables_growth - sales_growth
        if spread > 10:
            flags.append(_flag("Receivables vs Sales", "Warning", "HIGH", "Receivables growth materially exceeds sales growth.", round(spread, 2)))
            points += 2.0
        elif spread > 0:
            flags.append(_flag("Receivables vs Sales", "Caution", "MODERATE", "Receivables growth is above sales growth.", round(spread, 2)))
            points += 1.0
        else:
            flags.append(_flag("Receivables vs Sales", "Pass", "LOW", "Receivables do not outgrow sales.", round(spread, 2)))

    dso = _num(row.get("dso"))
    if dso is not None:
        if dso > 120:
            flags.append(_flag("DSO", "Warning", "HIGH", "Very high days sales outstanding.", dso))
            points += 2.0
        elif dso > 90:
            flags.append(_flag("DSO", "Caution", "MODERATE", "Elevated collection period.", dso))
            points += 1.0
        else:
            flags.append(_flag("DSO", "Pass", "LOW", "DSO is not unusually elevated.", dso))

    ccc = _num(row.get("ccc"))
    if ccc is not None:
        if ccc > 180:
            flags.append(_flag("Cash Conversion Cycle", "Warning", "HIGH", "Cash conversion cycle is very long.", ccc))
            points += 2.0
        elif ccc > 120:
            flags.append(_flag("Cash Conversion Cycle", "Caution", "MODERATE", "Cash conversion cycle is elevated.", ccc))
            points += 1.0

    dividends = _num(row.get("dividends_paid"))
    fcf = _num(row.get("fcf"))
    if dividends is not None and fcf is not None:
        if dividends > 0 and fcf < dividends:
            flags.append(_flag("Dividend Funding", "Warning", "HIGH", "Dividend cash outflow exceeds FCF; sustainability requires review.", round(fcf - dividends, 4)))
            points += 2.0
        elif dividends > 0:
            flags.append(_flag("Dividend Funding", "Pass", "LOW", "FCF appears to cover dividend cash outflow.", round(fcf - dividends, 4)))

    debt_trend = str(row.get("debt_trend", "")).strip().lower()
    if debt_trend:
        if any(token in debt_trend for token in ["rising", "increase", "up", "worsening"]):
            flags.append(_flag("Debt Trend", "Caution", "MODERATE", "Debt trend appears to be rising.", debt_trend))
            points += 1.0

    insider = str(row.get("insider_signal", "")).strip().lower()
    if insider:
        if "selling" in insider or "negative" in insider:
            flags.append(_flag("Insider Signal", "Caution", "MODERATE", "Insider signal indicates selling/caution.", insider))
            points += 1.0
        elif "buying" in insider or "positive" in insider:
            flags.append(_flag("Insider Signal", "Pass", "LOW", "Insider signal indicates buying/positive confidence.", insider))

    text_flags = scan_text_red_flags(report_text)
    flags.extend(text_flags)
    for tf in text_flags:
        points += 4.0 if tf["Severity"] == "CRITICAL" else 2.5

    # Severity
    if any(f["Severity"] == "CRITICAL" for f in flags) or points >= 10:
        risk = "CRITICAL"
        verdict = "Avoid / deep due diligence required"
    elif points >= 6:
        risk = "HIGH"
        verdict = "High red-flag risk"
    elif points >= 3:
        risk = "MODERATE"
        verdict = "Review carefully"
    else:
        risk = "LOW"
        verdict = "No major red-flag cluster detected"

    return {
        "fraud_risk_level": risk,
        "fraud_risk_points": round(points, 2),
        "verdict": verdict,
        "flags_table": pd.DataFrame(flags),
        "summary": verdict,
    }
