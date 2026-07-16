from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .psx_data import load_psx_yahoo_ohlcv
from .signals import analyze_symbol
from .pro_metrics import evaluate_psx_pro_score
from .risk_alerts import build_risk_warning
from .prediction_engine import run_prediction_engine
from .fib_engine import fibonacci_retracement_confluence
from .secular_engine import classify_secular_trend
from .market_hazard import detect_price_hazards
from .event_risk_monitor import build_news_event_risk_snapshot
from .confluence_engine import build_confluence_score
from .fundamental_master import normalize_fundamental_columns, score_fundamental_universe
from .fraud_risk_engine import score_financial_fakery
from .intrinsic_value_engine import intrinsic_value_composite
from .knowledge_integrator import decision_knowledge_tags, knowledge_registry_table, engine_coverage_table, knowledge_status_summary
from .internet_fundamental_fetcher import fetch_internet_fundamentals


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


DEFAULT_DISCIPLINE_RULES = {
    "minimum_confluence_for_entry": 60,
    "minimum_confluence_for_strong_entry": 70,
    "maximum_hazard_for_entry": {"LOW", "MODERATE"},
    "maximum_news_for_entry": {"LOW", "MODERATE"},
    "minimum_fundamental_score_for_investor": 55,
    "high_fraud_levels": {"HIGH", "CRITICAL"},
}



def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not df.columns.duplicated().any():
        return df
    result = pd.DataFrame(index=df.index)
    for col in dict.fromkeys(list(df.columns)):
        same = df.loc[:, df.columns == col]
        if isinstance(same, pd.DataFrame) and same.shape[1] > 1:
            series = same.iloc[:, 0].copy()
            for i in range(1, same.shape[1]):
                series = series.where(series.notna() & ~series.astype(str).str.strip().isin(["", "-", "nan", "None"]), same.iloc[:, i])
            result[col] = series
        else:
            result[col] = same.iloc[:, 0] if isinstance(same, pd.DataFrame) else same
    return result



def _dedupe_for_lookup(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Make key_col unique before pandas to_dict(orient='index')."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or key_col not in df.columns:
        return pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    out[key_col] = out[key_col].astype(str).str.upper().str.strip()
    out = out[out[key_col].notna() & out[key_col].astype(str).str.strip().ne("") & out[key_col].astype(str).str.upper().ne("NAN")]
    out = out.drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)
    return out


def _safe_to_dict_index(df: pd.DataFrame, key_col: str) -> dict:
    """Safe df.set_index(key_col).to_dict('index') replacement."""
    out = _dedupe_for_lookup(df, key_col)
    if out is None or out.empty or key_col not in out.columns:
        return {}
    try:
        return out.set_index(key_col, drop=True).to_dict("index")
    except Exception:
        lookup = {}
        for _, row in out.iterrows():
            key = str(row.get(key_col, "") or "").upper().strip()
            if key:
                lookup[key] = {k: v for k, v in row.to_dict().items() if k != key_col}
        return lookup


def _clean_symbol(value: Any) -> str:
    return str(value or "").upper().strip()


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_portfolio_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Flexible portfolio normalizer. It accepts common columns used in the user's
    prior portfolio sheets/PDF exports and returns:
    symbol, quantity, avg_buy, mtm_price (optional), company (optional).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "quantity", "avg_buy", "mtm_price", "company"])

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lookup = {"".join(ch for ch in str(c).lower() if ch.isalnum()): c for c in out.columns}

    alias_map = {
        "symbol": ["symbol", "ticker", "script", "code"],
        "quantity": ["quantity", "qty", "shares", "holding", "units"],
        "avg_buy": ["avgbuy", "averagebuy", "averageprice", "costprice", "avgcost", "buyprice"],
        "mtm_price": ["mtmprice", "currentprice", "marketprice", "lastprice", "cmp", "close"],
        "company": ["company", "companyname", "name", "names"],
    }

    ren: Dict[str, str] = {}
    for target, aliases in alias_map.items():
        for alias in aliases:
            if alias in lookup:
                ren[lookup[alias]] = target
                break

    out = out.rename(columns=ren)
    out = _collapse_duplicate_columns(out)
    if "symbol" not in out.columns:
        raise ValueError("Portfolio input needs a Symbol/Ticker column.")
    if "quantity" not in out.columns:
        out["quantity"] = np.nan
    if "avg_buy" not in out.columns:
        out["avg_buy"] = np.nan
    if "mtm_price" not in out.columns:
        out["mtm_price"] = np.nan
    if "company" not in out.columns:
        out["company"] = out["symbol"]

    out["symbol"] = out["symbol"].map(_clean_symbol)
    for col in ["quantity", "avg_buy", "mtm_price"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["symbol"] != ""].drop_duplicates(subset=["symbol"], keep="last")
    return out[["symbol", "company", "quantity", "avg_buy", "mtm_price"]].reset_index(drop=True)


def _fundamental_lookup(fundamentals_df: pd.DataFrame | None) -> Tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if fundamentals_df is None or fundamentals_df.empty:
        return {}, pd.DataFrame()
    normalized = normalize_fundamental_columns(fundamentals_df)
    scored = score_fundamental_universe(normalized)
    rows = _dedupe_for_lookup(scored.copy(), "symbol")
    lookup = _safe_to_dict_index(rows, "symbol") if not rows.empty else {}
    normalized = _dedupe_for_lookup(normalized, "symbol") if isinstance(normalized, pd.DataFrame) and not normalized.empty else normalized
    return lookup, normalized


def _safe_intrinsic_value(row: pd.Series | None) -> dict[str, Any]:
    if row is None or row.empty:
        return {}
    try:
        return intrinsic_value_composite(row)
    except Exception as exc:
        return {"valuation_grade": "Unavailable", "valuation_verdict": f"Valuation unavailable: {exc}"}


def _safe_fraud_score(row: pd.Series | None) -> dict[str, Any]:
    if row is None or row.empty:
        return {}
    try:
        return score_financial_fakery(row)
    except Exception as exc:
        return {"fraud_risk_level": "Unavailable", "verdict": f"Fraud-risk engine unavailable: {exc}"}


def _event_count_for_symbol(news: dict[str, Any], symbol: str) -> int:
    events = news.get("events")
    if not isinstance(events, pd.DataFrame) or events.empty or "Symbol" not in events.columns:
        return 0
    return int((events["Symbol"].astype(str).str.upper().str.strip() == symbol).sum())


def _decision_scorecard(
    *,
    confluence: dict[str, Any],
    pro: dict[str, Any],
    hazard: dict[str, Any],
    news: dict[str, Any],
    fundamental: dict[str, Any] | None,
    intrinsic: dict[str, Any] | None,
    fraud: dict[str, Any] | None,
    holding_row: pd.Series | None,
    discipline: dict[str, Any],
) -> tuple[str, str]:
    """Human-readable reason checklist for every BUY/HOLD/WAIT/AVOID decision."""
    score = _to_float(confluence.get("confluence_score"), 0.0) or 0.0
    entry_gate = _to_float(discipline.get("minimum_confluence_for_entry"), 60.0) or 60.0
    strong_gate = _to_float(discipline.get("minimum_confluence_for_strong_entry"), 70.0) or 70.0
    pro_score = _to_float(pro.get("pro_score"), None)
    trade_quality = str(pro.get("trade_quality") or "Unavailable")
    latest = pro.get("latest_metrics", {}) if isinstance(pro, dict) else {}
    rsi = _to_float(latest.get("rsi"), None)
    adx = _to_float(latest.get("adx"), None)
    volume_ratio = _to_float(latest.get("volume_ratio"), None)
    hazard_level = str(hazard.get("hazard_level") or "LOW").upper()
    news_level = str(news.get("risk_level") or "LOW").upper()
    signal = confluence.get("signal") if isinstance(confluence, dict) else None
    funda_score = _to_float((fundamental or {}).get("fundamental_score_pct"), None)
    funda_grade = str((fundamental or {}).get("fundamental_grade") or "Missing")
    val_grade = str((intrinsic or {}).get("valuation_grade") or "Unavailable")
    mos = _to_float((intrinsic or {}).get("best_margin_of_safety_pct"), None)
    fraud_level = str((fraud or {}).get("fraud_risk_level") or "Unavailable").upper()
    has_holding = holding_row is not None

    items: list[tuple[str, str, str]] = []
    items.append(("Confluence", "PASS" if score >= entry_gate else "FAIL", f"{score:.1f} vs entry gate {entry_gate:.0f}; strong gate {strong_gate:.0f}"))
    if pro_score is None:
        items.append(("PRO technical score", "MISSING", "No PRO score available from chart/indicator engine"))
    else:
        items.append(("PRO technical score", "PASS" if pro_score >= 55 else "WEAK", f"{pro_score:.1f}; quality={trade_quality}"))
    if rsi is None:
        items.append(("RSI", "MISSING", "RSI unavailable"))
    elif rsi >= 70:
        items.append(("RSI", "CAUTION", f"{rsi:.1f}; overbought / avoid chasing"))
    elif rsi <= 30:
        items.append(("RSI", "CAUTION", f"{rsi:.1f}; oversold, needs reversal confirmation"))
    elif rsi >= 50:
        items.append(("RSI", "PASS", f"{rsi:.1f}; bullish/neutral momentum"))
    else:
        items.append(("RSI", "WEAK", f"{rsi:.1f}; momentum below 50"))
    if adx is not None:
        items.append(("ADX / trend strength", "PASS" if adx >= 20 else "WEAK", f"{adx:.1f}"))
    if volume_ratio is not None:
        items.append(("Volume confirmation", "PASS" if volume_ratio >= 1 else "WEAK", f"{volume_ratio:.2f}x average"))
    items.append(("Price/news risk", "PASS" if hazard_level in discipline["maximum_hazard_for_entry"] and news_level in discipline["maximum_news_for_entry"] else "BLOCK", f"hazard={hazard_level}, news={news_level}"))
    if funda_score is None:
        items.append(("Fundamentals", "MISSING", "No matched fundamental row; decision is technical-only and conservative"))
    else:
        min_funda = _to_float(discipline.get("minimum_fundamental_score_for_investor"), 55.0) or 55.0
        items.append(("Fundamentals", "PASS" if funda_score >= min_funda else "WEAK", f"{funda_score:.1f}% ({funda_grade}); gate {min_funda:.0f}%"))
    if mos is None:
        items.append(("Valuation / MOS", "MISSING", f"valuation={val_grade}"))
    else:
        items.append(("Valuation / MOS", "PASS" if mos > 0 else "WEAK", f"MOS {mos:.1f}%; valuation={val_grade}"))
    items.append(("Fraud/red flags", "PASS" if fraud_level not in discipline["high_fraud_levels"] else "BLOCK", f"fraud risk={fraud_level}"))
    if has_holding and holding_row is not None:
        pnl_pct = _to_float(holding_row.get("pnl_pct"), None)
        if pnl_pct is not None:
            items.append(("Portfolio position", "INFO", f"holding exists; P&L {pnl_pct:.1f}%"))
    factors = confluence.get("factors", []) if isinstance(confluence, dict) else []
    if factors:
        top = sorted(factors, key=lambda x: abs(float(x.get("Points", 0) or 0)), reverse=True)[:5]
        for factor in top:
            pts = _to_float(factor.get("Points"), 0.0) or 0.0
            status = "ADD" if pts > 0 else "DEDUCT" if pts < 0 else "INFO"
            items.append((str(factor.get("Factor", "Confluence factor")), status, f"{pts:+.1f} pts — {factor.get('Reason', '')}"))

    scorecard = " | ".join([f"{name}: {status} ({detail})" for name, status, detail in items])
    short_reasons = []
    for name, status, detail in items:
        if status in {"FAIL", "BLOCK", "WEAK", "CAUTION", "MISSING"}:
            short_reasons.append(f"{name} {status.lower()}: {detail}")
    if not short_reasons:
        short_reasons.append("Major gates passed; use entry, stop-loss and position sizing rules.")
    return scorecard, " ; ".join(short_reasons[:7])


def _fundamental_reading(fundamental: dict[str, Any] | None, discipline: dict[str, Any]) -> dict[str, Any]:
    """Convert raw fundamental score into strict decision gates.

    Earlier versions allowed a stock to become BUY purely from technical confluence.
    This helper prevents wrong answers for investment decisions by making missing or
    weak fundamentals visible and by blocking fresh buys when the business data is
    not strong enough.
    """
    min_score = _to_float(discipline.get("minimum_fundamental_score_for_investor"), 55.0) or 55.0
    score = _to_float((fundamental or {}).get("fundamental_score_pct"), None)
    grade = str((fundamental or {}).get("fundamental_grade") or "Missing")
    if score is None:
        return {"status": "MISSING", "score": None, "grade": grade, "passes": False, "detail": "No matched fundamental row"}
    if score >= max(70.0, min_score + 12):
        status = "STRONG"
    elif score >= min_score:
        status = "ACCEPTABLE"
    elif score >= max(40.0, min_score - 15):
        status = "WEAK"
    else:
        status = "POOR"
    return {"status": status, "score": score, "grade": grade, "passes": score >= min_score, "detail": f"{score:.1f}% ({grade})"}


def _valuation_reading(intrinsic: dict[str, Any] | None) -> dict[str, Any]:
    mos = _to_float((intrinsic or {}).get("best_margin_of_safety_pct"), None)
    grade = str((intrinsic or {}).get("valuation_grade") or "Unavailable")
    verdict = str((intrinsic or {}).get("valuation_verdict") or "")
    if mos is None:
        status = "MISSING"
        passes = True  # do not block if valuation data is absent; show lower confidence instead
        detail = grade
    elif mos >= 20:
        status = "ATTRACTIVE"
        passes = True
        detail = f"MOS {mos:.1f}% ({grade})"
    elif mos >= 0:
        status = "FAIR"
        passes = True
        detail = f"MOS {mos:.1f}% ({grade})"
    elif mos > -15:
        status = "EXPENSIVE"
        passes = False
        detail = f"MOS {mos:.1f}% ({grade})"
    else:
        status = "OVERVALUED"
        passes = False
        detail = f"MOS {mos:.1f}% ({grade})"
    if verdict and verdict not in detail:
        detail = f"{detail}; {verdict}"
    return {"status": status, "mos": mos, "grade": grade, "passes": passes, "detail": detail}


def _technical_reading(confluence: dict[str, Any], pro: dict[str, Any], discipline: dict[str, Any]) -> dict[str, Any]:
    score = _to_float(confluence.get("confluence_score"), 0.0) or 0.0
    entry_gate = _to_float(discipline.get("minimum_confluence_for_entry"), 60.0) or 60.0
    strong_gate = _to_float(discipline.get("minimum_confluence_for_strong_entry"), 70.0) or 70.0
    latest = pro.get("latest_metrics", {}) if isinstance(pro, dict) else {}
    rsi = _to_float(latest.get("rsi"), None)
    pro_score = _to_float(pro.get("pro_score"), None)
    trade_quality = str(pro.get("trade_quality") or "Unavailable")
    if score >= strong_gate:
        status = "STRONG"
    elif score >= entry_gate:
        status = "ACTIONABLE"
    elif score >= max(45.0, entry_gate - 12):
        status = "WATCH"
    else:
        status = "WEAK"
    overbought = rsi is not None and rsi >= 72
    oversold = rsi is not None and rsi <= 30
    return {
        "status": status,
        "score": score,
        "pro_score": pro_score,
        "trade_quality": trade_quality,
        "rsi": rsi,
        "overbought": overbought,
        "oversold": oversold,
        "entry_gate": entry_gate,
        "strong_gate": strong_gate,
        "passes": score >= entry_gate and not overbought,
        "detail": f"confluence {score:.1f}, PRO {pro_score if pro_score is not None else 'N/A'}, RSI {rsi if rsi is not None else 'N/A'}",
    }


def _decision_confidence(
    technical: dict[str, Any], fundamental: dict[str, Any], valuation: dict[str, Any],
    hazard_level: str, news_level: str, fraud_level: str,
) -> tuple[float, str]:
    score = 0.0
    score += 30.0 if technical["status"] == "STRONG" else 22.0 if technical["status"] == "ACTIONABLE" else 12.0 if technical["status"] == "WATCH" else 4.0
    score += 30.0 if fundamental["status"] == "STRONG" else 22.0 if fundamental["status"] == "ACCEPTABLE" else 8.0 if fundamental["status"] == "WEAK" else 0.0
    score += 18.0 if valuation["status"] == "ATTRACTIVE" else 12.0 if valuation["status"] == "FAIR" else 6.0 if valuation["status"] == "MISSING" else 0.0
    score += 10.0 if hazard_level in {"LOW", "MODERATE"} else 0.0
    score += 6.0 if news_level in {"LOW", "MODERATE"} else 0.0
    score += 6.0 if fraud_level not in {"HIGH", "CRITICAL"} else 0.0
    score = max(0.0, min(100.0, score))
    if score >= 80:
        label = "High"
    elif score >= 60:
        label = "Medium"
    elif score >= 40:
        label = "Low / Review"
    else:
        label = "Low"
    return round(score, 1), label


def _autopilot_decision(
    *,
    symbol: str,
    confluence: dict[str, Any],
    pro: dict[str, Any],
    hazard: dict[str, Any],
    news: dict[str, Any],
    fundamental: dict[str, Any] | None,
    intrinsic: dict[str, Any] | None,
    fraud: dict[str, Any] | None,
    holding_row: pd.Series | None,
    discipline: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    """Strict all-engine decision logic.

    The old logic could mark BUY when technical confluence was high, even if
    fundamentals were missing/weak. This version separates trading readiness,
    investment quality, valuation, red flags and existing portfolio context.
    """
    has_holding = holding_row is not None
    hazard_level = str(hazard.get("hazard_level") or "LOW").upper()
    news_level = str(news.get("risk_level") or "LOW").upper()
    fraud_level = str((fraud or {}).get("fraud_risk_level") or "Unavailable").upper()
    technical = _technical_reading(confluence, pro, discipline)
    funda = _fundamental_reading(fundamental, discipline)
    valuation = _valuation_reading(intrinsic)
    risk_gate = hazard_level not in discipline["maximum_hazard_for_entry"] or news_level not in discipline["maximum_news_for_entry"]
    fraud_gate = fraud_level in discipline["high_fraud_levels"]
    scorecard, gate_breakdown = _decision_scorecard(confluence=confluence, pro=pro, hazard=hazard, news=news, fundamental=fundamental, intrinsic=intrinsic, fraud=fraud, holding_row=holding_row, discipline=discipline)
    confidence_score, confidence_label = _decision_confidence(technical, funda, valuation, hazard_level, news_level, fraud_level)

    pnl_pct = _to_float(holding_row.get("pnl_pct"), None) if has_holding and holding_row is not None else None
    bias = str((confluence or {}).get("signal") or "")
    signal_bias = str((pro or {}).get("latest_metrics", {}).get("bias") or "")

    label = "Discipline Gate"
    action = "WAIT"
    reasons: list[str] = []

    if fraud_gate:
        action = "REVIEW / REDUCE" if has_holding else "AVOID"
        label = "Red-Flag Override"
        reasons.append(f"Financial red-flag gate triggered: fraud-risk={fraud_level}.")
    elif risk_gate:
        action = "HOLD / WAIT" if has_holding else "AVOID FOR NOW"
        label = "Risk Override"
        reasons.append(f"Risk gate blocked fresh action: price hazard={hazard_level}, news/event risk={news_level}.")
    elif not has_holding and funda["status"] == "MISSING":
        if technical["status"] in {"STRONG", "ACTIONABLE"}:
            action = "TECHNICAL BUY SETUP - FUNDAMENTALS MISSING"
            label = "Technical Basis Only"
            reasons.append("Fundamental data was searched/checked but is still missing. This is not a fundamental investment approval; answer is given on technical basis only.")
        elif technical["status"] == "WATCH":
            action = "TECHNICAL WATCH - FUNDAMENTALS MISSING"
            label = "Technical Basis Only"
            reasons.append("Fundamental data is missing, so this is only a technical watchlist answer. Wait for price confirmation or provide fundamentals.")
        else:
            action = "WAIT - TECHNICAL WEAK / FUNDAMENTALS MISSING"
            label = "Technical Basis Only"
            reasons.append("Fundamental data is missing and the technical setup is weak; wait or provide required fundamentals.")
    elif not has_holding and not funda["passes"]:
        action = "AVOID / WAIT"
        label = "Weak Fundamentals"
        reasons.append(f"Fresh buy is blocked because fundamentals are {funda['status']} ({funda['detail']}).")
    elif not has_holding and not valuation["passes"]:
        action = "WATCH - VALUATION EXPENSIVE"
        label = "Valuation Gate"
        reasons.append(f"Fresh buy is not approved because valuation is {valuation['status']} ({valuation['detail']}).")
    elif not has_holding:
        if technical["status"] == "STRONG" and funda["passes"] and valuation["passes"]:
            action = "BUY CANDIDATE"
            label = "All-Engine Buy Candidate"
            reasons.append(f"Technical setup is {technical['status']} and fundamentals pass ({funda['detail']}).")
        elif technical["status"] in {"ACTIONABLE", "WATCH"}:
            action = "WATCH / CONDITIONAL BUY"
            label = "Wait for Entry Confirmation"
            reasons.append(f"Fundamentals pass, but technical setup is only {technical['status']}; wait for the planned entry trigger.")
        else:
            action = "WAIT"
            label = "Technical Gate"
            reasons.append(f"Fundamentals may pass, but technical setup is {technical['status']} ({technical['detail']}).")
    else:
        # Portfolio holdings: action must consider P&L and whether adding is justified.
        if funda["status"] == "MISSING":
            if technical["status"] == "STRONG" and pnl_pct is not None and pnl_pct < -5:
                action = "AVERAGE CAREFULLY - TECHNICAL BASIS ONLY"
                label = "Technical Basis Only"
                reasons.append("Holding exists but fundamentals are missing. Technical setup is strong, so averaging is only a technical-basis idea, not a fundamental approval.")
            elif technical["status"] in {"STRONG", "ACTIONABLE"}:
                action = "HOLD / BUY MORE ONLY ON TECHNICAL DIP"
                label = "Technical Basis Only"
                reasons.append("Holding exists but fundamentals are missing. Hold; any add/buy-more idea is technical-basis only until fundamentals are provided.")
            elif technical["status"] == "WATCH":
                action = "HOLD / TECHNICAL WATCH"
                label = "Technical Basis Only"
                reasons.append("Holding exists but fundamentals are missing. Hold on technical basis and wait for stronger confirmation.")
            else:
                action = "HOLD / REVIEW - FUNDAMENTALS MISSING"
                label = "Technical Basis Only"
                reasons.append("Holding exists but fundamentals are missing and chart support is weak. Do not average until data improves.")
        elif not funda["passes"] and technical["status"] in {"WEAK", "WATCH"}:
            action = "REVIEW / REDUCE"
            label = "Weak Business + Weak Chart"
            reasons.append(f"Holding has weak fundamentals ({funda['detail']}) and no strong technical support.")
        elif not valuation["passes"] and technical["status"] != "STRONG":
            action = "HOLD / TRIM ON STRENGTH"
            label = "Valuation Caution"
            reasons.append(f"Valuation looks {valuation['status']}; avoid adding and consider trimming on strength.")
        elif technical["status"] == "STRONG" and funda["passes"] and valuation["passes"]:
            if pnl_pct is not None and pnl_pct < -5:
                action = "AVERAGE CAREFULLY"
                label = "Average Only on Planned Level"
                reasons.append(f"Position is down {abs(pnl_pct):.1f}%, but fundamentals and technicals pass; average only near entry/support with stop discipline.")
            elif pnl_pct is not None and pnl_pct > 30:
                action = "HOLD / BUY MORE ONLY ON DIP"
                label = "Protect Profit"
                reasons.append(f"Position is up {pnl_pct:.1f}%; hold, protect profit, and buy more only on a pullback to the entry zone.")
            else:
                action = "BUY MORE / HOLD"
                label = "Accumulation Candidate"
                reasons.append("Existing holding has strong technical support and acceptable fundamentals/valuation.")
        elif technical["status"] in {"ACTIONABLE", "WATCH"} and funda["passes"]:
            action = "HOLD"
            label = "Hold / Wait for Stronger Entry"
            reasons.append("Fundamentals pass, but technical confirmation is not strong enough for buy-more or averaging.")
        else:
            action = "HOLD / REVIEW"
            label = "Mixed Evidence"
            reasons.append("Signals are mixed; hold only if it still fits your portfolio allocation and risk plan.")

    if technical["overbought"] and action in {"BUY CANDIDATE", "BUY MORE / HOLD", "AVERAGE CAREFULLY"}:
        action = "WATCH / DO NOT CHASE" if not has_holding else "HOLD / DO NOT ADD"
        label = "Overbought Guard"
        reasons.append(f"RSI is {technical['rsi']:.1f}; avoid chasing above the planned entry zone.")

    reasons.append(f"Technical: {technical['detail']}.")
    reasons.append(f"Fundamental: {funda['detail']}.")
    if funda["status"] == "MISSING":
        reasons.append("Decision basis: TECHNICAL BASIS ONLY because complete fundamental data could not be found/loaded.")
    reasons.append(f"Valuation: {valuation['detail']}.")
    reasons.append(f"Risk gates: hazard={hazard_level}, news={news_level}, fraud={fraud_level}.")
    if pnl_pct is not None:
        reasons.append(f"Portfolio P&L: {pnl_pct:.1f}%.")
    reasons.append("Main checklist: " + gate_breakdown + ".")
    reasons.append("Full scorecard: " + scorecard)

    diagnostics = {
        "technical_status": technical["status"],
        "fundamental_status": funda["status"],
        "fundamental_data_status": funda["status"],
        "valuation_status": valuation["status"],
        "decision_confidence_score": confidence_score,
        "decision_confidence": confidence_label,
        "data_completeness_pct": None,
    }
    return action, label, " ".join(reasons), diagnostics


def _risk_reward_targets(entry: Any, stop_loss: Any, bias: str | None = None) -> dict[str, float | None]:
    """Return TP1/TP2/TP3 using 1R/2R/3R from entry and stop."""
    e = _to_float(entry, None)
    s = _to_float(stop_loss, None)
    if e is None or s is None or e == s:
        return {"TP1": None, "TP2": None, "TP3": None}
    risk = abs(e - s)
    direction = -1.0 if str(bias or "").upper().startswith("BEAR") else 1.0
    return {
        "TP1": round(float(e + direction * risk), 8),
        "TP2": round(float(e + direction * 2 * risk), 8),
        "TP3": round(float(e + direction * 3 * risk), 8),
    }


def _support_resistance_levels(result: dict[str, Any]) -> tuple[list[float], list[float]]:
    sr = result.get("support_resistance", {}) if isinstance(result, dict) else {}
    supports = []
    resistances = []
    for value in sr.get("supports", []) or []:
        f = _to_float(value, None)
        if f is not None:
            supports.append(f)
    for value in sr.get("resistances", []) or []:
        f = _to_float(value, None)
        if f is not None:
            resistances.append(f)
    return sorted(set(supports)), sorted(set(resistances))


def _build_execution_levels(result: dict[str, Any], plan: dict[str, Any], bias: str | None = None) -> dict[str, Any]:
    """Create reliable Entry/SL/TP values even when the base trade_plan is incomplete.

    This fixes blank or unrealistic targets in result tables. For bullish/buy decisions
    it prefers the plan entry, then nearby support/close. For non-bullish cases it still
    produces a risk map, but labels the trigger as a review level.
    """
    frame = result.get("execution_frame") if isinstance(result, dict) else None
    close = None
    atr = None
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        close = _to_float(frame.get("close", pd.Series(dtype=float)).iloc[-1], None) if "close" in frame.columns else None
        if "atr" in frame.columns:
            atr = _to_float(frame["atr"].iloc[-1], None)
    if close is None:
        close = _to_float(plan.get("entry"), None)
    if close is None:
        return {"Entry": None, "Stop Loss": None, "TP1": None, "TP2": None, "TP3": None, "Entry Type": plan.get("order_type", "Review")}
    if atr is None or atr <= 0:
        atr = max(close * 0.025, 0.01)

    supports, resistances = _support_resistance_levels(result)
    nearest_support = max([s for s in supports if s <= close], default=(supports[-1] if supports else close - 2 * atr))
    nearest_resistance = min([r for r in resistances if r >= close], default=(resistances[0] if resistances else close + 2 * atr))
    bullish = not str(bias or "").upper().startswith("BEAR")

    entry = _to_float(plan.get("entry"), None)
    sl = _to_float(plan.get("stop_loss"), None)
    entry_type = str(plan.get("order_type") or "Review")

    # Reject obviously unrealistic entries far from current price.
    if entry is None or entry <= 0 or abs(entry - close) / max(close, 0.01) > 0.18:
        if bullish:
            entry = round(float(max(close, nearest_support)), 4)
            entry_type = "Buy near CMP/support" if nearest_support <= close else "Buy on breakout"
        else:
            entry = round(float(close), 4)
            entry_type = "Review / exit-risk level"
    if sl is None or sl <= 0 or (bullish and sl >= entry) or ((not bullish) and sl <= entry):
        if bullish:
            sl = round(float(min(nearest_support, entry - 1.5 * atr)), 4)
            if sl >= entry:
                sl = round(float(entry - 2.0 * atr), 4)
        else:
            sl = round(float(max(nearest_resistance, entry + 1.5 * atr)), 4)
            if sl <= entry:
                sl = round(float(entry + 2.0 * atr), 4)
    targets = _risk_reward_targets(entry, sl, bias)
    return {"Entry": round(float(entry), 4), "Stop Loss": round(float(sl), 4), **targets, "Entry Type": entry_type}


def _data_completeness_pct(fundamental: dict[str, Any] | None, intrinsic: dict[str, Any] | None, fraud: dict[str, Any] | None) -> float:
    checks = [
        _to_float((fundamental or {}).get("fundamental_score_pct"), None) is not None,
        _to_float((intrinsic or {}).get("best_margin_of_safety_pct"), None) is not None or bool((intrinsic or {}).get("valuation_grade")),
        bool((fraud or {}).get("fraud_risk_level")),
    ]
    return round(100.0 * sum(bool(x) for x in checks) / len(checks), 1)


def _fundamental_missing_indicators(fundamental: dict[str, Any] | None) -> list[str]:
    """List exact missing fundamental checks whenever the scorer can provide them."""
    detail = (fundamental or {}).get("detail_table")
    if isinstance(detail, pd.DataFrame) and not detail.empty and "rating" in detail.columns:
        miss = detail[detail["rating"].astype(str).str.lower().eq("missing")]
        names = []
        for _, row in miss.iterrows():
            bucket = str(row.get("Bucket", "")).strip()
            metric = str(row.get("Metric", "")).strip()
            name = f"{bucket}: {metric}" if bucket and metric else metric or bucket
            if name:
                names.append(name)
        return names[:40]
    if _to_float((fundamental or {}).get("fundamental_score_pct"), None) is None:
        return [
            "Growth: Revenue CAGR 3Y", "Growth: Profit CAGR 3Y", "Growth: EPS trend/acceleration",
            "Stability: CFO vs PAT", "Stability: Debt to Equity", "Stability: Current Ratio", "Stability: Interest Coverage",
            "Valuation: P/E", "Valuation: P/B", "Valuation: P/S", "Valuation: Dividend Yield", "Valuation: EV/EBITDA",
            "Inventory: Inventory Turnover", "Inventory: DSO", "Inventory: Cash Conversion Cycle",
            "Cashflow: FCF", "Cashflow: FCF Yield", "Cashflow: FCF/CFO", "Cashflow: CROIC",
        ]
    return []


def _missing_data_request(
    *,
    symbol: str,
    fundamental: dict[str, Any] | None,
    intrinsic: dict[str, Any] | None,
    fraud: dict[str, Any] | None,
    holding_row: pd.Series | None,
    is_portfolio_symbol: bool,
) -> tuple[str, str, str]:
    """Return user-facing required-data instructions for incomplete analysis.

    The Decision Center should not silently continue with weak inputs. Whenever a
    key dataset is absent, this creates an exact request that can be shown in the
    table, cards and symbol report.
    """
    missing_groups: list[str] = []
    fields: list[str] = []
    instructions: list[str] = []

    missing_funda_indicators = _fundamental_missing_indicators(fundamental)
    funda_score = _to_float((fundamental or {}).get("fundamental_score_pct"), None)
    missing_count = _to_float((fundamental or {}).get("missing"), None)
    if funda_score is None or (missing_count is not None and missing_count > 0):
        missing_groups.append("Fundamental indicators")
        fields.extend(missing_funda_indicators or [
            "Symbol", "Company", "Sector", "Growth metrics", "Stability metrics",
            "Valuation metrics", "Inventory metrics", "Cashflow metrics",
        ])
        if funda_score is None:
            instructions.append(
                f"Internet lookup could not provide complete fundamentals for {symbol}. Upload/paste a Google Sheet row or the 5 Sarmaaya screenshots "
                "(Growth, Stability, Valuation, Inventory, Cashflow). Until then, the answer is technical-basis only."
            )
        else:
            instructions.append(
                f"Some fundamental indicators are still missing for {symbol}. The Decision Center can continue, but confidence is reduced and any answer using this row is partial-fundamental plus technical basis."
            )

    mos = _to_float((intrinsic or {}).get("best_margin_of_safety_pct"), None)
    valuation_grade = str((intrinsic or {}).get("valuation_grade") or "").strip()
    if mos is None and not valuation_grade:
        missing_groups.append("Valuation / margin of safety")
        fields.extend(["Current price", "Fair value / intrinsic value", "Margin of safety", "Upside/downside"])
        instructions.append(
            f"Upload the Main Page screenshot or add Google Sheet columns for {symbol}: current price, fair/intrinsic value and margin of safety."
        )

    fraud_level = str((fraud or {}).get("fraud_risk_level") or "").strip()
    if not fraud_level:
        missing_groups.append("Financial red-flag / fraud-risk inputs")
        fields.extend([
            "Receivables trend", "Inventory trend", "Operating cashflow", "Net profit trend",
            "Debt / leverage", "Auditor or notes red flags",
        ])
        instructions.append(
            f"Add red-flag data for {symbol} so the bot can check earnings quality, cashflow quality and balance-sheet risk."
        )

    if is_portfolio_symbol:
        qty = _to_float(holding_row.get("quantity") if holding_row is not None else None, None)
        avg = _to_float(holding_row.get("avg_buy") if holding_row is not None else None, None)
        mtm = _to_float(holding_row.get("mtm_price") if holding_row is not None else None, None)
        if qty is None or avg is None:
            missing_groups.append("Portfolio holding details")
            fields.extend(["Portfolio symbol", "Quantity", "Average buy price", "Current/MTM price"])
            instructions.append(
                f"For portfolio action on {symbol}, provide quantity and average buy price in the portfolio file/PDF."
            )
        elif mtm is None:
            missing_groups.append("Portfolio current price")
            fields.append("Current/MTM price")
            instructions.append(f"Current/MTM price for {symbol} was not in portfolio; bot will use latest market close when available.")

    # Remove duplicates while keeping order.
    def uniq(values: list[str]) -> list[str]:
        seen = set()
        out = []
        for value in values:
            key = str(value).strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(str(value).strip())
        return out

    missing_groups = uniq(missing_groups)
    fields = uniq(fields)
    instructions = uniq(instructions)

    if not missing_groups:
        return "Complete", "None", "No missing critical data detected for this symbol."
    return " | ".join(missing_groups), ", ".join(fields), " ".join(instructions)



def _recent_price_momentum_guard(frame: pd.DataFrame | None) -> dict[str, Any]:
    """Price-action guard to prevent false SELL calls on strong green days.

    Conservative rule:
    SELL/REDUCE is allowed only when there is confirmed bearish evidence.
    A green/up day with no breakdown should be Neutral/Hold, not Sell.
    """
    guard = {
        "latest_close": None,
        "prev_close": None,
        "day_change": None,
        "day_change_pct": None,
        "three_candle_change_pct": None,
        "rsi": None,
        "ema21": None,
        "ema51": None,
        "support_break": False,
        "ma_bearish": False,
        "confirmed_bearish": False,
        "positive_momentum": False,
        "neutral_or_bullish": False,
        "reason": "No price frame available.",
    }
    try:
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) < 5:
            return guard

        df = frame.copy()
        # normalize column casing
        ren = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc in {"open", "high", "low", "close", "volume"} and c != lc:
                ren[c] = lc
        if ren:
            df = df.rename(columns=ren)
        if "close" not in df.columns:
            return guard

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 5:
            return guard

        latest = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        day_change = latest - prev
        day_change_pct = (day_change / prev * 100) if prev else 0.0
        base3 = float(close.iloc[-4]) if len(close) >= 4 else prev
        three_pct = ((latest - base3) / base3 * 100) if base3 else 0.0

        ema21 = close.ewm(span=21, adjust=False, min_periods=5).mean()
        ema51 = close.ewm(span=51, adjust=False, min_periods=10).mean()
        e21 = float(ema21.iloc[-1]) if pd.notna(ema21.iloc[-1]) else None
        e51 = float(ema51.iloc[-1]) if pd.notna(ema51.iloc[-1]) else None

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if len(rsi_series.dropna()) else None

        # Support break: latest close below recent rolling support, excluding current candle.
        low_series = pd.to_numeric(df["low"], errors="coerce") if "low" in df.columns else close
        support_20 = float(low_series.iloc[-21:-1].min()) if len(low_series) >= 22 else float(low_series.iloc[:-1].min())
        support_break = latest < support_20

        ma_bearish = bool(e21 is not None and e51 is not None and latest < e21 and e21 < e51)
        positive_momentum = bool(day_change > 0 and day_change_pct >= 0.5)
        strong_positive = bool(day_change > 0 and (day_change_pct >= 1.0 or three_pct >= 2.0))
        neutral_or_bullish = bool(day_change >= 0 or latest >= (e21 if e21 is not None else latest))
        confirmed_bearish = bool(
            (day_change_pct <= -1.5 or three_pct <= -3.0)
            and (support_break or ma_bearish)
            and (rsi is None or rsi < 48)
        )

        if confirmed_bearish:
            reason = f"Confirmed bearish: day {day_change:+.2f} ({day_change_pct:+.2f}%), 3-candle {three_pct:+.2f}%, support_break={support_break}, MA bearish={ma_bearish}, RSI={rsi if rsi is not None else 'N/A'}."
        elif strong_positive:
            reason = f"Strong positive momentum: day {day_change:+.2f} ({day_change_pct:+.2f}%), 3-candle {three_pct:+.2f}%; no confirmed sell breakdown."
        elif neutral_or_bullish:
            reason = f"Neutral/bullish guard: day {day_change:+.2f} ({day_change_pct:+.2f}%), price not in confirmed bearish breakdown."
        else:
            reason = f"Weak but not confirmed sell: day {day_change:+.2f} ({day_change_pct:+.2f}%), 3-candle {three_pct:+.2f}%, support_break={support_break}, MA bearish={ma_bearish}."

        guard.update({
            "latest_close": latest,
            "prev_close": prev,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
            "three_candle_change_pct": three_pct,
            "rsi": rsi,
            "ema21": e21,
            "ema51": e51,
            "support_break": support_break,
            "ma_bearish": ma_bearish,
            "confirmed_bearish": confirmed_bearish,
            "positive_momentum": positive_momentum,
            "neutral_or_bullish": neutral_or_bullish,
            "reason": reason,
        })
        return guard
    except Exception as exc:
        guard["reason"] = f"Momentum guard failed: {exc}"
        return guard


def _apply_false_sell_guard(action: str, label: str, reason: str, *, has_holding: bool, guard: dict[str, Any], hazard_level: str, fraud_level: str) -> tuple[str, str, str, dict[str, Any]]:
    """Override false SELL/REDUCE calls when market action is neutral/bullish.

    Does not override hard red-flag cases: HIGH/CRITICAL fraud or CRITICAL price hazard.
    """
    value = str(action or "").upper()
    hard_risk = str(fraud_level or "").upper() in {"HIGH", "CRITICAL"} or str(hazard_level or "").upper() == "CRITICAL"
    sell_like = any(x in value for x in ["REDUCE", "TRIM", "SELL", "AVOID"])
    protected = False
    why_not_sell = ""

    if has_holding and sell_like and not hard_risk and not bool(guard.get("confirmed_bearish")):
        # Strong up/neutral price action should not be shown as Sell.
        if bool(guard.get("positive_momentum")) or bool(guard.get("neutral_or_bullish")):
            action = "HOLD / NEUTRAL WATCH"
            label = "False-Sell Guard"
            protected = True
            why_not_sell = "Sell/Reduce blocked because recent price action is neutral/bullish and no confirmed bearish breakdown exists."
            reason = f"{why_not_sell} {guard.get('reason', '')} Original model action was {value}. " + str(reason or "")

    diagnostics = {
        "sell_guard_applied": protected,
        "why_not_sell": why_not_sell,
        "momentum_guard_reason": guard.get("reason", ""),
        "recent_price_change": guard.get("day_change"),
        "recent_price_change_pct": guard.get("day_change_pct"),
        "three_candle_change_pct": guard.get("three_candle_change_pct"),
        "confirmed_bearish": guard.get("confirmed_bearish"),
        "support_break": guard.get("support_break"),
        "ma_bearish": guard.get("ma_bearish"),
    }
    return action, label, reason, diagnostics



def _normalized_bot_decision(action: str, has_holding: bool) -> str:
    """Simplify institutional labels into user-facing portfolio/buy decisions."""
    value = str(action or "").upper()
    if has_holding:
        if "NEUTRAL WATCH" in value:
            return "HOLD / NEUTRAL"
        if "REDUCE" in value or "TRIM" in value or value == "AVOID":
            return "SELL / REDUCE"
        if "TECHNICAL BASIS" in value or "FUNDAMENTALS MISSING" in value:
            if "AVERAGE" in value:
                return "AVERAGE CAREFULLY (TECHNICAL BASIS)"
            if "BUY MORE" in value:
                return "BUY MORE / HOLD (TECHNICAL BASIS)"
            return "HOLD (TECHNICAL BASIS)"
        if "DO NOT ADD" in value:
            return "HOLD"
        if "BUY MORE" in value:
            return "BUY MORE"
        if "AVERAGE" in value:
            return "AVERAGE CAREFULLY"
        return "HOLD"
    if "BUY CANDIDATE" in value:
        return "BUY"
    if "TECHNICAL BUY SETUP" in value:
        return "TECHNICAL BUY SETUP"
    if "TECHNICAL WATCH" in value:
        return "TECHNICAL WATCH"
    if "CONDITIONAL BUY" in value:
        return "WATCH FOR ENTRY"
    if "FUNDAMENTAL DATA REQUIRED" in value:
        return "DATA REQUIRED"
    if "AVOID" in value:
        return "AVOID"
    if "VALUATION" in value or "WATCH" in value:
        return "WATCH FOR ENTRY"
    return "WAIT"

def _portfolio_lookup(portfolio_df: pd.DataFrame) -> dict[str, pd.Series]:
    if portfolio_df is None or portfolio_df.empty:
        return {}
    return {str(row["symbol"]).upper(): row for _, row in portfolio_df.iterrows()}


def run_autopilot_cycle(
    symbols: Iterable[str],
    *,
    portfolio_df: pd.DataFrame | None = None,
    fundamentals_df: pd.DataFrame | None = None,
    max_symbols: int = 30,
    analysis_tf: str = "1d",
    execution_tf: str = "1d",
    period: str = "2y",
    prediction_horizon: int = 5,
    stop_atr: float = 2.0,
    target_rr: float = 3.0,
    discipline_rules: dict[str, Any] | None = None,
    auto_fetch_missing_fundamentals: bool = True,
) -> dict[str, Any]:
    """
    One autonomous research cycle:
    1) news/event snapshot for the watchlist
    2) per-symbol multi-engine analysis
    3) rule-gated decisions for holdings and new opportunities
    4) machine-readable tables + concise portfolio brief
    """
    discipline = {**DEFAULT_DISCIPLINE_RULES, **(discipline_rules or {})}
    selected = []
    for symbol in symbols:
        s = _clean_symbol(symbol)
        if s and s not in selected:
            selected.append(s)
    selected = selected[: max(1, int(max_symbols or 1))]
    if not selected:
        raise ValueError("Autopilot requires at least one PSX symbol.")

    portfolio = normalize_portfolio_columns(portfolio_df)
    portfolio_lookup = _portfolio_lookup(portfolio)
    fundamental_lookup, fundamentals_norm = _fundamental_lookup(fundamentals_df)
    internet_fundamental_logs = pd.DataFrame()
    if auto_fetch_missing_fundamentals:
        missing_for_internet = [s for s in selected if s not in fundamental_lookup]
        if missing_for_internet:
            try:
                internet_df, internet_fundamental_logs = fetch_internet_fundamentals(missing_for_internet)
                if internet_df is not None and not internet_df.empty:
                    combined_fundamentals = pd.concat(
                        [fundamentals_df if fundamentals_df is not None else pd.DataFrame(), internet_df],
                        ignore_index=True, sort=False
                    )
                    fundamental_lookup, fundamentals_norm = _fundamental_lookup(combined_fundamentals)
            except Exception as exc:
                internet_fundamental_logs = pd.DataFrame([
                    {"Symbol": s, "Internet Fundamental Status": "FAILED", "Internet Fundamental Note": str(exc)}
                    for s in missing_for_internet
                ])
    internet_fundamental_logs = _dedupe_for_lookup(internet_fundamental_logs, "Symbol") if isinstance(internet_fundamental_logs, pd.DataFrame) and not internet_fundamental_logs.empty else internet_fundamental_logs
    internet_log_lookup = (
        _safe_to_dict_index(internet_fundamental_logs, "Symbol") if isinstance(internet_fundamental_logs, pd.DataFrame) and not internet_fundamental_logs.empty else {}
    )
    fundamentals_norm = _dedupe_for_lookup(fundamentals_norm, "symbol") if isinstance(fundamentals_norm, pd.DataFrame) and not fundamentals_norm.empty else fundamentals_norm
    fundamentals_row_lookup = (
        _safe_to_dict_index(fundamentals_norm, "symbol") if isinstance(fundamentals_norm, pd.DataFrame) and not fundamentals_norm.empty else {}
    )

    news = build_news_event_risk_snapshot(selected)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for symbol in selected:
        try:
            higher = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
            lower = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher,
                lower_df=lower,
                asset_class="Stock",
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                risk_context={"high_impact_news": news.get("risk_level") in {"HIGH", "MODERATE"}},
            )
            pro = evaluate_psx_pro_score(result)
            risk = build_risk_warning(
                result,
                pro,
                user_event_risk=news.get("risk_level") in {"HIGH", "MODERATE"},
                benchmark_conflict=False,
            )
            hazard = detect_price_hazards(
                result["execution_frame"],
                symbol=symbol,
                support_resistance=result.get("support_resistance", {}),
            )
            prediction = run_prediction_engine(
                result["execution_frame"],
                bias=result["signal"]["bias"],
                horizon=int(prediction_horizon),
                stop_atr=float(stop_atr),
                target_rr=float(target_rr),
                risk_severity=risk.get("risk_severity", "LOW"),
            )
            fib = fibonacci_retracement_confluence(result["execution_frame"], result["higher_trend"]["trend"])
            secular = classify_secular_trend(result.get("higher_frame", higher))
            confluence = build_confluence_score(
                result=result,
                pro=pro,
                prediction=prediction,
                fib=fib,
                secular=secular,
                hazard=hazard,
                news=news,
                trading_style="Position / Long-Term",
            )

            fundamental = fundamental_lookup.get(symbol, {})
            raw_funda = fundamentals_row_lookup.get(symbol)
            row_series = pd.Series(raw_funda) if raw_funda is not None else None
            intrinsic = _safe_intrinsic_value(row_series)
            fraud = _safe_fraud_score(row_series)

            holding = portfolio_lookup.get(symbol)
            last_close = _to_float(pro.get("latest_metrics", {}).get("close"), None)
            if holding is not None:
                holding = holding.copy()
                use_price = _to_float(holding.get("mtm_price"), None)
                if use_price is None:
                    holding["mtm_price"] = last_close
                    use_price = last_close
                qty = _to_float(holding.get("quantity"), None)
                avg = _to_float(holding.get("avg_buy"), None)
                if qty is not None and avg is not None and use_price is not None and avg != 0:
                    holding["investment"] = qty * avg
                    holding["mtm_total"] = qty * use_price
                    holding["pnl"] = holding["mtm_total"] - holding["investment"]
                    holding["pnl_pct"] = ((use_price - avg) / avg) * 100

            autopilot_action, autopilot_label, autopilot_reason, decision_diag = _autopilot_decision(
                symbol=symbol,
                confluence=confluence,
                pro=pro,
                hazard=hazard,
                news=news,
                fundamental=fundamental,
                intrinsic=intrinsic,
                fraud=fraud,
                holding_row=holding,
                discipline=discipline,
            )

            momentum_guard = _recent_price_momentum_guard(result.get("execution_frame"))
            autopilot_action, autopilot_label, autopilot_reason, guard_diag = _apply_false_sell_guard(
                autopilot_action,
                autopilot_label,
                autopilot_reason,
                has_holding=bool(holding is not None),
                guard=momentum_guard,
                hazard_level=hazard.get("hazard_level"),
                fraud_level=(fraud or {}).get("fraud_risk_level"),
            )
            decision_diag.update(guard_diag)

            plan = result.get("trade_plan", {})
            latest = pro.get("latest_metrics", {})
            execution_levels = _build_execution_levels(result, plan, result.get("signal", {}).get("bias"))
            decision_diag["data_completeness_pct"] = _data_completeness_pct(fundamental, intrinsic, fraud)
            internet_log = internet_log_lookup.get(symbol, {})
            missing_data_status, required_data_fields, missing_data_request = _missing_data_request(
                symbol=symbol,
                fundamental=fundamental,
                intrinsic=intrinsic,
                fraud=fraud,
                holding_row=holding,
                is_portfolio_symbol=bool(holding is not None),
            )
            bot_decision = _normalized_bot_decision(autopilot_action, bool(holding is not None))
            # Missing fundamentals no longer stops the table entirely. The bot first tries internet lookup;
            # if it still cannot find data, it gives a clearly labeled technical-basis answer.
            if missing_data_status != "Complete" and bot_decision in {"BUY", "BUY MORE", "AVERAGE CAREFULLY"}:
                bot_decision = "TECHNICAL BASIS - DATA PARTIAL"
            decision_basis = "Technical + Fundamental + Valuation"
            if _to_float((fundamental or {}).get("fundamental_score_pct"), None) is None:
                decision_basis = "Technical Basis Only - fundamentals missing/not found online"
            elif str(missing_data_status) != "Complete":
                decision_basis = "Partial Fundamentals + Technical Basis"
            knowledge_tags = decision_knowledge_tags(
                confluence=confluence, hazard=hazard, news=news, fundamental=fundamental,
                intrinsic=intrinsic, fraud=fraud, secular=secular,
            )

            current_price = use_price if holding is not None and use_price is not None else last_close
            current_price_source = "Portfolio MTM price" if holding is not None and _to_float(holding.get("mtm_price"), None) is not None else "Latest market/chart close"

            rows.append({
                "Symbol": symbol,
                "Current Price": current_price,
                "Current Price Source": current_price_source,
                "Portfolio Holding": bool(holding is not None),
                "Autopilot Action": autopilot_action,
                "Bot Decision": bot_decision,
                "Decision Label": autopilot_label,
                "Decision Reason": autopilot_reason,
                "Decision Scorecard": autopilot_reason.split("Full scorecard: ", 1)[1] if "Full scorecard: " in autopilot_reason else autopilot_reason,
                "Decision Confidence": decision_diag.get("decision_confidence"),
                "Decision Confidence %": decision_diag.get("decision_confidence_score"),
                "Sell Guard Applied": decision_diag.get("sell_guard_applied"),
                "Why Not Sell": decision_diag.get("why_not_sell"),
                "Momentum Guard": decision_diag.get("momentum_guard_reason"),
                "Recent Price Change": decision_diag.get("recent_price_change"),
                "Recent Price Change %": decision_diag.get("recent_price_change_pct"),
                "3-Candle Change %": decision_diag.get("three_candle_change_pct"),
                "Confirmed Bearish": decision_diag.get("confirmed_bearish"),
                "Support Break": decision_diag.get("support_break"),
                "MA Bearish": decision_diag.get("ma_bearish"),
                "Technical Status": decision_diag.get("technical_status"),
                "Fundamental Data Status": decision_diag.get("fundamental_data_status"),
                "Fundamental Source": "Internet fallback" if str(internet_log.get("Internet Fundamental Status", "")).upper() == "FOUND" else "Uploaded/Sheet/Image" if _to_float((fundamental or {}).get("fundamental_score_pct"), None) is not None else "Not found",
                "Internet Fundamental Status": internet_log.get("Internet Fundamental Status", "Not needed" if symbol in fundamental_lookup else "Not found"),
                "Internet Fundamental Note": internet_log.get("Internet Fundamental Note", ""),
                "Decision Basis": decision_basis,
                "Valuation Status": decision_diag.get("valuation_status"),
                "Data Completeness %": decision_diag.get("data_completeness_pct"),
                "Missing Data Status": missing_data_status,
                "Required Data Fields": required_data_fields,
                "Missing Data Request": missing_data_request,
                "Knowledge Applied": knowledge_tags,
                "Confluence Score": confluence.get("confluence_score"),
                "Confluence Grade": confluence.get("confluence_grade"),
                "Confluence Verdict": confluence.get("confluence_verdict"),
                "Signal Bias": result.get("signal", {}).get("bias"),
                "Signal Action": result.get("signal", {}).get("action"),
                "Signal Confidence": result.get("signal", {}).get("confidence"),
                "PRO Score": pro.get("pro_score"),
                "PRO Grade": pro.get("pro_grade"),
                "Trade Quality": pro.get("trade_quality"),
                "Hazard Level": hazard.get("hazard_level"),
                "News/Event Risk": news.get("risk_level"),
                "Symbol Event Rows": _event_count_for_symbol(news, symbol),
                "Prediction Verdict": prediction.get("prediction_verdict"),
                "Prediction Expected Return %": prediction.get("expected_return_pct"),
                "Prediction Up Probability %": prediction.get("probability_up_pct"),
                "Secular Bias": secular.get("secular_bias"),
                "Secular Score": secular.get("score"),
                "Fundamental Grade": fundamental.get("fundamental_grade"),
                "Fundamental Score %": fundamental.get("fundamental_score_pct"),
                "Valuation Grade": intrinsic.get("valuation_grade"),
                "Margin of Safety %": intrinsic.get("best_margin_of_safety_pct"),
                "Fraud Risk": fraud.get("fraud_risk_level"),
                "Entry": execution_levels.get("Entry"),
                "Entry Type": execution_levels.get("Entry Type"),
                "Stop Loss": execution_levels.get("Stop Loss"),
                "TP1": execution_levels.get("TP1"),
                "TP2": execution_levels.get("TP2"),
                "TP3": execution_levels.get("TP3"),
                "Take Profit": plan.get("take_profit"),
                "Last Close": latest.get("close"),
                "Market Current Price": current_price,
                "RSI": latest.get("rsi"),
                "ADX": latest.get("adx"),
                "Volume Ratio": latest.get("volume_ratio"),
                "Portfolio Qty": None if holding is None else holding.get("quantity"),
                "Portfolio Avg Buy": None if holding is None else holding.get("avg_buy"),
                "Portfolio MTM Price": None if holding is None else holding.get("mtm_price"),
                "Portfolio P&L": None if holding is None else holding.get("pnl"),
                "Portfolio P&L %": None if holding is None else holding.get("pnl_pct"),
            })
        except Exception as exc:
            failures.append({
                "Symbol": symbol,
                "Error": str(exc),
                "Missing Data Status": "Market/technical data unavailable",
                "Required Data Fields": "OHLCV data: date, open, high, low, close, volume",
                "Missing Data Request": f"Upload price history CSV/XLSX for {symbol} or check internet/data-source access. Required columns: date, open, high, low, close, volume.",
            })

    decisions = pd.DataFrame(rows)
    if not decisions.empty:
        decisions = decisions.sort_values(
            ["Portfolio Holding", "Confluence Score", "PRO Score", "Symbol"],
            ascending=[False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)

    failures_df = pd.DataFrame(failures)
    holdings = decisions[decisions["Portfolio Holding"] == True].copy() if not decisions.empty else pd.DataFrame()
    opportunities = decisions[decisions["Portfolio Holding"] == False].copy() if not decisions.empty else pd.DataFrame()

    decision_counts = decisions["Autopilot Action"].value_counts(dropna=False).to_dict() if not decisions.empty else {}
    action_text = ", ".join(f"{k}: {v}" for k, v in decision_counts.items()) if decision_counts else "No completed decisions"
    market_brief = pd.DataFrame([
        {
            "Symbols Requested": len(selected),
            "Symbols Completed": int(len(decisions)),
            "Failures": int(len(failures_df)),
            "News/Event Risk": news.get("risk_level"),
            "News/Event Notes": " | ".join(news.get("risk_notes", [])),
            "Action Mix": action_text,
            "Average Confluence": round(float(pd.to_numeric(decisions.get("Confluence Score"), errors="coerce").mean()), 2) if not decisions.empty else None,
            "High Hazard Count": int(decisions["Hazard Level"].isin(["HIGH", "CRITICAL"]).sum()) if not decisions.empty else 0,
        }
    ])

    return {
        "decisions": decisions,
        "holdings": holdings,
        "opportunities": opportunities,
        "failures": failures_df,
        "market_brief": market_brief,
        "news": news,
        "portfolio": portfolio,
        "knowledge_registry": knowledge_registry_table(),
        "engine_coverage": engine_coverage_table(),
        "knowledge_summary": pd.DataFrame([knowledge_status_summary()]),
    }


def persist_autopilot_outputs(bundle: dict[str, Any], output_dir: Path | None = None) -> dict[str, Path]:
    target = output_dir or DATA_DIR
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "decisions": target / "autopilot_decisions.csv",
        "holdings": target / "autopilot_holdings_actions.csv",
        "opportunities": target / "autopilot_opportunities.csv",
        "failures": target / "autopilot_failures.csv",
        "market_brief": target / "autopilot_market_brief.csv",
        "summary": target / "autopilot_latest_summary.md",
        "knowledge_registry": target / "autopilot_knowledge_registry.csv",
        "engine_coverage": target / "autopilot_engine_coverage.csv",
        "knowledge_summary": target / "autopilot_knowledge_summary.csv",
        "internet_fundamental_logs": target / "autopilot_internet_fundamental_logs.csv",
    }

    for key in ["decisions", "holdings", "opportunities", "failures", "market_brief", "knowledge_registry", "engine_coverage", "knowledge_summary", "internet_fundamental_logs"]:
        df = bundle.get(key)
        (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_csv(paths[key], index=False)

    decisions = bundle.get("decisions")
    market_brief = bundle.get("market_brief")
    summary_lines = [
        "# PSX Autopilot Latest Summary",
        "",
        "This report is generated by the autonomous research cycle. It does not place broker orders.",
        "",
    ]
    if isinstance(market_brief, pd.DataFrame) and not market_brief.empty:
        row = market_brief.iloc[0]
        summary_lines.extend([
            f"- Symbols completed: {row.get('Symbols Completed')} / {row.get('Symbols Requested')}",
            f"- News/Event risk: {row.get('News/Event Risk')}",
            f"- Average confluence: {row.get('Average Confluence')}",
            f"- Action mix: {row.get('Action Mix')}",
            "",
        ])
    if isinstance(decisions, pd.DataFrame) and not decisions.empty:
        summary_lines.append("## Top decisions")
        for _, row in decisions.head(12).iterrows():
            summary_lines.append(
                f"- **{row.get('Symbol')}** — {row.get('Autopilot Action')} | Confluence {row.get('Confluence Score')} | {row.get('Decision Label')}"
            )
    paths["summary"].write_text("\n".join(summary_lines), encoding="utf-8")
    return paths
