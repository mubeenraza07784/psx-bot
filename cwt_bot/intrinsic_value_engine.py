from __future__ import annotations

from typing import Any, Dict, List
import math
import numpy as np
import pandas as pd


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _clamp_growth(g: float | None, low: float = 5.0, high: float = 20.0, default: float = 9.0) -> float:
    if g is None or math.isnan(g):
        return default
    return max(low, min(high, float(g)))


def _discounted_stream(base: float, growth_pct: float, discount_pct: float, years: int) -> float:
    value = 0.0
    growth = growth_pct / 100.0
    discount = discount_pct / 100.0
    for year in range(1, years + 1):
        future = base * ((1 + growth) ** year)
        value += future / ((1 + discount) ** year)
    return value


def dcf_per_share_value(
    base_per_share: float | None,
    growth_pct: float | None,
    discount_rate_pct: float = 12.0,
    terminal_growth_pct: float = 4.0,
    growth_years: int = 10,
    terminal_years: int = 10,
) -> Dict[str, Any]:
    if base_per_share is None or base_per_share <= 0:
        return {"value": None, "status": "Unavailable", "warning": "Base per-share value is missing or non-positive."}

    g = _clamp_growth(growth_pct)
    stage1 = _discounted_stream(base_per_share, g, discount_rate_pct, growth_years)

    last_growth_stage = base_per_share * ((1 + g / 100.0) ** growth_years)
    stage2 = 0.0
    for year in range(1, terminal_years + 1):
        future = last_growth_stage * ((1 + terminal_growth_pct / 100.0) ** year)
        stage2 += future / ((1 + discount_rate_pct / 100.0) ** (growth_years + year))

    value = stage1 + stage2
    warning = None
    if growth_pct is not None and (growth_pct < 5 or growth_pct > 20):
        warning = "Growth assumption was capped to the 5%–20% range."
    return {
        "value": round(value, 4),
        "status": "OK",
        "growth_used_pct": round(g, 2),
        "discount_rate_pct": discount_rate_pct,
        "terminal_growth_pct": terminal_growth_pct,
        "warning": warning,
    }


def projected_fcf_value(
    six_year_avg_fcf_per_share: float | None,
    equity_per_share: float | None,
    growth_pct: float | None,
    growth_multiple_years: int = 6,
    equity_weight: float = 0.80,
) -> Dict[str, Any]:
    if six_year_avg_fcf_per_share is None or equity_per_share is None:
        return {"value": None, "status": "Unavailable", "warning": "Projected FCF inputs are missing."}
    g = _clamp_growth(growth_pct, low=4.0, high=15.0, default=9.0)
    growth_multiple = (1 + g / 100.0) ** growth_multiple_years
    value = (growth_multiple * six_year_avg_fcf_per_share) + (equity_weight * equity_per_share)
    return {
        "value": round(value, 4),
        "status": "OK",
        "growth_used_pct": round(g, 2),
        "growth_multiple": round(growth_multiple, 4),
        "warning": None,
    }


def peter_lynch_value(
    eps: float | None,
    growth_pct: float | None,
    peg_ratio: float | None = 1.0,
) -> Dict[str, Any]:
    if eps is None or growth_pct is None:
        return {"value": None, "status": "Unavailable", "warning": "EPS or growth assumption is missing."}
    g = _clamp_growth(growth_pct)
    peg = 1.0 if peg_ratio is None or peg_ratio <= 0 else float(peg_ratio)
    value = peg * g * eps
    return {
        "value": round(value, 4),
        "status": "OK",
        "growth_used_pct": round(g, 2),
        "peg_used": round(peg, 2),
        "warning": None,
    }


def _margin_of_safety(price: float | None, value: float | None) -> float | None:
    if price is None or value is None or value == 0:
        return None
    return round(((value - price) / value) * 100, 2)


def intrinsic_value_composite(row: pd.Series, assumptions: Dict[str, Any] | None = None) -> Dict[str, Any]:
    assumptions = assumptions or {}
    discount = float(assumptions.get("discount_rate_pct", 12.0))
    terminal_growth = float(assumptions.get("terminal_growth_pct", 4.0))
    growth_years = int(assumptions.get("growth_years", 10))
    terminal_years = int(assumptions.get("terminal_years", 10))

    price = _num(row.get("price"))
    eps = _num(row.get("eps"))
    fcf = _num(row.get("fcf"))
    shares = _num(row.get("shares_outstanding"))
    cash = _num(row.get("cash"))
    equity = _num(row.get("equity"))
    cash_per_share = _num(row.get("cash_per_share"))
    fcf_per_share = None
    if fcf is not None and shares not in (None, 0):
        fcf_per_share = fcf / shares
    if cash_per_share is None and cash is not None and shares not in (None, 0):
        cash_per_share = cash / shares
    equity_per_share = equity / shares if equity is not None and shares not in (None, 0) else _num(row.get("book_value_per_share"))

    growth_pct = _num(row.get("growth_rate"))
    if growth_pct is None:
        growth_pct = _num(row.get("ebitda_growth_avg"))
    if growth_pct is None:
        growth_pct = _num(row.get("eps_growth_avg"))

    fcf_ps_6y = _num(row.get("fcf_per_share_5y"))
    if fcf_ps_6y is None:
        fcf_ps_6y = fcf_per_share

    dcf_fcf = dcf_per_share_value(fcf_per_share, growth_pct, discount, terminal_growth, growth_years, terminal_years)
    dcf_eps = dcf_per_share_value(eps, growth_pct, discount, terminal_growth, growth_years, terminal_years)
    dcf_cash = dcf_per_share_value(cash_per_share, growth_pct, discount, terminal_growth, growth_years, terminal_years)
    projected_fcf = projected_fcf_value(fcf_ps_6y, equity_per_share, growth_pct)
    projected_cash = projected_fcf_value(cash_per_share, equity_per_share, growth_pct)
    lynch = peter_lynch_value(eps, growth_pct, peg_ratio=_num(row.get("peg")) or 1.0)

    methods = {
        "DCF FCF": dcf_fcf,
        "DCF EPS": dcf_eps,
        "DCF Cash": dcf_cash,
        "Projected FCF": projected_fcf,
        "Projected Cash": projected_cash,
        "Peter Lynch": lynch,
    }

    fcf_values = [m.get("value") for key, m in methods.items() if key in {"DCF FCF", "DCF EPS", "Projected FCF", "Peter Lynch"} and m.get("value") is not None]
    cash_values = [m.get("value") for key, m in methods.items() if key in {"DCF Cash", "DCF EPS", "Projected Cash", "Peter Lynch"} and m.get("value") is not None]
    composite_fcf = round(float(np.mean(fcf_values)), 4) if fcf_values else None
    composite_cash = round(float(np.mean(cash_values)), 4) if cash_values else None
    mos_fcf = _margin_of_safety(price, composite_fcf)
    mos_cash = _margin_of_safety(price, composite_cash)

    best_mos = max([v for v in [mos_fcf, mos_cash] if v is not None], default=None)
    if best_mos is None:
        grade = "Unavailable"
        verdict = "Inputs missing"
    elif best_mos >= 25:
        grade = "Attractive"
        verdict = "Margin of safety ≥25%"
    elif best_mos >= 0:
        grade = "Fair / Review"
        verdict = "Positive but limited margin of safety"
    else:
        grade = "Overvalued"
        verdict = "Intrinsic value below current price"

    warnings: List[str] = []
    if fcf_per_share is not None and fcf_per_share <= 0:
        warnings.append("FCF per share is non-positive; DCF-FCF reliability is lower.")
    if growth_pct is not None and (growth_pct < 5 or growth_pct > 20):
        warnings.append("Growth input was capped in the valuation model.")
    if len(fcf_values) < 3:
        warnings.append("FCF-based composite uses fewer than 3 valid methods.")
    if len(cash_values) < 3:
        warnings.append("Cash-based composite uses fewer than 3 valid methods.")

    method_rows = []
    for name, payload in methods.items():
        method_rows.append({
            "Method": name,
            "Value": payload.get("value"),
            "Status": payload.get("status"),
            "Warning": payload.get("warning"),
        })

    return {
        "current_price": price,
        "composite_intrinsic_value_fcf": composite_fcf,
        "margin_of_safety_fcf_pct": mos_fcf,
        "composite_intrinsic_value_cash": composite_cash,
        "margin_of_safety_cash_pct": mos_cash,
        "best_margin_of_safety_pct": best_mos,
        "valuation_grade": grade,
        "valuation_verdict": verdict,
        "warnings": warnings,
        "methods_table": pd.DataFrame(method_rows),
        "assumptions": {
            "discount_rate_pct": discount,
            "terminal_growth_pct": terminal_growth,
            "growth_years": growth_years,
            "terminal_years": terminal_years,
            "growth_input_pct": growth_pct,
        },
    }
