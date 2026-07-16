from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def _direction_score(direction: str, positive_if: str) -> int:
    d = str(direction or "").strip().lower()
    if d == positive_if.lower():
        return 1
    if d in {"stable", "neutral", "unchanged", "mixed"}:
        return 0
    return -1


def macro_checklist_score(
    inflation_status: str,
    policy_rate_direction: str,
    bond_yield_direction: str,
    current_account_direction: str,
    fiscal_account_direction: str,
    reserves_direction: str,
    currency_direction: str,
    oil_direction: str,
    market_pe: float | None = None,
    earnings_yield: float | None = None,
    bond_yield: float | None = None,
    global_risk_tone: str = "Neutral",
) -> Dict[str, Any]:
    """
    Macro checklist inspired by the uploaded Week 12 economy/stocks checklist and Pakistan profile PDFs.
    """
    rows: List[Dict[str, Any]] = []
    score = 0

    def add(area: str, status: str, impact: int, comment: str):
        nonlocal score
        score += impact
        label = "Positive" if impact > 0 else ("Neutral" if impact == 0 else "Negative")
        rows.append({"Area": area, "Status": status, "Impact": label, "Score": impact, "Comment": comment})

    inflation_l = str(inflation_status).lower()
    if "within" in inflation_l or "target" in inflation_l:
        add("Inflation", inflation_status, 1, "Inflation appears within/near policy target.")
    elif "above" in inflation_l or "rising" in inflation_l:
        add("Inflation", inflation_status, -1, "Inflation pressure may keep policy restrictive.")
    else:
        add("Inflation", inflation_status, 0, "Inflation status requires interpretation.")

    rate_score = _direction_score(policy_rate_direction, "falling")
    add("Interest Rates", policy_rate_direction, rate_score, "Falling rates are usually more equity-supportive; rising rates are restrictive.")

    by_score = _direction_score(bond_yield_direction, "falling")
    add("Bond Yields / T-Bills", bond_yield_direction, by_score, "Falling yields generally improve relative equity attractiveness.")

    cad_score = _direction_score(current_account_direction, "improving")
    add("Current Account", current_account_direction, cad_score, "Improving current account supports external stability.")

    fiscal_score = _direction_score(fiscal_account_direction, "improving")
    add("Fiscal Account", fiscal_account_direction, fiscal_score, "Improving fiscal position reduces macro pressure.")

    reserves_score = _direction_score(reserves_direction, "improving")
    add("Central Bank Reserves", reserves_direction, reserves_score, "Improving reserves support currency and confidence.")

    currency_l = str(currency_direction).lower()
    if currency_l in {"stable", "improving", "appreciating"}:
        add("Currency / PKR", currency_direction, 1, "Stable currency is broadly supportive for macro confidence.")
    elif currency_l in {"weakening", "depreciating", "volatile"}:
        add("Currency / PKR", currency_direction, -1, "Weak or volatile currency raises importer and inflation risk.")
    else:
        add("Currency / PKR", currency_direction, 0, "Currency impact is mixed.")

    oil_l = str(oil_direction).lower()
    if oil_l in {"falling", "stable low"}:
        add("Oil", oil_direction, 1, "Lower oil is generally positive for Pakistan's external balance.")
    elif oil_l in {"rising", "spiking"}:
        add("Oil", oil_direction, -1, "Higher oil can worsen inflation and import bill.")
    else:
        add("Oil", oil_direction, 0, "Oil impact is mixed.")

    global_l = str(global_risk_tone).lower()
    if "risk-on" in global_l or "supportive" in global_l:
        add("Global Risk Tone", global_risk_tone, 1, "Risk-on conditions are supportive for equities.")
    elif "risk-off" in global_l or "defensive" in global_l:
        add("Global Risk Tone", global_risk_tone, -1, "Risk-off conditions warrant defensive posture.")
    else:
        add("Global Risk Tone", global_risk_tone, 0, "Global tone is neutral/mixed.")

    if earnings_yield is not None and bond_yield is not None:
        spread = earnings_yield - bond_yield
        if spread > 0:
            add("Earnings Yield vs Bond Yield", f"{spread:.2f}% spread", 1, "Equities offer higher earnings yield than bonds.")
        elif spread > -2:
            add("Earnings Yield vs Bond Yield", f"{spread:.2f}% spread", 0, "Relative valuation is not clearly attractive.")
        else:
            add("Earnings Yield vs Bond Yield", f"{spread:.2f}% spread", -1, "Fixed income may be more attractive than equities.")

    if market_pe is not None:
        if market_pe <= 8:
            add("Market P/E", f"{market_pe:.2f}", 1, "Market valuation appears low on an absolute basis.")
        elif market_pe <= 12:
            add("Market P/E", f"{market_pe:.2f}", 0, "Market valuation appears moderate.")
        else:
            add("Market P/E", f"{market_pe:.2f}", -1, "Market valuation appears less attractive.")

    if score >= 6:
        regime = "Macro Bullish / Risk-On"
    elif score >= 2:
        regime = "Macro Constructive"
    elif score >= -1:
        regime = "Macro Neutral / Mixed"
    elif score >= -5:
        regime = "Macro Defensive"
    else:
        regime = "Macro High-Risk / Risk-Off"

    if "Bullish" in regime:
        action = "Equity exposure can be more constructive, while still respecting stock-level risk."
    elif "Constructive" in regime:
        action = "Selective equity exposure favored; prioritize stronger sectors and companies."
    elif "Neutral" in regime:
        action = "Use selective stock-picking and avoid over-aggressive allocation."
    elif "Defensive" in regime:
        action = "Favor defensives, quality balance sheets, dividends, and tighter risk controls."
    else:
        action = "Capital preservation is priority; avoid forcing risk-on trades."

    return {
        "macro_score": score,
        "macro_regime": regime,
        "quick_action": action,
        "checklist_table": pd.DataFrame(rows),
    }


def company_macro_sensitivity(
    sector: str,
    export_oriented: bool,
    import_dependent: bool,
    interest_rate_sensitive: str = "Medium",
    oil_sensitive: str = "Medium",
) -> pd.DataFrame:
    sector_l = str(sector or "").lower()
    rows = []

    def add(factor: str, impact: str, reason: str):
        rows.append({"Macro Factor": factor, "Company Impact": impact, "Reason": reason})

    if export_oriented:
        add("PKR Depreciation", "Positive", "Export revenue may translate into higher PKR sales.")
    elif import_dependent:
        add("PKR Depreciation", "Negative", "Imported inputs may raise costs.")
    else:
        add("PKR Depreciation", "Mixed", "Currency effect depends on input/output mix.")

    ir = str(interest_rate_sensitive).lower()
    if "high" in ir:
        add("Interest Rate Hikes", "Negative", "Finance cost or demand sensitivity is high.")
    elif "low" in ir:
        add("Interest Rate Hikes", "Limited", "Business is less rate-sensitive.")
    else:
        add("Interest Rate Hikes", "Moderate", "Some earnings or demand sensitivity is possible.")

    oil = str(oil_sensitive).lower()
    if "high" in oil and "oil" not in sector_l:
        add("Oil Price Rise", "Negative", "Energy/input costs may rise.")
    elif "oil" in sector_l:
        add("Oil Price Rise", "Positive/Mixed", "E&P may benefit; OMC/refining effects vary.")
    else:
        add("Oil Price Rise", "Mixed", "Indirect input-cost effect may matter.")

    if "bank" in sector_l:
        add("Rate Cuts", "Mixed", "Rate cuts may pressure spreads but support credit demand.")
    elif "cement" in sector_l or "auto" in sector_l:
        add("Rate Cuts", "Positive", "Lower rates may support construction/consumer demand.")
    elif "it" in sector_l or "technology" in sector_l:
        add("Rate Cuts", "Positive/Mixed", "Growth valuation may benefit, while export FX remains important.")
    else:
        add("Rate Cuts", "Mixed", "Sector-specific effect requires review.")

    return pd.DataFrame(rows)
