from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


PROFILE_MAP = {
    "Profile 1 — Pension / 20-Year Passive": {
        "label": "Profile 1",
        "style": "Pension / 20-Year Passive",
        "best_for": "No time to monitor markets for many years.",
        "modules": ["Pension / mutual fund selection", "Asset allocation", "Macro regime check"],
        "default_strategy": "Long-term fund allocation",
    },
    "Profile 2 — Passive Trader 1": {
        "label": "Profile 2",
        "style": "Passive Trader 1",
        "best_for": "Delegates investment management to funds or professionals.",
        "modules": ["Mutual fund scoring", "Interest-rate cycle", "Asset allocation"],
        "default_strategy": "Fund manager / mutual fund rotation",
    },
    "Profile 3 — Passive Trader 2": {
        "label": "Profile 3",
        "style": "Passive Trader 2",
        "best_for": "Buy-and-hold investor who can rotate sectors periodically.",
        "modules": ["Macro sector rotation", "Strategy 1/2", "Fundamentals"],
        "default_strategy": "Quality buy-and-hold",
    },
    "Profile 4 — Active Trader 1": {
        "label": "Profile 4",
        "style": "Active Trader 1",
        "best_for": "Monitors macro and interest-rate cycles and shifts allocation.",
        "modules": ["Macro checklist", "Interest-rate colors", "Sector rotation"],
        "default_strategy": "Macro rotation",
    },
    "Profile 5 — Active Trader 2": {
        "label": "Profile 5",
        "style": "Active Trader 2",
        "best_for": "Works on company fundamentals, business analysis, and sector selection.",
        "modules": ["Master fundamental score", "Intrinsic value", "Fraud-risk scanner"],
        "default_strategy": "Fundamental stock selection",
    },
    "Profile 6 — Swing Trader": {
        "label": "Profile 6",
        "style": "Swing Trader",
        "best_for": "Combines fundamentals with technical entries/exits.",
        "modules": ["Fundamentals", "CWT / MTF scenarios", "Prediction & loss control", "Trade tracker"],
        "default_strategy": "Fundamental + technical swing",
    },
    "Profile 7 — Day Trader": {
        "label": "Profile 7",
        "style": "Day Trader",
        "best_for": "Uses news, policy tone, seasonality, and TA for daily trades.",
        "modules": ["News/event watchtower", "Price hazards", "Scalping/intraday desk"],
        "default_strategy": "News-aware intraday trading",
    },
}


def recommend_profile(
    horizon_years: float,
    monitoring_frequency: str,
    prefers_funds: bool,
    uses_fundamentals: bool,
    uses_technicals: bool,
    trades_intraday: bool,
) -> Dict[str, Any]:
    """
    Heuristic mapping of user responses to the 7 investor profiles in the uploaded Week 12 file.
    """
    monitoring = monitoring_frequency.lower()

    if horizon_years >= 15 and monitoring in {"rarely", "annual"} and prefers_funds:
        key = "Profile 1 — Pension / 20-Year Passive"
    elif prefers_funds and monitoring in {"rarely", "annual", "quarterly"}:
        key = "Profile 2 — Passive Trader 1"
    elif horizon_years >= 5 and not uses_technicals and monitoring in {"quarterly", "monthly"}:
        key = "Profile 3 — Passive Trader 2"
    elif not uses_technicals and monitoring in {"monthly", "weekly"} and not trades_intraday:
        key = "Profile 4 — Active Trader 1"
    elif uses_fundamentals and not uses_technicals and not trades_intraday:
        key = "Profile 5 — Active Trader 2"
    elif uses_fundamentals and uses_technicals and not trades_intraday:
        key = "Profile 6 — Swing Trader"
    elif trades_intraday:
        key = "Profile 7 — Day Trader"
    else:
        key = "Profile 6 — Swing Trader" if uses_technicals else "Profile 5 — Active Trader 2"

    profile = PROFILE_MAP[key].copy()
    profile["profile_key"] = key
    return profile


def profile_catalog() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, data in PROFILE_MAP.items():
        rows.append({
            "Profile": data["label"],
            "Style": data["style"],
            "Best For": data["best_for"],
            "Default Strategy": data["default_strategy"],
            "Priority Modules": ", ".join(data["modules"]),
        })
    return pd.DataFrame(rows)
