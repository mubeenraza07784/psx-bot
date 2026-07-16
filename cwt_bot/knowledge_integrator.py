from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


# Complete integration registry for the uploaded PSX learning material, spreadsheets,
# earlier bot architecture, and the autonomous terminal modules.
# Each record maps an uploaded knowledge source to the engine that uses it.
KNOWLEDGE_REGISTRY: List[Dict[str, Any]] = [
    {
        "Category": "Technical Patterns",
        "Uploaded Knowledge": "Candlestick Patterns and Fibonacci",
        "Bot Integration": "Candlestick pattern engine + Fibonacci retracement confluence",
        "Modules": "patterns_candles.py, fib_engine.py, confluence_engine.py",
        "Autonomous Use": "Improves entry timing and confirms demand/supply reversal zones.",
        "Status": "Integrated",
    },
    {
        "Category": "Trading Framework",
        "Uploaded Knowledge": "Stocks (CWT), trend filtering, news impact, RRMS",
        "Bot Integration": "CWT structure, trend context, risk/reward, position planning",
        "Modules": "cwt.py, structures.py, risk.py, trade_plan.py, signals.py",
        "Autonomous Use": "Prevents random entries and forces trend/risk alignment.",
        "Status": "Integrated",
    },
    {
        "Category": "Market Regime",
        "Uploaded Knowledge": "Secular Bear Market / hedge-fund manager framework",
        "Bot Integration": "Long-cycle market regime and defensive gating",
        "Modules": "secular_engine.py, market_hazard.py, event_risk_monitor.py",
        "Autonomous Use": "Reduces aggressive entries during weak long-cycle conditions.",
        "Status": "Integrated",
    },
    {
        "Category": "Bank Fundamentals",
        "Uploaded Knowledge": "Financial Analysis of a Bank",
        "Bot Integration": "Bank growth, spreads, provisions, deposits, NPL and capital metrics",
        "Modules": "fundamental_master.py",
        "Autonomous Use": "Separates bank scoring from non-bank scoring instead of using one generic ratio set.",
        "Status": "Integrated",
    },
    {
        "Category": "Valuation & Quality",
        "Uploaded Knowledge": "Valuation, inventory analysis, cash-flow discipline",
        "Bot Integration": "Intrinsic value, margin of safety, cash-flow and working-capital checks",
        "Modules": "intrinsic_value_engine.py, fundamental_master.py",
        "Autonomous Use": "Blocks attractive charts when valuation or business quality is weak.",
        "Status": "Integrated",
    },
    {
        "Category": "Fraud / Red Flags",
        "Uploaded Knowledge": "Financial fakery / red-flag framework",
        "Bot Integration": "Cash-vs-profit checks, receivables flags, dividend sustainability, audit keyword risk",
        "Modules": "fraud_risk_engine.py",
        "Autonomous Use": "Overrides BUY logic when accounting or governance red flags cluster.",
        "Status": "Integrated",
    },
    {
        "Category": "Investor & Macro",
        "Uploaded Knowledge": "Investor profile, macro checklist, catalyst monitoring",
        "Bot Integration": "Investor suitability, macro sensitivity, event/catalyst screening",
        "Modules": "investor_profile_engine.py, macro_checklist_engine.py, catalyst_engine.py",
        "Autonomous Use": "Aligns decisions with holding period, macro backdrop, and event risk.",
        "Status": "Integrated",
    },
    {
        "Category": "Rankings & Data Quality",
        "Uploaded Knowledge": "PSX stock analysis bot pack, watchlist scorecards, sector rankings, dividend shortlist",
        "Bot Integration": "Fundamental scoring, sector tables, coverage awareness, quality/value/growth/balance-sheet weights",
        "Modules": "fundamental_master.py, app.py upload parsers",
        "Autonomous Use": "Uses uploaded ranking tables when supplied and keeps missing-data risk visible.",
        "Status": "Integrated",
    },
    {
        "Category": "Financial Document Intelligence",
        "Uploaded Knowledge": "PSX PDF processing / vector retrieval / metadata filtering architecture",
        "Bot Integration": "Knowledge map preserves statement-retrieval workflow and evidence-first analysis design",
        "Modules": "README integration notes; optional future local index bridge",
        "Autonomous Use": "Keeps the terminal designed to work with audited report context instead of blind summaries.",
        "Status": "Mapped",
    },
    {
        "Category": "Autonomous Workflow",
        "Uploaded Knowledge": "Earlier scanners, prediction bot, tracker, watchtower, alerts",
        "Bot Integration": "Unified autopilot cycle and saved output packs",
        "Modules": "autopilot_manager.py, autopilot_runner.py, app.py",
        "Autonomous Use": "Runs a single management cycle for holdings, opportunities, hazards, and diagnostics.",
        "Status": "Integrated",
    },
]


ENGINE_COVERAGE = {
    "Technical Pattern Engine": ["Candlestick", "Fibonacci", "CWT", "Trend", "Divergence"],
    "Risk Engine": ["Risk/Reward", "Hazards", "News/Event", "Secular Regime"],
    "Fundamental Engine": ["Non-bank ratios", "Bank ratios", "Sector ranking", "Dividend review"],
    "Valuation Engine": ["Intrinsic value", "Margin of safety", "Valuation multiples"],
    "Red-Flag Engine": ["Cash quality", "Receivables", "Audit keywords", "Dividend sustainability"],
    "Autonomous Manager": ["Holdings", "New opportunities", "Diagnostics", "Persisted outputs"],
}


def knowledge_registry_table() -> pd.DataFrame:
    return pd.DataFrame(KNOWLEDGE_REGISTRY)


def engine_coverage_table() -> pd.DataFrame:
    rows = []
    for engine, items in ENGINE_COVERAGE.items():
        rows.append({
            "Engine": engine,
            "Coverage": ", ".join(items),
            "Coverage Count": len(items),
            "Status": "Active",
        })
    return pd.DataFrame(rows)


def knowledge_status_summary() -> Dict[str, Any]:
    table = knowledge_registry_table()
    integrated = int((table["Status"].astype(str).str.lower() == "integrated").sum())
    mapped = int((table["Status"].astype(str).str.lower() == "mapped").sum())
    return {
        "Knowledge Sources": int(len(table)),
        "Integrated": integrated,
        "Mapped": mapped,
        "Active Engines": int(len(ENGINE_COVERAGE)),
        "Verdict": "All major uploaded frameworks are registered in the terminal brain.",
    }


def decision_knowledge_tags(
    *,
    confluence: Dict[str, Any] | None = None,
    hazard: Dict[str, Any] | None = None,
    news: Dict[str, Any] | None = None,
    fundamental: Dict[str, Any] | None = None,
    intrinsic: Dict[str, Any] | None = None,
    fraud: Dict[str, Any] | None = None,
    secular: Dict[str, Any] | None = None,
) -> str:
    tags: List[str] = []
    if confluence:
        tags.append("Pattern/CWT/Fib")
    if hazard or news:
        tags.append("Risk/Watchtower")
    if secular:
        tags.append("Secular Regime")
    if fundamental:
        tags.append("Fundamentals")
    if intrinsic:
        tags.append("Valuation")
    if fraud:
        tags.append("Red Flags")
    return " | ".join(tags) if tags else "Core signal only"
