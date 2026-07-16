from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

# Selector used by Streamlit scanners so uploaded PDFs, patterns, and scenario rules
# are visible and user-controllable instead of hidden inside the bot.
KNOWLEDGE_LAYERS: List[str] = [
    "Candlestick Patterns",
    "Chart Patterns: Wedge / Flag / Triangle / Rectangle",
    "Fibonacci Confluence",
    "CWT Alligator / Trend Structure",
    "Scenario 1/2/3 MTF Framework",
    "RSI / MACD Divergence",
    "Support / Resistance + Breakout",
    "Risk Reward / Position Sizing",
    "News / Event Risk Filter",
    "Secular Market Regime",
    "Fundamental Quality",
    "Valuation / Margin of Safety",
    "Financial Fakery / Red Flags",
    "Portfolio Action Rules",
]

KNOWLEDGE_PROFILES: Dict[str, List[str]] = {
    "Complete Uploaded Knowledge Brain": KNOWLEDGE_LAYERS,
    "Scenario Scanner Focus": [
        "Scenario 1/2/3 MTF Framework",
        "CWT Alligator / Trend Structure",
        "Support / Resistance + Breakout",
        "Risk Reward / Position Sizing",
        "News / Event Risk Filter",
    ],
    "Pattern Scanner Focus": [
        "Candlestick Patterns",
        "Chart Patterns: Wedge / Flag / Triangle / Rectangle",
        "Fibonacci Confluence",
        "RSI / MACD Divergence",
        "Support / Resistance + Breakout",
        "Risk Reward / Position Sizing",
    ],
    "PDF Trading Framework Focus": [
        "CWT Alligator / Trend Structure",
        "Scenario 1/2/3 MTF Framework",
        "Fibonacci Confluence",
        "Risk Reward / Position Sizing",
        "News / Event Risk Filter",
        "Secular Market Regime",
    ],
    "Fundamental + Valuation Confirmation": [
        "Fundamental Quality",
        "Valuation / Margin of Safety",
        "Financial Fakery / Red Flags",
        "Portfolio Action Rules",
    ],
    "Custom Selection": [],
}

LAYER_DATA_SOURCE: Dict[str, str] = {
    "Candlestick Patterns": "Uploaded candlestick pattern rules + patterns_candles engine",
    "Chart Patterns: Wedge / Flag / Triangle / Rectangle": "Uploaded chart-pattern logic + reversal/continuation engines",
    "Fibonacci Confluence": "Uploaded Fibonacci framework + fib_engine",
    "CWT Alligator / Trend Structure": "Uploaded CWT strategy PDFs + CWT engine",
    "Scenario 1/2/3 MTF Framework": "Uploaded Week 5/6 and Week 2/3/4 scenario systems",
    "RSI / MACD Divergence": "Uploaded divergence rules + divergence/pro metrics engines",
    "Support / Resistance + Breakout": "Uploaded technical framework + S/R engine",
    "Risk Reward / Position Sizing": "Uploaded RRMS/risk discipline PDFs + risk engine",
    "News / Event Risk Filter": "Uploaded news-avoidance rules + event risk monitor",
    "Secular Market Regime": "Uploaded secular market/bear-market framework + secular engine",
    "Fundamental Quality": "Uploaded fundamentals/ranking sheets + fundamental master engine",
    "Valuation / Margin of Safety": "Uploaded valuation/main-page screenshots or Google Sheet + intrinsic value engine",
    "Financial Fakery / Red Flags": "Uploaded financial-fakery framework + fraud risk engine",
    "Portfolio Action Rules": "Uploaded portfolio logic + Decision Center portfolio engine",
}

LAYER_SCANNER_ROLE: Dict[str, str] = {
    "Candlestick Patterns": "Pattern detector",
    "Chart Patterns: Wedge / Flag / Triangle / Rectangle": "Pattern detector",
    "Fibonacci Confluence": "Entry confirmation",
    "CWT Alligator / Trend Structure": "Trend / structure filter",
    "Scenario 1/2/3 MTF Framework": "Scenario selector",
    "RSI / MACD Divergence": "Momentum confirmation",
    "Support / Resistance + Breakout": "Entry and target mapping",
    "Risk Reward / Position Sizing": "Trade-plan gate",
    "News / Event Risk Filter": "Risk override",
    "Secular Market Regime": "Market-regime gate",
    "Fundamental Quality": "Fundamental confirmation when data is available",
    "Valuation / Margin of Safety": "Valuation confirmation when data is available",
    "Financial Fakery / Red Flags": "Red-flag override when data is available",
    "Portfolio Action Rules": "Hold / sell / average / buy-more logic when portfolio data is available",
}


def layers_for_profile(profile: str, custom_layers: List[str] | None = None) -> List[str]:
    if profile == "Custom Selection":
        return list(custom_layers or [])
    return list(KNOWLEDGE_PROFILES.get(profile, KNOWLEDGE_LAYERS))


def selector_summary(profile: str, layers: List[str]) -> Dict[str, Any]:
    return {
        "Knowledge Profile": profile,
        "Selected Layers": ", ".join(layers) if layers else "None selected",
        "Layer Count": len(layers),
        "Technical Layers": sum(1 for x in layers if x in KNOWLEDGE_LAYERS[:8]),
        "Fundamental/Risk Layers": sum(1 for x in layers if x in KNOWLEDGE_LAYERS[8:]),
    }


def selected_layer_table(layers: List[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Selected Layer": layer,
            "Data Source / Uploaded Material Used": LAYER_DATA_SOURCE.get(layer, "Registered uploaded knowledge"),
            "Scanner Role": LAYER_SCANNER_ROLE.get(layer, "Decision support"),
        }
        for layer in layers
    ])


def apply_knowledge_columns(df: pd.DataFrame, profile: str, layers: List[str], scanner_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.insert(0, "Knowledge Profile", profile)
    out.insert(1, "Selected Knowledge Layers", ", ".join(layers) if layers else "None selected")
    out.insert(2, "Scanner Mode", scanner_name)
    basis = []
    if any("Fundamental" in x or "Valuation" in x or "Fakery" in x for x in layers):
        basis.append("fundamental/valuation confirmation if uploaded/online data is available")
    if any("Pattern" in x or "Scenario" in x or "CWT" in x or "Divergence" in x for x in layers):
        basis.append("technical pattern + scenario scan")
    if any("Risk" in x or "News" in x or "Secular" in x for x in layers):
        basis.append("risk/regime filter")
    out["Decision Basis"] = "; ".join(basis) if basis else "selected scanner data"
    return out


def required_data_for_layers(layers: List[str]) -> List[str]:
    required: List[str] = []
    if "Fundamental Quality" in layers:
        required.append("Fundamental Google Sheet / Excel row or 5 Sarmaaya fundamental screenshots")
    if "Valuation / Margin of Safety" in layers:
        required.append("Main page screenshot or valuation columns: current price, fair value/intrinsic value, margin of safety")
    if "Financial Fakery / Red Flags" in layers:
        required.append("Cash-flow, receivables, debt, dividend and audit/red-flag indicators")
    if "Portfolio Action Rules" in layers:
        required.append("Portfolio PDF/Excel with symbol, quantity, average buy price, current/MTM price")
    if any(x in layers for x in ["Candlestick Patterns", "Chart Patterns: Wedge / Flag / Triangle / Rectangle", "CWT Alligator / Trend Structure", "Scenario 1/2/3 MTF Framework"]):
        required.append("OHLCV chart data for selected symbol/timeframe")
    return list(dict.fromkeys(required))
