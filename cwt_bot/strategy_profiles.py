from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


TRADING_PROFILES: Dict[str, Dict[str, Any]] = {
    "Scalping": {
        "analysis_tf": "30m",
        "execution_tf": "5m",
        "prediction_horizon": 3,
        "default_risk_pct": 0.15,
        "target_rr": 1.5,
        "stop_atr": 1.0,
        "description": "Very short holding period. Requires genuine intraday candles, strong liquidity, low spread/slippage, and fast alerts.",
        "data_warning": "Scalping requires real 5m/30m intraday OHLCV. Daily-only data is not suitable.",
        "preferred_scenario": "Scenario 1",
        "minimum_pro_score": 70,
    },
    "Intraday": {
        "analysis_tf": "1h",
        "execution_tf": "15m",
        "prediction_horizon": 5,
        "default_risk_pct": 0.22,
        "target_rr": 2.0,
        "stop_atr": 1.25,
        "description": "Same-day trading. Uses higher-timeframe bias with 15m entries and event/news avoidance.",
        "data_warning": "Intraday analysis requires 15m/1H OHLCV. Daily data cannot confirm intraday structure.",
        "preferred_scenario": "Scenario 1",
        "minimum_pro_score": 65,
    },
    "Short-Term Swing": {
        "analysis_tf": "1d",
        "execution_tf": "4h",
        "prediction_horizon": 10,
        "default_risk_pct": 1.0,
        "target_rr": 3.0,
        "stop_atr": 1.5,
        "description": "Multi-day swing trading. Aligns with the course reversal/continuation framework and 1:3 RR preference.",
        "data_warning": "4H candles are preferred for entries. If unavailable, use daily execution as a fallback with caution.",
        "preferred_scenario": "Scenario 1",
        "minimum_pro_score": 60,
    },
    "Position / Long-Term": {
        "analysis_tf": "1mo",
        "execution_tf": "1wk",
        "prediction_horizon": 15,
        "default_risk_pct": 1.5,
        "target_rr": 3.0,
        "stop_atr": 2.0,
        "description": "Longer holding period. Emphasizes secular trend, weekly/monthly structure, fundamentals, and macro event risk.",
        "data_warning": "Long-term mode works best with multi-year weekly/monthly OHLCV and fundamental context.",
        "preferred_scenario": "Scenario 1",
        "minimum_pro_score": 55,
    },
}


def get_profile(name: str) -> Dict[str, Any]:
    return TRADING_PROFILES.get(name, TRADING_PROFILES["Short-Term Swing"]).copy()


def profile_table():
    import pandas as pd
    rows = []
    for name, p in TRADING_PROFILES.items():
        rows.append({
            "Trading Style": name,
            "Analysis TF": p["analysis_tf"],
            "Execution TF": p["execution_tf"],
            "Prediction Horizon": p["prediction_horizon"],
            "Default Risk %": p["default_risk_pct"],
            "Target RR": p["target_rr"],
            "Stop ATR": p["stop_atr"],
            "Minimum PRO Score": p["minimum_pro_score"],
            "Description": p["description"],
        })
    return pd.DataFrame(rows)
