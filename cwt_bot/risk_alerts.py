from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def build_risk_warning(
    result: Dict[str, Any],
    pro: Dict[str, Any] | None = None,
    user_event_risk: bool = False,
    benchmark_conflict: bool = False,
) -> Dict[str, Any]:
    """
    Convert technical diagnostics into practical risk warnings.
    Severity:
      - LOW: no meaningful technical risk flag
      - MODERATE: one or more caution flags
      - HIGH: multiple material flags
      - CRITICAL: avoid/act fast due to invalid setup or major conflict
    """
    pro = pro or {}
    latest = pro.get("latest_metrics", {}) or {}

    warnings: List[str] = []
    urgent: List[str] = []
    risk_points = 0

    signal = result.get("signal", {}) or {}
    bias = signal.get("bias", "Neutral")
    action = signal.get("action", "WAIT")
    scenario = result.get("scenario", {}).get("label", "")
    cwt_state = result.get("cwt", {}).get("state", "")
    divergence = result.get("divergence", {}).get("label", "None")
    higher_trend = result.get("higher_trend", {}).get("trend", "Unknown")
    execution_trend = result.get("execution_trend", {}).get("trend", "Unknown")
    warnings_from_core = result.get("warnings", []) or []

    rsi = latest.get("rsi")
    adx = latest.get("adx")
    atr_pct = latest.get("atr_pct")
    volume_ratio = latest.get("volume_ratio")
    pro_score = pro.get("pro_score")
    trade_quality = pro.get("trade_quality", "Unclear")
    risk_level = pro.get("risk_level", "Unknown")

    # Event / benchmark warnings
    if user_event_risk:
        warnings.append("Event risk flagged: monetary policy, CPI, board meeting, or result announcement may distort the setup.")
        risk_points += 2
    if benchmark_conflict:
        warnings.append("Benchmark conflict flagged: KSE-100 / market bias may be against this setup.")
        risk_points += 2

    # Carry forward any pre-existing warnings from signal engine
    for item in warnings_from_core:
        if item and item not in warnings:
            warnings.append(str(item))
            risk_points += 1

    # Setup availability / CWT state
    if bias == "Neutral" or action == "WAIT":
        warnings.append("No clear directional bias: the bot is not showing a clean actionable BUY/SELL plan.")
        risk_points += 2
    if "Sleeping" in cwt_state or "Closed" in cwt_state:
        warnings.append("Sleeping/closed Alligator: trend continuation quality is weak or ranging.")
        risk_points += 2

    # Scenario risk
    if scenario.startswith("Scenario 2"):
        warnings.append("Scenario 2 pullback: entry timing requires more confirmation than Scenario 1.")
        risk_points += 1
    elif scenario.startswith("Scenario 3"):
        warnings.append("Scenario 3 sideways execution: breakout failure risk is higher.")
        risk_points += 2
    elif scenario.startswith("No HTF Trend"):
        warnings.append("No higher-timeframe trend detected: setup quality is reduced.")
        risk_points += 2

    # Volatility risk
    if atr_pct is not None:
        try:
            atr_val = float(atr_pct)
            if atr_val >= 7:
                warnings.append(f"Very high ATR volatility ({_fmt(atr_val)}%): stop-loss may be wider and slippage risk is higher.")
                risk_points += 3
            elif atr_val >= 4:
                warnings.append(f"High ATR volatility ({_fmt(atr_val)}%): position sizing should be reduced or reviewed.")
                risk_points += 2
        except Exception:
            pass

    # Volume quality
    if volume_ratio is not None:
        try:
            vol_val = float(volume_ratio)
            if vol_val < 0.60:
                warnings.append(f"Very weak relative volume ({_fmt(vol_val)}x): signal may lack participation/liquidity confirmation.")
                risk_points += 2
            elif vol_val < 0.85:
                warnings.append(f"Weak relative volume ({_fmt(vol_val)}x): confirm liquidity before acting.")
                risk_points += 1
        except Exception:
            pass

    # Trend strength
    if adx is not None:
        try:
            adx_val = float(adx)
            if adx_val < 15:
                warnings.append(f"Weak ADX trend strength ({_fmt(adx_val)}): trend-following signals are less reliable.")
                risk_points += 2
            elif adx_val < 20:
                warnings.append(f"Developing but not strong ADX trend ({_fmt(adx_val)}): confirmation is recommended.")
                risk_points += 1
        except Exception:
            pass

    # RSI overextension
    if rsi is not None:
        try:
            rsi_val = float(rsi)
            if bias == "Bullish" and rsi_val >= 72:
                warnings.append(f"RSI is overextended for a bullish entry ({_fmt(rsi_val)}): chasing risk is elevated.")
                risk_points += 2
            elif bias == "Bearish" and rsi_val <= 28:
                warnings.append(f"RSI is overextended for a bearish entry ({_fmt(rsi_val)}): late-entry risk is elevated.")
                risk_points += 2
        except Exception:
            pass

    # Divergence / trend contradiction
    if divergence not in {"None", "", None}:
        if bias == "Bullish" and "Bearish" in divergence:
            urgent.append("Bearish divergence conflicts with bullish trade bias.")
            risk_points += 3
        elif bias == "Bearish" and "Bullish" in divergence:
            urgent.append("Bullish divergence conflicts with bearish trade bias.")
            risk_points += 3

    if higher_trend in {"Bullish", "Bearish"} and execution_trend in {"Bullish", "Bearish"}:
        if higher_trend != execution_trend and scenario.startswith("Scenario 2"):
            warnings.append("Higher timeframe and execution timeframe are opposite: this is a pullback/reversal-sensitive zone.")
            risk_points += 1

    # Score / quality check
    if pro_score is not None:
        try:
            score_val = float(pro_score)
            if score_val < 45:
                warnings.append(f"Low PRO Score ({_fmt(score_val)}): setup does not meet stronger-quality conditions.")
                risk_points += 2
            elif score_val < 60:
                warnings.append(f"Moderate PRO Score ({_fmt(score_val)}): keep it on review/watchlist rather than rush entry.")
                risk_points += 1
        except Exception:
            pass

    if trade_quality == "Low Quality":
        warnings.append("Trade-quality label is Low Quality.")
        risk_points += 2

    # Determine risk band
    if urgent:
        severity = "CRITICAL"
    elif risk_points >= 8:
        severity = "HIGH"
    elif risk_points >= 4:
        severity = "MODERATE"
    else:
        severity = "LOW"

    if severity == "CRITICAL":
        quick_action = "Avoid new entry or reassess immediately; trend/analysis conflict is material."
    elif severity == "HIGH":
        quick_action = "Proceed only after manual confirmation; reduce risk or wait for cleaner confirmation."
    elif severity == "MODERATE":
        quick_action = "Review entry, volume, and event risk before acting."
    else:
        quick_action = "No major technical risk warning detected beyond normal trade discipline."

    return {
        "risk_severity": severity,
        "risk_points": risk_points,
        "risk_level_from_volatility": risk_level,
        "warnings": warnings,
        "urgent_warnings": urgent,
        "quick_action": quick_action,
        "risk_text": " | ".join(urgent + warnings),
    }


def concise_risk_columns(risk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Risk Alert": risk.get("risk_severity", "LOW"),
        "Risk Points": risk.get("risk_points", 0),
        "Quick Action": risk.get("quick_action", ""),
        "Risk Warnings": risk.get("risk_text", ""),
    }
