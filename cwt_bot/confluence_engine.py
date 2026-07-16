from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd


def build_confluence_score(
    result: Dict[str, Any],
    pro: Dict[str, Any] | None = None,
    prediction: Dict[str, Any] | None = None,
    fib: Dict[str, Any] | None = None,
    secular: Dict[str, Any] | None = None,
    hazard: Dict[str, Any] | None = None,
    news: Dict[str, Any] | None = None,
    trading_style: str = "Short-Term Swing",
) -> Dict[str, Any]:
    """
    Professional confluence score: combines uploaded strategy logic with
    technical, risk, prediction, and event-awareness signals.
    """
    pro = pro or {}
    prediction = prediction or {}
    fib = fib or {}
    secular = secular or {}
    hazard = hazard or {}
    news = news or {}

    score = 0.0
    max_score = 100.0
    rows: List[Dict[str, Any]] = []

    signal = result.get("signal", {}) or {}
    confidence = float(signal.get("confidence", 0) or 0)
    pro_score = pro.get("pro_score")
    mtf = result.get("mtf_scenario", result.get("scenario", {}))
    cwt = result.get("cwt_scenario", {})
    divergence = result.get("divergence", {}).get("label", "None")
    fvg = result.get("fvg", {})

    # CWT / base score
    base_points = min(25.0, confidence * 0.25)
    score += base_points
    rows.append({"Factor": "CWT Signal Confidence", "Points": round(base_points, 2), "Reason": f"{confidence:.0f}/100 signal confidence"})

    if pro_score is not None:
        pts = min(18.0, float(pro_score) * 0.18)
        score += pts
        rows.append({"Factor": "PRO Technical Score", "Points": round(pts, 2), "Reason": f"PRO Score {pro_score}"})

    # Preferred scenarios
    if str(mtf.get("number", "")).endswith("1") or "Scenario 1" in str(mtf.get("label", "")):
        score += 10
        rows.append({"Factor": "MTF Scenario 1", "Points": 10, "Reason": "Higher and execution timeframe align."})
    elif "Scenario 2" in str(mtf.get("label", "")):
        score += 5
        rows.append({"Factor": "MTF Scenario 2", "Points": 5, "Reason": "Pullback context; acceptable but secondary."})
    elif "Scenario 3" in str(mtf.get("label", "")):
        score += 2
        rows.append({"Factor": "MTF Scenario 3", "Points": 2, "Reason": "Sideways execution context."})

    if "Scenario 1" in str(cwt.get("label", "")):
        score += 8
        rows.append({"Factor": "CWT Trend Continuation", "Points": 8, "Reason": "Open Alligator continuation setup."})
    elif "Scenario 2" in str(cwt.get("label", "")):
        score += 5
        rows.append({"Factor": "CWT Reversal", "Points": 5, "Reason": "Jawline reversal setup."})
    elif "Scenario 3" in str(cwt.get("label", "")):
        score -= 8
        rows.append({"Factor": "Sleeping Alligator", "Points": -8, "Reason": "Range/no-trade state."})

    # Divergence
    if divergence not in {"None", None, ""}:
        bias = signal.get("bias", "Neutral")
        if ("Bullish" in divergence and bias == "Bullish") or ("Bearish" in divergence and bias == "Bearish"):
            score += 4
            rows.append({"Factor": "Aligned RSI Divergence", "Points": 4, "Reason": divergence})
        else:
            score -= 6
            rows.append({"Factor": "Conflicting RSI Divergence", "Points": -6, "Reason": divergence})

    # FVG
    if fvg.get("nearest_bullish_zone") or fvg.get("nearest_bearish_zone"):
        score += 3
        rows.append({"Factor": "FVG Context", "Points": 3, "Reason": "Fair Value Gap zone available for planning."})

    # Fibonacci
    if fib.get("status") == "OK" and fib.get("confluence"):
        pts = 6 if fib.get("preferred_retracement") else 4
        score += pts
        rows.append({"Factor": "Fibonacci Confluence", "Points": pts, "Reason": fib.get("message")})

    # Secular for long-term style
    if trading_style == "Position / Long-Term" and secular.get("status") == "OK":
        bias = secular.get("secular_bias", "")
        if "Bull" in bias:
            score += 8
            rows.append({"Factor": "Secular Bull Structure", "Points": 8, "Reason": bias})
        elif "Bear" in bias:
            score -= 10
            rows.append({"Factor": "Secular Bear Structure", "Points": -10, "Reason": bias})

    # Prediction
    verdict = prediction.get("prediction_verdict")
    if verdict == "PREDICTIVE SUPPORT":
        score += 8
        rows.append({"Factor": "Prediction Support", "Points": 8, "Reason": prediction.get("loss_control_action", "")})
    elif verdict == "LOSS-RISK ELEVATED":
        score -= 10
        rows.append({"Factor": "Predicted Loss Risk", "Points": -10, "Reason": prediction.get("loss_control_action", "")})
    elif verdict == "PREDICTION CONFLICT":
        score -= 8
        rows.append({"Factor": "Prediction Conflict", "Points": -8, "Reason": prediction.get("loss_control_action", "")})

    # Hazards
    hazard_level = hazard.get("hazard_level")
    if hazard_level == "CRITICAL":
        score -= 18
        rows.append({"Factor": "Critical Price Hazard", "Points": -18, "Reason": hazard.get("summary", "")})
    elif hazard_level == "HIGH":
        score -= 12
        rows.append({"Factor": "High Price Hazard", "Points": -12, "Reason": hazard.get("summary", "")})
    elif hazard_level == "MODERATE":
        score -= 6
        rows.append({"Factor": "Moderate Price Hazard", "Points": -6, "Reason": hazard.get("summary", "")})

    # News/event risk
    news_level = news.get("risk_level")
    if news_level == "HIGH":
        score -= 14
        rows.append({"Factor": "High News/Event Risk", "Points": -14, "Reason": "Macro/event warning detected."})
    elif news_level == "MODERATE":
        score -= 8
        rows.append({"Factor": "Moderate News/Event Risk", "Points": -8, "Reason": "Company/news event review recommended."})

    score = max(0.0, min(max_score, score))

    if score >= 80:
        grade = "A+"
        verdict = "Elite Confluence"
    elif score >= 70:
        grade = "A"
        verdict = "High Confluence"
    elif score >= 60:
        grade = "B+"
        verdict = "Actionable / Review"
    elif score >= 50:
        grade = "B"
        verdict = "Watchlist"
    elif score >= 40:
        grade = "C"
        verdict = "Weak / Wait"
    else:
        grade = "D"
        verdict = "Avoid / Not Ready"

    if hazard_level in {"HIGH", "CRITICAL"} or news_level == "HIGH":
        action = "Risk override: wait, reduce size, or avoid entry until hazard/event risk clears."
    elif score >= 70:
        action = "Setup has strong confluence. Use defined stop-loss, position size, and event checks."
    elif score >= 60:
        action = "Setup may be considered after confirmation and acceptable risk."
    else:
        action = "Do not force the trade; wait for clearer confluence."

    return {
        "confluence_score": round(score, 2),
        "confluence_grade": grade,
        "confluence_verdict": verdict,
        "quick_action": action,
        "factors": rows,
    }


def confluence_table(confluence: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(confluence.get("factors", []))
