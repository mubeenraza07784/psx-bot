from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import pandas as pd


DEFAULT_STATE_FILE = Path(__file__).resolve().parents[1] / "data" / "psx_alert_state.json"


TRACKED_FIELDS = {
    "bias": "Bias",
    "action": "Action",
    "setup": "Setup",
    "scenario": "MTF Scenario",
    "cwt_scenario": "CWT Scenario",
    "higher_trend": "Higher Trend",
    "execution_trend": "Execution Trend",
    "alligator_state": "Alligator",
    "divergence": "Divergence",
    "order_type": "Order Type",
    "trade_quality": "Trade Quality",
    "risk_severity": "Risk Alert",
    "pro_grade": "PRO Grade",
}


def _read_state(path: Path = DEFAULT_STATE_FILE) -> Dict[str, Any]:
    if not path.exists():
        return {"symbols": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"symbols": {}}
        if "symbols" not in data or not isinstance(data["symbols"], dict):
            data["symbols"] = {}
        return data
    except Exception:
        return {"symbols": {}}


def _write_state(data: Dict[str, Any], path: Path = DEFAULT_STATE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _snapshot_key(symbol: str, analysis_tf: str, execution_tf: str) -> str:
    return f"{symbol.upper().strip()}::{analysis_tf}::{execution_tf}"


def build_snapshot(
    symbol: str,
    analysis_tf: str,
    execution_tf: str,
    result: Dict[str, Any],
    pro: Dict[str, Any],
    risk: Dict[str, Any],
) -> Dict[str, Any]:
    signal = result.get("signal", {}) or {}
    plan = result.get("trade_plan", {}) or {}
    snapshot = {
        "symbol": symbol.upper().strip(),
        "analysis_tf": analysis_tf,
        "execution_tf": execution_tf,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bias": signal.get("bias"),
        "action": signal.get("action"),
        "setup": signal.get("setup_type"),
        "scenario": result.get("mtf_scenario", result.get("scenario", {})).get("label"),
        "cwt_scenario": result.get("cwt_scenario", {}).get("label"),
        "higher_trend": result.get("higher_trend", {}).get("trend"),
        "execution_trend": result.get("execution_trend", {}).get("trend"),
        "alligator_state": result.get("cwt", {}).get("state"),
        "divergence": result.get("divergence", {}).get("label"),
        "order_type": plan.get("order_type"),
        "entry": plan.get("entry"),
        "stop_loss": plan.get("stop_loss"),
        "take_profit": plan.get("take_profit"),
        "confidence": signal.get("confidence"),
        "pro_score": pro.get("pro_score"),
        "pro_grade": pro.get("pro_grade"),
        "trade_quality": pro.get("trade_quality"),
        "risk_severity": risk.get("risk_severity"),
        "risk_points": risk.get("risk_points"),
        "risk_text": risk.get("risk_text"),
    }
    return snapshot


def _as_num(value: Any):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def compare_snapshots(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []

    if not previous:
        alerts.append({
            "Alert Type": "BASELINE CREATED",
            "Severity": "INFO",
            "Field": "Snapshot",
            "Previous": "",
            "Current": "Initial saved state",
            "Message": "Baseline state created. Future runs will highlight changes.",
        })
        return alerts

    # Field changes
    important_critical = {"bias", "scenario", "higher_trend", "execution_trend", "risk_severity", "divergence"}
    for field, label in TRACKED_FIELDS.items():
        old = previous.get(field)
        new = current.get(field)
        if old != new:
            severity = "HIGH" if field in important_critical else "MODERATE"
            if field == "risk_severity" and new in {"HIGH", "CRITICAL"}:
                severity = "CRITICAL"
            alerts.append({
                "Alert Type": "ANALYSIS CHANGE",
                "Severity": severity,
                "Field": label,
                "Previous": old,
                "Current": new,
                "Message": f"{label} changed from {old} to {new}.",
            })

    # Confidence and pro score shifts
    prev_conf = _as_num(previous.get("confidence"))
    cur_conf = _as_num(current.get("confidence"))
    if prev_conf is not None and cur_conf is not None:
        delta = cur_conf - prev_conf
        if abs(delta) >= 12:
            severity = "HIGH" if abs(delta) >= 20 else "MODERATE"
            alerts.append({
                "Alert Type": "SCORE CHANGE",
                "Severity": severity,
                "Field": "CWT Confidence",
                "Previous": round(prev_conf, 2),
                "Current": round(cur_conf, 2),
                "Message": f"CWT confidence changed by {delta:+.2f} points.",
            })

    prev_pro = _as_num(previous.get("pro_score"))
    cur_pro = _as_num(current.get("pro_score"))
    if prev_pro is not None and cur_pro is not None:
        delta = cur_pro - prev_pro
        if abs(delta) >= 10:
            severity = "HIGH" if abs(delta) >= 18 else "MODERATE"
            alerts.append({
                "Alert Type": "SCORE CHANGE",
                "Severity": severity,
                "Field": "PRO Score",
                "Previous": round(prev_pro, 2),
                "Current": round(cur_pro, 2),
                "Message": f"PRO Score changed by {delta:+.2f} points.",
            })

    # Trade plan price changes
    for field, label in [("entry", "Entry"), ("stop_loss", "Stop Loss"), ("take_profit", "Take Profit")]:
        old = _as_num(previous.get(field))
        new = _as_num(current.get(field))
        if old is not None and new is not None and old != 0:
            pct = ((new - old) / abs(old)) * 100
            if abs(pct) >= 2:
                alerts.append({
                    "Alert Type": "TRADE PLAN CHANGE",
                    "Severity": "MODERATE",
                    "Field": label,
                    "Previous": round(old, 6),
                    "Current": round(new, 6),
                    "Message": f"{label} shifted by {pct:+.2f}%.",
                })

    if not alerts:
        alerts.append({
            "Alert Type": "NO MATERIAL CHANGE",
            "Severity": "LOW",
            "Field": "Snapshot",
            "Previous": "",
            "Current": "",
            "Message": "No material trend/analysis change detected versus the last saved run.",
        })
    return alerts


def check_and_update_symbol_alerts(
    symbol: str,
    analysis_tf: str,
    execution_tf: str,
    result: Dict[str, Any],
    pro: Dict[str, Any],
    risk: Dict[str, Any],
    path: Path = DEFAULT_STATE_FILE,
) -> Dict[str, Any]:
    state = _read_state(path)
    key = _snapshot_key(symbol, analysis_tf, execution_tf)
    previous = state["symbols"].get(key)
    current = build_snapshot(symbol, analysis_tf, execution_tf, result, pro, risk)
    alerts = compare_snapshots(previous, current)

    state["symbols"][key] = current
    _write_state(state, path)

    return {
        "key": key,
        "previous": previous,
        "current": current,
        "alerts": alerts,
        "state_file": str(path),
    }


def alerts_to_dataframe(alerts: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(alerts)


def load_alert_state_table(path: Path = DEFAULT_STATE_FILE) -> pd.DataFrame:
    state = _read_state(path)
    rows = []
    for key, snap in state.get("symbols", {}).items():
        rows.append({
            "Key": key,
            "Symbol": snap.get("symbol"),
            "Analysis TF": snap.get("analysis_tf"),
            "Execution TF": snap.get("execution_tf"),
            "Last Updated UTC": snap.get("timestamp_utc"),
            "Bias": snap.get("bias"),
            "Scenario": snap.get("scenario"),
            "Higher Trend": snap.get("higher_trend"),
            "Execution Trend": snap.get("execution_trend"),
            "PRO Score": snap.get("pro_score"),
            "PRO Grade": snap.get("pro_grade"),
            "Trade Quality": snap.get("trade_quality"),
            "Risk Alert": snap.get("risk_severity"),
        })
    return pd.DataFrame(rows)
