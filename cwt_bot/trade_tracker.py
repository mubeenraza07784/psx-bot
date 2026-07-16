from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import uuid
import pandas as pd


DEFAULT_TRACKER_FILE = Path(__file__).resolve().parents[1] / "data" / "psx_trade_tracker.csv"


TRACKER_COLUMNS = [
    "trade_id",
    "created_utc",
    "updated_utc",
    "symbol",
    "analysis_tf",
    "execution_tf",
    "status",
    "strategy",
    "scenario",
    "bias",
    "action",
    "order_type",
    "entry",
    "stop_loss",
    "take_profit",
    "rr",
    "position_size",
    "risk_amount",
    "pro_score",
    "pro_grade",
    "trade_quality",
    "risk_alert",
    "prediction_verdict",
    "probability_up_pct",
    "probability_down_pct",
    "probability_long_stop_hit_pct",
    "probability_long_tp_hit_pct",
    "probability_short_stop_hit_pct",
    "probability_short_tp_hit_pct",
    "expected_return_pct",
    "entry_filled_price",
    "exit_price",
    "exit_date_utc",
    "realized_pnl",
    "realized_pnl_pct",
    "notes",
]


def _ensure_tracker(path: Path = DEFAULT_TRACKER_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=TRACKER_COLUMNS).to_csv(path, index=False)


def load_trade_tracker(path: Path = DEFAULT_TRACKER_FILE) -> pd.DataFrame:
    _ensure_tracker(path)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=TRACKER_COLUMNS)
    for col in TRACKER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[TRACKER_COLUMNS]


def save_trade_tracker(df: pd.DataFrame, path: Path = DEFAULT_TRACKER_FILE) -> None:
    _ensure_tracker(path)
    work = df.copy()
    for col in TRACKER_COLUMNS:
        if col not in work.columns:
            work[col] = None
    work[TRACKER_COLUMNS].to_csv(path, index=False)


def append_trade_plan(
    symbol: str,
    analysis_tf: str,
    execution_tf: str,
    result: Dict[str, Any],
    pro: Dict[str, Any],
    risk: Dict[str, Any],
    prediction: Dict[str, Any] | None = None,
    position_size: Any = None,
    risk_amount: Any = None,
    notes: str = "",
    path: Path = DEFAULT_TRACKER_FILE,
) -> str:
    tracker = load_trade_tracker(path)
    signal = result.get("signal", {}) or {}
    plan = result.get("trade_plan", {}) or {}
    prediction = prediction or {}

    now = datetime.now(timezone.utc).isoformat()
    trade_id = str(uuid.uuid4())[:12]

    row = {
        "trade_id": trade_id,
        "created_utc": now,
        "updated_utc": now,
        "symbol": symbol.upper().strip(),
        "analysis_tf": analysis_tf,
        "execution_tf": execution_tf,
        "status": "PLANNED",
        "strategy": "PSX CWT PRO + Prediction",
        "scenario": result.get("scenario", {}).get("label"),
        "bias": signal.get("bias"),
        "action": signal.get("action"),
        "order_type": plan.get("order_type"),
        "entry": plan.get("entry"),
        "stop_loss": plan.get("stop_loss"),
        "take_profit": plan.get("take_profit"),
        "rr": plan.get("rr"),
        "position_size": position_size,
        "risk_amount": risk_amount,
        "pro_score": pro.get("pro_score"),
        "pro_grade": pro.get("pro_grade"),
        "trade_quality": pro.get("trade_quality"),
        "risk_alert": risk.get("risk_severity"),
        "prediction_verdict": prediction.get("prediction_verdict"),
        "probability_up_pct": prediction.get("probability_up_pct"),
        "probability_down_pct": prediction.get("probability_down_pct"),
        "probability_long_stop_hit_pct": prediction.get("probability_long_stop_hit_pct"),
        "probability_long_tp_hit_pct": prediction.get("probability_long_tp_hit_pct"),
        "probability_short_stop_hit_pct": prediction.get("probability_short_stop_hit_pct"),
        "probability_short_tp_hit_pct": prediction.get("probability_short_tp_hit_pct"),
        "expected_return_pct": prediction.get("expected_return_pct"),
        "entry_filled_price": None,
        "exit_price": None,
        "exit_date_utc": None,
        "realized_pnl": None,
        "realized_pnl_pct": None,
        "notes": notes,
    }

    tracker = pd.concat([tracker, pd.DataFrame([row])], ignore_index=True)
    save_trade_tracker(tracker, path)
    return trade_id


def update_trade_row(
    trade_id: str,
    status: str,
    entry_filled_price: float | None = None,
    exit_price: float | None = None,
    exit_date_utc: str | None = None,
    notes: str | None = None,
    path: Path = DEFAULT_TRACKER_FILE,
) -> pd.DataFrame:
    tracker = load_trade_tracker(path)
    if tracker.empty:
        raise ValueError("Trade tracker is empty.")
    mask = tracker["trade_id"].astype(str) == str(trade_id)
    if not mask.any():
        raise ValueError(f"Trade ID not found: {trade_id}")

    now = datetime.now(timezone.utc).isoformat()
    idx = tracker.index[mask][0]
    tracker.loc[idx, "status"] = status
    tracker.loc[idx, "updated_utc"] = now

    if entry_filled_price is not None:
        tracker.loc[idx, "entry_filled_price"] = entry_filled_price
    if exit_price is not None:
        tracker.loc[idx, "exit_price"] = exit_price
    if exit_date_utc is not None:
        tracker.loc[idx, "exit_date_utc"] = exit_date_utc
    if notes is not None:
        tracker.loc[idx, "notes"] = notes

    # Realized P&L calculation where possible
    try:
        entry = float(tracker.loc[idx, "entry_filled_price"])
        exit_ = float(tracker.loc[idx, "exit_price"])
        qty = float(tracker.loc[idx, "position_size"])
        bias = str(tracker.loc[idx, "bias"])
        if entry > 0 and qty > 0:
            pnl = (exit_ - entry) * qty if bias == "Bullish" else (entry - exit_) * qty
            pnl_pct = ((exit_ - entry) / entry) * 100 if bias == "Bullish" else ((entry - exit_) / entry) * 100
            tracker.loc[idx, "realized_pnl"] = round(pnl, 4)
            tracker.loc[idx, "realized_pnl_pct"] = round(pnl_pct, 4)
    except Exception:
        pass

    save_trade_tracker(tracker, path)
    return tracker


def tracker_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "total": 0,
            "planned": 0,
            "open": 0,
            "closed": 0,
            "tp": 0,
            "sl": 0,
            "cancelled": 0,
            "win_rate_pct": None,
            "realized_pnl": 0.0,
        }

    status = df["status"].fillna("").astype(str).str.upper()
    realized = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    closed_mask = status.isin(["CLOSED", "TP HIT", "SL HIT", "CANCELLED"])
    wins = status.isin(["TP HIT", "CLOSED"]) & (realized > 0)
    closed_trades = int(closed_mask.sum())
    win_rate = round(float(wins.sum()) / closed_trades * 100, 2) if closed_trades else None

    return {
        "total": int(len(df)),
        "planned": int((status == "PLANNED").sum()),
        "open": int((status == "OPEN").sum()),
        "closed": closed_trades,
        "tp": int((status == "TP HIT").sum()),
        "sl": int((status == "SL HIT").sum()),
        "cancelled": int((status == "CANCELLED").sum()),
        "win_rate_pct": win_rate,
        "realized_pnl": round(float(realized.sum()), 4),
    }
