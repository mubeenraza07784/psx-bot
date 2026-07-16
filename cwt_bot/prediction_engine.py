from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "rsi",
    "macd_hist",
    "adx",
    "atr_pct",
    "volume_ratio",
    "close_ma20_gap",
    "close_ma50_gap",
    "close_ma200_gap",
    "bb_position",
    "range_pct",
    "body_pct",
]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_prediction_frame(df: pd.DataFrame, horizon: int = 5, stop_atr: float = 1.5, target_rr: float = 3.0) -> pd.DataFrame:
    """
    Build time-series features and supervised labels.

    Labels:
    - target_up: future close is above current close after `horizon` bars
    - target_stop_hit: future low breaches current close - stop_atr * ATR
    - target_tp_hit: future high reaches current close + stop_atr * target_rr * ATR

    For bearish forecasts, the UI mirrors these probabilities using down-side interpretation.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    close = work["close"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    open_ = work["open"].astype(float)

    work["ret_1"] = close.pct_change(1) * 100
    work["ret_3"] = close.pct_change(3) * 100
    work["ret_5"] = close.pct_change(5) * 100
    work["ret_10"] = close.pct_change(10) * 100

    # Metrics are usually already available from PRO analysis; create fallbacks where needed.
    if "ma20" not in work.columns:
        work["ma20"] = close.rolling(20, min_periods=20).mean()
    if "ma50" not in work.columns:
        work["ma50"] = close.rolling(50, min_periods=50).mean()
    if "ma200" not in work.columns:
        work["ma200"] = close.rolling(200, min_periods=200).mean()

    if "macd_hist" not in work.columns:
        ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        work["macd_hist"] = macd - signal

    if "atr14" not in work.columns:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        work["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    if "atr_pct" not in work.columns:
        work["atr_pct"] = _safe_div(work["atr14"], close) * 100

    if "volume_ratio" not in work.columns:
        volume = work["volume"].astype(float) if "volume" in work.columns else pd.Series(0.0, index=work.index)
        avg_vol = volume.rolling(20, min_periods=1).mean()
        work["volume_ratio"] = _safe_div(volume, avg_vol)

    if "adx" not in work.columns:
        work["adx"] = np.nan

    if "rsi" not in work.columns:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        work["rsi"] = (100 - 100 / (1 + rs)).fillna(50.0)

    work["close_ma20_gap"] = _safe_div(close - work["ma20"], work["ma20"]) * 100
    work["close_ma50_gap"] = _safe_div(close - work["ma50"], work["ma50"]) * 100
    work["close_ma200_gap"] = _safe_div(close - work["ma200"], work["ma200"]) * 100

    if "bb_upper" not in work.columns or "bb_lower" not in work.columns:
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std(ddof=0)
        work["bb_upper"] = bb_mid + 2 * bb_std
        work["bb_lower"] = bb_mid - 2 * bb_std

    work["bb_position"] = _safe_div(close - work["bb_lower"], work["bb_upper"] - work["bb_lower"])
    work["range_pct"] = _safe_div(high - low, close) * 100
    work["body_pct"] = _safe_div((close - open_).abs(), close) * 100

    future_close = close.shift(-horizon)
    work["future_return_pct"] = _safe_div(future_close - close, close) * 100
    work["target_up"] = (future_close > close).astype(float)

    # Future window lows/highs aligned to current row
    future_lows = pd.concat([low.shift(-i) for i in range(1, horizon + 1)], axis=1)
    future_highs = pd.concat([high.shift(-i) for i in range(1, horizon + 1)], axis=1)
    work["future_min_low"] = future_lows.min(axis=1)
    work["future_max_high"] = future_highs.max(axis=1)

    long_stop = close - stop_atr * work["atr14"]
    long_tp = close + stop_atr * target_rr * work["atr14"]
    work["target_long_stop_hit"] = (work["future_min_low"] <= long_stop).astype(float)
    work["target_long_tp_hit"] = (work["future_max_high"] >= long_tp).astype(float)

    short_stop = close + stop_atr * work["atr14"]
    short_tp = close - stop_atr * target_rr * work["atr14"]
    work["target_short_stop_hit"] = (work["future_max_high"] >= short_stop).astype(float)
    work["target_short_tp_hit"] = (work["future_min_low"] <= short_tp).astype(float)

    work = work.replace([np.inf, -np.inf], np.nan)
    return work


def _usable_binary_target(y: pd.Series) -> bool:
    vals = y.dropna().astype(int)
    return len(vals) >= 80 and vals.nunique() >= 2 and vals.value_counts().min() >= 10


def _ts_validation_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    model: RandomForestClassifier,
    n_splits: int = 4,
) -> Dict[str, Any]:
    """
    Walk-forward style validation using TimeSeriesSplit.
    """
    if len(X) < 120:
        return {"accuracy": None, "brier": None, "folds": 0}

    n_splits = min(n_splits, max(2, len(X) // 60))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []

    for train_idx, test_idx in splitter.split(X):
        if len(train_idx) < 60 or len(test_idx) < 10:
            continue
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        all_true.extend(y.iloc[test_idx].astype(int).tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())

    if not all_true:
        return {"accuracy": None, "brier": None, "folds": 0}

    return {
        "accuracy": round(float(accuracy_score(all_true, all_pred)), 4),
        "brier": round(float(brier_score_loss(all_true, all_prob)), 4),
        "folds": n_splits,
    }


def _fit_classifier(X: pd.DataFrame, y: pd.Series, calibrate: bool = True):
    base = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    metrics = _ts_validation_probabilities(X, y, base)
    # Calibrated probabilities are preferable when enough data is available.
    if calibrate and len(X) >= 220:
        try:
            cv = TimeSeriesSplit(n_splits=3)
            model = CalibratedClassifierCV(
                estimator=base,
                method="sigmoid",
                cv=cv,
            )
            model.fit(X, y)
            calibrated = True
        except Exception:
            model = base.fit(X, y)
            calibrated = False
    else:
        model = base.fit(X, y)
        calibrated = False
    return model, metrics, calibrated


def _fit_regressor(X: pd.DataFrame, y: pd.Series):
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def _prob(model: Any, latest_x: pd.DataFrame) -> float | None:
    try:
        return round(float(model.predict_proba(latest_x)[0, 1]) * 100, 2)
    except Exception:
        return None


def _build_loss_control_verdict(
    bias: str,
    p_up: float | None,
    p_long_stop: float | None,
    p_long_tp: float | None,
    p_short_stop: float | None,
    p_short_tp: float | None,
    risk_severity: str | None = None,
) -> Dict[str, str]:
    risk_severity = risk_severity or "UNKNOWN"

    if bias == "Bullish":
        dir_prob = p_up
        stop_prob = p_long_stop
        tp_prob = p_long_tp
        aligned = dir_prob is not None and dir_prob >= 58
        loss_ok = stop_prob is not None and stop_prob <= 48
        target_ok = tp_prob is not None and tp_prob >= 28
        if aligned and loss_ok and target_ok and risk_severity not in {"HIGH", "CRITICAL"}:
            verdict = "PREDICTIVE SUPPORT"
            action = "Trade plan is statistically supportive, but use the defined stop and position size."
        elif dir_prob is not None and dir_prob < 45:
            verdict = "PREDICTION CONFLICT"
            action = "Bullish technical bias is not supported by the forecast model; avoid rushing entry."
        elif stop_prob is not None and stop_prob > 58:
            verdict = "LOSS-RISK ELEVATED"
            action = "Predicted stop-risk is elevated; reduce size, wait, or reject the setup."
        else:
            verdict = "REVIEW / WAIT"
            action = "Prediction is mixed; wait for stronger confirmation or lower risk."
    elif bias == "Bearish":
        p_down = None if p_up is None else round(100 - p_up, 2)
        dir_prob = p_down
        stop_prob = p_short_stop
        tp_prob = p_short_tp
        aligned = dir_prob is not None and dir_prob >= 58
        loss_ok = stop_prob is not None and stop_prob <= 48
        target_ok = tp_prob is not None and tp_prob >= 28
        if aligned and loss_ok and target_ok and risk_severity not in {"HIGH", "CRITICAL"}:
            verdict = "PREDICTIVE SUPPORT"
            action = "Bearish plan has model support, but use the defined stop and position size."
        elif dir_prob is not None and dir_prob < 45:
            verdict = "PREDICTION CONFLICT"
            action = "Bearish technical bias is not supported by the forecast model; avoid rushing entry."
        elif stop_prob is not None and stop_prob > 58:
            verdict = "LOSS-RISK ELEVATED"
            action = "Predicted stop-risk is elevated; reduce size, wait, or reject the setup."
        else:
            verdict = "REVIEW / WAIT"
            action = "Prediction is mixed; wait for stronger confirmation or lower risk."
    else:
        verdict = "NO DIRECTIONAL PLAN"
        action = "No strong bullish/bearish bias exists, so prediction should not be used for a trade entry."
    return {"prediction_verdict": verdict, "loss_control_action": action}


def run_prediction_engine(
    df: pd.DataFrame,
    bias: str = "Neutral",
    horizon: int = 5,
    stop_atr: float = 1.5,
    target_rr: float = 3.0,
    risk_severity: str | None = None,
) -> Dict[str, Any]:
    """
    Train symbol-specific walk-forward-aware models and forecast the latest bar.

    This is a decision-support forecast, not a guarantee of future returns.
    """
    frame = build_prediction_frame(df, horizon=horizon, stop_atr=stop_atr, target_rr=target_rr)
    if frame.empty:
        return {"status": "UNAVAILABLE", "message": "No data available for prediction."}

    feature_ready = frame.dropna(subset=FEATURE_COLUMNS + ["future_return_pct"]).copy()
    # Exclude the last horizon rows from supervised training because targets are incomplete.
    supervised = feature_ready.iloc[:-horizon].copy() if len(feature_ready) > horizon else pd.DataFrame()
    if supervised.empty or len(supervised) < 120:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "At least ~120 usable candles are needed for a more credible forecast.",
            "usable_rows": int(len(supervised)),
        }

    latest = frame.dropna(subset=FEATURE_COLUMNS).tail(1)
    if latest.empty:
        return {"status": "UNAVAILABLE", "message": "Latest feature row is incomplete."}

    X = supervised[FEATURE_COLUMNS].copy()
    latest_x = latest[FEATURE_COLUMNS].copy()

    targets = {
        "direction_up": supervised["target_up"],
        "long_stop_hit": supervised["target_long_stop_hit"],
        "long_tp_hit": supervised["target_long_tp_hit"],
        "short_stop_hit": supervised["target_short_stop_hit"],
        "short_tp_hit": supervised["target_short_tp_hit"],
    }

    models: Dict[str, Any] = {}
    validations: Dict[str, Any] = {}
    calibrated_flags: Dict[str, bool] = {}
    probabilities: Dict[str, float | None] = {}

    for name, y in targets.items():
        y = y.astype(int)
        if _usable_binary_target(y):
            model, metrics, calibrated = _fit_classifier(X, y, calibrate=True)
            models[name] = model
            validations[name] = metrics
            calibrated_flags[name] = calibrated
            probabilities[name] = _prob(model, latest_x)
        else:
            validations[name] = {"accuracy": None, "brier": None, "folds": 0}
            calibrated_flags[name] = False
            probabilities[name] = None

    regressor = _fit_regressor(X, supervised["future_return_pct"].astype(float))
    expected_return = round(float(regressor.predict(latest_x)[0]), 3)

    p_up = probabilities.get("direction_up")
    p_down = None if p_up is None else round(100 - p_up, 2)
    p_long_stop = probabilities.get("long_stop_hit")
    p_long_tp = probabilities.get("long_tp_hit")
    p_short_stop = probabilities.get("short_stop_hit")
    p_short_tp = probabilities.get("short_tp_hit")

    verdict = _build_loss_control_verdict(
        bias=bias,
        p_up=p_up,
        p_long_stop=p_long_stop,
        p_long_tp=p_long_tp,
        p_short_stop=p_short_stop,
        p_short_tp=p_short_tp,
        risk_severity=risk_severity,
    )

    latest_close = float(latest["close"].iloc[-1])
    atr = float(latest["atr14"].iloc[-1]) if pd.notna(latest["atr14"].iloc[-1]) else None
    if atr is not None:
        forecast_up_zone = round(latest_close + atr * stop_atr * target_rr, 4)
        forecast_down_zone = round(latest_close - atr * stop_atr * target_rr, 4)
        stop_long_zone = round(latest_close - atr * stop_atr, 4)
        stop_short_zone = round(latest_close + atr * stop_atr, 4)
    else:
        forecast_up_zone = forecast_down_zone = stop_long_zone = stop_short_zone = None

    direction_metrics = validations.get("direction_up", {})
    status = "OK" if p_up is not None else "PARTIAL"

    return {
        "status": status,
        "message": "Forecast generated from symbol-specific time-series features.",
        "horizon_bars": horizon,
        "stop_atr": stop_atr,
        "target_rr": target_rr,
        "usable_rows": int(len(supervised)),
        "latest_close": round(latest_close, 4),
        "expected_return_pct": expected_return,
        "probability_up_pct": p_up,
        "probability_down_pct": p_down,
        "probability_long_stop_hit_pct": p_long_stop,
        "probability_long_tp_hit_pct": p_long_tp,
        "probability_short_stop_hit_pct": p_short_stop,
        "probability_short_tp_hit_pct": p_short_tp,
        "forecast_up_zone": forecast_up_zone,
        "forecast_down_zone": forecast_down_zone,
        "atr_long_stop_zone": stop_long_zone,
        "atr_short_stop_zone": stop_short_zone,
        "validation_direction_accuracy": direction_metrics.get("accuracy"),
        "validation_direction_brier": direction_metrics.get("brier"),
        "validation_folds": direction_metrics.get("folds"),
        "direction_probability_calibrated": calibrated_flags.get("direction_up", False),
        "model_probabilities": probabilities,
        "model_validations": validations,
        **verdict,
    }


def prediction_summary_table(prediction: Dict[str, Any]) -> pd.DataFrame:
    if not prediction or prediction.get("status") not in {"OK", "PARTIAL"}:
        return pd.DataFrame([{
            "Status": prediction.get("status", "UNAVAILABLE") if prediction else "UNAVAILABLE",
            "Message": prediction.get("message", "No prediction."),
        }])

    rows = [
        {"Metric": "Forecast Horizon", "Value": f'{prediction["horizon_bars"]} bars'},
        {"Metric": "Expected Return %", "Value": prediction["expected_return_pct"]},
        {"Metric": "Probability Up %", "Value": prediction["probability_up_pct"]},
        {"Metric": "Probability Down %", "Value": prediction["probability_down_pct"]},
        {"Metric": "Long Stop Hit Probability %", "Value": prediction["probability_long_stop_hit_pct"]},
        {"Metric": "Long TP Hit Probability %", "Value": prediction["probability_long_tp_hit_pct"]},
        {"Metric": "Short Stop Hit Probability %", "Value": prediction["probability_short_stop_hit_pct"]},
        {"Metric": "Short TP Hit Probability %", "Value": prediction["probability_short_tp_hit_pct"]},
        {"Metric": "Direction Validation Accuracy", "Value": prediction["validation_direction_accuracy"]},
        {"Metric": "Direction Validation Brier", "Value": prediction["validation_direction_brier"]},
        {"Metric": "Probability Calibrated", "Value": prediction["direction_probability_calibrated"]},
        {"Metric": "Prediction Verdict", "Value": prediction["prediction_verdict"]},
        {"Metric": "Loss-Control Action", "Value": prediction["loss_control_action"]},
    ]
    return pd.DataFrame(rows)
