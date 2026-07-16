from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / max(length, 1), adjust=False, min_periods=length).mean()


def _safe_last(series: pd.Series, default=None):
    if series is None or len(series) == 0:
        return default
    value = series.iloc[-1]
    if pd.isna(value):
        return default
    return float(value)


def add_psx_pro_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float) if "volume" in out.columns else pd.Series(0.0, index=out.index)

    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma50"] = close.rolling(50, min_periods=50).mean()
    out["ma200"] = close.rolling(200, min_periods=200).mean()
    out["ema20"] = _ema(close, 20)
    out["ema50"] = _ema(close, 50)

    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd_line, 9)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width_pct"] = ((out["bb_upper"] - out["bb_lower"]) / close.replace(0, np.nan)) * 100

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr14 = _wilder(tr, 14)
    plus_di = 100 * _wilder(plus_dm, 14) / atr14.replace(0, np.nan)
    minus_di = 100 * _wilder(minus_dm, 14) / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder(dx, 14)

    out["atr14"] = atr14
    out["atr_pct"] = (atr14 / close.replace(0, np.nan)) * 100
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = adx

    out["avg_volume_20"] = volume.rolling(20, min_periods=1).mean()
    out["volume_ratio"] = volume / out["avg_volume_20"].replace(0, np.nan)
    out["return_1"] = close.pct_change() * 100
    out["return_5"] = close.pct_change(5) * 100
    out["return_20"] = close.pct_change(20) * 100
    out["recent_high_20"] = high.rolling(20, min_periods=5).max()
    out["recent_low_20"] = low.rolling(20, min_periods=5).min()
    out["breakout_up"] = close >= out["recent_high_20"].shift(1)
    out["breakdown_down"] = close <= out["recent_low_20"].shift(1)

    return out


def trend_stack_label(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Unclear"
    row = df.iloc[-1]
    close = float(row["close"])
    ma20 = row.get("ma20")
    ma50 = row.get("ma50")
    ma200 = row.get("ma200")

    if pd.notna(ma20) and pd.notna(ma50) and pd.notna(ma200):
        if close > ma20 > ma50 > ma200:
            return "Strong Bull Stack"
        if close < ma20 < ma50 < ma200:
            return "Strong Bear Stack"

    if pd.notna(ma20) and pd.notna(ma50):
        if close > ma20 > ma50:
            return "Bullish"
        if close < ma20 < ma50:
            return "Bearish"
    return "Mixed"


def _grade(score: float) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B+"
    if score >= 55:
        return "B"
    if score >= 45:
        return "C"
    return "D"


def _risk_level(atr_pct):
    if atr_pct is None:
        return "Unknown"
    if atr_pct <= 2:
        return "Low"
    if atr_pct <= 4:
        return "Moderate"
    if atr_pct <= 7:
        return "High"
    return "Very High"


def evaluate_psx_pro_score(result: Dict[str, Any]) -> Dict[str, Any]:
    frame = result.get("execution_frame")
    if frame is None or frame.empty:
        return {
            "pro_score": None,
            "pro_grade": "N/A",
            "trend_stack": "Unclear",
            "risk_level": "Unknown",
            "momentum_state": "Unknown",
            "volume_state": "Unknown",
            "adx_state": "Unknown",
            "trade_quality": "Unclear",
            "technical_notes": "",
            "latest_metrics": {},
        }

    df = add_psx_pro_metrics(frame)
    result["execution_frame"] = df
    row = df.iloc[-1]

    base_conf = float(result.get("signal", {}).get("confidence", 0) or 0)
    score = base_conf * 0.50
    notes: list[str] = [f"CWT confidence {base_conf:.0f}/100 contributes 50%."]

    scenario = result.get("scenario", {}).get("label", "")
    if scenario.startswith("Scenario 1"):
        score += 18
        notes.append("Scenario 1 alignment +18.")
    elif scenario.startswith("Scenario 2"):
        score += 10
        notes.append("Scenario 2 pullback context +10.")
    elif scenario.startswith("Scenario 3"):
        score += 6
        notes.append("Scenario 3 sideways execution context +6.")

    stack = trend_stack_label(df)
    bias = result.get("signal", {}).get("bias", "Neutral")
    if stack == "Strong Bull Stack" and bias == "Bullish":
        score += 12
        notes.append("Bullish MA stack aligns +12.")
    elif stack == "Strong Bear Stack" and bias == "Bearish":
        score += 12
        notes.append("Bearish MA stack aligns +12.")
    elif stack == "Bullish" and bias == "Bullish":
        score += 7
        notes.append("Directional MA structure aligns +7.")
    elif stack == "Bearish" and bias == "Bearish":
        score += 7
        notes.append("Directional MA structure aligns +7.")

    rsi = _safe_last(df["rsi"], 50.0)
    macd_hist = _safe_last(df["macd_hist"], 0.0)
    volume_ratio = _safe_last(df["volume_ratio"], None)
    adx = _safe_last(df["adx"], None)
    atr_pct = _safe_last(df["atr_pct"], None)
    ret5 = _safe_last(df["return_5"], None)

    if bias == "Bullish":
        if rsi is not None and 50 <= rsi <= 70:
            score += 7
            notes.append("RSI bullish but not overbought +7.")
        if macd_hist is not None and macd_hist > 0:
            score += 7
            notes.append("MACD histogram positive +7.")
    elif bias == "Bearish":
        if rsi is not None and 30 <= rsi <= 50:
            score += 7
            notes.append("RSI bearish but not oversold +7.")
        if macd_hist is not None and macd_hist < 0:
            score += 7
            notes.append("MACD histogram negative +7.")

    if volume_ratio is not None and volume_ratio >= 1.2:
        score += 6
        notes.append("Volume confirmation ≥1.2x +6.")
    elif volume_ratio is not None and volume_ratio < 0.7:
        score -= 4
        notes.append("Weak relative volume -4.")

    if adx is not None and adx >= 25:
        score += 6
        notes.append("ADX trend strength ≥25 +6.")
    elif adx is not None and adx < 15:
        score -= 4
        notes.append("Weak ADX <15 -4.")

    if bool(row.get("breakout_up", False)) and bias == "Bullish":
        score += 6
        notes.append("20-candle upside breakout +6.")
    if bool(row.get("breakdown_down", False)) and bias == "Bearish":
        score += 6
        notes.append("20-candle downside breakdown +6.")

    score = float(np.clip(score, 0, 100))
    grade = _grade(score)
    risk = _risk_level(atr_pct)

    if bias == "Bullish":
        momentum_state = "Bullish" if (rsi or 0) >= 50 and (macd_hist or 0) >= 0 else "Mixed"
    elif bias == "Bearish":
        momentum_state = "Bearish" if (rsi or 100) <= 50 and (macd_hist or 0) <= 0 else "Mixed"
    else:
        momentum_state = "Neutral"

    if volume_ratio is None:
        volume_state = "Unknown"
    elif volume_ratio >= 1.5:
        volume_state = "Strong"
    elif volume_ratio >= 1.0:
        volume_state = "Normal"
    else:
        volume_state = "Weak"

    if adx is None:
        adx_state = "Unknown"
    elif adx >= 25:
        adx_state = "Trending"
    elif adx >= 18:
        adx_state = "Developing"
    else:
        adx_state = "Weak Trend"

    if score >= 75 and risk in {"Low", "Moderate"}:
        trade_quality = "High Quality"
    elif score >= 60:
        trade_quality = "Actionable / Review"
    elif score >= 45:
        trade_quality = "Watchlist"
    else:
        trade_quality = "Low Quality"

    latest = {
        "rsi": None if rsi is None else round(rsi, 2),
        "macd_hist": None if macd_hist is None else round(macd_hist, 4),
        "volume_ratio": None if volume_ratio is None else round(volume_ratio, 2),
        "adx": None if adx is None else round(adx, 2),
        "atr_pct": None if atr_pct is None else round(atr_pct, 2),
        "return_5": None if ret5 is None else round(ret5, 2),
        "close": round(float(row["close"]), 4),
    }

    return {
        "pro_score": round(score, 2),
        "pro_grade": grade,
        "trend_stack": stack,
        "risk_level": risk,
        "momentum_state": momentum_state,
        "volume_state": volume_state,
        "adx_state": adx_state,
        "trade_quality": trade_quality,
        "technical_notes": " | ".join(notes),
        "latest_metrics": latest,
    }
