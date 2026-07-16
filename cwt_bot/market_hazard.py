from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def detect_price_hazards(
    df: pd.DataFrame,
    symbol: str = "",
    support_resistance: Dict[str, Any] | None = None,
    thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Detect risk conditions that often appear before/around adverse price moves:
    - gap shocks
    - abnormal true range
    - sudden volume spikes or dry liquidity
    - sharp drawdowns from recent highs
    - closes near support/resistance
    - volatility expansion after compression
    """
    thresholds = thresholds or {}
    gap_threshold_pct = thresholds.get("gap_threshold_pct", 2.5)
    range_atr_multiple = thresholds.get("range_atr_multiple", 2.0)
    volume_spike_ratio = thresholds.get("volume_spike_ratio", 2.0)
    low_volume_ratio = thresholds.get("low_volume_ratio", 0.55)
    drawdown_threshold_pct = thresholds.get("drawdown_threshold_pct", -8.0)
    support_buffer_atr = thresholds.get("support_buffer_atr", 0.5)

    if df is None or df.empty or len(df) < 25:
        return {
            "status": "INSUFFICIENT_DATA",
            "symbol": symbol,
            "hazard_level": "UNKNOWN",
            "alerts": [],
            "summary": "Not enough candles for hazard detection.",
        }

    work = df.copy()
    close = work["close"].astype(float)
    open_ = work["open"].astype(float)
    high = work["high"].astype(float)
    low = work["low"].astype(float)
    volume = work["volume"].astype(float) if "volume" in work.columns else pd.Series(0.0, index=work.index)
    prev_close = close.shift(1)

    if "atr14" in work.columns:
        atr = work["atr14"].astype(float)
    elif "atr" in work.columns:
        atr = work["atr"].astype(float)
    else:
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    avg_vol = volume.rolling(20, min_periods=5).mean()
    volume_ratio = volume / avg_vol.replace(0, np.nan)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    range_atr_ratio = true_range / atr.replace(0, np.nan)
    gap_pct = ((open_ - prev_close) / prev_close.replace(0, np.nan)) * 100
    candle_return_pct = close.pct_change() * 100
    recent_peak = close.rolling(20, min_periods=10).max()
    drawdown_pct = ((close - recent_peak) / recent_peak.replace(0, np.nan)) * 100

    alerts: List[Dict[str, Any]] = []
    points = 0

    latest = work.index[-1]
    latest_close = float(close.iloc[-1])
    latest_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else max(abs(latest_close)*0.02, 1e-9)
    latest_gap = float(gap_pct.iloc[-1]) if pd.notna(gap_pct.iloc[-1]) else 0.0
    latest_range_ratio = float(range_atr_ratio.iloc[-1]) if pd.notna(range_atr_ratio.iloc[-1]) else 0.0
    latest_vol_ratio = float(volume_ratio.iloc[-1]) if pd.notna(volume_ratio.iloc[-1]) else None
    latest_dd = float(drawdown_pct.iloc[-1]) if pd.notna(drawdown_pct.iloc[-1]) else 0.0
    latest_ret = float(candle_return_pct.iloc[-1]) if pd.notna(candle_return_pct.iloc[-1]) else 0.0

    if abs(latest_gap) >= gap_threshold_pct:
        severity = "HIGH" if abs(latest_gap) >= gap_threshold_pct * 1.6 else "MODERATE"
        alerts.append({
            "Type": "Gap Shock",
            "Severity": severity,
            "Value": round(latest_gap, 2),
            "Message": f"Opening gap of {latest_gap:+.2f}% detected versus previous close.",
        })
        points += 3 if severity == "HIGH" else 2

    if latest_range_ratio >= range_atr_multiple:
        severity = "HIGH" if latest_range_ratio >= range_atr_multiple * 1.5 else "MODERATE"
        alerts.append({
            "Type": "Range Expansion",
            "Severity": severity,
            "Value": round(latest_range_ratio, 2),
            "Message": f"True range is {latest_range_ratio:.2f}x ATR, signalling unusual volatility.",
        })
        points += 3 if severity == "HIGH" else 2

    if latest_vol_ratio is not None and latest_vol_ratio >= volume_spike_ratio:
        alerts.append({
            "Type": "Volume Spike",
            "Severity": "MODERATE",
            "Value": round(latest_vol_ratio, 2),
            "Message": f"Volume is {latest_vol_ratio:.2f}x its 20-period average.",
        })
        points += 1

    if latest_vol_ratio is not None and latest_vol_ratio <= low_volume_ratio:
        alerts.append({
            "Type": "Thin Liquidity",
            "Severity": "MODERATE",
            "Value": round(latest_vol_ratio, 2),
            "Message": f"Volume is only {latest_vol_ratio:.2f}x average; liquidity confirmation is weak.",
        })
        points += 1

    if latest_dd <= drawdown_threshold_pct:
        alerts.append({
            "Type": "Drawdown",
            "Severity": "HIGH" if latest_dd <= drawdown_threshold_pct * 1.5 else "MODERATE",
            "Value": round(latest_dd, 2),
            "Message": f"Price is {latest_dd:.2f}% below its recent 20-candle peak.",
        })
        points += 2

    if abs(latest_ret) >= 4:
        alerts.append({
            "Type": "Sharp Candle Move",
            "Severity": "HIGH" if abs(latest_ret) >= 6 else "MODERATE",
            "Value": round(latest_ret, 2),
            "Message": f"Latest candle return is {latest_ret:+.2f}%, indicating abrupt repricing.",
        })
        points += 2 if abs(latest_ret) >= 6 else 1

    sr = support_resistance or {}
    nearest_support = sr.get("nearest_support") or (sr.get("supports", [])[-1] if sr.get("supports") else None)
    nearest_resistance = sr.get("nearest_resistance") or (sr.get("resistances", [None])[0] if sr.get("resistances") else None)
    if nearest_support is not None:
        distance = latest_close - float(nearest_support)
        if 0 <= distance <= latest_atr * support_buffer_atr:
            alerts.append({
                "Type": "Support Proximity",
                "Severity": "MODERATE",
                "Value": round(distance, 4),
                "Message": "Price is very close to support; breakdown risk requires attention.",
            })
            points += 1
    if nearest_resistance is not None:
        distance = float(nearest_resistance) - latest_close
        if 0 <= distance <= latest_atr * support_buffer_atr:
            alerts.append({
                "Type": "Resistance Proximity",
                "Severity": "LOW",
                "Value": round(distance, 4),
                "Message": "Price is close to resistance; breakout or rejection risk is elevated.",
            })
            points += 0.5

    # Volatility compression -> expansion watch
    atr_pct = (atr / close.replace(0, np.nan)) * 100
    recent_atr = atr_pct.tail(5).mean()
    older_atr = atr_pct.tail(25).head(20).mean() if len(atr_pct) >= 25 else None
    if older_atr is not None and pd.notna(older_atr) and pd.notna(recent_atr):
        if recent_atr > older_atr * 1.35:
            alerts.append({
                "Type": "Volatility Expansion",
                "Severity": "MODERATE",
                "Value": round(float(recent_atr), 2),
                "Message": "ATR% is expanding materially versus its prior baseline.",
            })
            points += 1.5

    if points >= 7:
        level = "CRITICAL"
    elif points >= 4:
        level = "HIGH"
    elif points >= 2:
        level = "MODERATE"
    else:
        level = "LOW"

    if level == "CRITICAL":
        action = "Avoid fresh entries unless risk is specifically accepted; review news, liquidity, and stop placement immediately."
    elif level == "HIGH":
        action = "Reduce size or wait for stabilization/confirmation before acting."
    elif level == "MODERATE":
        action = "Trade only with clear entry, stop, and valid confirmation."
    else:
        action = "No major hazard alert beyond normal market risk."

    return {
        "status": "OK",
        "symbol": symbol,
        "hazard_level": level,
        "hazard_points": round(points, 2),
        "alerts": alerts,
        "summary": action,
        "latest_gap_pct": round(latest_gap, 2),
        "latest_range_atr_ratio": round(latest_range_ratio, 2),
        "latest_volume_ratio": None if latest_vol_ratio is None else round(latest_vol_ratio, 2),
        "latest_drawdown_pct": round(latest_dd, 2),
        "latest_candle_return_pct": round(latest_ret, 2),
    }


def hazard_alerts_dataframe(hazard: Dict[str, Any]) -> pd.DataFrame:
    alerts = hazard.get("alerts", []) if hazard else []
    return pd.DataFrame(alerts) if alerts else pd.DataFrame([{"Message": hazard.get("summary", "No hazard data.") if hazard else "No hazard data."}])
