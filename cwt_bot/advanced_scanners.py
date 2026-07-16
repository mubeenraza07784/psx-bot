from __future__ import annotations

from typing import Any, Callable, Optional
import pandas as pd

from .psx_data import fetch_psx_symbol_universe, load_psx_yahoo_ohlcv, load_psx_dps_ohlcv, resample_ohlcv
from .signals import analyze_symbol
from .pro_metrics import evaluate_psx_pro_score
from .risk_alerts import build_risk_warning, concise_risk_columns
from .state_alerts import check_and_update_symbol_alerts


ProgressCallback = Optional[Callable[[int, int, str], None]]


def parse_symbols(text: str) -> list[str]:
    raw = text.replace("\n", ",").replace(";", ",")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return list(dict.fromkeys(symbols))


def resolve_symbols(universe: str, custom_symbols_text: str) -> list[str]:
    if universe == "Custom Symbols":
        symbols = parse_symbols(custom_symbols_text)
        if not symbols:
            raise ValueError("Enter at least one custom PSX symbol.")
        return symbols
    return fetch_psx_symbol_universe(universe)


def _load_symbol_frames(
    symbol: str,
    data_source: str,
    analysis_tf: str,
    execution_tf: str,
    period: str,
    dps_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data_source == "Yahoo Finance PSX (.KA)":
        higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
        lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
        return higher_df, lower_df

    base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
    return resample_ohlcv(base_df, analysis_tf), resample_ohlcv(base_df, execution_tf)


def _common_row(symbol: str, result: dict[str, Any], pro: dict[str, Any], risk: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = result["trade_plan"]
    latest = pro.get("latest_metrics", {})
    row = {
        "Symbol": symbol,
        "Bias": result["signal"]["bias"],
        "Action": result["signal"]["action"],
        "Setup": result["signal"]["setup_type"],
        "MTF Scenario": result.get("mtf_scenario", result["scenario"])["label"],
        "CWT Setup Scenario": result.get("cwt_scenario", {}).get("label", ""),
        "Confidence": result["signal"]["confidence"],
        "Pro Score": pro["pro_score"],
        "Grade": pro["pro_grade"],
        "Trade Quality": pro["trade_quality"],
        "Trend Stack": pro["trend_stack"],
        "Risk": pro["risk_level"],
        "Momentum": pro["momentum_state"],
        "Volume": pro["volume_state"],
        "ADX State": pro["adx_state"],
        "RSI": latest.get("rsi"),
        "ADX": latest.get("adx"),
        "Vol Ratio": latest.get("volume_ratio"),
        "ATR %": latest.get("atr_pct"),
        "5P Return %": latest.get("return_5"),
        "Order": plan["order_type"],
        "Entry": plan["entry"],
        "SL": plan["stop_loss"],
        "TP 1:3": plan["take_profit"],
        "Divergence": result["divergence"]["label"],
        "Warnings": " | ".join(result["warnings"]),
    }
    if risk:
        row.update(concise_risk_columns(risk))
    return row


def scan_watchlist_pro(
    symbols: list[str],
    data_source: str,
    analysis_tf: str,
    execution_tf: str,
    period: str,
    dps_mode: str,
    risk_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback = None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    alert_rows: list[dict[str, Any]] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(i, total, symbol)
        try:
            higher_df, lower_df = _load_symbol_frames(symbol, data_source, analysis_tf, execution_tf, period, dps_mode)
            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher_df,
                lower_df=lower_df,
                asset_class="Stock",
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                risk_context=risk_context or {},
            )
            pro = evaluate_psx_pro_score(result)
            risk = build_risk_warning(
                result,
                pro,
                user_event_risk=bool((risk_context or {}).get("high_impact_news")),
                benchmark_conflict=bool((risk_context or {}).get("benchmark_conflict")),
            )
            alert_result = check_and_update_symbol_alerts(
                symbol=symbol,
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                result=result,
                pro=pro,
                risk=risk,
            )
            for alert in alert_result["alerts"]:
                alert_rows.append({"Symbol": symbol, **alert})
            rows.append(_common_row(symbol, result, pro, risk))
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Pro Score", "Confidence", "Symbol"], ascending=[False, False, True], na_position="last")
    alerts_df = pd.DataFrame(alert_rows)
    if not alerts_df.empty:
        alerts_df = alerts_df.sort_values(["Severity", "Symbol"], ascending=[True, True])
    return df, failures, alerts_df


def scan_patterns(
    universe: str,
    custom_symbols_text: str,
    data_source: str,
    analysis_tf: str,
    execution_tf: str,
    period: str,
    dps_mode: str,
    target: str,
    min_pro_score: float,
    max_symbols: int,
    risk_context: dict[str, Any] | None = None,
    progress_callback: ProgressCallback = None,
) -> tuple[pd.DataFrame, list[str]]:
    symbols = resolve_symbols(universe, custom_symbols_text)
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(i, total, symbol)
        try:
            higher_df, lower_df = _load_symbol_frames(symbol, data_source, analysis_tf, execution_tf, period, dps_mode)
            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher_df,
                lower_df=lower_df,
                asset_class="Stock",
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                risk_context=risk_context or {},
            )
            pro = evaluate_psx_pro_score(result)
            risk = build_risk_warning(
                result,
                pro,
                user_event_risk=bool((risk_context or {}).get("high_impact_news")),
                benchmark_conflict=bool((risk_context or {}).get("benchmark_conflict")),
            )
            if pro["pro_score"] is not None and pro["pro_score"] < min_pro_score:
                continue

            patterns = result["patterns"]
            reversal_names = [p.get("name", "") for p in patterns.get("reversal", [])]
            continuation_names = [p.get("name", "") for p in patterns.get("continuation", [])]
            candle_names = [p.get("name", "") for p in patterns.get("candlestick", [])]
            divergence = result["divergence"]["label"]

            include = False
            if target == "Any Reversal Pattern":
                include = bool(reversal_names)
            elif target == "Any Continuation Pattern":
                include = bool(continuation_names)
            elif target == "Bullish RSI Divergence":
                include = divergence == "Bullish RSI Divergence"
            elif target == "Bearish RSI Divergence":
                include = divergence == "Bearish RSI Divergence"
            elif target == "Any Candlestick Pattern":
                include = bool(candle_names)
            elif target in set(reversal_names + continuation_names + candle_names):
                include = True
            elif target == "High-Quality Bullish Setups":
                include = result["signal"]["bias"] == "Bullish" and pro["trade_quality"] in {"High Quality", "Actionable / Review"}
            elif target == "High-Quality Bearish Setups":
                include = result["signal"]["bias"] == "Bearish" and pro["trade_quality"] in {"High Quality", "Actionable / Review"}

            if include:
                row = _common_row(symbol, result, pro, risk)
                row["Reversal Patterns"] = ", ".join(reversal_names)
                row["Continuation Patterns"] = ", ".join(continuation_names)
                row["Candlestick Patterns"] = ", ".join(candle_names)
                rows.append(row)

        except Exception as exc:
            failures.append(f"{symbol}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Pro Score", "Confidence", "Symbol"], ascending=[False, False, True], na_position="last")
    return df, failures
