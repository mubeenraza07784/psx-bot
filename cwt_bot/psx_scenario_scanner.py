from __future__ import annotations

from typing import Callable, Any, Optional
import pandas as pd

from .psx_data import (
    fetch_psx_symbol_universe,
    load_psx_yahoo_ohlcv,
    load_psx_dps_ohlcv,
    resample_ohlcv,
)
from .signals import analyze_symbol
from .pro_metrics import evaluate_psx_pro_score
from .risk_alerts import build_risk_warning, concise_risk_columns


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _parse_custom_symbols(text: str) -> list[str]:
    raw = text.replace("\n", ",").replace(";", ",")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return list(dict.fromkeys(symbols))


def resolve_symbol_universe(
    universe: str,
    custom_symbols_text: str = "",
) -> list[str]:
    if universe == "Custom Symbols":
        symbols = _parse_custom_symbols(custom_symbols_text)
        if not symbols:
            raise ValueError("Enter at least one custom PSX symbol.")
        return symbols
    return fetch_psx_symbol_universe(universe)


def scenario_number_from_label(label: str) -> str:
    if label.startswith("Scenario 1"):
        return "Scenario 1"
    if label.startswith("Scenario 2"):
        return "Scenario 2"
    if label.startswith("Scenario 3"):
        return "Scenario 3"
    return "Other"


def scan_psx_for_scenario(
    selected_scenario: str,
    scenario_system: str,
    universe: str,
    custom_symbols_text: str,
    data_source: str,
    analysis_tf: str,
    execution_tf: str,
    period: str,
    dps_mode: str,
    risk_context: dict[str, Any] | None = None,
    max_symbols: int = 0,
    progress_callback: ProgressCallback = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Scan a PSX universe and return:
    - matched scenario rows

    scenario_system options:
    - "MTF Trade Scenario (Week 5/6)"
    - "CWT Setup Scenario (Week 2/3/4)"
    - all successfully analyzed rows
    - failed symbol messages
    """
    symbols = resolve_symbol_universe(universe, custom_symbols_text)
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]

    matches: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    total = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        if progress_callback:
            progress_callback(i, total, symbol)

        try:
            if data_source == "Yahoo Finance PSX (.KA)":
                higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
                lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
            else:
                base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
                higher_df = resample_ohlcv(base_df, analysis_tf)
                lower_df = resample_ohlcv(base_df, execution_tf)

            if higher_df.empty or lower_df.empty:
                raise ValueError("No OHLCV rows after loading/resampling.")

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

            plan = result["trade_plan"]
            mtf_scenario = result.get("mtf_scenario", result["scenario"])
            cwt_scenario = result.get("cwt_scenario", {"number": "Other", "label": "Unavailable"})

            if scenario_system == "CWT Setup Scenario (Week 2/3/4)":
                active_scenario = cwt_scenario
            else:
                active_scenario = mtf_scenario

            scenario_label = active_scenario.get("label", "Other")
            scenario_number = active_scenario.get("number", scenario_number_from_label(scenario_label))

            row = {
                "Symbol": symbol,
                "Scenario System": active_scenario.get("system", scenario_system),
                "Scenario": scenario_number,
                "Scenario Detail": scenario_label,
                "Scenario Side": active_scenario.get("side", "Neutral"),
                "Scenario Rule": active_scenario.get("rule", ""),
                "MTF Scenario": mtf_scenario.get("label", ""),
                "CWT Setup Scenario": cwt_scenario.get("label", ""),
                "Bias": result["signal"]["bias"],
                "Action": result["signal"]["action"],
                "Setup": result["signal"]["setup_type"],
                "Confidence": result["signal"]["confidence"],
                "Pro Score": pro["pro_score"],
                "Grade": pro["pro_grade"],
                "Trade Quality": pro["trade_quality"],
                "Trend Stack": pro["trend_stack"],
                "Risk": pro["risk_level"],
                "Momentum": pro["momentum_state"],
                "Volume": pro["volume_state"],
                "ADX State": pro["adx_state"],
                "RSI": pro["latest_metrics"].get("rsi"),
                "ADX": pro["latest_metrics"].get("adx"),
                "Vol Ratio": pro["latest_metrics"].get("volume_ratio"),
                "ATR %": pro["latest_metrics"].get("atr_pct"),
                "Higher Trend": result["higher_trend"]["trend"],
                "Execution Trend": result["execution_trend"]["trend"],
                "Market Phase": result["higher_trend"]["phase"],
                "Alligator": result["cwt"]["state"],
                "Divergence": result["divergence"]["label"],
                "Order": plan["order_type"],
                "Entry": plan["entry"],
                "SL": plan["stop_loss"],
                "TP 1:3": plan["take_profit"],
                "Warnings": " | ".join(result["warnings"]),
            }
            row.update(concise_risk_columns(risk))
            all_rows.append(row)
            if scenario_number == selected_scenario:
                matches.append(row)

        except Exception as exc:
            failures.append(f"{symbol}: {exc}")

    matched_df = pd.DataFrame(matches)
    all_df = pd.DataFrame(all_rows)

    if not matched_df.empty:
        matched_df = matched_df.sort_values(
            by=["Pro Score", "Confidence", "Symbol"],
            ascending=[False, False, True],
            na_position="last",
        )
    if not all_df.empty:
        all_df = all_df.sort_values(
            by=["Scenario", "Confidence", "Symbol"],
            ascending=[True, False, True],
            na_position="last",
        )
    return matched_df, all_df, failures
