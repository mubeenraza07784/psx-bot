from __future__ import annotations

from pathlib import Path
import json
import re
import time
import requests
from datetime import timedelta
from urllib.parse import quote
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from cwt_bot.psx_data import (
    load_psx_dps_ohlcv,
    load_psx_yahoo_ohlcv,
    load_psx_csv,
    resample_ohlcv,
)
from cwt_bot.psx_scenario_scanner import scan_psx_for_scenario
from cwt_bot.advanced_scanners import scan_patterns, scan_watchlist_pro, parse_symbols
from cwt_bot.signals import analyze_symbol
from cwt_bot.divergence import detect_rsi_divergence
from cwt_bot.pro_metrics import evaluate_psx_pro_score
from cwt_bot.fundamentals import prepare_fundamentals, build_fundamental_rankings
from cwt_bot.risk import risk_profile_for_execution_tf, calc_position_size
from cwt_bot.risk_alerts import build_risk_warning
from cwt_bot.state_alerts import check_and_update_symbol_alerts, alerts_to_dataframe, load_alert_state_table
from cwt_bot.prediction_engine import run_prediction_engine, prediction_summary_table
from cwt_bot.trade_tracker import append_trade_plan, load_trade_tracker, update_trade_row, tracker_summary
from cwt_bot.strategy_profiles import TRADING_PROFILES, get_profile, profile_table
from cwt_bot.fib_engine import fibonacci_retracement_confluence
from cwt_bot.secular_engine import classify_secular_trend
from cwt_bot.market_hazard import detect_price_hazards, hazard_alerts_dataframe
from cwt_bot.event_risk_monitor import build_news_event_risk_snapshot
from cwt_bot.confluence_engine import build_confluence_score, confluence_table
from cwt_bot.investor_profile_engine import recommend_profile, profile_catalog
from cwt_bot.fundamental_master import normalize_fundamental_columns, score_fundamental_universe, score_single_symbol
from cwt_bot.intrinsic_value_engine import intrinsic_value_composite
from cwt_bot.fraud_risk_engine import score_financial_fakery
from cwt_bot.strategy_engine import build_all_strategies, strategy_summary
from cwt_bot.catalyst_engine import scan_catalyst_text, catalyst_from_events
from cwt_bot.macro_checklist_engine import macro_checklist_score, company_macro_sensitivity
from cwt_bot.autopilot_manager import run_autopilot_cycle, persist_autopilot_outputs
from cwt_bot.knowledge_integrator import knowledge_registry_table, engine_coverage_table, knowledge_status_summary
from cwt_bot.knowledge_selector import (
    KNOWLEDGE_LAYERS,
    KNOWLEDGE_PROFILES,
    layers_for_profile,
    selector_summary,
    selected_layer_table,
    apply_knowledge_columns,
    required_data_for_layers,
)
from cwt_bot.external_links import read_google_sheet_table, read_google_drive_portfolio_pdf
from cwt_bot.structures import swing_points
from cwt_bot.fundamental_image_importer import (
    IMAGE_SECTION_ORDER,
    process_image_bundle,
    combine_structured_fundamentals,
)


st.set_page_config(page_title="PSX Institutional Intelligence Terminal", layout="wide", initial_sidebar_state="expanded")


def inject_institutional_terminal_theme(theme_name: str) -> None:
    is_dark = theme_name == "Midnight Terminal"
    if is_dark:
        palette = {
            "bg": "#07111d",
            "bg2": "#091827",
            "sidebar": "#081421",
            "card": "#0d1d2d",
            "card2": "#11263a",
            "line": "#203d56",
            "line_soft": "rgba(130, 176, 213, 0.18)",
            "text": "#eef5fb",
            "muted": "#9eb4c8",
            "primary": "#1595d3",
            "primary2": "#0c6ea7",
            "green": "#15a36d",
            "red": "#d94b57",
            "amber": "#d69a2d",
            "cyan": "#56c8ff",
            "input": "#0a1826",
            "shadow": "rgba(0, 0, 0, 0.34)",
        }
    else:
        palette = {
            "bg": "#f3f7fb",
            "bg2": "#eaf2f9",
            "sidebar": "#ffffff",
            "card": "#ffffff",
            "card2": "#f5f9fd",
            "line": "#c9dbe9",
            "line_soft": "rgba(44, 92, 128, 0.14)",
            "text": "#0c2538",
            "muted": "#557189",
            "primary": "#0e74ae",
            "primary2": "#095a89",
            "green": "#0b9563",
            "red": "#cf4150",
            "amber": "#c58b22",
            "cyan": "#1595c9",
            "input": "#ffffff",
            "shadow": "rgba(20, 53, 78, 0.12)",
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --term-bg: {palette['bg']};
            --term-bg2: {palette['bg2']};
            --term-sidebar: {palette['sidebar']};
            --term-card: {palette['card']};
            --term-card2: {palette['card2']};
            --term-line: {palette['line']};
            --term-line-soft: {palette['line_soft']};
            --term-text: {palette['text']};
            --term-muted: {palette['muted']};
            --term-primary: {palette['primary']};
            --term-primary2: {palette['primary2']};
            --term-green: {palette['green']};
            --term-red: {palette['red']};
            --term-amber: {palette['amber']};
            --term-cyan: {palette['cyan']};
            --term-input: {palette['input']};
            --term-shadow: {palette['shadow']};
        }}
        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {{
            background: radial-gradient(circle at 18% 0%, rgba(21,149,211,.12), transparent 28%),
                        linear-gradient(180deg, var(--term-bg) 0%, var(--term-bg2) 100%) !important;
            color: var(--term-text) !important;
        }}
        [data-testid="stAppViewBlockContainer"] {{
            max-width: 1520px;
            padding-top: 1.05rem;
            padding-bottom: 3rem;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--term-sidebar) 0%, var(--term-card) 100%) !important;
            border-right: 1px solid var(--term-line) !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem;
        }}
        [data-testid="collapsedControl"] {{
            color: var(--term-text) !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, div {{ color: var(--term-text); }}
        small, .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--term-muted) !important; }}
        .terminal-shell {{
            display: grid;
            gap: 14px;
            margin-bottom: 16px;
        }}
        .terminal-header {{
            padding: 22px 22px 18px 22px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(21,149,211,.95), rgba(13,29,45,.98) 62%);
            border: 1px solid var(--term-line);
            box-shadow: 0 16px 34px var(--term-shadow);
        }}
        .terminal-kicker {{
            display: inline-flex;
            align-items: center;
            padding: 6px 11px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,.22);
            background: rgba(255,255,255,.11);
            font-size: .72rem;
            letter-spacing: .09em;
            text-transform: uppercase;
            font-weight: 900;
            color: #fff !important;
        }}
        .terminal-header h1 {{
            margin: 10px 0 8px 0;
            font-size: clamp(1.65rem, 2.6vw, 2.7rem);
            font-weight: 950;
            color: #fff !important;
            line-height: 1.04;
        }}
        .terminal-header p {{
            margin: 0;
            max-width: 1080px;
            font-size: .98rem;
            color: rgba(255,255,255,.88) !important;
        }}
        .terminal-stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 12px;
        }}
        .terminal-stat {{
            background: linear-gradient(180deg, var(--term-card) 0%, var(--term-card2) 100%);
            border: 1px solid var(--term-line);
            border-radius: 17px;
            padding: 15px 16px;
            box-shadow: 0 10px 22px var(--term-shadow);
        }}
        .terminal-stat .label {{
            display: block;
            color: var(--term-muted) !important;
            font-size: .72rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 7px;
        }}
        .terminal-stat .value {{
            display: block;
            font-weight: 950;
            font-size: 1.16rem;
            line-height: 1.18;
        }}
        .terminal-stat .note {{
            display: block;
            margin-top: 7px;
            font-size: .80rem;
            line-height: 1.35;
            color: var(--term-muted) !important;
        }}
        .terminal-section-title {{
            margin-top: 18px;
            padding: 11px 14px;
            border-radius: 14px 14px 0 0;
            background: linear-gradient(90deg, var(--term-primary2), var(--term-primary));
            color: #fff !important;
            font-size: .84rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: .08em;
        }}
        .terminal-panel {{
            background: var(--term-card);
            border: 1px solid var(--term-line);
            border-top: 0;
            border-radius: 0 0 18px 18px;
            padding: 15px;
            margin-bottom: 16px;
        }}
        .terminal-chip {{
            display: inline-flex;
            align-items: center;
            padding: 7px 11px;
            margin: 0 7px 8px 0;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .05em;
            text-transform: uppercase;
            color: #fff !important;
        }}
        .chip-green {{ background: linear-gradient(135deg, #087a50, var(--term-green)); }}
        .chip-blue {{ background: linear-gradient(135deg, var(--term-primary2), var(--term-primary)); }}
        .chip-amber {{ background: linear-gradient(135deg, #a56f13, var(--term-amber)); }}
        .chip-red {{ background: linear-gradient(135deg, #a82f3b, var(--term-red)); }}
        .sidebar-brand {{
            background: linear-gradient(145deg, var(--term-card2), var(--term-card));
            border: 1px solid var(--term-line);
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 10px 20px var(--term-shadow);
        }}
        .sidebar-logo {{
            display: inline-flex;
            width: 38px;
            height: 38px;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--term-green), var(--term-primary));
            color: #fff !important;
            font-weight: 950;
            letter-spacing: .04em;
            margin-bottom: 10px;
        }}
        .sidebar-brand h3 {{ margin: 0; font-size: 1rem; font-weight: 950; }}
        .sidebar-brand p {{ margin: 6px 0 0 0; color: var(--term-muted) !important; font-size: .80rem; line-height: 1.35; }}
        .sidebar-note {{
            padding: 12px;
            border-radius: 14px;
            border: 1px solid var(--term-line);
            background: var(--term-card);
            color: var(--term-muted) !important;
            font-size: .80rem;
            line-height: 1.4;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; }}
        [data-testid="stSidebar"] [role="radiogroup"] label {{
            margin: 2px 0 !important;
            padding: 8px 9px !important;
            border-radius: 12px !important;
            border: 1px solid transparent !important;
            background: transparent !important;
            transition: all .12s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: var(--term-card2) !important;
            border-color: var(--term-line) !important;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--term-card) 0%, var(--term-card2) 100%);
            border: 1px solid var(--term-line);
            border-radius: 16px;
            padding: 13px 14px;
            min-height: 88px;
            box-shadow: 0 9px 19px var(--term-shadow);
        }}
        div[data-testid="stMetric"] label {{ color: var(--term-muted) !important; font-size: .72rem !important; font-weight: 900 !important; letter-spacing: .06em; text-transform: uppercase; }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: var(--term-text) !important; font-size: 1.27rem !important; font-weight: 950 !important; }}
        .stButton > button, button[kind="primary"] {{
            background: linear-gradient(135deg, var(--term-primary2), var(--term-primary)) !important;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,.16) !important;
            border-radius: 12px !important;
            min-height: 42px;
            font-weight: 950 !important;
            letter-spacing: .03em;
            box-shadow: 0 8px 18px var(--term-shadow);
        }}
        .stButton > button:hover {{ transform: translateY(-1px); filter: brightness(1.05); }}
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea {{
            background: var(--term-input) !important;
            border: 1px solid var(--term-line) !important;
            border-radius: 12px !important;
            color: var(--term-text) !important;
        }}
        [data-baseweb="input"] input, [data-baseweb="select"] span, textarea {{ color: var(--term-text) !important; }}
        [data-baseweb="select"] svg, [data-baseweb="input"] svg {{ fill: var(--term-text) !important; }}
        [data-testid="stFileUploaderDropzone"] {{
            background: var(--term-card) !important;
            border: 1px dashed var(--term-line) !important;
            border-radius: 15px !important;
        }}
        [data-testid="stExpander"] {{
            border: 1px solid var(--term-line) !important;
            border-radius: 15px !important;
            background: var(--term-card) !important;
            overflow: hidden;
        }}
        [data-testid="stExpander"] summary {{ background: var(--term-card2) !important; color: var(--term-text) !important; font-weight: 950 !important; }}
        div[data-testid="stAlert"] {{ border-radius: 15px !important; border: 1px solid var(--term-line) !important; box-shadow: 0 8px 18px var(--term-shadow); }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid var(--term-line);
            border-radius: 15px;
            overflow: hidden;
            background: var(--term-card);
        }}
        .stPlotlyChart {{ border: 1px solid var(--term-line); border-radius: 17px; overflow: hidden; background: var(--term-card); padding: 8px; }}
        hr {{ border-color: var(--term-line) !important; }}
        [data-baseweb="tab-list"] {{ gap: 8px; flex-wrap: wrap; background: transparent; margin-bottom: 12px; }}
        button[data-baseweb="tab"] {{
            height: auto !important; padding: 10px 13px !important; border-radius: 999px !important;
            border: 1px solid var(--term-line) !important; background: var(--term-card) !important;
            color: var(--term-text) !important; font-weight: 950 !important; white-space: normal !important; max-width: 220px;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: #fff !important; background: linear-gradient(135deg, var(--term-green), var(--term-primary)) !important; border-color: transparent !important; }}
        [data-baseweb="tab-highlight"] {{ display: none !important; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px !important; border-color: var(--term-line) !important; background: color-mix(in srgb, var(--term-card) 92%, transparent) !important; }}
        .stMultiSelect [data-baseweb="select"] {{ border-radius: 14px !important; }}
        @media (max-width: 860px) {{
            [data-testid="stAppViewBlockContainer"] {{ padding-left: .7rem; padding-right: .7rem; }}
            .terminal-header {{ border-radius: 17px; padding: 17px; }}
            .terminal-stat-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_terminal_sidebar() -> tuple[str, list[str], str]:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">PSX</div>
            <h3>Decision Terminal</h3>
            <p>Simplified professional workspace: fewer desks, clearer decisions, and multiple desks can stay open together.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    theme_name = st.sidebar.radio(
        "Appearance",
        ["Midnight Terminal", "Executive Light"],
        index=0,
        key="terminal_theme",
        help="Choose a professional dark or light terminal theme.",
    )

    desk_options = [
        "Decision Center",
        "Latest Divergence Scanner",
        "Fast Down Alert Scanner",
        "Portfolio WhatsApp Alert Watcher",
        "ChatGPT AI Decision Assistant",
        "Sarmaaya Data Import Center",
        "Portfolio Desk",
        "Stock Deep Dive",
        "Market & Technical Scanner",
        "Fundamental & Strategy Lab",
        "Risk, Alerts & Watchtower",
        "Knowledge & Macro",
    ]

    st.sidebar.markdown("### Simplified Workspaces")
    preset = st.sidebar.selectbox(
        "Workspace preset",
        [
            "Custom",
            "Daily Decision Desk",
            "Portfolio Review Desk",
            "Trading Research Desk",
            "Full Research Terminal",
            "Open Everything",
        ],
        index=1,
        key="terminal_workspace_preset",
        help="Pick a simple workspace, then adjust which desks remain open together.",
    )
    preset_modules = {
        "Daily Decision Desk": [
            "Decision Center",
            "Risk, Alerts & Watchtower",
        ],
        "Portfolio Review Desk": [
            "Decision Center",
            "Portfolio Desk",
            "Stock Deep Dive",
        ],
        "Trading Research Desk": [
            "Decision Center",
            "Latest Divergence Scanner",
            "Fast Down Alert Scanner",
            "Portfolio WhatsApp Alert Watcher",
            "ChatGPT AI Decision Assistant",
            "Sarmaaya Data Import Center",
            "Market & Technical Scanner",
            "Stock Deep Dive",
            "Risk, Alerts & Watchtower",
        ],
        "Full Research Terminal": [
            "Decision Center",
            "Latest Divergence Scanner",
            "Fast Down Alert Scanner",
            "Portfolio WhatsApp Alert Watcher",
            "ChatGPT AI Decision Assistant",
            "Sarmaaya Data Import Center",
            "Portfolio Desk",
            "Stock Deep Dive",
            "Market & Technical Scanner",
            "Fundamental & Strategy Lab",
            "Risk, Alerts & Watchtower",
            "Knowledge & Macro",
        ],
        "Open Everything": desk_options,
    }
    default_modules = preset_modules.get(
        preset,
        st.session_state.get("terminal_active_modules", ["Decision Center", "Risk, Alerts & Watchtower"]),
    )
    if preset != "Custom":
        st.session_state["terminal_active_modules"] = default_modules

    active_modules = st.sidebar.multiselect(
        "Desks open on screen",
        desk_options,
        default=st.session_state.get("terminal_active_modules", default_modules),
        key="terminal_active_modules",
        help="Select more than one desk. They remain visible together in the main workspace.",
    )
    if not active_modules:
        active_modules = ["Decision Center"]
        st.sidebar.info("At least one desk is needed. Decision Center is shown.")

    layout_mode = st.sidebar.radio(
        "Workspace layout",
        ["Accordion panels", "Stacked panels", "Split board"],
        index=0,
        key="terminal_multi_layout",
        help="Accordion is the cleanest. Stacked shows all desks directly. Split Board places desks in two columns.",
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-note">
            <strong>Simplified terminal active:</strong><br>
            {len(active_modules)} desk(s) open together.<br><br>
            <strong>Main desk:</strong><br>
            Decision Center answers Buy / Hold / Sell / Buy More / Average and gives Entry, TP1, TP2, TP3.
        </div>
        """,
        unsafe_allow_html=True,
    )
    return theme_name, active_modules, layout_mode

def render_terminal_header(active_modules: list[str]) -> None:
    st.markdown(
        f"""
        <div class="terminal-shell">
            <div class="terminal-header">
                <span class="terminal-kicker">PSX Autonomous Decision Terminal</span>
                <h1>Clear Decisions. Full Symbol Reports. Fewer Workspaces.</h1>
                <p>Active desks: {", ".join(active_modules[:5])}{" ..." if len(active_modules) > 5 else ""}. The Decision Center is now the primary place for buy ideas, portfolio actions, and price targets.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def command_center_panel() -> None:
    st.markdown(
        """
        <div>
            <span class="terminal-chip chip-green">Autopilot Ready</span>
            <span class="terminal-chip chip-blue">Portfolio + Fundamentals</span>
            <span class="terminal-chip chip-amber">Macro + Catalyst Watch</span>
            <span class="terminal-chip chip-red">Risk Control Layer</span>
        </div>
        <div class="terminal-stat-grid">
            <div class="terminal-stat"><span class="label">Autonomous Cycle</span><span class="value">Research → Rank → Action</span><span class="note">Use the Autopilot Manager to produce decisions for holdings and new ideas.</span></div>
            <div class="terminal-stat"><span class="label">Knowledge Brain</span><span class="value">Uploaded frameworks active</span><span class="note">Patterns, PDFs, valuations, risk logic, and ranking intelligence are registered.</span></div>
            <div class="terminal-stat"><span class="label">Data Inputs</span><span class="value">Files + Google links</span><span class="note">Portfolio Drive PDF and Fundamental Google Sheet links are supported.</span></div>
            <div class="terminal-stat"><span class="label">Execution Discipline</span><span class="value">Signal + Risk Gates</span><span class="note">Confluence, valuation, red flags, hazards, events, and loss control.</span></div>
        </div>
        <div class="terminal-section-title">Recommended Professional Route</div>
        <div class="terminal-panel">
            <div class="terminal-stat-grid">
                <div class="terminal-stat"><span class="label">Step 1</span><span class="value">Autopilot Manager</span><span class="note">Run the unified autonomous analysis cycle.</span></div>
                <div class="terminal-stat"><span class="label">Step 2</span><span class="value">Master Fundamentals</span><span class="note">Review quality, valuation, and rankings.</span></div>
                <div class="terminal-stat"><span class="label">Step 3</span><span class="value">Single Symbol PRO</span><span class="note">Validate action plans for selected stocks.</span></div>
                <div class="terminal-stat"><span class="label">Step 4</span><span class="value">Watchtower + Alerts</span><span class="note">Monitor risk, catalysts, and price hazards.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Layout", "Sidebar Terminal")
    c2.metric("Decision Stack", "Multi-Engine")
    c3.metric("Input Modes", "Upload + Links")
    c4.metric("Focus", "Professional PSX")



def _fmt_price(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def _selected_row(decisions: pd.DataFrame, symbol: str) -> pd.Series | None:
    if not isinstance(decisions, pd.DataFrame) or decisions.empty:
        return None
    rows = decisions[decisions["Symbol"].astype(str).str.upper().str.strip() == str(symbol).upper().strip()]
    return rows.iloc[0] if not rows.empty else None


def _render_decision_cards(frame: pd.DataFrame, *, title: str, key_prefix: str, empty_text: str) -> None:
    st.markdown(f"#### {title}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.info(empty_text)
        return

    usable = frame.head(12).reset_index(drop=True)
    for idx in range(0, len(usable), 2):
        left, right = st.columns(2)
        for offset, col in enumerate([left, right]):
            row_idx = idx + offset
            if row_idx >= len(usable):
                continue
            row = usable.iloc[row_idx]
            symbol = str(row.get("Symbol", "—"))
            decision = str(row.get("Bot Decision", row.get("Autopilot Action", "—")))
            conf = row.get("Confluence Score", "—")
            funda = row.get("Fundamental Score %", "—")
            entry = _fmt_price(row.get("Entry"))
            tp1 = _fmt_price(row.get("TP1"))
            tp2 = _fmt_price(row.get("TP2"))
            tp3 = _fmt_price(row.get("TP3"))
            confidence = row.get("Decision Confidence", "—")
            fstatus = row.get("Fundamental Data Status", "—")
            vstatus = row.get("Valuation Status", "—")
            entry_type = row.get("Entry Type", "—")
            missing_status = str(row.get("Missing Data Status", "Complete"))
            missing_note = "" if missing_status in {"Complete", "None", "nan", "—"} else f"<br><strong>Required data:</strong> {missing_status}"
            with col:
                st.markdown(
                    f"""
                    <div class="terminal-stat" style="margin-bottom: 10px;">
                        <span class="label">{decision} | Confidence: {confidence}</span>
                        <span class="value">{symbol}</span>
                        <span class="note">Confluence: {conf} | Fundamental: {funda} ({fstatus}) | Valuation: {vstatus}<br>Entry: {entry} ({entry_type}) | TP1: {tp1} | TP2: {tp2} | TP3: {tp3}{missing_note}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if missing_status not in {"Complete", "None", "nan", "—"}:
                    st.warning(str(row.get("Missing Data Request", "Please provide the missing data listed above.")))
                if st.button(f"Open full report: {symbol}", key=f"{key_prefix}_{symbol}_{row_idx}", use_container_width=True):
                    st.session_state["decision_center_selected_symbol"] = symbol
                    st.session_state["pro_single_symbol"] = symbol



def _render_missing_data_requests(decisions: pd.DataFrame, failures: pd.DataFrame | None = None) -> None:
    st.markdown("#### Required Data Requests")
    requests = []
    if isinstance(decisions, pd.DataFrame) and not decisions.empty and "Missing Data Status" in decisions.columns:
        miss = decisions[~decisions["Missing Data Status"].astype(str).isin(["Complete", "None", "nan", "—", ""])]
        for _, row in miss.iterrows():
            requests.append({
                "Symbol": row.get("Symbol"),
                "Missing Data": row.get("Missing Data Status"),
                "Required Fields": row.get("Required Data Fields"),
                "Request": row.get("Missing Data Request"),
            })
    if isinstance(failures, pd.DataFrame) and not failures.empty:
        for _, row in failures.iterrows():
            requests.append({
                "Symbol": row.get("Symbol"),
                "Missing Data": row.get("Missing Data Status", "Market/technical data unavailable"),
                "Required Fields": row.get("Required Data Fields", "OHLCV data"),
                "Request": row.get("Missing Data Request", row.get("Error")),
            })
    if not requests:
        st.success("No critical missing data detected in the latest run.")
        return
    req_df = pd.DataFrame(requests).drop_duplicates()
    st.warning("Some decisions are blocked or downgraded because required data is missing. Provide the requested data, then run Decision Center again.")
    st.dataframe(req_df, use_container_width=True, hide_index=True)


def _to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = value.replace('%', '').replace(',', '').strip()
            if cleaned in {'', '—', '-', 'nan', 'None'}:
                return default
            return float(cleaned)
        return float(value)
    except Exception:
        return default


def _smoothed_moving_average(series: pd.Series, period: int) -> pd.Series:
    """Williams-style smoothed moving average for Alligator lines."""
    values = pd.to_numeric(series, errors="coerce")
    return values.ewm(alpha=1 / float(period), adjust=False, min_periods=max(2, period // 2)).mean()


def _ensure_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add chart-only indicators without changing the core analysis engine."""
    out = df.copy()

    # Normalize possible Yahoo/CSV OHLCV column casing after direct timeframe loading.
    rename_map = {}
    for c in list(out.columns):
        lc = str(c).strip().lower()
        if lc in {"open", "high", "low", "close", "volume"} and c != lc:
            rename_map[c] = lc
    if rename_map:
        out = out.rename(columns=rename_map)

    if 'close' not in out.columns:
        return out

    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    for _c in ['open', 'high', 'low', 'volume']:
        if _c in out.columns:
            out[_c] = pd.to_numeric(out[_c], errors='coerce')

    # Always calculate RSI if it is missing or empty after timeframe reload/resample.
    try:
        if 'rsi' not in out.columns or pd.to_numeric(out.get('rsi'), errors='coerce').dropna().empty:
            delta = out['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            out['rsi'] = 100 - (100 / (1 + rs))
            out['rsi'] = out['rsi'].fillna(50)
    except Exception:
        pass
    for window in (20, 50, 200):
        key = f'ma{window}'
        if key not in out.columns:
            out[key] = out['close'].rolling(window, min_periods=max(5, window // 4)).mean()
    if 'ema9' not in out.columns:
        out['ema9'] = out['close'].ewm(span=9, adjust=False, min_periods=5).mean()
    if 'ema21' not in out.columns:
        out['ema21'] = out['close'].ewm(span=21, adjust=False, min_periods=8).mean()
    if 'ema51' not in out.columns:
        out['ema51'] = out['close'].ewm(span=51, adjust=False, min_periods=13).mean()
    if 'rsi' in out.columns:
        out['rsi'] = pd.to_numeric(out['rsi'], errors='coerce').fillna(50)
        if 'rsi_ma' not in out.columns or pd.to_numeric(out.get('rsi_ma'), errors='coerce').dropna().empty:
            out['rsi_ma'] = out['rsi'].rolling(9, min_periods=3).mean()
        if 'rsi_signal' not in out.columns or pd.to_numeric(out.get('rsi_signal'), errors='coerce').dropna().empty:
            out['rsi_signal'] = out['rsi_ma']
    if 'macd' not in out.columns or 'macd_signal' not in out.columns or 'macd_hist' not in out.columns:
        ema12 = out['close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = out['close'].ewm(span=26, adjust=False, min_periods=26).mean()
        out['macd'] = ema12 - ema26
        out['macd_signal'] = out['macd'].ewm(span=9, adjust=False, min_periods=9).mean()
        out['macd_hist'] = out['macd'] - out['macd_signal']
    if 'volume' in out.columns and 'volume_ma20' not in out.columns:
        out['volume_ma20'] = out['volume'].rolling(20, min_periods=5).mean()

    # Klinger Volume Oscillator: volume-flow trend confirmation.
    try:
        if {'high', 'low', 'close', 'volume'}.issubset(set(out.columns)) and ('klinger' not in out.columns or 'klinger_signal' not in out.columns):
            high_k = pd.to_numeric(out['high'], errors='coerce')
            low_k = pd.to_numeric(out['low'], errors='coerce')
            close_k = pd.to_numeric(out['close'], errors='coerce')
            vol_k = pd.to_numeric(out['volume'], errors='coerce').fillna(0)
            hlc_sum = high_k + low_k + close_k
            trend = pd.Series(np.where(hlc_sum > hlc_sum.shift(1), 1, -1), index=out.index).replace(0, method='ffill').fillna(1)
            dm = (high_k - low_k).abs()
            cm = dm.copy()
            for _i in range(1, len(out)):
                if trend.iloc[_i] == trend.iloc[_i - 1]:
                    cm.iloc[_i] = cm.iloc[_i - 1] + dm.iloc[_i]
                else:
                    cm.iloc[_i] = dm.iloc[_i - 1] + dm.iloc[_i]
            vf = vol_k * trend * (2 * ((dm / cm.replace(0, np.nan)) - 1)).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
            out['klinger'] = vf.ewm(span=34, adjust=False, min_periods=10).mean() - vf.ewm(span=55, adjust=False, min_periods=15).mean()
            out['klinger_signal'] = out['klinger'].ewm(span=13, adjust=False, min_periods=5).mean()
            out['klinger_hist'] = out['klinger'] - out['klinger_signal']
    except Exception:
        pass

    # Williams Alligator lines.
    # Jaw = 13-period smoothed median price shifted 8 bars forward
    # Teeth = 8-period smoothed median price shifted 5 bars forward
    # Lips = 5-period smoothed median price shifted 3 bars forward
    try:
        high_col = pd.to_numeric(out['high'], errors='coerce') if 'high' in out.columns else pd.to_numeric(out['close'], errors='coerce')
        low_col = pd.to_numeric(out['low'], errors='coerce') if 'low' in out.columns else pd.to_numeric(out['close'], errors='coerce')
        median_price = (high_col + low_col) / 2.0
        if 'jaw' not in out.columns or out['jaw'].dropna().empty:
            out['jaw'] = _smoothed_moving_average(median_price, 13).shift(8)
        if 'teeth' not in out.columns or out['teeth'].dropna().empty:
            out['teeth'] = _smoothed_moving_average(median_price, 8).shift(5)
        if 'lips' not in out.columns or out['lips'].dropna().empty:
            out['lips'] = _smoothed_moving_average(median_price, 5).shift(3)
        # Also provide explicit aliases for future chart modules.
        out['alligator_jaw'] = out['jaw']
        out['alligator_teeth'] = out['teeth']
        out['alligator_lips'] = out['lips']
    except Exception:
        pass
    return out


def _latest_indicator_rows(df: pd.DataFrame, result: dict, decision_row: dict | None = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame([{'Indicator': 'No data', 'Value': '—', 'Reading': 'Chart data unavailable'}])
    frame = _ensure_chart_indicators(df)
    latest = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) > 1 else latest
    close = _to_float(latest.get('close'))
    ma20 = _to_float(latest.get('ma20'))
    ma50 = _to_float(latest.get('ma50'))
    ma200 = _to_float(latest.get('ma200'))
    rsi_val = _to_float(latest.get('rsi'))
    vol = _to_float(latest.get('volume'))
    vol_ma = _to_float(latest.get('volume_ma20'))
    atr_val = _to_float(latest.get('atr'))
    macd_val = _to_float(latest.get('macd'))
    macd_signal = _to_float(latest.get('macd_signal'))
    macd_hist = _to_float(latest.get('macd_hist'))
    ema21 = _to_float(latest.get('ema21'))
    ema51 = _to_float(latest.get('ema51'))
    klinger_val = _to_float(latest.get('klinger'))
    klinger_signal = _to_float(latest.get('klinger_signal'))
    klinger_hist = _to_float(latest.get('klinger_hist'))

    rows = []
    if close is not None:
        rows.append({'Indicator': 'Last close', 'Value': round(close, 2), 'Reading': 'Current market price used for the chart'})
    if ma20 is not None and ma50 is not None:
        rows.append({'Indicator': 'MA20 vs MA50', 'Value': f"{ma20:.2f} / {ma50:.2f}", 'Reading': 'Bullish short trend' if ma20 > ma50 else 'Weak short trend / pullback'})
    if close is not None and ma200 is not None:
        rows.append({'Indicator': 'Price vs MA200', 'Value': f"{close:.2f} / {ma200:.2f}", 'Reading': 'Long-term uptrend filter positive' if close > ma200 else 'Below long-term trend filter'})
    if close is not None and ema21 is not None:
        rows.append({'Indicator': 'Price vs EMA21', 'Value': f"{close:.2f} / {ema21:.2f}", 'Reading': 'Short-term trend positive' if close > ema21 else 'Below EMA21 / short-term weakness'})
    if ema21 is not None and ema51 is not None:
        rows.append({'Indicator': 'EMA21 vs EMA51', 'Value': f"{ema21:.2f} / {ema51:.2f}", 'Reading': 'Bullish EMA alignment' if ema21 > ema51 else 'Bearish / weak EMA alignment'})
    if rsi_val is not None:
        if rsi_val >= 70:
            rsi_reading = 'Overbought / avoid chasing'
        elif rsi_val <= 30:
            rsi_reading = 'Oversold / watch reversal confirmation'
        elif rsi_val >= 55:
            rsi_reading = 'Bullish momentum'
        elif rsi_val <= 45:
            rsi_reading = 'Weak momentum'
        else:
            rsi_reading = 'Neutral momentum'
        rows.append({'Indicator': 'RSI 14', 'Value': round(rsi_val, 2), 'Reading': rsi_reading})
    if vol is not None and vol_ma not in (None, 0):
        rows.append({'Indicator': 'Volume vs 20-day avg', 'Value': f"{vol/vol_ma:.2f}x", 'Reading': 'Volume confirmation present' if vol >= vol_ma else 'Volume confirmation weak'})
    if atr_val is not None and close not in (None, 0):
        rows.append({'Indicator': 'ATR risk', 'Value': f"{(atr_val/close)*100:.2f}%", 'Reading': 'Higher volatility' if (atr_val/close) > 0.04 else 'Normal volatility'})
    if macd_val is not None and macd_signal is not None:
        macd_reading = 'Bullish MACD crossover' if macd_val > macd_signal else 'Bearish / weak MACD structure'
        hist_value = f" | Hist {macd_hist:.3f}" if macd_hist is not None else ''
        rows.append({'Indicator': 'MACD', 'Value': f"{macd_val:.3f} / {macd_signal:.3f}{hist_value}", 'Reading': macd_reading})
    if klinger_val is not None and klinger_signal is not None:
        klinger_reading = 'Bullish volume-flow confirmation' if klinger_val > klinger_signal else 'Bearish / weak volume-flow confirmation'
        kh = f" | Hist {klinger_hist:.3f}" if klinger_hist is not None else ''
        rows.append({'Indicator': 'Klinger Oscillator', 'Value': f"{klinger_val:.3f} / {klinger_signal:.3f}{kh}", 'Reading': klinger_reading})

    cwt = result.get('cwt', {}) if isinstance(result, dict) else {}
    if cwt:
        rows.append({'Indicator': 'CWT / Alligator', 'Value': cwt.get('setup', '—'), 'Reading': cwt.get('state', cwt.get('direction', '—'))})
    divergence = result.get('divergence', {}) if isinstance(result, dict) else {}
    if divergence:
        rows.append({'Indicator': 'RSI divergence', 'Value': divergence.get('bias', 'Neutral'), 'Reading': divergence.get('detail', 'No strong divergence signal')})
    signal = result.get('signal', {}) if isinstance(result, dict) else {}
    if signal:
        rows.append({'Indicator': 'Signal engine', 'Value': signal.get('action', '—'), 'Reading': f"Bias: {signal.get('bias', '—')} | Confidence: {signal.get('confidence', '—')}"})
    if decision_row:
        rows.append({'Indicator': 'Decision Center', 'Value': decision_row.get('Bot Decision', decision_row.get('Autopilot Action', '—')), 'Reading': decision_row.get('Decision Reason', '—')})
    return pd.DataFrame(rows)


def _pattern_rows_from_result(result: dict) -> pd.DataFrame:
    rows = []
    for group, items in (result.get('patterns', {}) if isinstance(result, dict) else {}).items():
        for item in items or []:
            rows.append({
                'Pattern Type': str(group).replace('_', ' ').title(),
                'Pattern': item.get('name', item.get('label', '—')),
                'Bias': item.get('bias', '—'),
                'Score': item.get('score', '—'),
                'Detail': item.get('detail', item.get('rule', '—')),
            })
    if not rows:
        rows.append({'Pattern Type': 'None', 'Pattern': 'No named pattern detected', 'Bias': 'Neutral', 'Score': '—', 'Detail': 'Use support/resistance, RSI, volume and trend confirmation instead.'})
    return pd.DataFrame(rows)


def _scorecard_table(scorecard_text: str) -> pd.DataFrame:
    rows = []
    for raw in str(scorecard_text or "").split(" | "):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        name, rest = raw.split(":", 1)
        status = ""
        detail = rest.strip()
        if "(" in detail and detail.endswith(")"):
            status = detail.split("(", 1)[0].strip()
            detail = detail.split("(", 1)[1][:-1].strip()
        rows.append({"Check": name.strip(), "Status": status or "INFO", "Reason": detail})
    if not rows:
        rows.append({"Check": "Decision reason", "Status": "INFO", "Reason": str(scorecard_text or "No detailed scorecard available")})
    return pd.DataFrame(rows)


def _render_indicator_pattern_chat(symbol: str, df: pd.DataFrame, result: dict, decision_row: dict | None = None) -> None:
    st.markdown('#### Selected Symbol — Indicator & Pattern Chat')
    indicator_df = _latest_indicator_rows(df, result, decision_row)
    pattern_df = _pattern_rows_from_result(result)

    latest = _ensure_chart_indicators(df).iloc[-1] if isinstance(df, pd.DataFrame) and not df.empty else {}
    close = _to_float(latest.get('close')) if hasattr(latest, 'get') else None
    entry = _to_float((decision_row or {}).get('Entry'))
    sl = _to_float((decision_row or {}).get('Stop Loss'))
    tp1 = _to_float((decision_row or {}).get('TP1'))
    decision = (decision_row or {}).get('Bot Decision', (decision_row or {}).get('Autopilot Action', '—'))
    signal = result.get('signal', {}) if isinstance(result, dict) else {}
    scenario = result.get('scenario', {}) if isinstance(result, dict) else {}
    cwt_scenario = result.get('cwt_scenario', {}) if isinstance(result, dict) else {}

    bullets = []
    if close is not None:
        bullets.append(f"Latest close is **{close:.2f}**.")
    if entry is not None:
        bullets.append(f"Decision Center entry is **{entry:.2f}**; avoid chasing far above entry.")
    if sl is not None and tp1 is not None:
        bullets.append(f"Risk map: SL **{sl:.2f}**, TP1 **{tp1:.2f}**.")
    if scenario:
        bullets.append(f"MTF scenario: **{scenario.get('number', '—')}** — {scenario.get('quality', '—')}.")
    if cwt_scenario:
        bullets.append(f"CWT scenario: **{cwt_scenario.get('number', '—')}** — {cwt_scenario.get('quality', '—')}.")
    if signal:
        bullets.append(f"Signal engine says **{signal.get('action', '—')}** with **{signal.get('bias', '—')}** bias and **{signal.get('confidence', '—')}** confidence.")
    bullets.append(f"Final bot decision: **{decision}**.")

    st.info(' '.join(bullets))
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('##### Indicator Readings')
        st.dataframe(indicator_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('##### Applied Patterns')
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)

def _render_symbol_decision_report(symbol: str, bundle: dict, fundamentals_df: pd.DataFrame | None = None) -> None:
    decisions = bundle.get("decisions", pd.DataFrame()) if isinstance(bundle, dict) else pd.DataFrame()
    row = _selected_row(decisions, symbol)
    if row is None:
        st.warning("The selected symbol is not available in the latest Decision Center run.")
        return

    decision = row.get("Bot Decision", row.get("Autopilot Action", "—"))
    st.markdown(f"### Full Decision Report — {symbol}")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Bot Decision", decision)
    a2.metric("Entry", _fmt_price(row.get("Entry")))
    a3.metric("TP1", _fmt_price(row.get("TP1")))
    a4.metric("TP2", _fmt_price(row.get("TP2")))
    a5.metric("TP3", _fmt_price(row.get("TP3")))

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Stop Loss", _fmt_price(row.get("Stop Loss")))
    b2.metric("Confluence", row.get("Confluence Score", "—"))
    b3.metric("PRO Score", row.get("PRO Score", "—"))
    b4.metric("Fundamental", row.get("Fundamental Score %", "—"))
    b5.metric("Fraud Risk", row.get("Fraud Risk", "—"))

    st.caption(f"Decision basis: {row.get('Decision Basis', '—')} | Fundamental source: {row.get('Fundamental Source', '—')} | Internet status: {row.get('Internet Fundamental Status', '—')}")
    if str(row.get('Decision Basis', '')).lower().startswith('technical basis'):
        st.warning("Fundamental data could not be found/loaded completely. This answer is given on TECHNICAL BASIS ONLY. Use smaller risk and provide missing fundamental indicators for a full investment decision.")

    reason = str(row.get("Decision Reason", "No reason available."))
    auto_action = str(row.get("Autopilot Action", "—"))
    label = str(row.get("Decision Label", "—"))
    knowledge = str(row.get("Knowledge Applied", "—"))
    st.markdown(
        f"""
        <div class="terminal-section-title">Decision Explanation</div>
        <div class="terminal-panel">
            <span class="terminal-chip chip-blue">{auto_action}</span>
            <span class="terminal-chip chip-green">{label}</span>
            <p><strong>Why:</strong> {reason}</p>
            <p><strong>Uploaded knowledge used:</strong> {knowledge}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    missing_status = str(row.get("Missing Data Status", "Complete"))
    if missing_status not in {"Complete", "None", "nan", "—", ""}:
        st.error("Required data is missing before this decision can be trusted fully.")
        st.dataframe(pd.DataFrame([
            {"Field": "Missing Data", "Value": row.get("Missing Data Status")},
            {"Field": "Required Fields", "Value": row.get("Required Data Fields")},
            {"Field": "Request", "Value": row.get("Missing Data Request")},
        ]), use_container_width=True, hide_index=True)

    st.markdown("#### Decision Reason Checklist")
    st.dataframe(_scorecard_table(str(row.get("Decision Scorecard", reason))), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Portfolio Snapshot")
        portfolio_rows = [
            {"Field": "Portfolio Holding", "Value": row.get("Portfolio Holding")},
            {"Field": "Quantity", "Value": row.get("Portfolio Qty")},
            {"Field": "Avg Buy", "Value": row.get("Portfolio Avg Buy")},
            {"Field": "MTM Price", "Value": row.get("Portfolio MTM Price")},
            {"Field": "P&L", "Value": row.get("Portfolio P&L")},
            {"Field": "P&L %", "Value": row.get("Portfolio P&L %")},
        ]
        st.dataframe(pd.DataFrame(portfolio_rows), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Fundamental & Valuation Snapshot")
        fundamental_rows = [
            {"Field": "Fundamental Grade", "Value": row.get("Fundamental Grade")},
            {"Field": "Fundamental Score %", "Value": row.get("Fundamental Score %")},
            {"Field": "Valuation Grade", "Value": row.get("Valuation Grade")},
            {"Field": "Margin of Safety %", "Value": row.get("Margin of Safety %")},
            {"Field": "Fraud Risk", "Value": row.get("Fraud Risk")},
            {"Field": "Secular Bias", "Value": row.get("Secular Bias")},
        ]
        st.dataframe(pd.DataFrame(fundamental_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Technical, Risk, and Prediction Snapshot")
    tech_rows = [
        {"Field": "Signal Bias", "Value": row.get("Signal Bias")},
        {"Field": "Signal Action", "Value": row.get("Signal Action")},
        {"Field": "Signal Confidence", "Value": row.get("Signal Confidence")},
        {"Field": "Hazard Level", "Value": row.get("Hazard Level")},
        {"Field": "News/Event Risk", "Value": row.get("News/Event Risk")},
        {"Field": "Prediction Verdict", "Value": row.get("Prediction Verdict")},
        {"Field": "Expected Return %", "Value": row.get("Prediction Expected Return %")},
        {"Field": "Probability Up %", "Value": row.get("Prediction Up Probability %")},
        {"Field": "RSI", "Value": row.get("RSI")},
        {"Field": "ADX", "Value": row.get("ADX")},
        {"Field": "Volume Ratio", "Value": row.get("Volume Ratio")},
    ]
    st.dataframe(pd.DataFrame(tech_rows), use_container_width=True, hide_index=True)

    report_cache_key = f"decision_report_cached_result_{symbol}"
    chart_refresh = st.button(
        "Refresh selected symbol chart data",
        key=f"decision_report_refresh_{symbol}",
        use_container_width=True,
        help="Checklist changes do not remove the report. Use this only when you want to reload live OHLCV data.",
    )

    if chart_refresh or report_cache_key not in st.session_state:
        try:
            higher = load_psx_yahoo_ohlcv(symbol, interval="1d", period="2y")
            lower = load_psx_yahoo_ohlcv(symbol, interval="1d", period="2y")
            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher,
                lower_df=lower,
                asset_class="Stock",
                analysis_tf="1d",
                execution_tf="1d",
                risk_context={"high_impact_news": str(row.get("News/Event Risk", "LOW")).upper() in {"HIGH", "MODERATE"}},
            )
            result["decision_row"] = row.to_dict()
            st.session_state[report_cache_key] = result
        except Exception as exc:
            if report_cache_key not in st.session_state:
                st.warning(f"The detailed live chart/report could not be rebuilt for {symbol}: {exc}")
                return
            st.warning(f"Live refresh failed, so the last saved chart/report is still shown: {exc}")

    result = st.session_state.get(report_cache_key)
    if not isinstance(result, dict):
        st.warning("No cached symbol chart result is available yet. Click refresh or run the Decision Center again.")
        return

    result["decision_row"] = row.to_dict()
    st.markdown("#### Clear Execution Chart — Price, Volume, RSI, Levels")
    render_chart_engine(
        result["execution_frame"],
        result,
        symbol=symbol,
        title=f"{symbol} — Decision Center Execution Chart",
        key_prefix=f"decision_report_{symbol}",
    )
    _render_indicator_pattern_chat(symbol, result["execution_frame"], result, row.to_dict())
    p_left, p_right = st.columns(2)
    with p_left:
        pattern_rows = []
        for group, items in result.get("patterns", {}).items():
            for item in items:
                pattern_rows.append({"Group": group, **item})
        st.markdown("#### Detected Patterns")
        st.dataframe(pd.DataFrame(pattern_rows) if pattern_rows else pd.DataFrame([{"Result": "No named pattern detected."}]), use_container_width=True, hide_index=True)
    with p_right:
        plan = result.get("trade_plan", {})
        st.markdown("#### Core Trade Plan")
        st.dataframe(pd.DataFrame([
            {"Field": "Order", "Value": plan.get("order_type")},
            {"Field": "Entry", "Value": plan.get("entry")},
            {"Field": "Stop Loss", "Value": plan.get("stop_loss")},
            {"Field": "Base Take Profit", "Value": plan.get("take_profit")},
            {"Field": "Risk/Reward", "Value": plan.get("rr")},
            {"Field": "Invalidation", "Value": plan.get("invalidation")},
        ]), use_container_width=True, hide_index=True)



def _reorder_decision_center_columns(df):
    """Show key Decision Center columns first, including Current Price."""
    try:
        if df is None or not hasattr(df, "columns") or getattr(df, "empty", True):
            return df
        preferred = [
            "Symbol", "Current Price", "Bot Decision", "Decision Label", "Portfolio Holding",
            "Portfolio Qty", "Portfolio Avg Buy", "Portfolio MTM Price", "Portfolio P&L", "Portfolio P&L %",
            "Recent Price Change", "Recent Price Change %", "Confirmed Bearish",
            "Entry", "Stop Loss", "TP1", "TP2", "TP3",
            "RSI", "Volume Ratio", "Confluence Score", "PRO Score",
            "Fundamental Score %", "Valuation Grade", "Margin of Safety %",
            "Decision Reason", "Why Not Sell", "Momentum Guard",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[cols]
    except Exception:
        return df


def decision_center_panel() -> None:
    st.subheader("Decision Center — Buy Ideas, Portfolio Actions, and Price Targets")
    st.write(
        "This is the simplified main brain of the bot. It scans your watchlist, reads your portfolio PDF/CSV and fundamental sheet, "
        "then answers: Which stock should I buy? For portfolio stocks: Hold, Sell/Reduce, Buy More, or Average Carefully. "
        "It also provides Entry, Stop Loss, TP1, TP2, and TP3."
    )
    st.info(_sarmaaya_full_bot_status_text())

    c1, c2 = st.columns([1.35, 1])
    symbols_text = c1.text_area(
        "Symbols to evaluate",
        value="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF",
        height=112,
        key="decision_center_symbols",
        help="Comma-separated PSX symbols. Add your portfolio and watchlist symbols here.",
    )
    with c2:
        max_symbols = st.number_input("Max symbols", min_value=1, max_value=100, value=25, step=1, key="decision_center_max_symbols")
        persist_outputs = st.checkbox("Save Decision Center outputs to data/", value=True, key="decision_center_persist")
        auto_fetch_fundamentals = st.checkbox(
            "Auto-find missing fundamentals from internet",
            value=True,
            key="decision_center_auto_fetch_fundamentals",
            help="If uploaded/Sheet/Image data is missing for a symbol, the bot will try public web sources first. If not found, it continues on technical basis and lists the missing indicators.",
        )
        run_label = "Run Complete Decision Analysis"

    i1, i2 = st.columns(2)
    portfolio_upload = i1.file_uploader(
        "Portfolio CSV / Excel",
        type=["csv", "xlsx", "xls"],
        key="decision_center_portfolio_upload",
        help="Optional file upload. Preferred columns: Symbol, Quantity, Avg_Buy, MTM_Price.",
    )
    fundamentals_upload = i2.file_uploader(
        "Fundamentals CSV / Excel",
        type=["csv", "xlsx", "xls"],
        key="decision_center_fundamentals_upload",
        help="Optional file upload for ranking, valuation, and fraud-risk checks.",
    )

    l1, l2 = st.columns(2)
    portfolio_drive_pdf_url = l1.text_input(
        "Portfolio Google Drive PDF link",
        value="",
        key="decision_center_portfolio_pdf_link",
        placeholder="Paste portfolio PDF Google Drive link",
    )
    fundamentals_sheet_url = l2.text_input(
        "Fundamental Google Sheet link",
        value="",
        key="decision_center_fundamentals_sheet_link",
        placeholder="Paste fundamental Google Sheet link",
    )
    st.caption("Use links or uploads. When a link is pasted, the link source is used for that dataset.")
    if fundamentals_upload is None and not str(fundamentals_sheet_url or "").strip() and has_imported_fundamental_images():
        st.info("No fundamental CSV/Sheet link is selected, so the Decision Center will use the latest Fundamental Image Import data as its fallback fundamentals source.")

    with st.container(border=True):
        st.markdown("### Decision discipline settings")
        d1, d2, d3, d4 = st.columns(4)
        entry_gate = d1.slider("Entry gate", 40, 85, 60, 1, key="decision_center_entry_gate")
        strong_gate = d2.slider("Strong buy gate", 50, 95, 70, 1, key="decision_center_strong_gate")
        investor_gate = d3.slider("Minimum fundamental score", 0, 90, 55, 1, key="decision_center_investor_gate")
        target_rr = d4.slider("Target R:R base", 1.0, 6.0, 3.0, 0.5, key="decision_center_target_rr")
        p1, p2 = st.columns(2)
        prediction_horizon = p1.slider("Prediction horizon", 3, 15, 5, 1, key="decision_center_prediction_horizon")
        stop_atr = p2.slider("Stop ATR", 0.75, 5.0, 2.0, 0.25, key="decision_center_stop_atr")

    run = st.button(run_label, type="primary", use_container_width=True, key="decision_center_run")
    if run:
        try:
            symbols = parse_symbols(symbols_text)
            portfolio_df = _sanitize_dataframe_for_decision_engine(read_portfolio_source(portfolio_upload, portfolio_drive_pdf_url))
            fundamentals_df = _dedupe_symbol_rows_for_decision(_sanitize_dataframe_for_decision_engine(read_fundamentals_source(fundamentals_upload, fundamentals_sheet_url)))
            st.session_state["full_bot_active_fundamentals_df"] = fundamentals_df
            bundle = run_autopilot_cycle(
                symbols,
                portfolio_df=portfolio_df,
                fundamentals_df=fundamentals_df,
                max_symbols=int(max_symbols),
                prediction_horizon=int(prediction_horizon),
                stop_atr=float(stop_atr),
                target_rr=float(target_rr),
                discipline_rules={
                    "minimum_confluence_for_entry": float(entry_gate),
                    "minimum_confluence_for_strong_entry": float(strong_gate),
                    "minimum_fundamental_score_for_investor": float(investor_gate),
                },
                auto_fetch_missing_fundamentals=bool(auto_fetch_fundamentals),
            )
            st.session_state["decision_center_bundle"] = bundle
            st.session_state["decision_center_fundamentals_df"] = fundamentals_df
            st.session_state["decision_center_portfolio_df"] = portfolio_df
            if persist_outputs:
                paths = persist_autopilot_outputs(bundle)
                st.success("Decision analysis complete. Outputs saved in data/.")
                st.caption("Saved: " + " | ".join(f"{k}: {v.name}" for k, v in paths.items()))
            else:
                st.success("Decision analysis complete.")
        except Exception as exc:
            st.error(f"Decision Center analysis failed: {exc}")
            try:
                st.caption("Debug: active fundamentals table shape/columns")
                fdbg = st.session_state.get("full_bot_active_fundamentals_df", pd.DataFrame())
                if isinstance(fdbg, pd.DataFrame) and not fdbg.empty:
                    st.write(f"Shape: {fdbg.shape}")
                    st.write("Columns:", list(fdbg.columns)[:80])
                    st.dataframe(fdbg.head(20), use_container_width=True, hide_index=True)
            except Exception:
                pass

    bundle = st.session_state.get("decision_center_bundle")
    if isinstance(bundle, dict):
        for _k in ["decisions", "holdings", "opportunities"]:
            if isinstance(bundle.get(_k), pd.DataFrame) and not bundle[_k].empty:
                bundle[_k] = _reorder_decision_center_columns(bundle[_k])
    if not isinstance(bundle, dict):
        st.info("Run the Decision Center to generate buy ideas, portfolio actions, and Entry/TP1/TP2/TP3 levels.")
        return

    decisions = bundle.get("decisions", pd.DataFrame())
    holdings = bundle.get("holdings", pd.DataFrame())
    opportunities = bundle.get("opportunities", pd.DataFrame())
    failures = bundle.get("failures", pd.DataFrame())
    market_brief = bundle.get("market_brief", pd.DataFrame())

    if isinstance(market_brief, pd.DataFrame) and not market_brief.empty:
        row = market_brief.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Completed", f"{int(row.get('Symbols Completed', 0))}/{int(row.get('Symbols Requested', 0))}")
        m2.metric("Event Risk", row.get("News/Event Risk", "N/A"))
        m3.metric("Avg Confluence", row.get("Average Confluence", "N/A"))
        m4.metric("High Hazards", row.get("High Hazard Count", 0))
        m5.metric("Failures", row.get("Failures", 0))

    if isinstance(opportunities, pd.DataFrame) and not opportunities.empty:
        buy_ideas = opportunities[opportunities["Bot Decision"].isin(["BUY", "WATCH FOR ENTRY"])].copy()
        if buy_ideas.empty:
            buy_ideas = opportunities.copy()
        buy_ideas = buy_ideas.sort_values(["Bot Decision", "Confluence Score", "PRO Score"], ascending=[True, False, False], na_position="last")
    else:
        buy_ideas = pd.DataFrame()

    if isinstance(holdings, pd.DataFrame) and not holdings.empty:
        portfolio_actions = holdings.copy().sort_values(["Bot Decision", "Confluence Score"], ascending=[True, False], na_position="last")
    else:
        portfolio_actions = pd.DataFrame()

    _render_decision_cards(
        buy_ideas,
        title="Best Stock Ideas to Buy / Watch",
        key_prefix="decision_buy_card",
        empty_text="No new non-portfolio buy/watch candidates were completed in this run.",
    )
    _render_decision_cards(
        portfolio_actions,
        title="Portfolio Actions — Hold / Sell / Buy More / Average",
        key_prefix="decision_portfolio_card",
        empty_text="No portfolio holdings matched the symbols evaluated in this run.",
    )

    _render_missing_data_requests(decisions, failures)

    st.markdown("#### Decision Table")
    show_cols = [
        "Symbol", "Portfolio Holding", "Bot Decision", "Autopilot Action", "Decision Confidence", "Decision Confidence %",
        "Technical Status", "Fundamental Data Status", "Valuation Status", "Data Completeness %",
        "Missing Data Status", "Required Data Fields", "Missing Data Request",
        "Confluence Score", "PRO Score", "Fundamental Score %", "Margin of Safety %", "Entry", "Entry Type", "Stop Loss", "TP1", "TP2", "TP3",
        "Fraud Risk", "Hazard Level", "News/Event Risk", "Portfolio P&L %", "Decision Reason", "Decision Scorecard",
    ]
    if isinstance(decisions, pd.DataFrame) and not decisions.empty:
        visible = [c for c in show_cols if c in decisions.columns]
        st.dataframe(decisions[visible], use_container_width=True, hide_index=True)
        st.download_button(
            "Download Decision Center CSV",
            decisions.to_csv(index=False).encode("utf-8"),
            file_name="psx_decision_center_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    symbols_available = decisions["Symbol"].astype(str).tolist() if isinstance(decisions, pd.DataFrame) and not decisions.empty else []
    if symbols_available:
        default_symbol = st.session_state.get("decision_center_selected_symbol", symbols_available[0])
        if default_symbol not in symbols_available:
            default_symbol = symbols_available[0]
        selected_symbol = st.selectbox(
            "Open or switch full symbol report",
            symbols_available,
            index=symbols_available.index(default_symbol),
            key="decision_center_report_select",
        )
        st.session_state["decision_center_selected_symbol"] = selected_symbol
        _render_symbol_decision_report(selected_symbol, bundle, st.session_state.get("decision_center_fundamentals_df"))


theme_mode, active_pages, workspace_layout = render_terminal_sidebar()
inject_institutional_terminal_theme(theme_mode)
render_terminal_header(active_pages)
st.caption(
    "Simplified terminal: use the Decision Center first, then open deeper workspaces only when needed. Multiple desks can still stay open together."
)
def chart_display_controls(key_prefix: str) -> dict:
    st.markdown("##### Chart Display Checklist")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        show_candles = st.checkbox("Candles", value=True, key=f"{key_prefix}_show_candles")
        candle_type = st.selectbox(
            "Candle Type",
            ["Simple Candles", "Heikin Ashi Candles"],
            index=0,
            key=f"{key_prefix}_candle_type",
            help="Simple candles show original OHLC. Heikin Ashi candles smooth price action and make trends easier to watch.",
        )
        clear_candles = st.checkbox("Clear Candles Mode", value=True, key=f"{key_prefix}_clear_candles", help="Makes candles larger/easier to read and reduces clutter.")
        show_levels = st.checkbox("Entry / SL / TP", value=True, key=f"{key_prefix}_show_levels")
    with c2:
        show_ma = st.checkbox("EMA / MAs", value=True, key=f"{key_prefix}_show_ma")
        show_ema21 = st.checkbox("EMA21", value=True, key=f"{key_prefix}_show_ema21")
        show_ema51 = st.checkbox("EMA51", value=True, key=f"{key_prefix}_show_ema51")
        show_sr = st.checkbox("Support / Resistance", value=True, key=f"{key_prefix}_show_sr")
    with c3:
        show_alligator = st.checkbox("Alligator Jaw / Teeth / Lips", value=True, key=f"{key_prefix}_show_alligator")
        show_volume = st.checkbox("Volume", value=True, key=f"{key_prefix}_show_volume")
    with c4:
        show_rsi = st.checkbox("RSI Panel", value=True, key=f"{key_prefix}_show_rsi")
        show_macd = st.checkbox("MACD Panel", value=False, key=f"{key_prefix}_show_macd")
    with c5:
        show_badges = st.checkbox("Buy/Sell Boxes + Pattern Note", value=True, key=f"{key_prefix}_show_badges")
        show_last_price = st.checkbox("Last Price Line", value=True, key=f"{key_prefix}_show_last_price")
    with c6:
        show_klinger = st.checkbox("Klinger Panel", value=False, key=f"{key_prefix}_show_klinger")
    c6a, c6b = st.columns(2)
    with c6a:
        show_pattern_overlay = st.checkbox(
            "Show detected patterns on chart",
            value=True,
            key=f"{key_prefix}_show_pattern_overlay",
            help="Draw visible labels, guide lines, zones, and targets for detected patterns so you can manually verify the setup.",
        )
    with c6b:
        show_divergence = st.checkbox(
            "Show divergence on chart + RSI",
            value=True,
            key=f"{key_prefix}_show_divergence",
            help="Mark bullish/bearish RSI divergence on both the price chart and the RSI panel.",
        )
    c7a, c7b = st.columns(2)
    with c7a:
        show_divergence_type = st.selectbox(
            "Divergence overlay type",
            ["Any", "Regular Bullish", "Regular Bearish", "Hidden Bullish", "Hidden Bearish", "Any Bullish", "Any Bearish"],
            index=0,
            key=f"{key_prefix}_show_divergence_type",
            help="Choose which divergence type should be drawn on the chart.",
        )
    with c7b:
        show_divergence_sensitivity = st.selectbox(
            "Divergence sensitivity",
            ["Sensitive", "Normal", "Strict"],
            index=0,
            key=f"{key_prefix}_show_divergence_sensitivity",
            help="Use Sensitive when visually clear divergences are not being detected.",
        )
    return {
        "candles": show_candles,
        "candle_type": candle_type,
        "clear_candles": clear_candles,
        "ma": show_ma,
        "ema21": show_ema21,
        "ema51": show_ema51,
        "klinger": show_klinger,
        "alligator": show_alligator,
        "support_resistance": show_sr,
        "levels": show_levels,
        "volume": show_volume,
        "rsi": show_rsi,
        "macd": show_macd,
        "badges": show_badges,
        "last_price": show_last_price,
        "pattern_overlay": show_pattern_overlay,
        "divergence": show_divergence,
        "divergence_type": show_divergence_type,
        "divergence_sensitivity": show_divergence_sensitivity,
    }


def _collect_detected_patterns(result: dict) -> list[dict]:
    rows: list[dict] = []
    if not isinstance(result, dict):
        return rows
    for group, items in (result.get("patterns") or {}).items():
        for item in items or []:
            row = {"group": group}
            if isinstance(item, dict):
                row.update(item)
            rows.append(row)
    return rows


def _add_sr_zones(fig: go.Figure, df: pd.DataFrame, supports: list, resistances: list) -> None:
    """Draw clearer support/resistance zones similar to TradingView markup."""
    if len(df) == 0:
        return
    close = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else max(close * 0.015, 0.01)
    zone_half = max(atr * 0.65, close * 0.010)
    x0 = df.index[0]
    x1 = df.index[-1]
    drawn = []

    def add_zone(level, zone_type="support"):
        if level is None or pd.isna(level):
            return
        level = float(level)
        if any(abs(level - prev) <= zone_half * 0.9 for prev in drawn):
            return
        drawn.append(level)
        if zone_type == "support":
            border_color = "#2EAF4A"
            fill = "rgba(255, 99, 110, 0.36)"
            label = "Support zone"
            yanchor = "bottom"
            ytext = level + zone_half * 1.05
        else:
            border_color = "#2EAF4A"
            fill = "rgba(255, 99, 110, 0.32)"
            label = "Resistance zone"
            yanchor = "top"
            ytext = level - zone_half * 1.05

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=level - zone_half, y1=level + zone_half,
            xref="x1", yref="y1",
            line=dict(color=border_color, width=2.2),
            fillcolor=fill,
            layer="below",
        )
        fig.add_annotation(
            x=x1, y=ytext, xref="x1", yref="y1",
            text=f"{label} {level:.2f}",
            showarrow=False,
            xanchor="right", yanchor=yanchor,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor=border_color, borderwidth=1,
            font=dict(size=10, color="#111827")
        )

    for level in list(supports[-4:]):
        add_zone(level, "support")
    for level in list(resistances[:4]):
        add_zone(level, "resistance")

def _add_pattern_overlays(fig: go.Figure, df: pd.DataFrame, result: dict) -> None:
    """Draw clearer, stronger pattern overlays like a manual TradingView markup."""
    try:
        patterns = _collect_detected_patterns(result)
        if not patterns or len(df) == 0:
            return

        chart = df.copy()
        last_x = chart.index[-1]
        close = float(chart["close"].iloc[-1])
        last_high = float(chart["high"].iloc[-1])
        last_low = float(chart["low"].iloc[-1])
        atr = float(chart["atr"].iloc[-1]) if "atr" in chart.columns and pd.notna(chart["atr"].iloc[-1]) else max(close * 0.015, 0.01)
        pad = max(atr * 0.8, close * 0.012)

        swings = swing_points(chart, window=3)
        highs = list(swings.get("highs", []) or [])
        lows = list(swings.get("lows", []) or [])

        sr = result.get("support_resistance", {}) if isinstance(result, dict) else {}
        supports = [float(v) for v in (sr.get("supports", []) or []) if v is not None and not pd.isna(v)]
        resistances = [float(v) for v in (sr.get("resistances", []) or []) if v is not None and not pd.isna(v)]

        pattern_names = [str(p.get("name", "")).strip() for p in patterns if p.get("name")]
        names_lower = {str(p.get("name", "")).lower(): p for p in patterns if p.get("name")}

        def _pt(obj):
            if not isinstance(obj, dict):
                return None
            pos = int(obj.get("pos", 0))
            pos = max(0, min(pos, len(chart) - 1))
            return {"time": chart.index[pos], "price": float(obj.get("price"))}

        def add_polyline(points, label="", color="#111111", width=3.0, dash="solid", markers=False):
            pts = [p for p in points if isinstance(p, dict) and p.get("time") is not None and p.get("price") is not None]
            if len(pts) < 2:
                return
            fig.add_trace(
                go.Scatter(
                    x=[p["time"] for p in pts],
                    y=[p["price"] for p in pts],
                    mode="lines+markers" if markers else "lines",
                    name=label or "Pattern",
                    line=dict(color=color, width=width, dash=dash),
                    marker=dict(size=7, color=color, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1, col=1
            )

        def add_level(y, label, color="#111111", dash="solid", width=2.4):
            if y is None or pd.isna(y):
                return
            y = float(y)
            fig.add_hline(
                y=y,
                line_dash=dash,
                line_color=color,
                line_width=width,
                annotation_text=label,
                annotation_position="top right",
                annotation_font_color=color,
                annotation_font_size=11,
                row=1, col=1
            )

        def add_target_arrow(y0, y1, label, color):
            if y1 is None or pd.isna(y1):
                return
            y1 = float(y1)
            fig.add_annotation(
                x=last_x, y=y1, xref="x1", yref="y1",
                ax=0, ay=(70 if y1 < y0 else -70),
                showarrow=True, arrowhead=3, arrowsize=1.25, arrowwidth=3.2, arrowcolor=color,
                text=label, font=dict(size=11, color=color),
                bgcolor="rgba(255,255,255,0.92)", bordercolor=color, borderwidth=1
            )

        # Candlestick pattern summary box
        candle_labels = [str(p.get("name")) for p in patterns if p.get("group") == "candlestick" and p.get("name")]
        if candle_labels:
            fig.add_annotation(
                x=chart.index[2] if len(chart) > 2 else last_x,
                y=last_high + pad * 2.0,
                xref="x1", yref="y1",
                text="Candlestick: " + " | ".join(candle_labels[:4]),
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="#111827", borderwidth=1,
                font=dict(size=11, color="#111827")
            )

        # Named neckline/level patterns
        level_map = {
            "double top": ("Double Top neckline", "#111111"),
            "double bottom": ("Double Bottom neckline", "#111111"),
            "triple top": ("Triple Top neckline", "#111111"),
            "triple bottom": ("Triple Bottom neckline", "#111111"),
            "head and shoulders": ("H&S neckline", "#111111"),
            "inverse head and shoulders": ("Inverse H&S neckline", "#111111"),
        }
        for key, (label, color) in level_map.items():
            if key in names_lower:
                add_level(names_lower[key].get("level"), label, color=color, dash="solid", width=2.8)

        # Generic pattern structure overlays using recent swing highs/lows.
        recent_highs = [_pt(p) for p in highs[-4:]]
        recent_lows = [_pt(p) for p in lows[-4:]]
        recent_highs = [p for p in recent_highs if p]
        recent_lows = [p for p in recent_lows if p]

        # Downtrend / supply line
        bearish_structure = {"descending triangle", "bearish flag", "bear flag", "head and shoulders", "double top", "triple top"}
        if any(n.lower() in bearish_structure for n in pattern_names) and len(recent_highs) >= 2:
            add_polyline([recent_highs[0], recent_highs[-1]], label="Bearish structure", color="#111111", width=3.2)

        # Uptrend / demand line
        bullish_structure = {"ascending triangle", "bullish flag", "bull flag", "inverse head and shoulders", "double bottom", "triple bottom"}
        if any(n.lower() in bullish_structure for n in pattern_names) and len(recent_lows) >= 2:
            add_polyline([recent_lows[0], recent_lows[-1]], label="Bullish structure", color="#111111", width=3.2)

        # Wedges / triangles / channels / rectangles: draw upper and lower boundaries.
        line_pattern_keywords = ("wedge", "triangle", "channel", "rectangle", "flag", "pennant")
        if any(any(k in n.lower() for k in line_pattern_keywords) for n in pattern_names):
            if len(recent_highs) >= 2:
                add_polyline([recent_highs[0], recent_highs[-1]], label="Upper boundary", color="#111111", width=3.0)
            if len(recent_lows) >= 2:
                add_polyline([recent_lows[0], recent_lows[-1]], label="Lower boundary", color="#111111", width=3.0)

        # Rectangle range zone if applicable
        if any("rectangle" in n.lower() or "range" in n.lower() for n in pattern_names) and recent_highs and recent_lows:
            top = max(p["price"] for p in recent_highs[-3:])
            bottom = min(p["price"] for p in recent_lows[-3:])
            fig.add_shape(
                type="rect",
                x0=recent_lows[0]["time"], x1=last_x,
                y0=bottom, y1=top,
                xref="x1", yref="y1",
                line=dict(color="#111111", width=2.8),
                fillcolor="rgba(0,0,0,0.04)",
                layer="below"
            )

        # Pattern levels & projections
        for p in patterns:
            nm = str(p.get("name", ""))
            lvl = p.get("level")
            proj = p.get("projection")
            if lvl is not None:
                add_level(lvl, f"{nm} level", color="#1D4ED8", dash="dash", width=2.2)
            if proj is not None:
                add_level(proj, f"{nm} target", color="#7C3AED", dash="dot", width=2.2)

        # Visual target arrow based on dominant bullish/bearish pattern.
        bullish_names = {"bullish flag", "ascending triangle", "inverse head and shoulders", "double bottom", "triple bottom", "falling wedge", "bullish pennant"}
        bearish_names = {"bearish flag", "descending triangle", "head and shoulders", "double top", "triple top", "rising wedge", "bearish pennant"}

        primary_target = None
        arrow_color = "#2563EB"
        arrow_text = "Pattern target"

        for p in patterns:
            nm = str(p.get("name", ""))
            lower_nm = nm.lower()
            if lower_nm in bullish_names:
                primary_target = p.get("projection") or p.get("level") or (resistances[0] if resistances else None)
                arrow_text = f"{nm} upside target"
                arrow_color = "#2563EB"
                break

        if primary_target is None:
            for p in patterns:
                nm = str(p.get("name", ""))
                lower_nm = nm.lower()
                if lower_nm in bearish_names:
                    primary_target = p.get("projection") or p.get("level") or (supports[-1] if supports else None)
                    arrow_text = f"{nm} downside target"
                    arrow_color = "#111111"
                    break

        if primary_target is not None:
            add_target_arrow(close, primary_target, arrow_text, arrow_color)

        # Small legend badge with active non-candle patterns.
        non_candle = [n for n in pattern_names if n and all(x not in n.lower() for x in ("engulf", "doji", "hammer", "harami", "morning", "evening", "star"))]
        if non_candle:
            fig.add_annotation(
                x=chart.index[2] if len(chart) > 2 else last_x,
                y=last_low - pad * 1.8,
                xref="x1", yref="y1",
                text="Pattern(s): " + " | ".join(non_candle[:4]),
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="#111827", borderwidth=1,
                font=dict(size=11, color="#111827")
            )

    except Exception:
        return

def _divergence_swing_points(df: pd.DataFrame, window: int = 2) -> dict:
    """Return practical swing points for divergence scanning."""
    try:
        if df is None or df.empty or len(df) < window * 2 + 3:
            return {"highs": [], "lows": []}
        highs, lows = [], []
        h = pd.to_numeric(df["high"], errors="coerce").values
        l = pd.to_numeric(df["low"], errors="coerce").values
        idx = list(df.index)
        for i in range(window, len(df) - window):
            if pd.isna(h[i]) or pd.isna(l[i]):
                continue
            if h[i] >= pd.Series(h[i - window : i + window + 1]).max():
                highs.append({"pos": i, "time": idx[i], "price": float(h[i])})
            if l[i] <= pd.Series(l[i - window : i + window + 1]).min():
                lows.append({"pos": i, "time": idx[i], "price": float(l[i])})
        return {"highs": highs, "lows": lows}
    except Exception:
        return {"highs": [], "lows": []}


def _all_divergence_points(df: pd.DataFrame, mode: str = "any", sensitivity: str = "Normal", lookback: int = 220, max_gap: int = 120) -> list[dict]:
    """Find RSI divergences using confirmed chart pivots and RSI pivots.

    Rules:
    - Regular bullish: price lower low + RSI higher low
    - Regular bearish: price higher high + RSI lower high
    - Hidden bullish: price higher low + RSI lower low
    - Hidden bearish: price lower high + RSI higher high

    The function checks multiple confirmed pivots, not only the last two candles.
    """
    if df is None or df.empty:
        return []
    try:
        chart = df.copy().tail(int(lookback)) if lookback and len(df) > int(lookback) else df.copy()
        if "rsi" not in chart.columns:
            chart = _ensure_chart_indicators(chart)
        chart["rsi"] = pd.to_numeric(chart["rsi"], errors="coerce")
        chart = chart.dropna(subset=["open", "high", "low", "close", "rsi"])
        if len(chart) < 20:
            return []

        sens = str(sensitivity or "Normal").lower()
        if sens == "strict":
            swing_window, min_price_pct, min_rsi_diff, min_gap = 3, 0.004, 3.0, 5
            bull_zone, bear_zone = 45, 55
        elif sens == "sensitive":
            swing_window, min_price_pct, min_rsi_diff, min_gap = 2, 0.0008, 0.6, 3
            bull_zone, bear_zone = 58, 42
        else:
            swing_window, min_price_pct, min_rsi_diff, min_gap = 2, 0.0018, 1.2, 4
            bull_zone, bear_zone = 52, 48

        swings = _divergence_swing_points(chart, window=swing_window)
        highs = swings.get("highs", [])[-16:]
        lows = swings.get("lows", [])[-16:]
        mode_l = str(mode or "any").lower()
        candidates = []

        def rsi_at(pt):
            return float(chart["rsi"].iloc[int(pt["pos"])])

        def add(label, bias, kind, p1, p2, r1, r2, detail, base_score):
            gap = int(p2["pos"]) - int(p1["pos"])
            if gap < min_gap or gap > int(max_gap):
                return
            price_move = abs(float(p2["price"]) - float(p1["price"])) / max(abs(float(p1["price"])), 1e-9)
            rsi_move = abs(float(r2) - float(r1))
            if price_move < min_price_pct or rsi_move < min_rsi_diff:
                return
            score = int(base_score + min(15, rsi_move) + min(10, price_move * 1000))
            candidates.append({
                "label": label,
                "bias": bias,
                "kind": kind,
                "price_points": [p1, p2],
                "rsi_points": [{"time": p1["time"], "price": float(r1)}, {"time": p2["time"], "price": float(r2)}],
                "score": min(99, score),
                "detail": detail,
                "pivot_gap": gap,
                "price_change_pct": round(price_move * 100, 3),
                "rsi_change": round(rsi_move, 2),
            })

        # High pivots: bearish divergences.
        for a in range(len(highs)):
            for b in range(a + 1, len(highs)):
                h1, h2 = highs[a], highs[b]
                r1, r2 = rsi_at(h1), rsi_at(h2)

                # Regular bearish
                if h2["price"] > h1["price"] and r2 < r1 and max(r1, r2) >= bear_zone:
                    add(
                        "Bearish RSI Divergence", "Bearish", "Regular Bearish",
                        h1, h2, r1, r2,
                        "Price made higher high but RSI made lower high.",
                        78,
                    )

                # Hidden bearish
                if h2["price"] < h1["price"] and r2 > r1 and max(r1, r2) >= bear_zone:
                    add(
                        "Hidden Bearish RSI Divergence", "Bearish", "Hidden Bearish",
                        h1, h2, r1, r2,
                        "Price made lower high but RSI made higher high.",
                        64,
                    )

        # Low pivots: bullish divergences.
        for a in range(len(lows)):
            for b in range(a + 1, len(lows)):
                l1, l2 = lows[a], lows[b]
                r1, r2 = rsi_at(l1), rsi_at(l2)

                # Regular bullish
                if l2["price"] < l1["price"] and r2 > r1 and min(r1, r2) <= bull_zone:
                    add(
                        "Bullish RSI Divergence", "Bullish", "Regular Bullish",
                        l1, l2, r1, r2,
                        "Price made lower low but RSI made higher low.",
                        78,
                    )

                # Hidden bullish
                if l2["price"] > l1["price"] and r2 < r1 and min(r1, r2) <= bull_zone:
                    add(
                        "Hidden Bullish RSI Divergence", "Bullish", "Hidden Bullish",
                        l1, l2, r1, r2,
                        "Price made higher low but RSI made lower low.",
                        64,
                    )

        if "bullish" in mode_l:
            candidates = [c for c in candidates if c.get("bias") == "Bullish"]
        elif "bearish" in mode_l:
            candidates = [c for c in candidates if c.get("bias") == "Bearish"]
        if "regular" in mode_l:
            candidates = [c for c in candidates if "Hidden" not in c.get("kind", "")]
        elif "hidden" in mode_l:
            candidates = [c for c in candidates if "Hidden" in c.get("kind", "")]

        dedup = {}
        for c in candidates:
            key = (c.get("kind"), int(c["price_points"][0]["pos"]), int(c["price_points"][1]["pos"]))
            if key not in dedup or c.get("score", 0) > dedup[key].get("score", 0):
                dedup[key] = c
        candidates = list(dedup.values())
        # Latest pivot first, then score.
        candidates.sort(key=lambda c: (int(c["price_points"][1]["pos"]), int(c.get("score", 0))), reverse=True)
        return candidates
    except Exception:
        return []


def _detect_divergence_points(df: pd.DataFrame, mode: str = "any", sensitivity: str = "Normal") -> dict | None:
    matches = _all_divergence_points(df, mode=mode, sensitivity=sensitivity, lookback=220, max_gap=120)
    return matches[0] if matches else None

def _add_divergence_overlays(fig: go.Figure, df: pd.DataFrame, result: dict, show_rsi_panel: bool = True, mode: str = "any", sensitivity: str = "Normal") -> None:
    if df is None or df.empty or "rsi" not in df.columns:
        return
    div = _detect_divergence_points(df, mode=mode, sensitivity=sensitivity)
    if not div:
        return

    bias = div.get("bias", "Neutral")
    color = "#2563EB" if bias == "Bullish" else "#EF4444"
    fill = "rgba(37,99,235,0.16)" if bias == "Bullish" else "rgba(239,68,68,0.14)"
    price_pts = div.get("price_points", [])
    rsi_pts = div.get("rsi_points", [])
    label = div.get("label", "RSI Divergence")
    kind = div.get("kind", label)

    if len(price_pts) >= 2:
        x0, x1 = price_pts[0]["time"], price_pts[1]["time"]
        y0, y1 = float(price_pts[0]["price"]), float(price_pts[1]["price"])

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines+markers",
                name=f"{kind} price",
                line=dict(color=color, width=3.2, dash="solid"),
                marker=dict(color=color, size=8, symbol="circle"),
                hovertemplate=kind + " price: %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_annotation(
            x=x1, y=y1, xref="x1", yref="y1",
            text=f"<b>{kind}</b><br>Price pivot",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.4, arrowcolor=color,
            ax=40, ay=-45 if bias == "Bearish" else 45,
            bgcolor="rgba(255,255,255,0.94)", bordercolor=color, borderwidth=1,
            font=dict(size=10, color=color),
        )

    if show_rsi_panel and len(rsi_pts) >= 2:
        x0, x1 = rsi_pts[0]["time"], rsi_pts[1]["time"]
        r0, r1 = float(rsi_pts[0]["price"]), float(rsi_pts[1]["price"])

        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0],
                y=[30 if bias == "Bullish" else 70, 30 if bias == "Bullish" else 70, r1, r0],
                fill="toself",
                mode="none",
                name=f"{kind} RSI zone",
                fillcolor=fill,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[r0, r1],
                mode="lines+markers",
                name=f"{kind} RSI",
                line=dict(color=color, width=3.2, dash="solid"),
                marker=dict(color=color, size=8, symbol="circle"),
                hovertemplate=kind + " RSI: %{y:.2f}<extra></extra>",
                showlegend=True,
            ),
            row=3,
            col=1,
        )

        fig.add_annotation(
            x=x1, y=r1, xref="x3", yref="y3",
            text=f"<b>{kind}</b><br>RSI confirmation",
            showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2.2, arrowcolor=color,
            ax=35, ay=-35 if bias == "Bearish" else 35,
            bgcolor="rgba(255,255,255,0.94)", bordercolor=color, borderwidth=1,
            font=dict(size=10, color=color),
        )



def _heikin_ashi_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Build Heikin Ashi OHLC for clearer trend-view candles.

    Indicators remain based on original OHLC/close; only the candle display changes.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    src = df.copy()
    o = pd.to_numeric(src["open"], errors="coerce")
    h = pd.to_numeric(src["high"], errors="coerce")
    l = pd.to_numeric(src["low"], errors="coerce")
    c = pd.to_numeric(src["close"], errors="coerce")

    ha_close = (o + h + l + c) / 4.0
    ha_open = ha_close.copy()
    if len(src) > 0:
        ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
        for i in range(1, len(src)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame({
        "open": ha_open,
        "high": ha_high,
        "low": ha_low,
        "close": ha_close,
    }, index=src.index)



def make_chart(df: pd.DataFrame, result: dict, title: str = "", visibility: dict | None = None) -> go.Figure:
    """Light technical chart with checklist-controlled layers."""
    visibility = visibility or {}
    show_candles = visibility.get("candles", True)
    candle_type = visibility.get("candle_type", "Simple Candles")
    clear_candles = visibility.get("clear_candles", True)
    show_ma = visibility.get("ma", True)
    show_ema21 = visibility.get("ema21", True)
    show_ema51 = visibility.get("ema51", True)
    show_klinger = visibility.get("klinger", False)
    show_alligator = visibility.get("alligator", False)
    show_sr = visibility.get("support_resistance", True)
    show_levels = visibility.get("levels", True)
    show_volume = visibility.get("volume", True)
    show_rsi = visibility.get("rsi", True)
    show_macd = visibility.get("macd", True)
    show_badges = visibility.get("badges", True)
    show_last_price = visibility.get("last_price", True)
    show_pattern_overlay = visibility.get("pattern_overlay", True)
    show_divergence = visibility.get("divergence", True)
    show_divergence_type = visibility.get("divergence_type", "Any")
    show_divergence_sensitivity = visibility.get("divergence_sensitivity", "Sensitive")

    df = _ensure_chart_indicators(df)
    if clear_candles and isinstance(df, pd.DataFrame) and len(df) > 160:
        df = df.tail(160).copy()
    if "rsi" in df.columns and "rsi_signal" not in df.columns:
        df = df.copy()
        df["rsi_signal"] = df["rsi"].rolling(9, min_periods=3).mean()

    candle_df = df[["open", "high", "low", "close"]].copy() if {"open", "high", "low", "close"}.issubset(set(df.columns)) else df.copy()
    if str(candle_type).startswith("Heikin"):
        ha = _heikin_ashi_ohlc(df)
        if isinstance(ha, pd.DataFrame) and not ha.empty:
            candle_df = ha

    show_oscillator_panel = bool(show_macd or show_klinger)
    panel_flags = [show_volume, show_rsi, show_oscillator_panel]
    active_panel_count = sum(1 for x in panel_flags if x)
    if clear_candles:
        if active_panel_count == 0:
            row_heights = [1.0, 0.001, 0.001, 0.001]
        elif active_panel_count == 1:
            row_heights = [0.82, 0.13 if show_volume else 0.001, 0.13 if show_rsi else 0.001, 0.13 if show_oscillator_panel else 0.001]
        elif active_panel_count == 2:
            row_heights = [0.72, 0.12 if show_volume else 0.001, 0.12 if show_rsi else 0.001, 0.12 if show_oscillator_panel else 0.001]
        else:
            row_heights = [0.66, 0.10, 0.11, 0.13]
    else:
        if active_panel_count == 0:
            row_heights = [1.0, 0.001, 0.001, 0.001]
        elif active_panel_count == 1:
            row_heights = [0.74, 0.18 if show_volume else 0.001, 0.18 if show_rsi else 0.001, 0.18 if show_oscillator_panel else 0.001]
        elif active_panel_count == 2:
            row_heights = [0.62, 0.14 if show_volume else 0.001, 0.14 if show_rsi else 0.001, 0.14 if show_oscillator_panel else 0.001]
        else:
            row_heights = [0.52, 0.14, 0.16, 0.18]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    close_series = pd.to_numeric(df.get("close"), errors="coerce")
    latest_close = _to_float(close_series.iloc[-1] if len(close_series) else None)
    prev_close = _to_float(close_series.iloc[-2] if len(close_series) > 1 else latest_close)
    delta = None if latest_close is None or prev_close is None else latest_close - prev_close
    pct = None if latest_close is None or prev_close in (None, 0) else (delta / prev_close) * 100

    if show_candles:
        fig.add_trace(
            go.Candlestick(
                x=candle_df.index,
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
                name="Heikin Ashi" if str(candle_type).startswith("Heikin") else "Price",
                increasing_line_color="#00A884",
                decreasing_line_color="#FF3131",
                increasing_fillcolor="#00A884",
                decreasing_fillcolor="#FF3131",
                whiskerwidth=0.85 if clear_candles else 0.45,
            ),
            row=1,
            col=1,
        )

    line_specs = [
        ("EMA9", "ema9", "#ff4d88", 1.3, show_ma),
        ("EMA21", "ema21", "#00A3A3", 1.45, show_ema21),
        ("EMA51", "ema51", "#8B5CF6", 1.45, show_ema51),
        ("MA20", "ma20", "#6abf69", 1.2, show_ma),
        ("MA50", "ma50", "#356dff", 1.4, show_ma),
        ("MA200", "ma200", "#94A3B8", 1.1, show_ma),
        ("Alligator Jaw 13/8", "jaw", "#2563EB", 1.8, show_alligator),
        ("Alligator Teeth 8/5", "teeth", "#E11D48", 1.8, show_alligator),
        ("Alligator Lips 5/3", "lips", "#16A34A", 1.8, show_alligator),
    ]
    visible_alligator_lines = 0
    for name, key, color, width, enabled in line_specs:
        if enabled and key in df.columns and not pd.to_numeric(df[key], errors="coerce").dropna().empty:
            if key in {"jaw", "teeth", "lips"}:
                visible_alligator_lines += 1
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[key],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=(width + 0.35 if clear_candles and key in {"ema21", "ema51"} else width)),
                    hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if show_alligator and visible_alligator_lines == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.98,
            text="Alligator enabled but not enough candles to draw shifted lines",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E11D48",
            font=dict(size=11, color="#E11D48"),
        )

    sr = result.get("support_resistance", {}) if isinstance(result, dict) else {}
    supports = [lvl for lvl in sr.get("supports", []) if lvl is not None]
    resistances = [lvl for lvl in sr.get("resistances", []) if lvl is not None]
    if show_sr:
        _add_sr_zones(fig, df, supports, resistances)
        for idx, lvl in enumerate(supports[-3:], start=1):
            fig.add_hline(y=lvl, line_dash="dot", line_color="#6B7280", line_width=0.9, annotation_text=f"S{idx}", annotation_position="top left", annotation_font_color="#6B7280", row=1, col=1)
        for idx, lvl in enumerate(resistances[:3], start=1):
            fig.add_hline(y=lvl, line_dash="dot", line_color="#6B7280", line_width=0.9, annotation_text=f"R{idx}", annotation_position="top left", annotation_font_color="#6B7280", row=1, col=1)

    decision_row = result.get("decision_row", {}) if isinstance(result, dict) else {}
    plan = result.get("trade_plan", {}) if isinstance(result, dict) else {}
    level_specs = [
        ("Entry", _to_float(decision_row.get("Entry", plan.get("entry"))), "dot", "#2962FF"),
        ("SL", _to_float(decision_row.get("Stop Loss", plan.get("stop_loss"))), "dot", "#F23645"),
        ("TP1", _to_float(decision_row.get("TP1", plan.get("take_profit"))), "dot", "#089981"),
        ("TP2", _to_float(decision_row.get("TP2")), "dot", "#089981"),
        ("TP3", _to_float(decision_row.get("TP3")), "dot", "#089981"),
    ]
    if show_levels:
        for label, value, dash, color in level_specs:
            if value is not None:
                fig.add_hline(y=value, line_dash=dash, line_color=color, line_width=1.2, annotation_text=label, annotation_position="top right", annotation_font_color=color, row=1, col=1)

    if show_last_price and latest_close is not None:
        fig.add_hline(y=latest_close, line_dash="dot", line_color="#10B981" if delta is not None and delta >= 0 else "#EF4444", line_width=1, annotation_text="Last", annotation_position="bottom right", annotation_font_color="#10B981" if delta is not None and delta >= 0 else "#EF4444", row=1, col=1)

    if show_volume and "volume" in df.columns:
        vol_colors = ["#089981" if c >= o else "#F23645" for o, c in zip(candle_df["open"], candle_df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker=dict(color=vol_colors), opacity=0.45 if clear_candles else 0.65), row=2, col=1)
        if "volume_ma20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["volume_ma20"], mode="lines", name="Vol MA20", line=dict(color="#4B5563", width=1.2)), row=2, col=1)
    else:
        fig.update_yaxes(visible=False, row=2, col=1)
        fig.update_xaxes(visible=False, row=2, col=1)

    if show_rsi:
        if "rsi" not in df.columns or pd.to_numeric(df.get("rsi"), errors="coerce").dropna().empty:
            # Last safety net: calculate RSI here so panel never disappears when checkbox is ON.
            try:
                delta = pd.to_numeric(df["close"], errors="coerce").diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
                df["rsi_signal"] = df["rsi"].rolling(9, min_periods=3).mean()
            except Exception:
                df["rsi"] = pd.Series([50] * len(df), index=df.index)
                df["rsi_signal"] = df["rsi"]

        fig.add_hrect(y0=30, y1=70, fillcolor="#7C4DFF", opacity=0.08, line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], mode="lines", name="RSI 14", line=dict(color="#7E57C2", width=1.6)), row=3, col=1)
        if "rsi_signal" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["rsi_signal"], mode="lines", name="RSI MA", line=dict(color="#E0B400", width=1.3)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#9CA3AF", line_width=1, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#D1D5DB", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#9CA3AF", line_width=1, row=3, col=1)
    else:
        fig.update_yaxes(visible=False, row=3, col=1)
        fig.update_xaxes(visible=False, row=3, col=1)

    if show_oscillator_panel:
        if show_macd and "macd" in df.columns and "macd_hist" in df.columns:
            hist_colors = ["#089981" if (pd.notna(v) and v >= 0) else "#F23645" for v in df["macd_hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD Hist", marker=dict(color=hist_colors), opacity=0.45), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color="#2962FF", width=1.5)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], mode="lines", name="MACD Signal", line=dict(color="#F59E0B", width=1.3)), row=4, col=1)
        if show_klinger and "klinger" in df.columns and "klinger_signal" in df.columns:
            if "klinger_hist" in df.columns:
                kh_colors = ["#10B981" if (pd.notna(v) and v >= 0) else "#EF4444" for v in df["klinger_hist"]]
                fig.add_trace(go.Bar(x=df.index, y=df["klinger_hist"], name="Klinger Hist", marker=dict(color=kh_colors), opacity=0.28), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["klinger"], mode="lines", name="Klinger", line=dict(color="#111827", width=1.4)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["klinger_signal"], mode="lines", name="Klinger Signal", line=dict(color="#DC2626", width=1.2, dash="dot")), row=4, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", line_width=1, row=4, col=1)
    else:
        fig.update_yaxes(visible=False, row=4, col=1)
        fig.update_xaxes(visible=False, row=4, col=1)

    if show_divergence and show_rsi:
        _add_divergence_overlays(fig, df, result if isinstance(result, dict) else {}, show_rsi_panel=show_rsi, mode=show_divergence_type, sensitivity=show_divergence_sensitivity)

    if show_pattern_overlay:
        _add_pattern_overlays(fig, df, result if isinstance(result, dict) else {})

    pattern_names = []
    for _group, items in (result.get("patterns", {}) if isinstance(result, dict) else {}).items():
        for item in items or []:
            name = item.get("name") or item.get("label")
            if name:
                pattern_names.append(str(name))
    pattern_summary = ", ".join(pattern_names[:8]) if pattern_names else "No strong named pattern"

    decision_text = str(decision_row.get("Bot Decision", decision_row.get("Autopilot Action", plan.get("order_type", "WATCH")))).upper()
    buy_badge_value = _to_float(decision_row.get("Entry", plan.get("entry")), latest_close)
    sell_badge_value = _to_float(decision_row.get("TP1", resistances[0] if resistances else latest_close), latest_close)

    if show_badges:
        if sell_badge_value is not None:
            fig.add_annotation(xref="paper", yref="paper", x=0.00, y=1.15, showarrow=False, align="center", text=f"<b>{sell_badge_value:.2f}</b><br>SELL", bordercolor="#E46A70", borderwidth=1.6, borderpad=8, bgcolor="#FFF7F7", font=dict(color="#E24B54", size=12))
        if buy_badge_value is not None:
            fig.add_annotation(xref="paper", yref="paper", x=0.08, y=1.15, showarrow=False, align="center", text=f"<b>{buy_badge_value:.2f}</b><br>BUY", bordercolor="#5C8FFF", borderwidth=1.6, borderpad=8, bgcolor="#F5F9FF", font=dict(color="#356DFF", size=12))
        fig.add_annotation(xref="paper", yref="paper", x=0.16, y=1.15, showarrow=False, align="left", text=f"<b>Decision:</b> {decision_text}", font=dict(color="#111827", size=12))
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=1.07, showarrow=False, align="left", text=f"<b>Patterns:</b> {pattern_summary}", font=dict(color="#374151", size=11))

    title_text = title or "Execution Chart"
    if str(candle_type).startswith("Heikin"):
        title_text = f"{title_text} | Heikin Ashi"
    if latest_close is not None:
        delta_text = "" if delta is None or pct is None else f" | Last: {latest_close:.2f} ({delta:+.2f}, {pct:+.2f}%)"
        title_text = f"{title_text}{delta_text}"

    fig.update_layout(
        height=(1180 if (show_rsi or show_macd or show_klinger or show_volume) else 900) if clear_candles else (1100 if (show_rsi or show_macd or show_klinger or show_volume) else 860),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.01,
        legend_xanchor="left",
        legend_x=0.25,
        title=dict(text=title_text, x=0.01, xanchor="left", font=dict(size=17, color="#111827")),
        margin=dict(l=20, r=55, t=110, b=30),
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#111827", family="Arial, sans-serif"),
        dragmode="pan",
        bargap=0.08,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F0F3F7", zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F3F7", zeroline=False, title_text="Price", side="right", row=1, col=1, tickformat=",.2f")
    fig.update_yaxes(showgrid=show_volume, gridcolor="#F3F4F6", zeroline=False, title_text="Volume", side="right", row=2, col=1)
    fig.update_yaxes(showgrid=show_rsi, gridcolor="#EEF2F7", zeroline=False, title_text="RSI", range=[0, 100], side="right", row=3, col=1)
    fig.update_yaxes(showgrid=show_macd, gridcolor="#EEF2F7", zeroline=False, title_text="MACD", side="right", row=4, col=1)
    return fig


def render_chart_with_downloads(fig: go.Figure, *, symbol: str, key_prefix: str, title: str | None = None) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
            "toImageButtonOptions": {"filename": f"{symbol}_chart", "format": "png", "scale": 2},
        },
    )
    c1, c2 = st.columns(2)
    html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    with c1:
        st.download_button("Download Chart (HTML)", data=html_bytes, file_name=f"{symbol}_chart.html", mime="text/html", key=f"{key_prefix}_html_download", use_container_width=True)
    try:
        png_bytes = pio.to_image(fig, format="png", width=1600, height=1100, scale=2)
        with c2:
            st.download_button("Download Chart (PNG)", data=png_bytes, file_name=f"{symbol}_chart.png", mime="image/png", key=f"{key_prefix}_png_download", use_container_width=True)
    except Exception:
        with c2:
            st.caption("PNG download becomes available when Plotly image export support (kaleido) is installed. HTML download is ready now.")


INVESTING_PSX_SLUGS = {
    "OGDC": "oil---gas-dev",
    "PPL": "pakistan-petroleum",
    "MARI": "mari-gas",
    "POL": "pakistan-oilfields",
    "PSO": "pakistan-state-oil",
    "UBL": "united-bank-ltd",
    "HBL": "habib-bank-ltd",
    "MCB": "mcb-bank",
    "BAFL": "bank-alfalah",
    "NBP": "national-bank-of-pakistan",
    "SYS": "systems-ltd",
    "HUBC": "hub-power-co",
    "LUCK": "lucky-cement",
    "DGKC": "dg-khan-cement",
    "MLCF": "maple-leaf-cement",
    "FCCL": "fauji-cement",
    "FFC": "fauji-fertilizer",
    "EFERT": "engro-fertilizers",
    "ENGROH": "engro-corp",
    "COLG": "colgate-palmolive-pakistan",
    "ABOT": "abbott-laboratories-pakistan",
    "NATF": "national-foods",
    "INDU": "indus-motor-co",
    "ATLH": "atlas-honda",
    "SAZEW": "sazgar-engineering",
}


def _investing_symbol_slug(symbol: str) -> str:
    cleaned = str(symbol or "").strip().upper().replace(" ", "")
    return INVESTING_PSX_SLUGS.get(cleaned, cleaned.lower())


def _investing_chart_url(symbol: str, manual_value: str = "", page_type: str = "Advanced chart") -> str:
    manual_value = str(manual_value or "").strip()
    if manual_value.startswith("http://") or manual_value.startswith("https://"):
        return manual_value
    slug = manual_value.strip("/") if manual_value else _investing_symbol_slug(symbol)
    if page_type == "Simple chart page":
        return f"https://www.investing.com/equities/{slug}-chart"
    if page_type == "Technical analysis page":
        return f"https://www.investing.com/equities/{slug}-technical"
    return f"https://www.investing.com/equities/{slug}-advanced-chart"


def render_investing_live_chart(symbol: str, key_prefix: str) -> None:
    st.markdown("##### Investing.com Chart Links")
    with st.container(border=True):
        st.markdown("### Investing.com chart options")
        c1, c2 = st.columns([1, 1])
        page_type = c1.selectbox(
            "Investing.com page",
            ["Advanced chart", "Simple chart page", "Technical analysis page"],
            index=0,
            key=f"{key_prefix}_inv_page_type",
        )
        open_mode = c2.selectbox(
            "Open mode",
            ["Open in new browser tab", "Show warning only"],
            index=0,
            key=f"{key_prefix}_inv_open_mode",
        )

        manual = st.text_input(
            "Investing.com slug or full chart URL",
            value=_investing_symbol_slug(symbol),
            key=f"{key_prefix}_inv_manual_slug",
            help="Example for OGDC: oil---gas-dev. You can also paste the full Investing.com advanced-chart URL.",
        )
        st.caption("Investing.com blocks most chart pages from being shown inside Streamlit iframes. Use the button below to open the real chart in a new tab. Keep Bot chart enabled for in-app charting.")

    url = _investing_chart_url(symbol, manual, page_type)
    search_url = f"https://www.investing.com/search/?q={symbol}"
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.link_button("Open Investing.com Chart", url, use_container_width=True)
    with c2:
        st.link_button("Open Investing.com Search", search_url, use_container_width=True)
    with c3:
        st.info(
            "Investing.com chart pages usually cannot render inside this app because the site prevents iframe embedding. "
            "This is a website security restriction, not a bot error. The reliable method is opening the chart in a new tab while using the bot chart inside Streamlit."
        )

    if open_mode == "Show warning only":
        st.warning("Iframe display is intentionally disabled to avoid blank Investing.com panels.")

    st.caption(
        "This Investing.com section is independent from the bot report. Changing Investing.com settings or bot checklist boxes will not clear the latest Decision Center results."
    )



CHART_TIMEFRAME_OPTIONS = [
    "Original Data",
    "1 minute",
    "5 minutes",
    "15 minutes",
    "30 minutes",
    "1 hour",
    "4 hour",
    "Daily",
    "Weekly",
    "Monthly",
]

_CHART_TIMEFRAME_RULES = {
    "1 minute": "1min",
    "5 minutes": "5min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "4 hour": "4h",
    "Daily": "1D",
    "Weekly": "1W",
    "Monthly": "1M",
}


def _infer_chart_base_minutes(df: pd.DataFrame) -> float | None:
    try:
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 3:
            return None
        idx = pd.to_datetime(df.index)
        deltas = pd.Series(idx).sort_values().diff().dropna()
        if deltas.empty:
            return None
        return float(deltas.median().total_seconds() / 60.0)
    except Exception:
        return None


def _target_minutes_from_label(label: str) -> float | None:
    label = str(label or "")
    if label == "1 minute":
        return 1
    if label == "5 minutes":
        return 5
    if label == "15 minutes":
        return 15
    if label == "30 minutes":
        return 30
    if label == "1 hour":
        return 60
    if label == "4 hour":
        return 240
    if label == "Daily":
        return 1440
    if label == "Weekly":
        return 10080
    if label == "Monthly":
        return 43200
    return None


def _resample_chart_timeframe(df: pd.DataFrame, timeframe_label: str) -> tuple[pd.DataFrame, str]:
    """Resample chart display to selected timeframe.

    Lower timeframes cannot be created from higher timeframe data.
    Example: daily data cannot become 1-minute candles. In that case the original data is kept.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df, "No chart data."

    label = str(timeframe_label or "Original Data")
    if label == "Original Data":
        return df, "Original loaded timeframe."

    rule = _CHART_TIMEFRAME_RULES.get(label)
    if not rule:
        return df, "Unknown chart timeframe; original data shown."

    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()

        base_min = _infer_chart_base_minutes(out)
        target_min = _target_minutes_from_label(label)

        if base_min is not None and target_min is not None and target_min < base_min * 0.75:
            return out, f"Current loaded data is {base_min:.0f}-minute; lower {label} candles need direct intraday data."

        agg = {}
        for col in out.columns:
            lc = str(col).lower()
            if lc == "open":
                agg[col] = "first"
            elif lc == "high":
                agg[col] = "max"
            elif lc == "low":
                agg[col] = "min"
            elif lc == "close":
                agg[col] = "last"
            elif lc == "volume":
                agg[col] = "sum"

        if not {"open", "high", "low", "close"}.issubset(set([str(c).lower() for c in agg.keys()])):
            return out, "OHLC columns not found; original data shown."

        resampled = out.resample(rule).agg(agg).dropna(subset=[c for c in out.columns if str(c).lower() in {"open", "high", "low", "close"}])
        if resampled.empty or len(resampled) < 5:
            return out, f"Not enough candles after {label} resampling; original data shown."

        resampled = _ensure_chart_indicators(resampled)
        return resampled, f"Chart resampled to {label}. Candles: {len(resampled)}."
    except Exception as exc:
        return df, f"Chart timeframe conversion failed: {exc}"




def _chart_tf_to_yahoo_params(label: str) -> tuple[str, str, bool]:
    """Return period, interval, needs_4h_resample."""
    label = str(label or "Original Data")
    if label == "1 minute":
        return "7d", "1m", False
    if label == "5 minutes":
        return "1mo", "5m", False
    if label == "15 minutes":
        return "2mo", "15m", False
    if label == "30 minutes":
        return "2mo", "30m", False
    if label == "1 hour":
        return "6mo", "1h", False
    if label == "4 hour":
        return "1y", "1h", True
    if label == "Daily":
        return "2y", "1d", False
    if label == "Weekly":
        return "5y", "1wk", False
    if label == "Monthly":
        return "10y", "1mo", False
    return "", "", False


def _load_chart_data_for_selected_timeframe(symbol: str, selected_chart_tf: str, fallback_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Load chart data directly for selected timeframe when possible.

    This fixes the issue where daily candles cannot be converted into 4H/1H/15M candles.
    """
    label = str(selected_chart_tf or "Original Data")
    if label == "Original Data":
        return fallback_df, "Original loaded timeframe."

    period, interval, needs_4h = _chart_tf_to_yahoo_params(label)
    if not period or not interval:
        return fallback_df, "Unknown timeframe; original chart shown."

    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return fallback_df, "No symbol available for direct timeframe loading; original chart shown."

    try:
        loaded = _cached_yahoo_ohlcv(symbol, period=period, interval=interval)
        if loaded is None or not isinstance(loaded, pd.DataFrame) or loaded.empty:
            return fallback_df, f"No direct {label} data returned for {symbol}; original chart shown."

        loaded = loaded.copy()
        try:
            loaded.index = pd.to_datetime(loaded.index)
            loaded = loaded.sort_index()
        except Exception:
            pass

        if needs_4h:
            loaded, msg = _resample_chart_timeframe(loaded, "4 hour")
            loaded = _ensure_chart_indicators(loaded)
            return loaded, f"Loaded 1H data and converted to 4 hour. {msg}"

        loaded = _ensure_chart_indicators(loaded)
        return loaded, f"Loaded direct {label} chart data for {symbol}. Candles: {len(loaded)}."
    except Exception as exc:
        return fallback_df, f"Could not load direct {label} chart for {symbol}: {exc}. Original chart shown."



def chart_date_range_controls(df: pd.DataFrame, key_prefix: str, symbol: str = "") -> tuple[pd.DataFrame, dict]:
    """Chart timeframe selector without visible From/To/Quick range controls.

    Selecting a timeframe now tries to load that timeframe directly.
    """
    if df is None or df.empty:
        return df, {"status": "empty"}

    base_df = df.copy()
    try:
        base_df.index = pd.to_datetime(base_df.index)
        base_df = base_df.sort_index()
    except Exception:
        pass

    st.markdown("##### Chart Timeframe")
    selected_chart_tf = st.selectbox(
        "Chart timeframe",
        CHART_TIMEFRAME_OPTIONS,
        index=0,
        key=f"{key_prefix}_chart_timeframe_select",
        help="Select display timeframe. Intraday timeframes load directly from Yahoo when available.",
    )

    chart_df, tf_msg = _load_chart_data_for_selected_timeframe(symbol, selected_chart_tf, base_df)
    st.caption(tf_msg)

    try:
        chart_df.index = pd.to_datetime(chart_df.index)
        chart_df = chart_df.sort_index()
    except Exception:
        st.warning("Chart index date/time could not be read, so original chart data is shown.")
        chart_df = base_df

    try:
        if selected_chart_tf == "1 minute":
            max_rows = 240
        elif selected_chart_tf in {"5 minutes", "15 minutes", "30 minutes"}:
            max_rows = 260
        elif selected_chart_tf == "1 hour":
            max_rows = 220
        elif selected_chart_tf == "4 hour":
            max_rows = 180
        elif selected_chart_tf == "Daily":
            max_rows = 180
        elif selected_chart_tf == "Weekly":
            max_rows = 160
        elif selected_chart_tf == "Monthly":
            max_rows = 120
        else:
            max_rows = 180

        filtered = chart_df.tail(max_rows).copy() if len(chart_df) > max_rows else chart_df.copy()

        from_date = pd.to_datetime(filtered.index.min()).date()
        to_date = pd.to_datetime(filtered.index.max()).date()

        st.caption(
            f"Visible candles: {len(filtered)} / {len(chart_df)} | "
            f"Range: {from_date} to {to_date}."
        )
        return filtered, {
            "status": "ok",
            "from": from_date,
            "to": to_date,
            "rows": len(filtered),
            "total_rows": len(chart_df),
            "timeframe": selected_chart_tf,
            "auto_range": True,
            "message": tf_msg,
        }
    except Exception:
        return chart_df, {
            "status": "ok",
            "from": "",
            "to": "",
            "rows": len(chart_df),
            "total_rows": len(chart_df),
            "timeframe": selected_chart_tf,
            "auto_range": True,
            "message": tf_msg,
        }



def render_chart_engine(df: pd.DataFrame, result: dict, *, symbol: str, title: str, key_prefix: str) -> None:
    st.markdown("##### In-App Bot Chart — TradingView-Style Presentation")
    chart_df, chart_range_info = chart_date_range_controls(df, key_prefix, symbol=symbol)
    pattern_df = _pattern_rows_from_result(result)
    if isinstance(pattern_df, pd.DataFrame) and not pattern_df.empty and str(pattern_df.iloc[0].get("Pattern", "")).lower() != "no named pattern detected":
        pattern_names = [str(x) for x in pattern_df["Pattern"].dropna().astype(str).tolist() if str(x).strip() and str(x).strip() != "—"]
        if pattern_names:
            st.info("Detected patterns on the current chart: " + " | ".join(pattern_names[:12]))
    else:
        st.info("No strong named pattern was detected in the current chart run. The bot chart still shows trend, support/resistance, RSI, MACD, and trade levels.")

    st.caption("For easy viewing keep Clear Candles Mode ON and use 3M/6M range. Use Candle Type to switch between Simple Candles and Heikin Ashi Candles.")
    visibility = chart_display_controls(key_prefix)
    range_suffix = ""
    if isinstance(chart_range_info, dict) and chart_range_info.get("status") == "ok":
        tf_label = chart_range_info.get("timeframe", "Original Data")
        range_suffix = f" | {tf_label}"
    fig = make_chart(chart_df, result, title=f"{title}{range_suffix}", visibility=visibility)
    render_chart_with_downloads(fig, symbol=symbol, key_prefix=key_prefix)

def scan_progress_ui():
    progress = st.progress(0, text="Preparing scan...")
    status = st.empty()

    def update(i: int, total: int, symbol: str):
        pct = 0 if total <= 0 else int(i / total * 100)
        progress.progress(min(pct, 100), text=f"Scanning {i}/{total}: {symbol}")
        status.caption(f"Current symbol: {symbol}")

    return progress, status, update


def overview_pro_metrics(pro: dict) -> None:
    latest = pro.get("latest_metrics", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PRO Score", pro["pro_score"])
    c2.metric("Grade", pro["pro_grade"])
    c3.metric("Trade Quality", pro["trade_quality"])
    c4.metric("Risk", pro["risk_level"])
    c5.metric("RSI", latest.get("rsi"))
    c6.metric("ADX", latest.get("adx"))



def load_frames_for_style(
    symbol: str,
    source: str,
    analysis_tf: str,
    execution_tf: str,
    period: str,
    dps_mode: str,
):
    if source.startswith("Yahoo Finance"):
        higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
        lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
        return higher_df, lower_df
    base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
    return resample_ohlcv(base_df, analysis_tf), resample_ohlcv(base_df, execution_tf)


def show_confluence_box(confluence: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Confluence Score", confluence.get("confluence_score"))
    c2.metric("Confluence Grade", confluence.get("confluence_grade"))
    c3.metric("Verdict", confluence.get("confluence_verdict"))
    verdict = confluence.get("confluence_verdict", "")
    action = confluence.get("quick_action", "")
    if "Avoid" in verdict or "Weak" in verdict:
        st.error(action)
    elif "High" in verdict or "Elite" in verdict:
        st.success(action)
    else:
        st.warning(action)
    factors = confluence_table(confluence)
    if not factors.empty:
        st.dataframe(factors, use_container_width=True, hide_index=True)






def knowledge_layer_selector_ui(key_prefix: str, default_profile: str = "Complete Uploaded Knowledge Brain") -> tuple[str, list[str]]:
    st.markdown("#### Uploaded Knowledge Selector")
    st.caption(
        "Select which uploaded PDFs, pattern rules, scenario systems, fundamental files, and risk frameworks should guide this scan."
    )
    k1, k2 = st.columns([1, 1.4])
    profile_options = list(KNOWLEDGE_PROFILES.keys())
    default_index = profile_options.index(default_profile) if default_profile in profile_options else 0
    profile = k1.selectbox(
        "Knowledge profile",
        profile_options,
        index=default_index,
        key=f"{key_prefix}_knowledge_profile",
    )
    default_layers = layers_for_profile(profile)
    custom_layers = []
    if profile == "Custom Selection":
        custom_layers = k2.multiselect(
            "Select uploaded-data layers",
            KNOWLEDGE_LAYERS,
            default=KNOWLEDGE_LAYERS,
            key=f"{key_prefix}_knowledge_layers",
        )
    else:
        k2.info("Active layers: " + ", ".join(default_layers))
    layers = layers_for_profile(profile, custom_layers)
    summary = selector_summary(profile, layers)
    s1, s2, s3 = st.columns(3)
    s1.metric("Selected Layers", summary["Layer Count"])
    s2.metric("Technical Layers", summary["Technical Layers"])
    s3.metric("Fundamental/Risk Layers", summary["Fundamental/Risk Layers"])
    with st.container(border=True):
        st.markdown("### Show exactly how uploaded data will be used")
        if layers:
            st.dataframe(selected_layer_table(layers), use_container_width=True, hide_index=True)
        required = required_data_for_layers(layers)
        if required:
            st.markdown("**Required data for full accuracy:**")
            for item in required:
                st.write(f"- {item}")
    return profile, layers

def uploaded_knowledge_brain_panel():
    st.subheader("Uploaded Knowledge Brain — complete integration registry")
    st.write(
        "This panel shows how the bot treats the uploaded PDFs, patterns, spreadsheets, ranking logic, "
        "and prior bot architecture as one knowledge system. The autonomous cycle uses these frameworks "
        "as decision gates, not as optional notes."
    )
    summary = knowledge_status_summary()
    a, b, c, d = st.columns(4)
    a.metric("Knowledge Sources", summary["Knowledge Sources"])
    b.metric("Integrated", summary["Integrated"])
    c.metric("Mapped", summary["Mapped"])
    d.metric("Active Engines", summary["Active Engines"])
    st.success(summary["Verdict"])

    st.markdown("#### Integration Registry")
    st.dataframe(knowledge_registry_table(), use_container_width=True, hide_index=True)

    st.markdown("#### Active Engine Coverage")
    st.dataframe(engine_coverage_table(), use_container_width=True, hide_index=True)

    st.info(
        "Document-intelligence architecture is mapped so the terminal stays evidence-first. "
        "When audited-report context or statement indexes are available locally, they can be bridged into this same terminal workflow."
    )

def autopilot_manager_panel():
    st.subheader("Autopilot Manager — one-click autonomous PSX research cycle")
    st.write(
        "This manager consolidates the uploaded trading-pattern, Fibonacci, psychology/risk, "
        "secular-trend, fundamentals, intrinsic-value, and red-flag frameworks into one disciplined cycle. "
        "It scans, triages, ranks, and produces portfolio/new-idea actions automatically; it does not place broker orders."
    )

    a1, a2, a3 = st.columns([1.25, 1, 1])
    symbols_text = a1.text_area(
        "Symbols to manage",
        value="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF",
        height=112,
        key="autopilot_symbols",
        help="Comma-separated PSX symbols. Use your portfolio names or watchlist names.",
    )
    max_symbols = a2.number_input("Max symbols this cycle", min_value=1, max_value=100, value=20, step=1, key="autopilot_max_symbols")
    persist_outputs = a3.checkbox("Save outputs to data/", value=True, key="autopilot_persist")
    auto_fetch_fundamentals = a3.checkbox(
        "Auto-find fundamentals online",
        value=True,
        key="autopilot_auto_fetch_fundamentals",
        help="Try public web sources for missing fundamentals before falling back to technical-basis decisions.",
    )

    b1, b2 = st.columns(2)
    portfolio_upload = b1.file_uploader(
        "Optional portfolio CSV / Excel",
        type=["csv", "xlsx", "xls"],
        key="autopilot_portfolio_upload",
        help="Preferred columns: Symbol, Quantity, Avg_Buy, MTM_Price. Flexible aliases are supported.",
    )
    fundamentals_upload = b2.file_uploader(
        "Optional fundamentals CSV / Excel",
        type=["csv", "xlsx", "xls"],
        key="autopilot_fundamentals_upload",
        help="Upload your ranking/fundamental dataset to enable Investogenie-style quality, valuation, and fakery gates.",
    )

    l1, l2 = st.columns(2)
    portfolio_drive_pdf_url = l1.text_input(
        "Portfolio Google Drive PDF link",
        value="",
        key="autopilot_portfolio_drive_pdf_url",
        placeholder="Paste Google Drive PDF link here",
        help="Share the PDF with link access. The bot will download it and try to extract the holdings table automatically.",
    )
    fundamentals_sheet_url = l2.text_input(
        "Fundamental Google Sheet link",
        value="",
        key="autopilot_fundamentals_sheet_url",
        placeholder="Paste Google Sheets link here",
        help="Share the sheet with link access. The bot converts the sheet tab to CSV and loads it into the fundamental engine.",
    )
    st.caption("Use upload or link. When a pasted link is provided, the bot uses the link source for that dataset.")
    if fundamentals_upload is None and not str(fundamentals_sheet_url or "").strip() and has_imported_fundamental_images():
        st.info("No fundamental CSV/Sheet link is selected, so Autopilot will use the latest Fundamental Image Import data as its fallback fundamentals source.")

    with st.container(border=True):
        st.markdown("### Autopilot discipline controls")
        c1, c2, c3, c4 = st.columns(4)
        entry_gate = c1.slider("Entry confluence gate", 40, 85, 60, 1, key="autopilot_entry_gate")
        strong_gate = c2.slider("Strong-entry gate", 50, 95, 70, 1, key="autopilot_strong_gate")
        prediction_horizon = c3.slider("Prediction horizon", 3, 15, 5, 1, key="autopilot_prediction_horizon")
        target_rr = c4.slider("Target R:R", 1.0, 6.0, 3.0, 0.5, key="autopilot_target_rr")
        d1, d2 = st.columns(2)
        stop_atr = d1.slider("Stop ATR", 0.75, 5.0, 2.0, 0.25, key="autopilot_stop_atr")
        investor_gate = d2.slider("Investor minimum fundamental score %", 0, 90, 55, 1, key="autopilot_investor_gate")

    run = st.button("Run Autonomous Management Cycle", type="primary", use_container_width=True, key="autopilot_run")
    if not run:
        st.info("Upload files or paste the Portfolio Google Drive PDF link and Fundamental Google Sheet link, then run the cycle. The bot will produce actions for holdings and fresh watchlist ideas in one pass.")
        return

    try:
        symbols = parse_symbols(symbols_text)
        portfolio_df = read_portfolio_source(portfolio_upload, portfolio_drive_pdf_url)
        fundamentals_df = read_fundamentals_source(fundamentals_upload, fundamentals_sheet_url)
        bundle = run_autopilot_cycle(
            symbols,
            portfolio_df=portfolio_df,
            fundamentals_df=fundamentals_df,
            max_symbols=int(max_symbols),
            prediction_horizon=int(prediction_horizon),
            stop_atr=float(stop_atr),
            target_rr=float(target_rr),
            discipline_rules={
                "minimum_confluence_for_entry": float(entry_gate),
                "minimum_confluence_for_strong_entry": float(strong_gate),
                "minimum_fundamental_score_for_investor": float(investor_gate),
            },
            auto_fetch_missing_fundamentals=bool(auto_fetch_fundamentals),
        )
        if persist_outputs:
            paths = persist_autopilot_outputs(bundle)
            st.success("Autopilot cycle complete and outputs saved locally in the data folder.")
            st.caption("Saved: " + " | ".join(f"{k}: {v.name}" for k, v in paths.items()))
        else:
            st.success("Autopilot cycle complete.")

        brief = bundle.get("market_brief", pd.DataFrame())
        if isinstance(brief, pd.DataFrame) and not brief.empty:
            row = brief.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Completed", f"{int(row.get('Symbols Completed', 0))}/{int(row.get('Symbols Requested', 0))}")
            m2.metric("Event Risk", row.get("News/Event Risk", "N/A"))
            m3.metric("Avg Confluence", row.get("Average Confluence", "N/A"))
            m4.metric("High Hazards", row.get("High Hazard Count", 0))
            m5.metric("Failures", row.get("Failures", 0))
            st.info(str(row.get("News/Event Notes", "No event-risk notes generated.")) or "No event-risk notes generated.")

        knowledge_summary = bundle.get("knowledge_summary", pd.DataFrame())
        if isinstance(knowledge_summary, pd.DataFrame) and not knowledge_summary.empty:
            k = knowledge_summary.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Knowledge Sources", int(k.get("Knowledge Sources", 0)))
            k2.metric("Integrated", int(k.get("Integrated", 0)))
            k3.metric("Mapped", int(k.get("Mapped", 0)))
            k4.metric("Active Engines", int(k.get("Active Engines", 0)))
            st.success(str(k.get("Verdict", "Knowledge registry active.")))

        decisions = bundle.get("decisions", pd.DataFrame())
        holdings = bundle.get("holdings", pd.DataFrame())
        opportunities = bundle.get("opportunities", pd.DataFrame())
        failures = bundle.get("failures", pd.DataFrame())

        preferred_cols = [
            "Symbol", "Portfolio Holding", "Autopilot Action", "Decision Label", "Confluence Score",
            "PRO Score", "Fundamental Score %", "Margin of Safety %", "Fraud Risk", "Hazard Level",
            "News/Event Risk", "Prediction Verdict", "Secular Bias", "Knowledge Applied", "Portfolio P&L %", "Decision Reason"
        ]
        visible_cols = [c for c in preferred_cols if isinstance(decisions, pd.DataFrame) and c in decisions.columns]

        st.markdown('<div class="dk-section-title">1. Master Autonomous Decisions</div>', unsafe_allow_html=True)
        st.markdown('<div class="dk-panel">', unsafe_allow_html=True)
        if isinstance(decisions, pd.DataFrame) and not decisions.empty:
            st.dataframe(decisions[visible_cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download autonomous decisions CSV",
                decisions.to_csv(index=False).encode("utf-8"),
                file_name="psx_autopilot_decisions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("No autonomous decisions were completed.")
        st.markdown('</div>', unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["Portfolio Management", "New Opportunities", "Failures / Diagnostics"])
        with t1:
            if isinstance(holdings, pd.DataFrame) and not holdings.empty:
                hc = [c for c in preferred_cols if c in holdings.columns]
                st.dataframe(holdings[hc], use_container_width=True, hide_index=True)
            else:
                st.info("No uploaded portfolio holdings matched the managed symbols.")
        with t2:
            if isinstance(opportunities, pd.DataFrame) and not opportunities.empty:
                oc = [c for c in preferred_cols if c in opportunities.columns]
                st.dataframe(opportunities[oc], use_container_width=True, hide_index=True)
            else:
                st.info("No non-portfolio opportunities completed in this cycle.")
        with t3:
            if isinstance(failures, pd.DataFrame) and not failures.empty:
                st.dataframe(failures, use_container_width=True, hide_index=True)
            else:
                st.success("No symbol-level failures in this cycle.")

    except Exception as exc:
        st.error(f"Autopilot cycle failed: {exc}")

def multi_style_trading_desk_panel():
    st.subheader("Multi-Style Trading Desk — Scalping, Intraday, Swing, and Long-Term")
    st.write(
        "This desk applies the uploaded CWT, market-stage, Dow Theory, reversal, continuation, "
        "candlestick, Fibonacci, FVG, RRMS, and news-risk logic to different trading horizons."
    )

    st.dataframe(profile_table(), use_container_width=True, hide_index=True)

    a1, a2, a3 = st.columns(3)
    trading_style = a1.selectbox(
        "Trading Style",
        list(TRADING_PROFILES.keys()),
        index=2,
        key="v40_style",
    )
    profile = get_profile(trading_style)
    symbol = a2.text_input("PSX Symbol", value="NBP", key="v40_style_symbol").strip().upper()
    source = a3.selectbox(
        "Data Source",
        ["Yahoo Finance PSX (.KA) — Recommended", "Experimental DPS Chart-Series Loader"],
        key="v40_style_source",
    )

    st.info(profile["description"] + " " + profile["data_warning"])

    b1, b2, b3, b4 = st.columns(4)
    analysis_tf = b1.selectbox(
        "Analysis TF",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
        index=(["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"].index(profile["analysis_tf"]) if profile["analysis_tf"] in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"] else 6),
        key="v40_style_atf",
    )
    execution_tf = b2.selectbox(
        "Execution TF",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
        index=(["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"].index(profile["execution_tf"]) if profile["execution_tf"] in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"] else 4),
        key="v40_style_etf",
    )
    period = b3.selectbox("History Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5, key="v40_style_period")
    dps_mode = b4.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v40_style_dps")

    c1, c2, c3, c4 = st.columns(4)
    event_risk_toggle = c1.checkbox("Apply event/news risk warning", value=True, key="v40_style_event")
    benchmark_conflict = c2.checkbox("KSE-100 / benchmark conflict", value=False, key="v40_style_benchmark")
    fetch_live_news = c3.checkbox("Fetch live official news/announcements", value=True, key="v40_style_fetch_news")
    use_prediction = c4.checkbox("Use prediction risk filter", value=True, key="v40_style_predict")

    with st.container(border=True):
        st.markdown("### Risk, RR, and Position Controls")
        r1, r2, r3, r4 = st.columns(4)
        account_balance = r1.number_input("Capital", value=100000.0, min_value=1.0, step=10000.0, key="v40_style_capital")
        risk_pct = r2.number_input("Risk %", value=float(profile["default_risk_pct"]), min_value=0.05, max_value=5.0, step=0.05, key="v40_style_risk")
        stop_atr = r3.number_input("Stop ATR", value=float(profile["stop_atr"]), min_value=0.5, max_value=5.0, step=0.25, key="v40_style_stop_atr")
        target_rr = r4.number_input("Target RR", value=float(profile["target_rr"]), min_value=1.0, max_value=6.0, step=0.5, key="v40_style_rr")

    run = st.button("Run Multi-Style Professional Analysis", type="primary", use_container_width=True, key="v40_style_run")
    if not run:
        return

    try:
        higher_df, lower_df = load_frames_for_style(symbol, source, analysis_tf, execution_tf, period, dps_mode)
        if len(higher_df) < 40 or len(lower_df) < 40:
            st.warning("Limited candle history for the selected style/timeframes. Conclusions may be less reliable.")

        result = analyze_symbol(
            symbol=symbol,
            higher_df=higher_df,
            lower_df=lower_df,
            asset_class="Stock",
            analysis_tf=analysis_tf,
            execution_tf=execution_tf,
            risk_context={
                "high_impact_news": event_risk_toggle,
                "benchmark_conflict": benchmark_conflict,
            },
        )
        pro = evaluate_psx_pro_score(result)
        risk_warning = build_risk_warning(
            result,
            pro,
            user_event_risk=event_risk_toggle,
            benchmark_conflict=benchmark_conflict,
        )
        fib = fibonacci_retracement_confluence(result["execution_frame"], result["higher_trend"]["trend"])
        secular = classify_secular_trend(result["higher_frame"] if "higher_frame" in result else higher_df)
        hazard = detect_price_hazards(
            result["execution_frame"],
            symbol=symbol,
            support_resistance=result.get("support_resistance", {}),
        )
        news = build_news_event_risk_snapshot([symbol]) if fetch_live_news else {"status": "DISABLED", "risk_level": "LOW", "risk_notes": [], "events": pd.DataFrame(), "errors": []}
        prediction = None
        if use_prediction:
            prediction = run_prediction_engine(
                result["execution_frame"],
                bias=result["signal"]["bias"],
                horizon=int(profile["prediction_horizon"]),
                stop_atr=float(stop_atr),
                target_rr=float(target_rr),
                risk_severity=risk_warning["risk_severity"],
            )
        confluence = build_confluence_score(
            result=result,
            pro=pro,
            prediction=prediction or {},
            fib=fib,
            secular=secular,
            hazard=hazard,
            news=news,
            trading_style=trading_style,
        )

        plan = result["trade_plan"]
        size = calc_position_size(
            account_balance=account_balance,
            risk_pct=risk_pct,
            entry=plan["entry"],
            stop_loss=plan["stop_loss"],
            contract_size=1.0,
        )

        h1, h2, h3, h4, h5, h6 = st.columns(6)
        h1.metric("Trading Style", trading_style)
        h2.metric("Bias", result["signal"]["bias"])
        h3.metric("MTF Scenario", result.get("mtf_scenario", result["scenario"])["number"] if result.get("mtf_scenario") else "")
        h4.metric("CWT Scenario", result.get("cwt_scenario", {}).get("number"))
        h5.metric("PRO Score", pro["pro_score"])
        h6.metric("Hazard Level", hazard.get("hazard_level"))

        show_confluence_box(confluence)

        st.subheader("News / Event Risk")
        if news.get("risk_level") == "HIGH":
            st.error("HIGH NEWS/EVENT RISK — " + " | ".join(news.get("risk_notes", [])))
        elif news.get("risk_level") == "MODERATE":
            st.warning("MODERATE NEWS/EVENT RISK — " + " | ".join(news.get("risk_notes", [])))
        else:
            st.success("No major official event-risk flag was generated in this run.")
        events_df = news.get("events")
        if isinstance(events_df, pd.DataFrame) and not events_df.empty:
            st.dataframe(events_df, use_container_width=True, hide_index=True)
        if news.get("errors"):
            st.caption("Some official sources could not be parsed: " + " | ".join(news["errors"]))

        st.subheader("Price-Hazard Early Warning")
        if hazard.get("hazard_level") in {"HIGH", "CRITICAL"}:
            st.error(f'{hazard.get("hazard_level")} HAZARD — {hazard.get("summary")}')
        elif hazard.get("hazard_level") == "MODERATE":
            st.warning(f'MODERATE HAZARD — {hazard.get("summary")}')
        else:
            st.success(hazard.get("summary", "No hazard summary."))
        st.dataframe(hazard_alerts_dataframe(hazard), use_container_width=True, hide_index=True)

        left, right = st.columns([1.15, 1])
        with left:
            render_chart_engine(
                result["execution_frame"],
                result,
                symbol=symbol,
                title=f"{symbol} — {trading_style} Chart",
                key_prefix=f"style_chart_{symbol}",
            )
            st.subheader("Fibonacci Retracement Confluence")
            if fib.get("status") == "OK":
                st.write(fib.get("message"))
                st.dataframe(pd.DataFrame(fib.get("levels", [])), use_container_width=True, hide_index=True)
            else:
                st.info(fib.get("message", fib.get("status")))
            st.subheader("Long-Term / Secular Trend Check")
            st.dataframe(pd.DataFrame([secular]), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Trade Plan & Loss Control")
            st.dataframe(pd.DataFrame([
                {"Field": "Order", "Value": plan["order_type"]},
                {"Field": "Entry", "Value": plan["entry"]},
                {"Field": "Stop Loss", "Value": plan["stop_loss"]},
                {"Field": "Take Profit", "Value": plan["take_profit"]},
                {"Field": "RR", "Value": plan["rr"]},
                {"Field": "Position Units", "Value": size["units"]},
                {"Field": "Risk Amount", "Value": size["risk_amount"]},
                {"Field": "Risk %", "Value": risk_pct},
            ]), use_container_width=True, hide_index=True)

            st.subheader("Prediction Filter")
            if prediction and prediction.get("status") in {"OK", "PARTIAL"}:
                p1, p2, p3 = st.columns(3)
                p1.metric("Verdict", prediction.get("prediction_verdict"))
                p2.metric("Expected Return %", prediction.get("expected_return_pct"))
                p3.metric("Prob. Up %", prediction.get("probability_up_pct"))
                st.dataframe(prediction_summary_table(prediction), use_container_width=True, hide_index=True)
            elif prediction:
                st.warning(prediction.get("message", "Prediction unavailable."))
            else:
                st.info("Prediction risk filter was disabled.")

            st.subheader("CWT Risk Warning")
            risk_text = f'{risk_warning["risk_severity"]} — {risk_warning["quick_action"]}'
            if risk_warning["risk_severity"] in {"HIGH", "CRITICAL"}:
                st.error(risk_text)
            elif risk_warning["risk_severity"] == "MODERATE":
                st.warning(risk_text)
            else:
                st.success(risk_text)
            if risk_warning["warnings"] or risk_warning["urgent_warnings"]:
                st.write(" | ".join(risk_warning["urgent_warnings"] + risk_warning["warnings"]))

        download_summary = pd.DataFrame([{
            "Symbol": symbol,
            "Trading Style": trading_style,
            "Analysis TF": analysis_tf,
            "Execution TF": execution_tf,
            "Bias": result["signal"]["bias"],
            "PRO Score": pro["pro_score"],
            "Confluence Score": confluence["confluence_score"],
            "Confluence Grade": confluence["confluence_grade"],
            "Hazard Level": hazard.get("hazard_level"),
            "News Risk": news.get("risk_level"),
            "Order": plan["order_type"],
            "Entry": plan["entry"],
            "Stop Loss": plan["stop_loss"],
            "Take Profit": plan["take_profit"],
        }])
        st.download_button(
            "Download Multi-Style Analysis Summary CSV",
            download_summary.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_{trading_style.lower().replace(' ', '_')}_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Multi-style trading desk failed: {exc}")

def single_symbol_panel():
    st.subheader("Single Symbol Professional Analysis")

    left_cfg, right_cfg = st.columns([1, 1])
    with left_cfg:
        source = st.selectbox(
            "Data Source",
            [
                "Yahoo Finance PSX (.KA) — Recommended",
                "PSX CSV Upload — Best for your own exact OHLCV",
                "Experimental DPS Chart-Series Loader",
            ],
            key="pro_single_source",
        )
        symbol = st.text_input("PSX Symbol", value="NBP", key="pro_single_symbol").strip().upper()
        period = st.selectbox("Yahoo History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4, key="pro_single_period")
        dps_mode = st.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="pro_single_dps_mode")
        csv_file = None
        if source.startswith("PSX CSV"):
            csv_file = st.file_uploader("Upload PSX OHLCV CSV", type=["csv"], key="pro_single_csv")

    with right_cfg:
        analysis_tf = st.selectbox("Higher / Analysis Timeframe", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="pro_single_analysis_tf")
        execution_tf = st.selectbox("Execution Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=0, key="pro_single_execution_tf")
        kse100_conflict = st.checkbox("KSE-100 daily trend conflicts with setup", value=False, key="pro_single_kse_conflict")
        event_risk = st.checkbox("Monetary policy / CPI / board-event risk", value=False, key="pro_single_event_risk")
        volume_shortlist = st.checkbox("Volume-leader shortlist confirmed", value=False, key="pro_single_volume")
        insider_review = st.checkbox("Insider transaction context reviewed", value=False, key="pro_single_insider")

    with st.container(border=True):
        st.markdown("### Risk & position sizing")
        r1, r2, r3 = st.columns(3)
        account_balance = r1.number_input("Capital / Account Value", value=100000.0, min_value=1.0, step=10000.0, key="pro_single_capital")
        risk_pct_override = r2.number_input("Risk % Override (0 = default)", value=0.0, min_value=0.0, step=0.05, key="pro_single_risk_override")
        shares_contract = r3.number_input("Contract Size (shares per unit)", value=1.0, min_value=0.000001, step=1.0, key="pro_single_contract")

    with st.container(border=True):
        st.markdown("### Prediction & loss-control settings")
        p1, p2, p3, p4 = st.columns(4)
        run_prediction = p1.checkbox("Run prediction engine", value=True, key="pro_single_run_prediction")
        prediction_horizon = p2.selectbox("Forecast horizon", [3, 5, 10, 15], index=1, key="pro_single_prediction_horizon")
        prediction_stop_atr = p3.number_input("Stop distance ATR", value=1.5, min_value=0.5, max_value=5.0, step=0.25, key="pro_single_prediction_stop_atr")
        prediction_target_rr = p4.number_input("Target RR", value=3.0, min_value=1.0, max_value=6.0, step=0.5, key="pro_single_prediction_target_rr")
        tracker_notes = st.text_input("Optional tracker note", value="", key="pro_single_tracker_notes")

    run = st.button("Run Professional Analysis", type="primary", use_container_width=True, key="pro_single_run")

    if run:
        try:
            if source.startswith("Yahoo Finance"):
                higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
                lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
                src_symbol = higher_df.attrs.get("source_symbol", symbol)
                st.info(f"Using Yahoo Finance PSX ticker: {src_symbol}")
            elif source.startswith("PSX CSV"):
                if csv_file is None:
                    raise ValueError("Upload a PSX CSV first.")
                base_df = load_psx_csv(csv_file)
                higher_df = resample_ohlcv(base_df, analysis_tf)
                lower_df = resample_ohlcv(base_df, execution_tf)
            else:
                base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
                source_warning = base_df.attrs.get("source_warning")
                if source_warning:
                    st.warning(source_warning)
                higher_df = resample_ohlcv(base_df, analysis_tf)
                lower_df = resample_ohlcv(base_df, execution_tf)

            result = analyze_symbol(
                symbol=symbol,
                higher_df=higher_df,
                lower_df=lower_df,
                asset_class="Stock",
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                risk_context={
                    "high_impact_news": event_risk,
                    "benchmark_conflict": kse100_conflict,
                },
            )
            pro = evaluate_psx_pro_score(result)
            risk_warning = build_risk_warning(
                result,
                pro,
                user_event_risk=event_risk,
                benchmark_conflict=kse100_conflict,
            )
            change_alerts = check_and_update_symbol_alerts(
                symbol=symbol,
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                result=result,
                pro=pro,
                risk=risk_warning,
            )

            prediction = None
            if run_prediction:
                prediction = run_prediction_engine(
                    result["execution_frame"],
                    bias=result["signal"]["bias"],
                    horizon=int(prediction_horizon),
                    stop_atr=float(prediction_stop_atr),
                    target_rr=float(prediction_target_rr),
                    risk_severity=risk_warning["risk_severity"],
                )

            shortlist_notes = []
            if volume_shortlist:
                shortlist_notes.append("Volume leader confirmation checked.")
            if insider_review:
                shortlist_notes.append("Insider context reviewed.")
            if shortlist_notes:
                st.info(" | ".join(shortlist_notes))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Bias", result["signal"]["bias"])
            c2.metric("Action", result["signal"]["action"])
            c3.metric("Setup", result["signal"]["setup_type"])
            c4.metric("MTF Scenario", result.get("mtf_scenario", result["scenario"])["label"])
            c5.metric("CWT Confidence", f'{result["signal"]["confidence"]}/100')

            overview_pro_metrics(pro)

            st.subheader("Risk Warning & Quick Action")
            risk_text = f'**{risk_warning["risk_severity"]} RISK** — {risk_warning["quick_action"]}'
            if risk_warning["risk_severity"] in {"CRITICAL", "HIGH"}:
                st.error(risk_text)
            elif risk_warning["risk_severity"] == "MODERATE":
                st.warning(risk_text)
            else:
                st.success(risk_text)

            if risk_warning["urgent_warnings"]:
                st.error("Urgent: " + " | ".join(risk_warning["urgent_warnings"]))
            if risk_warning["warnings"]:
                st.warning("Risk flags: " + " | ".join(risk_warning["warnings"]))

            st.subheader("Trend / Analysis Change Alerts")
            alerts_df = alerts_to_dataframe(change_alerts["alerts"])
            if not alerts_df.empty:
                urgent_count = int(alerts_df["Severity"].isin(["CRITICAL", "HIGH"]).sum()) if "Severity" in alerts_df.columns else 0
                if urgent_count > 0:
                    st.error(f"{urgent_count} high-priority change alert(s) detected versus the previous saved run.")
                else:
                    st.info("No high-priority change alert detected; see the alert table for details.")
                st.dataframe(alerts_df, use_container_width=True, hide_index=True)

            if result["warnings"]:
                st.warning("Core signal warnings: " + " | ".join(result["warnings"]))

            st.subheader("Prediction-Based Loss Control")
            if prediction is None:
                st.info("Prediction engine was disabled for this run.")
            elif prediction.get("status") in {"OK", "PARTIAL"}:
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Prediction Verdict", prediction.get("prediction_verdict"))
                p2.metric("Expected Return %", prediction.get("expected_return_pct"))
                p3.metric("Prob. Up %", prediction.get("probability_up_pct"))
                p4.metric("Prob. Down %", prediction.get("probability_down_pct"))

                if prediction.get("prediction_verdict") in {"PREDICTION CONFLICT", "LOSS-RISK ELEVATED"}:
                    st.error(prediction.get("loss_control_action"))
                elif prediction.get("prediction_verdict") == "PREDICTIVE SUPPORT":
                    st.success(prediction.get("loss_control_action"))
                else:
                    st.warning(prediction.get("loss_control_action"))

                st.dataframe(prediction_summary_table(prediction), use_container_width=True, hide_index=True)
            else:
                st.warning(prediction.get("message", "Prediction could not be generated."))

            render_chart_engine(
                result["execution_frame"],
                result,
                symbol=symbol,
                title=f"{symbol} — PRO Execution Chart",
                key_prefix=f"pro_chart_{symbol}",
            )

            left, right = st.columns([1.25, 1])
            with left:
                st.subheader("Detected Patterns")
                rows = []
                for group, items in result["patterns"].items():
                    for item in items:
                        rows.append({"Group": group, **item})
                st.dataframe(
                    pd.DataFrame(rows) if rows else pd.DataFrame([{"Result": "No named pattern detected."}]),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Technical Diagnostics")
                latest = pro["latest_metrics"]
                st.dataframe(pd.DataFrame([
                    {"Field": "MTF Scenario", "Value": result.get("mtf_scenario", result["scenario"])["label"]},
                    {"Field": "CWT Setup Scenario", "Value": result.get("cwt_scenario", {}).get("label")},
                    {"Field": "Higher Trend", "Value": result["higher_trend"]["trend"]},
                    {"Field": "Execution Trend", "Value": result["execution_trend"]["trend"]},
                    {"Field": "Market Phase", "Value": result["higher_trend"]["phase"]},
                    {"Field": "Trend Stack", "Value": pro["trend_stack"]},
                    {"Field": "Momentum", "Value": pro["momentum_state"]},
                    {"Field": "Volume State", "Value": pro["volume_state"]},
                    {"Field": "ADX State", "Value": pro["adx_state"]},
                    {"Field": "Volume Ratio", "Value": latest.get("volume_ratio")},
                    {"Field": "ATR %", "Value": latest.get("atr_pct")},
                    {"Field": "5-Period Return %", "Value": latest.get("return_5")},
                    {"Field": "Alligator", "Value": result["cwt"]["state"]},
                    {"Field": "Divergence", "Value": result["divergence"]["label"]},
                    {"Field": "Nearest Bullish FVG", "Value": result["fvg"]["nearest_bullish_zone"]},
                    {"Field": "Nearest Bearish FVG", "Value": result["fvg"]["nearest_bearish_zone"]},
                ]), use_container_width=True, hide_index=True)

                st.subheader("PRO Score Explanation")
                st.write(pro["technical_notes"])

            with right:
                st.subheader("Trade Plan")
                plan = result["trade_plan"]
                st.dataframe(pd.DataFrame([
                    {"Field": "Suggested Order", "Value": plan["order_type"]},
                    {"Field": "Entry", "Value": plan["entry"]},
                    {"Field": "Stop Loss", "Value": plan["stop_loss"]},
                    {"Field": "Take Profit 1:3", "Value": plan["take_profit"]},
                    {"Field": "RR", "Value": plan["rr"]},
                    {"Field": "Invalidation", "Value": plan["invalidation"]},
                    {"Field": "Rationale", "Value": plan["rationale"]},
                ]), use_container_width=True, hide_index=True)

                risk_pct = risk_pct_override if risk_pct_override > 0 else risk_profile_for_execution_tf(execution_tf)
                size = calc_position_size(
                    account_balance=account_balance,
                    risk_pct=risk_pct,
                    entry=plan["entry"],
                    stop_loss=plan["stop_loss"],
                    contract_size=shares_contract,
                )
                st.subheader("Position Sizing")
                st.dataframe(pd.DataFrame([
                    {"Field": "Risk %", "Value": f"{risk_pct:.2f}%"},
                    {"Field": "Risk Amount", "Value": size["risk_amount"]},
                    {"Field": "Price Risk / Share", "Value": size["risk_per_unit"]},
                    {"Field": "Shares / Units", "Value": size["units"]},
                ]), use_container_width=True, hide_index=True)

                if st.button("Add This Plan to Trade Tracker", key="pro_single_add_tracker", use_container_width=True):
                    trade_id = append_trade_plan(
                        symbol=symbol,
                        analysis_tf=analysis_tf,
                        execution_tf=execution_tf,
                        result=result,
                        pro=pro,
                        risk=risk_warning,
                        prediction=prediction or {},
                        position_size=size["units"],
                        risk_amount=size["risk_amount"],
                        notes=tracker_notes,
                    )
                    st.success(f"Trade plan added to tracker with ID: {trade_id}")

                summary_df = pd.DataFrame([{
                    "Symbol": symbol,
                    "Bias": result["signal"]["bias"],
                    "MTF Scenario": result.get("mtf_scenario", result["scenario"])["label"],
                    "CWT Setup Scenario": result.get("cwt_scenario", {}).get("label"),
                    "CWT Confidence": result["signal"]["confidence"],
                    "PRO Score": pro["pro_score"],
                    "Grade": pro["pro_grade"],
                    "Trade Quality": pro["trade_quality"],
                    "Order": plan["order_type"],
                    "Entry": plan["entry"],
                    "SL": plan["stop_loss"],
                    "TP": plan["take_profit"],
                }])
                st.download_button(
                    "Download Symbol Summary CSV",
                    summary_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{symbol}_psx_pro_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"PSX PRO analysis failed: {exc}")


def scenario_scanner_panel():
    st.subheader("Scenario Scanner — Fixed Scenario 1 / 2 / 3 Logic")
    st.caption("Now supports **All Timeframes**. Select All Timeframes in Analysis TF or Execution TF to scan scenarios across multiple charts and show exact timeframe in results.")
    st.info(
        "The uploaded course files use TWO different Scenario 1/2/3 systems. "
        "This scanner now keeps them separate: "
        "(1) Multi-Timeframe Trade Scenarios from Week 5/6, and "
        "(2) CWT Setup Scenarios from Week 2/3/4."
    )

    knowledge_profile, knowledge_layers = knowledge_layer_selector_ui(
        "scenario_scanner",
        default_profile="Scenario Scanner Focus",
    )

    s1, s2 = st.columns([1.4, 1])
    scenario_system = s1.selectbox(
        "Scenario System",
        ["MTF Trade Scenario (Week 5/6)", "CWT Setup Scenario (Week 2/3/4)"],
        index=0,
        key="v31_scenario_system",
    )
    selected_scenario = s2.radio("Scenario to find", ["Scenario 1", "Scenario 2", "Scenario 3"], key="v20_scenario")

    if scenario_system == "MTF Trade Scenario (Week 5/6)":
        st.markdown(
            "**MTF definitions:** Scenario 1 = HTF and execution TF aligned; "
            "Scenario 2 = HTF trend with opposite-direction pullback on execution TF; "
            "Scenario 3 = HTF trend with sideways execution TF."
        )
    else:
        st.markdown(
            "**CWT definitions:** Scenario 1 = trend continuation with open Alligator; "
            "Scenario 2 = jawline reversal; "
            "Scenario 3 = sleeping/closed Alligator."
        )

    c1, c2, c3 = st.columns(3)
    universe = c2.selectbox(
        "Universe",
        ["KSE-100 Constituents", "Eligible Scrips — All PSX Symbols", "Custom Symbols"],
        index=0,
        key="v20_scenario_universe",
    )
    max_symbols = c3.number_input("Max symbols (0 = all)", value=0, min_value=0, step=25, key="v20_scenario_max")

    custom_symbols_text = ""
    if universe == "Custom Symbols":
        custom_symbols_text = st.text_area(
            "Custom symbols",
            value="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF",
            key="v20_scenario_custom",
        )

    d1, d2, d3, d4 = st.columns(4)
    data_source = d1.selectbox(
        "Scanner Data Source",
        ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"],
        index=0,
        key="v20_scenario_source",
    )
    analysis_tf = d2.selectbox("Analysis TF", ANALYSIS_TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_scenario_atf")
    execution_tf = d3.selectbox("Execution TF", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_scenario_etf")
    period = d4.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="v20_scenario_period")

    e1, e2, e3 = st.columns(3)
    dps_mode = e1.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v20_scenario_dps")
    kse_conflict = e2.checkbox("Apply KSE-100 conflict warning", value=False, key="v20_scenario_conflict")
    event_risk = e3.checkbox("Apply macro/board-event warning", value=False, key="v20_scenario_event")

    run = st.button("Scan Selected Scenario", type="primary", use_container_width=True, key="v20_scenario_run")
    if run:
        progress, status, update = scan_progress_ui()
        try:
            matched_frames = []
            all_frames = []
            failures = []
            analysis_tfs = _selected_analysis_timeframes(analysis_tf)
            execution_tfs = _selected_execution_timeframes(execution_tf)
            total_jobs = len(analysis_tfs) * len(execution_tfs)
            job_no = 0

            for atf in analysis_tfs:
                for etf in execution_tfs:
                    job_no += 1
                    progress.progress(0, text=f"Scenario timeframe scan {job_no}/{total_jobs}: Analysis {atf} / Execution {etf}")
                    frame_matched, frame_all, frame_failures = scan_psx_for_scenario(
                        selected_scenario=selected_scenario,
                        scenario_system=scenario_system,
                        universe=universe,
                        custom_symbols_text=custom_symbols_text,
                        data_source=data_source,
                        analysis_tf=atf,
                        execution_tf=etf,
                        period=period,
                        dps_mode=dps_mode,
                        risk_context={
                            "high_impact_news": event_risk or ("News / Event Risk Filter" in knowledge_layers),
                            "benchmark_conflict": kse_conflict,
                            "knowledge_profile": knowledge_profile,
                            "knowledge_layers": knowledge_layers,
                        },
                        max_symbols=int(max_symbols),
                        progress_callback=update,
                    )

                    if isinstance(frame_matched, pd.DataFrame) and not frame_matched.empty:
                        frame_matched = frame_matched.copy()
                        frame_matched.insert(0, "Analysis TF", atf)
                        frame_matched.insert(1, "Execution TF", etf)
                        matched_frames.append(frame_matched)

                    if isinstance(frame_all, pd.DataFrame) and not frame_all.empty:
                        frame_all = frame_all.copy()
                        frame_all.insert(0, "Analysis TF", atf)
                        frame_all.insert(1, "Execution TF", etf)
                        all_frames.append(frame_all)

                    if frame_failures:
                        for f in frame_failures:
                            ff = str(f)
                            failures.append(f"{atf}/{etf}: {ff}")

            matched_df = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()
            all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

            if not matched_df.empty:
                dedup_cols = [c for c in ["Symbol", "Analysis TF", "Execution TF", "Scenario", "MTF Scenario", "CWT Scenario"] if c in matched_df.columns]
                if dedup_cols:
                    matched_df = matched_df.drop_duplicates(subset=dedup_cols, keep="first")
            if not all_df.empty:
                dedup_cols_all = [c for c in ["Symbol", "Analysis TF", "Execution TF"] if c in all_df.columns]
                if dedup_cols_all:
                    all_df = all_df.drop_duplicates(subset=dedup_cols_all, keep="first")

            matched_df = apply_knowledge_columns(matched_df, knowledge_profile, knowledge_layers, "Scenario Scanner")
            all_df = apply_knowledge_columns(all_df, knowledge_profile, knowledge_layers, "Scenario Scanner")
            progress.progress(100, text="Scenario scan complete.")
            status.empty()

            m1, m2, m3 = st.columns(3)
            m1.metric("Matched", len(matched_df))
            m2.metric("Analyzed", len(all_df))
            m3.metric("Unavailable", len(failures))

            st.subheader(f"{scenario_system} • {selected_scenario} Matches — Ranked by PRO Score / Timeframe")
            if matched_df.empty:
                st.info("No symbols matched this scenario.")
            else:
                min_score = st.slider("Filter minimum PRO Score", 0, 100, 0, key="v20_scenario_min_score")
                filtered = matched_df[matched_df["Pro Score"].fillna(0) >= min_score] if "Pro Score" in matched_df.columns else matched_df
                st.dataframe(filtered, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Scenario Matches CSV",
                    filtered.to_csv(index=False).encode("utf-8"),
                    file_name=f"psx_{selected_scenario.lower().replace(' ', '_')}_timeframe_matches.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.container(border=True):
                st.markdown("### Show all analyzed symbols")
                st.dataframe(all_df, use_container_width=True, hide_index=True) if not all_df.empty else st.write("No successful analyses.")
            with st.container(border=True):
                st.markdown("### Show failed/unavailable symbols")
                st.write("\n".join(failures) if failures else "No symbol-level failures.")
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Scenario scanner failed: {exc}")



TIMEFRAME_SELECTOR_OPTIONS = ["All Timeframes", "1d", "4h", "1h", "30m", "15m"]
ANALYSIS_TIMEFRAME_SELECTOR_OPTIONS = ["All Timeframes", "1d", "1wk", "1mo"]
ALL_SCAN_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

def _selected_execution_timeframes(selection: str) -> list[str]:
    if str(selection).strip() == "All Timeframes":
        return ALL_SCAN_TIMEFRAMES.copy()
    return [str(selection).strip()]


def _selected_analysis_timeframes(selection: str) -> list[str]:
    if str(selection).strip() == "All Timeframes":
        return ["1h", "4h", "1d", "1wk", "1mo"]
    return [str(selection).strip()]


def _scan_divergence_for_symbols(symbols: list[str], *, data_source: str, analysis_tf: str, execution_tf: str, period: str, dps_mode: str, divergence_type: str, max_symbols: int = 0, sensitivity: str = "Sensitive", lookback: int = 220, max_gap: int = 120, max_matches_per_symbol: int = 2):
    rows = []
    failures = []
    scan_symbols = symbols[:max_symbols] if max_symbols and max_symbols > 0 else symbols
    total = len(scan_symbols)
    progress = st.progress(0, text="Preparing divergence scan...")
    status = st.empty()

    for i, symbol in enumerate(scan_symbols, start=1):
        symbol = str(symbol).strip().upper()
        if not symbol:
            continue
        progress.progress(int(i / max(total, 1) * 100), text=f"Scanning divergence {i}/{total}: {symbol}")
        status.caption(f"Current symbol: {symbol}")

        try:
            selected_tfs = _fast_scan_timeframes(scan_mode, execution_tf)
            higher_cache = None

            for tf in selected_tfs:
                try:
                    if data_source == "Yahoo Finance PSX (.KA)":
                        if higher_cache is None:
                            higher_cache = load_psx_yahoo_ohlcv(symbol, period=period, interval=analysis_tf)
                        higher = higher_cache
                        lower = _cached_yahoo_ohlcv(symbol, period=period, interval=tf)
                    else:
                        higher = _cached_dps_ohlcv(symbol, mode=dps_mode)
                        lower = higher.copy()

                    result = analyze_symbol(
                        symbol=symbol,
                        higher_df=higher,
                        lower_df=lower,
                        asset_class="PSX",
                        analysis_tf=analysis_tf,
                        execution_tf=tf,
                    )
                    exec_frame = result.get("execution_frame", lower) if isinstance(result, dict) else lower
                    divergences = _all_divergence_points(
                        exec_frame,
                        mode=divergence_type,
                        sensitivity=sensitivity,
                        lookback=int(lookback),
                        max_gap=int(max_gap),
                    )[: int(max_matches_per_symbol)]

                    if divergences:
                        trade_plan = result.get("trade_plan", {}) if isinstance(result, dict) else {}
                        signal = result.get("signal", {}) if isinstance(result, dict) else {}
                        latest_close = None
                        try:
                            latest_close = float(result.get("execution_frame", lower)["close"].iloc[-1])
                        except Exception:
                            pass

                        for divergence in divergences:
                            p1, p2 = divergence.get("price_points", [{}, {}])
                            r1, r2 = divergence.get("rsi_points", [{}, {}])
                            rows.append({
                                "Symbol": symbol,
                                "Timeframe": tf,
                                "Divergence": divergence.get("label", "None"),
                                "Bias": divergence.get("bias", "Neutral"),
                                "Kind": divergence.get("kind", ""),
                                "Score": divergence.get("score", 0),
                                "Price Pivot 1": round(float(p1.get("price", 0)), 4) if p1 else None,
                                "Price Pivot 2": round(float(p2.get("price", 0)), 4) if p2 else None,
                                "RSI Pivot 1": round(float(r1.get("price", 0)), 2) if r1 else None,
                                "RSI Pivot 2": round(float(r2.get("price", 0)), 2) if r2 else None,
                                "Detail": divergence.get("detail", ""),
                                "Signal": signal.get("action", "WATCH"),
                                "Setup": signal.get("setup_type", ""),
                                "Confidence": signal.get("confidence", 0),
                                "Last Price": latest_close,
                                "Entry": trade_plan.get("entry"),
                                "Stop Loss": trade_plan.get("stop_loss"),
                                "TP1": trade_plan.get("take_profit"),
                                "_result": result,
                            })
                except Exception as tf_exc:
                    failures.append({"Symbol": symbol, "Timeframe": tf, "Error": str(tf_exc)})
        except Exception as exc:
            failures.append({"Symbol": symbol, "Timeframe": execution_tf, "Error": str(exc)})

    progress.empty()
    status.empty()
    return rows, failures



def _read_first_symbol_column(csv_path: Path) -> list[str]:
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Prefer symbol-like columns, otherwise first column.
            preferred = [c for c in df.columns if str(c).strip().lower() in {"symbol", "symbols", "scrip", "ticker", "code"}]
            col = preferred[0] if preferred else df.columns[0]
            return df[col].dropna().astype(str).str.replace(".KA", "", regex=False).str.upper().str.strip().unique().tolist()
    except Exception:
        return []
    return []


def _builtin_kse100_symbols() -> list[str]:
    # Fallback list to ensure KSE-100 option does not silently scan only a watchlist.
    return parse_symbols("""
        ABOT, ACPL, ADMM, AGP, AICL, AIRLINK, AKBL, APL, ATRL, AVN, BAFL,
        BAHL, BOP, BWCL, CEPB, CHCC, CNERGY, COLG, DAWH, DGKC, EFERT,
        ENGROH, EPCL, FABL, FATIMA, FCCL, FCEPL, FFC, FFBL, FML, FTRM,
        GAL, GHGL, GHNI, GLAXO, HBL, HUBC, ILP, INDU, ISL, JDWS, JVDC,
        KAPCO, KEL, KOHC, LUCK, MARI, MCB, MEBL, MLCF, MTL, NATF, NBP,
        NCL, NESTLE, NML, OGDC, PABC, PAEL, PAKT, PGLC, PIBTL, PIOC,
        POL, POML, PPL, PRL, PSO, PTC, RMPL, SCBPL, SEARL, SHEL, SHFA,
        SNGP, SYS, THCCL, TRG, UBL, UNITY, YOUW, ZIL
    """)


def _builtin_kmi30_symbols() -> list[str]:
    # Practical fallback list; CSV files override this when available.
    return parse_symbols("""
        OGDC, PPL, POL, MARI, HUBC, FFC, EFERT, FATIMA, FFBL, LUCK,
        MLCF, FCCL, DGKC, KOHC, CHCC, MEBL, BAHL, BAFL, AKBL, NBP,
        PSO, ATRL, PRL, CNERGY, SYS, TRG, AVN, ILP, INDU, PABC
    """)


def _resolve_latest_divergence_symbols(universe_mode: str, custom_symbols_text: str = "") -> list[str]:
    """Resolve symbols for Latest Divergence Scanner.

    Selected Symbols = pasted list only.
    KSE-100 Constituents = full KSE-100 list if file exists, otherwise built-in fallback.
    KMI-30 Constituents = full KMI-30 list if file exists, otherwise built-in fallback.
    All PSX Listed Symbols = full listed-symbol file if available, otherwise largest available watchlist.
    """
    mode = str(universe_mode or "").strip()
    if mode in {"Selected Symbols", "Custom symbols", "Custom Symbols"}:
        return parse_symbols(custom_symbols_text)

    if mode == "KSE-100 Constituents":
        for candidate in [
            Path("watchlists/kse100.csv"),
            Path("watchlists/kse_100.csv"),
            Path("watchlists/kse-100.csv"),
            Path("watchlists/psx_kse100.csv"),
            Path("watchlists/KSE100.csv"),
            Path("data/kse100.csv"),
            Path("data/kse_100.csv"),
        ]:
            symbols = _read_first_symbol_column(candidate)
            if len(symbols) >= 50:
                return symbols
        return _builtin_kse100_symbols()

    if mode == "KMI-30 Constituents":
        for candidate in [
            Path("watchlists/kmi30.csv"),
            Path("watchlists/kmi_30.csv"),
            Path("watchlists/kmi-30.csv"),
            Path("watchlists/psx_kmi30.csv"),
            Path("watchlists/KMI30.csv"),
            Path("data/kmi30.csv"),
            Path("data/kmi_30.csv"),
        ]:
            symbols = _read_first_symbol_column(candidate)
            if len(symbols) >= 20:
                return symbols
        return _builtin_kmi30_symbols()

    if mode in {"All PSX Symbols", "All PSX Listed Symbols", "Eligible Scrips — All PSX Symbols"}:
        for candidate in [
            Path("watchlists/all_psx_symbols.csv"),
            Path("watchlists/psx_all_symbols.csv"),
            Path("watchlists/psx_listed_symbols.csv"),
            Path("watchlists/eligible_scrips.csv"),
            Path("data/all_psx_symbols.csv"),
            Path("data/psx_all_symbols.csv"),
            Path("data/psx_listed_symbols.csv"),
            Path("watchlists/psx_watchlist.csv"),
        ]:
            symbols = _read_first_symbol_column(candidate)
            if len(symbols) >= 100:
                return symbols
        # Last fallback: combine known index lists.
        return sorted(set(_builtin_kse100_symbols() + _builtin_kmi30_symbols()))

    return parse_symbols(custom_symbols_text) or ["NBP", "SYS", "MARI", "FFC", "SAZEW", "PSX"]




@st.cache_data(show_spinner=False, ttl=1800)
def _cached_yahoo_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return load_psx_yahoo_ohlcv(symbol, period=period, interval=interval)


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_dps_ohlcv(symbol: str, mode: str) -> pd.DataFrame:
    return load_psx_dps_ohlcv(symbol, mode=mode)


def _fast_scan_timeframes(mode: str, selected_tf: str) -> list[str]:
    if selected_tf != "All Timeframes":
        return [selected_tf]
    mode = str(mode or "Balanced").lower()
    if mode == "fast":
        return ["1d"]
    if mode == "balanced":
        return ["1d", "4h"]
    if mode == "deep":
        return ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]
    return ["1d"]


def _scan_latest_divergence_for_symbols(symbols: list[str], *, data_source: str, analysis_tf: str, execution_tf: str, period: str, dps_mode: str, divergence_type: str, max_symbols: int = 0, sensitivity: str = "Sensitive", lookback: int = 220, max_gap: int = 120, latest_candles: int = 0, result_mode: str = "Latest divergence in lookback", scan_mode: str = "Fast", return_scope: str = "Latest/best per symbol"):
    """Find the most recent divergence per symbol across one or all timeframes."""
    rows, failures = [], []
    scan_symbols = symbols[:max_symbols] if max_symbols and max_symbols > 0 else symbols
    total = len(scan_symbols)
    progress = st.progress(0, text="Preparing latest divergence scan...")
    status = st.empty()
    for i, symbol in enumerate(scan_symbols, start=1):
        symbol = str(symbol).strip().upper()
        if not symbol:
            continue
        progress.progress(int(i / max(total, 1) * 100), text=f"Latest divergence scan {i}/{total}: {symbol}")
        status.caption(f"Current symbol: {symbol}")
        try:
            selected_tfs = _fast_scan_timeframes(scan_mode, execution_tf)
            higher_cache = {}
            best_matches = []
            for tf in selected_tfs:
                try:
                    if data_source == "Yahoo Finance PSX (.KA)":
                        if analysis_tf not in higher_cache:
                            higher_cache[analysis_tf] = _cached_yahoo_ohlcv(symbol, period=period, interval=analysis_tf)
                        higher = higher_cache[analysis_tf]
                        lower = _cached_yahoo_ohlcv(symbol, period=period, interval=tf)
                    else:
                        higher = _cached_dps_ohlcv(symbol, mode=dps_mode)
                        lower = higher.copy()
                    result = analyze_symbol(symbol=symbol, higher_df=higher, lower_df=lower, asset_class="PSX", analysis_tf=analysis_tf, execution_tf=tf)
                    exec_frame = result.get("execution_frame", lower) if isinstance(result, dict) else lower
                    divergences = _all_divergence_points(exec_frame, mode=divergence_type, sensitivity=sensitivity, lookback=int(lookback), max_gap=int(max_gap))
                    if divergences:
                        last_pos = len(exec_frame) - 1
                        for d in divergences:
                            try:
                                second_pos = int(d.get("price_points", [{}, {}])[1].get("pos", -9999))
                                candles_ago = int(last_pos - second_pos)
                            except Exception:
                                candles_ago = 9999
                            # latest_candles = 0 means no last-candle restriction.
                            if int(latest_candles) <= 0 or candles_ago <= int(latest_candles):
                                d = dict(d)
                                d["_candles_ago"] = candles_ago
                                d["_timeframe"] = tf
                                d["_result"] = result
                                d["_exec_frame"] = exec_frame
                                best_matches.append(d)
                except Exception as tf_exc:
                    failures.append({"Symbol": symbol, "Timeframe": tf, "Error": str(tf_exc)})
            if best_matches:
                if str(result_mode) == "Strongest divergence in lookback":
                    best_matches.sort(key=lambda x: (int(x.get("score", 0)), -int(x.get("_candles_ago", 9999))), reverse=True)
                else:
                    best_matches.sort(key=lambda x: (int(x.get("_candles_ago", 9999)), -int(x.get("score", 0))))

                matches_to_output = best_matches if str(return_scope) == "All divergences found" else best_matches[:1]
                for best in matches_to_output:
                    result = best.get("_result", {})
                    trade_plan = result.get("trade_plan", {}) if isinstance(result, dict) else {}
                    signal = result.get("signal", {}) if isinstance(result, dict) else {}
                    exec_frame = best.get("_exec_frame")
                    latest_close = None
                    try: latest_close = float(exec_frame["close"].iloc[-1])
                    except Exception: pass
                    p1, p2 = best.get("price_points", [{}, {}])
                    r1, r2 = best.get("rsi_points", [{}, {}])
                    rows.append({
                        "Symbol": symbol, "Latest Timeframe": best.get("_timeframe", ""), "Divergence": best.get("label", "None"),
                        "Bias": best.get("bias", "Neutral"), "Kind": best.get("kind", ""), "Score": best.get("score", 0),
                        "Candles Ago": best.get("_candles_ago", ""),
                        "Pivot Gap": best.get("pivot_gap", ""),
                        "Price Change %": best.get("price_change_pct", ""),
                        "RSI Change": best.get("rsi_change", ""),
                        "Price Pivot 1": round(float(p1.get("price", 0)), 4) if p1 else None,
                        "Price Pivot 2": round(float(p2.get("price", 0)), 4) if p2 else None,
                        "RSI Pivot 1": round(float(r1.get("price", 0)), 2) if r1 else None,
                        "RSI Pivot 2": round(float(r2.get("price", 0)), 2) if r2 else None,
                        "Detail": best.get("detail", ""), "Signal": signal.get("action", "WATCH"), "Setup": signal.get("setup_type", ""),
                        "Confidence": signal.get("confidence", 0), "Last Price": latest_close,
                        "Entry": trade_plan.get("entry"), "Stop Loss": trade_plan.get("stop_loss"), "TP1": trade_plan.get("take_profit"),
                        "_result": result,
                    })
        except Exception as exc:
            failures.append({"Symbol": symbol, "Timeframe": execution_tf, "Error": str(exc)})
    progress.empty(); status.empty()
    return rows, failures

def divergence_finder_panel():
    st.subheader("Divergence Finder")
    st.caption(
        "Scan PSX symbols for regular/hidden bullish or bearish RSI divergence. Select one timeframe or All Timeframes to scan 1D, 4H, 1H, 30M, and 15M together."
    )

    st.markdown("## 🔎 Latest Divergence Scanner — Find Latest Diversions")
    st.warning("For fast All PSX scan, use **Scan Speed Mode — Fast / Balanced / Deep**. Fast = Daily only, Balanced = Daily + 4H, Deep = Daily + 4H + 1H.")
    st.caption("Find the **latest or all divergences** from charts for Selected Symbols, full KSE-100, full KMI-30, or All PSX Listed Symbols.")

    l1, l2, l3, l4 = st.columns(4)
    latest_universe = l1.selectbox("Latest scan universe", ["Selected Symbols", "KSE-100 Constituents", "KMI-30 Constituents", "All PSX Listed Symbols"], index=0, key="latestdiv_universe")
    latest_tf = l2.selectbox("Latest scan timeframe", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="latestdiv_tf")
    latest_div_type = l3.selectbox("Latest divergence type", ["Any Divergence", "Regular Bullish RSI Divergence", "Regular Bearish RSI Divergence", "Hidden Bullish RSI Divergence", "Hidden Bearish RSI Divergence", "Any Bullish Divergence", "Any Bearish Divergence"], index=0, key="latestdiv_type")
    latest_scan_mode = l4.selectbox(
        "Scan Speed Mode — Fast / Balanced / Deep",
        ["Fast", "Balanced", "Deep"],
        index=0,
        key="latestdiv_speed_mode",
        help="Fast = Daily only. Balanced = Daily + 4H. Deep = Daily + 4H + 1H. Avoid 30M/15M for all PSX.",
    )

    x1, x2 = st.columns([1, 3])
    latest_max = x1.number_input("Max symbols", min_value=0, value=75, step=25, key="latestdiv_max_symbols")
    x2.info("For All PSX, use Fast mode first. Then scan Balanced/Deep only on shortlisted symbols.")

    y1, y2 = st.columns([1, 3])
    latest_result_scope = y1.selectbox(
        "Return results",
        ["Latest/best per symbol", "All divergences found"],
        index=0,
        key="latestdiv_result_scope",
        help="Use All divergences found when you want every divergence from KSE-100, KMI-30, or All PSX symbols.",
    )
    y2.info("KSE-100 scans the KSE-100 list; KMI-30 scans KMI-30; All PSX Listed Symbols scans the listed-symbol file if available.")

    latest_symbols_text = ""
    if latest_universe == "Selected Symbols":
        latest_symbols_text = st.text_area("Selected symbols for latest divergence scan", value="NBP, SYS, MARI, FFC, SAZEW, PSX", key="latestdiv_symbols")

    if st.checkbox("Use Recommended Fast All PSX Settings", value=False, key="latestdiv_fast_preset_help"):
        st.info("Recommended: Latest scan universe = All PSX Symbols, Latest scan timeframe = All Timeframes, Scan Speed Mode = Fast, Type = Any Divergence, Sensitivity = Sensitive, Limit by recent candles = OFF.")

    m1, m2, m3, m4 = st.columns(4)
    latest_data_source = m1.selectbox("Latest data source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="latestdiv_data_source")
    latest_analysis_tf = m2.selectbox("Latest analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="latestdiv_analysis_tf")
    latest_period = m3.selectbox("Latest Yahoo period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="latestdiv_period")
    latest_dps_mode = m4.selectbox("Latest DPS mode", ["daily", "intraday"], index=0, key="latestdiv_dps_mode")

    n1, n2, n3, n4 = st.columns(4)
    latest_sensitivity = n1.selectbox("Latest sensitivity", ["Sensitive", "Normal", "Strict"], index=0, key="latestdiv_sensitivity")
    latest_lookback = n2.number_input("Lookback candles", min_value=40, max_value=500, value=240, step=20, key="latestdiv_lookback")
    latest_gap = n3.number_input("Max pivot gap", min_value=10, max_value=220, value=140, step=10, key="latestdiv_gap")
    latest_result_mode = n4.selectbox(
        "Result mode",
        ["Latest divergence in lookback", "Strongest divergence in lookback"],
        index=0,
        key="latestdiv_result_mode",
        help="Latest = most recent divergence in the selected lookback. Strongest = highest score in the selected lookback.",
    )

    o1, o2 = st.columns([1, 3])
    limit_recent = o1.checkbox(
        "Limit by recent candles",
        value=False,
        key="latestdiv_limit_recent",
        help="Keep this off if it is not necessary for divergence to be inside last 35 candles.",
    )
    latest_candles = 0
    if limit_recent:
        latest_candles = o2.number_input("Maximum candles ago", min_value=5, max_value=200, value=35, step=5, key="latestdiv_recent_candles")
    else:
        o2.info("No last-candle restriction. Bot will find the latest divergence anywhere inside the selected lookback candles.")

    estimated_symbols = int(latest_max) if int(latest_max) > 0 else len(_resolve_latest_divergence_symbols(latest_universe, latest_symbols_text))
    selected_tf_count = len(_fast_scan_timeframes(latest_scan_mode, latest_tf))
    st.caption(f"Estimated chart requests: about {estimated_symbols * selected_tf_count}. Scanning: {', '.join(_fast_scan_timeframes(latest_scan_mode, latest_tf))}.")

    if st.button("Find Latest Divergences", type="primary", use_container_width=True, key="latestdiv_run"):
        latest_symbols = _resolve_latest_divergence_symbols(latest_universe, latest_symbols_text)
        rows, failures = _scan_latest_divergence_for_symbols(latest_symbols, data_source=latest_data_source, analysis_tf=latest_analysis_tf, execution_tf=latest_tf, period=latest_period, dps_mode=latest_dps_mode, divergence_type=latest_div_type, max_symbols=int(latest_max), sensitivity=latest_sensitivity, lookback=int(latest_lookback), max_gap=int(latest_gap), latest_candles=int(latest_candles), result_mode=latest_result_mode, scan_mode=latest_scan_mode, return_scope=latest_result_scope)
        st.session_state["latest_divergence_rows"] = rows
        st.session_state["latest_divergence_failures"] = failures

    latest_rows = st.session_state.get("latest_divergence_rows", [])
    latest_failures = st.session_state.get("latest_divergence_failures", [])
    if latest_rows:
        latest_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_result"} for r in latest_rows])
        st.success(f"Found {len(latest_df)} latest divergence stock(s).")
        st.dataframe(latest_df, use_container_width=True, hide_index=True)
        st.download_button("Download Latest Divergence Results", data=latest_df.to_csv(index=False).encode("utf-8"), file_name="psx_latest_divergence_results.csv", mime="text/csv", use_container_width=True, key="latestdiv_download")
        labels = [f"{r.get('Symbol')} | {r.get('Latest Timeframe', '')} | {r.get('Kind', r.get('Divergence', ''))} | {r.get('Candles Ago', '')} candles ago" for r in latest_rows]
        selected_label = st.selectbox("Open latest divergence chart", labels, key="latestdiv_chart_select")
        selected_idx = labels.index(selected_label) if selected_label in labels else 0
        selected_row = latest_rows[selected_idx]
        result = selected_row.get("_result", {})
        chart_df = result.get("execution_frame") if isinstance(result, dict) else None
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            st.info("This chart opens the latest divergence result. Keep **Show divergence on chart + RSI** enabled.")
            render_chart_engine(chart_df, result, symbol=selected_row.get("Symbol", ""), title=f"{selected_row.get('Symbol')} Latest Divergence | {selected_row.get('Latest Timeframe', '')}", key_prefix=f"latestdiv_{selected_idx}_{selected_row.get('Symbol', '')}")
    else:
        st.info("Recommended fast setting: Universe = All PSX or KSE-100, Timeframe = All Timeframes, Speed mode = Fast, Type = Any Divergence, Sensitivity = Sensitive, Limit by recent candles = OFF. Then run Balanced only on shortlisted symbols.")
    if latest_failures:
        with st.container(border=True):
            st.markdown("### Latest divergence failures / missing data (len(latest_failures))")
            st.dataframe(pd.DataFrame(latest_failures), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Full Divergence Finder")
    st.info("Speed Mode is available above in **Latest Divergence Scanner**. Use it for All PSX fast scan.")


    p1, p2, p3, p4 = st.columns(4)
    universe_mode = p1.selectbox(
        "Universe",
        ["Watchlist file", "Custom symbols"],
        index=0,
        key="divfinder_universe_mode",
    )
    divergence_type = p2.selectbox(
        "Divergence Type",
        ["Any Divergence", "Regular Bullish RSI Divergence", "Regular Bearish RSI Divergence", "Hidden Bullish RSI Divergence", "Hidden Bearish RSI Divergence", "Any Bullish Divergence", "Any Bearish Divergence"],
        index=0,
        key="divfinder_type",
    )
    data_source = p3.selectbox(
        "Data Source",
        ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"],
        key="divfinder_data_source",
    )
    max_symbols = p4.number_input(
        "Max symbols (0 = all)",
        min_value=0,
        value=100,
        step=25,
        key="divfinder_max_symbols",
    )

    q1, q2, q3, q4 = st.columns(4)
    analysis_tf = q1.selectbox("Analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="divfinder_analysis_tf")
    execution_tf = q2.selectbox("Execution TF", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="divfinder_execution_tf")
    period = q3.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="divfinder_period")
    dps_mode = q4.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="divfinder_dps_mode")

    s1, s2, s3, s4 = st.columns(4)
    sensitivity = s1.selectbox("Finder sensitivity", ["Sensitive", "Normal", "Strict"], index=0, key="divfinder_sensitivity")
    lookback = s2.number_input("Lookback candles", min_value=40, max_value=500, value=220, step=20, key="divfinder_lookback")
    max_gap = s3.number_input("Max pivot gap", min_value=10, max_value=200, value=120, step=10, key="divfinder_max_gap")
    max_matches_per_symbol = s4.number_input("Max matches / symbol", min_value=1, max_value=5, value=2, step=1, key="divfinder_max_matches")

    if universe_mode == "Custom symbols":
        raw_symbols = st.text_area(
            "Paste symbols separated by comma or new line",
            value="NBP, SYS, MARI, FFC, SAZEW, PSX",
            key="divfinder_custom_symbols",
        )
        symbols = parse_symbols(raw_symbols)
    else:
        try:
            watchlist_path = Path("watchlists/psx_watchlist.csv")
            if watchlist_path.exists():
                wl = pd.read_csv(watchlist_path)
                first_col = wl.columns[0]
                symbols = wl[first_col].dropna().astype(str).str.upper().tolist()
            else:
                symbols = ["NBP", "SYS", "MARI", "FFC", "SAZEW", "PSX"]
        except Exception:
            symbols = ["NBP", "SYS", "MARI", "FFC", "SAZEW", "PSX"]

    st.caption(f"Symbols prepared for scan: {len(symbols)}")

    run = st.button("Run Divergence Finder", type="primary", use_container_width=True, key="divfinder_run")
    if run:
        rows, failures = _scan_divergence_for_symbols(
            symbols,
            data_source=data_source,
            analysis_tf=analysis_tf,
            execution_tf=execution_tf,
            period=period,
            dps_mode=dps_mode,
            divergence_type=divergence_type,
            max_symbols=int(max_symbols),
            sensitivity=sensitivity,
            lookback=int(lookback),
            max_gap=int(max_gap),
            max_matches_per_symbol=int(max_matches_per_symbol),
        )

        st.session_state["divergence_finder_rows"] = rows
        st.session_state["divergence_finder_failures"] = failures

    rows = st.session_state.get("divergence_finder_rows", [])
    failures = st.session_state.get("divergence_finder_failures", [])

    if rows:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "_result"} for r in rows])
        st.success(f"Found {len(df)} divergence match(es).")
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Divergence Finder Results",
            data=csv,
            file_name="psx_divergence_finder_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="divfinder_download",
        )

        result_labels = [
            f"{r.get('Symbol')} | {r.get('Timeframe', '')} | {r.get('Kind', r.get('Divergence', ''))}"
            for r in rows
        ]
        selected_label = st.selectbox(
            "Open divergence chart",
            result_labels,
            key="divfinder_selected_symbol",
        )
        selected_idx = result_labels.index(selected_label) if selected_label in result_labels else 0
        selected_row = rows[selected_idx] if rows else None
        selected_symbol = selected_row.get("Symbol") if selected_row else ""
        if selected_row and isinstance(selected_row.get("_result"), dict):
            result = selected_row["_result"]
            chart_df = result.get("execution_frame")
            if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
                st.info("The opened chart includes divergence overlay controls. Keep **Show divergence on chart + RSI** enabled to see the same divergence lines on price and RSI.")
                render_chart_engine(
                    chart_df,
                    result,
                    symbol=selected_symbol,
                    title=f"{selected_symbol} Divergence Finder Chart | {selected_row.get('Timeframe', '')}",
                    key_prefix=f"divfinder_{selected_symbol}",
                )
            else:
                st.warning("Chart data is not available for the selected divergence result.")
    else:
        st.info("Run the scan to find divergence. Recommended first test: Any Divergence + Sensitive + 220 lookback candles.")

    if failures:
        with st.container(border=True):
            st.markdown("### Scan failures / missing data (len(failures))")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)



def mtf_scenario_divergence_scanner_panel():
    st.subheader("Scenario Finder — All Timeframe Divergence Mode")
    st.caption(
        "Select one timeframe or All Timeframes. The scanner will check divergence and scenario context across selected charts."
    )

    a1, a2, a3, a4 = st.columns(4)
    universe_mode = a1.selectbox("Scenario Universe", ["Watchlist file", "Custom symbols"], index=0, key="mtfscenario_universe")
    execution_tf = a2.selectbox("Scenario Timeframe", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="mtfscenario_tf")
    divergence_type = a3.selectbox(
        "Divergence Filter",
        ["Any Divergence", "Regular Bullish RSI Divergence", "Regular Bearish RSI Divergence", "Hidden Bullish RSI Divergence", "Hidden Bearish RSI Divergence", "Any Bullish Divergence", "Any Bearish Divergence"],
        index=0,
        key="mtfscenario_divtype",
    )
    max_symbols = a4.number_input("Max symbols", min_value=0, value=100, step=25, key="mtfscenario_max_symbols")

    b1, b2, b3, b4 = st.columns(4)
    data_source = b1.selectbox("Data Source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="mtfscenario_data_source")
    analysis_tf = b2.selectbox("Analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="mtfscenario_analysis_tf")
    period = b3.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="mtfscenario_period")
    sensitivity = b4.selectbox("Divergence Sensitivity", ["Sensitive", "Normal", "Strict"], index=0, key="mtfscenario_sensitivity")

    c1, c2, c3 = st.columns(3)
    lookback = c1.number_input("Lookback candles", min_value=40, max_value=500, value=220, step=20, key="mtfscenario_lookback")
    max_gap = c2.number_input("Max pivot gap", min_value=10, max_value=200, value=120, step=10, key="mtfscenario_max_gap")
    dps_mode = c3.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="mtfscenario_dps")

    if universe_mode == "Custom symbols":
        raw_symbols = st.text_area("Scenario custom symbols", value="NBP, SYS, MARI, FFC, SAZEW, PSX", key="mtfscenario_custom_symbols")
        symbols = parse_symbols(raw_symbols)
    else:
        try:
            watchlist_path = Path("watchlists/psx_watchlist.csv")
            if watchlist_path.exists():
                wl = pd.read_csv(watchlist_path)
                symbols = wl[wl.columns[0]].dropna().astype(str).str.upper().tolist()
            else:
                symbols = ["NBP", "SYS", "MARI", "FFC", "SAZEW", "PSX"]
        except Exception:
            symbols = ["NBP", "SYS", "MARI", "FFC", "SAZEW", "PSX"]

    st.caption(f"Scenario symbols prepared: {len(symbols)}")

    if st.button("Run Scenario Finder Across Timeframes", type="primary", use_container_width=True, key="mtfscenario_run"):
        rows, failures = _scan_divergence_for_symbols(
            symbols,
            data_source=data_source,
            analysis_tf=analysis_tf,
            execution_tf=execution_tf,
            period=period,
            dps_mode=dps_mode,
            divergence_type=divergence_type,
            max_symbols=int(max_symbols),
            sensitivity=sensitivity,
            lookback=int(lookback),
            max_gap=int(max_gap),
            max_matches_per_symbol=2,
        )

        enhanced = []
        for r in rows:
            rr = dict(r)
            res = rr.get("_result", {})
            if isinstance(res, dict):
                rr["Scenario"] = (res.get("scenario") or {}).get("label", "")
                rr["MTF Scenario"] = (res.get("mtf_scenario") or {}).get("label", "")
                rr["Higher Trend"] = (res.get("higher_trend") or {}).get("trend", "")
                rr["Execution Trend"] = (res.get("execution_trend") or {}).get("trend", "")
            enhanced.append(rr)
        st.session_state["mtfscenario_rows"] = enhanced
        st.session_state["mtfscenario_failures"] = failures

    rows = st.session_state.get("mtfscenario_rows", [])
    failures = st.session_state.get("mtfscenario_failures", [])

    if rows:
        display_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_result"} for r in rows])
        st.success(f"Scenario Finder found {len(display_df)} timeframe divergence match(es).")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Scenario Timeframe Divergence Results",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="psx_scenario_timeframe_divergence_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="mtfscenario_download",
        )

        labels = [
            f"{r.get('Symbol')} | {r.get('Timeframe', '')} | {r.get('Kind', r.get('Divergence', ''))} | {r.get('Scenario', '')}"
            for r in rows
        ]
        selected_label = st.selectbox("Open scenario divergence chart", labels, key="mtfscenario_chart_select")
        selected_idx = labels.index(selected_label) if selected_label in labels else 0
        selected_row = rows[selected_idx]
        result = selected_row.get("_result", {})
        chart_df = result.get("execution_frame") if isinstance(result, dict) else None
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            render_chart_engine(
                chart_df,
                result,
                symbol=selected_row.get("Symbol", ""),
                title=f"{selected_row.get('Symbol')} Scenario Divergence | {selected_row.get('Timeframe', '')}",
                key_prefix=f"mtfscenario_{selected_idx}_{selected_row.get('Symbol', '')}",
            )
    else:
        st.info("Run the Scenario Finder MTF scan. Recommended: All Timeframes + Any Divergence + Sensitive.")

    if failures:
        with st.container(border=True):
            st.markdown("### Scenario scan failures / missing data (len(failures))")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)



def pattern_scanner_panel():
    st.subheader("Pattern & Divergence Scanner")
    st.caption("Now supports **All Timeframes**. Select All Timeframes in Analysis TF or Execution TF to scan multiple charts and see exact timeframe in results.")

    knowledge_profile, knowledge_layers = knowledge_layer_selector_ui(
        "pattern_scanner",
        default_profile="Pattern Scanner Focus",
    )

    p1, p2, p3 = st.columns(3)
    target = p1.selectbox(
        "What to find",
        [
            "Any Reversal Pattern",
            "Any Continuation Pattern",
            "Rising Wedge",
            "Falling Wedge",
            "Bullish Flag",
            "Bearish Flag",
            "Pennant",
            "Ascending Triangle",
            "Descending Triangle",
            "Symmetrical Triangle",
            "Rectangle / Range Consolidation",
            "Double Top",
            "Double Bottom",
            "Triple Top",
            "Triple Bottom",
            "Head and Shoulders",
            "Inverse Head and Shoulders",
            "Bullish RSI Divergence",
            "Bearish RSI Divergence",
            "Any Candlestick Pattern",
            "High-Quality Bullish Setups",
            "High-Quality Bearish Setups",
        ],
        key="v20_pattern_target",
    )
    universe = p2.selectbox(
        "Universe",
        ["KSE-100 Constituents", "Eligible Scrips — All PSX Symbols", "Custom Symbols"],
        index=0,
        key="v20_pattern_universe",
    )
    max_symbols = p3.number_input("Max symbols (0 = all)", value=0, min_value=0, step=25, key="v20_pattern_max")

    custom_symbols_text = ""
    if universe == "Custom Symbols":
        custom_symbols_text = st.text_area(
            "Custom symbols",
            value="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF",
            key="v20_pattern_custom",
        )

    q1, q2, q3, q4 = st.columns(4)
    data_source = q1.selectbox("Data Source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="v20_pattern_source")
    analysis_tf = q2.selectbox("Analysis TF", ANALYSIS_TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_pattern_atf")
    execution_tf = q3.selectbox("Execution TF", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_pattern_etf")
    period = q4.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="v20_pattern_period")

    r1, r2, r3 = st.columns(3)
    dps_mode = r1.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v20_pattern_dps")
    min_score = r2.slider("Minimum PRO Score", 0, 100, 55, key="v20_pattern_min_score")
    event_risk = r3.checkbox("Apply event-risk warning", value=False, key="v20_pattern_event")

    run = st.button("Run Pattern Scanner", type="primary", use_container_width=True, key="v20_pattern_run")
    if run:
        progress, status, update = scan_progress_ui()
        try:
            all_frames = []
            all_failures = []
            analysis_tfs = _selected_analysis_timeframes(analysis_tf)
            execution_tfs = _selected_execution_timeframes(execution_tf)
            total_jobs = len(analysis_tfs) * len(execution_tfs)
            job_no = 0

            for atf in analysis_tfs:
                for etf in execution_tfs:
                    job_no += 1
                    progress.progress(0, text=f"Starting timeframe scan {job_no}/{total_jobs}: {atf} / {etf}")
                    frame_df, frame_failures = scan_patterns(
                        universe=universe,
                        custom_symbols_text=custom_symbols_text,
                        data_source=data_source,
                        analysis_tf=atf,
                        execution_tf=etf,
                        period=period,
                        dps_mode=dps_mode,
                        target=target,
                        min_pro_score=float(min_score),
                        max_symbols=int(max_symbols),
                        risk_context={
                            "high_impact_news": event_risk or ("News / Event Risk Filter" in knowledge_layers),
                            "knowledge_profile": knowledge_profile,
                            "knowledge_layers": knowledge_layers,
                        },
                        progress_callback=update,
                    )
                    if isinstance(frame_df, pd.DataFrame) and not frame_df.empty:
                        frame_df = frame_df.copy()
                        frame_df.insert(0, "Analysis TF", atf)
                        frame_df.insert(1, "Execution TF", etf)
                        all_frames.append(frame_df)
                    if frame_failures:
                        for f in frame_failures:
                            ff = dict(f) if isinstance(f, dict) else {"Error": str(f)}
                            ff["Analysis TF"] = atf
                            ff["Execution TF"] = etf
                            all_failures.append(ff)

            df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
            failures = all_failures
            if not df.empty:
                dedup_cols = [c for c in ["Symbol", "Analysis TF", "Execution TF", "Divergence", "Pattern", "Setup"] if c in df.columns]
                if dedup_cols:
                    df = df.drop_duplicates(subset=dedup_cols, keep="first")
            df = apply_knowledge_columns(df, knowledge_profile, knowledge_layers, "Pattern & Divergence Scanner")
            progress.progress(100, text="Pattern scan complete.")
            status.empty()

            c1, c2 = st.columns(2)
            c1.metric("Matches", len(df))
            c2.metric("Unavailable", len(failures))

            if df.empty:
                st.info("No matches found for the selected pattern/divergence condition.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Pattern Scan CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="psx_pattern_divergence_scan.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.container(border=True):
                st.markdown("### Show failed/unavailable symbols")
                st.write("\n".join(failures) if failures else "No symbol-level failures.")
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Pattern scanner failed: {exc}")


def watchlist_scorecard_panel():
    st.subheader("Watchlist PRO Scorecard")
    symbols_text = st.text_area(
        "Watchlist Symbols",
        value="ATLH,ATRL,DCR,EFERT,FFC,INDU,KOHC,MARI,NATF,NBP,OGDC,POL,SYS,UBL,SAZEW",
        height=100,
        key="v20_watchlist_symbols",
    )
    symbols = parse_symbols(symbols_text)

    c1, c2, c3, c4 = st.columns(4)
    data_source = c1.selectbox("Data Source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="v20_watch_source")
    analysis_tf = c2.selectbox("Analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="v20_watch_atf")
    execution_tf = c3.selectbox("Execution TF", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=0, key="v20_watch_etf")
    period = c4.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="v20_watch_period")

    d1, d2 = st.columns(2)
    dps_mode = d1.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v20_watch_dps")
    event_risk = d2.checkbox("Apply event-risk warning", value=False, key="v20_watch_event")

    run = st.button("Run Watchlist PRO Scorecard", type="primary", use_container_width=True, key="v20_watch_run")
    if run:
        if not symbols:
            st.error("Enter at least one watchlist symbol.")
            return
        progress, status, update = scan_progress_ui()
        try:
            df, failures, watch_alerts_df = scan_watchlist_pro(
                symbols=symbols,
                data_source=data_source,
                analysis_tf=analysis_tf,
                execution_tf=execution_tf,
                period=period,
                dps_mode=dps_mode,
                risk_context={"high_impact_news": event_risk},
                progress_callback=update,
            )
            progress.progress(100, text="Watchlist scan complete.")
            status.empty()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Symbols", len(symbols))
            c2.metric("Analyzed", len(df))
            c3.metric("A/A+ Grades", int(df["Grade"].isin(["A", "A+"]).sum()) if not df.empty else 0)
            c4.metric("Unavailable", len(failures))

            if df.empty:
                st.info("No watchlist results returned.")
            else:
                high_risk_count = int(df["Risk Alert"].isin(["HIGH", "CRITICAL"]).sum()) if "Risk Alert" in df.columns else 0
                if high_risk_count > 0:
                    st.error(f"{high_risk_count} watchlist symbol(s) currently carry HIGH/CRITICAL risk alerts.")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Watchlist Scorecard CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="psx_watchlist_pro_scorecard.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.subheader("Watchlist Trend / Analysis Change Alerts")
            if watch_alerts_df.empty:
                st.info("No alert table was produced.")
            else:
                high_alerts = int(watch_alerts_df["Severity"].isin(["CRITICAL", "HIGH"]).sum()) if "Severity" in watch_alerts_df.columns else 0
                if high_alerts > 0:
                    st.error(f"{high_alerts} high-priority trend/analysis change alert(s) detected in the watchlist scan.")
                else:
                    st.info("No high-priority watchlist change alerts detected.")
                st.dataframe(watch_alerts_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Watchlist Change Alerts CSV",
                    watch_alerts_df.to_csv(index=False).encode("utf-8"),
                    file_name="psx_watchlist_change_alerts.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.container(border=True):
                st.markdown("### Show failed/unavailable symbols")
                st.write("\n".join(failures) if failures else "No symbol-level failures.")
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Watchlist scorecard failed: {exc}")




def prediction_loss_control_panel():
    st.subheader("Prediction & Loss Control Lab")
    st.write(
        "This module combines the technical setup with symbol-specific time-series forecasts. "
        "It estimates directional probability, stop-risk probability, target-hit probability, and an expected-return estimate."
    )

    a1, a2, a3 = st.columns(3)
    symbol = a1.text_input("PSX Symbol", value="NBP", key="v30_pred_symbol").strip().upper()
    source = a2.selectbox(
        "Data Source",
        ["Yahoo Finance PSX (.KA) — Recommended", "Experimental DPS Chart-Series Loader"],
        key="v30_pred_source",
    )
    period = a3.selectbox("Yahoo History Period", ["6mo", "1y", "2y", "5y"], index=3, key="v30_pred_period")

    b1, b2, b3, b4 = st.columns(4)
    analysis_tf = b1.selectbox("Analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="v30_pred_atf")
    execution_tf = b2.selectbox("Execution TF", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=0, key="v30_pred_etf")
    horizon = b3.selectbox("Forecast Horizon", [3, 5, 10, 15], index=1, key="v30_pred_horizon")
    dps_mode = b4.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v30_pred_dps")

    c1, c2, c3, c4 = st.columns(4)
    stop_atr = c1.number_input("Stop ATR", value=1.5, min_value=0.5, max_value=5.0, step=0.25, key="v30_pred_stop_atr")
    target_rr = c2.number_input("Target RR", value=3.0, min_value=1.0, max_value=6.0, step=0.5, key="v30_pred_rr")
    event_risk = c3.checkbox("Event-risk warning", value=False, key="v30_pred_event_risk")
    kse_conflict = c4.checkbox("KSE-100 conflict warning", value=False, key="v30_pred_kse_conflict")

    run = st.button("Run Prediction & Loss-Control Analysis", type="primary", use_container_width=True, key="v30_pred_run")
    if not run:
        return

    try:
        if source.startswith("Yahoo Finance"):
            higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
            lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
        else:
            base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
            higher_df = resample_ohlcv(base_df, analysis_tf)
            lower_df = resample_ohlcv(base_df, execution_tf)

        result = analyze_symbol(
            symbol=symbol,
            higher_df=higher_df,
            lower_df=lower_df,
            asset_class="Stock",
            analysis_tf=analysis_tf,
            execution_tf=execution_tf,
            risk_context={
                "high_impact_news": event_risk,
                "benchmark_conflict": kse_conflict,
            },
        )
        pro = evaluate_psx_pro_score(result)
        risk_warning = build_risk_warning(
            result,
            pro,
            user_event_risk=event_risk,
            benchmark_conflict=kse_conflict,
        )
        prediction = run_prediction_engine(
            result["execution_frame"],
            bias=result["signal"]["bias"],
            horizon=int(horizon),
            stop_atr=float(stop_atr),
            target_rr=float(target_rr),
            risk_severity=risk_warning["risk_severity"],
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Bias", result["signal"]["bias"])
        c2.metric("Scenario", result["scenario"]["label"])
        c3.metric("PRO Score", pro["pro_score"])
        c4.metric("Risk Alert", risk_warning["risk_severity"])

        st.subheader("Forecast Verdict")
        if prediction.get("status") in {"OK", "PARTIAL"}:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Verdict", prediction.get("prediction_verdict"))
            v2.metric("Expected Return %", prediction.get("expected_return_pct"))
            v3.metric("Probability Up %", prediction.get("probability_up_pct"))
            v4.metric("Probability Down %", prediction.get("probability_down_pct"))

            if prediction.get("prediction_verdict") in {"PREDICTION CONFLICT", "LOSS-RISK ELEVATED"}:
                st.error(prediction.get("loss_control_action"))
            elif prediction.get("prediction_verdict") == "PREDICTIVE SUPPORT":
                st.success(prediction.get("loss_control_action"))
            else:
                st.warning(prediction.get("loss_control_action"))

            left, right = st.columns([1.1, 1])
            with left:
                st.dataframe(prediction_summary_table(prediction), use_container_width=True, hide_index=True)
            with right:
                st.subheader("ATR-Based Forecast Zones")
                st.dataframe(pd.DataFrame([
                    {"Zone": "Current Close", "Value": prediction.get("latest_close")},
                    {"Zone": "Bullish Target Zone", "Value": prediction.get("forecast_up_zone")},
                    {"Zone": "Bearish Target Zone", "Value": prediction.get("forecast_down_zone")},
                    {"Zone": "ATR Long Stop Zone", "Value": prediction.get("atr_long_stop_zone")},
                    {"Zone": "ATR Short Stop Zone", "Value": prediction.get("atr_short_stop_zone")},
                ]), use_container_width=True, hide_index=True)

            st.subheader("Risk Warnings")
            if risk_warning["risk_severity"] in {"HIGH", "CRITICAL"}:
                st.error(f'{risk_warning["risk_severity"]} RISK — {risk_warning["quick_action"]}')
            elif risk_warning["risk_severity"] == "MODERATE":
                st.warning(f'{risk_warning["risk_severity"]} RISK — {risk_warning["quick_action"]}')
            else:
                st.success(f'{risk_warning["risk_severity"]} RISK — {risk_warning["quick_action"]}')
            if risk_warning["warnings"] or risk_warning["urgent_warnings"]:
                st.write(" | ".join(risk_warning["urgent_warnings"] + risk_warning["warnings"]))

            st.download_button(
                "Download Prediction Summary CSV",
                prediction_summary_table(prediction).to_csv(index=False).encode("utf-8"),
                file_name=f"{symbol}_prediction_loss_control.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning(prediction.get("message", "Prediction engine could not generate a forecast."))

    except Exception as exc:
        st.error(f"Prediction module failed: {exc}")


def trade_tracker_panel():
    st.subheader("Trade Tracker & Loss-Control Journal")
    st.write(
        "Use this tracker to keep planned setups, paper trades, or real trades organized. "
        "Trade plans added from Single Symbol PRO appear here with their risk and prediction context."
    )

    tracker = load_trade_tracker()
    summary = tracker_summary(tracker)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Plans", summary["total"])
    c2.metric("Planned", summary["planned"])
    c3.metric("Open", summary["open"])
    c4.metric("Closed", summary["closed"])
    c5.metric("Win Rate %", summary["win_rate_pct"])
    c6.metric("Realized P&L", summary["realized_pnl"])

    if tracker.empty:
        st.info("Tracker is empty. Add a trade plan from the Single Symbol PRO tab.")
    else:
        status_filter = st.multiselect(
            "Filter tracker statuses",
            ["PLANNED", "OPEN", "TP HIT", "SL HIT", "CLOSED", "CANCELLED"],
            default=["PLANNED", "OPEN", "TP HIT", "SL HIT", "CLOSED", "CANCELLED"],
            key="v30_tracker_status_filter",
        )
        filtered = tracker.copy()
        if status_filter:
            filtered = filtered[filtered["status"].fillna("").astype(str).str.upper().isin(status_filter)]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Trade Tracker CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="psx_trade_tracker.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("Update a Tracked Trade")
        ids = tracker["trade_id"].dropna().astype(str).tolist()
        if ids:
            u1, u2, u3 = st.columns(3)
            trade_id = u1.selectbox("Trade ID", ids, key="v30_tracker_trade_id")
            status = u2.selectbox(
                "New Status",
                ["PLANNED", "OPEN", "TP HIT", "SL HIT", "CLOSED", "CANCELLED"],
                key="v30_tracker_new_status",
            )
            selected = tracker[tracker["trade_id"].astype(str) == str(trade_id)].iloc[0]
            entry_default = float(selected["entry_filled_price"]) if pd.notna(selected["entry_filled_price"]) else 0.0
            exit_default = float(selected["exit_price"]) if pd.notna(selected["exit_price"]) else 0.0
            entry_filled = u3.number_input("Entry Filled Price", value=entry_default, min_value=0.0, step=0.01, key="v30_tracker_entry_filled")

            v1, v2 = st.columns(2)
            exit_price = v1.number_input("Exit Price", value=exit_default, min_value=0.0, step=0.01, key="v30_tracker_exit_price")
            notes = v2.text_input("Update Notes", value="", key="v30_tracker_notes")

            if st.button("Update Tracked Trade", type="primary", use_container_width=True, key="v30_tracker_update"):
                update_trade_row(
                    trade_id=trade_id,
                    status=status,
                    entry_filled_price=entry_filled if entry_filled > 0 else None,
                    exit_price=exit_price if exit_price > 0 else None,
                    notes=notes if notes else None,
                )
                st.success("Trade tracker updated. Refresh or rerun this tab to view the latest table.")



def news_and_price_hazard_watchtower_panel():
    st.subheader("News & Price-Hazard Watchtower")
    st.write(
        "This watchtower checks official PSX announcements, PSX financial announcements, "
        "PSX notices, SBP monetary-policy calendar, PBS macro releases, and technical price-hazard conditions. "
        "It is designed to warn before trading into obvious event or volatility risk."
    )

    a1, a2, a3 = st.columns(3)
    symbols_text = a1.text_area(
        "Symbols to monitor",
        value="NBP,OGDC,MARI,SYS,UBL,SAZEW,NATF",
        height=110,
        key="v40_watchtower_symbols",
    )
    symbols = parse_symbols(symbols_text)
    source = a2.selectbox(
        "OHLCV Data Source",
        ["Yahoo Finance PSX (.KA) — Recommended", "Experimental DPS Chart-Series Loader"],
        key="v40_watchtower_source",
    )
    period = a3.selectbox("History Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="v40_watchtower_period")

    b1, b2, b3, b4 = st.columns(4)
    analysis_tf = b1.selectbox("Analysis TF", ["1h", "4h", "1d", "1wk", "1mo"], index=0, key="v40_watchtower_atf")
    execution_tf = b2.selectbox("Execution TF", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=0, key="v40_watchtower_etf")
    dps_mode = b3.selectbox("DPS Mode", ["daily", "intraday"], index=0, key="v40_watchtower_dps")
    max_symbols = b4.number_input("Max symbols to price-scan", value=20, min_value=1, max_value=200, step=5, key="v40_watchtower_max")

    run_news = st.button("Run Official News/Event Watch", type="primary", use_container_width=True, key="v40_watchtower_news_run")
    if run_news:
        try:
            news = build_news_event_risk_snapshot(symbols=symbols)
            st.subheader("Official News / Event Risk Snapshot")
            if news.get("risk_level") == "HIGH":
                st.error("HIGH NEWS/EVENT RISK — " + " | ".join(news.get("risk_notes", [])))
            elif news.get("risk_level") == "MODERATE":
                st.warning("MODERATE NEWS/EVENT RISK — " + " | ".join(news.get("risk_notes", [])))
            else:
                st.success("No major official news/event flag was generated in this run.")
            events = news.get("events")
            if isinstance(events, pd.DataFrame) and not events.empty:
                st.dataframe(events, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download News/Event Watch CSV",
                    events.to_csv(index=False).encode("utf-8"),
                    file_name="psx_official_news_event_watch.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No structured official-event rows were parsed for the selected symbols.")
            if news.get("errors"):
                with st.container(border=True):
                    st.markdown("### Source parsing notes")
                    st.write("\n".join(news["errors"]))
        except Exception as exc:
            st.error(f"Official news/event watch failed: {exc}")

    st.divider()
    run_hazard = st.button("Run Price-Hazard Early Warning Scan", type="primary", use_container_width=True, key="v40_watchtower_hazard_run")
    if run_hazard:
        if not symbols:
            st.error("Enter at least one symbol.")
            return

        progress, status, update = scan_progress_ui()
        rows = []
        alert_rows = []
        failures = []
        selected = symbols[: int(max_symbols)]
        total = len(selected)
        for i, symbol in enumerate(selected, start=1):
            update(i, total, symbol)
            try:
                higher_df, lower_df = load_frames_for_style(symbol, source, analysis_tf, execution_tf, period, dps_mode)
                result = analyze_symbol(
                    symbol=symbol,
                    higher_df=higher_df,
                    lower_df=lower_df,
                    asset_class="Stock",
                    analysis_tf=analysis_tf,
                    execution_tf=execution_tf,
                    risk_context={},
                )
                hazard = detect_price_hazards(
                    result["execution_frame"],
                    symbol=symbol,
                    support_resistance=result.get("support_resistance", {}),
                )
                rows.append({
                    "Symbol": symbol,
                    "Hazard Level": hazard.get("hazard_level"),
                    "Hazard Points": hazard.get("hazard_points"),
                    "Gap %": hazard.get("latest_gap_pct"),
                    "Range / ATR": hazard.get("latest_range_atr_ratio"),
                    "Volume Ratio": hazard.get("latest_volume_ratio"),
                    "Drawdown %": hazard.get("latest_drawdown_pct"),
                    "Candle Return %": hazard.get("latest_candle_return_pct"),
                    "Quick Action": hazard.get("summary"),
                })
                for alert in hazard.get("alerts", []):
                    alert_rows.append({"Symbol": symbol, **alert})
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")

        progress.progress(100, text="Price-hazard scan complete.")
        status.empty()

        hazard_df = pd.DataFrame(rows)
        alerts_df = pd.DataFrame(alert_rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Scanned", len(hazard_df))
        c2.metric("HIGH / CRITICAL Hazards", int(hazard_df["Hazard Level"].isin(["HIGH", "CRITICAL"]).sum()) if not hazard_df.empty else 0)
        c3.metric("Unavailable", len(failures))

        if hazard_df.empty:
            st.info("No price-hazard rows were produced.")
        else:
            order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3, "UNKNOWN": 4}
            hazard_df["_rank"] = hazard_df["Hazard Level"].map(order).fillna(5)
            hazard_df = hazard_df.sort_values(["_rank", "Hazard Points"], ascending=[True, False]).drop(columns=["_rank"])
            high_count = int(hazard_df["Hazard Level"].isin(["HIGH", "CRITICAL"]).sum())
            if high_count > 0:
                st.error(f"{high_count} symbol(s) show HIGH/CRITICAL price-hazard warnings.")
            st.dataframe(hazard_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Price-Hazard Scan CSV",
                hazard_df.to_csv(index=False).encode("utf-8"),
                file_name="psx_price_hazard_scan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.subheader("Detailed Hazard Alerts")
        if alerts_df.empty:
            st.info("No detailed hazard alerts were triggered.")
        else:
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Detailed Hazard Alerts CSV",
                alerts_df.to_csv(index=False).encode("utf-8"),
                file_name="psx_detailed_hazard_alerts.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.container(border=True):
            st.markdown("### Show unavailable symbols")
            st.write("\n".join(failures) if failures else "No symbol-level failures.")

    st.info(
        "This watchtower gives in-app warnings when you run it. For notifications while the app is closed, "
        "run the app on a server or schedule this scan locally with Windows Task Scheduler."
    )


def read_uploaded_table(uploaded_file):
    if uploaded_file is None:
        return None
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def imported_fundamental_image_table() -> pd.DataFrame | None:
    imported = st.session_state.get("fundamental_image_import_df")
    if isinstance(imported, pd.DataFrame) and not imported.empty:
        return imported.copy()
    return None


def has_imported_fundamental_images() -> bool:
    imported = imported_fundamental_image_table()
    return isinstance(imported, pd.DataFrame) and not imported.empty



def _sarmaaya_fill_table_to_wide_fundamentals(fill_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Sarmaaya Fundamental Fill Table into a normal wide fundamentals table."""
    if fill_df is None or not isinstance(fill_df, pd.DataFrame) or fill_df.empty:
        return pd.DataFrame()
    required = {"Symbol", "Fundamental Field", "Filled Value"}
    if not required.issubset(set(fill_df.columns)):
        return pd.DataFrame()

    rows = []
    for sym, group in fill_df.groupby(fill_df["Symbol"].astype(str).str.upper().str.strip()):
        if not sym or sym == "NAN":
            continue
        row = {"Symbol": sym, "symbol": sym, "Sarmaaya Fundamental Fill Used": "Yes"}
        filled_count = 0
        source_fields = []
        for _, r in group.iterrows():
            field = str(r.get("Fundamental Field", "") or "").strip()
            value = r.get("Filled Value", None)
            if not field:
                continue
            if pd.isna(value) or str(value).strip() in {"", "-", "None", "nan"}:
                continue
            row[field] = value
            filled_count += 1
            source_fields.append(field)
        row["Sarmaaya Filled Fields Count"] = filled_count
        row["Sarmaaya Filled Fields"] = ", ".join(source_fields[:80])
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_sarmaaya_fundamentals_into_base(base_df: pd.DataFrame) -> pd.DataFrame:
    """Use Sarmaaya saved data as a missing-fundamentals fallback across the full bot.

    If no uploaded/sheet/image fundamentals exist, the Sarmaaya wide table becomes the fundamentals table.
    If a fundamentals table exists, Sarmaaya fills blank/missing values by symbol and adds missing Sarmaaya-only symbols.
    """
    fill_df = _load_sarmaaya_fill_table_anytime()
    sarmaaya_wide = _sarmaaya_fill_table_to_wide_fundamentals(fill_df)
    if sarmaaya_wide is None or sarmaaya_wide.empty:
        return base_df if isinstance(base_df, pd.DataFrame) else pd.DataFrame()

    if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return sarmaaya_wide.copy()

    out = base_df.copy()

    # detect symbol column in base table
    symbol_col = None
    for c in out.columns:
        if str(c).strip().lower() in {"symbol", "symbols", "scrip", "ticker", "code"}:
            symbol_col = c
            break
    if symbol_col is None:
        out["Symbol"] = ""
        symbol_col = "Symbol"

    # Ensure all Sarmaaya fields exist in base.
    for c in sarmaaya_wide.columns:
        if c not in out.columns:
            out[c] = pd.NA

    out_symbols = out[symbol_col].astype(str).str.upper().str.strip()
    append_rows = []
    for _, srow in sarmaaya_wide.iterrows():
        sym = str(srow.get("Symbol", srow.get("symbol", "")) or "").upper().strip()
        if not sym:
            continue
        mask = out_symbols.eq(sym)
        if mask.any():
            for col in sarmaaya_wide.columns:
                val = srow.get(col, pd.NA)
                if pd.isna(val) or str(val).strip() in {"", "-", "None", "nan"}:
                    continue
                empty = out[col].isna() | out[col].astype(str).str.strip().isin(["", "-", "None", "nan"])
                out.loc[mask & empty, col] = val
            out.loc[mask, "Sarmaaya Fundamental Fill Used"] = "Yes"
        else:
            new_row = {c: pd.NA for c in out.columns}
            for col in sarmaaya_wide.columns:
                if col in new_row:
                    new_row[col] = srow.get(col, pd.NA)
            new_row[symbol_col] = sym
            if "Symbol" in new_row:
                new_row["Symbol"] = sym
            if "symbol" in new_row:
                new_row["symbol"] = sym
            new_row["Sarmaaya Fundamental Fill Used"] = "Yes"
            append_rows.append(new_row)

    if append_rows:
        out = pd.concat([out, pd.DataFrame(append_rows)], ignore_index=True, sort=False)

    return out


def read_fundamentals_source(uploaded_file=None, google_sheet_url: str = ""):
    """Load fundamentals and always merge saved Sarmaaya full-bot fundamentals safely."""
    url = str(google_sheet_url or "").strip()
    if url:
        base = _sanitize_dataframe_for_decision_engine(read_google_sheet_table(url))
        return _dedupe_symbol_rows_for_decision(_sanitize_dataframe_for_decision_engine(_force_object_columns(_merge_sarmaaya_full_bot_fundamentals(base))))

    uploaded = _sanitize_dataframe_for_decision_engine(read_uploaded_table(uploaded_file))
    if isinstance(uploaded, pd.DataFrame) and not uploaded.empty:
        return _dedupe_symbol_rows_for_decision(_sanitize_dataframe_for_decision_engine(_force_object_columns(_merge_sarmaaya_full_bot_fundamentals(uploaded))))

    image_table = _sanitize_dataframe_for_decision_engine(imported_fundamental_image_table())
    if isinstance(image_table, pd.DataFrame) and not image_table.empty:
        return _dedupe_symbol_rows_for_decision(_sanitize_dataframe_for_decision_engine(_force_object_columns(_merge_sarmaaya_full_bot_fundamentals(image_table))))

    return _dedupe_symbol_rows_for_decision(_sanitize_dataframe_for_decision_engine(_force_object_columns(_merge_sarmaaya_full_bot_fundamentals(pd.DataFrame()))))

def read_portfolio_source(uploaded_file=None, google_drive_pdf_url: str = ""):
    """Load portfolio from either a Google Drive PDF link or a local CSV/XLSX upload."""
    url = str(google_drive_pdf_url or "").strip()
    if url:
        return read_google_drive_portfolio_pdf(url)
    return read_uploaded_table(uploaded_file)


def investor_profile_panel():
    st.subheader("Investor Profile Engine")
    st.write(
        "This module converts the Week 12 investor-profile framework into a practical bot recommendation. "
        "It helps decide whether the terminal should behave like a passive allocator, active fundamental investor, swing trader, or day trader."
    )
    st.dataframe(profile_catalog(), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    horizon_years = c1.number_input("Investment / trading horizon in years", min_value=0.0, max_value=30.0, value=5.0, step=1.0, key="v50_profile_horizon")
    monitoring_frequency = c2.selectbox("How often can you monitor?", ["Rarely", "Annual", "Quarterly", "Monthly", "Weekly", "Daily"], index=3, key="v50_profile_monitor")
    prefers_funds = c3.checkbox("Prefer mutual funds / professional management", value=False, key="v50_profile_funds")

    d1, d2, d3 = st.columns(3)
    uses_fundamentals = d1.checkbox("I use fundamentals / business analysis", value=True, key="v50_profile_funda")
    uses_technicals = d2.checkbox("I use technical analysis for entry/exit", value=True, key="v50_profile_ta")
    trades_intraday = d3.checkbox("I take intraday/day trades", value=False, key="v50_profile_intraday")

    if st.button("Recommend My Investor Profile", type="primary", use_container_width=True, key="v50_profile_run"):
        profile = recommend_profile(
            horizon_years=horizon_years,
            monitoring_frequency=monitoring_frequency,
            prefers_funds=prefers_funds,
            uses_fundamentals=uses_fundamentals,
            uses_technicals=uses_technicals,
            trades_intraday=trades_intraday,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Recommended Profile", profile["label"])
        p2.metric("Style", profile["style"])
        p3.metric("Default Strategy", profile["default_strategy"])
        st.success(profile["best_for"])
        st.write("**Priority bot modules:** " + ", ".join(profile["modules"]))


def macro_checklist_panel():
    st.subheader("Macro Checklist & Company Sensitivity Matrix")
    st.write(
        "This panel converts the economy-and-stocks checklist into a scored top-down regime. "
        "Use it before sector or stock selection."
    )

    c1, c2, c3 = st.columns(3)
    inflation_status = c1.selectbox("Inflation status", ["Within/near target", "Above target", "Rising", "Falling", "Mixed"], index=0, key="v50_macro_inflation")
    policy_rate_direction = c2.selectbox("Policy rate direction", ["Falling", "Stable", "Rising"], index=1, key="v50_macro_rate")
    bond_yield_direction = c3.selectbox("Bond/T-bill yield direction", ["Falling", "Stable", "Rising"], index=1, key="v50_macro_bond")

    d1, d2, d3, d4 = st.columns(4)
    current_account_direction = d1.selectbox("Current account", ["Improving", "Stable", "Worsening"], index=0, key="v50_macro_cad")
    fiscal_account_direction = d2.selectbox("Fiscal account", ["Improving", "Stable", "Worsening"], index=1, key="v50_macro_fiscal")
    reserves_direction = d3.selectbox("SBP reserves", ["Improving", "Stable", "Worsening"], index=0, key="v50_macro_reserves")
    currency_direction = d4.selectbox("PKR currency trend", ["Stable", "Improving", "Weakening", "Volatile"], index=0, key="v50_macro_currency")

    e1, e2, e3, e4 = st.columns(4)
    oil_direction = e1.selectbox("Oil price trend", ["Falling", "Stable low", "Mixed", "Rising", "Spiking"], index=2, key="v50_macro_oil")
    market_pe = e2.number_input("KSE-100 / Market P/E", value=8.0, min_value=0.0, step=0.25, key="v50_macro_pe")
    earnings_yield = e3.number_input("Market earnings yield %", value=12.0, min_value=0.0, step=0.25, key="v50_macro_ey")
    bond_yield = e4.number_input("Bond / T-bill yield %", value=11.0, min_value=0.0, step=0.25, key="v50_macro_by")

    global_risk_tone = st.selectbox("Global risk tone", ["Risk-On / Supportive", "Neutral", "Risk-Off / Defensive"], index=1, key="v50_macro_global")

    if st.button("Score Macro Checklist", type="primary", use_container_width=True, key="v50_macro_run"):
        macro = macro_checklist_score(
            inflation_status=inflation_status,
            policy_rate_direction=policy_rate_direction,
            bond_yield_direction=bond_yield_direction,
            current_account_direction=current_account_direction,
            fiscal_account_direction=fiscal_account_direction,
            reserves_direction=reserves_direction,
            currency_direction=currency_direction,
            oil_direction=oil_direction,
            market_pe=market_pe,
            earnings_yield=earnings_yield,
            bond_yield=bond_yield,
            global_risk_tone=global_risk_tone,
        )
        m1, m2 = st.columns([1, 2])
        m1.metric("Macro Score", macro["macro_score"])
        m2.metric("Macro Regime", macro["macro_regime"])
        if "High-Risk" in macro["macro_regime"] or "Defensive" in macro["macro_regime"]:
            st.error(macro["quick_action"])
        elif "Constructive" in macro["macro_regime"] or "Bullish" in macro["macro_regime"]:
            st.success(macro["quick_action"])
        else:
            st.warning(macro["quick_action"])
        st.dataframe(macro["checklist_table"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Company Macro Sensitivity Matrix")
    f1, f2, f3, f4, f5 = st.columns(5)
    sector = f1.text_input("Sector", value="Cement", key="v50_macro_sector")
    export_oriented = f2.checkbox("Export oriented", value=False, key="v50_macro_export")
    import_dependent = f3.checkbox("Import dependent", value=True, key="v50_macro_import")
    interest_rate_sensitive = f4.selectbox("Rate sensitivity", ["Low", "Medium", "High"], index=2, key="v50_macro_rate_sens")
    oil_sensitive = f5.selectbox("Oil sensitivity", ["Low", "Medium", "High"], index=1, key="v50_macro_oil_sens")

    if st.button("Build Company Macro Matrix", use_container_width=True, key="v50_macro_company_run"):
        matrix = company_macro_sensitivity(
            sector=sector,
            export_oriented=export_oriented,
            import_dependent=import_dependent,
            interest_rate_sensitive=interest_rate_sensitive,
            oil_sensitive=oil_sensitive,
        )
        st.dataframe(matrix, use_container_width=True, hide_index=True)



def fundamental_image_import_center_panel() -> None:
    st.subheader("Fundamental Image Import Center — Growth to Margin of Safety")
    st.write(
        "Upload the six Sarmaaya / fundamental screenshots for one symbol, or paste a structured Google Sheet link. "
        "The bot will preserve the images, attempt local OCR extraction, map recognized metrics into the fundamental engine, "
        "and use the imported table as a fallback in the Decision Center and Autopilot when no CSV/XLSX or Sheet link is selected there."
    )

    meta1, meta2, meta3 = st.columns([0.8, 1.2, 1.0])
    import_symbol = meta1.text_input(
        "Symbol",
        value=str(st.session_state.get("fundamental_image_import_symbol", "")).upper(),
        key="fundamental_image_import_symbol",
        placeholder="e.g., MARI",
        help="Required. The six images are treated as one company's imported fundamental snapshot.",
    ).strip().upper()
    import_company = meta2.text_input(
        "Company name (optional)",
        value=str(st.session_state.get("fundamental_image_import_company", "")),
        key="fundamental_image_import_company",
        placeholder="e.g., Mari Energies Limited",
    ).strip()
    import_sector = meta3.text_input(
        "Sector (optional)",
        value=str(st.session_state.get("fundamental_image_import_sector", "Unknown")),
        key="fundamental_image_import_sector",
        placeholder="e.g., Oil & Gas",
    ).strip() or "Unknown"

    google_sheet_url = st.text_input(
        "Optional Google Sheet link for structured fundamental metrics",
        value="",
        key="fundamental_image_import_google_sheet",
        placeholder="Paste Google Sheets link here",
        help="Accepted formats: a standard wide fundamentals table with Symbol columns, or a long table using Category, Metric, Value columns.",
    )

    st.markdown("#### Upload the 6 required fundamental pages")
    slot_cols = st.columns(2)
    uploads = {}
    for idx, section in enumerate(IMAGE_SECTION_ORDER):
        col = slot_cols[idx % 2]
        label = f"{idx + 1}. {section} image"
        uploads[section] = col.file_uploader(
            label,
            type=["png", "jpg", "jpeg", "webp"],
            key=f"fundamental_image_upload_{idx}",
            help="Upload a clear screenshot. The bot will attempt OCR extraction, and you can use a Google Sheet as a structured fallback if OCR is unavailable.",
        )

    with st.container(border=True):
        st.markdown("### Accepted image set and data usage")
        st.markdown(
            """
            **Required image roles**  
            1. Growth  
            2. Stability  
            3. Valuation  
            4. Inventory  
            5. Cashflow  
            6. Main Page / Margin of Safety and summary metrics

            **How the bot uses them**  
            Recognized metrics are mapped into its fundamental engine wherever possible. The Main Page image can carry metrics such as current price, fair/intrinsic value, upside/downside, and margin-of-safety text when OCR can read them.

            **Google Sheet fallback**  
            A Google Sheet is the most reliable structured source. It can be a normal one-row-per-symbol fundamentals sheet, or a long sheet with `Category`, `Metric`, and `Value` columns for the six screenshot pages.
            """
        )

    process = st.button(
        "Process Fundamental Images / Google Sheet",
        type="primary",
        use_container_width=True,
        key="fundamental_image_import_process",
    )

    if process:
        try:
            if not import_symbol:
                raise ValueError("Please enter the symbol before processing the six image pages.")

            image_bytes = {}
            for section, uploaded in uploads.items():
                image_bytes[section] = uploaded.getvalue() if uploaded is not None else None

            image_count = sum(1 for value in image_bytes.values() if value)
            if image_count == 0 and not str(google_sheet_url or "").strip():
                raise ValueError("Upload at least one screenshot or paste a Google Sheet link before processing.")

            status_df, metric_long_df, image_wide_df = process_image_bundle(
                image_bytes,
                symbol=import_symbol,
                company=import_company,
                sector=import_sector,
            )

            google_df = None
            if str(google_sheet_url or "").strip():
                google_df = read_google_sheet_table(google_sheet_url)

            final_fundamentals_df, google_long_df = combine_structured_fundamentals(
                image_wide_df if image_count > 0 else None,
                google_df,
                symbol=import_symbol,
                company=import_company,
                sector=import_sector,
            )
            if final_fundamentals_df is None or final_fundamentals_df.empty:
                final_fundamentals_df = image_wide_df

            st.session_state["fundamental_image_import_status_df"] = status_df
            st.session_state["fundamental_image_import_metrics_df"] = metric_long_df
            st.session_state["fundamental_image_import_google_long_df"] = google_long_df
            st.session_state["fundamental_image_import_df"] = final_fundamentals_df
            st.session_state["fundamental_image_import_last_symbol"] = import_symbol
            st.success(
                f"Fundamental import completed for {import_symbol}. This data is now available as the fallback fundamentals source in Decision Center and Autopilot."
            )
        except Exception as exc:
            st.error(f"Fundamental Image Import failed: {exc}")

    status_df = st.session_state.get("fundamental_image_import_status_df")
    metric_long_df = st.session_state.get("fundamental_image_import_metrics_df")
    google_long_df = st.session_state.get("fundamental_image_import_google_long_df")
    final_df = st.session_state.get("fundamental_image_import_df")

    if isinstance(status_df, pd.DataFrame) and not status_df.empty:
        st.markdown("#### Image Processing Status")
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        parsed_total = int(pd.to_numeric(status_df.get("Parsed Metrics", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if parsed_total == 0 and status_df["Uploaded"].astype(str).str.upper().eq("YES").any():
            st.warning(
                "The images were received, but no usable Metric / Value rows were extracted. This usually means the local OCR engine is unavailable or the screenshot text layout is difficult. "
                "The Google Sheet link remains the reliable structured path. For OCR, install `pytesseract` dependencies and the desktop Tesseract engine on Windows."
            )

    if isinstance(metric_long_df, pd.DataFrame) and not metric_long_df.empty:
        st.markdown("#### OCR-Extracted Metric Rows")
        st.dataframe(metric_long_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download OCR Extracted Metrics CSV",
            metric_long_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{str(st.session_state.get('fundamental_image_import_last_symbol', 'symbol')).lower()}_image_ocr_metrics.csv",
            mime="text/csv",
            use_container_width=True,
            key="fundamental_image_ocr_download",
        )

    if isinstance(google_long_df, pd.DataFrame) and not google_long_df.empty:
        st.markdown("#### Google Sheet Long-Format Metrics Detected")
        st.dataframe(google_long_df, use_container_width=True, hide_index=True)

    if isinstance(final_df, pd.DataFrame) and not final_df.empty:
        st.markdown("#### Final Imported Fundamental Table Used by the Bot")
        st.dataframe(final_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Final Imported Fundamentals CSV",
            final_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{str(st.session_state.get('fundamental_image_import_last_symbol', 'symbol')).lower()}_final_imported_fundamentals.csv",
            mime="text/csv",
            use_container_width=True,
            key="fundamental_image_final_download",
        )
        st.info(
            "Use the Decision Center next. Leave the Decision Center fundamental CSV/Google Sheet inputs blank and it will automatically use this imported image/sheet fundamentals table as the fallback data source."
        )

def master_fundamentals_panel():
    st.subheader("Master Fundamentals, Intrinsic Value & Financial Fakery Scanner")
    st.write(
        "Upload a CSV/XLSX fundamentals file. The engine separates banks from non-financial companies, "
        "scores the uploaded Week 7–10 frameworks, estimates intrinsic value where inputs exist, and runs a red-flag scanner."
    )
    uploaded = st.file_uploader("Upload Fundamentals CSV/XLSX", type=["csv", "xlsx", "xls"], key="v50_master_funda_upload")
    google_sheet_url = st.text_input(
        "Or paste Fundamental Google Sheet link",
        value="",
        key="v50_master_funda_sheet_url",
        placeholder="Paste Google Sheets link here",
        help="The sheet must be link-accessible. When provided, this link source is used instead of the local upload.",
    )
    st.info(_sarmaaya_full_bot_status_text())
    sarmaaya_available = isinstance(_load_sarmaaya_fill_table_anytime(), pd.DataFrame) and not _load_sarmaaya_fill_table_anytime().empty
    if uploaded is None and not str(google_sheet_url or "").strip() and not has_imported_fundamental_images() and not sarmaaya_available:
        st.info("Upload a fundamentals dataset, paste a Google Sheet link, process images, or save Sarmaaya six-box data to unlock the master scoring engine.")
        return
    if uploaded is None and not str(google_sheet_url or "").strip() and has_imported_fundamental_images():
        st.info("Using the latest Fundamental Image Import table as the fallback fundamentals source.")

    try:
        raw = _sanitize_dataframe_for_decision_engine(read_fundamentals_source(uploaded, google_sheet_url))
        universe = score_fundamental_universe(raw)
        st.success(f"Loaded and scored {len(universe)} symbol(s).")
        st.subheader("Universe Ranking")
        display_cols = [c for c in ["symbol", "company", "sector", "is_bank", "fundamental_grade", "fundamental_score_pct", "good", "average", "bad", "missing"] if c in universe.columns]
        st.dataframe(universe[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download Master Fundamental Ranking CSV",
            universe.drop(columns=["detail_table"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="master_fundamental_ranking.csv",
            mime="text/csv",
            use_container_width=True,
        )

        symbols = universe["symbol"].astype(str).tolist()
        selected_symbol = st.selectbox("Choose a symbol for detailed review", symbols, key="v50_master_symbol")
        detail, summary, source_row = score_single_symbol(raw, selected_symbol)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Fundamental Grade", summary["grade"])
        s2.metric("Score %", summary["score_pct"])
        s3.metric("Good / Bad", f'{summary["good"]} / {summary["bad"]}')
        s4.metric("Missing Inputs", summary["missing"])

        st.subheader(f"{selected_symbol} — Detailed Good / Average / Bad Scorecard")
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.subheader("Intrinsic Value Composite")
        assumptions_cols = st.columns(4)
        discount_rate_pct = assumptions_cols[0].number_input("DCF discount rate %", value=12.0, min_value=1.0, max_value=40.0, step=0.5, key="v50_iv_discount")
        terminal_growth_pct = assumptions_cols[1].number_input("Terminal growth %", value=4.0, min_value=0.0, max_value=10.0, step=0.25, key="v50_iv_terminal")
        growth_years = assumptions_cols[2].number_input("Growth years", value=10, min_value=1, max_value=30, step=1, key="v50_iv_growth_years")
        terminal_years = assumptions_cols[3].number_input("Terminal years", value=10, min_value=1, max_value=30, step=1, key="v50_iv_terminal_years")
        intrinsic = intrinsic_value_composite(
            source_row,
            assumptions={
                "discount_rate_pct": discount_rate_pct,
                "terminal_growth_pct": terminal_growth_pct,
                "growth_years": int(growth_years),
                "terminal_years": int(terminal_years),
            },
        )
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Current Price", intrinsic["current_price"])
        i2.metric("Intrinsic FCF", intrinsic["composite_intrinsic_value_fcf"])
        i3.metric("Intrinsic Cash", intrinsic["composite_intrinsic_value_cash"])
        i4.metric("Best Margin of Safety %", intrinsic["best_margin_of_safety_pct"])
        if intrinsic["valuation_grade"] == "Attractive":
            st.success(intrinsic["valuation_verdict"])
        elif intrinsic["valuation_grade"] == "Overvalued":
            st.error(intrinsic["valuation_verdict"])
        else:
            st.warning(intrinsic["valuation_verdict"])
        if intrinsic["warnings"]:
            st.warning(" | ".join(intrinsic["warnings"]))
        st.dataframe(intrinsic["methods_table"], use_container_width=True, hide_index=True)

        st.subheader("Financial Fakery / Governance Red-Flag Scanner")
        report_text = st.text_area(
            "Optional: paste annual report / auditor report / announcement text to scan keywords",
            value="",
            height=140,
            key="v50_fraud_text",
        )
        fraud = score_financial_fakery(source_row, report_text=report_text)
        f1, f2 = st.columns(2)
        f1.metric("Fakery Risk Level", fraud["fraud_risk_level"])
        f2.metric("Risk Points", fraud["fraud_risk_points"])
        if fraud["fraud_risk_level"] in {"CRITICAL", "HIGH"}:
            st.error(fraud["verdict"])
        elif fraud["fraud_risk_level"] == "MODERATE":
            st.warning(fraud["verdict"])
        else:
            st.success(fraud["verdict"])
        if not fraud["flags_table"].empty:
            st.dataframe(fraud["flags_table"], use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Master fundamentals analysis failed: {exc}")


def strategy_builder_panel():
    st.subheader("Strategy Builder — Strategies 1 to 4")
    st.write(
        "Upload a CSV/XLSX fundamentals file. The engine reproduces the Week 12 strategy logic: "
        "Top 10% scoring, red-zone sector tilt, low-price/high-beta rotation, and intrinsic-value filter."
    )
    uploaded = st.file_uploader("Upload Fundamentals CSV/XLSX for Strategy Builder", type=["csv", "xlsx", "xls"], key="v50_strategy_upload")
    google_sheet_url = st.text_input(
        "Or paste Fundamental Google Sheet link for Strategy Builder",
        value="",
        key="v50_strategy_sheet_url",
        placeholder="Paste Google Sheets link here",
        help="The sheet must be link-accessible. When provided, this link source is used instead of the local upload.",
    )
    if uploaded is None and not str(google_sheet_url or "").strip() and not has_imported_fundamental_images():
        st.info("Upload a fundamentals dataset, paste a Google Sheet link, or process the six Fundamental Image Import pages to build strategy baskets.")
        return
    if uploaded is None and not str(google_sheet_url or "").strip() and has_imported_fundamental_images():
        st.info("Using the latest Fundamental Image Import table as the fallback fundamentals source for Strategy Builder.")

    try:
        raw = read_fundamentals_source(uploaded, google_sheet_url)
        norm = normalize_fundamental_columns(raw)
        scored = score_fundamental_universe(raw)
        raw_map = norm.set_index("symbol", drop=False)

        # Add valuation outputs where possible
        scored = scored.copy()
        intrinsic_rows = []
        for symbol in scored["symbol"].astype(str):
            if symbol in raw_map.index:
                iv = intrinsic_value_composite(raw_map.loc[symbol])
                intrinsic_rows.append({
                    "symbol": symbol,
                    "best_margin_of_safety_pct": iv["best_margin_of_safety_pct"],
                    "margin_of_safety_fcf_pct": iv["margin_of_safety_fcf_pct"],
                    "margin_of_safety_cash_pct": iv["margin_of_safety_cash_pct"],
                    "valuation_grade": iv["valuation_grade"],
                })
        if intrinsic_rows:
            scored = scored.merge(pd.DataFrame(intrinsic_rows), on="symbol", how="left")
        # Bring required strategy fields from normalized source
        merge_cols = [c for c in ["symbol", "price", "beta"] if c in norm.columns]
        if merge_cols:
            scored = scored.merge(norm[merge_cols].drop_duplicates("symbol"), on="symbol", how="left")

        strategies = build_all_strategies(scored)
        st.dataframe(strategy_summary(strategies), use_container_width=True, hide_index=True)

        tabs = st.tabs(["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"])
        for tab, name in zip(tabs, ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"]):
            with tab:
                df = strategies[name]
                if df.empty:
                    st.info(f"No candidates generated for {name}. Check whether the required fields are available.")
                else:
                    view = df.drop(columns=["detail_table"], errors="ignore")
                    st.dataframe(view, use_container_width=True, hide_index=True)
                    st.download_button(
                        f"Download {name} CSV",
                        view.to_csv(index=False).encode("utf-8"),
                        file_name=f"{name.lower().replace(' ', '_')}_candidates.csv",
                        mime="text/csv",
                        key=f"v50_strategy_download_{name}",
                        use_container_width=True,
                    )

    except Exception as exc:
        st.error(f"Strategy builder failed: {exc}")


def corporate_catalyst_panel():
    st.subheader("Corporate Catalyst & Explosive Stock Scanner")
    st.write(
        "This scanner turns the Assignment 10 framework into a practical catalyst detector. "
        "It searches for buybacks, capacity expansion, solar projects, M&A, production increases, and other special developments."
    )

    mode = st.radio("Scanner mode", ["Paste announcement text", "Fetch official PSX announcements for symbols"], horizontal=True, key="v50_catalyst_mode")
    if mode == "Paste announcement text":
        symbol = st.text_input("Symbol", value="KOHC", key="v50_catalyst_symbol")
        text = st.text_area("Paste announcement / corporate update text", height=180, key="v50_catalyst_text")
        if st.button("Scan Catalyst Text", type="primary", use_container_width=True, key="v50_catalyst_run_text"):
            scan = scan_catalyst_text(text, symbol=symbol)
            c1, c2, c3 = st.columns(3)
            c1.metric("Catalyst Score", scan["catalyst_score"])
            c2.metric("Catalyst Grade", scan["catalyst_grade"])
            c3.metric("Explosive Flag", scan["explosive_stock_flag"])
            if scan["catalysts"].empty:
                st.info("No catalyst keywords detected.")
            else:
                st.dataframe(scan["catalysts"], use_container_width=True, hide_index=True)
    else:
        symbols_text = st.text_area("Symbols", value="FFC,FFBL,GLAXO,THCCL,CSAP", height=90, key="v50_catalyst_symbols")
        symbols = parse_symbols(symbols_text)
        if st.button("Fetch Official Announcements & Scan Catalysts", type="primary", use_container_width=True, key="v50_catalyst_run_web"):
            try:
                news = build_news_event_risk_snapshot(symbols=symbols)
                events = news.get("events")
                if isinstance(events, pd.DataFrame) and not events.empty:
                    catalysts = catalyst_from_events(events, symbols=symbols)
                    st.dataframe(catalysts, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Corporate Catalyst Scan CSV",
                        catalysts.to_csv(index=False).encode("utf-8"),
                        file_name="corporate_catalyst_scan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No structured official announcement rows were parsed for the selected symbols.")
                if news.get("errors"):
                    with st.container(border=True):
                        st.markdown("### Source parsing notes")
                        st.write("\n".join(news["errors"]))
            except Exception as exc:
                st.error(f"Catalyst scanner failed: {exc}")

def alert_center_panel():
    st.subheader("Alert Center — Saved Trend / Analysis State")
    st.write(
        "This tab shows the most recently saved monitoring state for symbols analyzed in "
        "Single Symbol PRO or Watchlist PRO Scorecard. The bot compares each new run with "
        "the prior saved state and raises change alerts for trend, scenario, bias, risk, score, "
        "and trade-plan shifts."
    )

    states_df = load_alert_state_table()
    if states_df.empty:
        st.info("No saved alert state yet. Run Single Symbol PRO or Watchlist PRO Scorecard once to create the baseline.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored Snapshots", len(states_df))
    c2.metric("HIGH/CRITICAL Risk", int(states_df["Risk Alert"].isin(["HIGH", "CRITICAL"]).sum()) if "Risk Alert" in states_df.columns else 0)
    c3.metric("Bullish Bias", int((states_df["Bias"] == "Bullish").sum()) if "Bias" in states_df.columns else 0)
    c4.metric("Bearish Bias", int((states_df["Bias"] == "Bearish").sum()) if "Bias" in states_df.columns else 0)

    severity_filter = st.multiselect(
        "Filter by current risk alert",
        ["LOW", "MODERATE", "HIGH", "CRITICAL"],
        default=["LOW", "MODERATE", "HIGH", "CRITICAL"],
        key="v21_alert_center_filter",
    )
    filtered = states_df
    if severity_filter and "Risk Alert" in filtered.columns:
        filtered = filtered[filtered["Risk Alert"].isin(severity_filter)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Current Alert State CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="psx_current_alert_state.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Change alerts are generated when the same symbol/timeframe is analyzed again. "
        "For quick monitoring, run the Watchlist PRO Scorecard repeatedly during the session."
    )

def fundamentals_panel():
    st.subheader("Fundamentals & Rankings")
    st.write(
        "Upload a fundamentals CSV to generate PSX ranking tables. "
        "The bot accepts flexible column names such as Symbol, Sector, P/E, P/B, ROE, Dividend Yield, EPS Growth, Revenue Growth, Debt/Equity, Net Margin, and FCF Yield."
    )

    uploaded = st.file_uploader("Upload Fundamentals CSV/XLSX", type=["csv", "xlsx", "xls"], key="v20_funda_csv")
    google_sheet_url = st.text_input(
        "Or paste Fundamental Google Sheet link",
        value="",
        key="v20_funda_sheet_url",
        placeholder="Paste Google Sheets link here",
        help="The sheet must be link-accessible. When provided, this link source is used instead of the local upload.",
    )
    watchlist_text = st.text_area(
        "Optional watchlist symbols for fundamental scorecard",
        value="ATLH,EFERT,FFC,MARI,NATF,NBP,OGDC,POL,SYS,UBL",
        height=80,
        key="v20_funda_watchlist",
    )
    st.info(_sarmaaya_full_bot_status_text())
    sarmaaya_available = isinstance(_load_sarmaaya_fill_table_anytime(), pd.DataFrame) and not _load_sarmaaya_fill_table_anytime().empty
    if uploaded is None and not str(google_sheet_url or "").strip() and not has_imported_fundamental_images() and not sarmaaya_available:
        st.info("Upload a fundamentals CSV/XLSX, paste a Google Sheet link, process images, or save Sarmaaya six-box data to unlock ranking tables.")
        return
    if uploaded is None and not str(google_sheet_url or "").strip() and has_imported_fundamental_images():
        st.info("Using the latest Fundamental Image Import table as the fallback fundamentals source for rankings.")

    try:
        df_raw = _sanitize_dataframe_for_decision_engine(read_fundamentals_source(uploaded, google_sheet_url))
        f = prepare_fundamentals(df_raw)
        rankings = build_fundamental_rankings(f, watchlist=parse_symbols(watchlist_text))
        st.success(f"Loaded fundamentals for {f['symbol'].nunique()} symbols.")

        t1, t2, t3, t4, t5 = st.tabs([
            "Sector Ranking",
            "Top Fundamentals",
            "Dividend Shortlist",
            "Undervalued Shortlist",
            "Watchlist Fundamental Scorecard",
        ])

        with t1:
            st.dataframe(rankings["sector_ranking"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Sector Ranking CSV",
                rankings["sector_ranking"].to_csv(index=False).encode("utf-8"),
                "sector_ranking.csv",
                "text/csv",
            )
        with t2:
            st.dataframe(rankings["top_fundamentals"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Top Fundamentals CSV",
                rankings["top_fundamentals"].to_csv(index=False).encode("utf-8"),
                "top_fundamentals.csv",
                "text/csv",
            )
        with t3:
            st.dataframe(rankings["dividend_shortlist"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Dividend Shortlist CSV",
                rankings["dividend_shortlist"].to_csv(index=False).encode("utf-8"),
                "dividend_shortlist.csv",
                "text/csv",
            )
        with t4:
            st.dataframe(rankings["undervalued_shortlist"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Undervalued Shortlist CSV",
                rankings["undervalued_shortlist"].to_csv(index=False).encode("utf-8"),
                "undervalued_shortlist.csv",
                "text/csv",
            )
        with t5:
            scorecard = rankings.get("watchlist_fundamental_scorecard", pd.DataFrame())
            if not scorecard.empty:
                st.dataframe(scorecard, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Watchlist Fundamental Scorecard CSV",
                    scorecard.to_csv(index=False).encode("utf-8"),
                    "watchlist_fundamental_scorecard.csv",
                    "text/csv",
                )
            else:
                st.info("No watchlist symbols matched the uploaded fundamentals file.")

    except Exception as exc:
        st.error(f"Fundamentals ranking failed: {exc}")




def portfolio_desk_panel() -> None:
    st.subheader("Portfolio Desk")
    st.write("Use this desk when you want the original detailed autopilot portfolio workflow, full files/links inputs, and diagnostic tables.")
    autopilot_manager_panel()


def stock_deep_dive_desk_panel() -> None:
    st.subheader("Stock Deep Dive")
    st.write("Run a symbol-specific professional report with charts, patterns, risk, and trade plan diagnostics.")
    single_symbol_panel()



def _fast_down_alert_from_chart(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> dict:
    """Strict confirmed fast-down detector.

    Important rule:
    A stock is NOT marked as Fast Down only because RSI is weak or price is below MA.
    It must have a real recent price drop plus breakdown confirmation.
    """
    alert = {
        "Symbol": symbol,
        "Alert": "No",
        "Confirmed Fast Down": "No",
        "Severity": "Normal",
        "Score": 0,
        "Last Price": None,
        "1 Candle Drop %": None,
        "3 Candle Drop %": None,
        "5 Candle Drop %": None,
        "RSI": None,
        "Volume Ratio": None,
        "ATR %": None,
        "Nearest Support": None,
        "Price Drop Confirmed": "No",
        "Support Break": "No",
        "MA Breakdown": "No",
        "Volume Spike": "No",
        "ATR Shock": "No",
        "RSI Weak": "No",
        "Big Red Candle": "No",
        "Reason": "",
        "Why Not Alert": "",
        "Action Hint": "No urgent action",
    }

    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 30:
            alert["Why Not Alert"] = "Not enough chart candles for reliable scan."
            alert["Reason"] = "Not enough chart candles."
            return alert

        chart = _ensure_chart_indicators(df.copy())
        for c in ["open", "high", "low", "close"]:
            if c not in chart.columns:
                alert["Why Not Alert"] = f"Missing OHLC column: {c}"
                alert["Reason"] = f"Missing OHLC column: {c}"
                return alert

        open_ = pd.to_numeric(chart["open"], errors="coerce")
        high = pd.to_numeric(chart["high"], errors="coerce")
        low = pd.to_numeric(chart["low"], errors="coerce")
        close = pd.to_numeric(chart["close"], errors="coerce")
        volume = pd.to_numeric(chart.get("volume", pd.Series(index=chart.index, dtype=float)), errors="coerce")
        rsi = pd.to_numeric(chart.get("rsi", pd.Series(index=chart.index, dtype=float)), errors="coerce")

        valid = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "rsi": rsi
        }).dropna(subset=["open", "high", "low", "close"])

        if len(valid) < 30:
            alert["Why Not Alert"] = "Not enough valid OHLC candles after cleaning."
            alert["Reason"] = "Insufficient valid chart data."
            return alert

        open_ = valid["open"]
        high = valid["high"]
        low = valid["low"]
        close = valid["close"]
        volume = valid["volume"]
        rsi = valid["rsi"]

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        close_3 = float(close.iloc[-4]) if len(close) >= 4 else prev_close
        close_5 = float(close.iloc[-6]) if len(close) >= 6 else close_3
        last_open = float(open_.iloc[-1])
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])

        one_drop = (last_close - prev_close) / max(abs(prev_close), 1e-9) * 100
        three_drop = (last_close - close_3) / max(abs(close_3), 1e-9) * 100
        five_drop = (last_close - close_5) / max(abs(close_5), 1e-9) * 100

        last_rsi = float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else None
        prev_rsi = float(rsi.iloc[-4]) if len(rsi) >= 4 and pd.notna(rsi.iloc[-4]) else None

        body_pct = (last_close - last_open) / max(abs(last_open), 1e-9) * 100
        range_pct = abs(last_high - last_low) / max(abs(last_close), 1e-9) * 100
        close_position = (last_close - last_low) / max((last_high - last_low), 1e-9)

        tf = str(timeframe or "").lower()
        if tf in {"15m", "30m"}:
            min_1_drop, min_3_drop, hard_1_drop, hard_3_drop = -0.8, -1.6, -1.5, -2.8
        elif tf == "1h":
            min_1_drop, min_3_drop, hard_1_drop, hard_3_drop = -1.0, -2.0, -1.8, -3.5
        elif tf == "4h":
            min_1_drop, min_3_drop, hard_1_drop, hard_3_drop = -1.4, -2.8, -2.3, -4.5
        else:
            min_1_drop, min_3_drop, hard_1_drop, hard_3_drop = -2.0, -4.0, -3.0, -6.0

        # True price drop requirement.
        price_drop_confirmed = (one_drop <= min_1_drop) or (three_drop <= min_3_drop)
        hard_price_drop = (one_drop <= hard_1_drop) or (three_drop <= hard_3_drop)

        # Support levels, excluding current candle.
        support_10 = float(low.iloc[-11:-1].min()) if len(low) >= 12 else float(low.iloc[:-1].min())
        support_20 = float(low.iloc[-21:-1].min()) if len(low) >= 22 else support_10
        support_50 = float(low.iloc[-51:-1].min()) if len(low) >= 52 else support_20
        nearest_support = max(s for s in [support_10, support_20, support_50] if pd.notna(s))

        break_10 = last_close < support_10
        break_20 = last_close < support_20
        break_50 = last_close < support_50
        support_break = break_10 or break_20 or break_50

        # Moving averages.
        sma5 = close.rolling(5, min_periods=3).mean()
        sma10 = close.rolling(10, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=10).mean()
        below_sma5 = pd.notna(sma5.iloc[-1]) and last_close < float(sma5.iloc[-1])
        below_sma10 = pd.notna(sma10.iloc[-1]) and last_close < float(sma10.iloc[-1])
        below_sma20 = pd.notna(sma20.iloc[-1]) and last_close < float(sma20.iloc[-1])
        fresh_sma20_break = pd.notna(sma20.iloc[-1]) and pd.notna(sma20.iloc[-2]) and close.iloc[-2] >= sma20.iloc[-2] and last_close < sma20.iloc[-1]
        ma_breakdown = fresh_sma20_break or (below_sma5 and below_sma10 and below_sma20)

        # Candle confirmation.
        big_red = body_pct < 0 and abs(body_pct) >= abs(min_1_drop) and close_position <= 0.35
        panic_close = body_pct < 0 and close_position <= 0.20 and range_pct >= abs(min_1_drop) * 1.5

        # Volume confirmation.
        vol_ratio = None
        volume_spike = False
        if volume.notna().sum() >= 12 and pd.notna(volume.iloc[-1]):
            avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 22 else float(volume.iloc[:-1].mean())
            if avg_vol and avg_vol > 0:
                vol_ratio = float(volume.iloc[-1]) / avg_vol
                volume_spike = vol_ratio >= 1.35

        # ATR shock.
        prev_close_series = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close_series).abs(),
            (low - prev_close_series).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean()
        atr_pct = float(atr14.iloc[-1] / max(abs(last_close), 1e-9) * 100) if pd.notna(atr14.iloc[-1]) else None
        atr_shock = atr_pct is not None and (abs(one_drop) >= max(atr_pct * 1.15, abs(min_1_drop)) or abs(three_drop) >= atr_pct * 2.0)

        # RSI confirmation only; never standalone.
        rsi_weak = False
        rsi_collapse = False
        if last_rsi is not None:
            rsi_weak = last_rsi <= 40
            rsi_collapse = prev_rsi is not None and (last_rsi - prev_rsi) <= -9

        # Main confirmation rule:
        # Price drop + at least one real breakdown/candle confirmation.
        structural_confirmations = [
            support_break,
            ma_breakdown,
            big_red,
            panic_close,
            atr_shock,
        ]
        secondary_confirmations = [
            volume_spike,
            rsi_weak,
            rsi_collapse,
            five_drop <= min_3_drop,
        ]

        structural_count = sum(bool(x) for x in structural_confirmations)
        secondary_count = sum(bool(x) for x in secondary_confirmations)

        confirmed_fast_down = bool(
            price_drop_confirmed and (
                structural_count >= 1 or
                (hard_price_drop and secondary_count >= 1)
            )
        )

        score = 0
        reasons = []
        why_not = []

        if one_drop <= hard_1_drop:
            score += 22; reasons.append(f"hard 1-candle drop {one_drop:.2f}%")
        elif one_drop <= min_1_drop:
            score += 14; reasons.append(f"1-candle drop {one_drop:.2f}%")
        else:
            why_not.append(f"1-candle drop not enough ({one_drop:.2f}%)")

        if three_drop <= hard_3_drop:
            score += 22; reasons.append(f"hard 3-candle drop {three_drop:.2f}%")
        elif three_drop <= min_3_drop:
            score += 14; reasons.append(f"3-candle drop {three_drop:.2f}%")
        else:
            why_not.append(f"3-candle drop not enough ({three_drop:.2f}%)")

        if five_drop <= min_3_drop:
            score += 8; reasons.append(f"5-candle down pressure {five_drop:.2f}%")

        if support_break:
            if break_50:
                score += 22; reasons.append(f"broke 50-candle support {support_50:.2f}")
            elif break_20:
                score += 18; reasons.append(f"broke 20-candle support {support_20:.2f}")
            else:
                score += 12; reasons.append(f"broke 10-candle support {support_10:.2f}")
        else:
            why_not.append("no support breakdown")

        if ma_breakdown:
            score += 12; reasons.append("moving-average breakdown confirmed")
        else:
            why_not.append("no strong MA breakdown")

        if big_red or panic_close:
            score += 12; reasons.append(f"bearish candle closed near low; body {body_pct:.2f}%")
        else:
            why_not.append("last candle is not a strong bearish close")

        if volume_spike:
            score += 8; reasons.append(f"volume confirms selling {vol_ratio:.2f}x avg")
        if atr_shock:
            score += 8; reasons.append(f"drop exceeds volatility pressure; ATR {atr_pct:.2f}%")
        if rsi_weak:
            score += 5; reasons.append(f"RSI weak {last_rsi:.1f}")
        if rsi_collapse:
            score += 5; reasons.append(f"RSI fell fast {prev_rsi:.1f} → {last_rsi:.1f}")

        if not price_drop_confirmed:
            confirmed_fast_down = False
            why_not.insert(0, "price drop requirement not met")
        if price_drop_confirmed and structural_count == 0 and not (hard_price_drop and secondary_count >= 1):
            confirmed_fast_down = False
            why_not.insert(0, "price dropped, but no breakdown confirmation")

        if confirmed_fast_down:
            if score >= 75 or (hard_price_drop and support_break):
                severity, flag, action = "CRITICAL", "YES", "Confirmed fast down: check stop/support immediately; avoid averaging."
            elif score >= 55:
                severity, flag, action = "HIGH", "YES", "Confirmed fast down: wait for stabilization or follow exit/reduce plan."
            else:
                severity, flag, action = "MEDIUM", "YES", "Confirmed but moderate: monitor next candle/support."
        else:
            # Do not show weak false positives in main alert table.
            severity, flag, action = "Normal", "No", "No confirmed fast-down alert."

        alert.update({
            "Alert": flag,
            "Confirmed Fast Down": "Yes" if confirmed_fast_down else "No",
            "Severity": severity,
            "Score": int(min(score, 100)),
            "Current Price": round(last_close, 4),
            "Last Price": round(last_close, 4),
            "1 Candle Drop %": round(one_drop, 2),
            "3 Candle Drop %": round(three_drop, 2),
            "5 Candle Drop %": round(five_drop, 2),
            "RSI": round(last_rsi, 2) if last_rsi is not None else None,
            "Volume Ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "ATR %": round(atr_pct, 2) if atr_pct is not None else None,
            "Nearest Support": round(nearest_support, 4),
            "Price Drop Confirmed": "Yes" if price_drop_confirmed else "No",
            "Support Break": "Yes" if support_break else "No",
            "MA Breakdown": "Yes" if ma_breakdown else "No",
            "Volume Spike": "Yes" if volume_spike else "No",
            "ATR Shock": "Yes" if atr_shock else "No",
            "RSI Weak": "Yes" if rsi_weak else "No",
            "Big Red Candle": "Yes" if (big_red or panic_close) else "No",
            "Reason": "; ".join(reasons) if reasons else "No confirmed fast-down condition.",
            "Why Not Alert": "; ".join(dict.fromkeys(why_not)) if why_not else "",
            "Action Hint": action,
        })
        return alert

    except Exception as exc:
        alert["Reason"] = f"Alert check failed: {exc}"
        alert["Why Not Alert"] = f"Exception: {exc}"
        return alert




def _fast_up_alert_from_chart(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> dict:
    """Strict confirmed fast-up detector.

    A stock is marked Fast Up only when recent price rise is confirmed by breakout,
    MA strength, bullish candle, ATR impulse, volume, or RSI momentum.
    """
    alert = {
        "Symbol": symbol,
        "Alert": "No",
        "Confirmed Fast Up": "No",
        "Direction": "UP",
        "Severity": "Normal",
        "Score": 0,
        "Current Price": None,
        "Last Price": None,
        "1 Candle Rise %": None,
        "3 Candle Rise %": None,
        "5 Candle Rise %": None,
        "RSI": None,
        "Volume Ratio": None,
        "ATR %": None,
        "Nearest Resistance": None,
        "Price Rise Confirmed": "No",
        "Resistance Break": "No",
        "MA Breakout": "No",
        "Volume Spike": "No",
        "ATR Impulse": "No",
        "RSI Strong": "No",
        "Big Green Candle": "No",
        "Reason": "",
        "Why Not Alert": "",
        "Action Hint": "No urgent action",
    }

    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 30:
            alert["Why Not Alert"] = "Not enough chart candles for reliable scan."
            alert["Reason"] = "Not enough chart candles."
            return alert

        chart = _ensure_chart_indicators(df.copy())
        for c in ["open", "high", "low", "close"]:
            if c not in chart.columns:
                alert["Why Not Alert"] = f"Missing OHLC column: {c}"
                alert["Reason"] = f"Missing OHLC column: {c}"
                return alert

        open_ = pd.to_numeric(chart["open"], errors="coerce")
        high = pd.to_numeric(chart["high"], errors="coerce")
        low = pd.to_numeric(chart["low"], errors="coerce")
        close = pd.to_numeric(chart["close"], errors="coerce")
        volume = pd.to_numeric(chart.get("volume", pd.Series(index=chart.index, dtype=float)), errors="coerce")
        rsi = pd.to_numeric(chart.get("rsi", pd.Series(index=chart.index, dtype=float)), errors="coerce")

        valid = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "rsi": rsi
        }).dropna(subset=["open", "high", "low", "close"])

        if len(valid) < 30:
            alert["Why Not Alert"] = "Not enough valid OHLC candles after cleaning."
            alert["Reason"] = "Insufficient valid chart data."
            return alert

        open_ = valid["open"]
        high = valid["high"]
        low = valid["low"]
        close = valid["close"]
        volume = valid["volume"]
        rsi = valid["rsi"]

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        close_3 = float(close.iloc[-4]) if len(close) >= 4 else prev_close
        close_5 = float(close.iloc[-6]) if len(close) >= 6 else close_3
        last_open = float(open_.iloc[-1])
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])

        one_rise = (last_close - prev_close) / max(abs(prev_close), 1e-9) * 100
        three_rise = (last_close - close_3) / max(abs(close_3), 1e-9) * 100
        five_rise = (last_close - close_5) / max(abs(close_5), 1e-9) * 100

        last_rsi = float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else None
        prev_rsi = float(rsi.iloc[-4]) if len(rsi) >= 4 and pd.notna(rsi.iloc[-4]) else None

        body_pct = (last_close - last_open) / max(abs(last_open), 1e-9) * 100
        range_pct = abs(last_high - last_low) / max(abs(last_close), 1e-9) * 100
        close_position = (last_close - last_low) / max((last_high - last_low), 1e-9)

        tf = str(timeframe or "").lower()
        if tf in {"1m", "5m", "15m", "30m"}:
            min_1_rise, min_3_rise, hard_1_rise, hard_3_rise = 0.8, 1.6, 1.5, 2.8
        elif tf == "1h":
            min_1_rise, min_3_rise, hard_1_rise, hard_3_rise = 1.0, 2.0, 1.8, 3.5
        elif tf == "4h":
            min_1_rise, min_3_rise, hard_1_rise, hard_3_rise = 1.4, 2.8, 2.3, 4.5
        else:
            min_1_rise, min_3_rise, hard_1_rise, hard_3_rise = 2.0, 4.0, 3.0, 6.0

        price_rise_confirmed = (one_rise >= min_1_rise) or (three_rise >= min_3_rise)
        hard_price_rise = (one_rise >= hard_1_rise) or (three_rise >= hard_3_rise)

        resistance_10 = float(high.iloc[-11:-1].max()) if len(high) >= 12 else float(high.iloc[:-1].max())
        resistance_20 = float(high.iloc[-21:-1].max()) if len(high) >= 22 else resistance_10
        resistance_50 = float(high.iloc[-51:-1].max()) if len(high) >= 52 else resistance_20
        nearest_resistance = min(r for r in [resistance_10, resistance_20, resistance_50] if pd.notna(r))

        break_10 = last_close > resistance_10
        break_20 = last_close > resistance_20
        break_50 = last_close > resistance_50
        resistance_break = break_10 or break_20 or break_50

        sma5 = close.rolling(5, min_periods=3).mean()
        sma10 = close.rolling(10, min_periods=5).mean()
        sma20 = close.rolling(20, min_periods=10).mean()
        above_sma5 = pd.notna(sma5.iloc[-1]) and last_close > float(sma5.iloc[-1])
        above_sma10 = pd.notna(sma10.iloc[-1]) and last_close > float(sma10.iloc[-1])
        above_sma20 = pd.notna(sma20.iloc[-1]) and last_close > float(sma20.iloc[-1])
        fresh_sma20_break = pd.notna(sma20.iloc[-1]) and pd.notna(sma20.iloc[-2]) and close.iloc[-2] <= sma20.iloc[-2] and last_close > sma20.iloc[-1]
        ma_breakout = fresh_sma20_break or (above_sma5 and above_sma10 and above_sma20)

        big_green = body_pct > 0 and body_pct >= min_1_rise and close_position >= 0.65
        impulse_close = body_pct > 0 and close_position >= 0.80 and range_pct >= min_1_rise * 1.5

        vol_ratio = None
        volume_spike = False
        if volume.notna().sum() >= 12 and pd.notna(volume.iloc[-1]):
            avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 22 else float(volume.iloc[:-1].mean())
            if avg_vol and avg_vol > 0:
                vol_ratio = float(volume.iloc[-1]) / avg_vol
                volume_spike = vol_ratio >= 1.35

        prev_close_series = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close_series).abs(),
            (low - prev_close_series).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean()
        atr_pct = float(atr14.iloc[-1] / max(abs(last_close), 1e-9) * 100) if pd.notna(atr14.iloc[-1]) else None
        atr_impulse = atr_pct is not None and (abs(one_rise) >= max(atr_pct * 1.15, min_1_rise) or abs(three_rise) >= atr_pct * 2.0)

        rsi_strong = False
        rsi_surge = False
        if last_rsi is not None:
            rsi_strong = last_rsi >= 55
            rsi_surge = prev_rsi is not None and (last_rsi - prev_rsi) >= 8

        structural_confirmations = [resistance_break, ma_breakout, big_green, impulse_close, atr_impulse]
        secondary_confirmations = [volume_spike, rsi_strong, rsi_surge, five_rise >= min_3_rise]

        structural_count = sum(bool(x) for x in structural_confirmations)
        secondary_count = sum(bool(x) for x in secondary_confirmations)

        confirmed_fast_up = bool(
            price_rise_confirmed and (
                structural_count >= 1 or
                (hard_price_rise and secondary_count >= 1)
            )
        )

        score = 0
        reasons = []
        why_not = []

        if one_rise >= hard_1_rise:
            score += 22; reasons.append(f"hard 1-candle rise {one_rise:.2f}%")
        elif one_rise >= min_1_rise:
            score += 14; reasons.append(f"1-candle rise {one_rise:.2f}%")
        else:
            why_not.append(f"1-candle rise not enough ({one_rise:.2f}%)")

        if three_rise >= hard_3_rise:
            score += 22; reasons.append(f"hard 3-candle rise {three_rise:.2f}%")
        elif three_rise >= min_3_rise:
            score += 14; reasons.append(f"3-candle rise {three_rise:.2f}%")
        else:
            why_not.append(f"3-candle rise not enough ({three_rise:.2f}%)")

        if five_rise >= min_3_rise:
            score += 8; reasons.append(f"5-candle upside pressure {five_rise:.2f}%")

        if resistance_break:
            if break_50:
                score += 22; reasons.append(f"broke 50-candle resistance {resistance_50:.2f}")
            elif break_20:
                score += 18; reasons.append(f"broke 20-candle resistance {resistance_20:.2f}")
            else:
                score += 12; reasons.append(f"broke 10-candle resistance {resistance_10:.2f}")
        else:
            why_not.append("no resistance breakout")

        if ma_breakout:
            score += 12; reasons.append("moving-average breakout confirmed")
        else:
            why_not.append("no strong MA breakout")

        if big_green or impulse_close:
            score += 12; reasons.append(f"bullish candle closed near high; body {body_pct:.2f}%")
        else:
            why_not.append("last candle is not a strong bullish close")

        if volume_spike:
            score += 8; reasons.append(f"volume confirms buying {vol_ratio:.2f}x avg")
        if atr_impulse:
            score += 8; reasons.append(f"rise exceeds volatility pressure; ATR {atr_pct:.2f}%")
        if rsi_strong:
            score += 5; reasons.append(f"RSI strong {last_rsi:.1f}")
        if rsi_surge:
            score += 5; reasons.append(f"RSI rose fast {prev_rsi:.1f} → {last_rsi:.1f}")

        if not price_rise_confirmed:
            confirmed_fast_up = False
            why_not.insert(0, "price rise requirement not met")
        if price_rise_confirmed and structural_count == 0 and not (hard_price_rise and secondary_count >= 1):
            confirmed_fast_up = False
            why_not.insert(0, "price rose, but no breakout confirmation")

        if confirmed_fast_up:
            if score >= 75 or (hard_price_rise and resistance_break):
                severity, flag, action = "CRITICAL", "YES", "Confirmed fast up: check breakout strength; avoid chasing too far above support."
            elif score >= 55:
                severity, flag, action = "HIGH", "YES", "Confirmed fast up: watch pullback/retest or planned entry."
            else:
                severity, flag, action = "MEDIUM", "YES", "Confirmed upside momentum; monitor next candle."
        else:
            severity, flag, action = "Normal", "No", "No confirmed fast-up alert."

        alert.update({
            "Alert": flag,
            "Confirmed Fast Up": "Yes" if confirmed_fast_up else "No",
            "Severity": severity,
            "Score": int(min(score, 100)),
            "Current Price": round(last_close, 4),
            "Last Price": round(last_close, 4),
            "1 Candle Rise %": round(one_rise, 2),
            "3 Candle Rise %": round(three_rise, 2),
            "5 Candle Rise %": round(five_rise, 2),
            "RSI": round(last_rsi, 2) if last_rsi is not None else None,
            "Volume Ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "ATR %": round(atr_pct, 2) if atr_pct is not None else None,
            "Nearest Resistance": round(nearest_resistance, 4),
            "Price Rise Confirmed": "Yes" if price_rise_confirmed else "No",
            "Resistance Break": "Yes" if resistance_break else "No",
            "MA Breakout": "Yes" if ma_breakout else "No",
            "Volume Spike": "Yes" if volume_spike else "No",
            "ATR Impulse": "Yes" if atr_impulse else "No",
            "RSI Strong": "Yes" if rsi_strong else "No",
            "Big Green Candle": "Yes" if (big_green or impulse_close) else "No",
            "Reason": "; ".join(reasons) if reasons else "No confirmed fast-up condition.",
            "Why Not Alert": "; ".join(dict.fromkeys(why_not)) if why_not else "",
            "Action Hint": action,
        })
        return alert
    except Exception as exc:
        alert["Reason"] = f"Alert check failed: {exc}"
        alert["Why Not Alert"] = f"Exception: {exc}"
        return alert


def _scan_fast_down_alerts_for_symbols(symbols: list[str], *, universe_name: str, data_source: str, timeframe: str, period: str, dps_mode: str, max_symbols: int = 0, alert_mode: str = "Both"):
    rows, failures, diagnostics = [], [], []
    scan_symbols = symbols[:max_symbols] if max_symbols and max_symbols > 0 else symbols
    total = len(scan_symbols)
    progress = st.progress(0, text="Preparing fast up/down alert scan...")
    status = st.empty()

    mode = str(alert_mode or "Both")
    scan_down = mode in {"Both", "Fast Down Only"}
    scan_up = mode in {"Both", "Fast Up Only"}

    for i, symbol in enumerate(scan_symbols, start=1):
        symbol = str(symbol).strip().upper()
        if not symbol:
            continue
        progress.progress(int(i / max(total, 1) * 100), text=f"Fast up/down alert scan {i}/{total}: {symbol}")
        status.caption(f"Checking {symbol} on {timeframe}")
        try:
            if data_source == "Yahoo Finance PSX (.KA)":
                df = _cached_yahoo_ohlcv(symbol, period=period, interval=timeframe)
            else:
                df = _cached_dps_ohlcv(symbol, mode=dps_mode)

            if scan_down:
                row = _fast_down_alert_from_chart(df, symbol=symbol, timeframe=timeframe)
                row["Alert Type"] = "FAST DOWN"
                row["Direction"] = "DOWN"
                row["Universe"] = universe_name
                row["Timeframe"] = timeframe
                row["_df"] = df
                if row.get("Alert") == "YES" and row.get("Confirmed Fast Down") == "Yes":
                    rows.append(row)
                else:
                    diagnostics.append({k: v for k, v in row.items() if k != "_df"})

            if scan_up:
                row = _fast_up_alert_from_chart(df, symbol=symbol, timeframe=timeframe)
                row["Alert Type"] = "FAST UP"
                row["Direction"] = "UP"
                row["Universe"] = universe_name
                row["Timeframe"] = timeframe
                row["_df"] = df
                if row.get("Alert") == "YES" and row.get("Confirmed Fast Up") == "Yes":
                    rows.append(row)
                else:
                    diagnostics.append({k: v for k, v in row.items() if k != "_df"})

        except Exception as exc:
            failures.append({"Symbol": symbol, "Timeframe": timeframe, "Error": str(exc)})

    progress.empty()
    status.empty()
    severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "WATCH": 1, "Normal": 0}
    rows.sort(key=lambda r: (severity_rank.get(str(r.get("Severity", "")), 0), int(r.get("Score", 0))), reverse=True)
    diagnostics.sort(key=lambda r: int(r.get("Score", 0) or 0), reverse=True)
    st.session_state["fastdown_diagnostics"] = diagnostics[:300]
    return rows, failures


def _fastdown_message_history_path() -> Path:
    p = Path(".alert_state")
    p.mkdir(exist_ok=True)
    return p / "fastdown_whatsapp_message_history.txt"


def _load_fastdown_message_history() -> str:
    try:
        p = _fastdown_message_history_path()
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _save_fastdown_message_history(text_value: str) -> None:
    try:
        _fastdown_message_history_path().write_text(str(text_value or ""), encoding="utf-8")
    except Exception:
        pass


def _append_fastdown_message_history(new_message: str, *, max_chars: int = 0) -> str:
    """Append a scan message to persistent WhatsApp query history.

    max_chars=0 means unlimited history.
    """
    old = _load_fastdown_message_history().strip()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = f"\n\n===== FAST UP/DOWN SCAN {stamp} =====\n{str(new_message or '').strip()}\n"
    combined = (old + block).strip() if old else block.strip()
    if max_chars and max_chars > 0 and len(combined) > int(max_chars):
        combined = combined[-int(max_chars):]
    _save_fastdown_message_history(combined)
    return combined


def _build_fastdown_whatsapp_query(rows: list[dict], max_items: int = 8) -> str:
    """Build a compact WhatsApp message from fast up/down scan results."""
    if not rows:
        return "✅ PSX FAST UP/DOWN SCAN\nNo urgent fast-up or fast up/down alerts found in the latest scan."

    up_count = sum(1 for r in rows if str(r.get("Alert Type")) == "FAST UP")
    down_count = sum(1 for r in rows if str(r.get("Alert Type")) == "FAST DOWN")

    lines = [
        "🚨 PSX FAST UP/DOWN ALERT",
        f"Total alerts: {len(rows)} | Fast Up: {up_count} | Fast Down: {down_count}",
        "",
    ]
    for r in rows[: int(max_items)]:
        alert_type = r.get("Alert Type", "FAST ALERT")
        icon = "🟢" if alert_type == "FAST UP" else "🔴"
        if alert_type == "FAST UP":
            move_line = f"  Current Price: {r.get('Current Price', r.get('Last Price'))} | 1C: +{r.get('1 Candle Rise %')}% | 3C: +{r.get('3 Candle Rise %')}% | RSI: {r.get('RSI')}"
        else:
            move_line = f"  Current Price: {r.get('Current Price', r.get('Last Price'))} | 1C: {r.get('1 Candle Drop %')}% | 3C: {r.get('3 Candle Drop %')}% | RSI: {r.get('RSI')}"
        lines.extend([
            f"{icon} {r.get('Symbol')} | {alert_type} | {r.get('Severity')} | Score {r.get('Score')} | TF {r.get('Timeframe')}",
            move_line,
            f"  Reason: {r.get('Reason')}",
            f"  Action: {r.get('Action Hint', 'Check chart immediately')}",
            "",
        ])
    if len(rows) > int(max_items):
        lines.append(f"+ {len(rows) - int(max_items)} more alert(s). Open bot for full table.")
    lines.append("Action: Check chart immediately and decide entry / hold / reduce / wait.")
    return "\n".join(lines)



def fast_down_alert_scanner_panel():
    st.subheader("🚨 Fast Up & Fast Down Alert Scanner")
    st.caption("Auto scan every 30 seconds inside the Fast Up/Down scan block only. No full page/browser refresh.")

    a1, a2, a3, a4, a5 = st.columns(5)
    universe = a1.selectbox("Alert universe", ["Selected Symbols", "KSE-100 Constituents", "KMI-30 Constituents", "All PSX Listed Symbols"], index=0, key="fastdown_universe")
    alert_mode = a2.selectbox("Alert type", ["Both", "Fast Up Only", "Fast Down Only"], index=0, key="fastupdown_mode")
    timeframe = a3.selectbox("Alert timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=2, key="fastdown_tf")
    data_source = a4.selectbox("Alert data source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="fastdown_source")
    max_symbols = a5.number_input("Max symbols", min_value=0, value=100, step=25, key="fastdown_max")

    symbols_text = ""
    if universe == "Selected Symbols":
        symbols_text = st.text_area("Selected symbols to watch", value="ATLH, NBP, SYS, MARI, FFC, SAZEW, NATF", key="fastdown_symbols")

    b1, b2, b3, b4 = st.columns(4)
    period = b1.selectbox("Yahoo period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="fastdown_period")
    dps_mode = b2.selectbox("DPS mode", ["daily", "intraday"], index=0, key="fastdown_dps")
    auto_rescan = b3.checkbox("Auto re-scan every 30 sec", value=False, key="fastdown_auto_rescan")
    auto_send_whatsapp = b4.checkbox("Auto-send WhatsApp on alert", value=False, key="fastdown_auto_send_wa")

    st.info("For immediate action use 1H or 4H. Select Both to scan Fast Up and Fast Down together. The WhatsApp query box updates after every scan.")

    def _run_fastdown_scan_and_prepare_message(send_auto: bool = False):
        symbols = _resolve_latest_divergence_symbols(universe, symbols_text)
        rows, failures = _scan_fast_down_alerts_for_symbols(
            symbols,
            universe_name=universe,
            data_source=data_source,
            timeframe=timeframe,
            period=period,
            dps_mode=dps_mode,
            max_symbols=int(max_symbols),
            alert_mode=alert_mode,
        )
        st.session_state["fastdown_rows"] = rows
        st.session_state["fastdown_failures"] = failures
        st.session_state["fastdown_last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        latest_message = _build_fastdown_whatsapp_query(rows)
        # Append every scan message to unlimited WhatsApp query/message history.
        history_text = _append_fastdown_message_history(latest_message, max_chars=0)
        st.session_state["fastdown_whatsapp_query"] = history_text
        st.session_state["fastdown_latest_message"] = latest_message

        send_logs = []
        if send_auto and rows:
            message = st.session_state.get("fastdown_latest_message", "") if st.session_state.get("fastdown_send_latest_only", False) else st.session_state.get("fastdown_whatsapp_query", "")
            provider = st.session_state.get("fastdown_wa_provider", "Disabled / Test only")
            phone = st.session_state.get("fastdown_wa_phone", "")
            api_key = st.session_state.get("fastdown_wa_callmebot_key", "")
            account_sid = st.session_state.get("fastdown_wa_twilio_sid", "")
            auth_token = st.session_state.get("fastdown_wa_twilio_token", "")
            from_number = st.session_state.get("fastdown_wa_twilio_from", "")
            webhook_url = st.session_state.get("fastdown_wa_webhook_url", "")

            # One combined message per scan, with cooldown key.
            cooldown_key = f"FASTDOWN_COMBINED|{universe}|{timeframe}"
            state = _load_whatsapp_alert_state()
            now = time.time()
            last_sent = float(state.get(cooldown_key, 0) or 0)
            # Avoid duplicate WhatsApp spam: minimum 10 minutes for same universe/timeframe combined alert.
            if last_sent and (now - last_sent) < 600:
                send_logs.append({"Sent": False, "Info": "Combined alert cooldown active for 10 minutes."})
            else:
                ok, info = _send_whatsapp_alert_message(
                    message,
                    provider=provider,
                    phone=phone,
                    api_key=api_key,
                    account_sid=account_sid,
                    auth_token=auth_token,
                    from_number=from_number,
                    webhook_url=webhook_url,
                )
                send_logs.append({"Sent": ok, "Info": info})
                if ok:
                    state[cooldown_key] = now
                    _save_whatsapp_alert_state(state)
        st.session_state["fastdown_wa_send_logs"] = send_logs
        return rows, failures

    manual_scan = st.button("Run Fast Down Alert Scan Now", type="primary", use_container_width=True, key="fastdown_run")
    if manual_scan:
        _run_fastdown_scan_and_prepare_message(auto_send_whatsapp)

    if auto_rescan:
        if hasattr(st, "fragment"):
            st.success("Auto scan is ON: only the Fast Up/Down scan block reruns every 30 seconds. The full page/browser is not refreshed.")

            @st.fragment(run_every="30s")
            def _fastdown_auto_scan_fragment():
                st.markdown("#### Auto Fast Down Scan Block")
                last_auto = float(st.session_state.get("fastdown_last_auto_run", 0.0) or 0.0)
                now = time.time()

                # Fragment itself reruns every 30 seconds. This guard avoids accidental double-scan.
                if now - last_auto >= 25:
                    st.session_state["fastdown_last_auto_run"] = now
                    with st.spinner("Running Fast Up/Down scan only..."):
                        auto_rows, auto_failures = _run_fastdown_scan_and_prepare_message(auto_send_whatsapp)
                    st.caption(f"Auto Fast Up/Down scan completed at {st.session_state.get('fastdown_last_scan_time', '')}. Alerts: {len(auto_rows)} | Failed/unavailable: {len(auto_failures)}")
                else:
                    st.caption(f"Waiting for next 30-second Fast Up/Down scan. Last scan: {st.session_state.get('fastdown_last_scan_time', 'Not yet')}")

                live_rows = st.session_state.get("fastdown_rows", [])
                if live_rows:
                    live_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in live_rows])
                    preferred_cols = [c for c in [
                        "Symbol", "Alert Type", "Direction", "Severity", "Score", "Current Price", "Timeframe",
                        "1 Candle Rise %", "3 Candle Rise %", "1 Candle Drop %", "3 Candle Drop %",
                        "RSI", "Volume Ratio", "Reason", "Action Hint"
                    ] if c in live_df.columns]
                    if preferred_cols:
                        live_df = live_df[preferred_cols + [c for c in live_df.columns if c not in preferred_cols]]
                    st.error(f"Live auto-scan fast up/down alerts: {len(live_df)}")
                    st.dataframe(live_df, use_container_width=True, hide_index=True)
                else:
                    st.success("Live auto-scan: no fast up/down alerts in the latest scan.")

            _fastdown_auto_scan_fragment()
        else:
            st.warning("Your Streamlit version does not support fragment-only reruns. Auto page refresh has been disabled to avoid full-page refresh. Use the manual scan button, or upgrade Streamlit to use 30-second scan-block auto mode.")

    rows = st.session_state.get("fastdown_rows", [])
    failures = st.session_state.get("fastdown_failures", [])
    last_scan = st.session_state.get("fastdown_last_scan_time", "")

    if last_scan:
        st.caption(f"Last fast-down scan: {last_scan}")

    if rows:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])
        st.error(f"Fast-down alerts found: {len(df)}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download Fast Down Alerts", data=df.to_csv(index=False).encode("utf-8"), file_name="psx_fast_down_alerts.csv", mime="text/csv", use_container_width=True, key="fastdown_download")
    else:
        st.success("No fast up/down alerts currently shown. Run the scan or adjust universe/timeframe settings.")

    diagnostics = st.session_state.get("fastdown_diagnostics", [])
    if diagnostics:
        with st.container(border=True):
            st.markdown("### Why symbols were NOT marked Fast Down")
            diag_df = pd.DataFrame(diagnostics)
            show_cols = [c for c in [
                "Symbol", "Alert Type", "Confirmed Fast Up", "Confirmed Fast Down", "Score", "Current Price", "Last Price",
                "1 Candle Rise %", "3 Candle Rise %", "1 Candle Drop %", "3 Candle Drop %",
                "Resistance Break", "Support Break", "MA Breakout", "MA Breakdown", "Volume Spike",
                "ATR Impulse", "ATR Shock", "RSI", "Why Not Alert", "Reason"
            ] if c in diag_df.columns]
            st.dataframe(diag_df[show_cols] if show_cols else diag_df, use_container_width=True, hide_index=True)

    st.markdown("### WhatsApp Query / Message Box — Unlimited History")
    st.caption("Every scan appends a new message here. Old messages are not replaced. WhatsApp sends exactly the content inside this box.")

    # Load persistent history when page opens.
    if "fastdown_whatsapp_query" not in st.session_state:
        saved_history = _load_fastdown_message_history()
        st.session_state["fastdown_whatsapp_query"] = saved_history if saved_history else "Run scan to generate WhatsApp alert message."

    h1, h2, h3 = st.columns(3)
    if h1.button("Reload Saved Message History", use_container_width=True, key="fastdown_reload_history"):
        st.session_state["fastdown_whatsapp_query"] = _load_fastdown_message_history() or "No saved history yet."
    if h2.button("Clear Message Box + Saved History", use_container_width=True, key="fastdown_clear_history"):
        st.session_state["fastdown_whatsapp_query"] = ""
        _save_fastdown_message_history("")
    include_latest_only = h3.checkbox("Send latest scan only", value=False, key="fastdown_send_latest_only")

    box_value = st.session_state.get("fastdown_latest_message", "") if include_latest_only else st.session_state.get("fastdown_whatsapp_query", "")
    wa_query = st.text_area(
        "WhatsApp messages / query history",
        value=box_value,
        height=360,
        key="fastdown_whatsapp_query_box",
        help="Unlimited scan messages are appended here. You can edit before sending.",
    )

    if include_latest_only:
        st.session_state["fastdown_latest_message"] = wa_query
    else:
        st.session_state["fastdown_whatsapp_query"] = wa_query
        _save_fastdown_message_history(wa_query)

    st.download_button(
        "Download WhatsApp Message History",
        data=str(st.session_state.get("fastdown_whatsapp_query", "")).encode("utf-8"),
        file_name="fastdown_whatsapp_message_history.txt",
        mime="text/plain",
        use_container_width=True,
        key="fastdown_download_message_history",
    )

    st.markdown("### WhatsApp Send Settings")
    w1, w2, w3 = st.columns(3)
    provider = w1.selectbox("WhatsApp provider", ["Disabled / Test only", "CallMeBot", "Twilio WhatsApp", "Custom Webhook"], index=0, key="fastdown_wa_provider")
    phone = w2.text_input("WhatsApp number", value="", key="fastdown_wa_phone", help="Use country code, e.g. +923001234567")
    api_key = w3.text_input("CallMeBot API key", value="", type="password", key="fastdown_wa_callmebot_key")

    with st.container(border=True):
        st.markdown("### Twilio / Custom Webhook advanced settings")
        t1, t2 = st.columns(2)
        t1.text_input("Twilio Account SID", value="", key="fastdown_wa_twilio_sid")
        t2.text_input("Twilio Auth Token", value="", type="password", key="fastdown_wa_twilio_token")
        t3, t4 = st.columns(2)
        t3.text_input("Twilio From WhatsApp number", value="", key="fastdown_wa_twilio_from", help="Example: whatsapp:+14155238886")
        t4.text_input("Custom webhook URL", value="", key="fastdown_wa_webhook_url")

    send_now = st.button("Send WhatsApp Message From Query Box", use_container_width=True, key="fastdown_send_wa_now")
    if send_now:
        message_to_send = st.session_state.get("fastdown_latest_message", "") if st.session_state.get("fastdown_send_latest_only", False) else st.session_state.get("fastdown_whatsapp_query", "")
        ok, info = _send_whatsapp_alert_message(
            message_to_send,
            provider=provider,
            phone=phone,
            api_key=api_key,
            account_sid=st.session_state.get("fastdown_wa_twilio_sid", ""),
            auth_token=st.session_state.get("fastdown_wa_twilio_token", ""),
            from_number=st.session_state.get("fastdown_wa_twilio_from", ""),
            webhook_url=st.session_state.get("fastdown_wa_webhook_url", ""),
        )
        st.session_state["fastdown_wa_send_logs"] = [{"Sent": ok, "Info": info}]

    send_logs = st.session_state.get("fastdown_wa_send_logs", [])
    if send_logs:
        with st.container(border=True):
            st.markdown("### WhatsApp send log")
            st.dataframe(pd.DataFrame(send_logs), use_container_width=True, hide_index=True)

    if rows:
        labels = [f"{r.get('Symbol')} | {r.get('Severity')} | Score {r.get('Score')} | {r.get('Timeframe')}" for r in rows]
        selected = st.selectbox("Open alert chart", labels, key="fastdown_chart_select")
        idx = labels.index(selected) if selected in labels else 0
        row = rows[idx]
        chart_df = row.get("_df")
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            st.warning(f"{row.get('Symbol')} alert reason: {row.get('Reason')}")
            render_chart_engine(
                chart_df,
                {"signal": {"action": "FAST DOWN ALERT", "confidence": row.get("Score", 0)}, "trade_plan": {}},
                symbol=row.get("Symbol", ""),
                title=f"{row.get('Symbol')} Fast Down Alert | {row.get('Timeframe')}",
                key_prefix=f"fastdown_{idx}_{row.get('Symbol')}",
            )

    if failures:
        with st.container(border=True):
            st.markdown("### Unavailable / failed symbols (len(failures))")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)



def latest_divergence_scanner_desk_panel() -> None:
    st.subheader("Latest Divergence Scanner")
    st.success("Direct page only. Keep this open when using latest divergence scan. Do not open duplicate scanner inside another desk.")
    divergence_finder_panel()



def _alert_state_path() -> Path:
    p = Path(".alert_state")
    p.mkdir(exist_ok=True)
    return p / "whatsapp_portfolio_alerts.json"


def _load_whatsapp_alert_state() -> dict:
    try:
        p = _alert_state_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_whatsapp_alert_state(state: dict) -> None:
    try:
        _alert_state_path().write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _send_whatsapp_alert_message(message: str, *, provider: str, phone: str = "", api_key: str = "", account_sid: str = "", auth_token: str = "", from_number: str = "", webhook_url: str = "") -> tuple[bool, str]:
    """Send WhatsApp alert via supported providers.

    Providers:
    - CallMeBot: simple WhatsApp API. Needs phone and API key.
    - Twilio WhatsApp: needs SID, token, from, to.
    - Custom Webhook: posts JSON {'message': message, 'phone': phone}.
    """
    try:
        provider = str(provider or "").strip()
        if provider == "Disabled / Test only":
            return False, "Provider disabled. Message was not sent."

        if provider == "CallMeBot":
            if not phone or not api_key:
                return False, "CallMeBot requires phone and API key."
            url = "https://api.callmebot.com/whatsapp.php"
            params = {"phone": phone, "text": message, "apikey": api_key}
            resp = requests.get(url, params=params, timeout=15)
            ok = 200 <= resp.status_code < 300
            return ok, f"CallMeBot status {resp.status_code}: {resp.text[:160]}"

        if provider == "Twilio WhatsApp":
            if not (account_sid and auth_token and from_number and phone):
                return False, "Twilio requires Account SID, Auth Token, From WhatsApp number, and To phone."
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
            data = {
                "From": from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}",
                "To": phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}",
                "Body": message,
            }
            resp = requests.post(url, data=data, auth=(account_sid, auth_token), timeout=20)
            ok = 200 <= resp.status_code < 300
            return ok, f"Twilio status {resp.status_code}: {resp.text[:160]}"

        if provider == "Custom Webhook":
            if not webhook_url:
                return False, "Custom webhook requires URL."
            resp = requests.post(webhook_url, json={"phone": phone, "message": message}, timeout=20)
            ok = 200 <= resp.status_code < 300
            return ok, f"Webhook status {resp.status_code}: {resp.text[:160]}"

        return False, f"Unknown provider: {provider}"
    except Exception as exc:
        return False, f"WhatsApp send failed: {exc}"


def _portfolio_alert_message(row: dict) -> str:
    symbol = row.get("Symbol", "")
    sev = row.get("Severity", "")
    score = row.get("Score", "")
    tf = row.get("Timeframe", "")
    last_price = row.get("Last Price", "")
    one_drop = row.get("1 Candle Drop %", "")
    three_drop = row.get("3 Candle Drop %", "")
    rsi = row.get("RSI", "")
    reason = row.get("Reason", "")
    return (
        f"🚨 PSX PORTFOLIO ALERT\\n"
        f"Symbol: {symbol}\\n"
        f"Severity: {sev} | Score: {score}\\n"
        f"Timeframe: {tf}\\n"
        f"Last Price: {last_price}\\n"
        f"1-candle drop: {one_drop}%\\n"
        f"3-candle drop: {three_drop}%\\n"
        f"RSI: {rsi}\\n"
        f"Reason: {reason}\\n"
        f"Action: Open bot chart and decide quickly: hold / reduce / exit / wait."
    )


def _should_send_portfolio_alert(row: dict, cooldown_minutes: int, min_severity: str) -> tuple[bool, str]:
    severity_order = {"WATCH": 1, "HIGH": 2, "CRITICAL": 3}
    row_sev = str(row.get("Severity", "Normal")).upper()
    min_sev = str(min_severity or "HIGH").upper()
    if severity_order.get(row_sev, 0) < severity_order.get(min_sev, 2):
        return False, f"Severity {row_sev} below minimum {min_sev}."

    state = _load_whatsapp_alert_state()
    key = f"{row.get('Symbol')}|{row.get('Timeframe')}|{row.get('Severity')}"
    now = time.time()
    last_sent = float(state.get(key, 0) or 0)
    if last_sent and (now - last_sent) < int(cooldown_minutes) * 60:
        mins_left = int((int(cooldown_minutes) * 60 - (now - last_sent)) / 60)
        return False, f"Cooldown active for {key}, about {mins_left} minutes left."
    return True, key


def _mark_portfolio_alert_sent(key: str) -> None:
    state = _load_whatsapp_alert_state()
    state[key] = time.time()
    _save_whatsapp_alert_state(state)


def portfolio_whatsapp_alert_watcher_panel():
    st.subheader("📲 Portfolio WhatsApp Alert Watcher")
    st.caption("Auto-scan your portfolio and send WhatsApp alerts when urgent fast-down / crash conditions appear. Keep the bot running for automatic alerts.")

    st.warning("WhatsApp auto-sending requires an API provider. Use CallMeBot, Twilio WhatsApp, or your own webhook. Normal WhatsApp cannot be automated without an API.")

    p1, p2, p3, p4 = st.columns(4)
    scan_universe = p1.selectbox(
        "Portfolio source",
        ["Selected Portfolio Symbols", "KSE-100 Constituents", "KMI-30 Constituents", "All PSX Listed Symbols"],
        index=0,
        key="wa_portfolio_source",
    )
    timeframe = p2.selectbox("Alert timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"], index=0, key="wa_alert_tf")
    data_source = p3.selectbox("Data source", ["Yahoo Finance PSX (.KA)", "Experimental DPS Chart-Series Loader"], key="wa_alert_data_source")
    max_symbols = p4.number_input("Max symbols", min_value=0, value=75, step=25, key="wa_alert_max_symbols")

    symbols_text = ""
    if scan_universe == "Selected Portfolio Symbols":
        symbols_text = st.text_area(
            "Paste your portfolio symbols",
            value="ATLH, NBP, SYS, MARI, FFC, SAZEW, NATF, UBL, MEBL, HUBC",
            key="wa_portfolio_symbols",
            help="Paste only your portfolio symbols separated by comma or new line.",
        )

    q1, q2, q3, q4 = st.columns(4)
    period = q1.selectbox("Yahoo period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1, key="wa_alert_period")
    dps_mode = q2.selectbox("DPS mode", ["daily", "intraday"], index=0, key="wa_alert_dps")
    min_severity = q3.selectbox("Send alert from severity", ["WATCH", "HIGH", "CRITICAL"], index=1, key="wa_min_severity")
    cooldown = q4.number_input("Cooldown minutes / symbol", min_value=5, max_value=1440, value=60, step=5, key="wa_alert_cooldown")

    st.markdown("### WhatsApp Sending Settings")
    w1, w2, w3 = st.columns(3)
    provider = w1.selectbox("WhatsApp provider", ["Disabled / Test only", "CallMeBot", "Twilio WhatsApp", "Custom Webhook"], index=0, key="wa_provider")
    phone = w2.text_input("Your WhatsApp number", value="", key="wa_phone", help="Use country code, e.g. +923001234567")
    api_key = w3.text_input("CallMeBot API key", value="", type="password", key="wa_callmebot_key")

    with st.container(border=True):
        st.markdown("### Twilio / Custom Webhook advanced settings")
        t1, t2 = st.columns(2)
        account_sid = t1.text_input("Twilio Account SID", value="", key="wa_twilio_sid")
        auth_token = t2.text_input("Twilio Auth Token", value="", type="password", key="wa_twilio_token")
        t3, t4 = st.columns(2)
        from_number = t3.text_input("Twilio From WhatsApp number", value="", key="wa_twilio_from", help="Example: whatsapp:+14155238886")
        webhook_url = t4.text_input("Custom webhook URL", value="", key="wa_webhook_url")

    a1, a2, a3 = st.columns(3)
    auto_enabled = a1.checkbox("Enable auto scan while this page is open", value=False, key="wa_auto_scan")
    interval_sec = a2.number_input("Auto scan interval seconds", min_value=60, max_value=3600, value=300, step=60, key="wa_interval")
    send_whatsapp = a3.checkbox("Send WhatsApp if alert found", value=False, key="wa_send_enabled")

    st.info("Recommended: timeframe 1H, minimum severity HIGH, cooldown 60 minutes. For full portfolio, paste your portfolio symbols only for faster alerts.")

    def _run_portfolio_alert_scan(send_enabled: bool):
        symbols = _resolve_latest_divergence_symbols(scan_universe, symbols_text)
        rows, failures = _scan_fast_down_alerts_for_symbols(
            symbols,
            universe_name=scan_universe,
            data_source=data_source,
            timeframe=timeframe,
            period=period,
            dps_mode=dps_mode,
            max_symbols=int(max_symbols),
        )
        send_logs = []
        if rows and send_enabled:
            for row in rows:
                should_send, key_or_reason = _should_send_portfolio_alert(row, int(cooldown), min_severity)
                if should_send:
                    msg = _portfolio_alert_message(row)
                    ok, info = _send_whatsapp_alert_message(
                        msg,
                        provider=provider,
                        phone=phone,
                        api_key=api_key,
                        account_sid=account_sid,
                        auth_token=auth_token,
                        from_number=from_number,
                        webhook_url=webhook_url,
                    )
                    send_logs.append({"Symbol": row.get("Symbol"), "Sent": ok, "Info": info})
                    if ok:
                        _mark_portfolio_alert_sent(key_or_reason)
                else:
                    send_logs.append({"Symbol": row.get("Symbol"), "Sent": False, "Info": key_or_reason})
        st.session_state["wa_alert_rows"] = rows
        st.session_state["wa_alert_failures"] = failures
        st.session_state["wa_alert_send_logs"] = send_logs
        st.session_state["wa_last_scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return rows, failures, send_logs

    manual_scan = st.button("Scan Portfolio Now", type="primary", use_container_width=True, key="wa_manual_scan")
    if manual_scan:
        _run_portfolio_alert_scan(send_whatsapp)

    # Auto scan: this works while page is open. It triggers a rerun by browser refresh.
    if auto_enabled:
        last_auto = st.session_state.get("wa_last_auto_run", 0.0)
        now = time.time()
        if now - float(last_auto or 0) >= int(interval_sec):
            st.session_state["wa_last_auto_run"] = now
            _run_portfolio_alert_scan(send_whatsapp)
        components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload();}}, {int(interval_sec) * 1000});</script>",
            height=0,
        )
        st.success(f"Auto scan is ON. This page will refresh every {int(interval_sec)} seconds while the bot is open.")

    rows = st.session_state.get("wa_alert_rows", [])
    failures = st.session_state.get("wa_alert_failures", [])
    send_logs = st.session_state.get("wa_alert_send_logs", [])
    last_scan = st.session_state.get("wa_last_scan_time", "")

    if last_scan:
        st.caption(f"Last portfolio alert scan: {last_scan}")

    if rows:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])
        st.error(f"Portfolio alerts requiring attention: {len(df)}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        labels = [f"{r.get('Symbol')} | {r.get('Severity')} | Score {r.get('Score')} | {r.get('Timeframe')}" for r in rows]
        selected = st.selectbox("Open portfolio alert chart", labels, key="wa_alert_chart_select")
        idx = labels.index(selected) if selected in labels else 0
        row = rows[idx]
        chart_df = row.get("_df")
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            st.warning(row.get("Reason", ""))
            render_chart_engine(
                chart_df,
                {"signal": {"action": "PORTFOLIO FAST DOWN ALERT", "confidence": row.get("Score", 0)}, "trade_plan": {}},
                symbol=row.get("Symbol", ""),
                title=f"{row.get('Symbol')} Portfolio WhatsApp Alert | {row.get('Timeframe')}",
                key_prefix=f"wa_alert_{idx}_{row.get('Symbol')}",
            )
    else:
        st.success("No portfolio fast up/down alerts currently shown.")

    if send_logs:
        with st.container(border=True):
            st.markdown("### WhatsApp send log")
            st.dataframe(pd.DataFrame(send_logs), use_container_width=True, hide_index=True)

    if failures:
        with st.container(border=True):
            st.markdown("### Unavailable / failed symbols (len(failures))")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)



def fast_down_alert_scanner_desk_panel() -> None:
    st.success("Direct page for Fast Down / Crash Alert + WhatsApp message history. Keep this page open for 30-second auto re-scan.")
    fast_down_alert_scanner_panel()


def portfolio_whatsapp_alert_watcher_desk_panel() -> None:
    portfolio_whatsapp_alert_watcher_panel()



def _chatgpt_decision_prompt(context_text: str, mode: str = "Fast Down Alert") -> str:
    return f"""
You are an expert Pakistan Stock Exchange technical and portfolio risk assistant.

User profile:
- PSX investor
- Conservative risk preference
- Wants quick action alerts
- Needs simple hold / reduce / exit / wait guidance
- Do not guarantee profit. Mention confirmation and risk clearly.

Task mode: {mode}

Analyze the following bot output and produce:
1) Quick Action: HOLD / WAIT / REDUCE / EXIT / WATCH
2) Urgency: LOW / MEDIUM / HIGH / CRITICAL
3) Main reason in simple English
4) What to check on chart before action
5) WhatsApp-ready alert message, concise

Bot output:
{context_text}
"""


def _run_chatgpt_decision(api_key: str, context_text: str, model: str = "gpt-4.1-mini", mode: str = "Fast Down Alert") -> tuple[bool, str]:
    try:
        if OpenAI is None:
            return False, "OpenAI package is not installed. Run: pip install openai"
        if not api_key:
            return False, "OpenAI API key missing. Enter your API key in the bot."
        if not str(context_text or "").strip():
            return False, "No bot result/context was provided to ChatGPT."

        client = OpenAI(api_key=api_key)
        prompt = _chatgpt_decision_prompt(context_text, mode=mode)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful PSX trading-risk assistant. Be concise, practical, and risk-aware."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=700,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        return True, content or "No response returned."
    except Exception as exc:
        return False, f"ChatGPT API failed: {exc}"


def _latest_fastdown_context_for_ai() -> str:
    rows = st.session_state.get("fastdown_rows", [])
    if rows:
        try:
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "_df"} for r in rows])
            return df.head(25).to_csv(index=False)
        except Exception:
            return str(rows[:10])
    return str(st.session_state.get("fastdown_whatsapp_query", "") or "")


def chatgpt_ai_decision_assistant_panel():
    st.subheader("🤖 ChatGPT AI Decision Assistant")
    st.caption("Use OpenAI API to explain bot results, rewrite WhatsApp alerts, and summarize quick action. API key is entered here and not hard-coded.")

    c1, c2, c3 = st.columns(3)
    api_key = c1.text_input("OpenAI API key", value="", type="password", key="ai_openai_api_key")
    model = c2.selectbox("AI model", ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"], index=0, key="ai_model")
    mode = c3.selectbox("AI task mode", ["Fast Down Alert", "Portfolio Quick Action", "Divergence Explanation", "WhatsApp Alert Rewrite", "General Bot Result"], index=0, key="ai_task_mode")

    source = st.selectbox(
        "Use bot data from",
        ["Latest Fast Down Alert Results", "WhatsApp Query Box", "Sarmaaya Full Bot Fundamentals", "Manual Paste"],
        index=0,
        key="ai_context_source",
    )

    pending_context = st.session_state.pop("ai_pending_context_text", "") if "ai_pending_context_text" in st.session_state else ""

    if pending_context:
        default_context = pending_context
    elif source == "Latest Fast Down Alert Results":
        default_context = _latest_fastdown_context_for_ai()
    elif source == "WhatsApp Query Box":
        default_context = st.session_state.get("fastdown_whatsapp_query", "")
    elif source == "Sarmaaya Full Bot Fundamentals":
        fdf = _load_sarmaaya_fill_table_anytime()
        default_context = fdf.head(200).to_csv(index=False) if isinstance(fdf, pd.DataFrame) and not fdf.empty else ""
    else:
        default_context = ""

    context_text = st.text_area(
        "Bot result/context to send to ChatGPT",
        value=default_context,
        height=260,
        key="ai_context_text",
        help="This is the data ChatGPT will analyze. You can edit it before sending.",
    )

    b1, b2 = st.columns(2)
    run_ai = b1.button("Generate AI Decision", type="primary", use_container_width=True, key="ai_generate_decision")
    copy_to_wa = b2.checkbox("Save AI output into WhatsApp Query Box", value=True, key="ai_save_to_wa")

    if run_ai:
        ok, answer = _run_chatgpt_decision(api_key, context_text, model=model, mode=mode)
        st.session_state["ai_decision_output"] = answer
        st.session_state["ai_decision_ok"] = ok
        if ok and copy_to_wa:
            old = st.session_state.get("fastdown_whatsapp_query", "")
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            combined = (str(old).strip() + f"\n\n===== CHATGPT AI DECISION {stamp} =====\n" + answer.strip()).strip()
            st.session_state["fastdown_whatsapp_query"] = combined
            try:
                _save_fastdown_message_history(combined)
            except Exception:
                pass

    output = st.session_state.get("ai_decision_output", "")
    if output:
        if st.session_state.get("ai_decision_ok"):
            st.success("AI decision generated.")
        else:
            st.error("AI decision failed.")
        st.text_area("ChatGPT AI output", value=output, height=320, key="ai_decision_output_box")
        st.download_button(
            "Download AI Decision",
            data=str(output).encode("utf-8"),
            file_name="chatgpt_ai_decision.txt",
            mime="text/plain",
            use_container_width=True,
            key="ai_decision_download",
        )

    st.info("Install requirement if needed: pip install openai. ChatGPT output supports your decision; it does not replace risk management.")



def chatgpt_ai_decision_assistant_desk_panel() -> None:
    chatgpt_ai_decision_assistant_panel()



def _clean_sarmaaya_number(value):
    try:
        if value is None:
            return None
        s = str(value).replace(",", "").replace("%", "").replace("PKR", "").replace("Rs.", "").replace("Rs", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None
    except Exception:
        return None


def _extract_first_after_label(text_blob: str, label: str):
    """Very tolerant extraction for public Sarmaaya page text."""
    try:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*([\-]?\d+(?:\.\d+)?%?)"
        m = re.search(pattern, text_blob, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _fetch_sarmaaya_stock_snapshot(symbol: str) -> dict:
    """Experimental public Sarmaaya stock-page snapshot fetch.

    This is intentionally limited to visible public page text. For premium/login data,
    use CSV/Excel/paste import instead of bypassing the website.
    """
    symbol = str(symbol or "").strip().upper().replace(".KA", "")
    row = {
        "Symbol": symbol,
        "Sarmaaya URL": f"https://sarmaaya.pk/stocks/{symbol}",
        "Sarmaaya Fetch Status": "Not fetched",
        "Sarmaaya Note": "",
    }
    if not symbol:
        row["Sarmaaya Fetch Status"] = "Skipped"
        row["Sarmaaya Note"] = "Empty symbol."
        return row

    try:
        url = f"https://sarmaaya.pk/stocks/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; PSXBot/1.0; +local Streamlit app)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        row["HTTP Status"] = resp.status_code
        if resp.status_code != 200:
            row["Sarmaaya Fetch Status"] = "Failed"
            row["Sarmaaya Note"] = f"HTTP {resp.status_code}"
            return row

        html = resp.text or ""
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, "html.parser")
            text_blob = soup.get_text(" ", strip=True)
        else:
            text_blob = re.sub(r"<[^>]+>", " ", html)
            text_blob = re.sub(r"\s+", " ", text_blob)

        row["Sarmaaya Fetch Status"] = "Fetched"
        row["Page Text Sample"] = text_blob[:500]

        # Common visible fields / labels on Sarmaaya stock pages.
        label_map = {
            "Current Price": "share price today is PKR",
            "Day High": "high price is PKR",
            "Day Low": "low price is PKR",
            "Volume": "Volume traded in",
            "Intrinsic Value": "Intrinsic value",
            "Margin of Safety": "Margin of Safety",
            "Sarmaaya Score": "Sarmaaya Score",
            "Growth Score": "Growth",
            "Stability Score": "Stability",
            "Valuation Score": "Valuation",
            "Cashflow Score": "Cashflow",
            "Inventory Score": "Inventory",
        }

        # Price: try direct sentence first
        m_price = re.search(r"share price today is\s*PKR\s*([0-9,.]+)", text_blob, flags=re.IGNORECASE)
        if m_price:
            row["Current Price"] = _clean_sarmaaya_number(m_price.group(1))

        # Generic label captures
        for out_col, label in label_map.items():
            if out_col in row and row[out_col] not in (None, ""):
                continue
            val = _extract_first_after_label(text_blob, label)
            if val is not None:
                row[out_col] = _clean_sarmaaya_number(val)

        # Try score formats like Growth 12/20
        for col, lbl in [
            ("Growth Score", "Growth"),
            ("Stability Score", "Stability"),
            ("Valuation Score", "Valuation"),
            ("Cashflow Score", "Cashflow"),
            ("Inventory Score", "Inventory"),
        ]:
            if col not in row or row.get(col) is None:
                m = re.search(rf"{lbl}\s+([0-9]+(?:\.\d+)?)\s*/\s*20", text_blob, flags=re.IGNORECASE)
                if m:
                    row[col] = float(m.group(1))

        if len([k for k in row.keys() if k not in {"Symbol", "Sarmaaya URL", "Sarmaaya Fetch Status", "Sarmaaya Note", "HTTP Status", "Page Text Sample"}]) == 0:
            row["Sarmaaya Note"] = "Fetched page, but structured values were not visible in HTML. Use CSV/Excel/paste import for full data."
        else:
            row["Sarmaaya Note"] = "Public page fields extracted where visible."
        return row
    except Exception as exc:
        row["Sarmaaya Fetch Status"] = "Failed"
        row["Sarmaaya Note"] = str(exc)
        return row


def _read_uploaded_sarmaaya_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = str(uploaded_file.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.DataFrame()


def _save_sarmaaya_data(df: pd.DataFrame, label: str = "Sarmaaya") -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    st.session_state["sarmaaya_imported_data"] = df.copy()
    st.session_state["sarmaaya_imported_label"] = label
    try:
        out_dir = Path(".alert_state")
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / "sarmaaya_imported_data.csv", index=False)
    except Exception:
        pass


def _load_saved_sarmaaya_data() -> pd.DataFrame:
    try:
        p = Path(".alert_state") / "sarmaaya_imported_data.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()



SARMAAYA_SCREENSHOT_SECTIONS = ["Main / MOS", "Growth", "Stability", "Valuation", "Inventory", "Cashflow"]


def _ocr_image_file(uploaded_file) -> str:
    """OCR a Sarmaaya screenshot if Tesseract is installed. Returns empty string if unavailable."""
    try:
        if uploaded_file is None:
            return ""
        if Image is None or pytesseract is None:
            return ""
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def _extract_score_from_text(text_blob: str, section: str):
    try:
        # Sarmaaya cards usually show 13/20, 20/20, etc.
        m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*20", str(text_blob))
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _extract_rating_counts_from_text(text_blob: str) -> dict:
    txt = str(text_blob or "").upper()
    return {
        "Good Count": len(re.findall(r"\bGOOD\b", txt)),
        "Average Count": len(re.findall(r"\bAVERAGE\b", txt)),
        "Bad Count": len(re.findall(r"\bBAD\b", txt)),
    }


def _extract_main_sarmaaya_fields(text_blob: str) -> dict:
    txt = str(text_blob or "")
    row = {}
    patterns = {
        "Current Price": r"share price today is\s*PKR\s*([0-9,.]+)|\b([0-9,.]+)\s*(?:\n|\s)+[-+]?\d",
        "Intrinsic Value": r"Intrinsic\s+value\s*[:\-]?\s*([0-9,.]+)",
        "Margin of Safety %": r"Margin\s+of\s+Safety\s*[:\-]?\s*([0-9,.]+)\s*%",
        "Sarmaaya Score": r"Sarmaaya\s+Score\s*[:\-]?\s*([0-9,.]+)",
        "Explosive Ratio": r"Explosive\s+Ratio\s*[:\-]?\s*([0-9,.]+)",
        "Sarmaaya Aggressive Rank": r"Sarmaaya\s+Aggressive\s+Rank\s*[:\-]?\s*([0-9,.]+)",
    }
    for key, pat in patterns.items():
        try:
            m = re.search(pat, txt, flags=re.IGNORECASE)
            if m:
                vals = [g for g in m.groups() if g]
                if vals:
                    row[key] = _clean_sarmaaya_number(vals[0])
        except Exception:
            pass
    return row


def _default_sarmaaya_manual_rows(symbol: str) -> pd.DataFrame:
    rows = [
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Current Price", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Intrinsic Value", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Margin of Safety %", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Sarmaaya Score", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Explosive Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Main / MOS", "Metric": "Sarmaaya Aggressive Rank", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Growth", "Metric": "Net Profit CAGR", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Growth", "Metric": "EPS", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Growth", "Metric": "OPG CAGR operating", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Growth", "Metric": "Revenue CAGR", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Operating Margins", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Net Margin", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Tax Rate", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Current Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Total Debt", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Debt to Equity", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Interest Coverage Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Stability", "Metric": "Cash Flow from Operation", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Price to Earnings", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "PEG Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Earning Yield", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Price to Book Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Graham Value", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Price to Sales", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "Dividend Yield", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Valuation", "Metric": "EV/EBITDA", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Inventory", "Metric": "Inventory Turnover Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Inventory", "Metric": "Day Receivable Outstanding", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Inventory", "Metric": "Day Sales of Inventory", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Inventory", "Metric": "Days Payable Outstanding", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Inventory", "Metric": "Cash Conversion Cycle", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Cashflow", "Metric": "Free Cash Flow per Share", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Cashflow", "Metric": "Free Cash Flow per Sale", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Cashflow", "Metric": "Free Cash Flow per CFO", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Cashflow", "Metric": "Cash Return on Invested Capital", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
        {"Symbol": symbol, "Section": "Cashflow", "Metric": "Cash to Debt Ratio", "TTM": "", "2025": "", "2024": "", "2023": "", "2022": "", "2021": "", "Rating": ""},
    ]
    return pd.DataFrame(rows)


def _build_sarmaaya_summary_from_screenshots(symbol: str, section_texts: dict) -> pd.DataFrame:
    rows = []
    for section, txt in section_texts.items():
        score = _extract_score_from_text(txt, section)
        counts = _extract_rating_counts_from_text(txt)
        rows.append({
            "Symbol": symbol,
            "Section": section,
            "Section Score /20": score,
            "Good Count": counts["Good Count"],
            "Average Count": counts["Average Count"],
            "Bad Count": counts["Bad Count"],
            "OCR Text Available": "Yes" if str(txt or "").strip() else "No",
        })
    main_fields = _extract_main_sarmaaya_fields(section_texts.get("Main / MOS", ""))
    if main_fields:
        for k, v in main_fields.items():
            rows.append({
                "Symbol": symbol,
                "Section": "Main / MOS",
                "Section Score /20": None,
                "Metric": k,
                "Value": v,
                "Good Count": None,
                "Average Count": None,
                "Bad Count": None,
                "OCR Text Available": "Yes",
            })
    return pd.DataFrame(rows)



def _parse_sarmaaya_quick_text(symbol: str, raw_text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse copied Sarmaaya page/table text into a raw lines table and a section summary.

    This avoids screenshots/OCR. User copies visible Sarmaaya text/tables and pastes once.
    """
    symbol = str(symbol or "").strip().upper()
    raw = str(raw_text or "").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    sections = ["Margin of Safety", "Main", "Growth", "Stability", "Valuation", "Inventory", "Cashflow", "Cash Flow"]
    current_section = "Unknown"
    parsed_rows = []
    section_scores = {}

    for line in lines:
        line_clean = re.sub(r"\s+", " ", line).strip()
        for sec in sections:
            if re.fullmatch(sec, line_clean, flags=re.IGNORECASE) or line_clean.lower().startswith(sec.lower()):
                current_section = "Cashflow" if sec.lower() == "cash flow" else sec
                # capture section score if in same line e.g. Growth 20/20
                score = _extract_score_from_text(line_clean, current_section)
                if score is not None:
                    section_scores[current_section] = score
                break

        # capture standalone scores like 13/20 under section
        score = _extract_score_from_text(line_clean, current_section)
        if score is not None and current_section not in section_scores:
            section_scores[current_section] = score

        rating = ""
        if re.search(r"\bGOOD\b", line_clean, flags=re.IGNORECASE):
            rating = "GOOD"
        elif re.search(r"\bBAD\b", line_clean, flags=re.IGNORECASE):
            rating = "BAD"
        elif re.search(r"\bAVERAGE\b", line_clean, flags=re.IGNORECASE):
            rating = "AVERAGE"

        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", line_clean)
        parsed_rows.append({
            "Symbol": symbol,
            "Section": current_section,
            "Raw Line": line_clean,
            "Numbers Found": ", ".join(nums),
            "Rating Found": rating,
        })

    raw_df = pd.DataFrame(parsed_rows)

    # Main/MOS fields from full text
    main_fields = _extract_main_sarmaaya_fields(raw)

    summary_rows = []
    for sec in ["Main / MOS", "Growth", "Stability", "Valuation", "Inventory", "Cashflow"]:
        sec_aliases = [sec]
        if sec == "Main / MOS":
            sec_aliases = ["Main", "Margin of Safety", "Main / MOS"]
        sec_lines = raw_df[raw_df["Section"].isin(sec_aliases)] if not raw_df.empty else pd.DataFrame()
        ratings_text = " ".join(sec_lines["Raw Line"].astype(str).tolist()) if not sec_lines.empty else ""
        counts = _extract_rating_counts_from_text(ratings_text)
        score = None
        for alias in sec_aliases:
            if alias in section_scores:
                score = section_scores[alias]
                break
        summary_rows.append({
            "Symbol": symbol,
            "Section": sec,
            "Section Score /20": score,
            "Good Count": counts.get("Good Count", 0),
            "Average Count": counts.get("Average Count", 0),
            "Bad Count": counts.get("Bad Count", 0),
            "Lines Parsed": len(sec_lines) if not sec_lines.empty else 0,
        })

    for k, v in main_fields.items():
        summary_rows.append({
            "Symbol": symbol,
            "Section": "Main / MOS",
            "Metric": k,
            "Value": v,
            "Section Score /20": None,
            "Good Count": None,
            "Average Count": None,
            "Bad Count": None,
            "Lines Parsed": None,
        })

    summary_df = pd.DataFrame(summary_rows)
    return raw_df, summary_df



SARMAAYA_TO_FUNDAMENTAL_MAP = {
    # Main / MOS
    "Current Price": "Current Price",
    "Intrinsic Value": "Intrinsic Value",
    "Margin of Safety %": "Margin of Safety %",
    "Sarmaaya Score": "Sarmaaya Score",
    "Explosive Ratio": "Explosive Ratio",
    "Sarmaaya Aggressive Rank": "Sarmaaya Aggressive Rank",

    # Growth
    "Net Profit CAGR": "Net Profit CAGR",
    "EPS": "EPS",
    "OPG CAGR operating": "Operating Profit Growth CAGR",
    "Revenue CAGR": "Revenue CAGR",

    # Stability
    "Operating Margins": "Operating Margin",
    "Net Margin": "Net Margin",
    "Tax Rate": "Tax Rate",
    "Current Ratio": "Current Ratio",
    "Total Debt": "Total Debt",
    "Debt to Equity": "Debt to Equity",
    "Interest Coverage Ratio": "Interest Coverage Ratio",
    "Cash Flow from Operation": "Cash Flow from Operation",
    "Net Change in Cash": "Net Change in Cash",
    "CCFO vs CPAT": "CCFO vs CPAT",
    "ROE %": "ROE %",
    "Fixed Asset Turnover": "Fixed Asset Turnover",
    "Cash per Share": "Cash per Share",

    # Valuation
    "Price to Earnings": "P/E",
    "PEG Ratio": "PEG Ratio",
    "Earning Yield": "Earning Yield",
    "Price to Book Ratio": "P/B",
    "Graham Value": "Graham Value",
    "Price to Sales": "P/S",
    "Dividend Yield": "Dividend Yield",
    "EV/EBITDA": "EV/EBITDA",

    # Inventory
    "Inventory Turnover Ratio": "Inventory Turnover Ratio",
    "Day Receivable Outstanding": "Days Receivable Outstanding",
    "Day Sales of Inventory": "Days Sales of Inventory",
    "Days Payable Outstanding": "Days Payable Outstanding",
    "Cash Conversion Cycle": "Cash Conversion Cycle",

    # Cashflow
    "Free Cash Flow per Share": "Free Cash Flow per Share",
    "Free Cash Flow per Sale": "Free Cash Flow per Sale",
    "Free Cash Flow per CFO": "Free Cash Flow per CFO",
    "Cash Return on Invested Capital": "CROIC",
    "Cash to Debt Ratio": "Cash to Debt Ratio",
}


def _extract_metric_rows_from_sarmaaya_raw(symbol: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Extract metric/value/rating rows from parsed Sarmaaya raw lines.

    Uses the known metric names from the six Sarmaaya sections. It prefers TTM/first visible numeric value.
    """
    if raw_df is None or not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        return pd.DataFrame()

    metric_names = sorted(SARMAAYA_TO_FUNDAMENTAL_MAP.keys(), key=len, reverse=True)
    rows = []
    for _, row in raw_df.iterrows():
        line = str(row.get("Raw Line", "") or "").strip()
        section = str(row.get("Section", "") or "").strip()
        if not line:
            continue

        matched_metric = None
        for metric in metric_names:
            if metric.lower() in line.lower():
                matched_metric = metric
                break
        if not matched_metric:
            continue

        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", line)
        clean_numbers = [_clean_sarmaaya_number(n) for n in numbers]
        clean_numbers = [n for n in clean_numbers if n is not None]

        rating = str(row.get("Rating Found", "") or "")
        if not rating:
            if re.search(r"\bGOOD\b", line, flags=re.IGNORECASE):
                rating = "GOOD"
            elif re.search(r"\bBAD\b", line, flags=re.IGNORECASE):
                rating = "BAD"
            elif re.search(r"\bAVERAGE\b", line, flags=re.IGNORECASE):
                rating = "AVERAGE"

        # If line contains metric plus multiple years, first number is usually TTM/current.
        ttm_value = clean_numbers[0] if clean_numbers else None
        rows.append({
            "Symbol": symbol,
            "Section": section,
            "Sarmaaya Metric": matched_metric,
            "Fundamental Field": SARMAAYA_TO_FUNDAMENTAL_MAP.get(matched_metric, matched_metric),
            "Filled Value": ttm_value,
            "All Values Found": ", ".join(str(x) for x in clean_numbers),
            "Rating": rating,
            "Source": "Sarmaaya Six-Box Import",
            "Raw Line": line,
        })

    return pd.DataFrame(rows)


def _extract_main_summary_fields_as_fundamentals(symbol: str, summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if summary_df is None or not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        return pd.DataFrame()
    for _, row in summary_df.iterrows():
        metric = str(row.get("Metric", "") or "").strip()
        if not metric:
            continue
        field = SARMAAYA_TO_FUNDAMENTAL_MAP.get(metric, metric)
        val = row.get("Value", None)
        rows.append({
            "Symbol": symbol,
            "Section": str(row.get("Section", "") or "Main / MOS"),
            "Sarmaaya Metric": metric,
            "Fundamental Field": field,
            "Filled Value": val,
            "All Values Found": str(val),
            "Rating": "",
            "Source": "Sarmaaya Main/MOS Summary",
            "Raw Line": f"{metric}: {val}",
        })
    return pd.DataFrame(rows)


def _build_sarmaaya_fundamental_fill_table(symbol: str, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean table the bot can use to fill missing fundamental data."""
    metric_rows = _extract_metric_rows_from_sarmaaya_raw(symbol, raw_df)
    main_rows = _extract_main_summary_fields_as_fundamentals(symbol, summary_df)
    parts = [df for df in [main_rows, metric_rows] if isinstance(df, pd.DataFrame) and not df.empty]
    if not parts:
        return pd.DataFrame(columns=["Symbol", "Fundamental Field", "Filled Value", "Rating", "Source", "Raw Line"])

    fill_df = pd.concat(parts, ignore_index=True, sort=False)

    # Add section scores as fundamentals too.
    score_rows = []
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        for _, r in summary_df.iterrows():
            sec = str(r.get("Section", "") or "")
            score = r.get("Section Score /20", None)
            if pd.notna(score) and sec:
                score_rows.append({
                    "Symbol": symbol,
                    "Section": sec,
                    "Sarmaaya Metric": f"{sec} Score /20",
                    "Fundamental Field": f"Sarmaaya {sec} Score /20",
                    "Filled Value": score,
                    "All Values Found": str(score),
                    "Rating": "",
                    "Source": "Sarmaaya Section Score",
                    "Raw Line": f"{sec} score {score}/20",
                })
    if score_rows:
        fill_df = pd.concat([fill_df, pd.DataFrame(score_rows)], ignore_index=True, sort=False)

    # De-duplicate: keep first non-empty value per fundamental field.
    fill_df["Filled Value"] = fill_df["Filled Value"].replace("", pd.NA)
    fill_df = fill_df.dropna(subset=["Fundamental Field"])
    fill_df = fill_df.sort_values(["Symbol", "Fundamental Field", "Source"]).drop_duplicates(
        subset=["Symbol", "Fundamental Field"],
        keep="first",
    )
    return fill_df.reset_index(drop=True)





def _scalarize_cell(value):
    """Convert any list/tuple/Series/array/dict cell into a safe scalar string."""
    try:
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return pd.NA
            return _scalarize_cell(value.iloc[0])
    except Exception:
        pass
    try:
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                return pd.NA
            return " / ".join(str(x) for x in value)
    except Exception:
        pass
    try:
        if isinstance(value, dict):
            return json.dumps(value, default=str)
    except Exception:
        pass
    return value



def _dedupe_symbol_rows_for_decision(df: pd.DataFrame) -> pd.DataFrame:
    """Reset index and keep one row per symbol for Decision Center."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy().reset_index(drop=True)
    symbol_col = None
    for c in out.columns:
        if str(c).strip().lower() in {"symbol", "symbols", "scrip", "ticker", "code"}:
            symbol_col = c
            break
    if symbol_col:
        out[symbol_col] = out[symbol_col].astype(str).str.upper().str.strip()
        out = out[out[symbol_col].astype(str).str.strip().ne("")]
        out = out.drop_duplicates(subset=[symbol_col], keep="last").reset_index(drop=True)
    return out


def _sanitize_dataframe_for_decision_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every column is a real 1-D Series and every cell is scalar-safe."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy().reset_index(drop=True)

    # Remove duplicate column names by suffixing them. Duplicate labels can cause Series/DataFrame ambiguity.
    seen = {}
    new_cols = []
    for c in out.columns:
        name = str(c)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[str(c)]}"
        else:
            seen[name] = 0
        new_cols.append(name)
    out.columns = new_cols

    for c in list(out.columns):
        try:
            col = out[c]
            # If duplicate access still somehow returns DataFrame, collapse to first column.
            if isinstance(col, pd.DataFrame):
                out[c] = col.iloc[:, 0]
            out[c] = out[c].map(_scalarize_cell).astype("object")
        except Exception:
            try:
                out[c] = out[c].astype("object")
            except Exception:
                pass
    return out


def _safe_str_value(value):
    """Return a safe string/missing value for full-bot Sarmaaya dataframe storage."""
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass
    s = str(value).strip()
    if s.lower() in {"", "nan", "none"}:
        return pd.NA
    return s


def _safe_numeric_value(value):
    """Return numeric value only for numeric alias columns."""
    try:
        if value is None or pd.isna(value):
            return pd.NA
    except Exception:
        pass
    s = str(value).strip()
    if "/" in s:
        return pd.NA
    try:
        return float(str(value).replace(",", "").replace("%", "").replace("PKR", "").replace("Rs.", "").replace("Rs", "").strip())
    except Exception:
        return pd.NA


def _force_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid pandas/pyarrow dtype errors and 1-D Series errors."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = _sanitize_dataframe_for_decision_engine(df)
    for c in out.columns:
        try:
            out[c] = out[c].astype("object")
        except Exception:
            pass
    return out



def _sarmaaya_fill_to_full_bot_wide(fill_df: pd.DataFrame) -> pd.DataFrame:
    """Convert Sarmaaya fill table to a normal wide fundamental table for full bot use.

    Main fundamental fields are stored as strings to avoid dtype conflicts.
    Numeric aliases are stored as floats only in separate alias columns.
    """
    if fill_df is None or not isinstance(fill_df, pd.DataFrame) or fill_df.empty:
        return pd.DataFrame()

    required = {"Symbol", "Fundamental Field", "Filled Value"}
    if not required.issubset(set(fill_df.columns)):
        return pd.DataFrame()

    wide_rows = []
    for sym, group in fill_df.groupby(fill_df["Symbol"].astype(str).str.upper().str.strip()):
        if not sym or sym == "NAN":
            continue
        row = {
            "Symbol": sym,
            "symbol": sym,
            "Fundamental Source": "Sarmaaya Six-Box Saved Data",
            "Sarmaaya Fundamental Fill Used": "Yes",
        }
        fields = []
        for _, r in group.iterrows():
            field = str(r.get("Fundamental Field", "") or "").strip()
            val = r.get("Filled Value", None)
            if not field:
                continue
            safe_str = _safe_str_value(val)
            if pd.isna(safe_str):
                continue

            # Store original Sarmaaya fields as string-safe values.
            row[field] = safe_str
            fields.append(field)

            # Numeric aliases used by existing engines.
            aliases = {
                "P/E": ["pe", "PE", "Price to Earnings"],
                "P/B": ["pb", "PB", "Price to Book"],
                "P/S": ["ps", "PS", "Price to Sales"],
                "PEG Ratio": ["peg", "PEG"],
                "Earning Yield": ["earnings_yield", "Earnings Yield"],
                "Dividend Yield": ["dividend_yield", "Dividend %"],
                "EV/EBITDA": ["ev_ebitda"],

                "Revenue CAGR": ["revenue_growth", "Revenue Growth", "revenue_cagr_3y", "revenue_cagr_5y", "sales_growth"],
                "Net Profit CAGR": ["profit_growth", "EPS Growth", "profit_cagr_3y", "profit_cagr_5y"],
                "EPS": ["eps", "eps_trend_3y", "eps_trend_5y"],
                "OPG CAGR operating": ["ebitda_growth_avg"],

                "Operating Margin": ["operating_margin"],
                "Net Margin": ["net_margin"],
                "ROE %": ["roe", "ROE"],
                "Debt to Equity": ["debt_equity", "Debt/Equity", "Debt Equity", "debt_to_equity"],
                "Current Ratio": ["current_ratio"],
                "Interest Coverage Ratio": ["interest_coverage"],
                "Cash Flow from Operation": ["ccfo", "cfo_avg_3y", "cfo_avg_5y"],
                "CCFO vs CPAT": ["ccfo_cpat"],

                "Inventory Turnover Ratio": ["inventory_turnover"],
                "Day Receivable Outstanding": ["dso", "Days Sales Outstanding"],
                "Day Sales of Inventory": ["inventory_days"],
                "Days Payable Outstanding": ["payables_days"],
                "Cash Conversion Cycle": ["ccc"],

                "Free Cash Flow per Share": ["fcf_per_share_3y", "fcf_per_share_5y"],
                "Free Cash Flow per Sale": ["fcf_sales"],
                "Free Cash Flow per CFO": ["fcf_cfo_3y", "fcf_cfo_5y"],
                "Cash Return on Invested Capital": ["croic"],
                "Cash to Debt Ratio": ["cash_to_debt"],

                "Intrinsic Value": ["intrinsic_value"],
                "Margin of Safety %": ["margin_of_safety", "MOS"],
                "Sarmaaya Score": ["sarmaaya_score"],
                "Sarmaaya Aggressive Rank": ["sarmaaya_aggressive_rank"],
            }
            num_val = _safe_numeric_value(val)

            if field == "CCFO vs CPAT" and "/" in str(safe_str):
                try:
                    left, right = str(safe_str).replace(",", "").split("/", 1)
                    row["ccfo"] = float(left.strip())
                    row["cpat"] = float(right.strip())
                except Exception:
                    pass

            if field == "Free Cash Flow per Share" and "/" in str(safe_str):
                try:
                    left, right = str(safe_str).replace(",", "").split("/", 1)
                    if float(right.strip()) != 0:
                        fcfps = float(left.strip()) / float(right.strip())
                        row["fcf_per_share_3y"] = fcfps
                        row["fcf_per_share_5y"] = fcfps
                except Exception:
                    pass

            for alias in aliases.get(field, []):
                row[alias] = num_val if not pd.isna(num_val) else safe_str

        row["Sarmaaya Filled Fields Count"] = str(len(fields))
        row["Sarmaaya Filled Fields"] = ", ".join(fields[:120])
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    return _sanitize_dataframe_for_decision_engine(_force_object_columns(wide))



def _save_sarmaaya_full_bot_wide_table(fill_df: pd.DataFrame) -> pd.DataFrame:
    """Save a full-bot-ready Sarmaaya wide fundamentals file."""
    wide = _sarmaaya_fill_to_full_bot_wide(fill_df)
    if wide is None or not isinstance(wide, pd.DataFrame) or wide.empty:
        return pd.DataFrame()

    wide = _force_object_columns(wide)
    st.session_state["sarmaaya_full_bot_fundamentals_wide"] = wide.copy()
    try:
        alert_dir = Path(".alert_state")
        alert_dir.mkdir(exist_ok=True)
        wide.to_csv(alert_dir / "sarmaaya_full_bot_fundamentals_wide.csv", index=False)
    except Exception:
        pass
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        wide.to_csv(data_dir / "sarmaaya_full_bot_fundamentals_wide.csv", index=False)
    except Exception:
        pass
    return wide


def _load_sarmaaya_full_bot_wide_table() -> pd.DataFrame:
    """Load full-bot Sarmaaya wide table from session or disk."""
    try:
        df = st.session_state.get("sarmaaya_full_bot_fundamentals_wide", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        pass

    for p in [Path(".alert_state") / "sarmaaya_full_bot_fundamentals_wide.csv", Path("data") / "sarmaaya_full_bot_fundamentals_wide.csv"]:
        try:
            if p.exists():
                return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()


def _load_sarmaaya_fill_table_anytime() -> pd.DataFrame:
    """Load Sarmaaya fill table from session or saved files."""
    for key in ["sarmaaya_fundamental_fill_table", "sarmaaya_missing_fundamental_fill_df"]:
        try:
            df = st.session_state.get(key, pd.DataFrame())
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass
    try:
        p = Path(".alert_state") / "sarmaaya_fundamental_fill_table.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _merge_sarmaaya_full_bot_fundamentals(base_df: pd.DataFrame) -> pd.DataFrame:
    """Merge saved Sarmaaya fundamentals into any full-bot fundamentals table safely."""
    wide = _load_sarmaaya_full_bot_wide_table()
    if wide is None or not isinstance(wide, pd.DataFrame) or wide.empty:
        fill_df = _load_sarmaaya_fill_table_anytime()
        wide = _save_sarmaaya_full_bot_wide_table(fill_df)

    wide = _force_object_columns(wide)
    if wide is None or not isinstance(wide, pd.DataFrame) or wide.empty:
        return _force_object_columns(base_df) if isinstance(base_df, pd.DataFrame) else pd.DataFrame()

    if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return wide.copy()

    out = _force_object_columns(base_df)

    # detect symbol column
    symbol_col = None
    for c in out.columns:
        if str(c).strip().lower() in {"symbol", "symbols", "scrip", "ticker", "code"}:
            symbol_col = c
            break
    if symbol_col is None:
        out["Symbol"] = pd.NA
        symbol_col = "Symbol"

    # ensure all columns exist as object dtype
    for c in wide.columns:
        if c not in out.columns:
            out[c] = pd.Series([pd.NA] * len(out), dtype="object")
        else:
            out[c] = out[c].astype("object")

    base_symbols = out[symbol_col].astype(str).str.upper().str.strip()
    append_rows = []

    for _, wrow in wide.iterrows():
        sym = str(wrow.get("Symbol", wrow.get("symbol", "")) or "").upper().strip()
        if not sym:
            continue
        mask = base_symbols.eq(sym)
        if mask.any():
            for col in wide.columns:
                val = wrow.get(col, pd.NA)
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                if str(val).strip() in {"", "-", "None", "nan", "<NA>"}:
                    continue

                # Use object-safe assignment to avoid dtype errors.
                empty_mask = out[col].isna() | out[col].astype(str).str.strip().isin(["", "-", "None", "nan", "<NA>"])
                out.loc[mask & empty_mask, col] = str(val) if not isinstance(val, str) else val

            out.loc[mask, "Sarmaaya Fundamental Fill Used"] = "Yes"
            if "Fundamental Source" in out.columns:
                fs_empty = out["Fundamental Source"].isna() | out["Fundamental Source"].astype(str).str.strip().isin(["", "-", "None", "nan", "<NA>"])
                out.loc[mask & fs_empty, "Fundamental Source"] = "Sarmaaya Six-Box Saved Data"
        else:
            new_row = {c: pd.NA for c in out.columns}
            for col in wide.columns:
                if col in new_row:
                    val = wrow.get(col, pd.NA)
                    try:
                        new_row[col] = pd.NA if pd.isna(val) else (str(val) if not isinstance(val, str) else val)
                    except Exception:
                        new_row[col] = str(val)
            new_row[symbol_col] = sym
            if "Symbol" in new_row:
                new_row["Symbol"] = sym
            if "symbol" in new_row:
                new_row["symbol"] = sym
            new_row["Sarmaaya Fundamental Fill Used"] = "Yes"
            if "Fundamental Source" in new_row:
                new_row["Fundamental Source"] = "Sarmaaya Six-Box Saved Data"
            append_rows.append(new_row)

    if append_rows:
        out = pd.concat([out, pd.DataFrame(append_rows)], ignore_index=True, sort=False)

    return _force_object_columns(out)



def _sarmaaya_full_bot_status_text() -> str:
    wide = _load_sarmaaya_full_bot_wide_table()
    if wide is None or not isinstance(wide, pd.DataFrame) or wide.empty:
        fill_df = _load_sarmaaya_fill_table_anytime()
        wide = _sarmaaya_fill_to_full_bot_wide(fill_df)
    if wide is None or not isinstance(wide, pd.DataFrame) or wide.empty:
        return "No saved Sarmaaya full-bot fundamental data found."
    try:
        syms = wide["Symbol"].astype(str).str.upper().str.strip().dropna().unique().tolist()
        return f"Sarmaaya full-bot fundamentals active for {len(syms)} symbol(s): {', '.join(syms[:15])}"
    except Exception:
        return "Sarmaaya full-bot fundamentals are active."


def _save_sarmaaya_fundamental_fill_table(fill_df: pd.DataFrame) -> None:
    if fill_df is None or not isinstance(fill_df, pd.DataFrame) or fill_df.empty:
        return
    st.session_state["sarmaaya_fundamental_fill_table"] = fill_df.copy()
    try:
        out_dir = Path(".alert_state")
        out_dir.mkdir(exist_ok=True)
        fill_df.to_csv(out_dir / "sarmaaya_fundamental_fill_table.csv", index=False)
    except Exception:
        pass
    # Also save full-bot-ready wide fundamentals for Decision Center, Fundamental Lab, Stock Deep Dive, etc.
    try:
        _save_sarmaaya_full_bot_wide_table(fill_df)
    except Exception:
        pass


def _load_sarmaaya_fundamental_fill_table() -> pd.DataFrame:
    try:
        p = Path(".alert_state") / "sarmaaya_fundamental_fill_table.csv"
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()


def _apply_sarmaaya_fill_to_table(base_df: pd.DataFrame, fill_df: pd.DataFrame, symbol_col: str = "Symbol") -> pd.DataFrame:
    """Fill missing fundamental cells in an uploaded/base fundamental table using Sarmaaya fill table.

    This is generic: if a Fundamental Field matches a column in base_df, missing/blank values are filled.
    """
    if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty:
        return base_df
    if fill_df is None or not isinstance(fill_df, pd.DataFrame) or fill_df.empty:
        return base_df

    out = base_df.copy()
    if symbol_col not in out.columns:
        # try common symbol columns
        for c in out.columns:
            if str(c).strip().lower() in {"symbol", "scrip", "ticker", "code"}:
                symbol_col = c
                break
    if symbol_col not in out.columns:
        return out

    for _, f in fill_df.iterrows():
        sym = str(f.get("Symbol", "") or "").upper().strip()
        field = str(f.get("Fundamental Field", "") or "").strip()
        val = f.get("Filled Value", None)
        if not sym or not field or field not in out.columns:
            continue
        mask = out[symbol_col].astype(str).str.upper().str.strip().eq(sym)
        if not mask.any():
            continue
        empty_mask = out[field].isna() | out[field].astype(str).str.strip().isin(["", "-", "nan", "None"])
        out.loc[mask & empty_mask, field] = val
    return out



SARMAAYA_KNOWN_METRICS = [
    "Net Profit CAGR",
    "EPS",
    "OPG CAGR operating",
    "Revenue CAGR",
    "Operating Margins",
    "Net Margin",
    "Tax Rate",
    "Current Ratio",
    "Total Debt",
    "Debt to Equity",
    "Interest Coverage Ratio",
    "Cash Flow from Operation",
    "Net Change in Cash",
    "CCFO vs CPAT",
    "ROE %",
    "Fixed Asset Turnover",
    "Cash per Share",
    "Price to Earnings",
    "PEG Ratio",
    "Earning Yield",
    "Price to Book Ratio",
    "Graham Value",
    "Price to Sales",
    "Dividend Yield",
    "EV/EBITDA",
    "Inventory Turnover Ratio",
    "Day Receivable Outstanding",
    "Day Sales of Inventory",
    "Days Payable Outstanding",
    "Cash Conversion Cycle",
    "Free Cash Flow per Share",
    "Free Cash Flow per Sale",
    "Free Cash Flow per CFO",
    "Cash Return on Invested Capital",
    "Cash to Debt Ratio",
]


def _parse_sarmaaya_section_table(symbol: str, section: str, text_value: str) -> pd.DataFrame:
    """Parse one pasted Sarmaaya section into the clean displayed table.

    Output columns match the Sarmaaya layout:
    Metric | TTM | 2025 | 2024 | 2023 | 2022 | 2021 | Rating
    """
    symbol = str(symbol or "").strip().upper()
    raw = str(text_value or "").replace("\r", "\n")
    # Remove header-only lines but keep metric lines.
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in raw.split("\n") if ln.strip()]
    rows = []

    metric_names = sorted(SARMAAYA_KNOWN_METRICS, key=len, reverse=True)

    # Join "OPG CAGR" + "operating" if copied with a line break.
    repaired_lines = []
    skip_next = False
    for i, ln in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if ln.strip().lower() == "opg cagr" and i + 1 < len(lines) and lines[i + 1].strip().lower() == "operating":
            repaired_lines.append("OPG CAGR operating")
            skip_next = True
        else:
            repaired_lines.append(ln)
    lines = repaired_lines

    pending_row = None
    header_words = {"ratios", "ttm", "2025", "2024", "2023", "2022", "2021", "rating"}

    def finish_pending():
        nonlocal pending_row
        if pending_row is not None:
            rows.append(pending_row)
            pending_row = None

    for ln in lines:
        low = ln.lower().strip()
        if low in header_words or low in {"valuation", "growth", "stability", "inventory", "cashflow", "cash flow"}:
            continue
        # Section score row like 11/20 should not become a metric row.
        if re.fullmatch(r"\d+(?:\.\d+)?\s*/\s*20", ln):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", ln) and pending_row is None:
            continue

        rating_only = None
        if re.fullmatch(r"GOOD|BAD|AVERAGE", ln, flags=re.IGNORECASE):
            rating_only = ln.upper()
        if rating_only and pending_row is not None:
            pending_row["Rating"] = rating_only
            finish_pending()
            continue

        matched_metric = None
        for metric in metric_names:
            if ln.lower().startswith(metric.lower()):
                matched_metric = metric
                break

        if matched_metric:
            finish_pending()
            rest = ln[len(matched_metric):].strip()
            # Values may be separated by tabs originally, but after cleanup we parse tokens.
            tokens = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\-?\d+(?:,\d{3})*(?:\.\d+)?)?%?|-|GOOD|BAD|AVERAGE", rest, flags=re.IGNORECASE)
            vals = []
            rating = ""
            for t in tokens:
                tu = str(t).upper()
                if tu in {"GOOD", "BAD", "AVERAGE"}:
                    rating = tu
                else:
                    vals.append(t)

            # Some rows copied with rating on next line.
            pending_row = {
                "Symbol": symbol,
                "Section": section,
                "Metric": matched_metric,
                "TTM": vals[0] if len(vals) > 0 else "",
                "2025": vals[1] if len(vals) > 1 else "",
                "2024": vals[2] if len(vals) > 2 else "",
                "2023": vals[3] if len(vals) > 3 else "",
                "2022": vals[4] if len(vals) > 4 else "",
                "2021": vals[5] if len(vals) > 5 else "",
                "Rating": rating,
            }
            if rating:
                finish_pending()
            continue

        # If no known metric matched but it has enough numbers, keep as unknown metric line.
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\-?\d+(?:,\d{3})*(?:\.\d+)?)?%?|-", ln)
        if len(nums) >= 3:
            finish_pending()
            metric_guess = re.split(r"\s+-?\d", ln, maxsplit=1)[0].strip()
            rows.append({
                "Symbol": symbol,
                "Section": section,
                "Metric": metric_guess or ln[:40],
                "TTM": nums[0] if len(nums) > 0 else "",
                "2025": nums[1] if len(nums) > 1 else "",
                "2024": nums[2] if len(nums) > 2 else "",
                "2023": nums[3] if len(nums) > 3 else "",
                "2022": nums[4] if len(nums) > 4 else "",
                "2021": nums[5] if len(nums) > 5 else "",
                "Rating": "GOOD" if re.search(r"\bGOOD\b", ln, re.I) else ("BAD" if re.search(r"\bBAD\b", ln, re.I) else ("AVERAGE" if re.search(r"\bAVERAGE\b", ln, re.I) else "")),
            })

    finish_pending()
    cols = ["Symbol", "Section", "Metric", "TTM", "2025", "2024", "2023", "2022", "2021", "Rating"]
    return pd.DataFrame(rows, columns=cols)


def _build_sarmaaya_clean_tables(symbol: str, section_text_map: dict) -> pd.DataFrame:
    frames = []
    for section, txt in section_text_map.items():
        if str(txt or "").strip():
            frames.append(_parse_sarmaaya_section_table(symbol, section, txt))
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Section", "Metric", "TTM", "2025", "2024", "2023", "2022", "2021", "Rating"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _clean_sarmaaya_value_for_fill(value):
    """Use TTM value; if TTM is blank/dash, caller can fallback to 2025."""
    if value is None:
        return None
    s = str(value).strip()
    if s in {"", "-", "nan", "None"}:
        return None
    # Keep ratio strings like 23642.12/10521.66 and FCF per share strings.
    if "/" in s:
        return s
    return _clean_sarmaaya_number(s)


def _build_fill_table_from_clean_sarmaaya_table(clean_df: pd.DataFrame) -> pd.DataFrame:
    if clean_df is None or not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in clean_df.iterrows():
        metric = str(r.get("Metric", "") or "").strip()
        symbol = str(r.get("Symbol", "") or "").strip().upper()
        if not metric or not symbol:
            continue
        field = SARMAAYA_TO_FUNDAMENTAL_MAP.get(metric, metric)
        # Prefer TTM; if not available, use 2025.
        filled = _clean_sarmaaya_value_for_fill(r.get("TTM"))
        source_year = "TTM"
        if filled is None:
            filled = _clean_sarmaaya_value_for_fill(r.get("2025"))
            source_year = "2025 fallback"
        rows.append({
            "Symbol": symbol,
            "Section": r.get("Section", ""),
            "Sarmaaya Metric": metric,
            "Fundamental Field": field,
            "Filled Value": filled,
            "Source Period": source_year,
            "TTM": r.get("TTM", ""),
            "2025": r.get("2025", ""),
            "2024": r.get("2024", ""),
            "2023": r.get("2023", ""),
            "2022": r.get("2022", ""),
            "2021": r.get("2021", ""),
            "Rating": r.get("Rating", ""),
            "Source": "Sarmaaya Clean Six-Box Table",
            "Raw Line": f"{metric} | TTM {r.get('TTM','')} | 2025 {r.get('2025','')} | Rating {r.get('Rating','')}",
        })
    return pd.DataFrame(rows)



def _parse_sarmaaya_main_field_value(symbol: str, mos_text: str) -> pd.DataFrame:
    """Build Main/MOS Field | Value table like Sarmaaya summary examples."""
    txt = str(mos_text or "")
    lines = [ln.strip() for ln in txt.replace("\r", "\n").split("\n") if ln.strip()]
    joined = "\n".join(lines)

    def find_num(pattern, default=""):
        try:
            m = re.search(pattern, joined, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return default

    def next_value_after(label):
        for i, ln in enumerate(lines):
            if ln.strip().lower() == label.lower() and i + 1 < len(lines):
                return lines[i + 1].strip()
        return ""

    company = ""
    sector = ""
    # Common structure: Symbol, Company, Sector in first lines
    if len(lines) >= 2:
        company = lines[1]
    if len(lines) >= 3:
        sector = lines[2]

    current_price = find_num(r"share price today is\s*PKR\s*([0-9,.]+)", "")
    if not current_price:
        # fallback: first large price-looking number after symbol/company lines
        for ln in lines:
            if re.fullmatch(r"[0-9,.]+", ln) and _clean_sarmaaya_number(ln):
                current_price = ln
                break

    price_change = find_num(r"price change is\s*([\-0-9,.]+)", "")
    price_change_pct = find_num(r"\(([+-]?\d+(?:\.\d+)?)%\)", "")
    if price_change_pct and not price_change_pct.endswith("%"):
        price_change_pct = price_change_pct + "%"

    day_high = find_num(r"high price is\s*PKR\s*([0-9,.]+)", "")
    day_low = find_num(r"low price is\s*PKR\s*([0-9,.]+)", "")
    volume = find_num(r"Volume traded in\s+\w+\s+today is\s*([0-9,.]+)", "")
    intrinsic = next_value_after("Intrinsic value")
    mos = next_value_after("Margin of Safety")
    score = next_value_after("Sarmaaya Score")
    explosive = next_value_after("Explosive Ratio")
    aggressive = next_value_after("Sarmaaya Aggressive Rank")

    # Day range / 52 week range fallbacks based on labels
    day_low_fallback = day_high_fallback = wk_low = wk_high = ""
    for i, ln in enumerate(lines):
        if ln.lower() == "day range" and i + 2 < len(lines):
            day_low_fallback, day_high_fallback = lines[i+1], lines[i+2]
        if ln.lower() == "52 week range" and i + 2 < len(lines):
            wk_low, wk_high = lines[i+1], lines[i+2]

    rows = [
        ("Symbol", symbol),
        ("Company", company),
        ("Sector", sector),
        ("Current Price", current_price),
        ("Price Change", price_change),
        ("Price Change %", price_change_pct),
        ("Day Low", day_low or day_low_fallback),
        ("Day High", day_high or day_high_fallback),
        ("52 Week Low", wk_low),
        ("52 Week High", wk_high),
        ("Volume", volume),
        ("Intrinsic Value", intrinsic),
        ("Margin of Safety", mos),
        ("Sarmaaya Score", score),
        ("Explosive Ratio", explosive),
        ("Sarmaaya Aggressive Rank", aggressive),
    ]
    return pd.DataFrame([{"Field": k, "Value": v} for k, v in rows])


def _rating_counts(clean_df: pd.DataFrame, section: str) -> dict:
    if clean_df is None or clean_df.empty:
        return {"GOOD": 0, "AVERAGE": 0, "BAD": 0}
    sec = clean_df[clean_df["Section"].astype(str).eq(section)] if "Section" in clean_df.columns else pd.DataFrame()
    ratings = sec.get("Rating", pd.Series(dtype=str)).astype(str).str.upper()
    return {
        "GOOD": int((ratings == "GOOD").sum()),
        "AVERAGE": int((ratings == "AVERAGE").sum()),
        "BAD": int((ratings == "BAD").sum()),
    }


def _section_reading_text(clean_df: pd.DataFrame, section: str, main_df: pd.DataFrame | None = None) -> str:
    counts = _rating_counts(clean_df, section)
    good, avg, bad = counts["GOOD"], counts["AVERAGE"], counts["BAD"]

    def metric_rating(metric):
        try:
            sec = clean_df[clean_df["Section"].astype(str).eq(section)]
            row = sec[sec["Metric"].astype(str).str.lower().eq(metric.lower())]
            if not row.empty:
                return str(row.iloc[0].get("Rating", "")).upper()
        except Exception:
            pass
        return ""

    if section == "Growth":
        if bad == 0 and avg == 0 and good > 0:
            return "Growth section reading: very strong. All Growth indicators are marked GOOD, especially EPS, revenue growth, operating profit growth, and net profit growth."
        return f"Growth section reading: mixed. GOOD {good}, AVERAGE {avg}, BAD {bad}."

    if section == "Stability":
        debt_bad = metric_rating("Total Debt") == "BAD" or metric_rating("Debt to Equity") == "BAD"
        if debt_bad:
            return "Stability reading: mixed. Profitability and cash flow are strong, but Total Debt and Debt to Equity are weak points. For long-term investment, do not mark the stock fully clean until debt risk is considered."
        return f"Stability reading: GOOD {good}, AVERAGE {avg}, BAD {bad}."

    if section == "Valuation":
        return "Valuation reading: mixed. P/E, PEG, P/S and EV/EBITDA may look acceptable, but weak valuation ratings should prevent marking the stock as clearly undervalued on all metrics."

    if section == "Inventory":
        return "Inventory reading: mostly good. If TTM values are blank, the bot uses 2025 values as fallback for missing TTM inventory indicators."

    if section == "Cashflow":
        return "Cashflow reading: mixed. Strong cash generation is positive, but FCF per CFO and cash-to-debt weakness should be considered before long-term action."

    return f"{section} reading: GOOD {good}, AVERAGE {avg}, BAD {bad}."


def _main_value_from_table(main_df: pd.DataFrame, field: str) -> str:
    try:
        r = main_df[main_df["Field"].astype(str).str.lower().eq(field.lower())]
        if not r.empty:
            return str(r.iloc[0].get("Value", ""))
    except Exception:
        pass
    return ""


def _build_sarmaaya_final_summary(symbol: str, main_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    mos = _main_value_from_table(main_df, "Margin of Safety")
    growth_counts = _rating_counts(clean_df, "Growth")
    stability_counts = _rating_counts(clean_df, "Stability")
    valuation_counts = _rating_counts(clean_df, "Valuation")
    inventory_counts = _rating_counts(clean_df, "Inventory")
    cashflow_counts = _rating_counts(clean_df, "Cashflow")

    rows = [
        {"Section": "Margin of Safety", "Reading": f"Positive, {mos} MOS" if mos else "Available from Main/MOS box"},
        {"Section": "Growth", "Reading": "Strong, all GOOD" if growth_counts["BAD"] == 0 and growth_counts["AVERAGE"] == 0 and growth_counts["GOOD"] > 0 else f"GOOD {growth_counts['GOOD']}, AVERAGE {growth_counts['AVERAGE']}, BAD {growth_counts['BAD']}"},
        {"Section": "Stability", "Reading": "Mixed; debt and debt/equity are BAD" if stability_counts["BAD"] > 0 else f"GOOD {stability_counts['GOOD']}, AVERAGE {stability_counts['AVERAGE']}, BAD {stability_counts['BAD']}"},
        {"Section": "Valuation", "Reading": f"Mixed; GOOD {valuation_counts['GOOD']}, BAD {valuation_counts['BAD']}"},
        {"Section": "Inventory", "Reading": "Mostly GOOD" if inventory_counts["GOOD"] >= max(1, inventory_counts["BAD"]) else f"GOOD {inventory_counts['GOOD']}, BAD {inventory_counts['BAD']}"},
        {"Section": "Cashflow", "Reading": "Mixed; FCF per CFO and cash-to-debt may be BAD" if cashflow_counts["BAD"] > 0 else f"GOOD {cashflow_counts['GOOD']}, BAD {cashflow_counts['BAD']}"},
    ]
    return pd.DataFrame(rows)


def sarmaaya_quick_import_no_screenshot_panel():
    st.markdown("### ⚡ Sarmaaya Quick Import — Six Separate Boxes")
    st.caption("Paste Sarmaaya sections separately. The bot will display clean tables like Sarmaaya: Metric | TTM | 2025 | 2024 | 2023 | 2022 | 2021 | Rating.")

    q1, q2 = st.columns([1, 3])
    symbol = q1.text_input("Symbol", value="SRVI", key="sarmaaya_quick_symbol").strip().upper()
    sarmaaya_url = f"https://sarmaaya.pk/stocks/{symbol}" if symbol else "https://sarmaaya.pk/stocks/"
    q2.markdown(f"Open Sarmaaya page: {sarmaaya_url}")

    st.info(
        "Fast workflow: login to Sarmaaya once → open symbol page → copy each section table text → paste in the matching box below → click Parse and Save."
    )

    b1, b2 = st.columns(2)
    with b1:
        mos_text = st.text_area(
            "1) Margin of Safety / Main page",
            height=220,
            key="sarmaaya_quick_mos_text",
            help="Paste Main/Margin of Safety data: price, intrinsic value, MOS, Sarmaaya Score, Explosive Ratio, Aggressive Rank.",
        )
        stability_text = st.text_area(
            "3) Stability",
            height=220,
            key="sarmaaya_quick_stability_text",
            help="Paste Stability section table text.",
        )
        inventory_text = st.text_area(
            "5) Inventory",
            height=220,
            key="sarmaaya_quick_inventory_text",
            help="Paste Inventory section table text.",
        )

    with b2:
        growth_text = st.text_area(
            "2) Growth",
            height=220,
            key="sarmaaya_quick_growth_text",
            help="Paste Growth section table text.",
        )
        valuation_text = st.text_area(
            "4) Valuation",
            height=220,
            key="sarmaaya_quick_valuation_text",
            help="Paste Valuation section table text.",
        )
        cashflow_text = st.text_area(
            "6) Cashflow",
            height=220,
            key="sarmaaya_quick_cashflow_text",
            help="Paste Cashflow section table text.",
        )

    combined_text = "\n\n".join([
        "Main\n" + str(mos_text or ""),
        "Growth\n" + str(growth_text or ""),
        "Stability\n" + str(stability_text or ""),
        "Valuation\n" + str(valuation_text or ""),
        "Inventory\n" + str(inventory_text or ""),
        "Cashflow\n" + str(cashflow_text or ""),
    ])

    parse_now = st.button("1) Parse Six Sarmaaya Boxes and Save", type="primary", use_container_width=True, key="sarmaaya_quick_parse")
    if parse_now:
        raw_df, summary_df = _parse_sarmaaya_quick_text(symbol, combined_text)

        section_text_map = {
            "Growth": growth_text,
            "Stability": stability_text,
            "Valuation": valuation_text,
            "Inventory": inventory_text,
            "Cashflow": cashflow_text,
        }
        clean_df = _build_sarmaaya_clean_tables(symbol, section_text_map)
        main_df = _parse_sarmaaya_main_field_value(symbol, mos_text)
        final_summary_df = _build_sarmaaya_final_summary(symbol, main_df, clean_df)

        # Main / MOS fields are still extracted as summary fields.
        mos_raw_df, mos_summary_df = _parse_sarmaaya_quick_text(symbol, "Main\n" + str(mos_text or ""))
        if isinstance(mos_summary_df, pd.DataFrame) and not mos_summary_df.empty:
            summary_df = pd.concat([summary_df, mos_summary_df], ignore_index=True, sort=False).drop_duplicates()

        # Add separate section text status for better checking.
        section_status = pd.DataFrame([
            {"Symbol": symbol, "Section": "Main / MOS", "Text Length": len(str(mos_text or "")), "Pasted": "Yes" if str(mos_text or "").strip() else "No"},
            {"Symbol": symbol, "Section": "Growth", "Text Length": len(str(growth_text or "")), "Pasted": "Yes" if str(growth_text or "").strip() else "No"},
            {"Symbol": symbol, "Section": "Stability", "Text Length": len(str(stability_text or "")), "Pasted": "Yes" if str(stability_text or "").strip() else "No"},
            {"Symbol": symbol, "Section": "Valuation", "Text Length": len(str(valuation_text or "")), "Pasted": "Yes" if str(valuation_text or "").strip() else "No"},
            {"Symbol": symbol, "Section": "Inventory", "Text Length": len(str(inventory_text or "")), "Pasted": "Yes" if str(inventory_text or "").strip() else "No"},
            {"Symbol": symbol, "Section": "Cashflow", "Text Length": len(str(cashflow_text or "")), "Pasted": "Yes" if str(cashflow_text or "").strip() else "No"},
        ])

        st.session_state["sarmaaya_quick_raw_df"] = raw_df
        st.session_state["sarmaaya_quick_summary_df"] = summary_df
        st.session_state["sarmaaya_quick_clean_df"] = clean_df
        st.session_state["sarmaaya_quick_main_df"] = main_df
        st.session_state["sarmaaya_quick_final_summary_df"] = final_summary_df
        st.session_state["sarmaaya_quick_section_status"] = section_status

        combined = pd.concat([
            section_status.assign(TableType="Section Status"),
            main_df.assign(TableType="Main Field Value"),
            final_summary_df.assign(TableType="Final Summary"),
            summary_df.assign(TableType="Summary"),
            clean_df.assign(TableType="Clean Section Table"),
            raw_df.assign(TableType="Raw Lines")
        ], ignore_index=True, sort=False)

        _save_sarmaaya_data(combined, label="Sarmaaya Quick Import — Six Boxes")

        fill_from_clean = _build_fill_table_from_clean_sarmaaya_table(clean_df)
        fill_from_old = _build_sarmaaya_fundamental_fill_table(symbol, raw_df, summary_df)
        parts = [df for df in [fill_from_clean, fill_from_old] if isinstance(df, pd.DataFrame) and not df.empty]
        fill_df = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        if isinstance(fill_df, pd.DataFrame) and not fill_df.empty:
            fill_df = fill_df.drop_duplicates(subset=["Symbol", "Fundamental Field"], keep="first")
        _save_sarmaaya_fundamental_fill_table(fill_df)
        st.session_state["sarmaaya_missing_fundamental_fill_df"] = fill_df

        st.success("Six-section Sarmaaya data parsed into clean tables and saved for FULL BOT use: Decision Center, Stock Deep Dive, Fundamental Lab, rankings and AI context.")

    section_status = st.session_state.get("sarmaaya_quick_section_status", pd.DataFrame())
    summary_df = st.session_state.get("sarmaaya_quick_summary_df", pd.DataFrame())
    raw_df = st.session_state.get("sarmaaya_quick_raw_df", pd.DataFrame())

    if isinstance(section_status, pd.DataFrame) and not section_status.empty:
        st.markdown("#### Section Paste Status")
        st.dataframe(section_status, use_container_width=True, hide_index=True)

    main_df = st.session_state.get("sarmaaya_quick_main_df", pd.DataFrame())
    if isinstance(main_df, pd.DataFrame) and not main_df.empty:
        st.markdown("#### Main / Margin of Safety")
        st.dataframe(main_df, use_container_width=True, hide_index=True)

    clean_df = st.session_state.get("sarmaaya_quick_clean_df", pd.DataFrame())
    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        st.markdown("#### Clean Sarmaaya Section Tables")
        st.caption("This view matches Sarmaaya table format: Metric | TTM | 2025 | 2024 | 2023 | 2022 | 2021 | Rating.")
        for sec in ["Growth", "Stability", "Valuation", "Inventory", "Cashflow"]:
            sec_df = clean_df[clean_df["Section"].astype(str).eq(sec)] if "Section" in clean_df.columns else pd.DataFrame()
            if not sec_df.empty:
                st.markdown(f"##### {sec}")
                view_cols = ["Metric", "TTM", "2025", "2024", "2023", "2022", "2021", "Rating"]
                st.dataframe(sec_df[view_cols], use_container_width=True, hide_index=True)
                st.write(_section_reading_text(clean_df, sec, main_df=st.session_state.get("sarmaaya_quick_main_df", pd.DataFrame())))

        final_summary_df = st.session_state.get("sarmaaya_quick_final_summary_df", pd.DataFrame())
        if isinstance(final_summary_df, pd.DataFrame) and not final_summary_df.empty:
            st.markdown(f"#### {symbol} Sarmaaya summary after all 6 boxes")
            st.write("Overall, the stock can be judged from Growth, Stability, Valuation, Inventory, Cashflow and Margin of Safety together. The bot will use this summary plus clean tables in full fundamental checks.")
            st.dataframe(final_summary_df, use_container_width=True, hide_index=True)

    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        st.markdown("#### Parsed Sarmaaya Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        with st.container(border=True):
            st.markdown("### Parsed raw Sarmaaya lines")
            st.dataframe(raw_df, use_container_width=True, hide_index=True)

    fill_df = st.session_state.get("sarmaaya_missing_fundamental_fill_df", pd.DataFrame())
    if fill_df is None or not isinstance(fill_df, pd.DataFrame) or fill_df.empty:
        fill_df = _load_sarmaaya_fundamental_fill_table()
        if isinstance(fill_df, pd.DataFrame) and not fill_df.empty:
            st.session_state["sarmaaya_missing_fundamental_fill_df"] = fill_df

    if isinstance(fill_df, pd.DataFrame) and not fill_df.empty:
        st.markdown("#### Sarmaaya Fundamental Missing Data Fill Table")
        st.caption("These fields can now be used by the bot to fill missing fundamental data.")
        st.dataframe(fill_df, use_container_width=True, hide_index=True)

    st.markdown("### Quick Actions")
    qa1, qa2 = st.columns(2)

    if qa1.button("2) Save Quick Sarmaaya Data Into Bot", type="secondary", use_container_width=True, key="sarmaaya_save_quick_into_bot"):
        fill_df = st.session_state.get("sarmaaya_missing_fundamental_fill_df", pd.DataFrame())
        clean_df = st.session_state.get("sarmaaya_quick_clean_df", pd.DataFrame())
        main_df = st.session_state.get("sarmaaya_quick_main_df", pd.DataFrame())
        summary_df = st.session_state.get("sarmaaya_quick_summary_df", pd.DataFrame())
        raw_df = st.session_state.get("sarmaaya_quick_raw_df", pd.DataFrame())

        parts = []
        for label, df in [
            ("Main Field Value", main_df),
            ("Clean Section Table", clean_df),
            ("Summary", summary_df),
            ("Raw Lines", raw_df),
        ]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(df.assign(TableType=label))

        if parts:
            combined_save = pd.concat(parts, ignore_index=True, sort=False)
            _save_sarmaaya_data(combined_save, label="Sarmaaya Quick Six Boxes")
        if isinstance(fill_df, pd.DataFrame) and not fill_df.empty:
            _save_sarmaaya_fundamental_fill_table(fill_df)
            _save_sarmaaya_full_bot_wide_table(fill_df)
            st.success("Quick Sarmaaya data saved into full bot fundamentals.")
        else:
            st.warning("No parsed Sarmaaya fundamental fill table found. First click Parse Six Sarmaaya Boxes and Save.")

    all_fundamental_data = _load_sarmaaya_full_bot_wide_table()
    if all_fundamental_data is None or not isinstance(all_fundamental_data, pd.DataFrame) or all_fundamental_data.empty:
        all_fundamental_data = st.session_state.get("sarmaaya_full_bot_fundamentals_wide", pd.DataFrame())
    if all_fundamental_data is None or not isinstance(all_fundamental_data, pd.DataFrame) or all_fundamental_data.empty:
        all_fundamental_data = pd.DataFrame([{"Status": "No Sarmaaya fundamental data saved yet. First parse and save."}])

    qa2.download_button(
        "3) Download All Fundamental Data",
        data=all_fundamental_data.to_csv(index=False).encode("utf-8"),
        file_name="all_sarmaaya_full_bot_fundamental_data.csv",
        mime="text/csv",
        use_container_width=True,
        key="sarmaaya_download_all_fundamental_data",
    )




def sarmaaya_data_import_center_panel():
    st.subheader("📊 Sarmaaya Data Import Center")
    st.caption("Import Sarmaaya.pk data into the bot using CSV/Excel, pasted table data, or experimental public stock-page fetch.")

    st.warning("Easiest method: use Sarmaaya Quick Import — Six Boxes. Login once in browser, copy each Sarmaaya section text, paste in its own box.")

    mode = st.radio(
        "Import method",
        ["Sarmaaya Quick Import — Six Boxes", "Upload Sarmaaya CSV / Excel", "Paste Sarmaaya Table", "Upload Sarmaaya Screenshots", "Fetch Public Sarmaaya Stock Pages"],
        horizontal=True,
        key="sarmaaya_import_method",
    )

    imported = pd.DataFrame()

    if mode == "Sarmaaya Quick Import — Six Boxes":
        sarmaaya_quick_import_no_screenshot_panel()
        imported = st.session_state.get("sarmaaya_imported_data", pd.DataFrame())

    elif mode == "Upload Sarmaaya CSV / Excel":
        f = st.file_uploader("Upload Sarmaaya CSV / Excel file", type=["csv", "xlsx", "xls"], key="sarmaaya_file_upload")
        if f is not None:
            try:
                imported = _read_uploaded_sarmaaya_file(f)
                st.success(f"Loaded {len(imported)} rows from {f.name}.")
            except Exception as exc:
                st.error(f"File load failed: {exc}")

    elif mode == "Paste Sarmaaya Table":
        pasted = st.text_area(
            "Paste Sarmaaya table here",
            height=260,
            key="sarmaaya_paste_table",
            help="Copy table from Sarmaaya and paste it here. CSV/tab-separated text both supported.",
        )
        if pasted.strip():
            try:
                from io import StringIO
                sep = "\t" if "\t" in pasted else ","
                imported = pd.read_csv(StringIO(pasted), sep=sep)
                st.success(f"Parsed {len(imported)} rows from pasted table.")
            except Exception as exc:
                st.error(f"Paste parse failed: {exc}")

    elif mode == "Upload Sarmaaya Screenshots":
        st.info("Upload the six Sarmaaya screenshots by section. OCR is optional. If OCR is not installed, use the manual correction table below.")
        symbol = st.text_input("Symbol for these screenshots", value="SRVI", key="sarmaaya_screenshot_symbol").strip().upper()

        section_files = {}
        cols = st.columns(2)
        for idx, section in enumerate(SARMAAYA_SCREENSHOT_SECTIONS):
            with cols[idx % 2]:
                section_files[section] = st.file_uploader(
                    f"Upload {section} screenshot",
                    type=["png", "jpg", "jpeg"],
                    key=f"sarmaaya_screenshot_{section.replace(' ', '_').replace('/', '_')}",
                )

        run_ocr = st.button("Read Screenshots / Build Sarmaaya Template", type="primary", use_container_width=True, key="sarmaaya_screenshot_ocr")
        if run_ocr:
            section_texts = {}
            for section, file_obj in section_files.items():
                section_texts[section] = _ocr_image_file(file_obj) if file_obj is not None else ""
            summary_df = _build_sarmaaya_summary_from_screenshots(symbol, section_texts)
            manual_df = _default_sarmaaya_manual_rows(symbol)
            st.session_state["sarmaaya_screenshot_summary"] = summary_df
            st.session_state["sarmaaya_manual_correction_table"] = manual_df
            st.session_state["sarmaaya_ocr_texts"] = section_texts

        summary_df = st.session_state.get("sarmaaya_screenshot_summary", pd.DataFrame())
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            st.markdown("#### Screenshot OCR Summary")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        manual_table = st.session_state.get("sarmaaya_manual_correction_table")
        if manual_table is None or not isinstance(manual_table, pd.DataFrame) or manual_table.empty:
            manual_table = _default_sarmaaya_manual_rows(symbol)

        st.markdown("#### Manual Correction Table")
        st.caption("Fill or correct the values from Sarmaaya screenshots here. This is the most reliable method if OCR is not installed.")
        edited = st.data_editor(
            manual_table,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="sarmaaya_manual_editor",
        )
        imported = edited.copy()

        with st.container(border=True):
            st.markdown("### OCR raw text by section")
            texts = st.session_state.get("sarmaaya_ocr_texts", {})
            if texts:
                for sec, txt in texts.items():
                    st.text_area(sec, value=txt, height=140, key=f"sarmaaya_ocr_text_{sec}")
            else:
                st.write("No OCR text yet. Install Tesseract OCR and run again, or use manual table.")

    else:
        c1, c2 = st.columns([3, 1])
        symbols_text = c1.text_area(
            "Symbols to fetch from Sarmaaya public pages",
            value="PSX, SYS, NBP, MARI, FFC",
            key="sarmaaya_fetch_symbols",
        )
        max_symbols = c2.number_input("Max symbols", min_value=1, max_value=200, value=25, step=5, key="sarmaaya_fetch_max")
        if st.button("Fetch Sarmaaya Public Pages", type="primary", use_container_width=True, key="sarmaaya_fetch_run"):
            symbols = parse_symbols(symbols_text)[: int(max_symbols)]
            progress = st.progress(0, text="Fetching Sarmaaya pages...")
            rows = []
            for i, sym in enumerate(symbols, start=1):
                progress.progress(int(i / max(len(symbols), 1) * 100), text=f"Fetching {sym} ({i}/{len(symbols)})")
                rows.append(_fetch_sarmaaya_stock_snapshot(sym))
            progress.empty()
            imported = pd.DataFrame(rows)
            st.session_state["sarmaaya_last_fetch"] = imported

        if "sarmaaya_last_fetch" in st.session_state:
            imported = st.session_state["sarmaaya_last_fetch"]

    if isinstance(imported, pd.DataFrame) and not imported.empty:
        st.dataframe(imported, use_container_width=True, hide_index=True)
        if st.button("Save Sarmaaya Data Into Bot", type="primary", use_container_width=True, key="sarmaaya_save_data"):
            _save_sarmaaya_data(imported, label=mode)
            st.success("Sarmaaya data saved into bot session and local .alert_state/sarmaaya_imported_data.csv")

        st.download_button(
            "Download Cleaned Sarmaaya Data",
            data=imported.to_csv(index=False).encode("utf-8"),
            file_name="sarmaaya_imported_data.csv",
            mime="text/csv",
            use_container_width=True,
            key="sarmaaya_download_data",
        )

    saved = st.session_state.get("sarmaaya_imported_data")
    if saved is None or not isinstance(saved, pd.DataFrame) or saved.empty:
        saved = _load_saved_sarmaaya_data()
        if isinstance(saved, pd.DataFrame) and not saved.empty:
            st.session_state["sarmaaya_imported_data"] = saved

    st.markdown("### Full Bot Integration Status")
    st.success(_sarmaaya_full_bot_status_text())
    wide_fullbot = _load_sarmaaya_full_bot_wide_table()
    if isinstance(wide_fullbot, pd.DataFrame) and not wide_fullbot.empty:
        st.caption("This wide table is what the full bot uses automatically as fundamentals.")
        st.dataframe(wide_fullbot.head(100), use_container_width=True, hide_index=True)
    fill_for_full_bot = _load_sarmaaya_fill_table_anytime()
    if isinstance(fill_for_full_bot, pd.DataFrame) and not fill_for_full_bot.empty:
        st.caption("This table is automatically merged into Decision Center, Master Fundamentals, and Basic Fundamentals whenever data is missing.")
        st.dataframe(fill_for_full_bot.head(100), use_container_width=True, hide_index=True)

    if isinstance(saved, pd.DataFrame) and not saved.empty:
        st.markdown("### Saved Sarmaaya Data Available to Bot")
        st.info("This saved Sarmaaya data can now be copied into ChatGPT AI Decision Assistant or used for manual fundamental review.")
        st.dataframe(saved.head(100), use_container_width=True, hide_index=True)
        if st.button("Send Saved Sarmaaya Data to ChatGPT Context", use_container_width=True, key="sarmaaya_send_ai_context"):
            st.session_state["ai_pending_context_text"] = saved.head(50).to_csv(index=False)
            st.success("Saved Sarmaaya data prepared for ChatGPT AI context. Open ChatGPT AI Decision Assistant; it will load safely.")
    else:
        st.info("No saved Sarmaaya data yet.")

    fill_saved_global = st.session_state.get("sarmaaya_missing_fundamental_fill_df", pd.DataFrame())
    if fill_saved_global is None or not isinstance(fill_saved_global, pd.DataFrame) or fill_saved_global.empty:
        fill_saved_global = _load_sarmaaya_fundamental_fill_table()
    if isinstance(fill_saved_global, pd.DataFrame) and not fill_saved_global.empty:
        st.markdown("### Active Sarmaaya Missing Fundamental Fill Data")
        st.dataframe(fill_saved_global.head(150), use_container_width=True, hide_index=True)



def sarmaaya_data_import_center_desk_panel() -> None:
    sarmaaya_data_import_center_panel()


def market_technical_scanner_desk_panel() -> None:
    st.subheader("Market & Technical Scanner")
    st.info("Fast Down Alert Scanner and Latest Divergence Scanner are direct sidebar desks only, so they do not create duplicate widget keys.")
    st.warning("Fast Down Alert Scanner is now a direct left-sidebar desk only, to avoid duplicate Streamlit widget keys. Open **Fast Down Alert Scanner** from the sidebar for 30-second WhatsApp alerts.")
    st.warning("Latest Divergence Scanner is now a direct left-sidebar desk only, to avoid duplicate Streamlit widget keys. Open **Latest Divergence Scanner** from the sidebar.")
    with st.container(border=True):
        st.markdown("### Multi-Style Trading Desk")
        multi_style_trading_desk_panel()
    with st.container(border=True):
        st.markdown("### Scenario Scanner")
        scenario_scanner_panel()
    with st.container(border=True):
        st.markdown("### Pattern & Divergence Scanner")
        pattern_scanner_panel()
    with st.container(border=True):
        st.markdown("### Watchlist PRO Scorecard")
        watchlist_scorecard_panel()


def fundamental_strategy_lab_panel() -> None:
    st.subheader("Fundamental & Strategy Lab")
    with st.container(border=True):
        st.markdown("### Fundamental Image Import Center")
        fundamental_image_import_center_panel()
    with st.container(border=True):
        st.markdown("### Master Fundamentals")
        master_fundamentals_panel()
    with st.container(border=True):
        st.markdown("### Basic Fundamentals & Rankings")
        fundamentals_panel()
    with st.container(border=True):
        st.markdown("### Strategy Builder")
        strategy_builder_panel()
    with st.container(border=True):
        st.markdown("### Corporate Catalyst Scanner")
        corporate_catalyst_panel()


def risk_alerts_watchtower_panel() -> None:
    st.subheader("Risk, Alerts & Watchtower")
    with st.container(border=True):
        st.markdown("### News & Price Hazard Watchtower")
        news_and_price_hazard_watchtower_panel()
    with st.container(border=True):
        st.markdown("### Alert Center")
        alert_center_panel()
    with st.container(border=True):
        st.markdown("### Prediction & Loss Control")
        prediction_loss_control_panel()
    with st.container(border=True):
        st.markdown("### Trade Tracker")
        trade_tracker_panel()


def knowledge_macro_panel() -> None:
    st.subheader("Knowledge & Macro")
    with st.container(border=True):
        st.markdown("### Uploaded Knowledge Brain")
        uploaded_knowledge_brain_panel()
    with st.container(border=True):
        st.markdown("### Macro Checklist")
        macro_checklist_panel()
    with st.container(border=True):
        st.markdown("### Investor Profile")
        investor_profile_panel()


PAGE_RENDERERS = {
    "Decision Center": decision_center_panel,
    "Latest Divergence Scanner": latest_divergence_scanner_desk_panel,
    "Fast Down Alert Scanner": fast_down_alert_scanner_desk_panel,
    "Portfolio WhatsApp Alert Watcher": portfolio_whatsapp_alert_watcher_desk_panel,
    "ChatGPT AI Decision Assistant": chatgpt_ai_decision_assistant_desk_panel,
    "Sarmaaya Data Import Center": sarmaaya_data_import_center_desk_panel,
    "Portfolio Desk": portfolio_desk_panel,
    "Stock Deep Dive": stock_deep_dive_desk_panel,
    "Market & Technical Scanner": market_technical_scanner_desk_panel,
    "Fundamental & Strategy Lab": fundamental_strategy_lab_panel,
    "Risk, Alerts & Watchtower": risk_alerts_watchtower_panel,
    "Knowledge & Macro": knowledge_macro_panel,
}



def render_workspace_panels(active_panels: list[str], layout_mode: str) -> None:
    available_panels = [panel for panel in active_panels if panel in PAGE_RENDERERS]
    if not available_panels:
        available_panels = ["Decision Center"]

    st.markdown(
        f"""
        <div class="terminal-panel">
            <span class="terminal-chip chip-green">{len(available_panels)} panels open</span>
            <span class="terminal-chip chip-blue">Multi-workspace active</span>
            <span class="terminal-chip chip-amber">{layout_mode}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if layout_mode == "Split board":
        left, right = st.columns(2, gap="large")
        for idx, panel_name in enumerate(available_panels):
            target = left if idx % 2 == 0 else right
            with target:
                with st.container(border=True):
                    st.markdown(f"## {panel_name}")
                    PAGE_RENDERERS[panel_name]()
        return

    if layout_mode == "Accordion panels":
        for idx, panel_name in enumerate(available_panels):
            with st.container(border=True):
                st.markdown(f"## {panel_name}")
                PAGE_RENDERERS[panel_name]()
        return

    for panel_name in available_panels:
        with st.container(border=True):
            st.markdown(f"## {panel_name}")
            PAGE_RENDERERS[panel_name]()


render_workspace_panels(active_pages, workspace_layout)
