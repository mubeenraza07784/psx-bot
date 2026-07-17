from pathlib import Path
import re

ROOT = Path('.')
app = ROOT / 'app.py'
psx_data = ROOT / 'cwt_bot' / 'psx_data.py'
psx_scan = ROOT / 'cwt_bot' / 'psx_scenario_scanner.py'
for p in [app, psx_data, psx_scan]:
    if not p.exists():
        raise SystemExit(f"ERROR: {p} not found. Put this patch file in the bot root folder.")

(ROOT / 'requirements.txt').write_text("""streamlit==1.40.2
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
requests==2.32.3
yfinance==0.2.50
beautifulsoup4==4.12.3
html5lib==1.1
openpyxl==3.1.5
xlrd==2.0.1
Pillow==10.4.0
ta==0.11.0
scipy==1.14.1
scikit-learn==1.5.2
python-dateutil==2.9.0.post0
pytz==2024.2
""", encoding='utf-8')
(ROOT / 'packages.txt').write_text('libgl1\n', encoding='utf-8')
(ROOT / 'runtime.txt').write_text('python-3.11\n', encoding='utf-8')

# Disable every Streamlit expander, because app can render multiple panels together.
s = app.read_text(encoding='utf-8')
original = s

def make_title(arg_text: str) -> str:
    arg_text = arg_text.strip()
    m = re.match(r'^[fFrRbBuU]*([\'\"])(.*?)\1', arg_text)
    if m:
        title = m.group(2).replace('{', '').replace('}', '').replace('"', "'")
        return title[:120] if title else 'Details'
    return 'Details'

out = []
changed = 0
for line in s.splitlines():
    m = re.match(r'^(\s*)with\s+st\.expander\((.*)\):\s*$', line)
    if m:
        indent = m.group(1)
        first_arg = m.group(2).split(',', 1)[0]
        title = make_title(first_arg)
        out.append(f'{indent}with st.container(border=True):')
        out.append(f'{indent}    st.markdown("### {title}")')
        changed += 1
    else:
        out.append(line)
s = '\n'.join(out) + ('\n' if original.endswith('\n') else '')

# Safe scenario scanner defaults.
s = s.replace('max_symbols = c3.number_input("Max symbols (0 = all)", value=0, min_value=0, step=25, key="v20_scenario_max")',
              'max_symbols = c3.number_input("Max symbols (0 = all)", value=25, min_value=0, step=25, key="v20_scenario_max")')
s = s.replace('analysis_tf = d2.selectbox("Analysis TF", ANALYSIS_TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_scenario_atf")',
              'analysis_tf = d2.selectbox("Analysis TF", ANALYSIS_TIMEFRAME_SELECTOR_OPTIONS, index=1, key="v20_scenario_atf")')
s = s.replace('execution_tf = d3.selectbox("Execution TF", TIMEFRAME_SELECTOR_OPTIONS, index=0, key="v20_scenario_etf")',
              'execution_tf = d3.selectbox("Execution TF", TIMEFRAME_SELECTOR_OPTIONS, index=1, key="v20_scenario_etf")')
s = s.replace('period = d4.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=3, key="v20_scenario_period")',
              'period = d4.selectbox("Yahoo Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="v20_scenario_period")')

old_result = '''            st.subheader(f"{scenario_system} • {selected_scenario} Matches — Ranked by PRO Score / Timeframe")
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
'''
new_result = '''            st.subheader(f"{scenario_system} • {selected_scenario} Matches — Ranked by PRO Score / Timeframe")
            min_score = st.slider("Filter minimum PRO Score", 0, 100, 0, key="v20_scenario_min_score")

            if matched_df.empty:
                st.warning("No exact scenario match found with the current strict rules.")
                st.caption("Showing nearest analysed candidates instead, so the scanner always gives an answer.")

                if isinstance(all_df, pd.DataFrame) and not all_df.empty:
                    candidate_df = all_df.copy()
                    if "Pro Score" in candidate_df.columns:
                        candidate_df["_Rank Score"] = pd.to_numeric(candidate_df["Pro Score"], errors="coerce").fillna(0)
                    else:
                        candidate_df["_Rank Score"] = 0
                    if "Confidence" in candidate_df.columns:
                        candidate_df["_Rank Score"] = candidate_df["_Rank Score"] + pd.to_numeric(candidate_df["Confidence"], errors="coerce").fillna(0) / 10

                    candidate_df = candidate_df.sort_values("_Rank Score", ascending=False).head(50)
                    show_cols = [c for c in [
                        "Symbol", "Analysis TF", "Execution TF", "Scenario", "Scenario Detail", "Scenario Side",
                        "Bias", "Action", "Setup", "Confidence", "Pro Score", "Grade", "Trade Quality",
                        "Risk", "Momentum", "RSI", "ADX", "Vol Ratio", "Higher Trend", "Execution Trend",
                        "Alligator", "Divergence", "Order", "Entry", "SL", "TP 1:3", "Warnings"
                    ] if c in candidate_df.columns]

                    st.info("Nearest Scenario Candidates / Full Analysis Result")
                    st.dataframe(candidate_df[show_cols] if show_cols else candidate_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Nearest Scenario Candidates CSV",
                        candidate_df.drop(columns=["_Rank Score"], errors="ignore").to_csv(index=False).encode("utf-8"),
                        file_name=f"psx_{selected_scenario.lower().replace(' ', '_')}_nearest_candidates.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error("No symbols were successfully analysed.")
                    st.caption("This usually means data loading failed for every symbol. Check the failed/unavailable symbols section below.")
            else:
                filtered = matched_df[matched_df["Pro Score"].fillna(0) >= min_score] if "Pro Score" in matched_df.columns else matched_df
                st.success(f"Exact matches found: {len(filtered)}")
                st.dataframe(filtered, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Scenario Matches CSV",
                    filtered.to_csv(index=False).encode("utf-8"),
                    file_name=f"psx_{selected_scenario.lower().replace(' ', '_')}_timeframe_matches.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
'''
if old_result in s:
    s = s.replace(old_result, new_result, 1)
app.write_text(s, encoding='utf-8')

# psx_data: fallback universe, safe Yahoo period, 4h support.
s = psx_data.read_text(encoding='utf-8')
if 'import time' not in s:
    s = s.replace('import re\n', 'import re\nimport time\n', 1)
if 'FALLBACK_KSE100_SYMBOLS' not in s:
    s = s.replace('def find_column', '''# Built-in fallback universe used when PSX closes a connection on Streamlit Cloud.
FALLBACK_KSE100_SYMBOLS = [
    "ABL","ABOT","AGP","AICL","AIRLINK","AKBL","APL","ATLH","ATRL","AVN",
    "BAFL","BAHL","BOP","CHCC","COLG","DAWH","DGKC","EFERT","ENGROH","EPCL",
    "FABL","FATIMA","FCEPL","FFC","FFBL","GHGL","HBL","HUBC","ILP","INDU",
    "ISL","JDWS","JVDC","KOHC","LUCK","MARI","MCB","MEBL","MLCF","MTL",
    "NBP","NESTLE","NRL","OGDC","PABC","PAEL","PIOC","POL","PPL","PSO",
    "PSX","SAZEW","SEARL","SHFA","SRVI","SYS","THALL","THCCL","TRG","UBL","UNITY"
]
FALLBACK_ELIGIBLE_SYMBOLS = list(dict.fromkeys(FALLBACK_KSE100_SYMBOLS + [
    "AGIL","AHCL","AHL","AICL","AKDHL","ATIL","BIFO","BNL","BWHL","CENI",
    "CNERGY","DAWN","DCR","DYNO","EFUG","ENGROH","FCSC","FECTC","FCCL","FCEL",
    "FLYNG","GAL","GGL","GHNI","GRR","GWLC","HPL","IMAGE","JLICL","JSGCL",
    "LSEVL","MDTL","MCBIM","NATF","NETSOL","OTSU","PAKOXY","PNSC","POWER",
    "PTC","PTL","RCML","SAPT","SPEL","TPLI","TGL","ZAL"
]))


def find_column''', 1)

load_func = '''def _safe_yahoo_period(interval: str, period: str) -> str:
    tf = str(interval or "1d").lower().strip()
    requested = str(period or "1y").lower().strip()
    if tf in {"1m"}:
        return "7d"
    if tf in {"2m", "5m", "15m", "30m", "60m", "90m"}:
        return "60d"
    if tf in {"1h", "4h"}:
        return "6mo" if requested in {"1y", "2y", "5y", "max"} else requested
    return requested


def load_psx_yahoo_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    clean = symbol.strip().upper()
    yahoo_symbol = clean if clean.endswith(".KA") else f"{clean}.KA"
    requested_interval = str(interval or "1d").strip().lower()
    download_interval = "1h" if requested_interval == "4h" else requested_interval
    download_period = _safe_yahoo_period(requested_interval, period)
    df = load_yfinance_ohlcv(yahoo_symbol, interval=download_interval, period=download_period)
    if requested_interval == "4h":
        df = resample_ohlcv(df, "4h")
    df.attrs["source_symbol"] = yahoo_symbol
    df.attrs["download_interval"] = download_interval
    df.attrs["download_period"] = download_period
    return df
'''
s = re.sub(r'def _safe_yahoo_period\(.*?\n\ndef load_psx_yahoo_ohlcv\(symbol: str, interval: str, period: str\) -> pd\.DataFrame:\n.*?(?=\n\ndef load_psx_csv)', load_func, s, flags=re.S)
s = re.sub(r'def load_psx_yahoo_ohlcv\(symbol: str, interval: str, period: str\) -> pd\.DataFrame:\n.*?(?=\n\ndef load_psx_csv)', load_func, s, flags=re.S)

universe_func = '''def fetch_psx_symbol_universe(universe: str = "Eligible Scrips") -> list[str]:
    normalized = str(universe or "").strip().lower()
    if "kse" in normalized:
        url = "https://dps.psx.com.pk/indices/KSE100"
        fallback_symbols = FALLBACK_KSE100_SYMBOLS
    else:
        url = "https://dps.psx.com.pk/eligible-scrips"
        fallback_symbols = FALLBACK_ELIGIBLE_SYMBOLS
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            for table in tables:
                if table is None or table.empty:
                    continue
                for col in table.columns:
                    if "symbol" in str(col).strip().lower():
                        symbols = _clean_symbol_series(table[col])
                        if symbols:
                            return symbols
            for table in tables:
                if table is not None and not table.empty:
                    symbols = _clean_symbol_series(table.iloc[:, 0])
                    if symbols:
                        return symbols
            raise RuntimeError(f"Could not extract symbols from PSX universe page: {url}")
        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))
    print(f"PSX universe fetch failed, using built-in fallback for {universe}: {last_error}")
    return list(dict.fromkeys(fallback_symbols))
'''
s = re.sub(r'def fetch_psx_symbol_universe\(universe: str = "Eligible Scrips"\) -> list\[str\]:\n.*?raise RuntimeError\(f"Could not extract symbols from PSX universe page: \{url\}"\)\n', universe_func, s, flags=re.S)
psx_data.write_text(s, encoding='utf-8')

# scanner: load same TF once and fallback Yahoo -> DPS.
s = psx_scan.read_text(encoding='utf-8')
old_load = '''            if data_source == "Yahoo Finance PSX (.KA)":
                higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
                lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
            else:
                base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
                higher_df = resample_ohlcv(base_df, analysis_tf)
                lower_df = resample_ohlcv(base_df, execution_tf)
'''
new_load = '''            if data_source == "Yahoo Finance PSX (.KA)":
                try:
                    if str(analysis_tf).strip().lower() == str(execution_tf).strip().lower():
                        higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
                        lower_df = higher_df.copy()
                    else:
                        higher_df = load_psx_yahoo_ohlcv(symbol, interval=analysis_tf, period=period)
                        lower_df = load_psx_yahoo_ohlcv(symbol, interval=execution_tf, period=period)
                except Exception as yahoo_exc:
                    try:
                        base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
                        higher_df = resample_ohlcv(base_df, analysis_tf)
                        lower_df = resample_ohlcv(base_df, execution_tf)
                    except Exception as dps_exc:
                        raise RuntimeError(f"Yahoo failed: {yahoo_exc}; DPS fallback failed: {dps_exc}")
            else:
                base_df = load_psx_dps_ohlcv(symbol, mode=dps_mode)
                higher_df = resample_ohlcv(base_df, analysis_tf)
                lower_df = resample_ohlcv(base_df, execution_tf)
'''
if old_load in s:
    s = s.replace(old_load, new_load, 1)
psx_scan.write_text(s, encoding='utf-8')

print(f"OK: patch applied. Converted {changed} expander blocks if any were present.")
print("Next run: python -m py_compile app.py cwt_bot/*.py")
