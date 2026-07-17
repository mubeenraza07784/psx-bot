from pathlib import Path
import re

p = Path("cwt_bot") / "psx_data.py"
if not p.exists():
    raise SystemExit("ERROR: cwt_bot/psx_data.py not found. Put this patch file inside the bot root folder, then run it again.")

s = p.read_text(encoding="utf-8")

# Add time import for retry backoff.
if "import time" not in s:
    s = s.replace("import re\n", "import re\nimport time\n", 1)

fallback_block = r'''
# Built-in fallback universe used when PSX/Yahoo closes a connection on Streamlit Cloud.
# This is not claimed to be the latest official KSE-100 list; it is a safe starter universe
# so the scanner keeps running even when the remote universe page is temporarily unavailable.
FALLBACK_KSE100_SYMBOLS = [
    "ABL","ABOT","AGP","AICL","AIRLINK","AKBL","APL","ATLH","ATRL","AVN",
    "BAFL","BAHL","BOP","CHCC","COLG","DAWH","DGKC","EFERT","ENGROH","EPCL",
    "FABL","FATIMA","FCEPL","FFC","FFBL","GHGL","HBL","HUBC","ILP","INDU",
    "ISL","JDWS","JVDC","KOHC","LUCK","MARI","MCB","MEBL","MLCF","MTL",
    "NBP","NESTLE","NRL","OGDC","PABC","PAEL","PIOC","POL","PPL","PSO",
    "PSX","SAZEW","SEARL","SHFA","SRVI","SYS","THALL","THCCL","TRG","UBL",
    "UNITY"
]

FALLBACK_ELIGIBLE_SYMBOLS = list(dict.fromkeys(FALLBACK_KSE100_SYMBOLS + [
    "AGIL","AHCL","AHL","AICL","AKDHL","ATIL","BIFO","BNL","BWHL","CENI",
    "CNERGY","DAWN","DCR","DYNO","EFUG","ENGROH","FCSC","FECTC","FCCL","FCEL",
    "FLYNG","GAL","GGL","GHNI","GRR","GWLC","HPL","IMAGE","JLICL","JSGCL",
    "LSEVL","MDTL","MCBIM","NATF","NETSOL","OTSU","PAKOXY","PNSC","POWER",
    "PTC","PTL","RCML","SAPT","SPEL","TPLI","TGL","ZAL"
]))
'''

if "FALLBACK_KSE100_SYMBOLS" not in s:
    marker = "def find_column"
    if marker not in s:
        raise SystemExit("ERROR: Could not find insertion point in cwt_bot/psx_data.py")
    s = s.replace(marker, fallback_block + "\n\n" + marker, 1)

new_func = r'''def fetch_psx_symbol_universe(universe: str = "Eligible Scrips") -> list[str]:
    # Fetch PSX symbol universe with retry and a built-in fallback.
    # Streamlit Cloud sometimes receives:
    # RemoteDisconnected('Remote end closed connection without response')
    # from PSX/DPS pages. That should not crash Scenario Scanner.
    normalized = str(universe or "").strip().lower()
    if "kse" in normalized:
        url = "https://dps.psx.com.pk/indices/KSE100"
        fallback_symbols = FALLBACK_KSE100_SYMBOLS
    else:
        url = "https://dps.psx.com.pk/eligible-scrips"
        fallback_symbols = FALLBACK_ELIGIBLE_SYMBOLS

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "close",
    }

    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=12)
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))
            if not tables:
                raise RuntimeError(f"No HTML tables found on PSX universe page: {url}")

            # Prefer a table with a Symbol-like column.
            for table in tables:
                if table is None or table.empty:
                    continue
                for col in table.columns:
                    if "symbol" in str(col).strip().lower():
                        symbols = _clean_symbol_series(table[col])
                        if symbols:
                            return symbols

            # Fallback to first column of first non-empty table.
            for table in tables:
                if table is not None and not table.empty:
                    symbols = _clean_symbol_series(table.iloc[:, 0])
                    if symbols:
                        return symbols

            raise RuntimeError(f"Could not extract symbols from PSX universe page: {url}")

        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))

    # Do not crash the scanner when the remote site closes the connection.
    print(f"PSX universe fetch failed, using built-in fallback for {universe}: {last_error}")
    return list(dict.fromkeys(fallback_symbols))
'''

pattern = re.compile(
    r'def fetch_psx_symbol_universe\(universe: str = "Eligible Scrips"\) -> list\[str\]:\n'
    r'.*?'
    r'(?=\ndef |\Z)',
    re.DOTALL
)
s2, n = pattern.subn(new_func, s, count=1)
if n != 1:
    raise SystemExit("ERROR: Could not replace fetch_psx_symbol_universe. Send me cwt_bot/psx_data.py.")

p.write_text(s2, encoding="utf-8")
print("OK: Scenario Scanner remote-disconnect patch applied.")
print("Now run:")
print("python -m py_compile app.py cwt_bot/psx_data.py")
print("git add cwt_bot/psx_data.py")
print('git commit -m "Fix scenario scanner remote disconnect fallback"')
print("git push")
