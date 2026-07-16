from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple
import math
import re
import numpy as np
import pandas as pd


# Flexible alias map for user-uploaded CSV/XLSX-exported CSV datasets.
ALIASES: Dict[str, List[str]] = {
    "symbol": ["symbol", "ticker", "script", "code"],
    "company": ["company", "company name", "name"],
    "sector": ["sector", "industry", "sector name"],
    "is_bank": ["is bank", "bank", "financial institution"],
    "price": ["price", "share price", "current price", "market price", "cmp", "close"],
    "beta": ["beta"],
    "shares_outstanding": ["shares outstanding", "shares", "issued shares"],
    "market_cap": ["market cap", "market capitalization", "mcap"],
    "revenue_cagr_3y": ["revenue cagr 3y", "sales cagr 3y", "3y revenue cagr", "3 year sales cagr"],
    "revenue_cagr_5y": ["revenue cagr 5y", "sales cagr 5y", "5y revenue cagr", "5 year sales cagr"],
    "profit_cagr_3y": ["profit cagr 3y", "pat cagr 3y", "net profit cagr 3y"],
    "profit_cagr_5y": ["profit cagr 5y", "pat cagr 5y", "net profit cagr 5y"],
    "ebitda_growth_avg": ["ebitda growth", "average ebitda growth", "ebitda growth avg"],
    "eps_growth_avg": ["eps growth", "average eps growth", "eps growth avg"],
    "eps_trend_3y": ["eps 3y avg", "eps average 3y"],
    "eps_trend_5y": ["eps 5y avg", "eps average 5y"],
    "gross_margin": ["gross margin", "gpm"],
    "operating_margin": ["operating margin", "op margin", "ebit margin"],
    "net_margin": ["net margin", "npm", "profit margin"],
    "roe": ["roe", "return on equity"],
    "roa": ["roa", "return on assets"],
    "roic": ["roic", "return on invested capital"],
    "debt_to_equity": ["debt to equity", "d/e", "de ratio"],
    "current_ratio": ["current ratio"],
    "interest_coverage": ["interest coverage", "times interest earned"],
    "cash_per_share_3y": ["cash per share 3y", "cps 3y avg"],
    "cash_per_share_5y": ["cash per share 5y", "cps 5y avg"],
    "cfo_avg_3y": ["cfo 3y avg", "operating cash flow 3y avg"],
    "cfo_avg_5y": ["cfo 5y avg", "operating cash flow 5y avg"],
    "ccfo": ["ccfo", "cumulative cfo", "cumulative operating cash flow"],
    "cpat": ["cpat", "cumulative pat", "cumulative profit after tax"],
    "debt_trend": ["debt trend", "debt change"],
    "receivables_growth": ["receivables growth", "accounts receivable growth"],
    "sales_growth": ["sales growth", "revenue growth"],
    "dso": ["dso", "days sales outstanding"],
    "ccc": ["ccc", "cash conversion cycle"],
    "inventory_turnover": ["inventory turnover"],
    "inventory_days": ["inventory days", "dio"],
    "payables_days": ["payables days", "dpo"],
    "pe": ["pe", "p/e", "price earnings", "price to earnings"],
    "historical_pe": ["historical pe", "avg pe", "5y pe"],
    "sector_pe": ["sector pe", "industry pe"],
    "peg": ["peg"],
    "earnings_yield": ["earnings yield", "earning yield"],
    "bond_yield": ["bond yield", "tbill yield", "t bill yield", "risk free yield"],
    "pb": ["pb", "p/b", "price to book"],
    "ps": ["ps", "p/s", "price to sales", "price to sales ratio"],
    "dividend_yield": ["dividend yield", "yield"],
    "ev_ebitda": ["ev ebitda", "ev/ebitda"],
    "sector_ev_ebitda": ["sector ev ebitda", "industry ev ebitda"],
    "fcf": ["fcf", "free cash flow"],
    "fcf_per_share_3y": ["fcf per share 3y", "fcfps 3y avg"],
    "fcf_per_share_5y": ["fcf per share 5y", "fcfps 5y avg"],
    "fcf_sales": ["fcf sales", "fcf / sales", "fcf to sales"],
    "fcf_cfo_3y": ["fcf cfo 3y", "fcf/cfo 3y avg"],
    "fcf_cfo_5y": ["fcf cfo 5y", "fcf/cfo 5y avg"],
    "croic": ["croic", "cash return on invested capital"],
    "fcf_yield": ["fcf yield"],
    "book_value_per_share": ["book value per share", "bvps"],
    "cash_per_share": ["cash per share", "cps"],
    "eps": ["eps", "eps ttm", "earnings per share"],
    "growth_rate": ["growth rate", "intrinsic growth rate"],
    "equity": ["equity", "shareholders equity"],
    "debt": ["debt", "total debt"],
    "minority_interest": ["minority interest"],
    "cash": ["cash", "cash and cash equivalents"],
    "capex": ["capex", "capital expenditure"],
    "dividends_paid": ["dividends paid", "dividend cash outflow"],
    "promoter_holding": ["promoter holding", "sponsor holding"],
    "insider_signal": ["insider signal", "insider buying selling"],
    "free_float": ["free float", "free float percentage"],

    # Bank-specific
    "markup_growth": ["markup growth", "interest earned growth"],
    "net_spread_growth": ["net spread growth"],
    "profit_before_provision_growth": ["profit before provision growth", "pbp growth"],
    "bank_profit_growth": ["bank profit growth", "net profit growth"],
    "spread_ratio": ["spread ratio"],
    "bank_net_margin": ["bank net margin", "net margin banking"],
    "tax_ratio": ["tax ratio", "tax burden"],
    "nim": ["nim", "net interest margin"],
    "npl_ratio": ["npl gross loans", "npl / gross loans", "npl ratio"],
    "npl_trend_3y": ["npl 3y avg"],
    "npl_trend_5y": ["npl 5y avg"],
    "provision_ratio_3y": ["provision loans 3y avg", "provision ratio 3y"],
    "provision_ratio_5y": ["provision loans 5y avg", "provision ratio 5y"],
    "deposit_cagr": ["deposit cagr", "deposits growth"],
    "industry_deposit_cagr": ["industry deposit cagr"],
    "casa_ratio": ["casa ratio"],
    "adr": ["adr", "advances to deposit ratio"],
    "industry_adr": ["industry adr"],
    "idr": ["idr", "investments to deposit ratio"],
    "car": ["car", "capital adequacy ratio"],
}



def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate columns after alias-based renaming.

    When multiple aliases map to the same target (e.g., P/E, pe, Price to Earnings),
    pandas can return a DataFrame for df["pe"], which breaks pd.to_numeric.
    This coalesces duplicate labels into one 1-D Series.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if not out.columns.duplicated().any():
        return out
    result = pd.DataFrame(index=out.index)
    for col in dict.fromkeys(list(out.columns)):
        same = out.loc[:, out.columns == col]
        if isinstance(same, pd.DataFrame) and same.shape[1] > 1:
            series = same.iloc[:, 0].copy()
            for i in range(1, same.shape[1]):
                series = series.where(series.notna() & ~series.astype(str).str.strip().isin(["", "-", "nan", "None"]), same.iloc[:, i])
            result[col] = series
        else:
            result[col] = same.iloc[:, 0] if isinstance(same, pd.DataFrame) else same
    return result


def _clean(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, aliases: List[str]) -> str | None:
    cleaned = {_clean(c): c for c in df.columns}

    # Exact normalized match first.
    for alias in aliases:
        key = _clean(alias)
        if key in cleaned:
            return cleaned[key]

    # Conservative fuzzy match only for longer aliases.
    # This avoids short aliases like "pe" matching "percentage".
    for col in df.columns:
        cc = _clean(col)
        for alias in aliases:
            aa = _clean(alias)
            # Only allow alias-inside-column matching for meaningful long aliases.
            # Do NOT allow short column text inside long alias, e.g. P/E -> percentage.
            if len(aa) >= 5 and aa in cc:
                return col
    return None



def _as_num(value: Any) -> float | None:
    """Parse numeric value from strings including percentages and ratio strings."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).replace(",", "").replace("%", "").strip()
    if not s or s.lower() in {"nan", "none", "-", "<na>"}:
        return None
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    try:
        return float(s)
    except Exception:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else None


def _split_ratio_pair(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    s = str(value).replace(",", "").strip()
    if "/" not in s:
        return _as_num(s), None
    left, right = s.split("/", 1)
    return _as_num(left), _as_num(right)


def _first_num(row: pd.Series, keys: list[str]) -> float | None:
    for key in keys:
        if key in row.index:
            val = _as_num(row.get(key))
            if val is not None:
                return val
    return None


def _fill_sarmaaya_derived_master_fields(out: pd.DataFrame) -> pd.DataFrame:
    """Fill master-score internal columns from Sarmaaya six-box saved fields."""
    if out is None or out.empty:
        return out
    out = out.copy()

    required_cols = [
        "revenue_cagr_3y", "revenue_cagr_5y", "profit_cagr_3y", "profit_cagr_5y",
        "eps_trend_3y", "eps_trend_5y", "ccfo", "cpat", "dso", "ccc",
        "sales_growth", "receivables_growth", "fcf_per_share_3y", "fcf_per_share_5y",
        "fcf_sales", "fcf_cfo_3y", "fcf_cfo_5y", "croic", "earnings_yield", "bond_yield",
        "pe", "pb", "ps", "peg", "dividend_yield", "ev_ebitda",
    ]
    for col in required_cols:
        if col not in out.columns:
            out[col] = np.nan

    for idx, row in out.iterrows():
        rev = _first_num(row, ["Revenue CAGR", "Revenue Growth", "revenue_growth", "Revenue CAGR 3Y", "Revenue", "revenue_cagr_3y", "revenue_cagr_5y", "sales_growth"])
        pat = _first_num(row, ["Net Profit CAGR", "profit_growth", "EPS Growth", "Profit CAGR 3Y", "Net Profit", "profit_cagr_3y", "profit_cagr_5y", "bank_profit_growth"])
        eps = _first_num(row, ["EPS", "eps"])

        if pd.isna(out.at[idx, "revenue_cagr_3y"]) and rev is not None:
            out.at[idx, "revenue_cagr_3y"] = rev
        if pd.isna(out.at[idx, "revenue_cagr_5y"]) and rev is not None:
            out.at[idx, "revenue_cagr_5y"] = rev
        if pd.isna(out.at[idx, "profit_cagr_3y"]) and pat is not None:
            out.at[idx, "profit_cagr_3y"] = pat
        if pd.isna(out.at[idx, "profit_cagr_5y"]) and pat is not None:
            out.at[idx, "profit_cagr_5y"] = pat
        if pd.isna(out.at[idx, "eps_trend_3y"]) and eps is not None:
            out.at[idx, "eps_trend_3y"] = eps
        if pd.isna(out.at[idx, "eps_trend_5y"]) and eps is not None:
            out.at[idx, "eps_trend_5y"] = eps

        ccfo, cpat = None, None
        for key in ["CCFO vs CPAT", "ccfo_cpat"]:
            if key in row.index:
                ccfo, cpat = _split_ratio_pair(row.get(key))
                if ccfo is not None or cpat is not None:
                    break
        if ccfo is None:
            ccfo = _first_num(row, ["Cash Flow from Operation", "CFO", "ccfo", "cfo_avg_3y"])
        if cpat is None:
            cpat = _first_num(row, ["Net Profit CAGR", "PAT", "cpat", "profit_cagr_3y"])
        if pd.isna(out.at[idx, "ccfo"]) and ccfo is not None:
            out.at[idx, "ccfo"] = ccfo
        if pd.isna(out.at[idx, "cpat"]) and cpat is not None:
            out.at[idx, "cpat"] = cpat

        alias_map = {
            "pe": ["P/E", "PE", "Price to Earnings", "Price to Earnings Ratio", "pe"],
            "pb": ["P/B", "PB", "Price to Book", "Price to Book Ratio", "pb"],
            "ps": ["P/S", "PS", "Price to Sales", "Price to Sales Ratio", "ps"],
            "peg": ["PEG Ratio", "PEG", "peg"],
            "dividend_yield": ["Dividend Yield", "Dividend %", "dividend_yield"],
            "ev_ebitda": ["EV/EBITDA", "ev_ebitda"],
            "earnings_yield": ["Earning Yield", "Earnings Yield", "earnings_yield"],
        }
        for internal, aliases in alias_map.items():
            if pd.isna(out.at[idx, internal]):
                val = _first_num(row, aliases)
                if val is not None:
                    out.at[idx, internal] = val

        if pd.isna(out.at[idx, "bond_yield"]):
            out.at[idx, "bond_yield"] = 12.0

        if pd.isna(out.at[idx, "dso"]):
            val = _first_num(row, ["Day Receivable Outstanding", "Days Receivable Outstanding", "DSO", "dso"])
            if val is not None:
                out.at[idx, "dso"] = val
        if pd.isna(out.at[idx, "ccc"]):
            val = _first_num(row, ["Cash Conversion Cycle", "CCC", "ccc"])
            if val is not None:
                out.at[idx, "ccc"] = val
        if pd.isna(out.at[idx, "sales_growth"]) and rev is not None:
            out.at[idx, "sales_growth"] = rev
        if pd.isna(out.at[idx, "receivables_growth"]):
            dso_val = _as_num(out.at[idx, "dso"])
            sg_val = _as_num(out.at[idx, "sales_growth"])
            if dso_val is not None:
                out.at[idx, "receivables_growth"] = min(dso_val, sg_val if sg_val is not None else dso_val)

        fcfps = _first_num(row, ["Free Cash Flow per Share", "FCF per Share", "fcf_per_share", "fcf_per_share_3y", "fcf_per_share_5y", "fcf"])
        fcfs = _first_num(row, ["Free Cash Flow per Sale", "Free Cash Flow per Sales", "FCF / Sales", "fcf_sales"])
        fcfcfo = _first_num(row, ["Free Cash Flow per CFO", "FCF / CFO", "fcf_cfo", "fcf_cfo_3y", "fcf_cfo_5y"])
        croic_val = _first_num(row, ["Cash Return on Invested Capital", "CROIC", "croic"])

        if "Free Cash Flow per Share" in row.index and "/" in str(row.get("Free Cash Flow per Share")):
            try:
                left, right = str(row.get("Free Cash Flow per Share")).replace(",", "").split("/", 1)
                if float(right.strip()) != 0:
                    fcfps = float(left.strip()) / float(right.strip())
            except Exception:
                pass

        if pd.isna(out.at[idx, "fcf_per_share_3y"]) and fcfps is not None:
            out.at[idx, "fcf_per_share_3y"] = fcfps
        if pd.isna(out.at[idx, "fcf_per_share_5y"]) and fcfps is not None:
            out.at[idx, "fcf_per_share_5y"] = fcfps
        if pd.isna(out.at[idx, "fcf_sales"]) and fcfs is not None:
            out.at[idx, "fcf_sales"] = fcfs * 100 if abs(fcfs) <= 1 else fcfs
        if pd.isna(out.at[idx, "fcf_cfo_3y"]) and fcfcfo is not None:
            out.at[idx, "fcf_cfo_3y"] = fcfcfo
        if pd.isna(out.at[idx, "fcf_cfo_5y"]) and fcfcfo is not None:
            out.at[idx, "fcf_cfo_5y"] = fcfcfo
        if pd.isna(out.at[idx, "croic"]) and croic_val is not None:
            out.at[idx, "croic"] = croic_val

    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out



def normalize_fundamental_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ren: Dict[str, str] = {}
    used_source_cols = set()
    cleaned_cols = {_clean(c): c for c in out.columns}

    # First pass: exact alias matches across all targets. This prevents broad aliases
    # like "price" from stealing "Price to Sales" before the P/S target is checked.
    for target, aliases in ALIASES.items():
        for alias in aliases:
            key = _clean(alias)
            col = cleaned_cols.get(key)
            if col is not None and col not in used_source_cols:
                ren[col] = target
                used_source_cols.add(col)
                break

    # Second pass: conservative fuzzy matches for columns not mapped exactly.
    for target, aliases in ALIASES.items():
        if target in ren.values():
            continue
        for col in out.columns:
            if col in used_source_cols:
                continue
            cc = _clean(col)
            matched = False
            for alias in aliases:
                aa = _clean(alias)
                if len(aa) >= 8 and aa in cc:
                    ren[col] = target
                    used_source_cols.add(col)
                    matched = True
                    break
            if matched:
                break

    out = out.rename(columns=ren)
    out = _collapse_duplicate_columns(out)

    if "symbol" not in out.columns:
        raise ValueError("A Symbol/Ticker column is required.")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    if "company" not in out.columns:
        out["company"] = out["symbol"]
    if "sector" not in out.columns:
        out["sector"] = "Unknown"

    if "is_bank" not in out.columns:
        out["is_bank"] = out["sector"].astype(str).str.contains("bank", case=False, na=False)
    else:
        out["is_bank"] = out["is_bank"].astype(str).str.lower().isin(["1", "true", "yes", "y", "bank"])

    out = _fill_sarmaaya_derived_master_fields(out)

    numeric_exceptions = {"symbol", "company", "sector", "is_bank", "debt_trend", "insider_signal", "CCFO vs CPAT", "Free Cash Flow per Share"}
    for col in list(out.columns):
        if col not in numeric_exceptions:
            value = out[col]
            if isinstance(value, pd.DataFrame):
                value = _collapse_duplicate_columns(value).iloc[:, 0]
            out[col] = pd.to_numeric(value, errors="coerce")
    out = _collapse_duplicate_columns(out)
    return out


def _rating_good_avg_bad(value: float | None, good: bool | None = None, avg: bool | None = None, note: str = "") -> Dict[str, Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {"value": value, "rating": "Missing", "score": 0, "note": "Data missing"}
    if good is True:
        return {"value": value, "rating": "Good", "score": 2, "note": note}
    if avg is True:
        return {"value": value, "rating": "Average", "score": 1, "note": note}
    return {"value": value, "rating": "Bad", "score": 0, "note": note}


def _val(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _trend_better(row: pd.Series, short_key: str, long_key: str, label: str) -> Dict[str, Any]:
    s = _val(row, short_key)
    l = _val(row, long_key)
    if s is None or l is None:
        return _rating_good_avg_bad(None)
    return _rating_good_avg_bad(s - l, good=s > l, avg=abs(s - l) < 1e-9, note=f"{label}: 3Y={s:.2f}, 5Y={l:.2f}")


def score_nonbank(row: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []

    def add(bucket: str, metric: str, result: Dict[str, Any]):
        factors.append({"Bucket": bucket, "Metric": metric, **result})

    # Growth
    rev3 = _val(row, "revenue_cagr_3y")
    pat3 = _val(row, "profit_cagr_3y")
    add("Growth", "Revenue CAGR 3Y", _rating_good_avg_bad(rev3, good=(rev3 is not None and rev3 >= 15), avg=(rev3 is not None and 8 <= rev3 < 15), note="Target ≥15%"))
    add("Growth", "Profit CAGR 3Y", _rating_good_avg_bad(pat3, good=(pat3 is not None and pat3 >= 15), avg=(pat3 is not None and 8 <= pat3 < 15), note="Target ≥15%"))
    if rev3 is not None and pat3 is not None:
        add("Growth", "Sales-to-Profit Conversion", _rating_good_avg_bad(pat3 - rev3, good=pat3 >= rev3, avg=pat3 >= rev3 * 0.8, note="Profit growth should keep pace with sales growth"))
    else:
        add("Growth", "Sales-to-Profit Conversion", _rating_good_avg_bad(None))

    eps3 = _val(row, "eps_trend_3y")
    eps5 = _val(row, "eps_trend_5y")
    if eps3 is not None and eps5 is not None:
        add("Growth", "EPS Acceleration", _rating_good_avg_bad(eps3 - eps5, good=eps3 > eps5, avg=eps3 == eps5, note="3Y EPS average vs 5Y average"))
    else:
        add("Growth", "EPS Acceleration", _rating_good_avg_bad(None))

    # Stability / Quality
    ccfo = _val(row, "ccfo")
    cpat = _val(row, "cpat")
    if ccfo is not None and cpat is not None:
        add("Stability", "Cumulative CFO vs PAT", _rating_good_avg_bad(ccfo - cpat, good=ccfo >= cpat, avg=ccfo >= cpat * 0.8, note="Cash profits should support accounting profits"))
    else:
        add("Stability", "Cumulative CFO vs PAT", _rating_good_avg_bad(None))

    dte = _val(row, "debt_to_equity")
    add("Stability", "Debt to Equity", _rating_good_avg_bad(dte, good=(dte is not None and dte <= 1.0), avg=(dte is not None and 1.0 < dte <= 2.0), note="Lower leverage preferred"))

    current_ratio = _val(row, "current_ratio")
    add("Stability", "Current Ratio", _rating_good_avg_bad(current_ratio, good=(current_ratio is not None and current_ratio >= 1.5), avg=(current_ratio is not None and 1.0 <= current_ratio < 1.5), note="Liquidity buffer"))

    ic = _val(row, "interest_coverage")
    add("Stability", "Interest Coverage", _rating_good_avg_bad(ic, good=(ic is not None and ic >= 4), avg=(ic is not None and 2 <= ic < 4), note="Ability to service finance cost"))

    # Valuation
    pe = _val(row, "pe")
    add("Valuation", "P/E", _rating_good_avg_bad(pe, good=(pe is not None and pe <= 10), avg=(pe is not None and 10 < pe <= 15), note="PDF rule: ≤10 good, 10–15 average"))

    peg = _val(row, "peg")
    add("Valuation", "PEG", _rating_good_avg_bad(peg, good=(peg is not None and peg <= 1), avg=(peg is not None and 1 < peg <= 1.5), note="Growth-adjusted valuation"))

    ey = _val(row, "earnings_yield")
    by = _val(row, "bond_yield")
    if ey is not None and by is not None:
        add("Valuation", "Earnings Yield vs Bond Yield", _rating_good_avg_bad(ey - by, good=ey > by, avg=ey >= by * 0.9, note="Equity earnings yield should exceed bond yield"))
    else:
        add("Valuation", "Earnings Yield vs Bond Yield", _rating_good_avg_bad(None))

    pb = _val(row, "pb")
    add("Valuation", "P/B", _rating_good_avg_bad(pb, good=(pb is not None and pb <= 1.5), avg=(pb is not None and 1.5 < pb <= 2.5), note="PDF rule: ≤1.5 good"))

    ps = _val(row, "ps")
    add("Valuation", "P/S", _rating_good_avg_bad(ps, good=(ps is not None and ps <= 1.5), avg=(ps is not None and 1.5 < ps <= 3.0), note="PDF rule: ≤1.5 good, 1.5–3 average"))

    dy = _val(row, "dividend_yield")
    add("Valuation", "Dividend Yield", _rating_good_avg_bad(dy, good=(dy is not None and dy >= 4), avg=(dy is not None and 2 <= dy < 4), note="Dividend context; high growth reinvestors may still be acceptable"))

    eve = _val(row, "ev_ebitda")
    add("Valuation", "EV/EBITDA", _rating_good_avg_bad(eve, good=(eve is not None and eve <= 10), avg=(eve is not None and 10 < eve <= 14), note="PDF rule: ≤10 good"))

    if pe is not None and pb is not None:
        graham = pe * pb
        add("Valuation", "Graham P/E×P/B", _rating_good_avg_bad(graham, good=graham < 22.5, avg=22.5 <= graham <= 30, note="Benjamin Graham proxy"))
    else:
        add("Valuation", "Graham P/E×P/B", _rating_good_avg_bad(None))

    # Inventory / working capital
    inv_turn = _val(row, "inventory_turnover")
    add("Inventory", "Inventory Turnover", _rating_good_avg_bad(inv_turn, good=(inv_turn is not None and inv_turn >= 4), avg=(inv_turn is not None and 2 <= inv_turn < 4), note="Higher turnover is usually better"))

    dso = _val(row, "dso")
    add("Inventory", "Days Sales Outstanding", _rating_good_avg_bad(dso, good=(dso is not None and dso <= 60), avg=(dso is not None and 60 < dso <= 90), note="Receivable collection discipline"))

    ccc = _val(row, "ccc")
    add("Inventory", "Cash Conversion Cycle", _rating_good_avg_bad(ccc, good=(ccc is not None and ccc <= 90), avg=(ccc is not None and 90 < ccc <= 150), note="Lower is better"))

    receivables_growth = _val(row, "receivables_growth")
    sales_growth = _val(row, "sales_growth")
    if receivables_growth is not None and sales_growth is not None:
        add("Inventory", "Receivables vs Sales Growth", _rating_good_avg_bad(receivables_growth - sales_growth, good=receivables_growth <= sales_growth, avg=receivables_growth <= sales_growth * 1.15, note="Receivables should not grow much faster than sales"))
    else:
        add("Inventory", "Receivables vs Sales Growth", _rating_good_avg_bad(None))

    # FCF
    add("Free Cash Flow", "FCF per Share Trend", _trend_better(row, "fcf_per_share_3y", "fcf_per_share_5y", "FCF/share"))
    fcfs = _val(row, "fcf_sales")
    add("Free Cash Flow", "FCF / Sales", _rating_good_avg_bad(fcfs, good=(fcfs is not None and fcfs > 10), avg=(fcfs is not None and 5 <= fcfs <= 10), note="PDF rule: >10% good"))
    add("Free Cash Flow", "FCF / CFO Trend", _trend_better(row, "fcf_cfo_3y", "fcf_cfo_5y", "FCF/CFO"))
    croic = _val(row, "croic")
    add("Free Cash Flow", "CROIC", _rating_good_avg_bad(croic, good=(croic is not None and croic >= 13), avg=(croic is not None and 8 < croic < 13), note="PDF rule: ≥13% good, 8–13% average"))

    df = pd.DataFrame(factors)
    summary = summarize_factor_table(df)
    return df, summary


def score_bank(row: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []

    def add(bucket: str, metric: str, result: Dict[str, Any]):
        factors.append({"Bucket": bucket, "Metric": metric, **result})

    for key, label in [
        ("markup_growth", "Markup / Interest Earned Growth"),
        ("net_spread_growth", "Net Spread Growth"),
        ("profit_before_provision_growth", "Profit Before Provision Growth"),
        ("bank_profit_growth", "Net Profit Growth"),
    ]:
        v = _val(row, key)
        add("Bank Growth", label, _rating_good_avg_bad(v, good=(v is not None and v >= 15), avg=(v is not None and 8 <= v < 15), note="Bank lecture target ≥15%"))

    eps3 = _val(row, "eps_trend_3y")
    eps5 = _val(row, "eps_trend_5y")
    if eps3 is not None and eps5 is not None:
        add("Bank Growth", "EPS Acceleration", _rating_good_avg_bad(eps3 - eps5, good=eps3 > eps5, avg=eps3 == eps5, note="3Y EPS average vs 5Y average"))
    else:
        add("Bank Growth", "EPS Acceleration", _rating_good_avg_bad(None))

    spread = _val(row, "spread_ratio")
    add("Bank Stability", "Spread Ratio", _rating_good_avg_bad(spread, good=(spread is not None and spread >= 50), avg=(spread is not None and 30 <= spread < 50), note="PDF rule: ≥50% good"))

    npm = _val(row, "bank_net_margin")
    add("Bank Stability", "Net Profit Margin", _rating_good_avg_bad(npm, good=(npm is not None and npm >= 10), avg=(npm is not None and 5 <= npm < 10), note="PDF rule: ≥10% good"))

    nim = _val(row, "nim")
    add("Bank Stability", "Net Interest Margin", _rating_good_avg_bad(nim, good=(nim is not None and nim >= 4), avg=(nim is not None and 2.5 <= nim < 4), note="Prefer peer comparison when available"))

    npl = _val(row, "npl_ratio")
    add("Bank Asset Quality", "NPL / Gross Loans", _rating_good_avg_bad(npl, good=(npl is not None and npl <= 5), avg=(npl is not None and 5 < npl <= 8), note="PDF rule: ≤5% good"))

    add("Bank Asset Quality", "NPL Trend", _trend_better(row, "npl_trend_5y", "npl_trend_3y", "Lower 3Y NPL vs 5Y"))
    # Because _trend_better assumes short>long is good; invert by swapping inputs and note.
    if factors[-1]["rating"] != "Missing":
        val = factors[-1]["value"]
        factors[-1]["rating"] = "Good" if val > 0 else ("Average" if val == 0 else "Bad")
        factors[-1]["score"] = 2 if val > 0 else (1 if val == 0 else 0)
        factors[-1]["note"] = "Lower 3Y NPL average than 5Y average is preferred"

    p3 = _val(row, "provision_ratio_3y")
    p5 = _val(row, "provision_ratio_5y")
    if p3 is not None and p5 is not None:
        add("Bank Asset Quality", "Provision Ratio Trend", _rating_good_avg_bad(p5 - p3, good=p3 < p5, avg=p3 == p5, note="Lower 3Y provisions ratio than 5Y is preferred"))
    else:
        add("Bank Asset Quality", "Provision Ratio Trend", _rating_good_avg_bad(None))

    dep = _val(row, "deposit_cagr")
    idep = _val(row, "industry_deposit_cagr")
    if dep is not None and idep is not None:
        add("Bank Funding", "Deposit CAGR vs Industry", _rating_good_avg_bad(dep - idep, good=dep > idep, avg=dep >= idep * 0.9, note="Bank should grow deposits at least with industry"))
    else:
        add("Bank Funding", "Deposit CAGR vs Industry", _rating_good_avg_bad(None))

    casa = _val(row, "casa_ratio")
    add("Bank Funding", "CASA Ratio", _rating_good_avg_bad(casa, good=(casa is not None and casa >= 40), avg=(casa is not None and 30 <= casa < 40), note="Higher low-cost deposits are preferred"))

    adr = _val(row, "adr")
    iadr = _val(row, "industry_adr")
    if adr is not None and iadr is not None:
        # Course examples were context-specific; we score overly aggressive ADR as caution.
        add("Bank Funding", "ADR vs Industry", _rating_good_avg_bad(adr - iadr, good=adr <= iadr, avg=adr <= iadr * 1.15, note="Avoid excessive loan aggressiveness unless justified"))
    else:
        add("Bank Funding", "ADR vs Industry", _rating_good_avg_bad(None))

    car = _val(row, "car")
    add("Bank Solvency", "Capital Adequacy Ratio", _rating_good_avg_bad(car, good=(car is not None and car >= 11.5), avg=(car is not None and 10 <= car < 11.5), note="Use SBP minimum as floor"))

    ccfo = _val(row, "ccfo")
    cpat = _val(row, "cpat")
    if ccfo is not None and cpat is not None:
        add("Bank Cash Quality", "cCFO vs cPAT", _rating_good_avg_bad(ccfo - cpat, good=ccfo >= cpat, avg=ccfo >= cpat * 0.8, note="Cash generation should support reported profits"))
    else:
        add("Bank Cash Quality", "cCFO vs cPAT", _rating_good_avg_bad(None))

    df = pd.DataFrame(factors)
    summary = summarize_factor_table(df)
    return df, summary


def summarize_factor_table(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"good": 0, "average": 0, "bad": 0, "missing": 0, "score": 0, "max_score": 0, "grade": "N/A"}
    counts = df["rating"].value_counts().to_dict()
    score = float(df["score"].sum())
    max_score = float(len(df) * 2)
    pct = (score / max_score * 100) if max_score else 0
    if pct >= 85:
        grade = "A+"
    elif pct >= 75:
        grade = "A"
    elif pct >= 65:
        grade = "B+"
    elif pct >= 55:
        grade = "B"
    elif pct >= 45:
        grade = "C"
    else:
        grade = "D"
    return {
        "good": int(counts.get("Good", 0)),
        "average": int(counts.get("Average", 0)),
        "bad": int(counts.get("Bad", 0)),
        "missing": int(counts.get("Missing", 0)),
        "score": round(score, 2),
        "max_score": round(max_score, 2),
        "score_pct": round(pct, 2),
        "grade": grade,
    }


def score_fundamental_universe(df: pd.DataFrame) -> pd.DataFrame:
    norm = normalize_fundamental_columns(df)
    rows = []
    for _, row in norm.iterrows():
        detail, summary = score_bank(row) if bool(row.get("is_bank")) else score_nonbank(row)
        rows.append({
            "symbol": row["symbol"],
            "company": row.get("company"),
            "sector": row.get("sector"),
            "is_bank": bool(row.get("is_bank")),
            "fundamental_grade": summary["grade"],
            "fundamental_score_pct": summary["score_pct"],
            "good": summary["good"],
            "average": summary["average"],
            "bad": summary["bad"],
            "missing": summary["missing"],
            "detail_table": detail,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["fundamental_score_pct", "good", "bad"], ascending=[False, False, True])
    return out


def score_single_symbol(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any], pd.Series]:
    norm = normalize_fundamental_columns(df)
    match = norm[norm["symbol"].astype(str).str.upper() == str(symbol).upper()]
    if match.empty:
        raise ValueError(f"Symbol not found in fundamentals file: {symbol}")
    row = match.iloc[0]
    detail, summary = score_bank(row) if bool(row.get("is_bank")) else score_nonbank(row)
    return detail, summary, row
