from __future__ import annotations

from typing import Dict, Iterable, Any
import numpy as np
import pandas as pd


ALIASES = {
    "symbol": ["symbol", "ticker", "code", "script"],
    "company": ["company", "name", "company_name"],
    "sector": ["sector", "industry"],
    "pe": ["pe", "p/e", "price_to_earnings", "price earning ratio"],
    "pb": ["pb", "p/b", "price_to_book", "price book"],
    "roe": ["roe", "return_on_equity"],
    "roa": ["roa", "return_on_assets"],
    "dividend_yield": ["dividend_yield", "div yield", "yield", "dividend yield"],
    "debt_to_equity": ["debt_to_equity", "de", "d/e", "debt equity"],
    "eps_growth": ["eps_growth", "eps growth", "earnings_growth", "earnings growth"],
    "revenue_growth": ["revenue_growth", "revenue growth", "sales_growth", "sales growth"],
    "net_margin": ["net_margin", "net margin", "profit_margin", "profit margin"],
    "fcf_yield": ["fcf_yield", "fcf yield", "free_cash_flow_yield"],
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


def _clean(text: Any) -> str:
    return "".join(ch for ch in str(text).strip().lower() if ch.isalnum())


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    map_clean = {_clean(c): c for c in df.columns}
    for alias in aliases:
        key = _clean(alias)
        if key in map_clean:
            return map_clean[key]
    for col in df.columns:
        c = _clean(col)
        for alias in aliases:
            a = _clean(alias)
            if a and (a in c or c in a):
                return col
    return None


def _minmax_score(series: pd.Series, higher_is_better: bool = True, neutral: float = 50.0) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = _collapse_duplicate_columns(series).iloc[:, 0]
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(neutral, index=series.index)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(neutral, index=series.index)
    clipped = s.clip(lo, hi)
    score = (clipped - lo) / (hi - lo) * 100
    if not higher_is_better:
        score = 100 - score
    return score.fillna(neutral)


def prepare_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    raw = df.copy()
    ren: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        col = _find_col(raw, aliases)
        if col is not None:
            ren[col] = target
    out = raw.rename(columns=ren).copy()
    out = _collapse_duplicate_columns(out)

    if "symbol" not in out.columns:
        raise ValueError("Fundamentals file needs a Symbol/Ticker column.")

    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    if "company" not in out.columns:
        out["company"] = out["symbol"]
    if "sector" not in out.columns:
        out["sector"] = "Unknown"

    numeric_cols = [
        "pe", "pb", "roe", "roa", "dividend_yield", "debt_to_equity",
        "eps_growth", "revenue_growth", "net_margin", "fcf_yield",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = np.nan
        value = out[col]
        if isinstance(value, pd.DataFrame):
            value = _collapse_duplicate_columns(value).iloc[:, 0]
        out[col] = pd.to_numeric(value, errors="coerce")
    out = _collapse_duplicate_columns(out)

    value_parts = pd.DataFrame({
        "pe_score": _minmax_score(out["pe"], higher_is_better=False),
        "pb_score": _minmax_score(out["pb"], higher_is_better=False),
        "fcf_score": _minmax_score(out["fcf_yield"], higher_is_better=True),
    })
    quality_parts = pd.DataFrame({
        "roe_score": _minmax_score(out["roe"], higher_is_better=True),
        "roa_score": _minmax_score(out["roa"], higher_is_better=True),
        "margin_score": _minmax_score(out["net_margin"], higher_is_better=True),
        "debt_score": _minmax_score(out["debt_to_equity"], higher_is_better=False),
    })
    growth_parts = pd.DataFrame({
        "eps_growth_score": _minmax_score(out["eps_growth"], higher_is_better=True),
        "revenue_growth_score": _minmax_score(out["revenue_growth"], higher_is_better=True),
    })
    dividend_parts = pd.DataFrame({
        "yield_score": _minmax_score(out["dividend_yield"], higher_is_better=True),
    })

    out["value_score"] = value_parts.mean(axis=1).round(2)
    out["quality_score"] = quality_parts.mean(axis=1).round(2)
    out["growth_score"] = growth_parts.mean(axis=1).round(2)
    out["dividend_score"] = dividend_parts.mean(axis=1).round(2)

    out["fundamental_score"] = (
        0.35 * out["quality_score"]
        + 0.30 * out["value_score"]
        + 0.20 * out["growth_score"]
        + 0.15 * out["dividend_score"]
    ).round(2)

    out["fundamental_grade"] = pd.cut(
        out["fundamental_score"],
        bins=[-np.inf, 45, 60, 75, 85, np.inf],
        labels=["Weak", "Average", "Strong", "Very Strong", "Elite"],
    ).astype(str)

    out["valuation"] = pd.cut(
        out["value_score"],
        bins=[-np.inf, 45, 60, 75, np.inf],
        labels=["Expensive", "Fair", "Attractive", "Cheap"],
    ).astype(str)

    out["dividend_profile"] = pd.cut(
        out["dividend_score"],
        bins=[-np.inf, 45, 60, 75, np.inf],
        labels=["Low", "Moderate", "Good", "Strong"],
    ).astype(str)

    return out


def build_fundamental_rankings(fundamentals: pd.DataFrame, watchlist: Iterable[str] | None = None) -> Dict[str, pd.DataFrame]:
    if fundamentals is None or fundamentals.empty:
        return {}

    f = fundamentals.copy()
    sector_ranking = (
        f.groupby("sector", dropna=False)
        .agg(
            companies=("symbol", "count"),
            avg_fundamental_score=("fundamental_score", "mean"),
            avg_quality=("quality_score", "mean"),
            avg_value=("value_score", "mean"),
            avg_dividend=("dividend_score", "mean"),
            median_roe=("roe", "median"),
            median_yield=("dividend_yield", "median"),
        )
        .reset_index()
        .sort_values("avg_fundamental_score", ascending=False)
    )

    top_strongest = f.sort_values("fundamental_score", ascending=False).head(20)
    dividend_shortlist = f.sort_values(["dividend_score", "dividend_yield"], ascending=[False, False]).head(20)
    undervalued_shortlist = f.assign(
        undervalued_score=(0.65 * f["value_score"] + 0.35 * f["quality_score"]).round(2)
    ).sort_values("undervalued_score", ascending=False).head(20)

    output = {
        "sector_ranking": sector_ranking,
        "top_fundamentals": top_strongest,
        "dividend_shortlist": dividend_shortlist,
        "undervalued_shortlist": undervalued_shortlist,
    }

    if watchlist:
        wl = {str(x).upper().strip() for x in watchlist if str(x).strip()}
        scorecard = f[f["symbol"].isin(wl)].copy().sort_values("fundamental_score", ascending=False)
        output["watchlist_fundamental_scorecard"] = scorecard

    return output
