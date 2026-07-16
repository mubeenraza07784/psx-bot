from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import numpy as np


RED_ZONE_SECTORS = {
    "banks", "commercial banks", "oil", "oil & gas", "oil and gas",
    "fertilizer", "fertilizers", "utility", "utilities", "power", "chemicals", "chemical",
}

def _sector_match(sector: Any) -> bool:
    s = str(sector or "").strip().lower()
    return any(key in s for key in RED_ZONE_SECTORS)


def _top_percent(df: pd.DataFrame, score_col: str, pct: float = 0.10) -> pd.DataFrame:
    if df.empty or score_col not in df.columns:
        return pd.DataFrame()
    n = max(1, int(np.ceil(len(df) * pct)))
    return df.sort_values(score_col, ascending=False).head(n).copy()


def build_strategy_1(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 1:
    Invest Top 10% companies each year based on Scoring Model.
    """
    out = _top_percent(scored, "fundamental_score_pct", 0.10)
    if not out.empty:
        out["Strategy"] = "Strategy 1 — Top 10% Score"
        out["Selection Reason"] = "Top 10% by composite fundamental score"
    return out


def build_strategy_2(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 2:
    Strategy 1 + red-zone sector preference.
    """
    top = build_strategy_1(scored)
    if top.empty:
        return top
    red = top[top["sector"].apply(_sector_match)].copy()
    if red.empty:
        red = top.copy()
        red["Selection Reason"] = "No red-zone names in top 10%; showing Strategy 1 basket"
    else:
        red["Selection Reason"] = "Top score + red-zone sector filter"
    red["Strategy"] = "Strategy 2 — Top Score + Red Zone"
    return red


def build_strategy_3(scored: pd.DataFrame, lowest_price_count: int = 5) -> pd.DataFrame:
    """
    Strategy 3:
    Strategy 2 + lowest-price / high-beta rotation.
    """
    if scored.empty:
        return pd.DataFrame()
    base = build_strategy_2(scored)
    universe = base if not base.empty else scored.copy()

    if "price" not in universe.columns:
        return pd.DataFrame()
    work = universe.copy()
    if "beta" in work.columns:
        work["beta_rank"] = pd.to_numeric(work["beta"], errors="coerce").rank(ascending=False, method="min")
    else:
        work["beta_rank"] = np.nan
    work["price_rank"] = pd.to_numeric(work["price"], errors="coerce").rank(ascending=True, method="min")
    work["rotation_score"] = work["price_rank"].fillna(work["price_rank"].max() if work["price_rank"].notna().any() else 999)
    if work["beta_rank"].notna().any():
        work["rotation_score"] = work["rotation_score"] + work["beta_rank"].fillna(work["beta_rank"].max())
    out = work.sort_values(["rotation_score", "price"], ascending=[True, True]).head(lowest_price_count).copy()
    out["Strategy"] = "Strategy 3 — Red Zone + Low Price / High Beta"
    out["Selection Reason"] = "Lowest-price candidates with beta preference within the selected quality universe"
    return out


def build_strategy_4(scored: pd.DataFrame, minimum_margin_of_safety_pct: float = 25.0) -> pd.DataFrame:
    """
    Strategy 4:
    Strategy 3 + intrinsic value FCF/cash margin of safety filter.
    """
    base = build_strategy_3(scored)
    if base.empty:
        return base
    mos_cols = [c for c in ["best_margin_of_safety_pct", "margin_of_safety_fcf_pct", "margin_of_safety_cash_pct"] if c in base.columns]
    if not mos_cols:
        return pd.DataFrame()
    work = base.copy()
    work["strategy4_margin_of_safety"] = work[mos_cols].max(axis=1, skipna=True)
    out = work[work["strategy4_margin_of_safety"] >= minimum_margin_of_safety_pct].copy()
    out["Strategy"] = "Strategy 4 — Strategy 3 + Intrinsic Value Filter"
    out["Selection Reason"] = f"Strategy 3 candidate with margin of safety ≥{minimum_margin_of_safety_pct:.0f}%"
    return out


def build_all_strategies(scored: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "Strategy 1": build_strategy_1(scored),
        "Strategy 2": build_strategy_2(scored),
        "Strategy 3": build_strategy_3(scored),
        "Strategy 4": build_strategy_4(scored),
    }


def strategy_summary(strategies: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in strategies.items():
        rows.append({
            "Strategy": name,
            "Candidates": int(len(df)) if df is not None else 0,
            "Symbols": ", ".join(df["symbol"].astype(str).tolist()) if df is not None and not df.empty and "symbol" in df.columns else "",
        })
    return pd.DataFrame(rows)
