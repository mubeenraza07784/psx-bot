from __future__ import annotations

import io
import math
import re
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

from .fundamental_master import ALIASES


IMAGE_SECTION_ORDER = [
    "Growth",
    "Stability",
    "Valuation",
    "Inventory",
    "Cashflow",
    "Main Page / Margin of Safety",
]


def _clean_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or "metric"


def _parse_numeric(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    text = text.replace("Rs.", "").replace("PKR", "").replace("₨", "")
    text = text.replace("%", "")
    text = text.replace("x", "")
    text = text.replace("—", "").replace("-", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def canonical_metric_key(label: str) -> str | None:
    metric_clean = _clean_key(label)
    if not metric_clean:
        return None
    for target, aliases in ALIASES.items():
        for alias in aliases:
            alias_clean = _clean_key(alias)
            if not alias_clean:
                continue
            if alias_clean == metric_clean or alias_clean in metric_clean or metric_clean in alias_clean:
                return target

    custom_matches = {
        "marginofsafety": "margin_of_safety_pct",
        "safetymargin": "margin_of_safety_pct",
        "fairvalue": "intrinsic_value",
        "intrinsicvalue": "intrinsic_value",
        "estimatedvalue": "intrinsic_value",
        "upside": "upside_pct",
        "downside": "downside_pct",
        "currentprice": "price",
        "marketprice": "price",
        "discounttofairvalue": "margin_of_safety_pct",
    }
    for match, target in custom_matches.items():
        if match in metric_clean or metric_clean in match:
            return target
    return None


def _ocr_image_text(image_bytes: bytes) -> tuple[str, str]:
    """Best-effort local OCR. Requires Pillow + pytesseract + Tesseract executable."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception as exc:  # pragma: no cover - environment dependent
        return "", f"Pillow is not available: {exc}"

    try:
        import pytesseract
    except Exception as exc:  # pragma: no cover - environment dependent
        return "", f"pytesseract is not available: {exc}"

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Mild OCR preparation for dashboard screenshots: enlarge, grayscale, increase contrast.
        scale = 2 if max(image.size) < 2400 else 1
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale))
        gray = ImageOps.grayscale(image)
        gray = ImageEnhance.Contrast(gray).enhance(1.35)
        gray = gray.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(gray, config="--psm 6")
        return str(text or ""), "OCR complete"
    except Exception as exc:  # pragma: no cover - environment dependent
        return "", f"OCR could not run. Install the Tesseract desktop engine if needed. Details: {exc}"


def _candidate_line_pairs(line: str) -> Iterable[tuple[str, str]]:
    original = re.sub(r"\s+", " ", str(line or "").strip())
    if not original:
        return []

    # Metric: value, Metric | value, or Metric - value forms.
    for sep in [":", "|", "→", "=>"]:
        if sep in original:
            left, right = original.split(sep, 1)
            if left.strip() and right.strip():
                return [(left.strip(), right.strip())]

    # Last numeric token is often the value in screenshots.
    # Examples: "Revenue CAGR 3Y 18.4%", "P/E 7.6x".
    match = re.match(r"^(.*?)[\s]+([-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|X|Rs\.?|PKR)?)$", original)
    if match:
        return [(match.group(1).strip(), match.group(2).strip())]

    # Value first layouts: "18.4% Revenue CAGR 3Y".
    match = re.match(r"^([-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|x|X|Rs\.?|PKR)?)[\s]+(.*)$", original)
    if match:
        return [(match.group(2).strip(), match.group(1).strip())]

    return []


def parse_metric_value_text(text: str, category: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if len(line) < 3:
            continue
        for metric, value_text in _candidate_line_pairs(line):
            value_num = _parse_numeric(value_text)
            if value_num is None:
                continue
            canonical = canonical_metric_key(metric)
            rows.append({
                "Category": category,
                "Metric": metric,
                "Canonical Key": canonical or "",
                "Value": value_num,
                "Raw Value": value_text,
                "Raw Line": line,
                "Extraction": "OCR",
            })
    return pd.DataFrame(rows)


def long_metrics_from_google_sheet(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame | None:
    """Accept a Google Sheet in long form: Category, Metric, Value columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    lower = {_clean_key(col): col for col in df.columns}
    metric_col = lower.get("metric") or lower.get("indicator") or lower.get("ratio")
    value_col = lower.get("value") or lower.get("metricvalue") or lower.get("reading")
    if metric_col is None or value_col is None:
        return None
    category_col = lower.get("category") or lower.get("section") or lower.get("bucket")
    symbol_col = lower.get("symbol") or lower.get("ticker")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        metric = str(row.get(metric_col, "")).strip()
        value_text = str(row.get(value_col, "")).strip()
        value_num = _parse_numeric(value_text)
        if not metric or value_num is None:
            continue
        category = str(row.get(category_col, "Google Sheet")).strip() if category_col else "Google Sheet"
        row_symbol = str(row.get(symbol_col, symbol)).strip().upper() if symbol_col else str(symbol or "").strip().upper()
        rows.append({
            "Category": category or "Google Sheet",
            "Metric": metric,
            "Canonical Key": canonical_metric_key(metric) or "",
            "Value": value_num,
            "Raw Value": value_text,
            "Raw Line": f"{metric}: {value_text}",
            "Extraction": "Google Sheet",
            "Symbol": row_symbol,
        })
    return pd.DataFrame(rows)


def metric_long_to_wide(
    long_df: pd.DataFrame | None,
    *,
    symbol: str,
    company: str = "",
    sector: str = "Unknown",
) -> pd.DataFrame:
    symbol_clean = str(symbol or "").strip().upper()
    if not symbol_clean:
        raise ValueError("Symbol is required before converting imported metrics into bot fundamentals.")

    row: dict[str, Any] = {
        "symbol": symbol_clean,
        "company": str(company or symbol_clean).strip() or symbol_clean,
        "sector": str(sector or "Unknown").strip() or "Unknown",
    }
    category_counts: dict[str, int] = {}
    if isinstance(long_df, pd.DataFrame) and not long_df.empty:
        for _, metric_row in long_df.iterrows():
            category = str(metric_row.get("Category", "Imported")).strip() or "Imported"
            category_counts[category] = category_counts.get(category, 0) + 1
            metric = str(metric_row.get("Metric", "")).strip()
            canonical = str(metric_row.get("Canonical Key", "")).strip()
            value = metric_row.get("Value")
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            col = canonical or f"imported_{_slug(category)}_{_slug(metric)}"
            # Prefer the first canonical value; OCR duplicates are common.
            row.setdefault(col, value)

    for category, count in category_counts.items():
        row[f"imported_count_{_slug(category)}"] = int(count)
    return pd.DataFrame([row])


def process_image_bundle(
    images: Dict[str, bytes | None],
    *,
    symbol: str,
    company: str = "",
    sector: str = "Unknown",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return OCR status, long metrics, and a wide single-symbol fundamentals row."""
    statuses: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for category in IMAGE_SECTION_ORDER:
        image_bytes = images.get(category)
        if not image_bytes:
            statuses.append({
                "Image Slot": category,
                "Uploaded": "No",
                "OCR Status": "No image uploaded",
                "Parsed Metrics": 0,
            })
            continue
        text, status = _ocr_image_text(bytes(image_bytes))
        parsed = parse_metric_value_text(text, category)
        if not parsed.empty:
            frames.append(parsed)
        statuses.append({
            "Image Slot": category,
            "Uploaded": "Yes",
            "OCR Status": status,
            "Parsed Metrics": int(len(parsed)),
        })

    status_df = pd.DataFrame(statuses)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Category", "Metric", "Canonical Key", "Value", "Raw Value", "Raw Line", "Extraction"]
    )
    wide_df = metric_long_to_wide(long_df, symbol=symbol, company=company, sector=sector)
    return status_df, long_df, wide_df


def combine_structured_fundamentals(
    image_wide_df: pd.DataFrame | None,
    google_sheet_df: pd.DataFrame | None,
    *,
    symbol: str,
    company: str = "",
    sector: str = "Unknown",
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Build a final fundamentals table. Structured Google Sheet values override OCR values."""
    long_google = long_metrics_from_google_sheet(google_sheet_df, symbol=symbol)
    google_wide: pd.DataFrame | None = None
    if long_google is not None and not long_google.empty:
        google_wide = metric_long_to_wide(long_google, symbol=symbol, company=company, sector=sector)
    elif isinstance(google_sheet_df, pd.DataFrame) and not google_sheet_df.empty:
        google_wide = google_sheet_df.copy()

    if image_wide_df is None or image_wide_df.empty:
        return google_wide, long_google
    if google_wide is None or google_wide.empty:
        return image_wide_df.copy(), long_google

    # If the Google Sheet is a multi-symbol wide table, prefer it as the final canonical source.
    sheet_cols_clean = {_clean_key(col) for col in google_wide.columns}
    if "symbol" in sheet_cols_clean or "ticker" in sheet_cols_clean:
        try:
            symbols = google_wide[[col for col in google_wide.columns if _clean_key(col) in {"symbol", "ticker"}][0]].astype(str).str.upper().str.strip()
            if symbol.strip().upper() in set(symbols):
                return google_wide.copy(), long_google
        except Exception:
            pass

    image_row = image_wide_df.iloc[0].to_dict()
    sheet_row = google_wide.iloc[0].to_dict()
    # OCR row provides blanks; Google Sheet overrides keys when both are present.
    merged = {**image_row, **{k: v for k, v in sheet_row.items() if not (pd.isna(v) if not isinstance(v, (str, bytes)) else False)}}
    merged.setdefault("symbol", symbol.strip().upper())
    merged.setdefault("company", company.strip() or symbol.strip().upper())
    merged.setdefault("sector", sector.strip() or "Unknown")
    return pd.DataFrame([merged]), long_google
