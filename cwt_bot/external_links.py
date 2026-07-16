from __future__ import annotations

import io
import re
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


def _safe_request(url: str, *, timeout: int = 30) -> requests.Response:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 PSX-DASH/1.0"})
    response.raise_for_status()
    return response


def google_sheet_csv_export_url(sheet_url: str) -> str:
    """Convert a public/shared Google Sheets URL into a CSV-export URL."""
    url = str(sheet_url or "").strip()
    if not url:
        raise ValueError("Google Sheet link is empty.")
    if "docs.google.com/spreadsheets" not in url:
        raise ValueError("Please paste a valid Google Sheets link.")

    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Could not identify the Google Sheet ID from the link.")
    sheet_id = match.group(1)

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    gid = "0"
    if query.get("gid"):
        gid = query["gid"][0]
    else:
        frag = parsed.fragment or ""
        frag_match = re.search(r"gid=([0-9]+)", frag)
        if frag_match:
            gid = frag_match.group(1)

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


def read_google_sheet_table(sheet_url: str) -> pd.DataFrame:
    export_url = google_sheet_csv_export_url(sheet_url)
    response = _safe_request(export_url)
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type and b"Google" in response.content[:5000]:
        raise ValueError("Google Sheet could not be exported. Please enable link access or publish/share the sheet.")
    try:
        df = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        raise ValueError(f"Google Sheet was downloaded but could not be read as CSV: {exc}") from exc
    if df.empty:
        raise ValueError("Google Sheet loaded, but it is empty.")
    return df


def google_drive_file_id(drive_url: str) -> str:
    url = str(drive_url or "").strip()
    if not url:
        raise ValueError("Google Drive PDF link is empty.")
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not identify the Google Drive file ID. Paste a standard share link for the PDF.")


def google_drive_download_url(drive_url: str) -> str:
    file_id = google_drive_file_id(drive_url)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _download_google_drive_pdf(drive_url: str) -> bytes:
    response = _safe_request(google_drive_download_url(drive_url), timeout=45)
    content = response.content
    content_type = response.headers.get("Content-Type", "").lower()
    if content.lstrip().startswith(b"%PDF"):
        return content
    if "html" in content_type:
        raise ValueError("Google Drive returned a webpage instead of the PDF. Ensure the PDF is shared with link access and is not blocked by a download confirmation page.")
    raise ValueError("The Google Drive link did not return a PDF file.")


def _clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _table_to_frame(table: list[list[object]]) -> pd.DataFrame | None:
    if not table or len(table) < 2:
        return None
    cleaned = [[_clean_cell(cell) for cell in row] for row in table]
    # Remove fully blank rows.
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if len(cleaned) < 2:
        return None
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    headers = padded[0]
    if sum(bool(h) for h in headers) < max(1, width // 3):
        return None
    headers = [h or f"Column_{idx+1}" for idx, h in enumerate(headers)]
    frame = pd.DataFrame(padded[1:], columns=headers)
    frame = frame.dropna(how="all")
    return frame if not frame.empty else None


def _portfolio_relevance_score(frame: pd.DataFrame) -> int:
    joined = " ".join(str(col).lower() for col in frame.columns)
    score = 0
    tokens = {
        "symbol": 5, "ticker": 5, "qty": 4, "quantity": 4, "avg": 3, "buy": 3,
        "price": 2, "mtm": 2, "investment": 2, "pnl": 2, "profit": 2,
    }
    for token, points in tokens.items():
        if token in joined:
            score += points
    score += min(len(frame), 20) // 5
    return score


def read_google_drive_portfolio_pdf(drive_url: str) -> pd.DataFrame:
    """Download a shared Google Drive portfolio PDF and extract the most relevant holdings table."""
    pdf_bytes = _download_google_drive_pdf(drive_url)
    try:
        import pdfplumber
    except ImportError as exc:
        raise ValueError("PDF portfolio extraction requires pdfplumber. Install dependencies with: pip install -r requirements.txt") from exc

    frames: list[pd.DataFrame] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table in tables:
                frame = _table_to_frame(table)
                if frame is not None:
                    frames.append(frame)

    if not frames:
        raise ValueError("The PDF downloaded correctly, but no extractable portfolio table was found. Use a text/table PDF or upload CSV/XLSX as a fallback.")

    ranked = sorted(frames, key=lambda frame: (_portfolio_relevance_score(frame), len(frame)), reverse=True)
    best = ranked[0]
    if _portfolio_relevance_score(best) <= 0:
        raise ValueError("A table was found in the PDF, but it did not look like a portfolio holdings table. Upload CSV/XLSX or use a PDF with clear table headers.")
    return best.reset_index(drop=True)
