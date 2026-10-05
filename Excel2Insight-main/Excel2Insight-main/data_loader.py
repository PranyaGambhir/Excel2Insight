"""Stage 1 - read an uploaded CSV / Excel file into a typed Pandas DataFrame."""
from __future__ import annotations

import io
import re
from typing import BinaryIO

import pandas as pd
from pandas.api import types as pdt

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")
NULL_TOKENS = ["", " ", "na", "n/a", "N/A", "NA", "null", "NULL", "none", "None", "nan", "NaN", "-", "--", "?"]

_DATE_PATTERN = re.compile(
    r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"              # 2024-03-05, 05/03/2024
    r"|\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}"           # 5 Mar 2024
    r"|[A-Za-z]{3,9}\s\d{1,2},?\s\d{4}"             # March 5, 2024
)


_DMY = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}")


def detect_dayfirst(values: pd.Series) -> bool:
    """05/03/2024 is ambiguous. Look for a value that settles it (e.g. 25/03/2024 => day first)."""
    parts = values.astype(str).str.extract(_DMY).dropna().astype(int)
    if parts.empty:
        return False
    if (parts[0] > 12).any():
        return True
    return False


def _to_bytes(source: BinaryIO | bytes) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return source.getvalue()
    source.seek(0)
    return source.read()


def list_sheets(source: BinaryIO | bytes, filename: str) -> list[str]:
    """Sheet names for Excel files (empty list for CSV)."""
    if not filename.lower().endswith((".xlsx", ".xls")):
        return []
    return pd.ExcelFile(io.BytesIO(_to_bytes(source))).sheet_names


def load_data(source: BinaryIO | bytes, filename: str, sheet_name: str | int | None = None,
              dayfirst: bool | None = None) -> pd.DataFrame:
    """Read the file and infer better dtypes (numbers stored as text, date strings)."""
    name = filename.lower()
    raw = _to_bytes(source)

    if name.endswith(".csv"):
        df = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python", na_values=NULL_TOKENS)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode the CSV file (tried UTF-8 and Latin-1).")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name or 0, na_values=NULL_TOKENS)
    else:
        raise ValueError(f"Unsupported file type. Please upload one of: {', '.join(SUPPORTED_EXTENSIONS)}")

    if df.empty:
        raise ValueError("The file was read successfully but contains no rows.")

    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return infer_types(df, dayfirst=dayfirst)


def infer_types(df: pd.DataFrame, threshold: float = 0.9, dayfirst: bool | None = None) -> pd.DataFrame:
    """Convert text columns that are really numbers or dates.

    dayfirst=None auto-detects DD/MM vs MM/DD per column; pass True/False to force it."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pdt.is_numeric_dtype(s) or pdt.is_datetime64_any_dtype(s) or pdt.is_bool_dtype(s):
            continue
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            continue

        # numbers hidden in text: "1,200", "$45.50", "12%"
        strip_chars = r"[,\s$€£₹%]"
        numeric = pd.to_numeric(non_null.str.replace(strip_chars, "", regex=True), errors="coerce")
        if numeric.notna().mean() >= threshold:
            df[col] = pd.to_numeric(s.astype(str).str.replace(strip_chars, "", regex=True).where(s.notna()), errors="coerce")
            continue

        # date strings
        if non_null.str.contains(_DATE_PATTERN).mean() >= threshold:
            dfirst = detect_dayfirst(non_null) if dayfirst is None else dayfirst
            kwargs = {"dayfirst": True} if dfirst else {}
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed", **kwargs)
            if parsed.notna().mean() >= threshold:
                df[col] = pd.to_datetime(s, errors="coerce", format="mixed", **kwargs)
    return df
