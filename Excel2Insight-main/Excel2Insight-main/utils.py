"""Small helpers shared by every backend module."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pdt.is_numeric_dtype(df[c]) and not pdt.is_bool_dtype(df[c])]


def datetime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pdt.is_datetime64_any_dtype(df[c])]


def categorical_cols(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    """Non-numeric, non-datetime columns with a manageable number of distinct values."""
    out = []
    for c in df.columns:
        s = df[c]
        if pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
            continue
        if pdt.is_datetime64_any_dtype(s):
            continue
        if 1 < s.nunique(dropna=True) <= max_unique:
            out.append(c)
    return out


def is_id_like(s: pd.Series, name: str) -> bool:
    """Heuristic: identifier columns are not worth summarising or plotting."""
    n = s.notna().sum()
    if n == 0:
        return False
    name_hint = name.lower().replace(" ", "_").endswith(("id", "_id", "_no", "number", "index"))
    unique_ratio = s.nunique(dropna=True) / n
    is_int = pdt.is_integer_dtype(s)
    return bool(name_hint and unique_ratio > 0.9) or bool(is_int and unique_ratio > 0.98 and n > 20)


def column_kind(s: pd.Series) -> str:
    if pdt.is_bool_dtype(s):
        return "boolean"
    if pdt.is_numeric_dtype(s):
        return "numeric"
    if pdt.is_datetime64_any_dtype(s):
        return "datetime"
    n = max(len(s), 1)
    if s.nunique(dropna=True) <= max(20, int(0.05 * n)):
        return "categorical"
    return "text"
