"""Stage 2 - data profiling: types, missing values, duplicates, outliers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import column_kind, numeric_cols


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float] | None:
    s = s.dropna()
    if len(s) < 8:
        return None
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - k * iqr), float(q3 + k * iqr)


def profile_dataframe(df: pd.DataFrame, iqr_k: float = 1.5) -> dict:
    n_rows, n_cols = df.shape

    # --- types -------------------------------------------------------------
    types = pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "kind": [column_kind(df[c]) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "example": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df.columns],
        }
    )

    # --- missing -----------------------------------------------------------
    miss = df.isna().sum()
    missing = pd.DataFrame(
        {"column": list(miss.index), "missing": miss.values, "missing_pct": (miss.values / max(n_rows, 1) * 100).round(2)}
    ).sort_values("missing", ascending=False, ignore_index=True)

    # --- duplicates --------------------------------------------------------
    dup_count = int(df.duplicated().sum())

    # --- outliers (IQR rule) ----------------------------------------------
    rows = []
    for c in numeric_cols(df):
        b = iqr_bounds(df[c], iqr_k)
        if b is None:
            continue
        lo, hi = b
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append(
            {"column": c, "outliers": n_out, "outlier_pct": round(n_out / max(n_rows, 1) * 100, 2),
             "lower_bound": round(lo, 3), "upper_bound": round(hi, 3)}
        )
    outliers = pd.DataFrame(rows, columns=["column", "outliers", "outlier_pct", "lower_bound", "upper_bound"])
    if not outliers.empty:
        outliers = outliers.sort_values("outliers", ascending=False, ignore_index=True)

    # --- flags & score -----------------------------------------------------
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    high_missing = missing.loc[missing["missing_pct"] > 40, "column"].tolist()
    total_cells = max(n_rows * n_cols, 1)
    missing_pct = float(miss.sum() / total_cells * 100)
    dup_pct = dup_count / max(n_rows, 1) * 100
    outlier_pct = float(outliers["outliers"].sum() / total_cells * 100) if not outliers.empty else 0.0
    quality = float(np.clip(100 - missing_pct - 0.5 * dup_pct - 0.25 * outlier_pct, 0, 100))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "types": types,
        "missing": missing,
        "missing_cells_pct": round(missing_pct, 2),
        "duplicates": dup_count,
        "duplicates_pct": round(dup_pct, 2),
        "outliers": outliers,
        "constant_columns": constant_cols,
        "high_missing_columns": high_missing,
        "quality_score": round(quality, 1),
    }
