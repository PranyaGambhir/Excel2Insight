"""Stage 3 - automatic, transparent dataset cleaning (every action is logged)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from .profiling import iqr_bounds
from .utils import numeric_cols


@dataclass
class CleaningOptions:
    drop_duplicates: bool = True
    drop_empty: bool = True
    missing_col_threshold: float = 0.6      # drop columns with more than 60% missing
    numeric_impute: str = "median"          # median | mean | drop
    categorical_impute: str = "mode"        # mode | unknown | drop
    outliers: str = "none"                  # none | cap | remove
    iqr_k: float = 1.5


def clean_dataframe(df: pd.DataFrame, opts: CleaningOptions | None = None) -> tuple[pd.DataFrame, list[str]]:
    opts = opts or CleaningOptions()
    df = df.copy()
    log: list[str] = []
    start_rows, start_cols = df.shape

    # 1. tidy text values
    for c in df.columns:
        if not (pdt.is_numeric_dtype(df[c]) or pdt.is_datetime64_any_dtype(df[c]) or pdt.is_bool_dtype(df[c])):
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
    log.append("Trimmed whitespace in text columns.")

    # 2. empty rows / columns
    if opts.drop_empty:
        before = df.shape
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.shape != before:
            log.append(f"Dropped {before[0] - df.shape[0]} empty rows and {before[1] - df.shape[1]} empty columns.")

    # 3. columns that are mostly missing
    too_empty = [c for c in df.columns if df[c].isna().mean() > opts.missing_col_threshold]
    if too_empty:
        df = df.drop(columns=too_empty)
        log.append(f"Dropped columns with >{opts.missing_col_threshold:.0%} missing values: {', '.join(too_empty)}.")

    # 4. duplicates
    if opts.drop_duplicates:
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates(ignore_index=True)
            log.append(f"Removed {n} duplicate rows.")

    # 5. missing values
    for c in list(df.columns):
        n_missing = int(df[c].isna().sum())
        if n_missing == 0:
            continue
        s = df[c]
        if pdt.is_datetime64_any_dtype(s):
            log.append(f"'{c}': {n_missing} missing dates left as-is (excluded from time-series analysis).")
        elif pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s):
            if opts.numeric_impute == "drop":
                df = df[df[c].notna()]
                log.append(f"'{c}': dropped {n_missing} rows with missing values.")
            else:
                fill = s.median() if opts.numeric_impute == "median" else s.mean()
                df[c] = s.fillna(fill)
                log.append(f"'{c}': filled {n_missing} missing values with the {opts.numeric_impute} ({fill:.3g}).")
        else:
            if opts.categorical_impute == "drop":
                df = df[df[c].notna()]
                log.append(f"'{c}': dropped {n_missing} rows with missing values.")
            else:
                mode = s.mode(dropna=True)
                fill = mode.iloc[0] if (opts.categorical_impute == "mode" and not mode.empty) else "Unknown"
                df[c] = s.fillna(fill)
                log.append(f"'{c}': filled {n_missing} missing values with '{fill}'.")
    df = df.reset_index(drop=True)

    # 6. outliers (optional - off by default because outliers are often real business events)
    if opts.outliers != "none":
        for c in numeric_cols(df):
            b = iqr_bounds(df[c], opts.iqr_k)
            if b is None:
                continue
            lo, hi = b
            mask = (df[c] < lo) | (df[c] > hi)
            n = int(mask.sum())
            if not n:
                continue
            if opts.outliers == "cap":
                df[c] = df[c].clip(lo, hi)
                log.append(f"'{c}': capped {n} outliers to [{lo:.3g}, {hi:.3g}].")
            else:
                df = df[~mask]
                log.append(f"'{c}': removed {n} outlier rows.")
        df = df.reset_index(drop=True)

    log.append(f"Result: {start_rows} x {start_cols}  ->  {df.shape[0]} x {df.shape[1]} (rows x columns).")
    return df, log
