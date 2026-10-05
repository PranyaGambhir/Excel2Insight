"""Stage 4 - analytics engine: descriptive stats, correlations, trends, segment comparisons."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .utils import categorical_cols, datetime_cols, is_id_like, numeric_cols

# checked in priority order: headline business metrics first, generic ones last
_VALUE_PRIORITY = [
    re.compile(p, re.I) for p in (r"revenue|sales|turnover", r"profit|income|amount|total", r"cost|spend|expense",
                                  r"qty|quantity|units|volume", r"price|value")
]
_ADDITIVE_HINTS = re.compile(r"sales|revenue|amount|profit|income|total|cost|qty|quantity|units|spend", re.I)


# --------------------------------------------------------------------------- helpers
def pick_value_column(df: pd.DataFrame) -> str | None:
    """Best guess at the 'main metric' column (revenue, sales, amount, ...)."""
    nums = [c for c in numeric_cols(df) if not is_id_like(df[c], c) and df[c].nunique() > 1]
    if not nums:
        return None
    for pattern in _VALUE_PRIORITY:
        for c in nums:
            if pattern.search(c):
                return c
    return nums[0]


def pick_date_column(df: pd.DataFrame) -> str | None:
    dates = datetime_cols(df)
    if not dates:
        return None
    return max(dates, key=lambda c: df[c].nunique())


def auto_frequency(dates: pd.Series) -> str:
    span = (dates.max() - dates.min()).days
    if span <= 45:
        return "D"
    if span <= 200:
        return "W"
    if span <= 1500:
        return "M"
    return "Q"


def _strength(r: float) -> str:
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "negligible"


# --------------------------------------------------------------------------- descriptive
def descriptive_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nums = [c for c in numeric_cols(df) if not is_id_like(df[c], c)]
    numeric = pd.DataFrame()
    if nums:
        numeric = df[nums].describe().T
        numeric["skew"] = df[nums].skew()
        numeric["kurtosis"] = df[nums].kurt()
        numeric["cv_pct"] = (numeric["std"] / numeric["mean"].abs().replace(0, np.nan) * 100)
        pvals = {}
        for c in nums:
            x = df[c].dropna()
            pvals[c] = stats.normaltest(x).pvalue if len(x) >= 20 and x.nunique() > 1 else np.nan
        numeric["normality_p"] = pd.Series(pvals)
        numeric = numeric.round(3).reset_index().rename(columns={"index": "column"})

    cats = categorical_cols(df)
    cat_rows = []
    for c in cats:
        vc = df[c].value_counts(dropna=True)
        cat_rows.append({"column": c, "unique": int(vc.size), "top": str(vc.index[0]),
                         "top_count": int(vc.iloc[0]), "top_share_pct": round(vc.iloc[0] / vc.sum() * 100, 1)})
    return {"numeric": numeric, "categorical": pd.DataFrame(cat_rows)}


# --------------------------------------------------------------------------- correlations
def correlations(df: pd.DataFrame, method: str = "pearson", top_n: int = 10) -> dict:
    nums = [c for c in numeric_cols(df) if not is_id_like(df[c], c) and df[c].nunique() > 1][:20]
    if len(nums) < 2:
        return {"matrix": pd.DataFrame(), "pairs": pd.DataFrame(), "method": method}
    matrix = df[nums].corr(method=method)
    rows = []
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            sub = df[[a, b]].dropna()
            if len(sub) < 3:
                continue
            r, p = (stats.spearmanr if method == "spearman" else stats.pearsonr)(sub[a], sub[b])
            rows.append({"variable_1": a, "variable_2": b, "r": round(float(r), 3),
                         "p_value": float(p), "strength": _strength(r), "n": len(sub)})
    pairs = pd.DataFrame(rows, columns=["variable_1", "variable_2", "r", "p_value", "strength", "n"])
    if not pairs.empty:
        pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)
    return {"matrix": matrix.round(3), "pairs": pairs, "method": method}


# --------------------------------------------------------------------------- trends
def trend_analysis(df: pd.DataFrame, date_col: str, value_col: str,
                   freq: str | None = None, agg: str | None = None) -> dict | None:
    d = df[[date_col, value_col]].dropna()
    if len(d) < 3:
        return None
    freq = freq or auto_frequency(d[date_col])
    agg = agg or ("sum" if _ADDITIVE_HINTS.search(value_col) else "mean")

    grouped = d.groupby(d[date_col].dt.to_period(freq))[value_col].agg(agg)

    # A half-finished first/last period makes a *sum* look like a collapse - drop it (and say so).
    dropped: list[str] = []
    slack = pd.Timedelta(days={"D": 0, "W": 2, "M": 3, "Q": 5, "Y": 7}[freq])
    if agg == "sum" and freq != "D" and len(grouped) > 4:
        last, first = grouped.index[-1], grouped.index[0]
        if d[date_col].max().normalize() < last.end_time.normalize() - slack:
            dropped.append(str(last))
            grouped = grouped.iloc[:-1]
        if d[date_col].min().normalize() > first.start_time.normalize() + slack:
            dropped.append(str(first))
            grouped = grouped.iloc[1:]
    if len(grouped) < 3:
        return None

    y = grouped.to_numpy(dtype=float)
    x = np.arange(len(y))
    lr = stats.linregress(x, y)
    series = pd.DataFrame({"period": grouped.index.to_timestamp(), "value": y})
    series["pct_change"] = (series["value"].pct_change() * 100).replace([np.inf, -np.inf], np.nan)
    series["trend_fit"] = lr.intercept + lr.slope * x

    first, last = float(y[0]), float(y[-1])
    total_change = (last - first) / abs(first) * 100 if first else np.nan
    significant = lr.pvalue < 0.05
    direction = ("increasing" if lr.slope > 0 else "decreasing") if significant else "no statistically significant trend"

    label = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}[freq]
    return {
        "date_col": date_col, "value_col": value_col, "freq": freq, "freq_label": label, "agg": agg,
        "n_periods": len(y), "dropped_partial_periods": dropped, "first_value": first, "last_value": last,
        "total_change_pct": None if np.isnan(total_change) else round(float(total_change), 1),
        "latest_change_pct": None if np.isnan(series["pct_change"].iloc[-1]) else round(float(series["pct_change"].iloc[-1]), 1),
        "slope_per_period": float(lr.slope), "r_squared": round(float(lr.rvalue ** 2), 3),
        "p_value": float(lr.pvalue), "direction": direction,
        "peak_period": str(grouped.idxmax()), "peak_value": float(grouped.max()),
        "trough_period": str(grouped.idxmin()), "trough_value": float(grouped.min()),
        "series": series,
    }


# --------------------------------------------------------------------------- segments
def category_breakdown(df: pd.DataFrame, cat_col: str, value_col: str,
                       agg: str = "sum", top_n: int = 10) -> dict:
    g = df.groupby(cat_col, dropna=False)[value_col].agg(agg).sort_values(ascending=False)
    table = g.head(top_n).rename("value").reset_index().rename(columns={cat_col: "category"})
    table["category"] = table["category"].astype(str)
    if agg == "sum" and g.sum() != 0:
        table["share_pct"] = (table["value"] / g.sum() * 100).round(1)

    # do the groups genuinely differ? (Kruskal-Wallis is robust to non-normal data)
    p_value = None
    groups = [x.dropna().to_numpy() for _, x in df.groupby(cat_col)[value_col] if x.notna().sum() >= 3]
    if 2 <= len(groups) <= 50:
        try:
            p_value = float(stats.kruskal(*groups).pvalue)
        except ValueError:
            p_value = None
    return {"category_col": cat_col, "value_col": value_col, "agg": agg, "table": table,
            "kruskal_p": p_value, "n_groups": int(g.size)}


# --------------------------------------------------------------------------- orchestration
def run_analytics(df: pd.DataFrame, value_col: str | None = None, date_col: str | None = None,
                  freq: str | None = None, corr_method: str = "pearson") -> dict:
    value_col = value_col or pick_value_column(df)
    date_col = date_col or pick_date_column(df)

    result: dict = {
        "descriptive": descriptive_stats(df),
        "correlation": correlations(df, corr_method),
        "value_col": value_col,
        "date_col": date_col,
        "trend": None,
        "breakdowns": [],
    }
    if value_col and date_col:
        result["trend"] = trend_analysis(df, date_col, value_col, None if freq in (None, "auto") else freq)
    if value_col:
        agg = "sum" if _ADDITIVE_HINTS.search(value_col) else "mean"
        cats = [c for c in categorical_cols(df, 30)][:3]
        result["breakdowns"] = [category_breakdown(df, c, value_col, agg) for c in cats]
    return result
