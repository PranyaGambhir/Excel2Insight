"""Stage 5 - interactive Plotly charts and the auto-generated dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import is_id_like, numeric_cols

TEMPLATE = "plotly_white"
COLORS = px.colors.qualitative.Set2


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=20, r=20, t=60, b=20),
                      title_font_size=16, legend_title_text="")
    return fig


def fig_trend(trend: dict) -> go.Figure:
    s = trend["series"]
    fig = go.Figure()
    fig.add_scatter(x=s["period"], y=s["value"], mode="lines+markers", name=trend["value_col"], line=dict(width=3))
    fig.add_scatter(x=s["period"], y=s["trend_fit"], mode="lines", name="Linear trend", line=dict(dash="dash"))
    fig.update_layout(title=f"{trend['agg'].title()} of {trend['value_col']} per {trend['freq_label']}",
                      xaxis_title=trend["date_col"], yaxis_title=trend["value_col"])
    return _style(fig)


def fig_growth(trend: dict) -> go.Figure:
    s = trend["series"].dropna(subset=["pct_change"])
    fig = px.bar(s, x="period", y="pct_change", color=(s["pct_change"] >= 0).map({True: "Growth", False: "Decline"}),
                 color_discrete_map={"Growth": "#2a9d8f", "Decline": "#e76f51"},
                 title=f"Period-over-period change (%) per {trend['freq_label']}")
    return _style(fig, 340)


def fig_category(bd: dict) -> go.Figure:
    t = bd["table"].sort_values("value")
    fig = px.bar(t, x="value", y="category", orientation="h", text_auto=".3s",
                 title=f"{bd['agg'].title()} of {bd['value_col']} by {bd['category_col']}",
                 labels={"value": bd["value_col"], "category": bd["category_col"]})
    fig.update_traces(marker_color=COLORS[0])
    return _style(fig, max(320, 40 * len(t) + 120))


def fig_corr_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", title="Correlation matrix")
    return _style(fig, 500)


def fig_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, marginal="box", title=f"Distribution of {col}")
    return _style(fig, 400)


def fig_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None) -> go.Figure:
    data = df.sample(5000, random_state=0) if len(df) > 5000 else df
    fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7, title=f"{y} vs {x}")
    return _style(fig)


def fig_missing(missing: pd.DataFrame) -> go.Figure:
    m = missing[missing["missing"] > 0].sort_values("missing_pct")
    fig = px.bar(m, x="missing_pct", y="column", orientation="h", title="Missing values by column (%)",
                 labels={"missing_pct": "% missing"})
    fig.update_traces(marker_color="#e76f51")
    return _style(fig, max(300, 32 * len(m) + 120))


def custom_chart(df: pd.DataFrame, kind: str, x: str, y: str | None = None, color: str | None = None) -> go.Figure:
    data = df.sample(5000, random_state=0) if len(df) > 5000 and kind == "scatter" else df
    if kind == "histogram":
        fig = px.histogram(data, x=x, color=color, marginal="box")
    elif kind == "box":
        fig = px.box(data, x=color, y=x) if color else px.box(data, y=x)
    elif kind == "line":
        fig = px.line(data.sort_values(x), x=x, y=y, color=color)
    elif kind == "bar":
        fig = px.bar(data.groupby(x, as_index=False)[y].sum(), x=x, y=y)
    else:
        fig = px.scatter(data, x=x, y=y, color=color, opacity=0.7)
    return _style(fig)


def build_dashboard(df: pd.DataFrame, analytics: dict, profile: dict | None = None) -> dict[str, go.Figure]:
    """Pick sensible charts automatically from the analytics results."""
    figs: dict[str, go.Figure] = {}
    trend = analytics.get("trend")
    if trend:
        figs["Trend"] = fig_trend(trend)
        figs["Growth"] = fig_growth(trend)
    for bd in analytics.get("breakdowns", [])[:2]:
        figs[f"By {bd['category_col']}"] = fig_category(bd)

    corr = analytics["correlation"]
    if not corr["matrix"].empty:
        figs["Correlations"] = fig_corr_heatmap(corr["matrix"])
        if not corr["pairs"].empty:
            top = corr["pairs"].iloc[0]
            figs[f"{top['variable_1']} vs {top['variable_2']}"] = fig_scatter(df, top["variable_1"], top["variable_2"])

    nums = [c for c in numeric_cols(df) if not is_id_like(df[c], c) and df[c].nunique() > 5][:3]
    for c in nums:
        figs[f"Distribution: {c}"] = fig_distribution(df, c)

    if profile is not None and (profile["missing"]["missing"] > 0).any():
        figs["Missing values"] = fig_missing(profile["missing"])
    return figs
