"""Streamlit front-end: wires the whole pipeline together.

USER -> upload -> Pandas load -> profile -> clean -> analytics -> Plotly visuals -> LLM insights -> HTML report
Every stage stores its result in st.session_state, so tabs/buttons can be used in any order
and results survive Streamlit's rerun-on-every-click model.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from backend import analytics, cleaning, data_loader, llm_insight, profiling, report, visualization
from backend.utils import categorical_cols, datetime_cols, numeric_cols

try:  # optional: load API keys from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="AI Data Analytics Workflow", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sales_sample.csv"
DEFAULTS = {
    "source": None, "filename": None, "load_sig": None, "upload_sig": None,
    "raw_df": None, "profile": None, "clean_df": None, "profile_clean": None, "clean_log": [],
    "analytics": None, "figs": None, "insights": None, "insight_source": None, "insight_version": 0,
    "report_html": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = list(val) if isinstance(val, list) else val
S = st.session_state

PIPELINE = ["profile", "clean_df", "analytics", "figs", "insights", "report_html"]


def clear_from(stage: str) -> None:
    """Invalidate a stage and everything downstream of it."""
    for k in PIPELINE[PIPELINE.index(stage):]:
        S[k] = None
    if stage in ("profile", "clean_df"):
        S.clean_log, S.profile_clean = [], None


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.getenv(name, "")
    except Exception:  # no secrets.toml present
        return os.getenv(name, "")


def working_df() -> pd.DataFrame:
    return S.clean_df if S.clean_df is not None else S.raw_df


# --------------------------------------------------------------------------- pipeline stages
def do_profile() -> None:
    S.profile = profiling.profile_dataframe(S.raw_df)


def do_clean(opts: cleaning.CleaningOptions) -> None:
    if S.profile is None:
        do_profile()
    clear_from("clean_df")
    S.clean_df, S.clean_log = cleaning.clean_dataframe(S.raw_df, opts)
    S.profile_clean = profiling.profile_dataframe(S.clean_df)


def do_analytics(**kwargs) -> None:
    if S.profile is None:
        do_profile()
    clear_from("analytics")
    S.analytics = analytics.run_analytics(working_df(), **kwargs)


def do_visuals() -> None:
    if S.analytics is None:
        do_analytics()
    clear_from("figs")
    S.figs = visualization.build_dashboard(working_df(), S.analytics, S.profile)


def do_insights(provider: str, api_key: str, model: str, business_context: str) -> None:
    if S.analytics is None:
        do_analytics()
    S.insights, S.insight_source = llm_insight.generate_insights(
        S.profile, S.analytics, S.clean_log, provider=provider, api_key=api_key, model=model,
        business_context=business_context)
    S.insight_version += 1
    S.report_html = None


def do_report(title: str, embed_js: bool) -> None:
    if S.figs is None:
        do_visuals()
    if S.insights is None:
        S.insights, S.insight_source = llm_insight.rule_based_insights(S.profile, S.analytics), "rule-based"
    S.report_html = report.build_html_report(
        filename=S.filename, profile=S.profile, analytics=S.analytics, figures=S.figs,
        insights_md=S.insights, insight_source=S.insight_source, cleaning_log=S.clean_log,
        clean_shape=None if S.clean_df is None else S.clean_df.shape, title=title, embed_plotlyjs=embed_js)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 AI Analytics")
    st.caption("Upload → Profile → Clean → Analyze → Visualize → Insights → Report")

    uploaded = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx", "xls"])
    if uploaded is not None and S.upload_sig != (uploaded.name, uploaded.size):
        S.upload_sig, S.source, S.filename = (uploaded.name, uploaded.size), uploaded.getvalue(), uploaded.name
    if SAMPLE_PATH.exists() and st.button("Use sample dataset", width="stretch"):
        S.upload_sig, S.source, S.filename = None, SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name

    sheet, dayfirst = None, None
    if S.source is not None:
        sheets = data_loader.list_sheets(S.source, S.filename)
        if len(sheets) > 1:
            sheet = st.selectbox("Excel sheet", sheets)
        order = st.radio("Ambiguous dates (05/03/2024)", ["Auto-detect", "Day first (DD/MM)", "Month first (MM/DD)"])
        dayfirst = {"Auto-detect": None, "Day first (DD/MM)": True, "Month first (MM/DD)": False}[order]

        sig = (S.filename, len(S.source), sheet, dayfirst)
        if S.load_sig != sig:
            try:
                S.raw_df = data_loader.load_data(S.source, S.filename, sheet, dayfirst)
                S.load_sig = sig
                clear_from("profile")
            except Exception as exc:
                S.raw_df, S.load_sig = None, None
                st.error(f"Could not read the file: {exc}")

    st.divider()
    st.subheader("LLM insight engine")
    provider = st.selectbox("Provider", ["Anthropic", "OpenAI", "None (rule-based)"])
    default_model = llm_insight.DEFAULT_MODELS.get(provider, "")
    model = st.text_input("Model", value=default_model, disabled=not default_model)
    env_key = secret("ANTHROPIC_API_KEY" if provider == "Anthropic" else "OPENAI_API_KEY")
    api_key = st.text_input("API key", type="password", value=env_key, disabled=not default_model,
                            help="Or set ANTHROPIC_API_KEY / OPENAI_API_KEY in .env or Streamlit secrets.")
    business_context = st.text_area("Business context (optional)", placeholder="e.g. Monthly electronics sales for an Indian retailer")
    st.caption("Only aggregated statistics are sent to the LLM — never raw rows.")

    st.divider()
    if st.button("▶ Run full pipeline", type="primary", width="stretch", disabled=S.raw_df is None):
        with st.spinner("Running pipeline…"):
            do_profile()
            do_clean(cleaning.CleaningOptions())
            do_analytics()
            do_visuals()
            do_insights(provider, api_key, model, business_context)
            do_report("Data Analytics Report", False)
        st.success("Pipeline complete — see the tabs.")

# --------------------------------------------------------------------------- header + progress
st.title("AI-Powered Data Analytics Workflow")
if S.raw_df is None:
    st.info("👈 Upload a CSV/Excel file (or click **Use sample dataset**) to begin.")
    st.stop()

stages = [("Loaded", True), ("Profiled", S.profile is not None), ("Cleaned", S.clean_df is not None),
          ("Analyzed", S.analytics is not None), ("Visualized", S.figs is not None),
          ("Insights", S.insights is not None), ("Report", S.report_html is not None)]
for col, (name, done) in zip(st.columns(len(stages)), stages):
    col.markdown(f"{'✅' if done else '⬜'} **{name}**")

tabs = st.tabs(["1 · Data", "2 · Profiling", "3 · Cleaning", "4 · Analytics", "5 · Visualization", "6 · AI Insights", "7 · Report"])

# --------------------------------------------------------------------------- 1 data
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(S.raw_df):,}")
    c2.metric("Columns", S.raw_df.shape[1])
    c3.metric("File", S.filename)
    st.dataframe(S.raw_df.head(500))

# --------------------------------------------------------------------------- 2 profiling
with tabs[1]:
    if st.button("🔍 Run data profiling", type="primary"):
        do_profile()
    p = S.profile
    if p is None:
        st.caption("Detects data types, missing values, duplicates and outliers.")
    else:
        m = st.columns(5)
        m[0].metric("Quality score", f"{p['quality_score']}/100")
        m[1].metric("Missing cells", f"{p['missing_cells_pct']}%")
        m[2].metric("Duplicate rows", p["duplicates"])
        m[3].metric("Constant columns", len(p["constant_columns"]))
        m[4].metric("Memory", f"{p['memory_mb']} MB")
        st.subheader("Column types")
        st.dataframe(p["types"], hide_index=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Missing values")
            st.dataframe(p["missing"], hide_index=True)
        with right:
            st.subheader("Outliers (IQR rule)")
            st.dataframe(p["outliers"], hide_index=True)

# --------------------------------------------------------------------------- 3 cleaning
with tabs[2]:
    with st.expander("Cleaning options", expanded=True):
        a, b, c = st.columns(3)
        drop_dups = a.checkbox("Remove duplicate rows", True)
        drop_empty = a.checkbox("Remove empty rows/columns", True)
        col_thr = a.slider("Drop columns with more than … missing (fraction)", 0.1, 1.0, 0.6, 0.05)
        num_imp = b.selectbox("Missing numbers", ["median", "mean", "drop"])
        cat_imp = b.selectbox("Missing categories", ["mode", "unknown", "drop"])
        out_mode = c.selectbox("Outliers", ["none", "cap", "remove"], help="Default keeps outliers: they are often real business events.")
        iqr_k = c.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.5)
    if st.button("🧹 Clean dataset", type="primary"):
        do_clean(cleaning.CleaningOptions(drop_dups, drop_empty, col_thr, num_imp, cat_imp, out_mode, iqr_k))
    if S.clean_df is not None:
        pc = S.profile_clean
        m = st.columns(4)
        m[0].metric("Rows", f"{len(S.clean_df):,}", f"{len(S.clean_df) - len(S.raw_df):,}")
        m[1].metric("Columns", S.clean_df.shape[1], S.clean_df.shape[1] - S.raw_df.shape[1])
        m[2].metric("Missing cells", f"{pc['missing_cells_pct']}%")
        m[3].metric("Quality score", f"{pc['quality_score']}/100")
        st.subheader("What changed")
        for line in S.clean_log:
            st.markdown(f"- {line}")
        st.dataframe(S.clean_df.head(500))
        st.download_button("⬇ Download cleaned CSV", S.clean_df.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv")
    else:
        st.caption("Analytics will use the raw data until you clean it.")

# --------------------------------------------------------------------------- 4 analytics
with tabs[3]:
    df = working_df()
    nums, dates = numeric_cols(df), datetime_cols(df)
    a, b, c, d = st.columns(4)
    val_choice = a.selectbox("Main metric", ["auto"] + nums)
    date_choice = b.selectbox("Date column", ["auto"] + dates)
    freq_choice = c.selectbox("Trend period", ["auto", "D", "W", "M", "Q", "Y"])
    method = d.selectbox("Correlation", ["pearson", "spearman"])
    if st.button("📈 Run analytics", type="primary"):
        do_analytics(value_col=None if val_choice == "auto" else val_choice,
                     date_col=None if date_choice == "auto" else date_choice,
                     freq=freq_choice, corr_method=method)
    an = S.analytics
    if an:
        t = an["trend"]
        if t:
            st.subheader(f"Trend — {t['agg']} of {t['value_col']} per {t['freq_label']}")
            m = st.columns(4)
            m[0].metric("Direction", t["direction"].split(" (")[0].capitalize())
            m[1].metric("First → last", f"{t['total_change_pct']:+.1f}%" if t["total_change_pct"] is not None else "n/a")
            m[2].metric("Latest period", f"{t['latest_change_pct']:+.1f}%" if t["latest_change_pct"] is not None else "n/a")
            m[3].metric("R² / p-value", f"{t['r_squared']:.2f} / {t['p_value']:.3f}")
            if t["dropped_partial_periods"]:
                st.caption(f"Incomplete period(s) excluded: {', '.join(t['dropped_partial_periods'])}")
        else:
            st.info("No trend computed (needs a date column and a numeric metric).")
        st.subheader("Descriptive statistics")
        st.dataframe(an["descriptive"]["numeric"], hide_index=True)
        if not an["descriptive"]["categorical"].empty:
            st.dataframe(an["descriptive"]["categorical"], hide_index=True)
        st.subheader("Strongest correlations")
        st.dataframe(an["correlation"]["pairs"], hide_index=True)
        for bd in an["breakdowns"]:
            sig = "" if bd["kruskal_p"] is None else (" — groups differ significantly" if bd["kruskal_p"] < 0.05 else " — no significant difference")
            st.subheader(f"{bd['value_col']} by {bd['category_col']}{sig}")
            st.dataframe(bd["table"], hide_index=True)

# --------------------------------------------------------------------------- 5 visualization
with tabs[4]:
    if st.button("📊 Generate dashboard", type="primary"):
        do_visuals()
    if S.figs:
        items = list(S.figs.items())
        for i in range(0, len(items), 2):
            for col, (name, fig) in zip(st.columns(2), items[i:i + 2]):
                col.plotly_chart(fig, key=f"dash_{name}")
    with st.expander("🎨 Build your own chart"):
        df = working_df()
        kind = st.selectbox("Chart type", ["scatter", "line", "bar", "histogram", "box"])
        cols = list(df.columns)
        x = st.selectbox("X (or the variable for histogram/box)", cols)
        y = st.selectbox("Y", cols, index=min(1, len(cols) - 1))
        color = st.selectbox("Colour by", [None] + categorical_cols(df, 15))
        if st.button("Draw chart"):
            st.plotly_chart(visualization.custom_chart(df, kind, x, y, color), key="custom_chart")

# --------------------------------------------------------------------------- 6 insights
with tabs[5]:
    if st.button("🤖 Generate business insights", type="primary"):
        with st.spinner("Asking the LLM…" if api_key and provider != "None (rule-based)" else "Summarising…"):
            do_insights(provider, api_key, model, business_context)
    if S.insights:
        st.caption(f"Source: {S.insight_source}")
        st.markdown(S.insights)
        with st.expander("✏️ Edit before exporting"):
            S.insights = st.text_area("Markdown", S.insights, height=350, key=f"edit_{S.insight_version}")
    else:
        st.caption("Without an API key you'll get a rule-based summary; with one, an LLM-written narrative.")

# --------------------------------------------------------------------------- 7 report
with tabs[6]:
    title = st.text_input("Report title", "Data Analytics Report")
    embed = st.checkbox("Embed Plotly.js (works offline, +4 MB)", False)
    if st.button("📄 Build report", type="primary"):
        with st.spinner("Rendering report…"):
            do_report(title, embed)
    if S.report_html:
        st.download_button("⬇ Download HTML report", S.report_html.encode("utf-8"), "analytics_report.html",
                           "text/html", type="primary")
        st.caption("Tip: open the file in a browser and use Print → Save as PDF for a PDF copy.")
        if hasattr(st, "iframe"):                       # Streamlit >= 1.5x
            st.iframe(S.report_html, height=900)
        else:                                           # older Streamlit
            components.html(S.report_html, height=900, scrolling=True)
