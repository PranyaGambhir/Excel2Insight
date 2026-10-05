"""Stage 6 - LLM insight engine.

Only *aggregated* results (profile summary, statistics, trends, segment tables) are sent to the
model - never raw rows. If no API key is configured, or the call fails, a deterministic
rule-based summary is returned so the app always produces something useful.
"""
from __future__ import annotations

import json

import pandas as pd

DEFAULT_MODELS = {"Anthropic": "claude-sonnet-5", "OpenAI": "gpt-4o-mini"}

SYSTEM_PROMPT = """You are a senior business data analyst writing for non-technical decision makers.
You receive JSON with aggregated statistics computed from a dataset. Write insights that are:
- GROUNDED: use only numbers that appear in the JSON. Never invent figures, products, regions or causes.
- HONEST ABOUT CAUSALITY: correlations and segment differences are "associated with", not "caused by",
  unless the JSON itself proves the cause. Mention p-values only in plain language (e.g. "statistically significant").
- CAVEATED: call out data-quality issues (missing data, duplicates, outliers). If trend.dropped_partial_periods is non-empty, say those incomplete periods were excluded.
- ACTIONABLE: finish with concrete next steps.

Format the answer in Markdown with exactly these sections:
## Executive summary   (2-3 sentences)
## Key findings        (4-6 bullets, each containing at least one number)
## Drivers and segments (what is associated with the main metric)
## Risks and data caveats
## Recommended actions  (3 bullets)
"""


# --------------------------------------------------------------------------- context
def _records(df: pd.DataFrame, n: int = 12) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.head(n).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def build_context(profile: dict, analytics: dict, cleaning_log: list[str] | None = None,
                  business_context: str = "") -> str:
    trend = analytics.get("trend")
    ctx = {
        "business_context": business_context or "not provided",
        "dataset": {k: profile[k] for k in ("n_rows", "n_cols", "missing_cells_pct", "duplicates", "quality_score")},
        "data_quality": {
            "high_missing_columns": profile["high_missing_columns"],
            "outlier_columns": _records(profile["outliers"], 6),
        },
        "cleaning_actions": (cleaning_log or [])[:10],
        "main_metric": analytics.get("value_col"),
        "numeric_summary": _records(analytics["descriptive"]["numeric"][["column", "mean", "std", "min", "50%", "max", "skew"]]
                                    if not analytics["descriptive"]["numeric"].empty else pd.DataFrame(), 10),
        "top_correlations": _records(analytics["correlation"]["pairs"], 8),
        "trend": None if not trend else {
            **{k: trend[k] for k in ("date_col", "value_col", "freq_label", "agg", "n_periods", "first_value", "last_value",
                                     "dropped_partial_periods", "total_change_pct", "latest_change_pct", "direction", "r_squared", "p_value",
                                     "peak_period", "peak_value", "trough_period", "trough_value")},
            "recent_periods": _records(trend["series"][["period", "value", "pct_change"]].tail(8), 8),
        },
        "segments": [
            {"column": b["category_col"], "metric": b["value_col"], "agg": b["agg"],
             "groups_differ_significantly_p": b["kruskal_p"], "top_groups": _records(b["table"], 6)}
            for b in analytics.get("breakdowns", [])
        ],
    }
    return json.dumps(ctx, default=str, indent=1)[:14000]


# --------------------------------------------------------------------------- providers
def _call_anthropic(context: str, api_key: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=1800, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Aggregated analytics JSON:\n{context}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_openai(context: str, api_key: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Aggregated analytics JSON:\n{context}"}],
    )
    return resp.choices[0].message.content


def generate_insights(profile: dict, analytics: dict, cleaning_log: list[str] | None = None,
                      provider: str = "Anthropic", api_key: str = "", model: str = "",
                      business_context: str = "") -> tuple[str, str]:
    """Return (markdown_text, source) where source is 'anthropic', 'openai' or 'rule-based'."""
    if provider in DEFAULT_MODELS and api_key:
        context = build_context(profile, analytics, cleaning_log, business_context)
        model = model or DEFAULT_MODELS[provider]
        try:
            text = (_call_anthropic if provider == "Anthropic" else _call_openai)(context, api_key, model)
            if text and text.strip():
                return text.strip(), f"{provider.lower()} ({model})"
        except Exception as exc:  # network, auth, quota, model name ...
            fallback = rule_based_insights(profile, analytics)
            return f"> ⚠️ LLM call failed ({type(exc).__name__}: {exc}). Showing rule-based insights instead.\n\n{fallback}", "rule-based"
    return rule_based_insights(profile, analytics), "rule-based"


# --------------------------------------------------------------------------- offline fallback
def _fmt(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def rule_based_insights(profile: dict, analytics: dict) -> str:
    lines = ["## Executive summary"]
    trend = analytics.get("trend")
    metric = analytics.get("value_col")
    if trend and trend["total_change_pct"] is not None:
        lines.append(f"{metric} shows **{trend['direction']}** across {trend['n_periods']} {trend['freq_label']}s "
                     f"({trend['total_change_pct']:+.1f}% from first to last period).")
    else:
        lines.append(f"The dataset has {profile['n_rows']:,} rows and {profile['n_cols']} columns "
                     f"with a data-quality score of {profile['quality_score']}/100.")

    lines += ["", "## Key findings"]
    if trend:
        lines.append(f"- Peak {trend['freq_label']}: **{trend['peak_period']}** ({_fmt(trend['peak_value'])}); "
                     f"lowest: **{trend['trough_period']}** ({_fmt(trend['trough_value'])}).")
        if trend["latest_change_pct"] is not None:
            lines.append(f"- The latest {trend['freq_label']} changed by {trend['latest_change_pct']:+.1f}% versus the previous one.")
        lines.append(f"- Linear trend explains {trend['r_squared']:.0%} of the variation (p = {trend['p_value']:.3f}).")
    pairs = analytics["correlation"]["pairs"]
    for _, r in pairs.head(3).iterrows():
        lines.append(f"- {r['variable_1']} and {r['variable_2']} have a {r['strength']} "
                     f"{'positive' if r['r'] > 0 else 'negative'} relationship (r = {r['r']:.2f}).")

    lines += ["", "## Drivers and segments"]
    if analytics.get("breakdowns"):
        for b in analytics["breakdowns"]:
            top = b["table"].iloc[0]
            share = f" ({top['share_pct']}% of total)" if "share_pct" in b["table"] else ""
            sig = ""
            if b["kruskal_p"] is not None:
                sig = " Differences between groups are statistically significant." if b["kruskal_p"] < 0.05 else " Differences between groups are not statistically significant."
            lines.append(f"- By **{b['category_col']}**, *{top['category']}* leads on {metric}{share}.{sig}")
    else:
        lines.append("- No categorical segments were found to compare.")

    lines += ["", "## Risks and data caveats"]
    lines.append(f"- {profile['missing_cells_pct']}% of cells were missing and {profile['duplicates']} duplicate rows were found in the raw data.")
    if not profile["outliers"].empty and profile["outliers"]["outliers"].sum():
        worst = profile["outliers"].iloc[0]
        lines.append(f"- Outliers detected, most in **{worst['column']}** ({worst['outliers']} rows) - verify whether they are real events or errors.")
    if trend and trend.get("dropped_partial_periods"):
        lines.append(f"- Incomplete period(s) {', '.join(trend['dropped_partial_periods'])} were excluded from the trend so totals are not understated.")
    else:
        lines.append("- The most recent period may be incomplete; interpret the latest change with care.")

    lines += ["", "## Recommended actions",
              "- Validate the top segments and any outliers with the business owner.",
              "- Track the main metric on a regular cadence to confirm the trend persists.",
              "- Add an API key in the sidebar to get narrative, LLM-written insights."]
    return "\n".join(lines)
