# 📊 AI-Powered Data Analytics Workflow (Streamlit)

Upload an Excel/CSV file and walk it through a full analytics pipeline — profiling, cleaning,
statistics, interactive charts, LLM-written business insights — then download a single shareable HTML report.

```
USER → Upload Excel/CSV → Streamlit Frontend → Python Backend → Pandas → Data Profiling → Clean Dataset
     → Analytics Engine → Visualization → LLM Insight Engine → Business Insights → Final Interactive Report
```

## Folder structure

```
ai-analytics-workflow/
├── app.py                    # Streamlit UI + orchestration (session state, buttons, tabs)
├── backend/
│   ├── data_loader.py        # CSV/Excel → DataFrame, type inference (numbers-as-text, dates, DD/MM vs MM/DD)
│   ├── profiling.py          # types, missing values, duplicates, IQR outliers, quality score
│   ├── cleaning.py           # configurable auto-clean; every action is logged
│   ├── analytics.py          # describe(), correlations (SciPy), trends (linregress), segment tests (Kruskal-Wallis)
│   ├── visualization.py      # Plotly figures + auto-built dashboard
│   ├── llm_insight.py        # Anthropic / OpenAI insights + rule-based offline fallback
│   ├── report.py             # Jinja2 → self-contained HTML report
│   └── utils.py              # column-type helpers shared by all modules
├── sample_data/
│   ├── generate_sample.py    # builds a deliberately messy sales dataset
│   └── sales_sample.csv
├── .streamlit/config.toml    # theme + upload limit
├── .streamlit/secrets.toml.example
├── .env.example
├── render.yaml               # Render blueprint
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add ANTHROPIC_API_KEY or OPENAI_API_KEY (optional)
streamlit run app.py
```

Click **Use sample dataset** in the sidebar to try it instantly, or **▶ Run full pipeline** to run every stage at once.
No API key? The app still works — the Insights tab falls back to a deterministic rule-based summary.

## How the components connect

| Stage | Module → function | Input → Output |
|---|---|---|
| Upload | `data_loader.load_data` | file bytes → typed `DataFrame` (`raw_df`) |
| Profile | `profiling.profile_dataframe` | `raw_df` → dict of DataFrames + quality score |
| Clean | `cleaning.clean_dataframe` | `raw_df` + `CleaningOptions` → `(clean_df, log)` |
| Analyze | `analytics.run_analytics` | `clean_df` → stats, correlations, trend, segment tables |
| Visualize | `visualization.build_dashboard` | `clean_df` + analytics → `{name: plotly Figure}` |
| Insights | `llm_insight.generate_insights` | profile + analytics **(aggregates only)** → Markdown |
| Report | `report.build_html_report` | everything above → one HTML string (download) |

`app.py` holds each stage's output in `st.session_state`. Streamlit reruns the script on every click, so
session state is what lets you use tabs in any order; re-running a stage automatically invalidates the stages after it.

### Design decisions worth knowing

- **LLM never sees raw rows.** `build_context()` sends only aggregates (summary stats, top correlations, trend, top segments).
- **Grounded prompts.** The system prompt forbids inventing numbers and requires "associated with" wording for
  correlations — so you get "Revenue rose 49% and laptops drive 37% of sales", not a made-up "due to product X".
- **Partial periods are dropped.** A half-finished last month would make a monthly *sum* look like a crash, so incomplete edge periods are excluded from the trend (and reported).
- **Outliers are kept by default.** They are often real events (a big order). Choose *cap* or *remove* in the Cleaning tab.
- **Date ambiguity.** `05/03/2024` is auto-detected as day-first only if some value proves it (e.g. `25/03/2024`); override in the sidebar.
- **Report is XSS-safe.** Raw HTML in LLM output or user edits is escaped before it reaches the report.

## Deploy

### Option A — Streamlit Community Cloud (easiest)
1. Push this folder to a GitHub repo (`.env` and `secrets.toml` are git-ignored).
2. Go to <https://share.streamlit.io> → **New app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings** → choose Python 3.11/3.12 and paste your keys under **Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."
   ```
4. **Deploy**. Every `git push` redeploys.

### Option B — Render
1. Push to GitHub, then Render dashboard → **New → Blueprint** (uses `render.yaml`) — or **New → Web Service** manually with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
2. Add `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` under **Environment**. Set `PYTHON_VERSION` (e.g. `3.12.3`).
3. Deploy. The free plan sleeps after inactivity (first load is slow) and has ~512 MB RAM — for very large files, sample the data first.

## Customising

- **Different LLM / model**: edit `DEFAULT_MODELS` in `llm_insight.py` (or type any model name in the sidebar).
- **New chart**: add a `fig_*` function in `visualization.py` and register it in `build_dashboard`.
- **New analysis** (forecasting, seasonality, A/B tests): add a function in `analytics.py` (e.g. `statsmodels` seasonal decomposition), return it from `run_analytics`, and mention it in `llm_insight.build_context`.
- **PDF export**: open the HTML report in a browser → *Print → Save as PDF* (the print stylesheet is included), or add `weasyprint`/`playwright` to `report.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not read the file" | Check the extension; `.xls` needs `xlrd` (in requirements). |
| Dates parsed wrongly | Use the *Ambiguous dates* radio in the sidebar. |
| LLM call failed banner | Check the key/model name; the rule-based summary is shown instead. |
| Report charts blank offline | Tick **Embed Plotly.js** before building the report. |
