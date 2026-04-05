# CLAUDE.md — Your Data Wrestler

This file documents the codebase for AI assistants working on this project.

## Project Overview

**Your Data Wrestler** is a Streamlit web application that transforms raw data into actionable insights through AI-powered analysis, automated cleaning, and dynamic visualizations. Users upload CSV/Excel/TXT files and interact with their data via natural language queries, automated cleaning suggestions, and one-click chart generation.

- **Live app:** https://your-data-wrestler.streamlit.app/
- **Author:** Bernardo Forbes Costa PhD
- **License:** MIT 2025

---

## Repository Structure

```
Your-Data-Wrestler/
├── main.py           # Streamlit UI, session state, page layout (747 lines)
├── utils.py          # All business logic, OpenAI API calls, data helpers (784 lines)
├── prompts.py        # LLM prompt templates (164 lines)
├── requirements.txt  # Python dependencies
├── runtime.txt       # Python version pin (3.11.9)
├── packages.txt      # System-level apt packages (python3-dev, build-essential)
├── .streamlit/
│   └── config.toml   # Streamlit server config (port 8501, headless, CORS off)
├── README.md
└── LICENSE
```

There is no `tests/` directory, no Makefile, no `pyproject.toml`, and no CI/CD workflow files.

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| App framework | Streamlit | 1.37.1 |
| Data processing | Pandas | 2.2.1 |
| Numerical | NumPy | 1.26.4 |
| Visualization | Plotly | 5.24.1 |
| Static chart export | Kaleido | 0.2.1 |
| AI / LLM | OpenAI SDK | 1.83.0 |
| Excel support | Openpyxl | 3.1.2 |
| Image processing | Pillow | 10.4.0 |
| Retry logic | Tenacity | >=8.0.0 |
| Env vars | python-dotenv | 1.0.1 |
| Runtime | Python | 3.11.9 |

---

## Running Locally

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Start the app
streamlit run main.py
# Opens at http://localhost:8501
```

The app runs without an API key — AI features are silently disabled, but file upload, data display, and manual cleaning still work.

---

## Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Optional | Powers all AI features. If absent, AI tabs are disabled. |

**Local:** place in `.env` (already in `.gitignore`).  
**Production:** configured in the Streamlit Cloud dashboard as a secret (`st.secrets`).

The code in `utils.py` checks both:
```python
# utils.py - _get_openai_client()
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv("OPENAI_API_KEY")
```

---

## Architecture and Code Conventions

### Module responsibilities

- **`main.py`** — owns all Streamlit UI rendering, tab layout, file upload handling, and session state reads/writes. Does NOT call OpenAI directly.
- **`utils.py`** — owns all data processing logic, every OpenAI API call, caching, and visualization construction. No Streamlit imports except `st.secrets` and `st.cache_data`.
- **`prompts.py`** — owns all raw LLM prompt strings. Update prompts here, not inline in `utils.py`.

### Session state

Persistent state lives in `st.session_state`. Key keys:

| Key | Type | Purpose |
|---|---|---|
| `df` | `pd.DataFrame` | The active dataset |
| `domain` | `str` | Inferred data domain (e.g., "Healthcare") |
| `cleaning_suggestions` | `list[dict]` | Pending cleaning actions |
| `generated_figs` | `list` | Plotly figures produced this session |
| `analysis_history` | `list[dict]` | Q&A pairs from the analysis tab |

A new file upload clears all session state.

### LLM usage (GPT-3.5-turbo)

All calls go through `utils.py`. The pattern:

1. Build a prompt from `prompts.py`
2. Hash the prompt with MD5 → check `_response_cache`
3. If cache miss, call `openai.chat.completions.create(model="gpt-3.5-turbo", temperature=0.7, max_tokens=500)`
4. Store result in `_response_cache` (LRU, max 100 entries)
5. Parse/validate the response before returning

Functions:
- `analyze_data(df, question, domain)` — natural language Q&A
- `get_cleaning_suggestions(df)` → list of 10 possible cleaning actions
- `get_visualization_suggestions(df, domain)` → JSON list of chart configs
- `infer_domain(df)` → short domain string
- `get_plot_labels(chart_type, x_col, y_col)` → axis/title labels

### Visualization

`create_visualization(df, chart_type, x_col, y_col, ...)` in `utils.py` supports 11 chart types:

`histogram`, `scatter`, `bar`, `box`, `line`, `stacked_bar`, `area`, `pie`, `heatmap`, `violin`, `density`

All charts use the `plotly_dark` template. Export is available as HTML (interactive) or PNG (via Kaleido).

### Data cleaning

Ten cleaning action types are hard-coded in `main.py`:

`remove_high_missing`, `fill_missing`, `remove_duplicates`, `lowercase_strings`, `strip_whitespace`, `convert_numeric`, `remove_outliers`, `standardize_dates`, `remove_empty_rows`, `fix_data_types`

Cleaning is applied in-place to `st.session_state.df`.

### Naming conventions

- Public helpers: `snake_case` (e.g., `get_dataset_info`, `analyze_data`)
- Private/internal helpers: leading underscore (e.g., `_get_openai_client`, `_response_cache`)
- No type annotations in the existing codebase — avoid adding them to unchanged code
- No docstrings in the existing codebase — avoid adding them to unchanged code

---

## Deployment

The app is deployed on **Streamlit Cloud** and auto-deploys on push to `main`.

```bash
# Deploy by pushing to main
git push origin main
```

There is no CI pipeline. There are no pre-commit hooks. Broken code pushed to `main` will break the live app immediately.

---

## No Tests

There are no automated tests. Manual testing through the Streamlit UI is the current approach. Do not add a test framework unless explicitly asked — the project has no test infrastructure to build on.

---

## Common Tasks

### Add a new chart type

1. Add the type string to the supported types list in `create_visualization()` in `utils.py`
2. Add a `elif chart_type == "your_type":` branch in `create_visualization()`
3. Update the prompt in `prompts.py` so the LLM knows it can suggest the new type

### Add a new cleaning action

1. Add the action key to the `cleaning_actions` dict in `main.py`
2. Add a matching `elif action == "your_action":` block in the cleaning execution logic in `main.py`
3. Update the cleaning suggestions prompt in `prompts.py`

### Change LLM behavior

Edit `prompts.py` only. Do not embed prompt strings inline in `utils.py` or `main.py`.

### Update a dependency

Edit `requirements.txt`. Use exact pinned versions (`==`) for packages that have caused compatibility issues: `numpy`, `plotly`, `pillow`, `kaleido`. Use `>=` only for utility packages like `tenacity`.

---

## Known Issues and Constraints

- **Kaleido** PNG export is fragile across environments — version is pinned to `0.2.1` specifically because newer versions broke on Streamlit Cloud.
- **NumPy** is pinned to `1.26.4` to avoid breaking changes in 2.x.
- **SSL errors** on the OpenAI client are caught silently; the client returns `None` and AI features degrade gracefully.
- The app is stateless between sessions — no database, no persistent storage.
- File size is limited by Streamlit's default upload limit (200 MB).
