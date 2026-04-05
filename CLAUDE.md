# CLAUDE.md — Your Data Wrestler

This file provides guidance for AI assistants (Claude, Copilot, etc.) working on this codebase.

## Project Overview

**Your Data Wrestler** is a Streamlit-based web application for AI-powered data analysis, cleaning, and visualization. Users upload CSV/Excel files and interact with their data through natural language questions and AI-generated suggestions powered by OpenAI GPT-3.5-turbo.

**Tech stack**: Python 3.11.9, Streamlit 1.37.1, Pandas, Plotly, OpenAI API

## Repository Structure

```
Your-Data-Wrestler/
├── main.py          # Streamlit app entry point & UI (747 lines)
├── utils.py         # Core logic, AI integration, data processing (784 lines)
├── prompts.py       # LLM prompt templates (164 lines)
├── requirements.txt # Pinned Python dependencies
├── packages.txt     # System packages for Streamlit Cloud (python3-dev, build-essential)
├── runtime.txt      # Python version pin (3.11.9)
├── .streamlit/
│   └── config.toml  # Server settings (headless, port 8501, no CORS/CSRF)
├── README.md        # User-facing documentation
└── .gitignore       # Excludes venv, .env, __pycache__, etc.
```

## Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
cp .env.example .env  # or create .env manually
# Add: OPENAI_API_KEY=sk-...

# Run the app
streamlit run main.py
```

The app runs at `http://localhost:8501` by default.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for AI features) | OpenAI API key for GPT-3.5-turbo calls |

The key is loaded via `python-dotenv` from `.env` (local) or Streamlit secrets (cloud deployment). The app degrades gracefully when the key is absent — AI features return empty results rather than crashing.

## Architecture & Data Flow

```
User uploads file (CSV/TXT/XLSX/XLS)
    → main.py reads file into pandas DataFrame
    → Stored in st.session_state["df"]
    → infer_domain() auto-detects data domain
    → User interactions trigger utils.py functions
    → utils.py formats prompts from prompts.py
    → OpenAI GPT-3.5-turbo returns structured JSON
    → utils.py parses responses with fallbacks
    → main.py renders results in Streamlit UI
```

### Module Responsibilities

**`main.py`** — UI only. Handles:
- Streamlit page config (Matrix-style green theming)
- Session state initialization and lifecycle
- File upload and format detection
- Section toggles via sidebar checkboxes (overview, cleaning, visualization)
- Rendering suggestions returned from utils.py

**`utils.py`** — All logic. Handles:
- OpenAI client setup and caching (`lru_cache` + MD5 dict cache)
- Dataset introspection (`get_dataset_info`)
- Natural language Q&A (`analyze_data`)
- Cleaning suggestions and application (`get_cleaning_suggestions`, `apply_quick_fixes`)
- Visualization suggestions and creation (`get_visualization_suggestions`, `create_visualization`)
- Robust JSON parsing from LLM responses (`_extract_json_block`, `_parse_json_safe`)

**`prompts.py`** — Prompt templates only. Three templates:
- `ANALYSIS_PROMPT` — natural language Q&A
- `CLEANING_SUGGESTIONS_PROMPT` — structured cleaning suggestions (or `"no_clean_needed"`)
- `VISUALIZATION_SUGGESTIONS_PROMPT` — 3-5 visualization suggestions from 11 supported types

## Key Conventions

### Naming
- Private helpers prefixed with underscore: `_get_openai_client`, `_parse_json_safe`
- Session state keys: `snake_case` (e.g., `cleaning_suggestions`, `generated_figs`)
- All functions: `snake_case`

### LLM Integration Pattern
1. Build prompt string with dataset info + domain context
2. Call `get_openai_response(prompt, cache_key)` — checks cache first
3. Parse response with `_parse_json_safe` (tries direct parse → JSON block extraction → fallback)
4. Never crash on bad LLM output — always return empty/default structure

### Caching
- `@lru_cache(maxsize=100)` on `_get_openai_client`
- Dict-based response cache keyed by MD5 hash of the prompt
- Always check cache before making API calls

### Session State
- All state initialized at app start in `main.py`
- Detect file changes by comparing `df` shape/columns and clear stale suggestions
- Track `generated_figs` (AI) and custom figures separately

### Error Handling
- Use `st.error()`, `st.warning()`, `st.spinner()` for user-facing errors
- Use try/except broadly around data type conversions
- Never let missing columns or unexpected dtypes crash the app

### Supported Cleaning Actions (10)
`remove_duplicates`, `fill_missing_mean`, `fill_missing_mode`, `fill_missing_zero`, `fill_missing_empty`, `convert_to_numeric`, `convert_to_datetime`, `remove_outliers`, `strip_whitespace`, `standardize_case`

### Supported Visualization Types (11)
`histogram`, `scatter`, `bar`, `box`, `line`, `stacked_bar`, `area`, `pie`, `heatmap`, `violin`, `density`

## Deployment

The app is deployed on **Streamlit Cloud**. Deployment is triggered by pushing to the `main` branch. There is no CI/CD pipeline — deployments are manual git pushes.

- `packages.txt` provides system-level packages for Streamlit Cloud build
- `runtime.txt` pins the Python version
- All dependencies in `requirements.txt` are pinned to exact versions for reproducibility — keep them pinned when adding new ones

## Testing

There is currently **no test suite**. Manual testing is done by running `streamlit run main.py` locally and exercising the UI. When adding tests, use `pytest` and place test files in a `tests/` directory.

## Important Notes for AI Assistants

1. **Dependency versions are intentionally pinned** — do not upgrade versions without verifying Streamlit Cloud compatibility. Past issues with Plotly (kaleido) required specific version pinning.

2. **Do not add persistent storage** — the app is stateless by design; session state is lost on browser refresh. This is intentional for privacy (no user data stored).

3. **OpenAI model is GPT-3.5-turbo** — kept for cost reasons. Do not change to GPT-4 without user instruction.

4. **JSON parsing is deliberately defensive** — `_extract_json_block` and `_parse_json_safe` exist because LLMs sometimes wrap JSON in markdown or add commentary. Do not simplify this logic.

5. **No authentication** — the app has no user auth. Do not add `OPENAI_API_KEY` to any committed file.

6. **Streamlit re-runs on every interaction** — all code at module level runs on each user action. Use `st.session_state` to persist data across re-runs.

7. **The `main` branch is production** — always develop on feature branches and merge via PR.
