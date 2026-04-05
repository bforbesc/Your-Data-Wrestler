# CLAUDE.md — Your Data Wrestler

This file provides guidance for AI assistants (Claude, Copilot, etc.) working on this codebase.

## Project Overview

**Your Data Wrestler** is a Streamlit-based web application for AI-powered data analysis, cleaning, and visualization. Users upload CSV/Excel files and interact with their data through natural language questions and AI-generated suggestions powered by **Strands Agents SDK** with OpenAI GPT-3.5-turbo.

**Tech stack**: Python 3.11.9, Streamlit 1.37.1, Pandas, Plotly, Strands Agents SDK, OpenAI

## Repository Structure

```
Your-Data-Wrestler/
├── main.py          # Streamlit app entry point & UI (747 lines)
├── utils.py         # Core logic, Strands AI integration, data processing
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
| `OPENAI_API_KEY` | Yes (for AI features) | OpenAI API key used by the Strands agent |

The key is loaded via `python-dotenv` from `.env` (local) or Streamlit secrets (cloud deployment). The app degrades gracefully when the key is absent — AI features return empty results rather than crashing.

## Architecture & Data Flow

```
User uploads file (CSV/TXT/XLSX/XLS)
    → main.py reads file into pandas DataFrame
    → Stored in st.session_state["df"]
    → infer_domain() auto-detects data domain via Strands agent
    → User interactions trigger utils.py functions
    → utils.py formats prompts from prompts.py
    → _call_agent() invokes a Strands Agent (OpenAI GPT-3.5-turbo)
    → Agent optionally calls data-inspection tools (list_columns, etc.)
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
- Strands `OpenAIModel` singleton (`_get_strands_model`)
- Five `@tool`-decorated data-inspection functions the agent can call
- `_call_agent(prompt, df, use_tools)` — core agent invocation with MD5 caching
- Dataset introspection (`get_dataset_info`)
- Natural language Q&A (`analyze_data`)
- Cleaning suggestions and application (`get_cleaning_suggestions`, `apply_quick_fixes`)
- Visualization suggestions and creation (`get_visualization_suggestions`, `create_visualization`)
- Robust JSON parsing from LLM responses (`_extract_json_block`, `_parse_json_safe`)

**`prompts.py`** — Prompt templates only. Three templates:
- `ANALYSIS_PROMPT` — natural language Q&A
- `CLEANING_SUGGESTIONS_PROMPT` — structured cleaning suggestions (or `"no_clean_needed"`)
- `VISUALIZATION_SUGGESTIONS_PROMPT` — 3-5 visualization suggestions from 11 supported types

## Strands Agent Integration

### How It Works

The Strands SDK replaces direct OpenAI API calls. Instead of a raw `client.chat.completions.create()`, a `strands.Agent` is created per invocation and handed a set of tools the agent can optionally call before producing its final answer.

```python
from strands import Agent, tool
from strands.models import OpenAIModel

model = OpenAIModel(
    client_args={"api_key": api_key},
    model_id="gpt-3.5-turbo",
    params={"temperature": 0.7, "max_tokens": 1000},
)
agent = Agent(model=model, tools=[list_columns, get_column_statistics, ...])
response: str = agent("Your prompt here")
```

### Available Tools (5)

All tools access the module-level `_current_df` variable which `_call_agent` sets before each invocation:

| Tool | When the agent uses it |
|------|----------------------|
| `list_columns()` | To verify column names and types |
| `get_column_statistics(column_name)` | To get min/max/mean/top values for a column |
| `get_missing_value_info()` | To check missing value counts across all columns |
| `get_sample_rows(n_rows)` | To examine actual data values |
| `check_duplicates()` | To check for duplicate rows |

### `_call_agent` signature

```python
def _call_agent(prompt: str, df: pd.DataFrame, use_tools: bool = True) -> str
```

- `use_tools=True` — for open-ended analysis (Q&A, domain inference, cleaning/viz suggestions)
- `use_tools=False` — for structured text-to-JSON conversion where tool calls would disrupt output format

### Caching

Responses are cached in `_response_cache` (module-level dict) keyed by MD5 hash of the prompt, identical to the previous implementation. `OpenAIModel` is cached as a module-level singleton via `_get_strands_model()`.

## Key Conventions

### Naming
- Private helpers prefixed with underscore: `_get_strands_model`, `_parse_json_safe`, `_call_agent`
- Session state keys: `snake_case` (e.g., `cleaning_suggestions`, `generated_figs`)
- All functions: `snake_case`

### LLM Integration Pattern
1. Build prompt string with dataset info + domain context (from `prompts.py` templates)
2. Call `_call_agent(prompt, df, use_tools=True/False)`
3. Parse response with `_parse_json_safe` where JSON is expected
4. Never crash on bad LLM output — always return empty/default structure

### Session State
- All state initialized at app start in `main.py`
- Detect file changes by comparing filename and clear stale suggestions
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
- All dependencies in `requirements.txt` are pinned to exact versions for reproducibility — keep them pinned when adding new ones (`strands-agents[openai]` should be pinned after verifying the version that works on Streamlit Cloud)

## Testing

There is currently **no test suite**. Manual testing is done by running `streamlit run main.py` locally and exercising the UI. When adding tests, use `pytest` and place test files in a `tests/` directory.

## Important Notes for AI Assistants

1. **Dependency versions are intentionally pinned** — do not upgrade versions without verifying Streamlit Cloud compatibility. Past issues with Plotly (kaleido) required specific version pinning. Pin `strands-agents` to an exact version once confirmed working.

2. **Do not add persistent storage** — the app is stateless by design; session state is lost on browser refresh. This is intentional for privacy (no user data stored).

3. **OpenAI model is GPT-3.5-turbo** — kept for cost reasons. Do not change to GPT-4 without user instruction. Model ID is set in `_get_strands_model()` in `utils.py`.

4. **JSON parsing is deliberately defensive** — `_extract_json_block` and `_parse_json_safe` exist because LLMs sometimes wrap JSON in markdown or add commentary. Do not simplify this logic.

5. **No authentication** — the app has no user auth. Do not add `OPENAI_API_KEY` to any committed file.

6. **Streamlit re-runs on every interaction** — all code at module level runs on each user action. Use `st.session_state` to persist data across re-runs. The `_current_df` module-level variable is safe here because Streamlit is single-threaded per session.

7. **The `main` branch is production** — always develop on feature branches and merge via PR.

8. **Strands agent is created fresh per call** — `_call_agent` creates a new `Agent` instance each time (conversation history is not needed across calls). The `OpenAIModel` instance is reused via `_get_strands_model()`.

9. **Tool use is controlled per call** — pass `use_tools=False` to `_call_agent` when the response must be a clean JSON blob (e.g., the second-pass JSON conversion in `get_visualization_suggestions`). Tool call output appearing in the response would break `_parse_json_safe`.
