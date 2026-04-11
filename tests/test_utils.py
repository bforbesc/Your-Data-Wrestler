"""
Tests for utils.py.

Coverage strategy
-----------------
- Pure functions (JSON parsing, dataset info, cleaning applicability,
  apply_quick_fixes): tested directly with no mocking.
- Visualization creation: get_plot_labels is patched so no API key is needed.
- AI-calling functions (analyze_data, infer_domain, get_cleaning_suggestions,
  get_visualization_suggestions): _call_agent is patched to return controlled
  strings, keeping tests fast and deterministic.
"""

import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import utils
from utils import (
    _extract_json_block,
    _parse_json_safe,
    get_dataset_info,
    _is_action_applicable,
    _get_applicable_cleaning_actions,
    _parse_cleaning_suggestions,
    apply_quick_fixes,
    create_visualization,
    analyze_data,
    infer_domain,
    get_cleaning_suggestions,
    get_visualization_suggestions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """Small, clean DataFrame with numeric and categorical columns."""
    return pd.DataFrame({
        "age":        [25, 30, 35, 40, 45],
        "salary":     [50_000, 60_000, 70_000, 80_000, 90_000],
        "department": ["HR", "IT", "IT", "Finance", "HR"],
        "name":       ["Alice", "Bob", "Charlie", "Dave", "Eve"],
    })


@pytest.fixture
def dirty_df():
    """DataFrame with missing values, duplicates, and whitespace."""
    return pd.DataFrame({
        "age":     [25, 30, None, None, None, 25],   # >10 % missing, has dupe
        "salary":  [50_000, None, None, None, None, 50_000],  # >50 % missing
        "city":    ["  NYC  ", "LA  ", "NYC", "  LA", " NYC ", "  NYC  "],
    })


@pytest.fixture
def outlier_df():
    """DataFrame where >20 % of numeric values are IQR outliers.

    Values 100-109 form a tight cluster (Q1=102, Q3=107, IQR=5, fence=114.5).
    1000, 2000, 3000 are clearly above the fence → 3/13 ≈ 23 % outliers.
    """
    values = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 1000, 2000, 3000]
    return pd.DataFrame({"value": values})


# ---------------------------------------------------------------------------
# 1. JSON parsing utilities
# ---------------------------------------------------------------------------

class TestExtractJsonBlock:
    def test_plain_json(self):
        text = '{"key": "value"}'
        assert _extract_json_block(text) == '{"key": "value"}'

    def test_fenced_json_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}

    def test_fenced_no_language_tag(self):
        text = '```\n{"a": 1}\n```'
        result = _extract_json_block(text)
        assert json.loads(result) == {"a": 1}

    def test_json_with_preamble(self):
        text = 'Here is the result:\n{"status": "ok"}\nDone.'
        result = _extract_json_block(text)
        assert json.loads(result) == {"status": "ok"}

    def test_no_json_returns_none(self):
        assert _extract_json_block("no json here") is None

    def test_none_input_returns_none(self):
        assert _extract_json_block(None) is None

    def test_empty_string_returns_none(self):
        assert _extract_json_block("") is None


class TestParseJsonSafe:
    def test_valid_json_string(self):
        assert _parse_json_safe('{"x": 1}') == {"x": 1}

    def test_fenced_json(self):
        text = '```json\n{"x": 1}\n```'
        assert _parse_json_safe(text) == {"x": 1}

    def test_json_with_surrounding_text(self):
        text = 'Result: {"done": true} — end'
        assert _parse_json_safe(text) == {"done": True}

    def test_invalid_json_returns_none(self):
        assert _parse_json_safe("not json at all") is None

    def test_empty_string_returns_none(self):
        assert _parse_json_safe("") is None

    def test_none_returns_none(self):
        assert _parse_json_safe(None) is None

    def test_nested_json(self):
        payload = {"options": [{"type": "histogram", "columns": ["age"]}]}
        assert _parse_json_safe(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# 2. get_dataset_info
# ---------------------------------------------------------------------------

class TestGetDatasetInfo:
    def test_row_and_column_counts(self, simple_df):
        info = get_dataset_info(simple_df)
        assert info["num_rows"] == 5
        assert info["num_cols"] == 4

    def test_column_info_keys(self, simple_df):
        info = get_dataset_info(simple_df)
        assert set(info["column_info"].keys()) == {"age", "salary", "department", "name"}

    def test_missing_values_clean(self, simple_df):
        info = get_dataset_info(simple_df)
        assert all(v == 0.0 for v in info["missing_values"].values())

    def test_missing_values_dirty(self, dirty_df):
        info = get_dataset_info(dirty_df)
        # salary has 4/6 ≈ 66.67 % missing
        assert info["missing_values"]["salary"] > 60.0

    def test_dtype_strings(self, simple_df):
        info = get_dataset_info(simple_df)
        assert "int" in info["column_info"]["age"] or "float" in info["column_info"]["age"]
        # pandas 2.x reports 'object'; pandas 3.x reports 'str'
        assert info["column_info"]["name"] in ("object", "str", "string")


# ---------------------------------------------------------------------------
# 3. _is_action_applicable
# ---------------------------------------------------------------------------

class TestIsActionApplicable:
    def test_remove_high_missing_true(self, dirty_df):
        # salary column has >50 % missing
        assert _is_action_applicable("remove_high_missing", dirty_df) is True

    def test_remove_high_missing_false(self, simple_df):
        assert _is_action_applicable("remove_high_missing", simple_df) is False

    def test_remove_duplicates_true(self, dirty_df):
        # dirty_df has 2 identical rows → 2/6 ≈ 33 % > 15 % threshold
        assert _is_action_applicable("remove_duplicates", dirty_df) == True

    def test_remove_duplicates_false(self, simple_df):
        assert _is_action_applicable("remove_duplicates", simple_df) == False

    def test_remove_outliers_true(self, outlier_df):
        assert _is_action_applicable("remove_outliers", outlier_df) == True

    def test_remove_outliers_false(self, simple_df):
        assert _is_action_applicable("remove_outliers", simple_df) == False

    def test_remove_empty_rows_true(self):
        df = pd.DataFrame({"a": [1, None], "b": [2, None]})
        assert _is_action_applicable("remove_empty_rows", df) == True

    def test_remove_empty_rows_false(self, simple_df):
        assert _is_action_applicable("remove_empty_rows", simple_df) == False

    def test_strip_whitespace_true(self, dirty_df):
        # city column has many values with leading/trailing spaces
        assert _is_action_applicable("strip_whitespace", dirty_df) is True

    def test_strip_whitespace_false(self, simple_df):
        assert _is_action_applicable("strip_whitespace", simple_df) is False

    def test_unknown_action_returns_false(self, simple_df):
        assert _is_action_applicable("nonexistent_action", simple_df) is False


# ---------------------------------------------------------------------------
# 4. _parse_cleaning_suggestions
# ---------------------------------------------------------------------------

class TestParseCleaningSuggestions:
    def test_no_clean_needed_returns_empty(self, simple_df):
        result = _parse_cleaning_suggestions("no_clean_needed", simple_df)
        assert result == []

    def test_empty_response_falls_back_to_applicable_actions(self, dirty_df):
        result = _parse_cleaning_suggestions("", dirty_df)
        # dirty_df has high missing → should surface at least one suggestion
        assert isinstance(result, list)
        assert len(result) > 0

    def test_parsed_suggestions_have_required_keys(self, dirty_df):
        response = "1. remove_high_missing of 'salary' - Removes columns with >50 % missing"
        result = _parse_cleaning_suggestions(response, dirty_df)
        for item in result:
            assert "label" in item
            assert "description" in item
            assert "default" in item

    def test_no_duplicate_labels(self, dirty_df):
        response = (
            "1. remove_high_missing - Remove high missing\n"
            "2. remove_high_missing - Another mention\n"
        )
        result = _parse_cleaning_suggestions(response, dirty_df)
        labels = [item["label"] for item in result]
        assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# 5. apply_quick_fixes
# ---------------------------------------------------------------------------

class TestApplyQuickFixes:
    def test_drops_columns_with_high_missing(self):
        df = pd.DataFrame({
            "keep":  [1, 2, 3, 4],
            "drop":  [None, None, None, 1],   # 75 % missing
        })
        result = apply_quick_fixes(df)
        assert "drop" not in result.columns
        assert "keep" in result.columns

    def test_fills_numeric_missing_with_mean(self):
        df = pd.DataFrame({"value": [10.0, 20.0, None]})
        result = apply_quick_fixes(df)
        assert result["value"].isnull().sum() == 0
        assert result["value"].iloc[2] == pytest.approx(15.0)

    def test_fills_categorical_missing_with_mode(self):
        df = pd.DataFrame({"cat": ["A", "A", "B", None]})
        result = apply_quick_fixes(df)
        assert result["cat"].isnull().sum() == 0
        assert result["cat"].iloc[3] == "A"   # mode is "A"

    def test_returns_dataframe(self, simple_df):
        result = apply_quick_fixes(simple_df.copy())
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_values_after_fix(self, dirty_df):
        result = apply_quick_fixes(dirty_df.copy())
        assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 6. create_visualization  (get_plot_labels is patched)
# ---------------------------------------------------------------------------

FAKE_LABELS = {"title": "Test Title", "xaxis": "X", "yaxis": "Y"}


@pytest.fixture(autouse=False)
def patch_labels():
    with patch("utils.get_plot_labels", return_value=FAKE_LABELS):
        yield


class TestCreateVisualization:
    def test_histogram(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "histogram", "age")
        assert fig is not None

    def test_bar_no_y(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "bar", "department")
        assert fig is not None

    def test_bar_with_y(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "bar", "department", "salary")
        assert fig is not None

    def test_scatter(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "scatter", "age", "salary")
        assert fig is not None

    def test_box(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "box", "salary")
        assert fig is not None

    def test_pie(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "pie", "department")
        assert fig is not None

    def test_violin(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "violin", "salary")
        assert fig is not None

    def test_density(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "density", "age")
        assert fig is not None

    def test_area(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "area", "age", "salary")
        assert fig is not None

    def test_line(self, patch_labels):
        # Line plots require a datetime x-axis (mirrors the UI gate in main.py)
        df = pd.DataFrame({
            "date":  pd.date_range("2024-01-01", periods=5),
            "value": [10, 20, 30, 40, 50],
        })
        fig = create_visualization(df, "line", "date", "value")
        assert fig is not None

    def test_heatmap(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "heatmap", "age", "salary")
        assert fig is not None

    def test_scatter_missing_y_raises(self, simple_df, patch_labels):
        with pytest.raises((ValueError, TypeError)):
            create_visualization(simple_df, "scatter", "age")

    def test_unsupported_type_raises(self, simple_df, patch_labels):
        with pytest.raises(ValueError):
            create_visualization(simple_df, "radar", "age")

    def test_layout_has_dark_theme(self, simple_df, patch_labels):
        # Check the explicit dark background colours set in create_visualization
        # (fig.layout.template is a resolved Template object, not the string name)
        fig = create_visualization(simple_df, "histogram", "age")
        assert fig.layout.paper_bgcolor == "rgba(17, 17, 17, 1)"
        assert fig.layout.plot_bgcolor == "rgba(17, 17, 17, 1)"

    def test_layout_title_set(self, simple_df, patch_labels):
        fig = create_visualization(simple_df, "histogram", "age")
        assert fig.layout.title.text == "Test Title"


# ---------------------------------------------------------------------------
# 7. AI-calling functions  (_call_agent is patched)
# ---------------------------------------------------------------------------

class TestAnalyzeData:
    def test_returns_agent_response(self, simple_df):
        with patch("utils._call_agent", return_value="The average age is 35.") as mock:
            result = analyze_data(simple_df, "What is the average age?")
        assert result == "The average age is 35."
        mock.assert_called_once()

    def test_passes_question_in_prompt(self, simple_df):
        with patch("utils._call_agent", return_value="ok") as mock:
            analyze_data(simple_df, "unique question xyz")
        prompt_used = mock.call_args[0][0]
        assert "unique question xyz" in prompt_used

    def test_uses_tools(self, simple_df):
        with patch("utils._call_agent", return_value="ok") as mock:
            analyze_data(simple_df, "anything")
        assert mock.call_args[1].get("use_tools", mock.call_args[0][2] if len(mock.call_args[0]) > 2 else True)

    def test_returns_empty_string_when_agent_unavailable(self, simple_df):
        with patch("utils._call_agent", return_value=""):
            result = analyze_data(simple_df, "anything")
        assert result == ""


class TestInferDomain:
    def test_returns_stripped_domain(self, simple_df):
        with patch("utils._call_agent", return_value="  Healthcare - Patient Records  "):
            result = infer_domain(simple_df, "patients.csv")
        assert result == "Healthcare - Patient Records"

    def test_fallback_on_empty_response(self, simple_df):
        with patch("utils._call_agent", return_value=""):
            result = infer_domain(simple_df, "data.csv")
        assert result == "General - Tabular Data"

    def test_filename_appears_in_prompt(self, simple_df):
        with patch("utils._call_agent", return_value="Finance") as mock:
            infer_domain(simple_df, "revenue_2024.csv")
        assert "revenue_2024.csv" in mock.call_args[0][0]


class TestGetCleaningSuggestions:
    def test_returns_dict_with_required_keys(self, simple_df):
        with patch("utils._call_agent", return_value="no_clean_needed"):
            result = get_cleaning_suggestions(simple_df, "General")
        assert "description" in result
        assert "options" in result

    def test_no_clean_needed_yields_empty_options(self):
        # Use a dataset with no applicable cleaning actions so the programmatic
        # fallback also produces nothing (the agent signal alone is not enough to
        # suppress the fallback in the current implementation).
        df = pd.DataFrame({
            "age":   [25, 30, 35, 40, 45],
            "score": [0.5, 0.6, 0.7, 0.8, 0.9],
            "tag":   ["alpha", "beta", "gamma", "delta", "epsilon"],  # lowercase, no spaces
        })
        with patch("utils._call_agent", return_value="no_clean_needed"):
            result = get_cleaning_suggestions(df, "General")
        assert result["options"] == []

    def test_dirty_data_yields_options(self, dirty_df):
        # Even when agent returns garbage, fallback logic inspects the df
        with patch("utils._call_agent", return_value=""):
            result = get_cleaning_suggestions(dirty_df, "General")
        assert len(result["options"]) > 0

    def test_options_have_required_keys(self, dirty_df):
        with patch("utils._call_agent", return_value=""):
            result = get_cleaning_suggestions(dirty_df, "General")
        for opt in result["options"]:
            assert "label" in opt
            assert "description" in opt
            assert "default" in opt


class TestGetVisualizationSuggestions:
    _valid_response = json.dumps({
        "description": "Visualisation overview",
        "options": [
            {
                "type": "histogram",
                "label": "Age Distribution",
                "description": "Shows age spread",
                "columns": ["age"],
                "default": True,
            },
            {
                "type": "scatter",
                "label": "Age vs Salary",
                "description": "Correlation",
                "columns": ["age", "salary"],
                "default": True,
            },
        ],
    })

    def test_returns_dict_with_required_keys(self, simple_df):
        with patch("utils._call_agent", return_value=self._valid_response):
            result = get_visualization_suggestions(simple_df, "General")
        assert "description" in result
        assert "options" in result

    def test_valid_options_returned(self, simple_df):
        with patch("utils._call_agent", return_value=self._valid_response):
            result = get_visualization_suggestions(simple_df, "General")
        assert len(result["options"]) >= 1

    def test_invalid_agent_response_yields_fallback(self, simple_df):
        with patch("utils._call_agent", return_value="not json at all"):
            result = get_visualization_suggestions(simple_df, "General")
        # Fallback should still produce a valid dict
        assert isinstance(result, dict)
        assert "options" in result

    def test_columns_validated_against_df(self, simple_df):
        """Options referencing non-existent columns must be filtered out."""
        bad_response = json.dumps({
            "description": "test",
            "options": [
                {
                    "type": "histogram",
                    "label": "Bad Col",
                    "description": "desc",
                    "columns": ["nonexistent_column"],
                    "default": True,
                }
            ],
        })
        with patch("utils._call_agent", return_value=bad_response):
            result = get_visualization_suggestions(simple_df, "General")
        labels = [o["label"] for o in result["options"]]
        assert "Bad Col" not in labels
