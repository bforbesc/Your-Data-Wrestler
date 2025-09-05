"""
Helper functions for the data analysis app.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import os
from typing import Dict, List, Tuple, Optional
from prompts import ANALYSIS_PROMPT, CLEANING_SUGGESTIONS_PROMPT, VISUALIZATION_SUGGESTIONS_PROMPT
from dotenv import load_dotenv
import hashlib
import json
from functools import lru_cache
import numpy as np


# ----------------------
# JSON parsing utilities
# ----------------------
def _extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON object from an LLM response.

    Looks for the first '{' and the last '}' and attempts to parse that slice.
    Also handles fenced code blocks like ```json ... ``` by stripping fences.
    """
    if text is None:
        return None

    cleaned = text.strip()

    # Strip fenced code block wrappers if present
    if cleaned.startswith("```"):
        # Remove first fence line
        lines = cleaned.splitlines()
        # Drop the opening fence (possibly with a language tag)
        if lines:
            lines = lines[1:]
        # Drop a trailing closing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Fallback: slice from first '{' to last '}'
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]

    return None

def _parse_json_safe(text: str) -> Optional[dict]:
    """Try to parse JSON from text; return dict or None on failure."""
    if not text:
        return None
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting a JSON-looking block
    block = _extract_json_block(text)
    if block is None:
        return None
    try:
        return json.loads(block)
    except Exception:
        return None

# Load environment variables from .env file
load_dotenv()

# Lazy OpenAI client initialization to avoid crashes when SSL/CA bundle is misconfigured
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    try:
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except Exception:
        # If client cannot be created (e.g., SSL CA path invalid), disable AI features gracefully
        return None

# Cache for storing API responses
response_cache = {}

def get_cache_key(prompt: str) -> str:
    """Generate a cache key for a prompt."""
    return hashlib.md5(prompt.encode()).hexdigest()

@lru_cache(maxsize=100)
def get_openai_response(prompt: str) -> str:
    """Get response from OpenAI API with caching."""
    cache_key = get_cache_key(prompt)
    
    # Check cache first
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    client = _get_openai_client()
    if client is None:
        return ""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        result = response.choices[0].message.content
        
        # Store in cache
        response_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return ""

def get_dataset_info(df: pd.DataFrame) -> Dict:
    """Get basic information about the dataset."""
    return {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "column_info": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    }

def analyze_data(df: pd.DataFrame, question: str) -> str:
    """Analyze data using OpenAI API with optimized prompt."""
    info = get_dataset_info(df)
    
    # Optimize prompt to use fewer tokens
    prompt = ANALYSIS_PROMPT.format(
        num_rows=info["num_rows"],
        num_cols=info["num_cols"],
        column_info=json.dumps(info["column_info"]),  # Convert to JSON string to reduce tokens
        missing_values=json.dumps(info["missing_values"]),  # Convert to JSON string to reduce tokens
        question=question
    )
    return get_openai_response(prompt)

def get_cleaning_suggestions(df: pd.DataFrame, domain: str) -> dict:
    """Get data cleaning suggestions using OpenAI API with structured approach."""
    info = get_dataset_info(df)
    
    prompt = CLEANING_SUGGESTIONS_PROMPT.format(
        num_rows=info["num_rows"],
        num_cols=info["num_cols"],
        column_info=json.dumps(info["column_info"]),
        missing_values=json.dumps(info["missing_values"]),
        domain=domain
    )
    response = get_openai_response(prompt)
    
    # Parse the structured response and map to supported cleaning actions
    options = _parse_cleaning_suggestions(response, df)
    
    # If no valid options found, check what's actually needed
    if not options:
        # Define the mappings here for the fallback
        action_mapping = {
            "remove_high_missing": "Remove columns with >50% missing values",
            "fill_missing": "Fill missing values with mean (numeric) or mode (categorical)",
            "remove_duplicates": "Remove duplicate rows",
            "lowercase_strings": "Convert string columns to lowercase",
            "strip_whitespace": "Remove leading/trailing whitespace",
            "convert_numeric": "Convert string numbers to numeric format",
            "remove_outliers": "Remove extreme outliers from numeric columns",
            "standardize_dates": "Standardize date column formats",
            "remove_empty_rows": "Remove completely empty rows",
            "fix_data_types": "Fix incorrect data types"
        }
        
        descriptions = {
            "Remove columns with >50% missing values": "Drop columns that have more than 50% missing values",
            "Fill missing values with mean (numeric) or mode (categorical)": "Fill missing values using appropriate statistical methods",
            "Remove duplicate rows": "Remove any duplicate entries in the dataset",
            "Convert string columns to lowercase": "Standardize text data by converting to lowercase",
            "Remove leading/trailing whitespace": "Clean up text data by removing extra spaces",
            "Convert string numbers to numeric format": "Convert numeric data stored as strings to proper numeric format",
            "Remove extreme outliers from numeric columns": "Remove statistical outliers from numeric columns",
            "Standardize date column formats": "Convert date columns to consistent datetime format",
            "Remove completely empty rows": "Remove rows where all values are missing",
            "Fix incorrect data types": "Convert columns to their appropriate data types"
        }
        
        options = _get_applicable_cleaning_actions(df, action_mapping, descriptions)
    
    return {
        "description": "Data cleaning suggestions based on dataset analysis",
        "options": options
    }

def _parse_cleaning_suggestions(response: str, df: pd.DataFrame) -> list:
    """Parse cleaning suggestions and map to supported actions."""
    options = []
    
    # Mapping from action types to supported labels
    action_mapping = {
        "remove_high_missing": "Remove columns with >50% missing values",
        "fill_missing": "Fill missing values with mean (numeric) or mode (categorical)",
        "remove_duplicates": "Remove duplicate rows",
        "lowercase_strings": "Convert string columns to lowercase",
        "strip_whitespace": "Remove leading/trailing whitespace",
        "convert_numeric": "Convert string numbers to numeric format",
        "remove_outliers": "Remove extreme outliers from numeric columns",
        "standardize_dates": "Standardize date column formats",
        "remove_empty_rows": "Remove completely empty rows",
        "fix_data_types": "Fix incorrect data types"
    }
    
    # Default descriptions
    descriptions = {
        "Remove columns with >50% missing values": "Drop columns that have more than 50% missing values",
        "Fill missing values with mean (numeric) or mode (categorical)": "Fill missing values using appropriate statistical methods",
        "Remove duplicate rows": "Remove any duplicate entries in the dataset",
        "Convert string columns to lowercase": "Standardize text data by converting to lowercase",
        "Remove leading/trailing whitespace": "Clean up text data by removing extra spaces",
        "Convert string numbers to numeric format": "Convert numeric data stored as strings to proper numeric format",
        "Remove extreme outliers from numeric columns": "Remove statistical outliers from numeric columns",
        "Standardize date column formats": "Convert date columns to consistent datetime format",
        "Remove completely empty rows": "Remove rows where all values are missing",
        "Fix incorrect data types": "Convert columns to their appropriate data types"
    }
    
    # Check if the AI response indicates no cleaning is needed
    if "no_clean_needed" in response.lower():
        return []  # Return empty list to indicate no cleaning needed
    
    # If the AI response is empty or malformed, fall back to checking what's actually needed
    if not response or len(response.strip()) < 10:
        return _get_applicable_cleaning_actions(df, action_mapping, descriptions)
    
    # Parse the response line by line
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
            
        # Extract action type and columns
        try:
            # Look for action types in the line
            for action_type, label in action_mapping.items():
                if action_type in line.lower():
                    # Check if this action is applicable to the dataset
                    if _is_action_applicable(action_type, df):
                        options.append({
                            "label": label,
                            "description": descriptions[label],
                            "default": True
                        })
                        break
        except Exception:
            continue
    
    # If no options were parsed from AI response, fall back to checking what's actually needed
    if not options:
        return _get_applicable_cleaning_actions(df, action_mapping, descriptions)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_options = []
    for option in options:
        if option["label"] not in seen:
            seen.add(option["label"])
            unique_options.append(option)
    
    return unique_options

def _get_applicable_cleaning_actions(df: pd.DataFrame, action_mapping: dict, descriptions: dict) -> list:
    """Get cleaning actions that are actually applicable to the dataset."""
    options = []
    
    for action_type, label in action_mapping.items():
        if _is_action_applicable(action_type, df):
            options.append({
                "label": label,
                "description": descriptions[label],
                "default": True
            })
    
    return options

def _is_action_applicable(action_type: str, df: pd.DataFrame) -> bool:
    """Check if a cleaning action is applicable to the dataset."""
    if action_type == "remove_high_missing":
        return any(df[col].isnull().mean() > 0.5 for col in df.columns)
    elif action_type == "fill_missing":
        # Only suggest if there are missing values but not too many (not >50%)
        # Also require that missing values are not just a few scattered ones
        has_missing = df.isnull().any().any()
        if not has_missing:
            return False
        
        # Check if missing values are significant (more than 10% of data)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        significant_missing = missing_cells > total_cells * 0.10
        
        # Don't suggest if any column has >50% missing
        no_high_missing = not any(df[col].isnull().mean() > 0.5 for col in df.columns)
        
        return significant_missing and no_high_missing
    elif action_type == "remove_duplicates":
        # Only suggest if there are significant duplicates (more than 15% of rows)
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        return duplicate_rows > total_rows * 0.15
    elif action_type == "lowercase_strings":
        # Only suggest if there are string columns with significant mixed case
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    # Count how many values have uppercase letters
                    uppercase_count = sum(1 for val in sample_values if any(c.isupper() for c in str(val)))
                    # Only suggest if more than 60% of values have uppercase letters
                    if uppercase_count > len(sample_values) * 0.6:
                        return True
        return False
    elif action_type == "strip_whitespace":
        # Only suggest if there are string columns with significant whitespace issues
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    # Count how many values have whitespace issues
                    whitespace_count = sum(1 for val in sample_values if str(val) != str(val).strip())
                    # Only suggest if more than 50% of values have whitespace issues
                    if whitespace_count > len(sample_values) * 0.5:
                        return True
        return False
    elif action_type == "convert_numeric":
        # Only suggest if there are string columns that look like numbers
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    # More strict numeric detection
                    numeric_like_count = 0
                    for val in sample_values:
                        val_str = str(val).strip()
                        # Must look like a real number (not just digits)
                        if (val_str.replace('.', '').replace('-', '').replace(',', '').isdigit() and 
                            len(val_str) > 0 and val_str != '0' and '.' in val_str):
                            numeric_like_count += 1
                    # Only suggest if at least 95% of values look like numbers AND they have decimal points
                    if numeric_like_count > len(sample_values) * 0.95:
                        return True
        return False
    elif action_type == "remove_outliers":
        # Only suggest if there are numeric columns with outliers and enough data
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and len(df) > 10:  # Need at least 10 rows
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division by zero
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    # Only suggest if there are significant outliers (more than 20% of data)
                    if len(outliers) > len(df) * 0.20:
                        return True
        return False
    elif action_type == "standardize_dates":
        # Only suggest if there are columns that look like dates but aren't datetime
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                # More strict date detection - must look like actual dates
                date_like_count = 0
                for val in sample_values:
                    val_str = str(val).strip()
                    # Check for common date patterns
                    if (len(val_str) >= 8 and 
                        (val_str.count('/') == 2 or val_str.count('-') == 2) and
                        any(c.isdigit() for c in val_str)):
                        date_like_count += 1
                # Only suggest if at least 80% of values look like dates
                if date_like_count > len(sample_values) * 0.8:
                    return True
        return False
    elif action_type == "remove_empty_rows":
        # Only suggest if there are completely empty rows
        return df.isnull().all(axis=1).any()
    elif action_type == "fix_data_types":
        # Only suggest if there are obvious type mismatches
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(100)
                if len(sample_values) > 0:
                    # Check if most values look like numbers but are stored as strings
                    numeric_like_count = 0
                    for val in sample_values:
                        val_str = str(val).strip()
                        # More strict numeric detection - must have decimal points
                        if (val_str.replace('.', '').replace('-', '').replace(',', '').isdigit() and 
                            len(val_str) > 0 and val_str != '0' and '.' in val_str):
                            numeric_like_count += 1
                    # Only suggest if at least 98% of values look like numbers with decimals
                    if numeric_like_count > len(sample_values) * 0.98:
                        return True
        return False
    return False

def get_visualization_suggestions(df: pd.DataFrame, domain: str) -> dict:
    """Get visualization suggestions using OpenAI API with optimized prompt."""
    info = get_dataset_info(df)
    
    # Get visualization suggestions using the new prompt
    prompt = VISUALIZATION_SUGGESTIONS_PROMPT.format(
        num_rows=info["num_rows"],
        num_cols=info["num_cols"],
        column_info=json.dumps(info["column_info"]),  # Convert to JSON string to reduce tokens
        domain=domain
    )
    response = get_openai_response(prompt)
    
    # Parse the response to extract structured suggestions
    prompt = f"""Given these visualization suggestions:
{response}

Convert them to JSON format with this EXACT structure:
{{
    "description": "Brief overview of visualization strategy",
    "options": [
        {{
            "type": "histogram|scatter|bar|box|line|stacked_bar|area|pie|heatmap|violin|density",
            "label": "Clear descriptive label",
            "description": "What this visualization reveals",
            "columns": ["exact_column_name_1", "exact_column_name_2_if_needed"],
            "default": true
        }}
    ]
}}

RULES:
1. Use ONLY the supported plot types listed above
2. Use EXACT column names from the dataset
3. For single-column plots: use 1 column in the array
4. For two-column plots: use exactly 2 columns in the array
5. Set default: true for the first 2 suggestions, false for the rest
6. Generate exactly 3-5 suggestions maximum

Dataset columns available: {list(df.columns)}"""
    
    structured_response = get_openai_response(prompt)
    parsed = _parse_json_safe(structured_response)
    if isinstance(parsed, dict):
        # Ensure required keys exist
        description = parsed.get("description") or (response if isinstance(response, str) else "Visualization suggestions")
        options = parsed.get("options")
        if not isinstance(options, list) or not options:
            # Create comprehensive fallback suggestions based on data types
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            categorical_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
            
            options = []
            suggestion_count = 0
            
            # Priority 1: Histogram for first numeric column
            if numeric_cols and suggestion_count < 5:
                options.append({
                    "type": "histogram",
                    "label": f"{numeric_cols[0]} Distribution",
                    "description": f"Shows the distribution pattern of {numeric_cols[0]}",
                    "columns": [numeric_cols[0]],
                    "default": True
                })
                suggestion_count += 1
            
            # Priority 2: Scatter plot for first two numeric columns
            if len(numeric_cols) >= 2 and suggestion_count < 5:
                options.append({
                    "type": "scatter",
                    "label": f"{numeric_cols[0]} vs {numeric_cols[1]} Relationship",
                    "description": f"Shows correlation between {numeric_cols[0]} and {numeric_cols[1]}",
                    "columns": [numeric_cols[0], numeric_cols[1]],
                    "default": True
                })
                suggestion_count += 1
            
            # Priority 3: Bar chart for first categorical column
            if categorical_cols and suggestion_count < 5:
                options.append({
                    "type": "bar",
                    "label": f"{categorical_cols[0]} Frequency",
                    "description": f"Shows frequency distribution of {categorical_cols[0]}",
                    "columns": [categorical_cols[0]],
                    "default": False
                })
                suggestion_count += 1
            
            # Priority 4: Box plot for second numeric column (if exists)
            if len(numeric_cols) >= 2 and suggestion_count < 5:
                options.append({
                    "type": "box",
                    "label": f"{numeric_cols[1]} Box Plot",
                    "description": f"Shows quartiles and outliers for {numeric_cols[1]}",
                    "columns": [numeric_cols[1]],
                    "default": False
                })
                suggestion_count += 1
            
            # Priority 5: Pie chart for second categorical column (if exists)
            if len(categorical_cols) >= 2 and suggestion_count < 5:
                options.append({
                    "type": "pie",
                    "label": f"{categorical_cols[1]} Proportions",
                    "description": f"Shows proportional distribution of {categorical_cols[1]}",
                    "columns": [categorical_cols[1]],
                    "default": False
                })
                suggestion_count += 1
        
        # Validate and clean up options
        valid_options = []
        for option in options:
            if not isinstance(option, dict):
                continue
                
            plot_type = option.get("type", "")
            columns = option.get("columns", [])
            
            # Validate plot type
            if plot_type not in ["histogram", "scatter", "bar", "box", "line", "stacked_bar", "area", "pie", "heatmap", "violin", "density"]:
                continue
                
            # Validate columns exist in dataset
            valid_columns = [col for col in columns if col in df.columns]
            if not valid_columns:
                continue
                
            # Validate column count matches plot type requirements
            if plot_type in ["histogram", "bar", "box", "pie", "violin", "density"] and len(valid_columns) != 1:
                continue
            elif plot_type in ["scatter", "line", "area", "heatmap"] and len(valid_columns) < 2:
                continue
            elif plot_type == "stacked_bar" and len(valid_columns) < 2:
                continue
                
            # Create clean option
            clean_option = {
                "type": plot_type,
                "label": option.get("label", f"{plot_type.title()} of {', '.join(valid_columns)}"),
                "description": option.get("description", f"Shows {plot_type} for {', '.join(valid_columns)}"),
                "columns": valid_columns,
                "default": option.get("default", False)
            }
            valid_options.append(clean_option)
        
        # Remove duplicates based on (type, columns) combination
        seen_combinations = set()
        unique_options = []
        for option in valid_options:
            key = (option["type"], tuple(sorted(option["columns"])))
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_options.append(option)
        
        return {"description": description, "options": unique_options}

    # Fallback to default structure if parsing fails
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    
    fallback_options = []
    if numeric_cols:
        fallback_options.append({
            "type": "histogram",
            "label": f"{numeric_cols[0]} Distribution",
            "description": f"Shows the distribution of {numeric_cols[0]}",
            "columns": [numeric_cols[0]],
            "default": True
        })
    
    if categorical_cols:
        fallback_options.append({
            "type": "bar",
            "label": f"{categorical_cols[0]} Frequency",
            "description": f"Shows frequency distribution of {categorical_cols[0]}",
            "columns": [categorical_cols[0]],
            "default": False
        })
    
    return {
        "description": "Basic visualization suggestions",
        "options": fallback_options
    }

def get_plot_labels(df: pd.DataFrame, plot_type: str, x_col: str, y_col: Optional[str] = None) -> Dict[str, str]:
    """Generate better plot labels using OpenAI with optimized prompt."""
    # Optimize prompt to use fewer tokens
    prompt = f"""Generate plot labels for:
Type: {plot_type}
X: {x_col}
Y: {y_col if y_col else 'count'}
Columns: {json.dumps(df.columns.tolist())}

Format as JSON:
{{
    "title": "Plot title",
    "xaxis": "X-axis label",
    "yaxis": "Y-axis label"
}}"""
    
    response = get_openai_response(prompt)
    try:
        return json.loads(response)
    except:
        # Fallback to default labels if parsing fails
        return {
            'title': f"{plot_type.title()} Plot of {x_col}" + (f" vs {y_col}" if y_col else ""),
            'xaxis': x_col,
            'yaxis': y_col if y_col else "Count"
        }

def create_visualization(df: pd.DataFrame, plot_type: str, x_col: str, y_col: Optional[str] = None) -> go.Figure:
    """Create a Plotly visualization based on the specified type."""
    # Get better labels using OpenAI
    labels = get_plot_labels(df, plot_type, x_col, y_col)
    
    # Check if x_col is datetime
    is_time_series = pd.api.types.is_datetime64_any_dtype(df[x_col])
    
    if plot_type == "bar":
        if y_col:
            # Use graph_objects to avoid express validators issues
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col]))
        else:
            # Compute counts per category explicitly to avoid issues with None y
            counts = (
                df[x_col]
                .astype(str)
                .fillna("<NA>")
                .value_counts(dropna=False)
                .reset_index()
            )
            counts.columns = [x_col, "count"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=counts[x_col], y=counts["count"]))
    elif plot_type == "histogram":
        fig = px.histogram(df, x=x_col)
    elif plot_type == "box":
        fig = px.box(df, x=x_col, y=y_col if y_col else None)
    elif plot_type == "scatter":
        if not y_col:
            raise ValueError("y_col is required for scatter plot")
        fig = px.scatter(df, x=x_col, y=y_col)
    elif plot_type == "line" and is_time_series:
        # For time series data, create a line plot
        fig = px.line(df, x=x_col, y=y_col if y_col else None)
    elif plot_type == "stacked_bar":
        if not y_col:
            raise ValueError("y_col is required for stacked bar plot")
        # Create a stacked bar chart
        fig = px.bar(df, x=x_col, y=y_col, color=x_col)
    elif plot_type == "area":
        if not y_col:
            raise ValueError("y_col is required for area plot")
        # Create an area chart
        fig = px.area(df, x=x_col, y=y_col)
    elif plot_type == "pie":
        # Create a pie chart
        value_counts = df[x_col].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index)
    elif plot_type == "heatmap":
        if not y_col:
            raise ValueError("y_col is required for heatmap")
        # Create a correlation heatmap
        numeric_df = df[[x_col, y_col]].select_dtypes(include=[np.number])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        else:
            raise ValueError("Heatmap requires numeric columns")
    elif plot_type == "violin":
        # Create a violin plot
        fig = px.violin(df, y=x_col)
    elif plot_type == "density":
        # Create a density plot (using histogram with density)
        fig = px.histogram(df, x=x_col, histnorm='density')
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Update layout with better labels and dark theme
    fig.update_layout(
        title=labels['title'],
        xaxis_title=labels['xaxis'],
        yaxis_title=labels['yaxis'],
        template='plotly_dark',
        font=dict(size=12, color='white'),  # Ensure text is white
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=True,
        # Set dark background colors that work for both display and export
        plot_bgcolor='rgba(17, 17, 17, 1)',  # Dark background for plot area
        paper_bgcolor='rgba(17, 17, 17, 1)',  # Dark background for paper
        # Ensure axis colors are visible on dark background
        xaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',
            linecolor='rgba(128, 128, 128, 0.8)',
            tickcolor='rgba(128, 128, 128, 0.8)',
            title_font_color='white',
            tickfont_color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.3)',
            linecolor='rgba(128, 128, 128, 0.8)',
            tickcolor='rgba(128, 128, 128, 0.8)',
            title_font_color='white',
            tickfont_color='white'
        )
    )
    
    # Explicitly set colors for traces to ensure they work in both display and PNG export
    # Define a color palette that works well with dark theme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Update traces with explicit colors
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'marker'):
            trace.marker.color = colors[i % len(colors)]
        if hasattr(trace, 'line'):
            trace.line.color = colors[i % len(colors)]
        if hasattr(trace, 'fillcolor'):
            trace.fillcolor = colors[i % len(colors)]
    
    # Add hover template
    if is_time_series:
        fig.update_traces(
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y}<extra></extra>"
        )
    else:
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>"
        )
    
    return fig

def apply_quick_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-click fixes to the dataset."""
    # Drop columns with more than 50% missing values
    threshold = 0.5
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    df = df.drop(columns=cols_to_drop)
    
    # Fill remaining missing values with appropriate defaults
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def infer_domain(df: pd.DataFrame, filename: str) -> str:
    """Infer the domain of the dataset using OpenAI API with optimized prompt."""
    info = get_dataset_info(df)
    
    # Optimize prompt to use fewer tokens
    prompt = f"""Analyze dataset:
File: {filename}
Columns: {json.dumps(info['column_info'])}
Rows: {info['num_rows']}

Respond with domain (e.g., "Healthcare - Patient Records", "E-commerce - Sales")."""
    
    domain = get_openai_response(prompt)
    if isinstance(domain, str) and domain.strip():
        return domain.strip()
    # Fallback domain if AI unavailable
    return "General - Tabular Data"