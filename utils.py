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

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        return "Error: Unable to get response from OpenAI API. Please check your API key and internet connection."

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
    """Get data cleaning suggestions using OpenAI API with optimized prompt."""
    info = get_dataset_info(df)
    
    # Optimize prompt to use fewer tokens
    prompt = CLEANING_SUGGESTIONS_PROMPT.format(
        num_rows=info["num_rows"],
        num_cols=info["num_cols"],
        column_info=json.dumps(info["column_info"]),  # Convert to JSON string to reduce tokens
        missing_values=json.dumps(info["missing_values"]),  # Convert to JSON string to reduce tokens
        domain=domain
    )
    response = get_openai_response(prompt)
    
    # Parse the response to extract structured suggestions
    prompt = f"""Format these cleaning suggestions as JSON:
{response}

Structure:
{{
    "description": "Brief explanation",
    "options": [
        {{
            "label": "Action label",
            "description": "Brief explanation",
            "default": true/false
        }}
    ]
}}"""
    
    try:
        structured_response = get_openai_response(prompt)
        return json.loads(structured_response)
    except:
        # Fallback to default structure if parsing fails
        return {
            "description": response,
            "options": [
                {"label": "Remove columns with >50% missing values", "description": "Drop columns that have more than 50% missing values", "default": True},
                {"label": "Fill missing values with mean (numeric) or mode (categorical)", "description": "Fill missing values using appropriate statistical methods", "default": True},
                {"label": "Remove duplicate rows", "description": "Remove any duplicate entries in the dataset", "default": True},
                {"label": "Convert string columns to lowercase", "description": "Standardize text data by converting to lowercase", "default": False},
                {"label": "Remove leading/trailing whitespace", "description": "Clean up text data by removing extra spaces", "default": True}
            ]
        }

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
    prompt = f"""Given these specific visualization suggestions:
{response}

Format them as JSON with this structure:
{{
    "description": "A brief overview of the visualization strategy for this specific dataset",
    "options": [
        {{
            "type": "visualization_type (histogram/scatter/bar/box)",
            "label": "A clear, specific label describing this exact visualization",
            "description": "Detailed explanation of what this specific visualization reveals about this dataset",
            "columns": ["exact_column_names_to_use"],
            "default": true/false
        }}
    ]
}}

Make sure to:
1. Use the exact column names mentioned in the suggestions
2. Keep the descriptions specific to this dataset
3. Include all relevant details from the suggestions"""
    
    try:
        structured_response = get_openai_response(prompt)
        return json.loads(structured_response)
    except:
        # Fallback to default structure if parsing fails
        return {
            "description": response,
            "options": [
                {
                    "type": "histogram",
                    "label": "Distribution Analysis",
                    "description": "Shows the distribution of numerical variables",
                    "columns": [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                    "default": True
                },
                {
                    "type": "scatter",
                    "label": "Relationship Analysis",
                    "description": "Shows relationships between numerical variables",
                    "columns": [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
                    "default": True
                },
                {
                    "type": "bar",
                    "label": "Categorical Analysis",
                    "description": "Shows frequency of categorical variables",
                    "columns": [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])],
                    "default": True
                }
            ]
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
        fig = px.bar(df, x=x_col, y=y_col if y_col else None)
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
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Update layout with better labels
    fig.update_layout(
        title=labels['title'],
        xaxis_title=labels['xaxis'],
        yaxis_title=labels['yaxis'],
        template="plotly_white",
        font=dict(size=12),
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=True
    )
    
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
    
    domain = get_openai_response(prompt).strip()
    return domain 