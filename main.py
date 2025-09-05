"""
Main Streamlit app for interactive data analysis.
"""

import streamlit as st
import pandas as pd
from utils import (
    get_dataset_info,
    analyze_data,
    get_cleaning_suggestions,
    get_visualization_suggestions,
    create_visualization,
    apply_quick_fixes,
    infer_domain
)
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Your Data Wrestler",
    page_icon="🤼‍♀️",
    layout="wide"
)

# Custom CSS for Matrix-style headings
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #00FF00 !important;
        font-family: monospace;
        # text-shadow: 0 0 6px #00FF00;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "cleaning_suggestions" not in st.session_state:
    st.session_state.cleaning_suggestions = None
if "visualization_suggestions" not in st.session_state:
    st.session_state.visualization_suggestions = None
if "domain" not in st.session_state:
    st.session_state.domain = "Analyzing data to determine domain..."
if "editing_domain" not in st.session_state:
    st.session_state.editing_domain = False
if "generated_figs" not in st.session_state:
    st.session_state.generated_figs = []  # list of dicts: {"fig": go.Figure, "label": str, "filename": str}
if "custom_figs" not in st.session_state:
    st.session_state.custom_figs = []

# Main content
st.title("Your Data Wrestler")
st.write("Upload a file to begin analyzing your data.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "xlsx", "xls"])

# Sidebar
st.sidebar.title("Settings")
show_overview = st.sidebar.checkbox("Data Overview", value=True)
show_cleaning = st.sidebar.checkbox("Data Cleaning", value=True)
show_visualization = st.sidebar.checkbox("Data Visualization", value=True)

if uploaded_file is not None:
    # Read the file based on its type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'txt':
            # Try to detect delimiter for txt files
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()
        
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()
    
    # Domain Context
    st.subheader("Domain Context")
    if st.session_state.domain == "Analyzing data to determine domain...":
        with st.spinner("Inferring domain context..."):
            st.session_state.domain = infer_domain(st.session_state.df, uploaded_file.name)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.editing_domain:
            new_domain = st.text_input(
                "Edit Domain Context",
                value=st.session_state.domain,
                label_visibility="collapsed"
            )
            if new_domain != st.session_state.domain:
                st.session_state.domain = new_domain
                st.session_state.cleaning_suggestions = None
                st.session_state.visualization_suggestions = None
        else:
            st.write(f"**Current Domain:** {st.session_state.domain}")
    
    with col2:
        if st.button("Edit Domain" if not st.session_state.editing_domain else "Save Domain"):
            st.session_state.editing_domain = not st.session_state.editing_domain
            st.rerun()
    
    # Natural language analysis
    st.subheader("Ask Questions About Your Data")
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Analyzing..."):
            answer = analyze_data(df, question)
            st.write(answer)
    
    # Display basic metadata
    if show_overview:
        st.header("Data Overview")
        info = get_dataset_info(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", info["num_rows"])
            st.metric("Number of Columns", info["num_cols"])
        
        with col2:
            st.write("Column Types:")
            st.write(info["column_info"])
        
        # Missing values
        st.write("Missing Values (%):")
        st.write(info["missing_values"])
        
        # Display top 5 rows
        st.subheader("Data Preview")
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    # Data Cleaning Section
    if show_cleaning:
        st.header("Data Cleaning")
        
        # Cleaning suggestions
        st.subheader("Data Cleaning Suggestions")
        if st.session_state.cleaning_suggestions is None:
            with st.spinner("Generating cleaning suggestions..."):
                try:
                    st.session_state.cleaning_suggestions = get_cleaning_suggestions(df, st.session_state.domain)
                except Exception as e:
                    st.error(f"Error generating cleaning suggestions: {str(e)}")
                    st.session_state.cleaning_suggestions = {
                        "description": "Unable to generate cleaning suggestions. Please try again.",
                        "options": [
                            {"label": "Remove columns with >50% missing values", "description": "Drop columns that have more than 50% missing values", "default": True},
                            {"label": "Fill missing values with mean (numeric) or mode (categorical)", "description": "Fill missing values using appropriate statistical methods", "default": True},
                            {"label": "Remove duplicate rows", "description": "Remove any duplicate entries in the dataset", "default": True},
                            {"label": "Convert string columns to lowercase", "description": "Standardize text data by converting to lowercase", "default": False},
                            {"label": "Remove leading/trailing whitespace", "description": "Clean up text data by removing extra spaces", "default": True}
                        ]
                    }
        
        if isinstance(st.session_state.cleaning_suggestions, dict) and "description" in st.session_state.cleaning_suggestions:
            # Determine if there are any supported actions; if none, show clean message here too
            _allowed_labels_preview = {
                "Remove columns with >50% missing values",
                "Fill missing values with mean (numeric) or mode (categorical)",
                "Remove duplicate rows",
                "Convert string columns to lowercase",
                "Remove leading/trailing whitespace",
            }
            _options = st.session_state.cleaning_suggestions.get("options", [])
            _supported_exists = any(
                isinstance(opt, dict) and opt.get("label") in _allowed_labels_preview for opt in _options
            )
            if _supported_exists:
                st.write(st.session_state.cleaning_suggestions["description"])
            else:
                st.success("Dataset appears clean ✅ No cleaning actions recommended.")
        else:
            st.warning("Cleaning suggestions were malformed; using safe defaults.")
            st.session_state.cleaning_suggestions = {
                "description": "Default cleaning suggestions applied.",
                "options": [
                    {"label": "Remove columns with >50% missing values", "description": "Drop columns that have more than 50% missing values", "default": True},
                    {"label": "Fill missing values with mean (numeric) or mode (categorical)", "description": "Fill missing values using appropriate statistical methods", "default": True},
                    {"label": "Remove duplicate rows", "description": "Remove any duplicate entries in the dataset", "default": True},
                    {"label": "Convert string columns to lowercase", "description": "Standardize text data by converting to lowercase", "default": False},
                    {"label": "Remove leading/trailing whitespace", "description": "Clean up text data by removing extra spaces", "default": True}
                ]
            }
        
        # Data cleaning options
        st.subheader("Apply Data Cleaning")
        selected_options = {}
        
        if isinstance(st.session_state.cleaning_suggestions, dict) and "options" in st.session_state.cleaning_suggestions:
            # Only show supported cleaning actions
            ALLOWED_CLEANING_LABELS = {
                "Remove columns with >50% missing values",
                "Fill missing values with mean (numeric) or mode (categorical)",
                "Remove duplicate rows",
                "Convert string columns to lowercase",
                "Remove leading/trailing whitespace",
            }

            filtered_options = [
                opt for opt in st.session_state.cleaning_suggestions["options"]
                if isinstance(opt, dict) and opt.get("label") in ALLOWED_CLEANING_LABELS
            ]

            if not filtered_options:
                st.success("Dataset appears clean ✅ No cleaning actions recommended.")
                filtered_options = []

            if filtered_options:
                for option in filtered_options:
                    selected_options[option["label"]] = st.checkbox(
                        f"{option['label']} - {option['description']}", 
                        value=option.get("default", False)
                    )
            
            if filtered_options and st.button("Apply Selected Cleaning"):
                cleaned_df = df.copy()
                
                if selected_options.get("Remove columns with >50% missing values", False):
                    threshold = 0.5
                    cols_to_drop = [col for col in cleaned_df.columns if cleaned_df[col].isnull().mean() > threshold]
                    cleaned_df = cleaned_df.drop(columns=cols_to_drop)
                
                if selected_options.get("Fill missing values with mean (numeric) or mode (categorical)", False):
                    for col in cleaned_df.columns:
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            try:
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].astype(float).mean())
                            except Exception:
                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        else:
                            try:
                                mode_series = cleaned_df[col].mode()
                                if not mode_series.empty:
                                    cleaned_df[col] = cleaned_df[col].fillna(mode_series.iloc[0])
                            except Exception:
                                pass
                
                if selected_options.get("Remove duplicate rows", False):
                    cleaned_df = cleaned_df.drop_duplicates()
                
                if selected_options.get("Convert string columns to lowercase", False):
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].str.lower()
                
                if selected_options.get("Remove leading/trailing whitespace", False):
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            cleaned_df[col] = cleaned_df[col].str.strip()
                
                # Show a quick preview of cleaned data
                st.subheader("Cleaned Data Preview")
                st.dataframe(
                    cleaned_df.head(100),
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )

                # Option to replace current dataset with cleaned version
                if st.checkbox("Replace current dataset with cleaned version"):
                    st.session_state.df = cleaned_df
                    st.success("Replaced dataset with cleaned version. Subsequent analyses will use cleaned data.")

                # Download button for cleaned data
                csv = cleaned_df.to_csv(index=False)
                original_filename = uploaded_file.name
                cleaned_filename = original_filename.rsplit('.', 1)[0] + '_cleaned.csv'
                st.download_button(
                    label="Download Cleaned Data",
                    data=csv,
                    file_name=cleaned_filename,
                    mime='text/csv'
                )
        else:
            st.warning("No cleaning options found; using safe defaults.")
            st.session_state.cleaning_suggestions = {
                "description": "Default cleaning suggestions applied.",
                "options": [
                    {"label": "Remove columns with >50% missing values", "description": "Drop columns that have more than 50% missing values", "default": True},
                    {"label": "Fill missing values with mean (numeric) or mode (categorical)", "description": "Fill missing values using appropriate statistical methods", "default": True},
                    {"label": "Remove duplicate rows", "description": "Remove any duplicate entries in the dataset", "default": True},
                    {"label": "Convert string columns to lowercase", "description": "Standardize text data by converting to lowercase", "default": False},
                    {"label": "Remove leading/trailing whitespace", "description": "Clean up text data by removing extra spaces", "default": True}
                ]
            }
    
    # Data Visualization Section
    if show_visualization:
        st.header("Data Visualization")
        
        # Visualization suggestions
        st.subheader("Data Visualization Suggestions")
        if st.session_state.visualization_suggestions is None:
            with st.spinner("Generating visualization suggestions..."):
                try:
                    st.session_state.visualization_suggestions = get_visualization_suggestions(df, st.session_state.domain)
                except Exception as e:
                    st.error(f"Error generating visualization suggestions: {str(e)}")
                    st.session_state.visualization_suggestions = {
                        "description": "Unable to generate visualization suggestions. Please try again.",
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
        
        if isinstance(st.session_state.visualization_suggestions, dict) and "description" in st.session_state.visualization_suggestions:
            st.write(st.session_state.visualization_suggestions["description"])
        else:
            st.error("Invalid visualization suggestions format. Please try refreshing the page.")
            st.session_state.visualization_suggestions = None
        
        # Visualization options
        st.subheader("Apply Data Visualization Suggestions")
        selected_visualizations = {}
        
        if isinstance(st.session_state.visualization_suggestions, dict) and "options" in st.session_state.visualization_suggestions:
            for option in st.session_state.visualization_suggestions["options"]:
                selected_visualizations[option["label"]] = st.checkbox(
                    f"{option['label']} - {option['description']}", 
                    value=option["default"]
                )
            
            if st.button("Generate Selected Visualizations"):
                new_figs = []
                for viz_type, selected in selected_visualizations.items():
                    if selected:
                        option = next((opt for opt in st.session_state.visualization_suggestions["options"] if opt["label"] == viz_type), None)
                        if option and option["columns"]:
                            if option["type"] == "histogram":
                                for col in option["columns"]:
                                    fig = create_visualization(df, "histogram", col)
                                    new_figs.append({"fig": fig, "label": f"{col} Distribution", "filename": f"histogram_{col}"})
                            elif option["type"] == "scatter":
                                for i, col1 in enumerate(option["columns"]):
                                    for col2 in option["columns"][i+1:]:
                                        fig = create_visualization(df, "scatter", col1, col2)
                                        new_figs.append({"fig": fig, "label": f"{col1} vs {col2} Scatter", "filename": f"scatter_{col1}_{col2}"})
                            elif option["type"] == "bar":
                                for col in option["columns"]:
                                    fig = create_visualization(df, "bar", col)
                                    new_figs.append({"fig": fig, "label": f"{col} Bar Chart", "filename": f"bar_{col}"})
                st.session_state.generated_figs.extend(new_figs)
                st.rerun()
        else:
            st.error("No visualization options available. Please try refreshing the page.")
            st.session_state.visualization_suggestions = None

        # Custom visualization controls
        st.subheader("Create Custom Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            # Check for datetime columns
            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            plot_types = ["bar", "histogram", "box", "scatter"]
            if datetime_cols:
                plot_types.append("line")
            
            plot_type = st.selectbox(
                "Select Plot Type",
                plot_types
            )
            x_col = st.selectbox("Select X-axis Column", df.columns)
        
        with col2:
            y_col = None
            if plot_type in ["bar", "box", "scatter", "line"]:
                y_col = st.selectbox("Select Y-axis Column", df.columns)
        
        if st.button("Generate Custom Plot"):
            try:
                fig = create_visualization(df, plot_type, x_col, y_col)
                st.session_state.custom_figs.append({
                    "fig": fig,
                    "label": f"Custom: {plot_type.title()} - {x_col}{(' vs ' + y_col) if y_col else ''}",
                    "filename": f"{plot_type}_{x_col}_{y_col if y_col else 'count'}"
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        # Render saved figures with download buttons so they persist after downloads
        def _render_saved_figs(title: str, items: list):
            if not items:
                return
            st.subheader(title)
            for item in items:
                fig = item.get("fig")
                label = item.get("label", "Plot")
                base = item.get("filename", "plot")
                st.plotly_chart(fig, use_container_width=True)
                try:
                    html = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label=f"Download {label} as HTML",
                        data=html,
                        file_name=f"{base}.html",
                        mime="text/html"
                    )
                    try:
                        png_bytes = fig.to_image(format="png", scale=2)
                        st.download_button(
                            label=f"Download {label} as PNG",
                            data=png_bytes,
                            file_name=f"{base}.png",
                            mime="image/png"
                        )
                    except Exception as e_png:
                        st.error(f"Error exporting PNG: {str(e_png)}")
                except Exception as e_html:
                    st.error(f"Error saving plot: {str(e_html)}")

        _render_saved_figs("Generated Visualizations", st.session_state.generated_figs)
        _render_saved_figs("Custom Visualizations", st.session_state.custom_figs)