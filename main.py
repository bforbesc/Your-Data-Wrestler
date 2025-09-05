
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
    # Check if this is a new file upload (different from current session)
    current_file_name = st.session_state.get('current_file_name')
    if current_file_name != uploaded_file.name:
        # Clear all previous analysis when new file is uploaded
        st.session_state.generated_figs = []
        st.session_state.custom_figs = []
        st.session_state.cleaning_suggestions = None
        st.session_state.visualization_suggestions = None
        st.session_state.domain = "Analyzing data to determine domain..."
        st.session_state.editing_domain = False
        st.session_state.current_file_name = uploaded_file.name
        st.rerun()
    
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
        
        # Check if we have cleaning suggestions and determine if dataset is clean
        dataset_needs_cleaning = False
        
        if isinstance(st.session_state.cleaning_suggestions, dict) and "description" in st.session_state.cleaning_suggestions and "options" in st.session_state.cleaning_suggestions:
            # Determine if there are any supported actions; if none, show clean message here too
            ALLOWED_CLEANING_LABELS = {
                "Remove columns with >50% missing values",
                "Fill missing values with mean (numeric) or mode (categorical)",
                "Remove duplicate rows",
                "Convert string columns to lowercase",
                "Remove leading/trailing whitespace",
                "Convert string numbers to numeric format",
                "Remove extreme outliers from numeric columns",
                "Standardize date column formats",
                "Remove completely empty rows",
                "Fix incorrect data types"
            }
            _options = st.session_state.cleaning_suggestions.get("options", [])
            filtered_options = [
                opt for opt in _options
                if isinstance(opt, dict) and opt.get("label") in ALLOWED_CLEANING_LABELS
            ]
            if filtered_options:
                st.write(st.session_state.cleaning_suggestions["description"])
                dataset_needs_cleaning = True
            else:
                # Force clean message if no applicable actions found
                st.success("Dataset appears clean ✅ No cleaning actions recommended.")
                dataset_needs_cleaning = False
        else:
            st.warning("Cleaning suggestions were malformed; checking what's actually needed.")
            # Import the function we need
            from utils import _get_applicable_cleaning_actions
            
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
            st.session_state.cleaning_suggestions = {
                "description": "Data cleaning suggestions based on dataset analysis",
                "options": options
            }
        
        # Data cleaning options - only show if there are cleaning actions available
        if dataset_needs_cleaning and isinstance(st.session_state.cleaning_suggestions, dict) and "options" in st.session_state.cleaning_suggestions:
            # Get the filtered options (we already calculated these above)
            ALLOWED_CLEANING_LABELS = {
                "Remove columns with >50% missing values",
                "Fill missing values with mean (numeric) or mode (categorical)",
                "Remove duplicate rows",
                "Convert string columns to lowercase",
                "Remove leading/trailing whitespace",
                "Convert string numbers to numeric format",
                "Remove extreme outliers from numeric columns",
                "Standardize date column formats",
                "Remove completely empty rows",
                "Fix incorrect data types"
            }

            filtered_options = [
                opt for opt in st.session_state.cleaning_suggestions["options"]
                if isinstance(opt, dict) and opt.get("label") in ALLOWED_CLEANING_LABELS
            ]

            # Show cleaning options without the "Apply Data Cleaning" header
            selected_options = {}
            for option in filtered_options:
                selected_options[option["label"]] = st.checkbox(
                    f"{option['label']} - {option['description']}", 
                    value=option.get("default", False)
                )
            
            # Apply button outside the loop
            if st.button("Apply Selected Cleaning"):
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
                
                if selected_options.get("Convert string numbers to numeric format", False):
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            # Try to convert to numeric, keeping original if conversion fails
                            try:
                                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                            except Exception:
                                pass
                
                if selected_options.get("Remove extreme outliers from numeric columns", False):
                    for col in cleaned_df.columns:
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            Q1 = cleaned_df[col].quantile(0.25)
                            Q3 = cleaned_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            # Remove outliers using IQR method
                            cleaned_df = cleaned_df[~((cleaned_df[col] < Q1 - 1.5 * IQR) | (cleaned_df[col] > Q3 + 1.5 * IQR))]
                
                if selected_options.get("Standardize date column formats", False):
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            try:
                                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='ignore')
                            except Exception:
                                pass
                
                if selected_options.get("Remove completely empty rows", False):
                    cleaned_df = cleaned_df.dropna(how='all')
                
                if selected_options.get("Fix incorrect data types", False):
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            # Try to convert to numeric first
                            try:
                                numeric_converted = pd.to_numeric(cleaned_df[col], errors='coerce')
                                if not numeric_converted.isna().all():
                                    cleaned_df[col] = numeric_converted
                                    continue
                            except Exception:
                                pass
                            # Try to convert to datetime
                            try:
                                datetime_converted = pd.to_datetime(cleaned_df[col], errors='coerce')
                                if not datetime_converted.isna().all():
                                    cleaned_df[col] = datetime_converted
                            except Exception:
                                pass
                
                # Show a quick preview of cleaned data
                st.subheader("Cleaned Data Preview")
                st.dataframe(
                    cleaned_df.head(100),
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )

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
                # Clear existing generated plots to show only newly selected ones
                st.session_state.generated_figs = []
                new_figs = []
                # Track what we're creating in this batch to prevent duplicates within the batch
                batch_plots = set()
                
                
                for viz_type, selected in selected_visualizations.items():
                    if selected:
                        option = next((opt for opt in st.session_state.visualization_suggestions["options"] if opt["label"] == viz_type), None)
                        if option and option["columns"]:
                            # Filter columns to only include those that exist in the dataframe
                            valid_columns = [col for col in option["columns"] if col in df.columns]
                            if not valid_columns:
                                st.warning(f"No valid columns found for {viz_type}. Available columns: {list(df.columns)}")
                                continue
                            
                            # Generate the appropriate plot based on type
                            if option["type"] == "histogram":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Distribution"
                                filename = f"histogram_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "histogram", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)  # Track this batch
                                    except Exception as e:
                                        st.warning(f"Error creating histogram for {col}: {str(e)}")
                                else:
                                    st.info(f"Histogram for {col} already being created in this batch. Skipping.")
                            elif option["type"] == "scatter":
                                # Use only the first two valid columns to ensure 1:1 mapping
                                if len(valid_columns) >= 2:
                                    col1, col2 = valid_columns[0], valid_columns[1]
                                    label = f"{col1} vs {col2} Scatter"
                                    filename = f"scatter_{col1}_{col2}"
                                    plot_key = (label, filename)
                                    
                                    if plot_key not in batch_plots:
                                        try:
                                            fig = create_visualization(df, "scatter", col1, col2)
                                            new_figs.append({"fig": fig, "label": label, "filename": filename})
                                            batch_plots.add(plot_key)  # Track this batch
                                        except Exception as e:
                                            st.warning(f"Error creating scatter plot for {col1} vs {col2}: {str(e)}")
                                    else:
                                        st.info(f"Scatter plot for {col1} vs {col2} already being created in this batch. Skipping.")
                                else:
                                    st.warning(f"Need at least 2 columns for scatter plot. Available: {valid_columns}")
                            elif option["type"] == "bar":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Bar Chart"
                                filename = f"bar_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "bar", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)  # Track this batch
                                    except Exception as e:
                                        st.warning(f"Error creating bar chart for {col}: {str(e)}")
                                else:
                                    st.info(f"Bar chart for {col} already being created in this batch. Skipping.")
                                        
                            # Handle additional plot types that might be in recommendations
                            elif option["type"] == "box":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Box Plot"
                                filename = f"box_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "box", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)
                                    except Exception as e:
                                        st.warning(f"Error creating box plot for {col}: {str(e)}")
                                        
                            elif option["type"] == "line":
                                # Use only the first two valid columns to ensure 1:1 mapping
                                if len(valid_columns) >= 2:
                                    col1, col2 = valid_columns[0], valid_columns[1]
                                    label = f"{col1} vs {col2} Line Plot"
                                    filename = f"line_{col1}_{col2}"
                                    plot_key = (label, filename)
                                    
                                    if plot_key not in batch_plots:
                                        try:
                                            fig = create_visualization(df, "line", col1, col2)
                                            new_figs.append({"fig": fig, "label": label, "filename": filename})
                                            batch_plots.add(plot_key)
                                        except Exception as e:
                                            st.warning(f"Error creating line plot for {col1} vs {col2}: {str(e)}")
                                else:
                                    st.warning(f"Line plot requires at least 2 columns, but only {len(valid_columns)} valid columns found for {viz_type}")
                                
                            elif option["type"] == "stacked_bar":
                                # Use only the first two valid columns to ensure 1:1 mapping
                                if len(valid_columns) >= 2:
                                    col1, col2 = valid_columns[0], valid_columns[1]
                                    label = f"{col1} vs {col2} Stacked Bar"
                                    filename = f"stacked_bar_{col1}_{col2}"
                                    plot_key = (label, filename)
                                    
                                    if plot_key not in batch_plots:
                                        try:
                                            fig = create_visualization(df, "stacked_bar", col1, col2)
                                            new_figs.append({"fig": fig, "label": label, "filename": filename})
                                            batch_plots.add(plot_key)
                                        except Exception as e:
                                            st.warning(f"Error creating stacked bar for {col1} vs {col2}: {str(e)}")
                                else:
                                    st.warning(f"Stacked bar requires at least 2 columns, but only {len(valid_columns)} valid columns found for {viz_type}")
                                
                            elif option["type"] == "area":
                                # Use only the first two valid columns to ensure 1:1 mapping
                                if len(valid_columns) >= 2:
                                    col1, col2 = valid_columns[0], valid_columns[1]
                                    label = f"{col1} vs {col2} Area Chart"
                                    filename = f"area_{col1}_{col2}"
                                    plot_key = (label, filename)
                                    
                                    if plot_key not in batch_plots:
                                        try:
                                            fig = create_visualization(df, "area", col1, col2)
                                            new_figs.append({"fig": fig, "label": label, "filename": filename})
                                            batch_plots.add(plot_key)
                                        except Exception as e:
                                            st.warning(f"Error creating area chart for {col1} vs {col2}: {str(e)}")
                                else:
                                    st.warning(f"Area chart requires at least 2 columns, but only {len(valid_columns)} valid columns found for {viz_type}")
                                
                            elif option["type"] == "pie":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Pie Chart"
                                filename = f"pie_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "pie", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)
                                    except Exception as e:
                                        st.warning(f"Error creating pie chart for {col}: {str(e)}")
                                        
                            elif option["type"] == "heatmap":
                                # Use only the first two valid columns to ensure 1:1 mapping
                                if len(valid_columns) >= 2:
                                    col1, col2 = valid_columns[0], valid_columns[1]
                                    label = f"{col1} vs {col2} Heatmap"
                                    filename = f"heatmap_{col1}_{col2}"
                                    plot_key = (label, filename)
                                    
                                    if plot_key not in batch_plots:
                                        try:
                                            fig = create_visualization(df, "heatmap", col1, col2)
                                            new_figs.append({"fig": fig, "label": label, "filename": filename})
                                            batch_plots.add(plot_key)
                                        except Exception as e:
                                            st.warning(f"Error creating heatmap for {col1} vs {col2}: {str(e)}")
                                else:
                                    st.warning(f"Heatmap requires at least 2 columns, but only {len(valid_columns)} valid columns found for {viz_type}")
                                
                            elif option["type"] == "violin":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Violin Plot"
                                filename = f"violin_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "violin", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)
                                    except Exception as e:
                                        st.warning(f"Error creating violin plot for {col}: {str(e)}")
                                        
                            elif option["type"] == "density":
                                # Use only the first valid column to ensure 1:1 mapping
                                col = valid_columns[0]
                                label = f"{col} Density Plot"
                                filename = f"density_{col}"
                                plot_key = (label, filename)
                                
                                if plot_key not in batch_plots:
                                    try:
                                        fig = create_visualization(df, "density", col)
                                        new_figs.append({"fig": fig, "label": label, "filename": filename})
                                        batch_plots.add(plot_key)
                                    except Exception as e:
                                        st.warning(f"Error creating density plot for {col}: {str(e)}")
                                        
                            elif option["type"] not in ["histogram", "scatter", "bar", "box", "line", "stacked_bar", "area", "pie", "heatmap", "violin", "density"]:
                                st.warning(f"Unsupported plot type '{option['type']}' for {viz_type}. Skipping.")
                        else:
                            st.warning(f"❌ No option found for '{viz_type}' or option has no columns")
                
                if new_figs:
                    st.session_state.generated_figs = new_figs
                    st.success(f"✅ Generated {len(new_figs)} visualizations!")
                    for fig_info in new_figs:
                        st.write(f"  - {fig_info['label']}")
                else:
                    st.error("❌ No visualizations generated. Please check the debug info above for issues.")
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
            plot_types = ["bar", "histogram", "box", "scatter", "pie", "violin", "density", "stacked_bar", "area", "heatmap"]
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
                # Clear previous custom plots to show only the new one
                st.session_state.custom_figs = []
                
                # Check for duplicates in custom visualizations
                custom_label = f"Custom: {plot_type.title()} - {x_col}{(' vs ' + y_col) if y_col else ''}"
                existing_custom_labels = {fig["label"] for fig in st.session_state.custom_figs}
                
                if custom_label in existing_custom_labels:
                    st.warning(f"Custom plot '{custom_label}' already exists. Please choose different columns or plot type.")
                else:
                    fig = create_visualization(df, plot_type, x_col, y_col)
                    st.session_state.custom_figs.append({
                        "fig": fig,
                        "label": custom_label,
                        "filename": f"{plot_type}_{x_col}_{y_col if y_col else 'count'}"
                    })
                    st.success(f"Created custom plot: {custom_label}")
                    st.rerun()
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        # Render saved figures with download buttons so they persist after downloads
        def _render_saved_figs(title: str, items: list):
            if not items:
                return
            st.subheader(title)
            for i, item in enumerate(items):
                fig = item.get("fig")
                label = item.get("label", "Plot")
                base = item.get("filename", "plot")
                st.plotly_chart(fig, use_container_width=True)
                try:
                    html = fig.to_html(
                        include_plotlyjs='cdn',
                        config={
                            'displayModeBar': True,
                            'displaylogo': False,
                            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                        }
                    )
                    st.download_button(
                        label=f"Download {label} as HTML",
                        data=html,
                        file_name=f"{base}.html",
                        mime="text/html",
                        key=f"html_download_{title}_{i}_{base}"
                    )
                    try:
                        png_bytes = fig.to_image(format="png", scale=2)
                        st.download_button(
                            label=f"Download {label} as PNG",
                            data=png_bytes,
                            file_name=f"{base}.png",
                            mime="image/png",
                            key=f"png_download_{title}_{i}_{base}"
                        )
                    except Exception as e_png:
                        st.error(f"Error exporting PNG: {str(e_png)}")
                except Exception as e_html:
                    st.error(f"Error saving plot: {str(e_html)}")

        _render_saved_figs("Generated Visualizations", st.session_state.generated_figs)
        _render_saved_figs("Custom Visualizations", st.session_state.custom_figs)