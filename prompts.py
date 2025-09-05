"""
Prompt templates for the data analysis app.
"""

ANALYSIS_PROMPT = """Analyze dataset:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Missing: {missing_values}

Question: {question}

Provide concise answer. If visualization needed, suggest plot type."""

CLEANING_SUGGESTIONS_PROMPT = """Analyze this dataset and suggest data cleaning actions:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Missing: {missing_values}
Domain: {domain}

CRITICAL REQUIREMENTS:
1. If the dataset appears CLEAN and well-structured, respond with: "no_clean_needed"
2. If cleaning is needed, suggest EXACTLY 3-5 specific cleaning actions
3. Use ONLY these exact cleaning action types: remove_high_missing, fill_missing, remove_duplicates, lowercase_strings, strip_whitespace, convert_numeric, remove_outliers, standardize_dates, remove_empty_rows, fix_data_types
4. Each suggestion must specify EXACT column names from the dataset when applicable
5. Each suggestion must be unique and non-overlapping
6. ONLY suggest actions that are ACTUALLY NEEDED - be very conservative

CLEANING ACTION RULES:
- remove_high_missing: Use ONLY when columns have >50% missing values (specify column names)
- fill_missing: Use ONLY when columns have <50% missing values AND missing values are significant (>1% of data) (specify column names and method: mean/mode)
- remove_duplicates: Use ONLY when dataset has duplicate rows (no specific columns needed)
- lowercase_strings: Use ONLY when string columns have significant mixed casing (>30% of values have uppercase) (specify column names)
- strip_whitespace: Use ONLY when string columns have significant whitespace issues (>20% of values have leading/trailing spaces) (specify column names)
- convert_numeric: Use ONLY when numeric data is stored as strings AND most values (>80%) look like numbers with decimals (specify column names)
- remove_outliers: Use ONLY when there are significant outliers (>5% of data) in numeric columns (specify column names)
- standardize_dates: Use ONLY when date columns have inconsistent formats AND most values (>50%) look like dates (specify column names)
- remove_empty_rows: Use ONLY when there are completely empty rows (no specific columns needed)
- fix_data_types: Use ONLY when columns have obvious type mismatches AND most values (>90%) look like numbers with decimals (specify column names and target type)

IMPORTANT: Do NOT suggest actions if the data is already clean or if the issues are minor!

FORMAT: For each suggestion, provide:
1. Action type (from the list above)
2. Exact column names to use (when applicable)
3. Brief description of what it fixes
4. Why it's valuable for this dataset

Example format for dirty data:
1. remove_high_missing of 'column1', 'column2' - Removes columns with >50% missing data
2. fill_missing of 'age' with mean, 'category' with mode - Fills remaining missing values appropriately
3. remove_duplicates - Removes duplicate rows from dataset
4. lowercase_strings of 'name', 'category' - Standardizes text data casing
5. convert_numeric of 'price', 'quantity' - Converts string numbers to proper numeric format

Example format for clean data:
no_clean_needed

IMPORTANT: If the data looks clean, well-structured, and ready for analysis, respond with exactly "no_clean_needed" (no other text).

Generate suggestions following this format."""

VISUALIZATION_SUGGESTIONS_PROMPT = """Analyze this dataset and suggest EXACTLY 3-5 specific visualizations:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Domain: {domain}

CRITICAL REQUIREMENTS:
1. Suggest EXACTLY 3-5 visualizations (no more, no less)
2. Use ONLY these exact plot types: histogram, scatter, bar, box, line, stacked_bar, area, pie, heatmap, violin, density
3. Each suggestion must specify EXACT column names from the dataset
4. Each suggestion must be unique and non-overlapping
5. Prioritize the most insightful visualizations for this specific dataset

PLOT TYPE RULES:
- histogram: Use with 1 numerical column only
- scatter: Use with exactly 2 numerical columns
- bar: Use with 1 categorical column only  
- box: Use with 1 numerical column only
- line: Use with exactly 2 numerical columns
- stacked_bar: Use with 1 categorical + 1 numerical column
- area: Use with exactly 2 numerical columns
- pie: Use with 1 categorical column only
- heatmap: Use with 2+ numerical columns (will use first 2)
- violin: Use with 1 numerical column only
- density: Use with 1 numerical column only

FORMAT: For each suggestion, provide:
1. Plot type (from the list above)
2. Exact column names to use
3. Brief description of what it reveals
4. Why it's valuable for this dataset

Example format:
1. histogram of 'age' - Shows age distribution patterns
2. scatter of 'income' vs 'spending' - Reveals spending-income correlation
3. bar of 'category' - Shows category frequency distribution

Generate exactly 3-5 suggestions following this format."""

VISUALIZATION_PROMPT = """You are a data visualization expert. Given this dataset:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Domain: {domain}

Provide a comprehensive set of visualization suggestions that would be insightful for this specific dataset and domain. Use ONLY these supported plot types:

SUPPORTED PLOT TYPES:
- histogram: For single numerical column distributions
- scatter: For relationships between two numerical columns  
- bar: For single categorical column frequencies
- box: For single numerical column distribution analysis
- line: For trends between two numerical columns
- stacked_bar: For categorical breakdowns with two columns
- area: For filled area charts between two numerical columns
- pie: For single categorical column proportions
- heatmap: For correlation matrices between two numerical columns
- violin: For single numerical column distribution shapes
- density: For single numerical column density curves

1. Distribution Analysis
- Histograms for numerical columns
  - Identify data spread and outliers
  - Check for normal distribution
- Box plots for comparing distributions
  - Show quartiles and outliers
  - Compare across categories
- Violin plots for distribution shapes
- Density plots for smooth distribution curves

2. Relationship Analysis
- Scatter plots for numerical pairs
  - Show correlations
  - Identify clusters
- Bar charts for categorical data
  - Compare frequencies
  - Show proportions
- Heatmaps for correlation matrices
- Stacked bar charts for categorical breakdowns

3. Time Series Analysis (if applicable)
- Line plots for trends
  - Show changes over time
  - Identify patterns
- Area charts for filled trend visualization

4. Domain-Specific Visualizations
- Pie charts for proportions
- Industry standard plots
  - Common metrics
  - Key performance indicators
- Specialized comparisons
  - Domain-specific metrics
  - Custom aggregations

For each suggested visualization, explain:
- What insights it would reveal
- Why it's particularly relevant for this data
- How to interpret the results
- What patterns to look for

Format your response as a detailed list of visualization suggestions, with clear explanations of their value and interpretation.""" 