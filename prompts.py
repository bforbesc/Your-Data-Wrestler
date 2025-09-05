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

CLEANING_SUGGESTIONS_PROMPT = """Suggest data cleaning for:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Missing: {missing_values}
Domain: {domain}

Focus on:
1. Missing values
2. Data types
3. Outliers
4. Domain needs

List actionable items."""

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