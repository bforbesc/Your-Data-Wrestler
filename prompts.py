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

VISUALIZATION_SUGGESTIONS_PROMPT = """Analyze this specific dataset and suggest visualizations:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Domain: {domain}

IMPORTANT: Your suggestions MUST:
1. Reference EXACT column names from the dataset
2. Include specific measurements or thresholds when relevant
3. Explain why each visualization is particularly relevant for THIS dataset
4. Consider the domain context and what insights would be most valuable

Example of a good suggestion:
"Create a scatter plot of bill_length_mm vs flipper_length_mm, colored by species, to show how these measurements correlate across different penguin species. This is particularly relevant because it can reveal if certain species have distinct morphological characteristics."

Example of a bad suggestion (too generic):
"Create scatter plots for numerical columns to show relationships between variables."

For each suggested visualization:
1. Specify the exact columns to use
2. Explain what specific patterns or relationships to look for
3. Describe why these insights matter for this dataset
4. Include any relevant thresholds or measurements

If the dataset contains time-based columns:
1. Suggest specific time-based visualizations using the exact time column names
2. Consider seasonal patterns, trends, and cycles relevant to this domain
3. Suggest appropriate time-based aggregations for this specific data

Format your response as a list of specific visualization suggestions, each with:
- The exact columns to use
- The type of plot
- What specific insights to look for
- Why these insights matter for this dataset."""

VISUALIZATION_PROMPT = """You are a data visualization expert. Given this dataset:
Rows: {num_rows}
Cols: {num_cols}
Info: {column_info}
Domain: {domain}

Provide a comprehensive set of visualization suggestions that would be insightful for this specific dataset and domain. For each visualization, include:

1. Distribution Analysis
- Histograms for numerical columns
  - Identify data spread and outliers
  - Check for normal distribution
- Box plots for comparing distributions
  - Show quartiles and outliers
  - Compare across categories

2. Relationship Analysis
- Scatter plots for numerical pairs
  - Show correlations
  - Identify clusters
- Bar charts for categorical data
  - Compare frequencies
  - Show proportions

3. Time Series Analysis (if applicable)
- Line plots for trends
  - Show changes over time
  - Identify patterns
- Seasonal decomposition
  - Show cyclical patterns
  - Identify trends

4. Domain-Specific Visualizations
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