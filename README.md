# Data Analysis Assistant

An interactive Streamlit app that allows users to analyze CSV files using natural language and AI-powered insights.

## Features

- Upload and analyze CSV files
- View basic dataset metadata
- Ask questions about your data using natural language
- Get AI-powered data cleaning suggestions
- Generate visualizations using Plotly
- Domain-specific analysis (generic, healthcare, e-commerce)

## Setup

1. Clone this repository
2. Create a virtual environment using uv:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```

5. Add your OpenAI API key to the `.env` file

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a CSV file and start exploring your data!

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in requirements.txt

## Note

This app uses OpenAI's GPT-3.5 API for natural language processing and analysis. Make sure you have a valid API key and sufficient credits in your OpenAI account. 