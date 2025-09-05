# Your Data Wrestler 🤼

An intelligent Streamlit application that transforms raw data into actionable insights through AI-powered analysis, automated cleaning, and dynamic visualizations.

🔗 [https://your-data-wrestler.streamlit.app/](https://your-data-wrestler.streamlit.app/)

## Features

### 📊 **Data Analysis & Exploration**
- Upload and analyze multiple file formats (CSV, TXT, XLSX, XLS)
- Interactive data overview with comprehensive metadata
- Natural language querying - ask questions about your data in plain English
- Domain-specific analysis with automatic context detection
- Real-time data preview with customizable display options

### 🧹 **Intelligent Data Cleaning**
- AI-powered cleaning suggestions based on data patterns
- Automated detection of data quality issues
- One-click application of cleaning operations:
  - Remove columns with high missing values (>50%)
  - Fill missing values using statistical methods (mean/mode)
  - Remove duplicate rows and empty entries
  - Standardize text data (lowercase, whitespace trimming)
  - Convert string numbers to proper numeric format
  - Remove statistical outliers
  - Standardize date formats
  - Fix incorrect data types
- Download cleaned datasets

### 📈 **Advanced Visualizations**
- **Automated Visualization Suggestions** - AI recommends optimal chart types
- **Custom Visualization Builder** - Create charts with full control
- **Multiple Chart Types**:
  - Distribution plots (histogram, density, violin)
  - Relationship plots (scatter, line, area)
  - Categorical analysis (bar, pie, stacked bar)
  - Statistical plots (box plot, heatmap)
- **Export Options** - Download as HTML or PNG
- **Interactive Charts** - Powered by Plotly for rich interactivity

### 🎯 **Smart Features**
- **Domain Detection** - Automatically identifies data context (healthcare, e-commerce, etc.)
- **Editable Domain Context** - Customize analysis focus
- **Session Management** - Maintains state across interactions
- **Error Handling** - Graceful handling of data issues
- **Responsive Design** - Optimized for different screen sizes

## Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for AI-powered analysis)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-data-wrestler.git
   cd your-data-wrestler
   ```

2. **Create and activate a virtual environment**
   
   **Using uv (recommended):**
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```
   
   **Using venv (alternative):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   touch .env  # On Unix/macOS
   # or
   type nul > .env  # On Windows
   ```
   
   Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Quick Start
```bash
streamlit run main.py
```

The app will be available at `http://localhost:8501`

## Usage

### Getting Started

1. **Launch the application**
   ```bash
   streamlit run main.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your data** - Supported formats:
   - CSV files (`.csv`)
   - Excel files (`.xlsx`, `.xls`)
   - Text files (`.txt`) with delimited data

### Workflow

#### 1. **Data Upload & Overview**
- Upload your dataset using the file uploader
- View automatic data overview with:
  - Row and column counts
  - Data types and missing value percentages
  - Interactive data preview

#### 2. **Domain Context**
- The app automatically detects your data's domain (healthcare, e-commerce, etc.)
- Edit the domain context if needed for more relevant analysis

#### 3. **Natural Language Analysis**
- Ask questions about your data in plain English
- Get AI-powered insights and explanations
- Examples:
  - "What are the top 5 products by sales?"
  - "Show me the correlation between age and income"
  - "What patterns do you see in the missing data?"

#### 4. **Data Cleaning**
- Review AI-suggested cleaning operations
- Select which cleaning steps to apply
- Download the cleaned dataset

#### 5. **Visualization**
- Choose from AI-recommended visualizations
- Create custom charts with full control
- Download visualizations as HTML or PNG

### Tips for Best Results

- **File Size**: Works best with datasets under 100MB
- **Data Quality**: Cleaner data produces better AI insights
- **Questions**: Be specific in your natural language queries
- **Domain Context**: Set the correct domain for more relevant suggestions

## Technical Details

### Dependencies
- **Streamlit** (1.32.0) - Web application framework
- **Pandas** (2.2.1) - Data manipulation and analysis
- **OpenAI** (1.83.0) - AI-powered analysis and insights
- **Plotly** (5.19.0) - Interactive visualizations
- **Python-dotenv** (1.0.1) - Environment variable management
- **Kaleido** (0.2.1) - Static image export for Plotly
- **Openpyxl** (3.1.2) - Excel file support

### Supported File Formats
- **CSV** - Comma-separated values
- **TXT** - Text files with delimited data (auto-detects separator)
- **XLSX** - Excel 2007+ format
- **XLS** - Legacy Excel format

### Visualization Types
- **Distribution**: Histogram, Density, Violin plots
- **Relationships**: Scatter, Line, Area charts
- **Categorical**: Bar, Pie, Stacked Bar charts
- **Statistical**: Box plots, Heatmaps

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 100MB free space for dependencies
- **Internet**: Required for OpenAI API calls

## API Usage

This application uses OpenAI's GPT-3.5 API for:
- Natural language data analysis
- Data cleaning suggestions
- Visualization recommendations
- Domain context detection

**Important Notes:**
- Requires a valid OpenAI API key
- API usage is charged per request
- Ensure sufficient credits in your OpenAI account
- Consider data privacy when uploading sensitive information

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-username/your-data-wrestler/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**Made with ❤️ for data enthusiasts who want to wrestle their data into submission!** 🤼‍♂️ 
