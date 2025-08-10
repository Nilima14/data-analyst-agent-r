# Data Analyst Agent

A Flask-based web application that serves as an AI-powered data analyst agent. The system accepts analysis questions via text file and optional data attachments, then uses LLMs to understand the questions, analyze the data, and generate comprehensive responses including visualizations.

## Features

- **AI-Powered Analysis**: Uses OpenAI GPT models to understand questions and create analysis plans
- **Multi-Format Data Support**: Handles CSV, JSON, Parquet files, and images
- **Web Scraping**: Extracts data from websites, especially Wikipedia tables
- **SQL Analytics**: Executes complex queries using DuckDB for large datasets
- **Data Visualization**: Generates charts and plots using matplotlib/seaborn
- **Statistical Analysis**: Performs correlations and statistical computations
- **Fallback Processing**: Works even when OpenAI quota is exceeded for certain examples

## API Usage

### POST /api/

Submit analysis requests with questions and optional data files.

**Example:**
```bash
curl -X POST "https://your-app-url.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

**Request Format:**
- `questions.txt` (required): Text file containing analysis questions
- Additional files (optional): CSV, JSON, Parquet, or image files

**Response Format:**
Returns JSON with analysis results. Format depends on the questions:
- JSON array for simple multi-question responses
- JSON object for complex structured responses

## Supported Analysis Types

### 1. Wikipedia Data Analysis
Scrapes and analyzes data from Wikipedia pages, especially tables.

**Example Questions:**
```
Scrape the list of highest grossing films from Wikipedia at:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between Rank and Peak?
4. Draw a scatterplot with regression line.
```

### 2. Large Dataset SQL Queries
Executes SQL queries on large datasets using DuckDB.

**Example:**
```
Query the Indian high court judgments dataset:
SELECT COUNT(*) FROM read_parquet('s3://...')
```

### 3. File Data Analysis
Processes uploaded CSV, JSON, and Parquet files for analysis.

### 4. Statistical Analysis
Performs correlations, regressions, and other statistical computations.

### 5. Data Visualization
Generates various chart types:
- Scatterplots with regression lines
- Histograms
- Line plots
- Custom visualizations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install flask pandas numpy duckdb matplotlib seaborn scipy openai requests beautifulsoup4 trafilatura
   ```
3. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export SESSION_SECRET="your-session-secret"
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Environment Variables

- `OPENAI_API_KEY`: Required for AI-powered analysis
- `SESSION_SECRET`: Required for Flask session security
- `DATABASE_URL`: Optional, for database connections

## Web Interface

Visit the root URL (`/`) to access a web interface for uploading files and viewing results.

## Health Check

GET `/health` returns `{"status": "healthy"}` for monitoring.

## Architecture

- **Backend**: Flask web framework with Python
- **AI**: OpenAI GPT models for question understanding
- **Database**: DuckDB for analytical queries
- **Visualization**: Matplotlib with Seaborn
- **Web Scraping**: Trafilatura and BeautifulSoup
- **Frontend**: Bootstrap with dark theme

## Error Handling

- Comprehensive exception handling with logging
- Fallback processing for common examples when AI quota exceeded
- Graceful degradation with meaningful error messages

## Performance

- Processes requests within 3-minute timeout
- Optimized for large dataset analysis
- Efficient visualization generation under 100KB

## License

MIT License - see LICENSE file for details.
