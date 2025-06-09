# EDGAR 10-K/10-Q Fetcher

Fetch and analyze 10-K and 10-Q filings via EDGAR APIs with AI summarization and W&B logging.

## Quick Start

### Setup
```bash
export OPENAI_API_KEY="your-openai-key"
export WANDB_API_KEY="your-wandb-key"
pip install secedgar weave wandb openai requests
```

### Usage Examples
```bash
# Get 10-K annual report
python edgar_fetch_and_summarize.py --ticker MSFT --form 10-K --year 2023

# Get 10-Q quarterly report
python edgar_fetch_and_summarize.py --ticker AAPL --form 10-Q --year 2023

# List available filings
python edgar_fetch_and_summarize.py --ticker TSLA --form 10-K --year 2023 --list-filings
```

## Supported Filing Types

- **10-K**: Annual comprehensive business report
- **10-Q**: Quarterly financial report

## Key Features

- **Year-based Search**: Fetch filings from specific years
- **AI Summarization**: Uses `filing_summary.summarize_filing()` Weave operation with OpenAI integration
- **W&B Integration**: Automatic logging of raw filings and summaries as artifacts
- **CIK Resolution**: Supports ticker symbols, company names, or direct CIK input
- **In-memory Processing**: Efficient HTML retrieval and processing

## Arguments

- `--ticker`: Stock ticker (e.g., 'MSFT'), company name, or CIK
- `--form`: Filing type ('10-K' or '10-Q')
- `--year`: 4-digit year (e.g., '2023')
- `--output-dir`: Output directory (default: './outputs')
- `--list-filings`: List available filings and exit

## Output

- Raw HTML files logged as W&B artifacts
- AI-generated summaries saved as JSON with metadata
- Comprehensive filing metadata including dates and URLs

## Dependencies

Requires `filing_summary.py` with `summarize_filing()` Weave operation for OpenAI-powered AI processing.