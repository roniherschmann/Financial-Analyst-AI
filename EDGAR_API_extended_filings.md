# EDGAR Extended Filings Fetcher

Fetch and analyze extended SEC filing types (8-K, Forms 3/4/5, 20-F, DEF 14A) via EDGAR APIs with AI summarization and W&B logging.

## Supported Filing Types

- **8-K**: Real-time material events disclosure
- **Form 3**: Initial insider ownership filing  
- **Form 4**: Insider ownership changes (filed within 2 days)
- **Form 5**: Annual insider trade summary
- **20-F**: Annual report for non-U.S. companies
- **DEF 14A**: Proxy statements for shareholder votes

## Quick Start

### Setup
```bash
export OPENAI_API_KEY="your-openai-key"
export WANDB_API_KEY="your-wandb-key"
pip install secedgar weave wandb openai requests
```

# Get recent 8-K filing
python edgar_extended_filetypes.py --ticker MSFT --form 8-K --days-back 30

# Get proxy statement from specific year
python edgar_extended_filetypes.py --ticker AAPL --form "DEF 14A" --year 2023

# List available filings
python edgar_extended_filetypes.py --ticker TSLA --list-filings

# Get second most recent Form 4
python edgar_extended_filetypes.py --ticker NVDA --form 4 --days-back 90 --filing-index 1

## Key Features

- **Flexible Search**: By year or days back from today
- **AI Summarization**: Uses existing `filing_summary.summarize_filing()` Weave operation
- **W&B Integration**: Automatic logging of raw filings and summaries as artifacts
- **Smart Defaults**: Appropriate lookback periods for different filing types
- **Rich Metadata**: Captures filing dates, URLs, and document details

## Dependencies

Requires `filing_summary.py` with `summarize_filing()` Weave operation for AI processing.

## Output

- Raw HTML files logged as W&B artifacts
- AI-generated summaries saved as JSON with metadata
- Comprehensive logging for agent performance tracking