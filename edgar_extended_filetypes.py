# edgar_extended_filings.py
# Author: Roni Herschmann
# Fetch extended SEC filing types (8-K, Forms 3/4/5, 20-F, DEF 14A) via EDGAR APIs, 
# summarize, and log to W&B in memory.
# ──────────────────────────────────────────────────────────────────────────────

from secedgar.cik_lookup import get_cik_map
from filing_summary import summarize_filing  # Your existing Weave op

import requests
import weave
import wandb
import argparse
import json
import os
import openai
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

# ─────────────────── Initialize Weave & W&B ───────────────────
weave.init("AI-Analyst")

# Verify OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not found")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")

# Initialize OpenAI client for potential direct usage
openai.api_key = os.getenv("OPENAI_API_KEY")

# Extended form types with descriptions
SUPPORTED_FORMS = {
    "8-K": "Real-time disclosure of material events or corporate changes",
    "3": "Initial filing showing insider ownership when someone becomes an officer/director/10%+ holder",
    "4": "Changes in insider ownership (buy/sell/exercise) - filed within 2 business days",
    "5": "Annual summary of insider trades not required to be reported earlier",
    "20-F": "Annual report equivalent to 10-K for non-U.S. companies trading on U.S. exchanges",
    "DEF 14A": "Proxy statement sent to shareholders ahead of votes (board nominees, compensation, proposals)"
}

def get_cik_from_ticker(identifier: str) -> str:
    """
    Convert a ticker or exact company name into a zero-padded CIK string.
    Uses secedgar.get_cik_map (cached internally).
    """
    identifier = identifier.strip().upper()
    cik_map = get_cik_map(user_agent="Your Name your-email@example.com")

    # 1) Check ticker dictionary (e.g. 'MSFT' -> '0000789019')
    if identifier in cik_map["ticker"]:
        return cik_map["ticker"][identifier].zfill(10)
    # 2) Check company-name dictionary (e.g. 'MICROSOFT CORPORATION' -> '0000789019')
    if identifier in cik_map["title"]:
        return cik_map["title"][identifier].zfill(10)
    # 3) If already a 10-digit numeric string, treat as CIK
    if identifier.isdigit() and len(identifier) == 10:
        return identifier.zfill(10)
    raise RuntimeError(f"Could not resolve identifier '{identifier}' to a CIK.")


def get_company_submissions(cik: str) -> dict:
    """
    Get all submissions for a company using SEC's submissions API.
    """
    # Remove leading zeros from CIK for the API
    cik_no_zeros = str(int(cik))
    
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": "Your Name your-email@example.com",
        "Accept": "application/json"
    }
    
    print(f"DEBUG: Fetching submissions from {url}")
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    
    return resp.json()


def find_filing_by_criteria(
    submissions: dict, 
    form_type: str, 
    year: Optional[int] = None,
    days_back: Optional[int] = None,
    limit: int = 1
) -> List[Dict[str, str]]:
    """
    Find filings based on different criteria:
    - For 8-K, Forms 3/4/5: Can search by days_back from today
    - For 20-F, DEF 14A: Usually search by year
    - Returns list of filings with most recent first
    """
    recent_filings = submissions.get("filings", {}).get("recent", {})
    
    form_types = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    
    print(f"DEBUG: Total filings found: {len(form_types)}")
    
    # Calculate date range if days_back is specified
    cutoff_date = None
    if days_back:
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        print(f"DEBUG: Looking for {form_type} filings since {cutoff_date}")
    
    # Find matching filings
    matching_filings = []
    
    for i in range(len(form_types)):
        if form_types[i] == form_type:
            filing_date = filing_dates[i]
            
            # Apply filters
            include_filing = True
            
            if year and not filing_date.startswith(str(year)):
                include_filing = False
            
            if days_back and filing_date < cutoff_date:
                include_filing = False
            
            if include_filing:
                matching_filings.append({
                    "index": i,
                    "filing_date": filing_date,
                    "accession_number": accession_numbers[i],
                    "primary_document": primary_documents[i],
                    "form_type": form_types[i]
                })
                print(f"DEBUG: Found {form_type} filed on {filing_date}")
    
    if not matching_filings:
        # Provide helpful error message with available options
        all_matching_years = set()
        recent_count = 0
        
        for i in range(len(form_types)):
            if form_types[i] == form_type:
                all_matching_years.add(filing_dates[i][:4])
                if filing_dates[i] >= (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"):
                    recent_count += 1
        
        available_years_str = ", ".join(sorted(all_matching_years, reverse=True))
        error_msg = f"No {form_type} found"
        
        if year:
            error_msg += f" for year {year}"
        if days_back:
            error_msg += f" in last {days_back} days"
        
        error_msg += f". Available years: {available_years_str}"
        error_msg += f". Recent filings (last 90 days): {recent_count}"
        
        raise RuntimeError(error_msg)
    
    # Sort by filing date (most recent first) and apply limit
    matching_filings.sort(key=lambda x: x["filing_date"], reverse=True)
    return matching_filings[:limit]


@weave.op()
def fetch_extended_filing_html(
    zero_cik: str,
    form_type: str,
    year: Optional[str] = None,
    days_back: Optional[int] = None,
    filing_index: int = 0
) -> Tuple[str, Dict[str, str]]:
    """
    Fetch HTML for extended filing types with flexible search criteria.
    Returns: (html_content, filing_metadata)
    """
    # Get all company submissions
    submissions = get_company_submissions(zero_cik)
    
    # Find matching filings
    year_int = int(year) if year else None
    matching_filings = find_filing_by_criteria(
        submissions, form_type, year_int, days_back, limit=filing_index + 1
    )
    
    if filing_index >= len(matching_filings):
        raise RuntimeError(f"Requested filing index {filing_index} but only found {len(matching_filings)} matching filings")
    
    selected_filing = matching_filings[filing_index]
    
    # Build URL
    accession_no_clean = selected_filing["accession_number"].replace("-", "")
    cik_no_zeros = str(int(zero_cik))
    
    archives_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_clean}/{selected_filing['primary_document']}"
    
    print(f"✅ Found {form_type} filed on {selected_filing['filing_date']}")
    print(f"DEBUG: Fetching from URL: {archives_url}")
    
    headers = {"User-Agent": "Your Name your-email@example.com"}
    res = requests.get(archives_url, headers=headers)
    res.raise_for_status()

    # Return both content and metadata
    metadata = {
        "filing_date": selected_filing["filing_date"],
        "accession_number": selected_filing["accession_number"],
        "primary_document": selected_filing["primary_document"],
        "form_type": selected_filing["form_type"],
        "url": archives_url
    }

    return res.text, metadata


def list_available_filings(zero_cik: str, ticker: str, form_types: List[str], limit: int = 10):
    """
    List recent filings of specified types for a company.
    """
    submissions = get_company_submissions(zero_cik)
    recent = submissions.get("filings", {}).get("recent", {})
    
    print(f"\nRecent filings for {ticker} (CIK: {zero_cik}):")
    print(f"{'Form':<10} {'Filing Date':<12} {'Description':<60}")
    print("-" * 85)
    
    count = 0
    for i in range(len(recent.get("form", []))):
        form = recent["form"][i]
        if form in form_types:
            filing_date = recent["filingDate"][i]
            description = SUPPORTED_FORMS.get(form, "Unknown form type")
            print(f"{form:<10} {filing_date:<12} {description[:58]}")
            count += 1
            if count >= limit:
                break
    
    if count == 0:
        print("No matching filings found.")
    else:
        print(f"\nShowing {count} most recent filings of requested types.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch extended SEC filing types via EDGAR APIs, summarize, and log to W&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported filing types:
{chr(10).join([f"  {k}: {v}" for k, v in SUPPORTED_FORMS.items()])}

Examples:
  # Get most recent 8-K filing
  python edgar_extended_filings.py --ticker MSFT --form 8-K --days-back 30

  # Get DEF 14A from specific year
  python edgar_extended_filings.py --ticker AAPL --form "DEF 14A" --year 2023

  # List recent filings
  python edgar_extended_filings.py --ticker TSLA --list-filings
        """
    )
    
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="Stock ticker, exact company name, or zero-padded CIK."
    )
    parser.add_argument(
        "--form", choices=list(SUPPORTED_FORMS.keys()), required=False,
        help="Type of filing to fetch."
    )
    parser.add_argument(
        "--year", type=str,
        help="4-digit calendar year (e.g. '2023'). Best for 20-F and DEF 14A."
    )
    parser.add_argument(
        "--days-back", type=int,
        help="Look for filings within this many days from today. Good for 8-K and Forms 3/4/5."
    )
    parser.add_argument(
        "--filing-index", type=int, default=0,
        help="Which filing to select if multiple match (0=most recent, 1=second most recent, etc.)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory where summary JSON will be saved."
    )
    parser.add_argument(
        "--list-filings", action="store_true",
        help="List all available filings for the company and exit."
    )
    parser.add_argument(
        "--project", type=str, default="ai-analyst",
        help="W&B project name."
    )

    args = parser.parse_args()

    # Initialize W&B
    wandb.init(project=args.project, job_type="edgar_extended_filings")

    # Resolve ticker/name to zero-padded CIK
    zero_cik = get_cik_from_ticker(args.ticker)
    print(f"Resolved identifier '{args.ticker}' → CIK {zero_cik}")

    # List filings option
    if args.list_filings:
        list_available_filings(zero_cik, args.ticker, list(SUPPORTED_FORMS.keys()))
        return

    # Validate required arguments
    if not args.form:
        parser.error("--form is required when not using --list-filings")

    if not args.year and not args.days_back:
        print("Warning: Neither --year nor --days-back specified. Will fetch most recent filing.")
        # Default to looking back 365 days for most forms, 730 for annual forms
        if args.form in ["20-F", "DEF 14A"]:
            args.days_back = 730
        else:
            args.days_back = 365

    # Fetch filing
    try:
        raw_html, metadata = fetch_extended_filing_html(
            zero_cik, args.form, args.year, args.days_back, args.filing_index
        )
        print(f"✅ Retrieved {args.form} filed on {metadata['filing_date']}, length={len(raw_html)} chars")
    except Exception as e:
        print(f"❌ Error fetching filing: {e}")
        return

    # Create unique identifier for this filing
    filing_id = f"{zero_cik}_{args.form.replace(' ', '_')}_{metadata['filing_date'].replace('-', '')}"

    # Log raw HTML as W&B Artifact
    raw_artifact = wandb.Artifact(
        name=f"{filing_id}_raw_html", 
        type="raw_filing",
        metadata=metadata
    )
    temp_folder = Path("./temp_filings")
    temp_folder.mkdir(exist_ok=True)
    temp_path = temp_folder / f"{filing_id}.htm"
    temp_path.write_text(raw_html, encoding="utf-8", errors="ignore")
    raw_artifact.add_file(str(temp_path))
    wandb.log_artifact(raw_artifact)
    temp_path.unlink()
    print(f"✅ Logged raw HTML as Artifact: {filing_id}_raw_html")

    # Summarize via existing Weave op
    print("🤖 Generating summary using AI...")
    summary = summarize_filing(raw_html)
    print("✅ Summary generated")

    # Create comprehensive summary document
    summary_doc = {
        "ticker_or_name": args.ticker,
        "cik": zero_cik,
        "form_type": args.form,
        "form_description": SUPPORTED_FORMS[args.form],
        "filing_metadata": metadata,
        "search_criteria": {
            "year": args.year,
            "days_back": args.days_back,
            "filing_index": args.filing_index
        },
        "summary": summary,
        "generated_at": datetime.now().isoformat()
    }

    # Save summary JSON
    out_folder = Path(args.output_dir)
    out_folder.mkdir(exist_ok=True)
    summary_path = out_folder / f"{filing_id}_summary.json"
    
    with open(summary_path, "w") as f:
        json.dump(summary_doc, f, indent=2)

    # Log summary as W&B Artifact
    summary_artifact = wandb.Artifact(
        name=f"{filing_id}_summary",
        type="summary",
        metadata={
            "form_type": args.form,
            "filing_date": metadata["filing_date"],
            "company_ticker": args.ticker
        }
    )
    summary_artifact.add_file(str(summary_path))
    wandb.log_artifact(summary_artifact)
    
    # Log key metrics to W&B
    wandb.log({
        "form_type": args.form,
        "filing_date": metadata["filing_date"],
        "html_length": len(raw_html),
        "company_cik": zero_cik,
        "filing_year": int(metadata["filing_date"][:4])
    })

    print(f"✅ Logged summary JSON as Artifact: {filing_id}_summary")
    print(f"✅ Summary saved to: {summary_path}")
    
    # Display key information
    print(f"\n📊 Filing Summary:")
    print(f"   Company: {args.ticker} (CIK: {zero_cik})")
    print(f"   Form: {args.form} - {SUPPORTED_FORMS[args.form]}")
    print(f"   Filed: {metadata['filing_date']}")
    print(f"   Document: {metadata['primary_document']}")


if __name__ == "__main__":
    main()