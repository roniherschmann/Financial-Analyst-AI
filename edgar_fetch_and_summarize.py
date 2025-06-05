# edgar_fetch_and_summarize.py
# Author: Roni Herschmann
# Find any company's 10-K/10-Q via EDGAR APIs, summarize, and log to W&B in memory.
# ──────────────────────────────────────────────────────────────────────────────

from secedgar.cik_lookup import get_cik_map
from filing_summary import summarize_filing  # Your existing Weave op

import requests
import weave
import wandb
import argparse
import json
from pathlib import Path
from datetime import datetime

# ─────────────────── Initialize Weave & W&B ───────────────────
weave.init("AI-Analyst")
wandb.init(project="ai-analyst", job_type="edgar_fetch_and_summarize")


def get_cik_from_ticker(identifier: str) -> str:
    """
    Convert a ticker or exact company name into a zero-padded CIK string.
    Uses secedgar.get_cik_map (cached internally).
    """
    identifier = identifier.strip().upper()
    cik_map = get_cik_map(user_agent="Your Name your-email@example.com")

    # 1) Check ticker dictionary (e.g. 'MSFT' -> '0000789019')
    if identifier in cik_map["ticker"]:
        return cik_map["ticker"][identifier].zfill(10)  # Ensure zero-padded CIK
    # 2) Check company-name dictionary (e.g. 'MICROSOFT CORPORATION' -> '0000789019')
    if identifier in cik_map["title"]:
        return cik_map["title"][identifier].zfill(10)  # Ensure zero-padded CIK
    # 3) If already a 10-digit numeric string, treat as CIK
    if identifier.isdigit() and len(identifier) == 10:
        return identifier.zfill(10)  # Ensure zero-padded CIK
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


def find_filing_in_year(submissions: dict, form_type: str, year: int) -> tuple[str, str, str]:
    """
    Find the most recent filing of the specified type in the given year.
    Returns: (accession_number, primary_document, filing_date)
    """
    recent_filings = submissions.get("filings", {}).get("recent", {})
    
    form_types = recent_filings.get("form", [])
    filing_dates = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    
    print(f"DEBUG: Total filings found: {len(form_types)}")
    
    # Find all filings of the requested type in the specified year
    matching_filings = []
    
    for i in range(len(form_types)):
        if form_types[i] == form_type and filing_dates[i].startswith(str(year)):
            matching_filings.append({
                "index": i,
                "filing_date": filing_dates[i],
                "accession_number": accession_numbers[i],
                "primary_document": primary_documents[i]
            })
            print(f"DEBUG: Found {form_type} filed on {filing_dates[i]}")
    
    if not matching_filings:
        # List available years for this form type
        available_years = set()
        for i in range(len(form_types)):
            if form_types[i] == form_type:
                available_years.add(filing_dates[i][:4])
        
        available_years_str = ", ".join(sorted(available_years, reverse=True))
        raise RuntimeError(
            f"No {form_type} found for year {year}. "
            f"Available years for {form_type}: {available_years_str}"
        )
    
    # Sort by filing date and get the most recent
    matching_filings.sort(key=lambda x: x["filing_date"], reverse=True)
    latest = matching_filings[0]
    
    print(f"DEBUG: Selected filing from {latest['filing_date']}")
    
    return (
        latest["accession_number"],
        latest["primary_document"],
        latest["filing_date"]
    )


@weave.op()
def fetch_filing_html(
    zero_cik: str,
    form_type: str,
    year: str
) -> str:
    """
    1) Get company submissions to find the right filing
    2) Download raw HTML from EDGAR Archives directly into memory.
    Returns the HTML string.
    """
    # Get all company submissions
    submissions = get_company_submissions(zero_cik)
    
    # Find the specific filing
    accession_number, primary_document, filing_date = find_filing_in_year(
        submissions, form_type, int(year)
    )
    
    # Remove dashes from accession number for URL
    accession_no_clean = accession_number.replace("-", "")
    
    # Remove leading zeros from CIK for the URL
    cik_no_zeros = str(int(zero_cik))
    
    archives_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_clean}/{primary_document}"
    print(f"✅ Found {form_type} filed on {filing_date}")
    print(f"DEBUG: Fetching from URL: {archives_url}")
    
    headers = {"User-Agent": "Your Name your-email@example.com"}
    res = requests.get(archives_url, headers=headers)
    res.raise_for_status()

    return res.text  # Full HTML in memory


def main():
    parser = argparse.ArgumentParser(
        description="Fetch any company's 10-K/10-Q via EDGAR APIs, summarize, and log to W&B in memory."
    )
    parser.add_argument(
        "--ticker", type=str, required=True,
        help="Stock ticker (e.g. 'MSFT') or exact company name (e.g. 'MICROSOFT CORPORATION') or zero-padded CIK."
    )
    parser.add_argument(
        "--form", choices=["10-K", "10-Q"], required=True,
        help="Type of filing: '10-K' or '10-Q'."
    )
    parser.add_argument(
        "--year", type=str, required=True,
        help="4-digit calendar year (e.g. '2023')."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Directory where summary JSON will be saved."
    )
    parser.add_argument(
        "--list-filings", action="store_true",
        help="List all available filings for the company and exit."
    )
    args = parser.parse_args()

    # 1) Resolve ticker/name to zero-padded CIK
    zero_cik = get_cik_from_ticker(args.ticker)
    print(f"Resolved identifier '{args.ticker}' → CIK {zero_cik}")

    # Optional: List all filings
    if args.list_filings:
        submissions = get_company_submissions(zero_cik)
        recent = submissions.get("filings", {}).get("recent", {})
        
        print(f"\nRecent filings for {args.ticker}:")
        print(f"{'Form Type':<10} {'Filing Date':<12} {'Period Date':<12}")
        print("-" * 40)
        
        for i in range(min(20, len(recent.get("form", [])))):
            form = recent["form"][i]
            if form in ["10-K", "10-Q"]:
                filing_date = recent["filingDate"][i]
                period_date = recent.get("reportDate", ["N/A"])[i]
                print(f"{form:<10} {filing_date:<12} {period_date:<12}")
        return

    # 2) Fetch raw HTML in memory via Weave
    raw_html = fetch_filing_html(zero_cik, args.form, args.year)
    print(f"✅ Retrieved raw {args.form} for CIK={zero_cik}, year={args.year}, length={len(raw_html)} chars")

    # 3) Log raw HTML as W&B Artifact (temporary file only for upload)
    artifact_name = f"{zero_cik}_{args.form}_{args.year}_raw_html"
    raw_artifact = wandb.Artifact(name=artifact_name, type="raw_filing")
    temp_folder = Path("./temp_filings")
    temp_folder.mkdir(exist_ok=True)
    temp_path = temp_folder / f"{zero_cik}_{args.form}_{args.year}.htm"
    temp_path.write_text(raw_html, encoding="utf-8", errors="ignore")
    raw_artifact.add_file(str(temp_path))
    wandb.log_artifact(raw_artifact)
    temp_path.unlink()
    print(f"✅ Logged raw HTML as Artifact: {artifact_name}")

    # 4) Summarize via your existing Weave op
    summary = summarize_filing(raw_html)
    print("DEBUG: summary type:", type(summary))

    # 5) Save & log summary JSON as W&B Artifact
    summary_doc = {
        "ticker_or_name": args.ticker,
        "cik": zero_cik,
        "form": args.form,
        "year": args.year,
        "summary": summary
    }
    out_folder = Path(args.output_dir)
    out_folder.mkdir(exist_ok=True)
    summary_path = out_folder / f"{zero_cik}_{args.form}_{args.year}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_doc, f, indent=2)

    summary_art = wandb.Artifact(
        name=f"{zero_cik}_{args.form}_{args.year}_summary",
        type="summary"
    )
    summary_art.add_file(str(summary_path))
    wandb.log_artifact(summary_art)
    print(f"✅ Logged summary JSON as Artifact: {zero_cik}_{args.form}_{args.year}_summary")


if __name__ == "__main__":
    main()