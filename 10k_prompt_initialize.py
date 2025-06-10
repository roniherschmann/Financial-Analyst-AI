import os
from dotenv import load_dotenv
import weave
from weave import Prompt
from typing import Dict, List, Any

# Load environment variables from .env
load_dotenv()

# Initialize Weave for your W&B project
weave.init("financial-advisor")

class TenKPrompt(Prompt):
    """
    Custom Prompt for 10-K Analysis with structured fields:
    - client_tpl: User instruction template
    - system_config: Structured system metadata
    - guidelines: Extraction and formatting instructions
    - task: Desired output description
    - output_style: Formatting requirements
    - workflow: Step-by-step pipeline guidance
    """
    client_tpl: str = (
        "Create a report on {ticker}’s {year} 10-K. "
        "Highlight business overview, segment profitability, growth metrics, debt/financial health, "
        "cash flow metrics, management tone, stock option disclosures (for younger firms), dividends (for mature firms), R&D spend, risk factors, and any legal or governance issues."
    )
    system_config: Dict[str, Any] = {
        "topic": "SEC Filings – 10-K Analysis",
        "subject": "Full-cycle parsing and structured summarization of {ticker}’s {year} Form 10-K",
        "persona": "Data-driven equity research associate—detail-oriented, questioning, grounded in value theory",
        "tone": "Institutional, skeptical, data-supported, and technical",
        "context": "Equity research report and portfolio due diligence for an asset manager or hedge fund team.",
        "knowledge_scope": (
            "Focus on the {year} 10-K; lightly cross-reference with consensus estimates, past relevant data of {ticker} and market data from that fiscal year."
        ),
        "called_apis": [
            "SEC EDGAR API",
            "FRED (for macro context)",
            "YahooFinance/Bloomberg (for market data & peer comparables)",
            "internal financial templates"
        ]
    }
    guidelines: List[str] = [
        "1. Begin with a concise business overview:  \n"
        "   • Describe the company’s core operations, key products/services, and competitive positioning.  \n"
        "   • Note any material corporate changes over the year (acquisitions, divestitures, reorganizations).",
        "2. Segment Profitability & Growth:  \n"
        "   • Break out annual revenue, gross margin, operating margin, and net margin by segment.  \n"
        "   • Provide a table showing segment revenue and margin contributions vs. prior year.  \n"
        "   • Call out fast-growing or declining segments and question any inconsistencies.",
        "3. Income Statement Analysis:  \n"
        "   • Highlight YoY changes in total revenue, COGS, SG&A, R&D, and net income.  \n"
        "   • Flag significant one-time items (impairments, write-downs, restructuring charges).  \n"
        "   • Identify any non-recurring gains or losses.",
        "4. Debt & Financial Health:  \n"
        "   • Summarize total debt, maturity schedule, interest rates, and leverage ratios (e.g., Debt/EBITDA).  \n"
        "   • Note covenant changes, new debt issuances, or refinancing.  \n"
        "   • Call out any red flags (e.g., large covenant waivers, rising interest expense).",
        "5. Cash Flow & Liquidity Metrics:  \n"
        "   • Provide operating cash flow, free cash flow, and capital expenditures.  \n"
        "   • Discuss working capital changes (DSO, DPO, inventory days).  \n"
        "   • Identify unusual cash items (e.g., litigation settlements).",
        "6. R&D Spend & Capex:  \n"
        "   • Quantify annual R&D expenses and compare to prior year.  \n"
        "   • Note any commentary on R&D pipeline or capitalization policies.  \n"
        "   • Summarize capital expenditure plans and depreciation policy changes.",
        "7. Management Tone & Forward Outlook:  \n"
        "   • Capture MD&A tone: optimistic, cautious, defensive—include direct quotes when available.  \n"
        "   • Extract any forward-looking guidance or qualitative outlook statements.  \n"
        "   • If guidance was withdrawn or modified, note the rationale.",
        "8. Stock Options, Dividends & Shareholder Returns:  \n"
        "   • For younger firms: detail stock option grants, dilution effects, vesting modifications.  \n"
        "   • For mature firms: summarize declared dividends, payout ratio, share buybacks.  \n"
        "   • Identify any changes in capital return policy.",
        "9. Risk Factors & Legal/Governance Issues:  \n"
        "   • Extract top 5 risk factors; prioritize those that could materially impact future performance.  \n"
        "   • Summarize any ongoing litigation, regulatory investigations, or contingent liabilities.  \n"
        "   • Note changes in board composition, auditor fees, or corporate governance disclosures.",
        "10. Footnotes & Off-Balance-Sheet Items:  \n"
        "    • Highlight any off-balance-sheet liabilities (leases, guarantees).  \n"
        "    • Call out related-party transactions or unusual accounting policies.",
        "11. Key Ratios & Peer Comparison:  \n"
        "    • Compute and compare key ratios (ROIC, ROE, current ratio, debt/EBITDA) vs. peer averages.  \n"
        "    • Label any outliers or anomalies.",
        "12. Avoid Boilerplate:  \n"
        "    • Do not restate generic MD&A paragraphs verbatim.  \n"
        "    • Avoid spending excessive time on line items unchanged from prior year.",
        "13. User Engagement:  \n"
        "    • At the end, ask: “Is there a specific section or metric from this 10-K you want to explore further?”",
        "14. Add Macro/Industry Context:  \n"
        "    • Mention relevant macro drivers (interest rate changes, commodity prices, regulatory shifts) that affected the company over the fiscal year."
    ]
    task: str = (
        "Produce a 3-to-6-page summary (paragraphs + bullet points + tables) covering:  \n"
        "• Executive Business Overview  \n"
        "• Segment Profitability & Growth (with comparison table)  \n"
        "• Income Statement Highlights  \n"
        "• Debt & Financial Health Analysis  \n"
        "• Cash Flow & Liquidity  \n"
        "• R&D Spend & CapEx  \n"
        "• Management Tone & Forward Outlook  \n"
        "• Stock Options / Dividends / Buybacks  \n"
        "• Risk Factors & Legal/Governance Notes  \n"
        "• Footnotes & Off-Balance-Sheet Items  \n"
        "• Key Ratios & Peer Comparison  \n"
        "• Macro/Industry Context  \n"
        "• Concluding Question for Further Exploration"
    )
    output_style: str = (
        "Word-doc style (internal-report format). Use numbered sections and subheadings.  \n"
        "Embed tables for:  \n"
        "  - Segment revenue & margin comparison (current vs. prior year),  \n"
        "  - Debt maturity schedule summary,  \n"
        "  - Key ratio comparison vs. peers.  \n"
        "If requested by user, adapt to PPT-style headings and bullet charts."
    )
    workflow: List[Dict[str, str]] = [
        {"step": 1, "description": "Detect '10-K' request and parse placeholders {ticker} and {year} from user prompt."},
        {"step": 2, "description": "Fetch the specified 10-K PDF/text from SEC EDGAR API for ticker: {ticker}, year: {year}."},
        {"step": 3, "description": "Parse Business Overview, MD&A, Financial Statements, Footnotes, Risk Factors, Legal & Governance sections."},
        {"step": 4, "description": "Extract KPIs: segment revenues, gross margin, operating margin, net margin, debt schedules, interest expense, cash flow line items, R&D, stock option tables, dividend/buyback figures, and governance disclosures."},
        {"step": 5, "description": "Compute YoY deltas for key metrics. Compare to consensus estimates and peer averages if available."},
        {"step": 6, "description": "Identify red flags: unusual one-time charges, litigation reserves, covenant breaches, off-balance-sheet liabilities."},
        {"step": 7, "description": "Capture direct quotes from MD&A on strategy, outlook, and risk; annotate tone (optimistic, cautious, etc.)."},
        {"step": 8, "description": "Assemble the structured summary following the ‘task’ outline. Insert tables and bullet points."},
        {"step": 9, "description": "Add a concluding question: “Is there a specific section or metric from this 10-K you want to explore further?”"}
    ]
    def format(self, **kwargs) -> List[Dict[str, str]]:
        # Fill client prompt
        client = self.client_tpl.format(**kwargs)
        # Serialize system_config for the system message
        sys_parts = []
        for key, val in self.system_config.items():
            if isinstance(val, str):
                sys_parts.append(f"{key}: {val.format(**kwargs)}")
            else:
                sys_parts.append(f"{key}: {', '.join(val)}")
        # Combine all sections into one system message
        full_system = "\n".join(sys_parts + self.guidelines + [self.task, self.output_style])
        # Return messages list
        return [
            {"role": "system", "content": full_system},
            {"role": "user",   "content": client}
        ]

# Instantiate and publish
tenk_prompt = TenKPrompt()
weave.publish(tenk_prompt, name="generic_10k_prompt")

# Example usage
if __name__ == "__main__":
    import os
    from openai import OpenAI

    # Load your LLM API keys via environment
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in your .env"

    # Initialize the OpenAI client (v1.x style)
    client = OpenAI(api_key=api_key)

    # Format the prompt for a specific ticker/year
    messages = tenk_prompt.format(ticker="AAPL", year=2024)

    # Call the new chat completions endpoint
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    # Extract and print the assistant’s response
    print(response.choices[0].message.content)