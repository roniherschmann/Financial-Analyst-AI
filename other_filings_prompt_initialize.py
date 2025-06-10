# author: Roni Herschmann
# initialization script for a Weave prompt to analyze other SEC filings
# other_filings_prompt_initialize.py

import os
from dotenv import load_dotenv
import weave
from weave import Prompt
from typing import Dict, List, Any

# Load environment variables from .env
load_dotenv()

# Initialize Weave for your W&B project
weave.init("financial-advisor")

class OtherFilingsPrompt(Prompt):
    """
    Custom Prompt for Other SEC Filings Analysis (General Catch-All):
    - client_tpl: User instruction template
    - system_config: Structured system metadata
    - guidelines: Extraction and formatting instructions
    - task: Desired output description
    - output_style: Formatting requirements
    - workflow: Step-by-step pipeline guidance
    """
    client_tpl: str = (
        "Summarize {ticker}'s {form_type} filed on {filing_date}. "
        "Highlight material facts, key disclosures, and investor implications."
    )
    system_config: Dict[str, Any] = {
        "topic": "SEC Filings – General Catch-All",
        "subject": "Structured summarization of {ticker}'s {form_type}",
        "persona": "Data-driven equity research associate—detail-oriented, questioning, grounded in value theory",
        "tone": "Institutional, skeptical, data-supported, and technical",
        "context": "Equity research and compliance monitoring for a hedge fund or asset manager.",
        "knowledge_scope": (
            "Restrict to the specified filing only; do not reference other filings unless directly necessary for clarity."
        ),
        "called_apis": [
            "SEC EDGAR API",
            "YahooFinance/Bloomberg (for market reaction)",
            "internal tracking templates"
        ]
    }
    guidelines: List[str] = [
        "1. Identify Filing Purpose & Sections:  \n"
        "   • Determine why the form was filed (e.g., S-3: secondary offering; 8-K: corporate event; S-1: IPO registration).  \n"
        "   • List main sections or exhibits relevant to this form.",
        "2. Extract Material Facts:  \n"
        "   • Summarize the primary disclosures (e.g., offering terms, material agreements, legal notices).  \n"
        "   • Provide dates, parties involved, and core terms where applicable.",
        "3. Investor Implications:  \n"
        "   • Comment on how the disclosure could affect financial metrics or risk profile.  \n"
        "   • Note any anticipated market reaction or regulatory concerns.",
        "4. Red Flags & Unusual Items:  \n"
        "   • Flag any late or amended filings, unusual disclaimers, or ambiguous language.  \n"
        "   • Identify any related-party or conflict-of-interest disclosures.",
        "5. Context & Cross-References (Minimal):  \n"
        "   • If needed, briefly reference a related 10-K/10-Q/8-K for context—only if material to this form.  \n"
        "   • Otherwise, avoid extraneous cross-referencing.",
        "6. Avoid Boilerplate:  \n"
        "   • Do not copy standard legal disclaimers or forward-looking statements.  \n"
        "   • Focus on what is new or changed compared to prior filings.",
        "7. User Engagement:  \n"
        "   • At the end, ask: 'Is there a specific detail or section of this filing you want to discuss further?'"
    ]
    task: str = (
        "Produce a 1-to-3-page summary (paragraphs + bullet points) covering:  \n"
        "• Filing Purpose & Main Sections  \n"
        "• Material Facts & Key Terms  \n"
        "• Investor Implications  \n"
        "• Red Flags & Unusual Items  \n"
        "• Minimal Context or Cross-References if Necessary  \n"
        "• Concluding Question for Further Exploration"
    )
    output_style: str = (
        "Word-doc style (internal-report format). Use numbered sections and subheadings. "
        "Embed a small table if summarizing multiple exhibits or key data points."
    )
    workflow: List[Dict[str, str]] = [
        {"step": 1, "description": "Detect request and parse placeholders {ticker}, {form_type}, {filing_date}."},
        {"step": 2, "description": "Fetch the specified filing PDF/text from SEC EDGAR API for {ticker}, {form_type}, {filing_date}."},
        {"step": 3, "description": "Parse main sections and extract material disclosures (e.g., offering prospectus, agreements, risk disclosures)."},
        {"step": 4, "description": "Extract core facts: dates, parties, terms, figures—whatever is central to this form's purpose."},
        {"step": 5, "description": "Assess investor impact: financial, operational, or governance implications."},
        {"step": 6, "description": "Identify red flags: late/amended filings, confusing language, related-party issues."},
        {"step": 7, "description": "Assemble the structured summary following the 'task' outline. Use bullet points for clarity."},
        {"step": 8, "description": "Add a concluding question: 'Is there a specific detail or section of this filing you want to discuss further?'"}
    ]

    def format(self, **kwargs) -> List[Dict[str, str]]:
        # Fill client prompt
        client = self.client_tpl.format(**kwargs)
        # Serialize system_config
        sys_parts = []
        for key, val in self.system_config.items():
            if isinstance(val, str):
                sys_parts.append(f"{key}: {val.format(**kwargs)}")
            else:
                sys_parts.append(f"{key}: {', '.join(val)}")
        full_system = "\n".join(sys_parts + self.guidelines + [self.task, self.output_style])
        return [
            {"role": "system", "content": full_system},
            {"role": "user",   "content": client}
        ]

# Instantiate and publish
other_filings_prompt = OtherFilingsPrompt()
weave.publish(other_filings_prompt, name="generic_other_filings_prompt")

# Example usage
if __name__ == "__main__":
    import os
    from openai import OpenAI

    # Load your LLM API keys via environment
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in your .env"

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Format the prompt for a specific ticker/form_type/filing_date
    messages = other_filings_prompt.format(
        ticker="AAPL", 
        form_type="S-3", 
        filing_date="2024-06-15"
    )

    # Call the chat completions endpoint
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    # Extract and print the assistant's response
    print(response.choices[0].message.content)