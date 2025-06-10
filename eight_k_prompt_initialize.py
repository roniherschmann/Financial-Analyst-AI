# author: Roni Herschmann
# initialization script for a Weave prompt to analyze 8-K filings
# eight_k_prompt_initialize.py

import os
from dotenv import load_dotenv
import weave
from weave import Prompt
from typing import Dict, List, Any

# Load environment variables from .env
load_dotenv()

# Initialize Weave for your W&B project
weave.init("financial-advisor")

class EightKPrompt(Prompt):
    """
    Custom Prompt for 8-K Analysis:
    - client_tpl: User instruction template
    - system_config: Structured system metadata
    - guidelines: Extraction and formatting instructions
    - task: Desired output description
    - output_style: Formatting requirements
    - workflow: Step-by-step pipeline guidance
    """
    client_tpl: str = (
        "Summarize {ticker}'s Form 8-K filed on {filing_date}. "
        "Identify the event category, material details, and investor impact."
    )
    system_config: Dict[str, Any] = {
        "topic": "SEC Filings – 8-K Real-Time Disclosure",
        "subject": "Structured extraction and summarization of {ticker}'s Form 8-K for {filing_date}",
        "persona": "Data-driven equity research associate—detail-oriented, questioning, grounded in value theory",
        "tone": "Institutional, skeptical, data-supported, explanatory, and technical",
        "context": "Equity research and real-time monitoring of corporate events for a hedge fund, asset manager or trader.",
        "knowledge_scope": (
            "Restrict to the specified 8-K filing; do not refer to prior filings unless they directly relate to this event."
        ),
        "called_apis": [
            "SEC EDGAR API",
            "YahooFinance/Bloomberg (for stock price reaction)",
            "internal event-tracking templates"
        ]
    }
    guidelines: List[str] = [
        "1. Identify the Item Number(s) under which the 8-K was filed (e.g., Item 1.01 – Entry into a Material Definitive Agreement).",
        "2. Summarize the material event:  \n"
        "   • Describe the nature of the event (e.g., 'entered a binding purchase agreement,' 'CEO resignation,' 'financial restatement').  \n"
        "   • Provide relevant dates, parties involved, and core terms (e.g., agreement value, termination clauses).",
        "3. Assess Investor Impact:  \n"
        "   • What financial or operational impact is implied?  \n"
        "   • How might this event affect revenue, cash flow, or risk profile?",
        "4. Stock Price Reaction (if available):  \n"
        "   • Pull intraday price/trading-volume data around {filing_date}.  \n"
        "   • Note any abnormal price movement or volume spike.",
        "5. Materiality & Red Flags:  \n"
        "   • Flag if the event suggests potential liquidity stress, covenant breaches, or litigation risk.  \n"
        "   • Highlight any unusual disclosures (e.g., late filings, contradictory language).",
        "6. Related Party or Insider Aspects:  \n"
        "   • If the 8-K involves a related party transaction, name the insider and describe the relationship.  \n"
        "   • Flag any insider trading windows or blackout issues implied.",
        "7. Avoid Boilerplate:  \n"
        "   • Do not copy standard 8-K boilerplate language (e.g., 'forward-looking statements' disclaimers).  \n"
        "   • Focus on new, material facts specific to this filing.",
        "8. User Engagement:  \n"
        "   • At the end, ask: 'Is there a particular section of this 8-K you want to dive deeper into?'",
        "9. Add Brief Context:  \n"
        "   • Mention any related recent events (e.g., prior 8-Ks, 10-K risk factors) only if directly relevant."
    ]
    task: str = (
        "Produce a 1-to-2-page summary (paragraphs + bullet points) covering:  \n"
        "• Item Number(s) & Event Type  \n"
        "• Material Details (dates, parties, terms)  \n"
        "• Investor Impact Analysis  \n"
        "• Stock Price Reaction (brief)  \n"
        "• Materiality & Red Flags  \n"
        "• Related-Party/Insider Considerations  \n"
        "• Concluding Question for Further Exploration"
    )
    output_style: str = (
        "Word-doc style (internal-disclosure report). Use numbered sections and subheadings. "
        "Embed a small table if summarizing multiple related items or comparison to prior similar filings."
    )
    workflow: List[Dict[str, str]] = [
        {"step": 1, "description": "Detect '8-K' request and parse placeholders {ticker} and {filing_date} from user prompt."},
        {"step": 2, "description": "Fetch the specified Form 8-K PDF/text from SEC EDGAR API for ticker: {ticker}, filed on {filing_date}."},
        {"step": 3, "description": "Parse Item Numbers and extract relevant sections (Description of Event, Exhibits)."},
        {"step": 4, "description": "Extract material facts: event type, parties, key financial/contractual terms, effective dates."},
        {"step": 5, "description": "Fetch short-term stock price and volume data around {filing_date} to note abnormal movements."},
        {"step": 6, "description": "Identify red flags: unusual disclosures, potential covenant issues, related-party transactions."},
        {"step": 7, "description": "Assemble the structured summary following the 'task' outline. Use bullet points for clarity."},
        {"step": 8, "description": "Add a concluding question: 'Is there a particular section of this 8-K you want to dive deeper into?'"}
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
eight_k_prompt = EightKPrompt()
weave.publish(eight_k_prompt, name="generic_8k_prompt")

# Example usage
if __name__ == "__main__":
    import os
    from openai import OpenAI

    # Load your LLM API keys via environment
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "Please set OPENAI_API_KEY in your .env"

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Format the prompt for a specific ticker/filing_date
    messages = eight_k_prompt.format(ticker="AAPL", filing_date="2024-12-15")

    # Call the chat completions endpoint
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
    )

    # Extract and print the assistant's response
    print(response.choices[0].message.content)