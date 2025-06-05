# Author: Roni Herschmann
# filing_summary.py
import weave
from openai import OpenAI

# Initialize the OpenAI client (make sure OPENAI_API_KEY is exported)
client = OpenAI()

@weave.op()
def summarize_filing(raw_text: str) -> dict:
    """
    Given the full text of a 10-K or 10-Q, return a JSON object with:
      • kpis: top 3–5 metrics and values
      • segments: a map of segment names → 2-3 sentence summaries
      • tone_summary: 1 paragraph of overall management tone
      • financial_performance_summary: 3-4 sentences on revenue/profit trends
      • peer_metric_recs: instruction for which peer multiples to fetch next
    """
    prompt = f"""
You are a financial analyst LLM. Below is the full text of a 10-K or 10-Q. Return a JSON object with exactly these fields:

1. "kpis": {{ (metric → value) for the top 3–5 quantitative KPIs (e.g., Revenue, EPS, Free Cash Flow) in the most recent period }}
2. "segments": {{ (segment_name → 2–3 sentence summary of that segment’s performance) }}
3. "tone_summary": "1 paragraph summarizing overall management tone (e.g. optimistic, cautious)"
4. "financial_performance_summary": "3–4 sentence summary of revenue/profit/trend changes versus prior period"
5. "peer_metric_recs": "A short instruction on which peers/metrics to fetch next (e.g., 'Fetch EV/EBITDA for MSFT, GOOGL, AMZN')"

Return only valid JSON—no additional commentary.

FILING TEXT:
\"\"\"
{raw_text[:100000]}
\"\"\"
"""
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt}
        ],
        # force the LLM to emit raw JSON
        response_format={ "type": "json_object" }
    )
    # Because of response_format, `.message.content` is already a dict
    return response.choices[0].message.content


if __name__ == "__main__":
    # Initialize Weave under the same W&B project name
    weave.init("AI-Analyst")

    # --- TEST WITH DUMMY FILING TEXT ---
    # Replace this with real 10-K/10-Q text once you have EDGAR retrieval in place.
    dummy_text = """
Apple Inc. posted revenue of $260 billion in fiscal 2022, up 8% year-over-year. 
iPhone segment revenue grew 10% to $183 billion, while Services revenue reached $68 billion. 
Management’s tone remained cautiously optimistic, citing supply chain constraints as a drag on margins. 
Net income was $57 billion, down 3% from the prior year. 
Fetch EV/EBITDA for MSFT, GOOGL, AMZN.
"""

    summary = summarize_filing(dummy_text)
    print(summary)