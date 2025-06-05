# filing_summary.py
import weave
from openai import OpenAI
import json                   # ← add

client = OpenAI()

@weave.op()
def summarize_filing(raw_text: str) -> dict:
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
        response_format={ "type": "json_object" }
    )

    result = response.choices[0].message.content

    # If it’s not a dict yet, parse from JSON text:
    if not isinstance(result, dict):
        try:
            result = json.loads(result)
        except Exception as e:
            raise ValueError(
                f"LLM did not return valid JSON; got {type(result)}:\n{result}"
            ) from e

    return result


if __name__ == "__main__":
    weave.init("AI-Analyst")

    dummy_text = """
Apple Inc. posted revenue of $260 billion in fiscal 2022, up 8% year-over-year. 
iPhone segment revenue grew 10% to $183 billion, while Services revenue reached $68 billion. 
Management’s tone remained cautiously optimistic, citing supply chain constraints as a drag on margins. 
Net income was $57 billion, down 3% from the prior year. 
Fetch EV/EBITDA for MSFT, GOOGL, AMZN.
"""

    summary = summarize_filing(dummy_text)
    print("DEBUG summary:", summary, type(summary))