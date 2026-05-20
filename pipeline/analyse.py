"""
pipeline/analyse.py
Multi-agent LLM pipeline — 5 agents with different perspectives,
synthesised into prioritised action items.

Each review is passed through:
  Agent 1 — Product Manager: feature gaps, UX issues
  Agent 2 — Customer Support Lead: urgent recurring pain
  Agent 3 — Growth Analyst: churn and acquisition blockers
  Agent 4 — Skeptic: filters noise from real issues
  Agent 5 — Synthesiser: resolves conflicts, outputs priority + confidence

Install: pip install anthropic pandas tqdm
"""

import os
import json
import time
import pandas as pd
from tqdm import tqdm
import anthropic
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# AGENT PROMPTS
# ─────────────────────────────────────────────

AGENT_PROMPTS = {
    "product_manager": """You are a Product Manager analysing a user review for {product}, {product_description}.
Your job: identify product gaps, UX friction, and missing features.
Focus on: what is the user actually struggling to do? What feature would fix it?

Review: {review}

Respond in JSON only:
{{"insight": "<1-2 sentences>", "category": "<UX|Feature Gap|Performance|Reliability|Pricing|Other>", "severity": "<High|Medium|Low>"}}\n""",

    "cs_lead": """You are a Customer Support Lead analysing a user review for {product}, {product_description}.
Your job: identify urgent recurring pain points that are flooding support queues.
Focus on: is this a common complaint pattern? How frustrated is the user? Is this urgent?

Review: {review}

Respond in JSON only:
{{"insight": "<1-2 sentences>", "urgency": "<Critical|High|Medium|Low>", "support_category": "<Payments|Cancellation|Driver|App Bug|Tracking|Pricing|Other>"}}\n""",

    "growth_analyst": """You are a Growth Analyst analysing a user review for {product}, {product_description}.
Your job: identify signals that indicate churn risk or acquisition blockers.
Focus on: would this make someone uninstall the app? Would this stop someone from recommending it? Is it a trust issue?

Review: {review}

Respond in JSON only:
{{"insight": "<1-2 sentences>", "churn_risk": "<High|Medium|Low>", "growth_impact": "<Retention|Acquisition|Both|Neither>"}}\n""",

    "skeptic": """You are a Devil's Advocate analysing a user review for {product}, {product_description}.
Your job: challenge whether this complaint reflects a real product problem or is just noise (user error, unrealistic expectation, one-off incident).
Be rigorous — most reviews are noise. Only flag as actionable if you are confident it reflects a systemic issue.

Review: {review}

Respond in JSON only:
{{"is_actionable": true/false, "reasoning": "<1 sentence>", "noise_type": "<User Error|Unrealistic Expectation|One-off Incident|Systemic Issue|null>"}}\n""",
}

SYNTHESISER_PROMPT = """You are a Chief of Staff synthesising 4 analyst perspectives on a user review for {product}.

Review: {review}

PM Analysis: {pm}
CS Lead Analysis: {cs}
Growth Analyst Analysis: {growth}
Skeptic Verdict: {skeptic}

Your job: reconcile all perspectives and output a final prioritised action item.
If the Skeptic says it's not actionable, explain why but still note the theme.

Respond in JSON only:
{{
  "final_category": "<UX|Feature Gap|Performance|Reliability|Pricing|Driver Experience|Payment|Other>",
  "priority": "<P1-Critical|P2-High|P3-Medium|P4-Low>",
  "confidence": <0.0-1.0>,
  "action_item": "<specific actionable recommendation in 1 sentence>",
  "key_insight": "<what this review reveals about the product in 1 sentence>",
  "sentiment": "<Positive|Negative|Neutral|Mixed>"
}}\n"""


# ─────────────────────────────────────────────
# CALL LLM
# ─────────────────────────────────────────────
def call_llm(prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "parse_failed", "raw": text}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e)}
    return {}


# ─────────────────────────────────────────────
# ANALYSE SINGLE REVIEW
# ─────────────────────────────────────────────
def analyse_review(review_text: str) -> dict:
    ctx = {"product": PRODUCT, "product_description": PRODUCT_DESCRIPTION, "review": review_text}

    pm     = call_llm(AGENT_PROMPTS["product_manager"].format(**ctx))
    cs     = call_llm(AGENT_PROMPTS["cs_lead"].format(**ctx))
    growth = call_llm(AGENT_PROMPTS["growth_analyst"].format(**ctx))
    skeptic = call_llm(AGENT_PROMPTS["skeptic"].format(**ctx))

    synth = call_llm(SYNTHESISER_PROMPT.format(
        product=PRODUCT,
        review=review_text,
        pm=json.dumps(pm),
        cs=json.dumps(cs),
        growth=json.dumps(growth),
        skeptic=json.dumps(skeptic)
    ))

    return {
        "pm_insight":         pm.get("insight", ""),
        "pm_category":        pm.get("category", ""),
        "pm_severity":        pm.get("severity", ""),
        "cs_insight":         cs.get("insight", ""),
        "cs_urgency":         cs.get("urgency", ""),
        "cs_support_category": cs.get("support_category", ""),
        "growth_insight":     growth.get("insight", ""),
        "churn_risk":         growth.get("churn_risk", ""),
        "growth_impact":      growth.get("growth_impact", ""),
        "is_actionable":      skeptic.get("is_actionable", True),
        "skeptic_reasoning":  skeptic.get("reasoning", ""),
        "final_category":     synth.get("final_category", ""),
        "priority":           synth.get("priority", ""),
        "confidence":         synth.get("confidence", 0),
        "action_item":        synth.get("action_item", ""),
        "key_insight":        synth.get("key_insight", ""),
        "sentiment":          synth.get("sentiment", ""),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_analysis(input_path: str = None, output_path: str = None, sample: int = None):
    input_path  = input_path  or os.path.join(RAW_DATA_DIR, "raw_reviews.csv")
    output_path = output_path or ENRICHED_CSV_PATH

    print(f"\n{'='*50}")
    print(f"  Multi-Agent Analysis: {PRODUCT}")
    print(f"{'='*50}\n")

    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} reviews")

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42)
        print(f"  Sampling {len(df)} reviews for analysis")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analysing"):
        analysis = analyse_review(str(row["review_text"]))
        results.append(analysis)
        time.sleep(0.5)  # rate limit buffer

    analysis_df = pd.DataFrame(results)
    enriched_df = pd.concat([df.reset_index(drop=True), analysis_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched_df.to_csv(output_path, index=False)

    print(f"\n  ✓ Enriched CSV saved to {output_path}")
    print(f"\n  Priority breakdown:")
    if "priority" in enriched_df.columns:
        print(enriched_df["priority"].value_counts().to_string())

    return enriched_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Analyse only N reviews (for testing)")
    args = parser.parse_args()
    run_analysis(sample=args.sample)
