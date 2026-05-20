"""
pipeline/enrich.py
──────────────────
LLM enrichment pipeline using Groq (free tier).
Reads clean_reviews.csv, enriches each review with LLM analysis,
saves incrementally to enriched_reviews.csv.

Usage:
    export GROQ_API_KEY=your_key_here
    python pipeline/enrich.py                        # uses config.py product
    python pipeline/enrich.py --product TraderSync   # override product
    python pipeline/enrich.py --product Skydo --resume  # skip already-done rows

Rate limits (Groq free tier):
    30 req/min · 14,400 req/day · processes ~5 reviews/request → ~370 reviews in ~3 min
"""

import os, sys, json, time, argparse, re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from groq import Groq
except ImportError:
    print("❌ Run: pip install groq")
    sys.exit(1)

# ── Product registry (mirrors dashboard) ─────────────────────────────────────
BASE = Path(__file__).parent.parent

PRODUCTS = {
    "Porter": {
        "clean_csv":    BASE / "data/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/porter/processed/enriched_reviews.csv",
        "product_type": "logistics",
        "description":  "last-mile logistics and moving platform in India",
    },
    "Skydo": {
        "clean_csv":    BASE / "data/skydo/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/skydo/processed/enriched_reviews.csv",
        "product_type": "payments",
        "description":  "international payments and remittance platform for Indian freelancers and businesses",
    },
    "Tradezella": {
        "clean_csv":    BASE / "data/tradezella/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradezella/processed/enriched_reviews.csv",
        "product_type": "trading_journal",
        "description":  "trading journal and performance analytics platform for retail traders",
    },
    "TraderSync": {
        "clean_csv":    BASE / "data/tradersync/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradersync/processed/enriched_reviews.csv",
        "product_type": "trading_journal",
        "description":  "trading journal and performance analytics platform for retail traders",
    },
    "FirstClub": {
        "clean_csv":    BASE / "data/firstclub/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/firstclub/processed/enriched_reviews.csv",
        "product_type": "marketplace",
        "description":  "premium D2C subscription delivering quality groceries and essentials in minutes",
    },
}

# ── Groq model ────────────────────────────────────────────────────────────────
MODEL       = "llama-3.1-8b-instant"   # fast, free, good for structured tasks
BATCH_SIZE  = 5                         # reviews per API call
RPM_LIMIT   = 28                        # stay under 30 RPM
SLEEP_BATCH = 60 / RPM_LIMIT           # ~2.1 sec between calls

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior product analyst. You receive batches of user reviews for a software product and return structured JSON analysis.

For EACH review return exactly this JSON object:
{
  "sentiment": "Positive" | "Negative" | "Neutral" | "Mixed",
  "sentiment_reason": "one sentence explaining why",
  "themes": ["list", "of", "relevant", "themes"],
  "specific_issue": "exact problem or praise in concrete terms, or null",
  "feature_request": "specific feature the user wants built, or null",
  "churn_risk": "Low" | "Medium" | "High",
  "churn_reason": "why churn risk is this level, or null",
  "priority": "P1" | "P2" | "P3" | "P4",
  "priority_reason": "why this priority",
  "action_item": "specific actionable recommendation for the product team, or null",
  "key_insight": "one crisp sentence summarising what matters most about this review",
  "is_feature_request": true | false,
  "is_bug_report": true | false,
  "is_churn_signal": true | false,
  "competitor_mentioned": "competitor name if mentioned, or null"
}

Priority guide:
  P1 = data loss, app broken, can't use core feature, security issue
  P2 = significant friction, billing problem, feature missing blocking workflow
  P3 = UX issue, minor bug, improvement request
  P4 = cosmetic, nice-to-have, general praise

Churn risk guide:
  High   = user explicitly switching, cancelled, or threatening to leave
  Medium = strong frustration, comparing to competitors
  Low    = minor issue or positive review

Return a JSON array with one object per review, in the same order as input.
Return ONLY the JSON array, no markdown, no explanation."""


def build_user_prompt(reviews: list[dict], product_name: str, product_desc: str) -> str:
    lines = [f"Product: {product_name} — {product_desc}\n"]
    for i, r in enumerate(reviews):
        platform = r.get("platform", "unknown")
        rating   = r.get("rating", "")
        rating_str = f" | Rating: {rating}★" if pd.notna(rating) and rating != "" else ""
        text = str(r.get("review_text", "")).strip()[:600]
        lines.append(f"Review {i+1} [{platform}{rating_str}]:\n{text}")
    return "\n\n".join(lines)


def call_groq(client: Groq, reviews: list[dict], product_name: str,
              product_desc: str, retries: int = 3) -> list[dict]:
    prompt = build_user_prompt(reviews, product_name, product_desc)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            results = json.loads(raw)
            if isinstance(results, list) and len(results) == len(reviews):
                return results
            # Sometimes model wraps in an object
            if isinstance(results, dict):
                for v in results.values():
                    if isinstance(v, list) and len(v) == len(reviews):
                        return v
            print(f"  ⚠️  Got {len(results) if isinstance(results, list) else 'non-list'} results for {len(reviews)} reviews, retrying...")

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 60
                print(f"  ⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ⚠️  API error (attempt {attempt+1}): {e}")

        time.sleep(2 ** attempt)

    # Fallback: return empty enrichment for this batch
    return [_empty_enrichment() for _ in reviews]


def _empty_enrichment() -> dict:
    return {
        "sentiment": None, "sentiment_reason": None, "themes": [],
        "specific_issue": None, "feature_request": None,
        "churn_risk": None, "churn_reason": None,
        "priority": None, "priority_reason": None,
        "action_item": None, "key_insight": None,
        "is_feature_request": False, "is_bug_report": False,
        "is_churn_signal": False, "competitor_mentioned": None,
    }


def enrich_product(product_name: str, resume: bool = False):
    cfg = PRODUCTS[product_name]

    # ── Load clean reviews ────────────────────────────────────────────────────
    if not cfg["clean_csv"].exists():
        print(f"❌ Clean CSV not found: {cfg['clean_csv']}")
        return

    df = pd.read_csv(cfg["clean_csv"])
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 15].reset_index(drop=True)
    total = len(df)
    print(f"\n{'='*55}")
    print(f"  Enriching: {product_name}  ({total} reviews)")
    print(f"  Model:     {MODEL}")
    print(f"  Output:    {cfg['enriched_csv']}")
    print(f"{'='*55}\n")

    # ── Resume: skip already-enriched rows ────────────────────────────────────
    enriched_path = cfg["enriched_csv"]
    enriched_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if resume and enriched_path.exists():
        existing = pd.read_csv(enriched_path)
        already_done = set(existing["review_text"].tolist())
        print(f"  ↩  Resuming — {len(already_done)} rows already enriched\n")

    # ── Enrich in batches ─────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("❌ Set GROQ_API_KEY environment variable first")
        return

    client = Groq(api_key=api_key)

    rows_to_enrich = df[~df["review_text"].isin(already_done)].copy()
    print(f"  Processing {len(rows_to_enrich)} reviews in batches of {BATCH_SIZE}...\n")

    all_enriched = []
    batches = [rows_to_enrich.iloc[i:i+BATCH_SIZE]
               for i in range(0, len(rows_to_enrich), BATCH_SIZE)]

    with tqdm(total=len(rows_to_enrich), unit="review", ncols=70) as pbar:
        for batch_df in batches:
            reviews = batch_df.to_dict("records")
            results = call_groq(client, reviews, product_name,
                                cfg["description"])

            for orig_row, enrichment in zip(reviews, results):
                merged = {**orig_row, **enrichment}
                # Flatten themes list to string for CSV
                if isinstance(merged.get("themes"), list):
                    merged["themes"] = " | ".join(merged["themes"])
                all_enriched.append(merged)

            # Save incrementally after every batch
            out_df = pd.DataFrame(all_enriched)
            if already_done and enriched_path.exists():
                existing = pd.read_csv(enriched_path)
                existing = existing[existing["review_text"].isin(already_done)]
                out_df = pd.concat([existing, out_df], ignore_index=True)
            out_df.to_csv(enriched_path, index=False)

            pbar.update(len(reviews))
            time.sleep(SLEEP_BATCH)

    # ── Summary ───────────────────────────────────────────────────────────────
    final = pd.read_csv(enriched_path)
    print(f"\n✅ Done! {len(final)} reviews enriched → {enriched_path}")
    print(f"\n  Sentiment breakdown:")
    if "sentiment" in final.columns:
        print(final["sentiment"].value_counts().to_string())
    print(f"\n  Priority breakdown:")
    if "priority" in final.columns:
        print(final["priority"].value_counts().to_string())
    print(f"\n  Churn risk:")
    if "churn_risk" in final.columns:
        print(final["churn_risk"].value_counts().to_string())
    feat_reqs = final["is_feature_request"].sum() if "is_feature_request" in final.columns else 0
    bugs      = final["is_bug_report"].sum()      if "is_bug_report"      in final.columns else 0
    churns    = final["is_churn_signal"].sum()    if "is_churn_signal"    in final.columns else 0
    print(f"\n  Feature requests: {feat_reqs}")
    print(f"  Bug reports:      {bugs}")
    print(f"  Churn signals:    {churns}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config.config import PRODUCT as CONFIG_PRODUCT

    parser = argparse.ArgumentParser(description="LLM enrichment pipeline via Groq")
    parser.add_argument("--product", default=CONFIG_PRODUCT,
                        choices=list(PRODUCTS.keys()),
                        help="Product to enrich (default: from config.py)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already in enriched_reviews.csv")
    args = parser.parse_args()

    enrich_product(args.product, resume=args.resume)
