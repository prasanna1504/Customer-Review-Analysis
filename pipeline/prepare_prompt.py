"""
pipeline/prepare_prompt.py
──────────────────────────
Generates ready-to-paste prompt files for manual LLM enrichment.
Paste each file into Claude.ai (or any model), save the JSON response,
then run parse_output.py to merge into enriched_reviews.csv.

Usage:
    python pipeline/prepare_prompt.py --product TraderSync
    python pipeline/prepare_prompt.py --product Skydo --batch-size 100
"""

import os, sys, argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

BASE = Path(__file__).parent.parent

PRODUCTS = {
    "Porter": {
        "clean_csv":    BASE / "data/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/porter/processed/enriched_reviews.csv",
        "description":  "last-mile logistics and moving platform in India (customers book trucks/movers)",
        "product_type": "logistics",
    },
    "Skydo": {
        "clean_csv":    BASE / "data/skydo/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/skydo/processed/enriched_reviews.csv",
        "description":  "international payments and remittance platform for Indian freelancers and businesses",
        "product_type": "payments",
    },
    "Tradezella": {
        "clean_csv":    BASE / "data/tradezella/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradezella/processed/enriched_reviews.csv",
        "description":  "trading journal and performance analytics platform for retail traders",
        "product_type": "trading_journal",
    },
    "TraderSync": {
        "clean_csv":    BASE / "data/tradersync/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradersync/processed/enriched_reviews.csv",
        "description":  "trading journal and performance analytics platform for retail traders",
        "product_type": "trading_journal",
    },
    "FirstClub": {
        "clean_csv":    BASE / "data/firstclub/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/firstclub/processed/enriched_reviews.csv",
        "description":  "premium D2C subscription delivering quality groceries and essentials in minutes",
        "product_type": "marketplace",
    },
}

# ── Category options per product type ────────────────────────────────────────
CATEGORIES = {
    "logistics":       "UX | Feature Gap | Performance | Reliability | Pricing | Driver Behaviour | Safety & Damage | Payment | Support Quality | Booking & Cancellation | Other",
    "payments":        "UX | Feature Gap | Performance | Reliability | Pricing | KYC & Verification | Transfer Speed | Fees & Rates | Bank Integration | Security | Support Quality | Other",
    "trading_journal": "UX | Feature Gap | Performance | Reliability | Pricing | Trade Import | Analytics | Broker Compatibility | Mobile | Bugs | Support Quality | Other",
    "marketplace": "UX | Product Quality | Delivery & Speed | Pricing & Value | App Performance | Subscription | Customer Support | Returns & Refunds | Selection & Discovery | Packaging | Tracking | Other",
}

SUPPORT_CATEGORIES = {
    "logistics":       "Payments | Cancellation | Driver | App Bug | Tracking | Pricing | Account | Other",
    "payments":        "Transfer | KYC | Account Access | Fees | App Bug | Security | Limits | Other",
    "trading_journal": "Import | Sync | Billing | App Bug | Data | Broker | Account | Other",
    "marketplace": "Delivery | Returns & Refunds | Billing | App Bug | Product Quality | Account | Tracking | Subscription | Other",
}


# ── The single mega-prompt system block ──────────────────────────────────────
SYSTEM_PROMPT = """You are a senior product analyst combining four perspectives into one verdict:

1. PRODUCT MANAGER (10 yrs, consumer mobile) — finds product gaps engineering can fix
2. CS LEAD (50k+ tickets) — spots support queue fires and operational urgency
3. GROWTH ANALYST — detects churn intent, referral poison, and trust breaks
4. SKEPTIC / CALIBRATOR — filters noise before the team wastes time on it

Your output is ONE final JSON object per review. No separate agent outputs.

Internal reasoning rules (apply silently, don't output them):
  • If the review is vague venting with no specific behaviour → P4, is_actionable=false
  • If user error or unrealistic expectation → P4, confidence ≤ 0.4
  • If money/safety trust break + churn intent → P1 automatically
  • If CS urgency is Critical AND churn is High → P1 automatically
  • Confidence = how much evidence supports the verdict (0.0–1.0)
  • Default to is_actionable=true when any specific behaviour is named

action_item must be scope-able by an engineer. "Improve the app" = rejected.
"Add confirmation step before charging cancellation fee" = accepted."""


def build_prompt(batch_df: pd.DataFrame, product_name: str,
                 product_desc: str, product_type: str,
                 batch_num: int, total_batches: int,
                 global_start_idx: int) -> str:

    cats         = CATEGORIES.get(product_type, CATEGORIES["trading_journal"])
    support_cats = SUPPORT_CATEGORIES.get(product_type, SUPPORT_CATEGORIES["trading_journal"])

    lines = []

    # ── Instructions ─────────────────────────────────────────────────────────
    lines.append(SYSTEM_PROMPT)
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"PRODUCT: {product_name}")
    lines.append(f"DESCRIPTION: {product_desc}")
    if total_batches > 1:
        lines.append(f"BATCH: {batch_num} of {total_batches}")
    lines.append("=" * 60)
    lines.append("")

    # ── Few-shot example ─────────────────────────────────────────────────────
    lines.append("<example>")
    lines.append("Input review [1] [trustpilot | ★1]:")
    lines.append('  "Tried to cancel my subscription before renewal. '
                 'Was charged anyway. Support said they cant refund '
                 'because its past the billing date. I\'ve been a '
                 'customer for 2 years. Switching to a competitor."')
    lines.append("")
    lines.append("Expected output (just the object for review 1, inside the array):")
    lines.append("""{
  "idx": 1,
  "sentiment": "Negative",
  "sentiment_reason": "User was charged against their intent and support refused to help, compounded by loyalty context.",
  "themes": ["Billing", "Cancellation", "Support Quality"],
  "specific_issue": "Subscription charge applied after cancellation attempt; support denied refund citing billing date policy.",
  "feature_request": null,
  "churn_risk": "High",
  "churn_reason": "Explicit switch intent stated; loyalty context makes it a trust break.",
  "trust_break": true,
  "priority": "P1",
  "priority_reason": "Money trust break + explicit churn + 2-year customer loss.",
  "is_actionable": true,
  "is_feature_request": false,
  "is_bug_report": false,
  "is_churn_signal": true,
  "noise_type": "None",
  "competitor_mentioned": null,
  "action_item": "Add a grace-period cancellation window (e.g. 48h after billing) and empower support to issue one-time loyalty refunds.",
  "key_insight": "Rigid billing policy is actively converting long-term customers into churned detractors.",
  "confidence": 0.92
}""")
    lines.append("</example>")
    lines.append("")

    # ── Output schema ─────────────────────────────────────────────────────────
    lines.append("<output_schema>")
    lines.append(f"""Return a JSON array with exactly {len(batch_df)} objects, one per review, in order.
Return ONLY the JSON array — no markdown fences, no explanation, no preamble.

Each object:
{{
  "idx": <integer matching the review number>,
  "sentiment": "Positive" | "Negative" | "Neutral" | "Mixed",
  "sentiment_reason": "<1 sentence>",
  "themes": ["<up to 3 themes from: {cats}>"],
  "specific_issue": "<concrete problem or praise, null if none>",
  "feature_request": "<specific feature wanted, null if none>",
  "churn_risk": "Low" | "Medium" | "High",
  "churn_reason": "<why, null if Low>",
  "trust_break": <true | false>,
  "priority": "P1" | "P2" | "P3" | "P4",
  "priority_reason": "<1 sentence>",
  "is_actionable": <true | false>,
  "is_feature_request": <true | false>,
  "is_bug_report": <true | false>,
  "is_churn_signal": <true | false>,
  "noise_type": "User Error" | "Unrealistic Expectation" | "One-off Incident" | "Vague Venting" | "None",
  "support_category": "<from: {support_cats}>",
  "competitor_mentioned": "<name or null>",
  "action_item": "<engineer-scope-able recommendation, null if P4>",
  "key_insight": "<1 crisp sentence>",
  "confidence": <float 0.0–1.0>
}}""")
    lines.append("</output_schema>")
    lines.append("")

    # ── Reviews ───────────────────────────────────────────────────────────────
    lines.append("<reviews>")
    for i, (_, row) in enumerate(batch_df.iterrows(), start=global_start_idx):
        platform = str(row.get("platform", "unknown"))
        rating   = row.get("rating", "")
        rating_s = f" | ★{int(float(rating))}" if pd.notna(rating) and str(rating).strip() not in ("", "nan") else ""
        date_s   = str(row.get("date", ""))[:10]
        date_s   = f" | {date_s}" if date_s and date_s not in ("nan", "NaT", "") else ""
        text     = str(row.get("review_text", "")).strip()[:600]
        lines.append(f"[{i}] [{platform}{rating_s}{date_s}]")
        lines.append(text)
        lines.append("")
    lines.append("</reviews>")

    return "\n".join(lines)


def prepare(product_name: str, batch_size: int = 150):
    cfg = PRODUCTS[product_name]

    if not cfg["clean_csv"].exists():
        print(f"❌ Clean CSV not found: {cfg['clean_csv']}")
        return

    df = pd.read_csv(cfg["clean_csv"])
    df = df[df["review_text"].astype(str).str.len() > 15].reset_index(drop=True)
    total = len(df)

    out_dir = BASE / "pipeline" / "prompts" / product_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files
    for f in out_dir.glob("*"):
        f.unlink()

    batches       = [df.iloc[i:i+batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)

    print(f"\n{'='*55}")
    print(f"  Product : {product_name}")
    print(f"  Reviews : {total}")
    print(f"  Batches : {total_batches}  (≤{batch_size} each)")
    print(f"  Output  : {out_dir}/")
    print(f"{'='*55}\n")

    for i, chunk in enumerate(batches, start=1):
        global_start = (i - 1) * batch_size + 1
        prompt_text  = build_prompt(
            chunk, product_name, cfg["description"], cfg["product_type"],
            i, total_batches, global_start
        )

        fname = out_dir / f"{product_name.lower()}_prompt_b{i:02d}.txt"
        fname.write_text(prompt_text, encoding="utf-8")

        tokens_est = int(len(prompt_text.split()) * 1.35)
        reviews_range = f"{global_start}–{global_start + len(chunk) - 1}"
        print(f"  ✅ Batch {i:2d}/{total_batches}  reviews {reviews_range:>8}  "
              f"~{tokens_est:,} tokens  →  {fname.name}")

    # ── HOW_TO_USE ────────────────────────────────────────────────────────────
    how_to = out_dir / "HOW_TO_USE.txt"
    response_files = "\n".join(
        f"  batch {i} → {product_name.lower()}_response_b{i:02d}.json"
        for i in range(1, total_batches + 1)
    )
    how_to.write_text(f"""HOW TO ENRICH {product_name.upper()} REVIEWS
{"=" * 50}

STEP 1 — Open claude.ai/new  (use Claude Sonnet or Opus)

STEP 2 — For each prompt file:
  • Click the paperclip / attachment icon
  • Upload the .txt file  (or copy-paste the contents)
  • Hit send

STEP 3 — When the model finishes:
  • Select ALL of the JSON output  (starts with [  ends with ])
  • Save it as a .json file in THIS folder:

{response_files}

  If the model says it was truncated, reply:
  "Continue the JSON array from where you left off"
  Then append the continuation manually before saving.

STEP 4 — Run the parser:
  python pipeline/parse_output.py --product {product_name}

  Output: data/{product_name.lower()}/processed/enriched_reviews.csv
  The dashboard will automatically use this file next time it loads.

NOTES:
  • {batch_size} reviews per batch — Claude Sonnet handles this in one shot
  • You can use ChatGPT or Gemini too — same prompt works
  • Each batch is independent; run them in any order
""", encoding="utf-8")

    print(f"\n  📄 Instructions → {how_to.name}")
    print(f"\n  When done with Claude, run:")
    print(f"      python pipeline/parse_output.py --product {product_name}\n")


if __name__ == "__main__":
    from config.config import PRODUCT as CONFIG_PRODUCT
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default=CONFIG_PRODUCT, choices=list(PRODUCTS.keys()))
    parser.add_argument("--batch-size", type=int, default=150)
    args = parser.parse_args()
    prepare(args.product, args.batch_size)
