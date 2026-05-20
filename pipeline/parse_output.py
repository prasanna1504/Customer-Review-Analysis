"""
pipeline/parse_output.py
────────────────────────
Merges Claude.ai JSON responses with the original clean CSV to produce
enriched_reviews.csv — the file the dashboard reads for deep analysis.

Usage:
    python pipeline/parse_output.py --product TraderSync
    python pipeline/parse_output.py --product Skydo
    python pipeline/parse_output.py --product Porter --batch-size 100

Expects response files at:
    pipeline/prompts/{product_lower}/{product_lower}_response_b01.json
    pipeline/prompts/{product_lower}/{product_lower}_response_b02.json
    ...

Output:
    data/{product_lower}/processed/enriched_reviews.csv
"""

import os, sys, json, re, argparse
import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

BASE = Path(__file__).parent.parent

PRODUCTS = {
    "Porter": {
        "clean_csv":    BASE / "data/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/porter/processed/enriched_reviews.csv",
    },
    "Skydo": {
        "clean_csv":    BASE / "data/skydo/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/skydo/processed/enriched_reviews.csv",
    },
    "Tradezella": {
        "clean_csv":    BASE / "data/tradezella/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradezella/processed/enriched_reviews.csv",
    },
    "TraderSync": {
        "clean_csv":    BASE / "data/tradersync/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/tradersync/processed/enriched_reviews.csv",
    },
    "FirstClub": {
        "clean_csv":    BASE / "data/firstclub/processed/clean_reviews.csv",
        "enriched_csv": BASE / "data/firstclub/processed/enriched_reviews.csv",
    },
}

# All expected fields from the LLM response schema
LLM_FIELDS = [
    "sentiment", "sentiment_reason", "themes", "specific_issue",
    "feature_request", "churn_risk", "churn_reason", "trust_break",
    "priority", "priority_reason", "is_actionable", "is_feature_request",
    "is_bug_report", "is_churn_signal", "noise_type", "support_category",
    "competitor_mentioned", "action_item", "key_insight", "confidence",
]


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if the model wrapped its output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> list:
    """
    Try to extract a JSON array from raw text.
    Handles:
      - Clean array
      - Array wrapped in markdown fences
      - Partial output (truncated) — salvages complete objects
    """
    text = _strip_fences(text)

    # Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        # Wrapped in an object key
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    # Salvage: find the opening bracket and try progressively shorter slices
    start = text.find("[")
    if start == -1:
        return []

    # Try adding a closing bracket to handle truncated output
    candidate = text[start:]
    for suffix in ["", "]", "\n]", ",null]"]:
        try:
            parsed = json.loads(candidate + suffix)
            if isinstance(parsed, list):
                print(f"    ⚠️  JSON was truncated — salvaged {len(parsed)} objects (used suffix: {repr(suffix)})")
                return [obj for obj in parsed if isinstance(obj, dict)]
        except json.JSONDecodeError:
            pass

    # Last resort: extract individual objects with regex
    objects = []
    depth = 0
    buf = []
    in_obj = False
    for ch in candidate:
        if ch == "{":
            depth += 1
            in_obj = True
        if in_obj:
            buf.append(ch)
        if ch == "}":
            depth -= 1
            if depth == 0 and in_obj:
                raw_obj = "".join(buf)
                try:
                    obj = json.loads(raw_obj)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                buf = []
                in_obj = False

    if objects:
        print(f"    ⚠️  Recovered {len(objects)} objects via regex extraction")
    return objects


def load_response_file(path: Path) -> list[dict]:
    """Load and parse a single response file, return list of enrichment dicts."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"    ✗ Cannot read {path.name}: {e}")
        return []

    objects = _extract_json_array(raw)
    if not objects:
        print(f"    ✗ No valid JSON array found in {path.name}")
    return objects


def parse_product(product_name: str, batch_size: int = 150):
    cfg = PRODUCTS[product_name]
    slug = product_name.lower()

    # ── 1. Load source reviews ────────────────────────────────────────────────
    if not cfg["clean_csv"].exists():
        print(f"❌ Clean CSV not found: {cfg['clean_csv']}")
        return

    df = pd.read_csv(cfg["clean_csv"])
    df = df[df["review_text"].astype(str).str.len() > 15].reset_index(drop=True)
    total = len(df)
    print(f"\n{'='*55}")
    print(f"  Product      : {product_name}")
    print(f"  Source rows  : {total}")
    print(f"{'='*55}\n")

    # ── 2. Find response files ────────────────────────────────────────────────
    prompts_dir = BASE / "pipeline" / "prompts" / slug
    if not prompts_dir.exists():
        print(f"❌ Prompts directory not found: {prompts_dir}")
        print(f"   Run: python pipeline/prepare_prompt.py --product {product_name}")
        return

    response_files = sorted(prompts_dir.glob(f"{slug}_response_b*.json"))
    if not response_files:
        print(f"❌ No response files found in {prompts_dir}")
        print(f"   Expected pattern: {slug}_response_b01.json, {slug}_response_b02.json, ...")
        print(f"\n   Steps to generate them:")
        print(f"   1. Run: python pipeline/prepare_prompt.py --product {product_name}")
        print(f"   2. Open each .txt file in {prompts_dir}/")
        print(f"   3. Paste into claude.ai, save the JSON response as the .json filename above")
        return

    print(f"  Found {len(response_files)} response file(s):")
    for f in response_files:
        size_kb = f.stat().st_size / 1024
        print(f"    • {f.name}  ({size_kb:.1f} KB)")
    print()

    # ── 3. Parse all responses → build idx → enrichment map ──────────────────
    enrichment_map: dict[int, dict] = {}   # 1-based idx → enrichment dict
    total_parsed = 0
    total_skipped = 0

    for resp_file in response_files:
        # Infer batch number from filename: *_b01.json → 1
        m = re.search(r"_b(\d+)\.json$", resp_file.name)
        batch_num = int(m.group(1)) if m else 0
        global_start = (batch_num - 1) * batch_size + 1

        objects = load_response_file(resp_file)

        for i, obj in enumerate(objects):
            # Prefer explicit idx field; fall back to position-based
            idx = obj.get("idx")
            if idx is None:
                idx = global_start + i
            else:
                idx = int(idx)

            if idx < 1 or idx > total:
                total_skipped += 1
                continue

            enrichment_map[idx] = obj
            total_parsed += 1

        print(f"    ✅ {resp_file.name}  → {len(objects)} objects (batch {batch_num}, rows {global_start}–{global_start+len(objects)-1})")

    print(f"\n  Parsed total : {total_parsed} enriched objects")
    if total_skipped:
        print(f"  Skipped      : {total_skipped} out-of-range idx values")
    missing = total - len(enrichment_map)
    if missing:
        print(f"  ⚠️  Missing    : {missing} rows have no LLM enrichment (will keep original data)")

    # ── 4. Merge enrichment into source DataFrame ─────────────────────────────
    # Add LLM columns (NaN by default for un-enriched rows)
    for col in LLM_FIELDS:
        df[col] = None

    enriched_count = 0
    for idx_1based, enrichment in enrichment_map.items():
        row_i = idx_1based - 1   # convert to 0-based
        for field in LLM_FIELDS:
            val = enrichment.get(field)
            # Flatten themes list → pipe-separated string for CSV compatibility
            if field == "themes" and isinstance(val, list):
                val = " | ".join(val)
            df.at[row_i, field] = val
        enriched_count += 1

    print(f"\n  Enriched rows: {enriched_count} / {total}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_path = cfg["enriched_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n  ✅ Saved → {out_path}")
    print(f"  Columns: {list(df.columns)}\n")

    # ── 6. Quick stats ────────────────────────────────────────────────────────
    enriched_df = df[df["sentiment"].notna()]
    if len(enriched_df) == 0:
        print("  ℹ️  No enriched rows to summarise.")
        return

    print(f"  {'─'*40}")
    print(f"  Sentiment breakdown  ({len(enriched_df)} enriched rows):")
    print(enriched_df["sentiment"].value_counts().to_string())

    if "priority" in enriched_df.columns and enriched_df["priority"].notna().any():
        print(f"\n  Priority breakdown:")
        print(enriched_df["priority"].value_counts().to_string())

    if "churn_risk" in enriched_df.columns and enriched_df["churn_risk"].notna().any():
        print(f"\n  Churn risk:")
        print(enriched_df["churn_risk"].value_counts().to_string())

    for flag in ["is_feature_request", "is_bug_report", "is_churn_signal"]:
        if flag in enriched_df.columns:
            n = enriched_df[flag].astype(str).str.lower().isin(["true", "1"]).sum()
            label = flag.replace("is_", "").replace("_", " ").title()
            print(f"\n  {label}: {n}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from config.config import PRODUCT as CONFIG_PRODUCT
    except ImportError:
        CONFIG_PRODUCT = "TraderSync"

    parser = argparse.ArgumentParser(
        description="Merge Claude.ai JSON responses into enriched_reviews.csv"
    )
    parser.add_argument(
        "--product",
        default=CONFIG_PRODUCT,
        choices=list(PRODUCTS.keys()),
        help="Product to parse (default: from config.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help="Batch size used when running prepare_prompt.py (default: 150)",
    )
    args = parser.parse_args()
    parse_product(args.product, args.batch_size)
