# ============================================================
# config.py — change this to run the pipeline for any product
# ============================================================

PRODUCT = "TraderSync"
PRODUCT_DESCRIPTION = "trading journal and performance analytics platform for traders"

# Google Play app IDs
PLAYSTORE_CUSTOMER_APP_ID = "com.tradersync"
PLAYSTORE_DRIVER_APP_ID   = ""

# Reddit queries
REDDIT_QUERIES = ["TraderSync review", "TraderSync trading journal", "TraderSync vs"]

# Twitter/X search queries
TWITTER_QUERIES = ["TraderSync", "@tradersync", "TraderSync trading journal"]

# Trustpilot
TRUSTPILOT_COMPANY_URL = "https://www.trustpilot.com/review/tradersync.com"

# How many reviews to fetch per source
MAX_REVIEWS_PER_SOURCE = 200

# Anthropic model to use for analysis
LLM_MODEL = "claude-sonnet-4-20250514"

# Output paths (product-specific so runs don't overwrite each other)
_slug = PRODUCT.lower().replace(" ", "_")
RAW_DATA_DIR       = f"data/{_slug}/raw"
PROCESSED_DATA_DIR = f"data/{_slug}/processed"
ENRICHED_CSV_PATH  = f"data/{_slug}/processed/enriched_reviews.csv"
