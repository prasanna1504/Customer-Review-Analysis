"""
scrapers/scrape_all.py
Scrapes Porter reviews from Google Play (customer + driver), Reddit, Trustpilot, and X/Twitter.
Outputs a single unified CSV: data/raw/raw_reviews.csv

Install deps:
    pip install google-play-scraper praw requests beautifulsoup4 pandas playwright
    playwright install chromium
"""

import os
import re
import time
import pandas as pd
from datetime import datetime, timezone
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import *


# ─────────────────────────────────────────────
# 1. GOOGLE PLAY SCRAPER
# ─────────────────────────────────────────────
def scrape_playstore(app_id: str, label: str, count: int = MAX_REVIEWS_PER_SOURCE) -> pd.DataFrame:
    try:
        from google_play_scraper import reviews, Sort
        print(f"  Scraping Play Store: {label} ({app_id})...")
        result, _ = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=count
        )
        rows = []
        for r in result:
            rows.append({
                "platform": f"playstore_{label}",
                "date": r.get("at", datetime.now(timezone.utc)).strftime("%Y-%m-%d") if hasattr(r.get("at"), "strftime") else str(r.get("at", "")),
                "author": r.get("userName", ""),
                "rating": r.get("score", None),
                "review_text": r.get("content", ""),
                "source_url": f"https://play.google.com/store/apps/details?id={app_id}"
            })
        df = pd.DataFrame(rows)
        print(f"    ✓ {len(df)} reviews fetched")
        return df
    except Exception as e:
        print(f"    ✗ Play Store scrape failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 2. REDDIT SCRAPER (using requests + JSON API)
# ─────────────────────────────────────────────
def scrape_reddit(queries: list, count: int = MAX_REVIEWS_PER_SOURCE) -> pd.DataFrame:
    import requests
    print(f"  Scraping Reddit...")
    headers = {"User-Agent": "porter-intelligence-research/1.0"}
    rows = []
    for query in queries:
        try:
            url = f"https://www.reddit.com/search.json?q={query.replace(' ', '+')}&sort=new&limit=100&type=link"
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                p = post.get("data", {})
                text = p.get("selftext", "") or p.get("title", "")
                if len(text.strip()) < 20:
                    continue
                rows.append({
                    "platform": "reddit",
                    "date": datetime.fromtimestamp(p.get("created_utc", 0), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "author": p.get("author", ""),
                    "rating": None,
                    "review_text": (p.get("title", "") + " " + p.get("selftext", "")).strip(),
                    "source_url": "https://reddit.com" + p.get("permalink", "")
                })
            time.sleep(2)
        except Exception as e:
            print(f"    ✗ Reddit query '{query}' failed: {e}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["review_text"])
    print(f"    ✓ {len(df)} Reddit posts fetched")
    return df


# ─────────────────────────────────────────────
# 3. TRUSTPILOT SCRAPER (Playwright)
# ─────────────────────────────────────────────
def scrape_trustpilot(url: str, pages: int = 15) -> pd.DataFrame:
    print(f"  Scraping Trustpilot (Playwright)...")
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
    except ImportError:
        print("    ✗ playwright not installed. Run: pip install playwright && playwright install chromium")
        return pd.DataFrame()

    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            consecutive_empty = 0
            for pg in range(1, pages + 1):
                page_url = f"{url}?page={pg}"
                try:
                    page.goto(page_url, wait_until='domcontentloaded', timeout=30000)
                    page.wait_for_timeout(3000)
                    html = page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    articles = soup.find_all('article', {'data-service-review-card-paper': True})
                    if not articles:
                        consecutive_empty += 1
                        if consecutive_empty >= 2:
                            break  # two empty pages in a row = done
                        continue
                    consecutive_empty = 0
                    for a in articles:
                        try:
                            rating_el = a.find('div', {'data-service-review-rating': True})
                            text_el = a.find('p', {'data-service-review-text-typography': True})
                            time_el = a.find('time')
                            aside_el = a.find('aside')
                            title_el = a.find('h2', {'data-service-review-title-typography': True})

                            text = text_el.get_text(strip=True) if text_el else ''
                            title = title_el.get_text(strip=True) if title_el else ''
                            full_text = (title + ' ' + text).strip()

                            if not full_text:
                                continue

                            rating_str = rating_el.get('data-service-review-rating', '') if rating_el else ''
                            date_raw = time_el.get('datetime', '')[:10] if time_el else ''
                            author = aside_el.get('aria-label', '').replace('Info for ', '') if aside_el else ''

                            rows.append({
                                "platform": "trustpilot",
                                "date": date_raw,
                                "author": author,
                                "rating": int(rating_str) if rating_str.isdigit() else None,
                                "review_text": full_text,
                                "source_url": url
                            })
                        except Exception:
                            continue
                except Exception as e:
                    print(f"    ✗ Trustpilot page {pg} failed: {e}")
                    break

            browser.close()
    except Exception as e:
        print(f"    ✗ Trustpilot scrape failed: {e}")

    df = pd.DataFrame(rows)
    print(f"    ✓ {len(df)} Trustpilot reviews fetched")
    return df


# ─────────────────────────────────────────────
# 4. X / TWITTER SCRAPER (Scweet — no API key)
# ─────────────────────────────────────────────
def scrape_twitter(queries: list, max_per_query: int = 100) -> pd.DataFrame:
    print(f"  Scraping X/Twitter (Scweet)...")

    auth_token = os.environ.get("X_AUTH_TOKEN", "")
    if not auth_token:
        print("    ✗ X_AUTH_TOKEN not set — skipping.")
        print("      Get it from x.com: DevTools → Application → Cookies → auth_token")
        print("      Then: export X_AUTH_TOKEN=your_token_here")
        return pd.DataFrame()

    try:
        from Scweet import Scweet
    except ImportError:
        print("    ✗ Scweet not installed. Run: pip install scweet")
        return pd.DataFrame()

    rows = []
    try:
        from Scweet import ScweetConfig
        config = ScweetConfig(
            daily_requests_limit=500,
            daily_tweets_limit=5000,
            auth_cooldown_s=0,
            cooldown_default_s=5,
        )
        scraper = Scweet(auth_token=auth_token, config=config)
        for query in queries:
            try:
                tweets = scraper.search(
                    query,
                    lang="en",
                    limit=max_per_query,
                    tweet_type="Latest",
                )
                for t in tweets:
                    text = t.get("text", "").strip()
                    if len(text) < 20:
                        continue
                    user = t.get("user", {})
                    rows.append({
                        "platform": "twitter",
                        "date": str(t.get("timestamp", ""))[:10],
                        "author": user.get("screen_name", ""),
                        "rating": None,
                        "review_text": text,
                        "source_url": t.get("tweet_url", ""),
                    })
                time.sleep(2)
            except Exception as e:
                print(f"    ✗ Twitter query '{query}' failed: {e}")
    except Exception as e:
        print(f"    ✗ Scweet init failed: {e}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["review_text"]) if rows else pd.DataFrame()
    print(f"    ✓ {len(df)} tweets fetched")
    return df


# ─────────────────────────────────────────────
# 5. GOOGLE MAPS SCRAPER (Playwright)
# ─────────────────────────────────────────────
def scrape_google_maps(reviews_url: str, max_reviews: int = 200) -> pd.DataFrame:
    """
    Scrape Google Maps reviews using Playwright.
    reviews_url: direct link that opens the reviews panel, e.g.
        https://www.google.com/search?q=FirstClub#lrd=0x...:0x...,1,,,,
    """
    print(f"  Scraping Google Maps reviews...")
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup
    except ImportError:
        print("    ✗ playwright not installed. Run: pip install playwright && playwright install chromium")
        return pd.DataFrame()

    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            page.goto(reviews_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(4000)

            # Sort by newest if the sort button is available
            try:
                sort_btn = page.query_selector('[data-sort-id="newestFirst"], [aria-label*="Sort"], button:has-text("Most relevant")')
                if sort_btn:
                    sort_btn.click()
                    page.wait_for_timeout(2000)
                    newest_opt = page.query_selector('[data-sort-id="newestFirst"], [data-value="newestFirst"]')
                    if newest_opt:
                        newest_opt.click()
                        page.wait_for_timeout(2000)
            except Exception:
                pass

            # Scroll to load more reviews
            scrollable = page.query_selector('[jsname="fk8dgd"], .review-dialog-list, [role="main"]')
            scroll_target = scrollable or page
            for _ in range(max_reviews // 10):
                try:
                    if scrollable:
                        scrollable.evaluate("el => el.scrollTop += 2000")
                    else:
                        page.evaluate("window.scrollBy(0, 2000)")
                    page.wait_for_timeout(1500)
                except Exception:
                    break

            # Try "More" buttons to expand truncated text
            try:
                more_btns = page.query_selector_all('button[aria-label="See more"], .review-more-link, button:has-text("More")')
                for btn in more_btns[:50]:
                    try:
                        btn.click()
                        page.wait_for_timeout(200)
                    except Exception:
                        pass
            except Exception:
                pass

            html = page.content()
            browser.close()

            soup = BeautifulSoup(html, "html.parser")

            # Google Maps review selectors (these can change)
            review_containers = (
                soup.find_all("div", {"data-review-id": True}) or
                soup.find_all("div", class_=re.compile(r"review-item|jJc9Ad|WMbnJf"))
            )

            for container in review_containers:
                try:
                    # Rating
                    rating_el = container.find(attrs={"aria-label": re.compile(r"\d+ star")})
                    rating = None
                    if rating_el:
                        m = re.search(r"(\d+)\s+star", rating_el.get("aria-label", ""))
                        if m:
                            rating = int(m.group(1))

                    # Text
                    text_el = container.find("span", {"data-expandable-section": True}) or \
                              container.find("span", class_=re.compile(r"review-full-text|wiI7pd|Jtu6Td"))
                    text = text_el.get_text(strip=True) if text_el else ""

                    # Date
                    date_el = container.find("span", class_=re.compile(r"dehysf|rsqaWe"))
                    date_str = date_el.get_text(strip=True) if date_el else ""

                    # Author
                    author_el = container.find("div", class_=re.compile(r"d4r55|TSUbDb"))
                    author = author_el.get_text(strip=True) if author_el else ""

                    if len(text.strip()) < 10:
                        continue

                    rows.append({
                        "platform": "google_maps",
                        "date": date_str,
                        "author": author,
                        "rating": rating,
                        "review_text": text,
                        "source_url": reviews_url,
                    })
                except Exception:
                    continue

    except Exception as e:
        print(f"    ✗ Google Maps scrape failed: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_text"])
    print(f"    ✓ {len(df)} Google Maps reviews fetched")
    return df


# ─────────────────────────────────────────────
# 6. APPLE APP STORE SCRAPER
# ─────────────────────────────────────────────
def scrape_app_store(app_id: str, app_name: str, country: str = "in",
                     count: int = 200) -> pd.DataFrame:
    """
    Scrape Apple App Store reviews using app-store-scraper.
    Install: pip install app-store-scraper
    app_id: numeric ID from App Store URL (e.g. '6744534743')
    app_name: slug from App Store URL (e.g. 'firstclub-quality-in-minutes')
    """
    print(f"  Scraping App Store: {app_name} ({app_id})...")
    try:
        from app_store_scraper import AppStore
    except ImportError:
        print("    ✗ app-store-scraper not installed. Run: pip install app-store-scraper")
        return pd.DataFrame()

    try:
        app = AppStore(country=country, app_name=app_name, app_id=app_id)
        app.review(how_many=count, sleep=2)
        reviews = app.reviews

        rows = []
        for r in reviews:
            date_val = r.get("date", "")
            if hasattr(date_val, "strftime"):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)[:10]

            rows.append({
                "platform": "app_store",
                "date": date_str,
                "author": r.get("userName", ""),
                "rating": r.get("rating", None),
                "review_text": r.get("review", ""),
                "source_url": f"https://apps.apple.com/{country}/app/{app_name}/id{app_id}",
            })

        df = pd.DataFrame(rows)
        print(f"    ✓ {len(df)} App Store reviews fetched")
        return df

    except Exception as e:
        print(f"    ✗ App Store scrape failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 7. CLEAN + UNIFY
# ─────────────────────────────────────────────
def clean_and_unify(dfs: list) -> pd.DataFrame:
    df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    df = df[df["review_text"].str.len() > 20]
    df = df.drop_duplicates(subset=["review_text"])
    df = df.reset_index(drop=True)

    print(f"\n  ✓ Total unified reviews: {len(df)}")
    print(f"  Platform breakdown:\n{df['platform'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────
# 8. PRODUCT REGISTRY FOR SCRAPERS
# ─────────────────────────────────────────────
SCRAPER_PRODUCTS = {
    "Porter": {
        "raw_data_dir":           "data/raw",
        "playstore_customer_id":  "in.porter.android",
        "playstore_driver_id":    "",
        "reddit_queries":         ["Porter app review", "Porter logistics India", "Porter truck booking"],
        "trustpilot_url":         "",
        "twitter_queries":        ["Porter app India", "@porterapp", "Porter logistics"],
        "google_maps_url":        "",
        "appstore_id":            "",
        "appstore_name":          "",
        "appstore_country":       "in",
    },
    "Skydo": {
        "raw_data_dir":           "data/skydo/raw",
        "playstore_customer_id":  "com.skydo.app",
        "playstore_driver_id":    "",
        "reddit_queries":         ["Skydo review", "Skydo payments India", "Skydo vs wise"],
        "trustpilot_url":         "https://www.trustpilot.com/review/skydo.com",
        "twitter_queries":        ["Skydo", "@skydopay", "Skydo payments"],
        "google_maps_url":        "",
        "appstore_id":            "",
        "appstore_name":          "",
        "appstore_country":       "in",
    },
    "Tradezella": {
        "raw_data_dir":           "data/tradezella/raw",
        "playstore_customer_id":  "com.tradezella.app",
        "playstore_driver_id":    "",
        "reddit_queries":         ["Tradezella review", "Tradezella trading journal", "Tradezella vs"],
        "trustpilot_url":         "https://www.trustpilot.com/review/tradezella.com",
        "twitter_queries":        ["Tradezella", "@tradezella", "Tradezella journal"],
        "google_maps_url":        "",
        "appstore_id":            "",
        "appstore_name":          "",
        "appstore_country":       "us",
    },
    "TraderSync": {
        "raw_data_dir":           "data/tradersync/raw",
        "playstore_customer_id":  "com.tradersync",
        "playstore_driver_id":    "",
        "reddit_queries":         ["TraderSync review", "TraderSync trading journal", "TraderSync vs"],
        "trustpilot_url":         "https://www.trustpilot.com/review/tradersync.com",
        "twitter_queries":        ["TraderSync", "@tradersync", "TraderSync trading journal"],
        "google_maps_url":        "",
        "appstore_id":            "",
        "appstore_name":          "",
        "appstore_country":       "us",
    },
    "FirstClub": {
        "raw_data_dir":           "data/firstclub/raw",
        "playstore_customer_id":  "com.firstclub.app",
        "playstore_driver_id":    "",
        "reddit_queries":         ["FirstClub app review", "FirstClub grocery delivery", "FirstClub subscription"],
        "trustpilot_url":         "",
        "twitter_queries":        ["FirstClub", "@firstclubapp", "FirstClub delivery"],
        "google_maps_url":        "https://www.google.com/search?q=FirstClub#lrd=0x3bae1390768e3ee7:0x3f8295009e707a10,1,,,,",
        "appstore_id":            "6744534743",
        "appstore_name":          "firstclub-quality-in-minutes",
        "appstore_country":       "in",
    },
}


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def run_scrapers(product_name: str = None):
    cfg_globals = None
    if product_name and product_name in SCRAPER_PRODUCTS:
        cfg_globals = SCRAPER_PRODUCTS[product_name]
    else:
        # Fall back to config.py globals
        product_name = PRODUCT
        cfg_globals = {
            "raw_data_dir":           RAW_DATA_DIR,
            "playstore_customer_id":  PLAYSTORE_CUSTOMER_APP_ID,
            "playstore_driver_id":    getattr(sys.modules[__name__], "PLAYSTORE_DRIVER_APP_ID", ""),
            "reddit_queries":         REDDIT_QUERIES,
            "trustpilot_url":         TRUSTPILOT_COMPANY_URL,
            "twitter_queries":        TWITTER_QUERIES,
            "google_maps_url":        "",
            "appstore_id":            "",
            "appstore_name":          "",
            "appstore_country":       "in",
        }

    raw_data_dir = cfg_globals["raw_data_dir"]

    print(f"\n{'='*50}")
    print(f"  Scraping reviews for: {product_name}")
    print(f"{'='*50}\n")

    os.makedirs(raw_data_dir, exist_ok=True)

    dfs = []

    if cfg_globals.get("playstore_customer_id"):
        dfs.append(scrape_playstore(cfg_globals["playstore_customer_id"], "customer"))
    else:
        print("  Skipping Play Store customer (no app ID configured)")

    if cfg_globals.get("playstore_driver_id"):
        dfs.append(scrape_playstore(cfg_globals["playstore_driver_id"], "driver"))

    if cfg_globals.get("reddit_queries"):
        dfs.append(scrape_reddit(cfg_globals["reddit_queries"]))

    if cfg_globals.get("trustpilot_url"):
        dfs.append(scrape_trustpilot(cfg_globals["trustpilot_url"]))
    else:
        print("  Skipping Trustpilot (no URL configured)")

    if cfg_globals.get("google_maps_url"):
        dfs.append(scrape_google_maps(cfg_globals["google_maps_url"], max_reviews=MAX_REVIEWS_PER_SOURCE))
    else:
        print("  Skipping Google Maps (no URL configured)")

    if cfg_globals.get("appstore_id"):
        dfs.append(scrape_app_store(
            cfg_globals["appstore_id"],
            cfg_globals["appstore_name"],
            cfg_globals.get("appstore_country", "in"),
            count=MAX_REVIEWS_PER_SOURCE,
        ))
    else:
        print("  Skipping App Store (no app ID configured)")

    twitter_df = scrape_twitter(cfg_globals.get("twitter_queries", []))
    output_path = os.path.join(raw_data_dir, "raw_reviews.csv")
    if twitter_df.empty and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        existing_twitter = existing[existing["platform"] == "twitter"]
        if not existing_twitter.empty:
            print(f"    ↩ Twitter scrape failed — keeping {len(existing_twitter)} existing tweets")
            twitter_df = existing_twitter
    dfs.append(twitter_df)

    df = clean_and_unify(dfs)
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to {output_path}")
    return df


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(description="Scrape reviews for a product")
    _parser.add_argument("--product", default=None, choices=list(SCRAPER_PRODUCTS.keys()),
                         help="Product to scrape (default: from config.py)")
    _args = _parser.parse_args()
    run_scrapers(_args.product)
