"""
dashboard/app.py
Multi-product Customer Intelligence Dashboard
No LLM required — uses VADER sentiment + TF-IDF phrase extraction
"""

import os
import re
import sys
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import Counter

# ── optional NLP deps ──────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _TFIDF_OK = True
except ImportError:
    _TFIDF_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

PRODUCTS = {
    "FirstClub": {
        "raw_csv":      os.path.join(BASE_DIR, "data/firstclub/raw/raw_reviews.csv"),
        "clean_csv":    os.path.join(BASE_DIR, "data/firstclub/processed/clean_reviews.csv"),
        "enriched_csv": os.path.join(BASE_DIR, "data/firstclub/processed/enriched_reviews.csv"),
        "product_type": "marketplace",
        "emoji":        "🛒",
        "description":  "Premium D2C subscription — quality groceries & essentials",
    },
    "Tradezella": {
        "raw_csv":      os.path.join(BASE_DIR, "data/tradezella/raw/raw_reviews.csv"),
        "clean_csv":    os.path.join(BASE_DIR, "data/tradezella/processed/clean_reviews.csv"),
        "enriched_csv": os.path.join(BASE_DIR, "data/tradezella/processed/enriched_reviews.csv"),
        "product_type": "trading_journal",
        "emoji":        "📈",
        "description":  "Trading journal & performance analytics",
    },
    "TraderSync": {
        "raw_csv":      os.path.join(BASE_DIR, "data/tradersync/raw/raw_reviews.csv"),
        "clean_csv":    os.path.join(BASE_DIR, "data/tradersync/processed/clean_reviews.csv"),
        "enriched_csv": os.path.join(BASE_DIR, "data/tradersync/processed/enriched_reviews.csv"),
        "product_type": "trading_journal",
        "emoji":        "📊",
        "description":  "Trading journal & performance analytics platform",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# THEME PATTERNS  (domain-aware keyword clusters)
# ══════════════════════════════════════════════════════════════════════════════
THEME_PATTERNS = {
    "logistics": {
        "Delivery Speed":          r"\b(delay|late|slow|fast|quick|on.?time|hour|wait|delivery time)\b",
        "Driver Behaviour":        r"\b(driver|captain|rude|polite|behaviour|professional|helpful|attitude)\b",
        "Pricing & Charges":       r"\b(price|cost|charge|expensive|cheap|fee|extra|overcharg|fare|rate)\b",
        "App & UX":                r"\b(app|ui|interface|bug|crash|glitch|update|feature|navigation|map)\b",
        "Customer Support":        r"\b(support|help|response|customer service|refund|complaint|resolve|chat)\b",
        "Booking & Cancellation":  r"\b(book|cancel|reschedul|confirm|slot|availab)\b",
        "Safety & Damage":         r"\b(damage|broken|safe|theft|stolen|careful|mishandl)\b",
        "Payment & Refunds":       r"\b(payment|refund|cashback|wallet|upi|cash|transaction|deduct)\b",
    },
    "payments": {
        "Transfer Speed":          r"\b(fast|slow|instant|quick|delay|time|hour|minute|transfer speed|swift)\b",
        "Fees & Rates":            r"\b(fee|charge|rate|cost|expensive|cheap|forex|spread|hidden|markup)\b",
        "KYC & Verification":      r"\b(kyc|verif|document|id|proof|aadhar|pan|passport|compliance|limit)\b",
        "App & UX":                r"\b(app|ui|interface|bug|crash|glitch|update|feature|dashboard|portal)\b",
        "Customer Support":        r"\b(support|help|response|customer service|chat|email|ticket|resolve)\b",
        "Bank & Integration":      r"\b(bank|account|integrate|swift|iban|wire|neft|imps|connect)\b",
        "Security & Trust":        r"\b(secure|safe|trust|fraud|scam|otp|2fa|risk|protect)\b",
        "Limits & Restrictions":   r"\b(limit|restrict|block|hold|freeze|cap|maximum|minimum)\b",
    },
    "trading_journal": {
        "Trade Import & Sync":     r"\b(import|sync|connect|broker|csv|auto|manual|upload|fetch|pull)\b",
        "Analytics & Reports":     r"\b(analytic|report|stat|metric|performance|insight|chart|graph|data|win rate|p&l|profit|loss)\b",
        "App & UX":                r"\b(app|ui|interface|bug|crash|glitch|update|feature|design|layout|navigation)\b",
        "Pricing & Plans":         r"\b(price|cost|plan|subscription|fee|expensive|cheap|free|tier|pro|premium)\b",
        "Customer Support":        r"\b(support|help|response|customer service|chat|email|ticket|resolve|reply)\b",
        "Journal & Notes":         r"\b(journal|note|tag|comment|emotion|mindset|playbook|strategy|rule)\b",
        "Broker Compatibility":    r"\b(broker|thinkorswim|td ameritrade|interactive brokers|ibkr|webull|robinhood|etrade|tasty|tradovate|ninjatrader|tradier|mt4|mt5)\b",
        "Mobile Experience":       r"\b(mobile|iphone|android|ios|phone|tablet|app store|google play)\b",
        "Bugs & Reliability":      r"\b(bug|crash|error|freeze|slow|glitch|broken|issue|problem|fix|lag)\b",
        "Comparison & Switching":  r"\b(tradervue|edgewonk|myfxbook|stocktrak|tradeviz|vs\s+\w+|switch(ed|ing)?\s+(to|from|away)|moved?\s+to|compared?\s+to|cancel(l?ed)?\s+(sub|account)|left\s+for|alternative\s+to)\b",
    },
    "marketplace": {
        "Delivery & Speed":        r"\b(deliver|fast|slow|quick|late|delay|on.?time|hour|express|same.?day|rush)\b",
        "Product Quality":         r"\b(quality|fresh|stale|damage|broken|expire|tamper|organic|premium|authentic|rotten)\b",
        "App & UX":                r"\b(app|ui|interface|bug|crash|glitch|update|feature|navigation|load|screen)\b",
        "Pricing & Value":         r"\b(price|cost|expensive|cheap|value|worth|discount|offer|deal|overpriced)\b",
        "Subscription & Plans":    r"\b(subscri|member|plan|renew|cancel|pause|tier|premium|free.?trial|autorenew)\b",
        "Customer Support":        r"\b(support|help|response|customer service|refund|complaint|resolve|chat|call|agent)\b",
        "Returns & Refunds":       r"\b(return|refund|replace|wrong|missing|incorrect|credit|compensat|reimburse)\b",
        "Selection & Discovery":   r"\b(selection|variety|range|catalog|product|brand|availab|stock|out.?of.?stock|choice)\b",
        "Packaging":               r"\b(pack|box|bag|container|spill|leak|seal|wrap|damage|tamper|broken|crush)\b",
        "Order Tracking":          r"\b(track|update|notif|eta|status|live|real.?time|alert|where.?is|location)\b",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    border-left: 4px solid #3b82f6;
}
.quote-card {
    background: #0f172a;
    border-left: 3px solid #64748b;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 6px 0;
    font-style: italic;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.theme-bar-label {
    font-size: 0.82rem;
    color: #94a3b8;
}
.big-number {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_analyse(product_name: str) -> pd.DataFrame:
    cfg = PRODUCTS[product_name]
    # Try enriched CSV first (LLM pipeline output), then clean, then raw
    for path in [cfg.get("enriched_csv",""), cfg["clean_csv"], cfg["raw_csv"]]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["review_text"] = df["review_text"].astype(str).str.strip()
            df = df[df["review_text"].str.len() > 15].copy()

            use_llm_data = product_name in ["Tradezella", "TraderSync", "FirstClub"]

            # ── Sentiment via VADER + context correction ──────────────────────
            if use_llm_data and "sentiment" in df.columns and df["sentiment"].notna().any():
                if "sentiment_score" not in df.columns:
                    def map_score(s):
                        if pd.isna(s): return 0.0
                        if str(s).lower() == "positive": return 0.5
                        if str(s).lower() == "negative": return -0.5
                        return 0.0
                    df["sentiment_score"] = df["sentiment"].apply(map_score)
            elif _VADER_OK:
                sia = SentimentIntensityAnalyzer()

                # Patterns where past-tense struggle + product success = Positive
                # e.g. "helped me pass", "finally profitable", "changed my trading"
                _SUCCESS_STORY = re.compile(
                    r"(helped?\s+me|thanks?\s+to|because\s+of|using\s+\w+\s+i|"
                    r"finally\s+(profitable|pass|success|made|broke|achiev)|"
                    r"(pass(ed)?|achiev\w+|improv\w+|fix\w+|solv\w+)\s+.{0,40}(challeng|problem|issue|goal)|"
                    r"(best|great|amazing|love|recommend|game.?changer|life.?chang)\s+.{0,30}(tool|app|platform|software|journal)|"
                    r"(i\s+was\s+(losing|failing|struggling).{0,60}(now|until|before|but).{0,60}(profit|pass|better|improv|win)))",
                    re.IGNORECASE
                )
                # Patterns that are genuinely negative about the product
                _PRODUCT_COMPLAINT = re.compile(
                    r"(doesn'?t\s+work|not\s+work|broken|useless|waste\s+of|"
                    r"cancelled?\s+my\s+(sub|account)|refund|charged\s+(me\s+)?twice|"
                    r"data\s+(lost|missing|wrong|incorrect)|can'?t\s+(import|sync|connect|login|access)|"
                    r"(terrible|horrible|awful|disgusting|garbage|trash)\s+(app|tool|software|platform|support|service))",
                    re.IGNORECASE
                )

                def contextual_sentiment(text: str, rating=None) -> tuple[str, float]:
                    """
                    Priority order:
                      1. Star rating (ground truth) — if present, use it directly.
                         4-5 ★ → Positive | 3 ★ → Neutral | 1-2 ★ → Negative
                      2. VADER + success-story correction — for unrated text (Reddit/Twitter).
                    """
                    t = str(text)

                    # ── 1. Rating override (highest trust) ───────────────────
                    try:
                        r = float(rating)
                        if not pd.isna(r):
                            if r >= 4:   return "Positive", round((r - 3) / 2, 2)   # 0.5–1.0
                            if r == 3:   return "Neutral",  0.0
                            if r <= 2:   return "Negative", round((r - 3) / 2, 2)   # -0.5 to -1.0
                    except (TypeError, ValueError):
                        pass

                    # ── 2. VADER for unrated reviews ─────────────────────────
                    raw = sia.polarity_scores(t)["compound"]

                    # Success-story correction: VADER wrongly reads past-tense
                    # struggle words as negative even in positive testimonials
                    if raw < -0.05:
                        if _SUCCESS_STORY.search(t) and not _PRODUCT_COMPLAINT.search(t):
                            return "Positive", 0.35

                    if raw >= 0.05:   return "Positive", raw
                    if raw <= -0.05:  return "Negative", raw
                    return "Neutral", raw

                # Rating must be normalised first so we can pass it in
                if "rating" not in df.columns:
                    df["rating"] = None
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

                sentiments = df.apply(
                    lambda row: contextual_sentiment(row["review_text"], row["rating"]),
                    axis=1
                )
                df["sentiment"]       = sentiments.apply(lambda x: x[0])
                df["sentiment_score"] = sentiments.apply(lambda x: x[1])
                df["sentiment_raw_vader"] = df["review_text"].apply(
                    lambda t: sia.polarity_scores(str(t))["compound"]
                )
                df["sentiment_corrected"] = df["sentiment_score"] != df["sentiment_raw_vader"]

            elif "sentiment" not in df.columns:
                df["sentiment"]       = "Unknown"
                df["sentiment_score"] = 0.0

            # ── Rating normalise (if VADER block was skipped) ────────────────
            if "rating" not in df.columns:
                df["rating"] = None
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

            # ── Theme tagging ────────────────────────────────────────────────
            patterns = THEME_PATTERNS.get(cfg["product_type"], {})
            
            if use_llm_data and "themes" in df.columns:
                import ast
                def parse_themes(t_val):
                    if pd.isna(t_val): return []
                    if isinstance(t_val, list): return t_val
                    try: return ast.literal_eval(t_val)
                    except: return []
                df["parsed_themes"] = df["themes"].apply(parse_themes)
                
                # Make sure all standard themes are present as columns (even if 0)
                for theme in patterns.keys():
                    df[f"theme_{theme}"] = df["parsed_themes"].apply(lambda x: theme in x if isinstance(x, list) else False)
                
                # Also capture any new themes the LLM invented
                all_found_themes = set(t for tlist in df["parsed_themes"] if isinstance(tlist, list) for t in tlist)
                for theme in all_found_themes:
                    if theme and f"theme_{theme}" not in df.columns:
                        df[f"theme_{theme}"] = df["parsed_themes"].apply(lambda x: theme in x if isinstance(x, list) else False)
            else:
                for theme, pat in patterns.items():
                    df[f"theme_{theme}"] = df["review_text"].str.lower().str.contains(
                        pat, flags=re.IGNORECASE, regex=True, na=False
                    )

            return df

    return pd.DataFrame()


def get_top_phrases(texts: pd.Series, top_n: int = 20, ngram_range=(2, 3)) -> list[tuple]:
    """Return (phrase, count) list ordered by TF-IDF score."""
    clean = texts.dropna().astype(str)
    clean = clean[clean.str.len() > 10]
    if len(clean) < 5 or not _TFIDF_OK:
        # Fallback: simple word frequency
        words = " ".join(clean.tolist()).lower().split()
        stops = {"the","a","an","is","it","in","on","to","of","and","or","for",
                 "was","are","i","my","me","you","with","this","that","they",
                 "have","has","be","at","by","we","as","so","but","not","if",
                 "from","do","app","very","just","can","will","been","also",
                 "their","its","our","there","when","all","more","than","had"}
        freq = Counter(w for w in words if w not in stops and len(w) > 3)
        return freq.most_common(top_n)
    try:
        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=5000,
            stop_words="english",
            min_df=2,
        )
        X = vec.fit_transform(clean)
        scores = np.asarray(X.sum(axis=0)).flatten()
        terms = vec.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_n]
        return [(terms[i], int(scores[i])) for i in top_idx]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def render_quote(text: str, platform: str = "", rating=None):
    stars = ""
    if rating and not math.isnan(float(rating)):
        r = int(float(rating))
        stars = "⭐" * r + " "
    plat_tag = f"<span style='color:#64748b;font-size:0.75rem'>[{platform}]</span> " if platform else ""
    st.markdown(
        f"<div class='quote-card'>{stars}{plat_tag}{text[:350]}{'…' if len(text)>350 else ''}</div>",
        unsafe_allow_html=True
    )


def importance_bar(label: str, count: int, max_count: int, color: str = "#3b82f6"):
    pct = count / max_count if max_count > 0 else 0
    bar_html = f"""
    <div style="margin:4px 0">
      <div style="display:flex;justify-content:space-between;margin-bottom:2px">
        <span class="theme-bar-label">{label}</span>
        <span style="font-size:0.8rem;color:#94a3b8;font-weight:600">{count}</span>
      </div>
      <div style="background:#1e293b;border-radius:4px;height:8px">
        <div style="background:{color};width:{pct*100:.1f}%;height:8px;border-radius:4px"></div>
      </div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)


def platform_sentiment_donut(df_plat: pd.DataFrame, title: str):
    counts = df_plat["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8", "Unknown": "#475569"}
    fig = px.pie(counts, values="Count", names="Sentiment",
                 color="Sentiment", color_discrete_map=color_map,
                 hole=0.55, title=title)
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                      showlegend=True, height=260,
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#cbd5e1")
    return fig


def render_platform_tab(df: pd.DataFrame, platform_key: str, product_type: str, tab_key: str = ""):
    """Full deep-dive tab for a single platform. tab_key must be unique per call."""
    k = tab_key or platform_key   # unique prefix for all chart keys in this tab
    sub = df[df["platform"].str.lower().str.contains(platform_key, na=False)].copy()

    if sub.empty:
        st.info(f"No data available for **{platform_key}** yet.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", len(sub))
    pos_pct = round((sub["sentiment"] == "Positive").mean() * 100, 1)
    neg_pct = round((sub["sentiment"] == "Negative").mean() * 100, 1)
    c2.metric("Positive %", f"{pos_pct}%")
    c3.metric("Negative %", f"{neg_pct}%")
    if sub["rating"].notna().any():
        avg_r = sub["rating"].mean()
        c4.metric("Avg Rating", f"{avg_r:.1f} ⭐")
    else:
        c4.metric("Avg Rating", "N/A")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(platform_sentiment_donut(sub, "Sentiment Split"),
                        use_container_width=True, key=f"{k}_donut")

    with col_right:
        if sub["rating"].notna().sum() > 5:
            fig = px.histogram(
                sub.dropna(subset=["rating"]),
                x="rating", nbins=5,
                color_discrete_sequence=["#3b82f6"],
                title="Rating Distribution"
            )
            fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                              bargap=0.1)
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_rating_hist")
        else:
            time_sub = sub.dropna(subset=["date"]).copy()
            if not time_sub.empty:
                time_sub["month"] = time_sub["date"].dt.to_period("M").astype(str)
                monthly = time_sub.groupby("month").size().reset_index(name="count")
                fig = px.bar(monthly, x="month", y="count",
                             color_discrete_sequence=["#3b82f6"], title="Volume Over Time")
                fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10),
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
                st.plotly_chart(fig, use_container_width=True, key=f"{k}_vol_time")

    # ── Theme Analysis ────────────────────────────────────────────────────────
    st.subheader("🎯 Theme Frequency (What users actually talk about)")
    patterns = THEME_PATTERNS.get(product_type, {})
    if patterns:
        theme_counts = {}
        for theme, pat in patterns.items():
            theme_counts[theme] = int(
                sub["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False).sum()
            )
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=["Theme", "Mentions"])
        theme_df = theme_df[theme_df["Mentions"] > 0].sort_values("Mentions", ascending=False)

        if not theme_df.empty:
            max_m = theme_df["Mentions"].max()
            colors = ["#3b82f6","#6366f1","#8b5cf6","#a855f7","#ec4899","#f43f5e","#f97316","#eab308","#22c55e","#14b8a6"]
            for i, row in theme_df.iterrows():
                importance_bar(row["Theme"], row["Mentions"], max_m, colors[i % len(colors)])

    # ── Top Phrases ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔤 Most Frequent Phrases (TF-IDF weighted)")
    phrases = get_top_phrases(sub["review_text"], top_n=25)
    if phrases:
        phrase_df = pd.DataFrame(phrases, columns=["Phrase", "Score"])
        fig = px.bar(
            phrase_df.head(20), x="Score", y="Phrase", orientation="h",
            color="Score", color_continuous_scale="Blues"
        )
        fig.update_layout(
            height=500, margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{k}_phrases")

    # ── Sentiment breakdown per theme ─────────────────────────────────────────
    if patterns:
        st.subheader("😊 / 😠 Sentiment by Theme")
        rows = []
        for theme, pat in patterns.items():
            mask = sub["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
            theme_sub = sub[mask]
            if len(theme_sub) < 2:
                continue
            pos = (theme_sub["sentiment"] == "Positive").sum()
            neg = (theme_sub["sentiment"] == "Negative").sum()
            neu = (theme_sub["sentiment"] == "Neutral").sum()
            rows.append({"Theme": theme, "Positive": int(pos), "Negative": int(neg), "Neutral": int(neu)})

        if rows:
            sent_theme_df = pd.DataFrame(rows).sort_values("Negative", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Positive", x=sent_theme_df["Theme"], y=sent_theme_df["Positive"], marker_color="#22c55e"))
            fig.add_trace(go.Bar(name="Negative", x=sent_theme_df["Theme"], y=sent_theme_df["Negative"], marker_color="#ef4444"))
            fig.add_trace(go.Bar(name="Neutral",  x=sent_theme_df["Theme"], y=sent_theme_df["Neutral"],  marker_color="#64748b"))
            fig.update_layout(
                barmode="stack", height=350,
                margin=dict(t=20, b=60, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                xaxis_tickangle=-30
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{k}_sent_theme")

    # ── Sample quotes ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Representative Reviews")
    col_p, col_n = st.columns(2)
    with col_p:
        st.markdown("**✅ Positive highlights**")
        pos_reviews = sub[sub["sentiment"] == "Positive"].sort_values("sentiment_score", ascending=False).head(6)
        for _, r in pos_reviews.iterrows():
            render_quote(r["review_text"], r.get("platform",""), r.get("rating", None))
    with col_n:
        st.markdown("**❌ Negative highlights (pain points)**")
        neg_reviews = sub[sub["sentiment"] == "Negative"].sort_values("sentiment_score").head(6)
        for _, r in neg_reviews.iterrows():
            render_quote(r["review_text"], r.get("platform",""), r.get("rating", None))


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCT INTELLIGENCE PANEL
# ══════════════════════════════════════════════════════════════════════════════
def extract_phrases_for_sentiment(df_sub: pd.DataFrame, sentiment: str, top_n: int = 6) -> list[str]:
    """Return top TF-IDF phrases from reviews of a given sentiment."""
    texts = df_sub[df_sub["sentiment"] == sentiment]["review_text"]
    phrases = get_top_phrases(texts, top_n=top_n, ngram_range=(1, 3))
    return [p for p, _ in phrases]


def extract_feature_requests(df: pd.DataFrame, use_llm: bool = False) -> list[tuple[str, int]]:
    """Find user feature-request sentences. Excludes company/support responses and damage complaints."""
    if use_llm and "is_feature_request" in df.columns:
        req_df = df[df["is_feature_request"] == True]
        results = []
        for _, row in req_df.head(6).iterrows():
            text_snippet = row.get("action_item") or row.get("specific_issue") or str(row.get("review_text", ""))[:120]
            if pd.isna(text_snippet): text_snippet = str(row.get("review_text", ""))[:120]
            results.append((text_snippet, 1))
        return results

    req_pat = re.compile(
        r"(wish\s+(it|there|they|you|the)\s+(had|would|could|support)|"
        r"would\s+(love|like)\s+(to\s+(have|see|get)|a\s+\w+|an?\s+\w+|if\s+they)|"
        r"please\s+add\b|feature\s+request|"
        r"need\s+(a|an|the)\s+\w+\s+(feature|option|way|ability|support)|"
        r"should\s+(add|include|support|have\s+(a|an))\s+\w+|"
        r"hoping\s+for|add\s+(support\s+for|the\s+ability|an?\s+option)|"
        r"it\s+would\s+be\s+(great|nice|helpful|awesome)\s+if|"
        r"missing\s+(a\s+)?(feature|option|way|ability|dark\s*mode|support\s+for)|"
        r"would\s+be\s+(better|great|perfect)\s+(with|if\s+(it|they|there\s+was)))",
        re.IGNORECASE
    )
    # Exclude: company/support responses, damage-missing-item complaints, generic negatives
    suppress_pat = re.compile(
        r"(we\s+would\s+(like|love|be)|please\s+provide|kindly\s+(confirm|share|send)|"
        r"our\s+team|do\s+let\s+us\s+know|reach\s+out\s+to\s+us|"
        r"dm\s+(us|me)|direct\s+message|escalat|investigate\s+this|"
        r"items?\s+(are\s+)?(missing|missing\s+from)|things?\s+missing|"
        r"no\s+response\s+from|unable\s+to\s+solve|internal\s+coordination)",
        re.IGNORECASE
    )
    # Only use negative/neutral reviews — positive reviews rarely contain real requests
    request_df = df[df["sentiment"].isin(["Negative", "Neutral"])]
    snippets: Counter = Counter()
    for text in request_df["review_text"].dropna():
        for sentence in re.split(r'[.!?\n]', str(text)):
            s = sentence.strip()
            if req_pat.search(s) and not suppress_pat.search(s) and 25 < len(s) < 180:
                snippets[s[:120]] += 1
    return snippets.most_common(6)


def extract_churn_signals(df: pd.DataFrame, use_llm: bool = False) -> list[tuple[str, int]]:
    """Reviews where user switched away or threatened to cancel."""
    if use_llm and "is_churn_signal" in df.columns:
        churn_df = df[df["is_churn_signal"] == True]
        results = []
        for _, row in churn_df.head(4).iterrows():
            text_snippet = row.get("churn_reason") or row.get("specific_issue") or str(row.get("review_text", ""))[:120]
            if pd.isna(text_snippet): text_snippet = str(row.get("review_text", ""))[:120]
            results.append((text_snippet, 1))
        return results

    churn_pat = re.compile(
        r"(cancel(l?ed)?(\s+my)?\s+(sub|account|plan)|uninstall|switch(ed|ing)?\s+(to|away|back)|"
        r"going\s+back\s+to|moved?\s+to\s+\w+|left\s+(for|because)|quit\s+(using|the)|"
        r"deleted?\s+the?\s+app|looking\s+for\s+(an?\s+)?alternative|"
        r"(tradervue|edgewonk|tradezella|tradersync|myfxbook|excel|spreadsheet)\s+is\s+better)",
        re.IGNORECASE
    )
    matches = df[df["review_text"].str.contains(churn_pat, regex=True, na=False)]
    snippets: Counter = Counter()
    for text in matches["review_text"]:
        for sentence in re.split(r'[.!?\n]', str(text)):
            if churn_pat.search(sentence) and len(sentence.strip()) > 15:
                snippets[sentence.strip()[:120]] += 1
    return snippets.most_common(4)


def extract_competitor_mentions(df: pd.DataFrame, product_name: str) -> dict[str, int]:
    """Count how often each competitor is mentioned."""
    product_type = PRODUCTS.get(product_name, {}).get("product_type", "trading_journal")
    
    COMPETITOR_PATTERNS = {
        "trading_journal": {
            "Tradezella":    r"\btradezella\b",
            "TraderSync":    r"\btradersync\b",
            "TraderVue":     r"\btradervue\b",
            "Edgewonk":      r"\bedgewonk\b",
            "TradesViz":     r"\btradesviz\b",
            "Myfxbook":      r"\bmyfxbook\b",
            "Excel/Sheets":  r"\b(excel|google sheets?|spreadsheet)\b",
            "Stocktrak":     r"\bstocktrak\b",
        },
        "payments": {
            "Wise":          r"\bwise\b",
            "Payoneer":      r"\bpayoneer\b",
            "Razorpay":      r"\brazorpay\b",
            "PayPal":        r"\bpaypal\b",
            "Stripe":        r"\bstripe\b",
            "Skydo":         r"\bskydo\b",
        },
        "logistics": {
            "Porter":        r"\bporter\b",
            "Dunzo":         r"\bdunzo\b",
            "Shadowfax":     r"\bshadowfax\b",
            "WeFast":        r"\bwefast\b",
            "Borzo":         r"\bborzo\b",
            "Lalamove":      r"\blalamove\b",
        },
        "marketplace": {
            "Zepto":         r"\bzepto\b",
            "Blinkit":       r"\bblinkit\b",
            "Swiggy Instamart": r"\b(swiggy|instamart)\b",
            "BigBasket":     r"\b(bigbasket|bb\sdaily|bbdaily)\b",
            "Amazon Fresh":  r"\b(amazon fresh|amazon)\b",
            "Country Delight": r"\bcountry delight\b",
            "Milkbasket":    r"\bmilkbasket\b",
            "Supr Daily":    r"\bsupr daily\b",
            "FirstClub":     r"\bfirstclub\b",
        }
    }
    
    competitors = COMPETITOR_PATTERNS.get(product_type, {})
    results = {}
    for name, pat in competitors.items():
        if name.lower() == product_name.lower():
            continue
        cnt = int(df["review_text"].str.contains(pat, flags=re.IGNORECASE, regex=True, na=False).sum())
        if cnt > 0:
            results[name] = cnt
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def _card(bg: str, border: str, title_color: str, title: str, body_html: str) -> str:
    """Build a self-contained card as a single HTML string."""
    return f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
                padding:18px 20px;height:100%;box-sizing:border-box">
      <div style="color:{title_color};font-weight:700;font-size:0.95rem;
                  margin-bottom:12px;letter-spacing:0.02em">{title}</div>
      {body_html}
    </div>
    """

def _tag(text: str, bg: str, color: str) -> str:
    return (f"<span style='background:{bg};color:{color};border-radius:4px;"
            f"padding:2px 8px;margin:2px 2px 2px 0;display:inline-block;"
            f"font-size:0.76rem'>{text}</span>")

def _quote_row(text: str, border: str, bg: str, color: str, badge: str = "") -> str:
    badge_html = f"<span style='float:right;font-size:0.72rem;color:{border};opacity:.8'>{badge}</span>" if badge else ""
    return (f"<div style='margin:5px 0;padding:7px 10px;background:{bg};"
            f"border-left:3px solid {border};border-radius:4px;clear:both'>"
            f"{badge_html}"
            f"<span style='color:{color};font-size:0.81rem;font-style:italic'>&ldquo;{text}&rdquo;</span>"
            f"</div>")

def _bar_row(label: str, count: int, max_count: int, label_color: str,
             bar_color: str, bar_bg: str, meta: str = "") -> str:
    pct = int(count / max(max_count, 1) * 100)
    meta_html = f"<span style='color:{bar_color};font-size:0.75rem'>{meta}</span>" if meta else ""
    return f"""
    <div style='margin:7px 0'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px'>
        <span style='color:{label_color};font-size:0.84rem;font-weight:600'>{label}</span>
        {meta_html}
      </div>
      <div style='background:{bar_bg};border-radius:4px;height:6px'>
        <div style='background:{bar_color};width:{pct}%;height:6px;border-radius:4px'></div>
      </div>
    </div>"""


def render_intelligence_panel(df: pd.DataFrame, product_type: str, product_name: str):
    st.subheader("🧠 Product Intelligence")
    st.caption("Auto-extracted from all reviews — think PM · Growth · CS · Competitive")

    patterns = THEME_PATTERNS.get(product_type, {})

    # ── Pre-compute all data first (no st calls yet) ──────────────────────────
    theme_health = []
    for theme, pat in patterns.items():
        mask = df["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
        sub  = df[mask]
        if len(sub) < 2:
            continue
        pos   = int((sub["sentiment"] == "Positive").sum())
        neg   = int((sub["sentiment"] == "Negative").sum())
        score = (pos - neg) / max(len(sub), 1)
        theme_health.append({"theme": theme, "total": len(sub),
                              "pos": pos, "neg": neg, "score": score})

    theme_health.sort(key=lambda x: x["total"], reverse=True)
    strengths  = sorted([t for t in theme_health if t["score"] >  0.1],  key=lambda x: -x["score"])[:3]
    weaknesses = sorted([t for t in theme_health if t["score"] < -0.05], key=lambda x:  x["score"])[:3]
    # Fallback: if no clear strengths, use the least-negative themes
    relative_strengths = sorted(theme_health, key=lambda x: -x["score"])[:3] if not strengths else []

    praised    = extract_phrases_for_sentiment(df, "Positive", top_n=6)
    complained = extract_phrases_for_sentiment(df, "Negative", top_n=6)
    use_llm_data = product_name in ["Tradezella", "TraderSync", "FirstClub"]
    feat_reqs  = extract_feature_requests(df, use_llm=use_llm_data)
    churn      = extract_churn_signals(df, use_llm=use_llm_data)
    comp_map   = extract_competitor_mentions(df, product_name)

    # ── ROW 1: Strengths | Weaknesses ────────────────────────────────────────
    c1, c2 = st.columns(2)

    # — Strengths (or relative best if no clear positives) —
    if strengths:
        title_str = "💪 Strengths — What users love"
        rows = "".join(
            f"<div style='margin:5px 0'>"
            f"<span style='color:#86efac;font-weight:600'>{t['theme']}</span> "
            f"<span style='color:#4ade80;font-size:0.78rem'>· {round(t['pos']/max(t['total'],1)*100)}% positive · {t['total']} mentions</span>"
            f"</div>"
            for t in strengths
        )
        tags = "".join(_tag(p, "#065f46", "#a7f3d0") for p in praised) if praised else ""
        tag_section = f"<div style='margin-top:10px;color:#6ee7b7;font-size:0.75rem;margin-bottom:4px'>Top praised phrases</div>{tags}" if tags else ""
        body = rows + tag_section
    elif relative_strengths:
        title_str = "📊 Best Performing Areas — Least negative themes"
        rows = "".join(
            f"<div style='margin:5px 0'>"
            f"<span style='color:#86efac;font-weight:600'>{t['theme']}</span> "
            f"<span style='color:#4ade80;font-size:0.78rem'>· {round(t['pos']/max(t['total'],1)*100)}% positive · {t['total']} mentions</span>"
            f"</div>"
            for t in relative_strengths
        )
        note = "<div style='margin-top:8px;color:#4ade80;font-size:0.75rem;opacity:0.7'>⚠️ Overall sentiment is negative — these are the least-complained areas</div>"
        body = rows + note
    else:
        title_str = "💪 Strengths — What users love"
        body = "<div style='color:#6ee7b7;font-size:0.84rem'>Not enough data yet.</div>"

    c1.markdown(_card("#052e16", "#166534", "#4ade80", title_str, body),
                unsafe_allow_html=True)

    # — Weaknesses —
    if weaknesses:
        rows = "".join(
            f"<div style='margin:5px 0'>"
            f"<span style='color:#fca5a5;font-weight:600'>{t['theme']}</span> "
            f"<span style='color:#f87171;font-size:0.78rem'>· {round(t['neg']/max(t['total'],1)*100)}% negative · {t['neg']} complaints</span>"
            f"</div>"
            for t in weaknesses
        )
        tags = "".join(_tag(p, "#450a0a", "#fca5a5") for p in complained) if complained else ""
        tag_section = f"<div style='margin-top:10px;color:#fca5a5;font-size:0.75rem;margin-bottom:4px'>Top complaint phrases</div>{tags}" if tags else ""
        body = rows + tag_section
    else:
        body = "<div style='color:#fca5a5;font-size:0.84rem'>No dominant weakness signal found.</div>"

    c2.markdown(_card("#2d0a0a", "#7f1d1d", "#f87171", "⚠️ Weaknesses — Where users are frustrated", body),
                unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── ROW 2: Feature Requests | Competitive Landscape ──────────────────────
    c3, c4 = st.columns(2)

    # — Feature Requests —
    if feat_reqs:
        body = "".join(
            _quote_row(s[:110], "#d97706", "#292524", "#fde68a",
                       f"×{freq}" if freq > 1 else "")
            for s, freq in feat_reqs
        )
    else:
        body = "<div style='color:#fde68a;font-size:0.84rem'>No explicit feature requests detected.</div>"

    c3.markdown(_card("#1c1917", "#78350f", "#fbbf24", "🆕 Feature Requests — What users want built", body),
                unsafe_allow_html=True)

    # — Competitive Landscape — with positive/negative context per competitor —
    if comp_map:
        max_c = max(comp_map.values())
        rows = []
        for comp, cnt in list(comp_map.items())[:7]:
            # Count positive vs negative mentions of this competitor
            comp_pat = re.compile(re.escape(comp), re.IGNORECASE)
            comp_mask = df["review_text"].str.contains(comp_pat, regex=True, na=False)
            comp_sub  = df[comp_mask]
            c_pos = (comp_sub["sentiment"] == "Positive").sum()
            c_neg = (comp_sub["sentiment"] == "Negative").sum()
            # Context badge: is competitor praised or complained about alongside product?
            if c_pos > c_neg:
                ctx = f"<span style='color:#4ade80;font-size:0.72rem'>▲ {c_pos} positive</span>"
            elif c_neg > c_pos:
                ctx = f"<span style='color:#f87171;font-size:0.72rem'>▼ {c_neg} negative</span>"
            else:
                ctx = f"<span style='color:#94a3b8;font-size:0.72rem'>{cnt} neutral</span>"
            pct = int(cnt / max_c * 100)
            rows.append(
                f"<div style='margin:7px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px'>"
                f"<span style='color:#bfdbfe;font-size:0.84rem;font-weight:600'>{comp}</span>{ctx}</div>"
                f"<div style='background:#1e3a8a;border-radius:4px;height:5px'>"
                f"<div style='background:#3b82f6;width:{pct}%;height:5px;border-radius:4px'></div></div>"
                f"</div>"
            )
        body = "".join(rows)
    else:
        body = ("<div style='color:#93c5fd;font-size:0.84rem'>"
                "No competitor names found in reviews.<br>"
                "<span style='opacity:0.6;font-size:0.78rem'>Users may not be naming alternatives directly.</span></div>")

    c4.markdown(_card("#0c1445", "#1e3a8a", "#93c5fd", "🥊 Competitive Landscape — Who users compare to", body),
                unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── ROW 3: Churn Signals | Build a Competitor ────────────────────────────
    c5, c6 = st.columns(2)

    # — Churn Signals —
    if churn:
        body = "".join(_quote_row(s[:110], "#7c3aed", "#2e1065", "#e9d5ff") for s, _ in churn)
    else:
        body = "<div style='color:#e9d5ff;font-size:0.84rem'>No strong churn signals detected.</div>"

    c5.markdown(_card("#1a0a2e", "#581c87", "#c084fc", "🚨 Churn Signals — Users switching or cancelling", body),
                unsafe_allow_html=True)

    # — If You're Building a Competitor —
    gaps = []
    for t in weaknesses[:3]:
        neg_phrases = extract_phrases_for_sentiment(
            df[df["review_text"].str.lower().str.contains(
                patterns.get(t["theme"], "x^"), flags=re.IGNORECASE, regex=True, na=False)],
            "Negative", top_n=1
        )
        hint = f' — e.g. "{neg_phrases[0][:60]}"' if neg_phrases else ""
        gaps.append(f"<b>{t['theme']}</b> has {t['neg']} complaints{hint}")

    for snippet, _ in (feat_reqs or [])[:2]:
        gaps.append(f"Unbuilt: &ldquo;{snippet[:80]}&rdquo;")

    if comp_map:
        top_comp = next(iter(comp_map))
        top_cnt  = comp_map[top_comp]
        gaps.append(f"<b>{top_comp}</b> is mentioned {top_cnt}× — users are already comparing")

    if gaps:
        body = "".join(
            f"<div style='margin:5px 0;padding:7px 11px;background:#0f2744;"
            f"border-left:3px solid #0ea5e9;border-radius:4px'>"
            f"<span style='color:#bae6fd;font-size:0.81rem'>{g}</span></div>"
            for g in gaps[:5]
        )
    else:
        body = "<div style='color:#bae6fd;font-size:0.84rem'>Not enough signal yet.</div>"

    c6.markdown(_card("#0a1628", "#0e4d91", "#38bdf8",
                      "🏗️ If You're Building a Competitor — Exploit these gaps", body),
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Product selector
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Customer Intelligence")
    st.divider()

    product_options = list(PRODUCTS.keys())
    product_name = st.selectbox(
        "Select Product",
        product_options,
        index=product_options.index("FirstClub"),
        format_func=lambda p: f"{PRODUCTS[p]['emoji']} {p}"
    )
    cfg = PRODUCTS[product_name]

    st.caption(cfg["description"])
    st.divider()

    # Date filter
    st.markdown("### 📅 Date Filter")
    date_filter_on = st.checkbox("Enable date filter", value=False)

    st.divider()
    st.markdown("### ℹ️ Data Sources")
    for path_label, path_val in [("Clean CSV", cfg["clean_csv"]), ("Raw CSV", cfg["raw_csv"])]:
        exists = os.path.exists(path_val)
        icon = "✅" if exists else "❌"
        st.caption(f"{icon} {path_label}")

    st.divider()
    if not _VADER_OK:
        st.warning("Install vaderSentiment for sentiment analysis:\n`pip install vaderSentiment`")
    if not _TFIDF_OK:
        st.warning("Install scikit-learn for TF-IDF:\n`pip install scikit-learn`")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"Loading {product_name} data…"):
    df_raw = load_and_analyse(product_name)

if df_raw.empty:
    st.error(
        f"No data found for **{product_name}**.\n\n"
        f"Expected:\n- `{cfg['clean_csv']}`\n- `{cfg['raw_csv']}`\n\n"
        "Run the scraper pipeline first."
    )
    st.stop()

# Apply date filter
df = df_raw.copy()
if date_filter_on and df["date"].notna().any():
    min_d = df["date"].min().date()
    max_d = df["date"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=[min_d, max_d],
                                        min_value=min_d, max_value=max_d)
    if len(date_range) == 2:
        df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title(f"{cfg['emoji']} {product_name} — Customer Intelligence")
st.caption(
    f"{cfg['description']} · **{len(df):,} reviews** across "
    f"**{df['platform'].nunique()} platforms** · "
    f"Updated: {datetime.now().strftime('%d %b %Y')}"
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_themes, tab_platforms, tab_trustpilot, tab_twitter, tab_reddit, tab_playstore, tab_gmaps, tab_appstore, tab_trends, tab_raw, tab_files = st.tabs([
    "📈 Overview",
    "🎯 Deep Themes",
    "📊 Platform Breakdown",
    "⭐ Trustpilot",
    "🐦 Twitter / X",
    "🤝 Reddit",
    "📱 Play Store",
    "🗺️ Google Maps",
    "🍎 App Store",
    "📅 Trends",
    "📁 Raw Data",
    "📂 File Browser",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # ── Top KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Reviews", f"{len(df):,}")
    k2.metric("Platforms", df["platform"].nunique())
    pos_pct = round((df["sentiment"] == "Positive").mean() * 100, 1)
    neg_pct = round((df["sentiment"] == "Negative").mean() * 100, 1)
    k3.metric("Positive", f"{pos_pct}%")
    k4.metric("Negative", f"{neg_pct}%")
    if df["rating"].notna().any():
        k5.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
    else:
        k5.metric("Net Sentiment", f"{pos_pct - neg_pct:+.1f}%")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Distribution")
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        color_map = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8", "Unknown": "#475569"}
        fig = px.pie(sent_counts, values="Count", names="Sentiment",
                     color="Sentiment", color_discrete_map=color_map, hole=0.5)
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(fig, use_container_width=True, key="ov_sentiment_donut")

    with c2:
        st.subheader("Reviews by Platform")
        plat_counts = df["platform"].value_counts().reset_index()
        plat_counts.columns = ["Platform", "Count"]
        # sentiment breakdown
        plat_sent = df.groupby(["platform","sentiment"]).size().reset_index(name="Count")
        fig = px.bar(plat_sent, x="platform", y="Count", color="sentiment",
                     color_discrete_map=color_map, barmode="stack")
        fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          legend_title="Sentiment")
        st.plotly_chart(fig, use_container_width=True, key="ov_platform_bar")

    # ── Overall theme importance ───────────────────────────────────────────────
    st.subheader("🎯 Theme Importance Map (frequency = importance)")
    patterns = THEME_PATTERNS.get(cfg["product_type"], {})
    if patterns:
        theme_counts = {
            theme: int(df["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False).sum())
            for theme, pat in patterns.items()
        }
        theme_df = pd.DataFrame(list(theme_counts.items()), columns=["Theme", "Mentions"])
        theme_df = theme_df[theme_df["Mentions"] > 0].sort_values("Mentions", ascending=False)

        col_bar, col_radar = st.columns([1.2, 1])
        with col_bar:
            fig = px.bar(theme_df, x="Mentions", y="Theme", orientation="h",
                         color="Mentions", color_continuous_scale="Blues")
            fig.update_layout(
                height=350, margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1"
            )
            st.plotly_chart(fig, use_container_width=True, key="ov_theme_bar")

        with col_radar:
            cats  = theme_df["Theme"].tolist()
            vals  = theme_df["Mentions"].tolist()
            if len(cats) >= 3:
                fig2 = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.2)",
                    line_color="#3b82f6"
                ))
                fig2.update_layout(
                    polar=dict(radialaxis=dict(visible=True, color="#64748b")),
                    height=350, margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1"
                )
                st.plotly_chart(fig2, use_container_width=True, key="ov_theme_radar")

    # ── Product Intelligence Panel ────────────────────────────────────────────
    st.divider()
    render_intelligence_panel(df, cfg["product_type"], product_name)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP THEMES
# ══════════════════════════════════════════════════════════════════════════════
with tab_themes:
    st.subheader("🎯 Deep Theme Intelligence — A to Z")
    st.caption("Each theme: frequency across all reviews, positive/negative split, top phrases, representative quotes")

    patterns = THEME_PATTERNS.get(cfg["product_type"], {})
    if not patterns:
        st.info("No theme patterns configured for this product type.")
    else:
        # Build theme stats
        theme_stats = []
        for theme, pat in patterns.items():
            mask = df["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
            theme_df_sub = df[mask]
            if len(theme_df_sub) == 0:
                continue
            pos = (theme_df_sub["sentiment"] == "Positive").sum()
            neg = (theme_df_sub["sentiment"] == "Negative").sum()
            neu = (theme_df_sub["sentiment"] == "Neutral").sum()
            avg_score = theme_df_sub["sentiment_score"].mean() if "sentiment_score" in theme_df_sub.columns else 0
            theme_stats.append({
                "theme": theme, "total": len(theme_df_sub),
                "positive": int(pos), "negative": int(neg), "neutral": int(neu),
                "avg_sentiment": round(float(avg_score), 3),
                "mask": mask,
            })

        theme_stats.sort(key=lambda x: x["total"], reverse=True)

        # Theme selector
        theme_names = [t["theme"] for t in theme_stats]
        selected_theme = st.selectbox("Select Theme to Deep-Dive", theme_names,
                                       format_func=lambda t: f"{t} ({next(s['total'] for s in theme_stats if s['theme']==t)} mentions)")

        st.divider()

        # Show all themes overview first
        st.subheader("📊 All Themes — Weighted Overview")
        max_total = theme_stats[0]["total"] if theme_stats else 1
        for ts in theme_stats:
            col_label, col_bar_area = st.columns([0.25, 0.75])
            with col_label:
                health = "🟢" if ts["positive"] > ts["negative"] else ("🔴" if ts["negative"] > ts["positive"] else "🟡")
                st.markdown(f"{health} **{ts['theme']}**")
                st.caption(f"{ts['total']} mentions · +{ts['positive']} / -{ts['negative']}")
            with col_bar_area:
                bar_data = pd.DataFrame({
                    "type": ["Positive", "Negative", "Neutral"],
                    "count": [ts["positive"], ts["negative"], ts["neutral"]]
                })
                fig = px.bar(bar_data, x="count", y="type", orientation="h",
                             color="type",
                             color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#64748b"})
                fig.update_layout(height=80, margin=dict(t=0,b=0,l=0,r=0),
                                  showlegend=False, yaxis_title="",
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                                  xaxis_title="")
                st.plotly_chart(fig, use_container_width=True,
                                key=f"dt_theme_bar_{ts['theme'].replace(' ','_')}")

        st.divider()

        # ── Deep dive into selected theme ─────────────────────────────────────
        selected_stat = next((s for s in theme_stats if s["theme"] == selected_theme), None)
        if selected_stat:
            st.subheader(f"🔍 Deep Dive: {selected_theme}")
            theme_sub = df[selected_stat["mask"]].copy()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Mentions", selected_stat["total"])
            m2.metric("Positive",       selected_stat["positive"])
            m3.metric("Negative",       selected_stat["negative"])
            sentiment_label = "😊 Positive" if selected_stat["avg_sentiment"] > 0.05 else ("😠 Negative" if selected_stat["avg_sentiment"] < -0.05 else "😐 Neutral")
            m4.metric("Overall Tone", sentiment_label)

            col_phrases, col_plat = st.columns(2)

            with col_phrases:
                st.markdown("**🔤 Key Phrases**")
                phrases = get_top_phrases(theme_sub["review_text"], top_n=15, ngram_range=(1,3))
                if phrases:
                    max_p = phrases[0][1]
                    for phrase, score in phrases:
                        importance_bar(phrase, score, max_p)

            with col_plat:
                st.markdown("**📊 Platform Distribution**")
                plat_dist = theme_sub["platform"].value_counts().reset_index()
                plat_dist.columns = ["Platform", "Count"]
                fig = px.pie(plat_dist, values="Count", names="Platform",
                             hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
                st.plotly_chart(fig, use_container_width=True,
                                key=f"dt_plat_dist_{selected_theme.replace(' ','_')}")

            # Volume trend
            if theme_sub["date"].notna().any():
                st.markdown("**📅 Mention Trend Over Time**")
                trend = theme_sub.dropna(subset=["date"]).copy()
                trend["month"] = trend["date"].dt.to_period("M").astype(str)
                trend_grouped = trend.groupby(["month","sentiment"]).size().reset_index(name="count")
                fig = px.bar(trend_grouped, x="month", y="count", color="sentiment",
                             color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#64748b"},
                             barmode="stack")
                fig.update_layout(height=220, margin=dict(t=10,b=10,l=10,r=10),
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                                  xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"dt_trend_{selected_theme.replace(' ','_')}")

            # Quotes
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("**✅ Positive quotes**")
                for _, r in theme_sub[theme_sub["sentiment"]=="Positive"].head(5).iterrows():
                    render_quote(r["review_text"], r.get("platform",""), r.get("rating"))
            with col_neg:
                st.markdown("**❌ Negative quotes**")
                for _, r in theme_sub[theme_sub["sentiment"]=="Negative"].head(5).iterrows():
                    render_quote(r["review_text"], r.get("platform",""), r.get("rating"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PLATFORM BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab_platforms:
    st.subheader("📊 Cross-Platform Comparison")

    platforms_in_data = df["platform"].value_counts()

    # Summary table
    rows = []
    for plat, cnt in platforms_in_data.items():
        sub = df[df["platform"] == plat]
        pos = round((sub["sentiment"] == "Positive").mean() * 100, 1)
        neg = round((sub["sentiment"] == "Negative").mean() * 100, 1)
        avg_r = f"{sub['rating'].mean():.1f}" if sub["rating"].notna().any() else "—"
        rows.append({"Platform": plat, "Reviews": int(cnt), "Positive %": pos, "Negative %": neg, "Avg Rating": avg_r})

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # Heatmap: theme × platform
    patterns = THEME_PATTERNS.get(cfg["product_type"], {})
    if patterns:
        st.subheader("🗺️ Theme × Platform Heatmap")
        heat_data = {}
        for plat in platforms_in_data.index:
            plat_sub = df[df["platform"] == plat]
            heat_data[plat] = {
                theme: int(plat_sub["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False).sum())
                for theme, pat in patterns.items()
            }

        heat_df = pd.DataFrame(heat_data).fillna(0)
        heat_df = heat_df.loc[heat_df.sum(axis=1) > 0]

        if not heat_df.empty:
            fig = px.imshow(
                heat_df,
                color_continuous_scale="Blues",
                aspect="auto",
                title="Theme mentions per platform"
            )
            fig.update_layout(
                height=400, margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1"
            )
            st.plotly_chart(fig, use_container_width=True, key="pb_heatmap")

    # Sentiment × Platform scatter
    st.subheader("📈 Sentiment Score Distribution by Platform")
    if "sentiment_score" in df.columns:
        fig = px.box(
            df[df["sentiment_score"].notna()],
            x="platform", y="sentiment_score",
            color="platform",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=350, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="pb_box")


# ══════════════════════════════════════════════════════════════════════════════
# TABS 4-7 — PER PLATFORM DEEP DIVES
# ══════════════════════════════════════════════════════════════════════════════
with tab_trustpilot:
    st.subheader("⭐ Trustpilot Deep Dive")
    render_platform_tab(df, "trustpilot", cfg["product_type"], tab_key="tp")

with tab_twitter:
    st.subheader("🐦 Twitter / X Deep Dive")
    render_platform_tab(df, "twitter", cfg["product_type"], tab_key="tw")

with tab_reddit:
    st.subheader("🤝 Reddit Deep Dive")
    render_platform_tab(df, "reddit", cfg["product_type"], tab_key="rd")

with tab_playstore:
    st.subheader("📱 Play Store Deep Dive")
    render_platform_tab(df, "playstore", cfg["product_type"], tab_key="ps")

with tab_gmaps:
    st.subheader("🗺️ Google Maps Deep Dive")
    render_platform_tab(df, "google_maps", cfg["product_type"], tab_key="gm")

with tab_appstore:
    st.subheader("🍎 App Store Deep Dive")
    render_platform_tab(df, "app_store", cfg["product_type"], tab_key="as")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.subheader("📅 Temporal Trends")

    time_df = df.dropna(subset=["date"]).copy()
    if time_df.empty:
        st.info("No date data available.")
    else:
        time_df["month"] = time_df["date"].dt.to_period("M").astype(str)
        time_df["week"]  = time_df["date"].dt.to_period("W").astype(str)

        granularity = st.radio("Granularity", ["Monthly", "Weekly"], horizontal=True)
        period_col = "month" if granularity == "Monthly" else "week"

        # Volume over time by platform
        vol_plat = time_df.groupby([period_col, "platform"]).size().reset_index(name="count")
        fig = px.line(vol_plat, x=period_col, y="count", color="platform",
                      title="Review Volume Over Time (by Platform)",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350, margin=dict(t=40,b=20,l=20,r=20),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True, key="tr_vol_platform")

        # Sentiment trend
        sent_trend = time_df.groupby([period_col, "sentiment"]).size().reset_index(name="count")
        fig2 = px.area(sent_trend, x=period_col, y="count", color="sentiment",
                       title="Sentiment Volume Over Time",
                       color_discrete_map={"Positive":"#22c55e","Negative":"#ef4444","Neutral":"#64748b"})
        fig2.update_layout(height=300, margin=dict(t=40,b=20,l=20,r=20),
                           paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True, key="tr_sent_area")

        # Rolling average sentiment score
        if "sentiment_score" in time_df.columns:
            st.subheader("📉 Sentiment Score Rolling Average")
            monthly_score = (
                time_df.groupby(period_col)["sentiment_score"]
                .mean()
                .reset_index()
                .rename(columns={"sentiment_score": "avg_score"})
            )
            fig3 = px.line(monthly_score, x=period_col, y="avg_score",
                           title="Average Sentiment Score Over Time")
            fig3.add_hline(y=0, line_dash="dash", line_color="#64748b")
            fig3.update_layout(height=280, margin=dict(t=40,b=20,l=20,r=20),
                               paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                               xaxis_tickangle=-30)
            st.plotly_chart(fig3, use_container_width=True, key="tr_rolling_avg")

        # Theme frequency over time
        st.subheader("🎯 Theme Frequency Over Time")
        patterns = THEME_PATTERNS.get(cfg["product_type"], {})
        if patterns:
            selected_themes_trend = st.multiselect(
                "Select themes",
                list(patterns.keys()),
                default=list(patterns.keys())[:3]
            )
            if selected_themes_trend:
                trend_rows = []
                for theme in selected_themes_trend:
                    pat = patterns[theme]
                    mask = time_df["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
                    theme_time = time_df[mask].groupby(period_col).size().reset_index(name="count")
                    theme_time["theme"] = theme
                    trend_rows.append(theme_time)

                if trend_rows:
                    trend_combined = pd.concat(trend_rows)
                    fig4 = px.line(trend_combined, x=period_col, y="count", color="theme",
                                   color_discrete_sequence=px.colors.qualitative.Set1)
                    fig4.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20),
                                       paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                                       xaxis_tickangle=-30)
                    st.plotly_chart(fig4, use_container_width=True, key="tr_theme_freq")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — RAW DATA & DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("📁 Raw Data Explorer")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        plat_filter = st.multiselect("Platform", df["platform"].unique().tolist(),
                                      default=df["platform"].unique().tolist())
    with fc2:
        sent_filter = st.multiselect("Sentiment", df["sentiment"].unique().tolist(),
                                      default=df["sentiment"].unique().tolist())
    with fc3:
        search_text = st.text_input("Search review text", "")

    filtered_raw = df[df["platform"].isin(plat_filter) & df["sentiment"].isin(sent_filter)]
    if search_text:
        filtered_raw = filtered_raw[filtered_raw["review_text"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Showing {len(filtered_raw):,} of {len(df):,} reviews")

    display_cols = [c for c in ["date","platform","rating","sentiment","sentiment_score","review_text","author","source_url"]
                    if c in filtered_raw.columns]
    st.dataframe(
        filtered_raw[display_cols].sort_values("date", ascending=False, na_position="last"),
        use_container_width=True,
        height=450,
        column_config={
            "review_text":     st.column_config.TextColumn("Review", width="large"),
            "sentiment_score": st.column_config.NumberColumn("Score", format="%.2f"),
            "source_url":      st.column_config.LinkColumn("Source"),
        }
    )

    st.divider()
    st.subheader("⬇️ Download Data")

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_all = filtered_raw.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Reviews (CSV)",
            data=csv_all,
            file_name=f"{product_name.lower().replace(' ','_')}_reviews_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl2:
        # Theme summary CSV
        patterns = THEME_PATTERNS.get(cfg["product_type"], {})
        if patterns:
            theme_rows = []
            for theme, pat in patterns.items():
                mask = df["review_text"].str.lower().str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
                ts = df[mask]
                theme_rows.append({
                    "Theme": theme,
                    "Total_Mentions": len(ts),
                    "Positive": (ts["sentiment"]=="Positive").sum(),
                    "Negative": (ts["sentiment"]=="Negative").sum(),
                    "Neutral":  (ts["sentiment"]=="Neutral").sum(),
                    "Avg_Sentiment_Score": round(ts["sentiment_score"].mean(), 3) if "sentiment_score" in ts.columns else "",
                })
            theme_summary_csv = pd.DataFrame(theme_rows).to_csv(index=False)
            st.download_button(
                "📥 Download Theme Summary (CSV)",
                data=theme_summary_csv,
                file_name=f"{product_name.lower().replace(' ','_')}_theme_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with dl3:
        # Full unfiltered
        csv_full = df.to_csv(index=False)
        st.download_button(
            "📥 Download All Reviews (CSV)",
            data=csv_full,
            file_name=f"{product_name.lower().replace(' ','_')}_all_reviews.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — FILE BROWSER
# ══════════════════════════════════════════════════════════════════════════════
with tab_files:
    import glob
    from pathlib import Path

    st.subheader("📂 File Browser — All Project CSVs")
    st.caption("Click any file to view it inline. All files are read-only.")

    # ── Discover all CSVs in the project ──────────────────────────────────────
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    all_csvs  = sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True))

    if not all_csvs:
        st.info("No CSV files found in the `data/` directory.")
    else:
        # Build a display table for the file tree
        file_meta = []
        for fpath in all_csvs:
            p = Path(fpath)
            rel = p.relative_to(Path(data_root).parent)   # relative to project root
            size_kb = round(p.stat().st_size / 1024, 1)
            modified = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            # Peek at row count without loading full file
            try:
                with open(fpath) as f:
                    row_count = sum(1 for _ in f) - 1   # subtract header
            except Exception:
                row_count = -1
            file_meta.append({
                "File": str(rel),
                "Rows": row_count,
                "Size (KB)": size_kb,
                "Modified": modified,
                "_path": fpath,
            })

        meta_df = pd.DataFrame(file_meta)

        # ── File tree ─────────────────────────────────────────────────────────
        st.markdown("### 🗂️ Available Files")

        # Group by product folder
        product_folders = {}
        for entry in file_meta:
            parts = Path(entry["File"]).parts   # e.g. ('data','tradersync','raw','raw_reviews.csv')
            folder = "/".join(parts[:-1]) if len(parts) > 1 else "root"
            product_folders.setdefault(folder, []).append(entry)

        for folder, entries in sorted(product_folders.items()):
            with st.expander(f"📁 {folder}  ({len(entries)} file{'s' if len(entries)>1 else ''})", expanded=False):
                for entry in entries:
                    fname = Path(entry["File"]).name
                    col_name, col_rows, col_size, col_btn = st.columns([3, 1, 1, 1])
                    col_name.markdown(f"**{fname}**")
                    col_rows.caption(f"{entry['Rows']:,} rows")
                    col_size.caption(f"{entry['Size (KB)']} KB")
                    # Use session_state to track which file is open
                    btn_key = f"view_{entry['_path']}"
                    if col_btn.button("👁 View", key=btn_key, use_container_width=True):
                        st.session_state["file_viewer_path"] = entry["_path"]

        st.divider()

        # ── File viewer ───────────────────────────────────────────────────────
        st.markdown("### 🔍 File Viewer")

        # Dropdown as alternative to button clicks
        file_labels = [str(Path(f).relative_to(Path(data_root).parent)) for f in all_csvs]
        default_idx = 0
        if "file_viewer_path" in st.session_state:
            try:
                default_idx = all_csvs.index(st.session_state["file_viewer_path"])
            except ValueError:
                default_idx = 0

        selected_label = st.selectbox(
            "Select file to view",
            file_labels,
            index=default_idx,
            key="file_selector_dropdown"
        )
        selected_path = all_csvs[file_labels.index(selected_label)]
        st.session_state["file_viewer_path"] = selected_path

        # Load and display
        try:
            fview = pd.read_csv(selected_path)

            # ── Metadata strip ────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows",    f"{len(fview):,}")
            m2.metric("Columns", len(fview.columns))
            m3.metric("Size",    f"{round(Path(selected_path).stat().st_size/1024,1)} KB")
            m4.metric("Modified", datetime.fromtimestamp(Path(selected_path).stat().st_mtime).strftime("%d %b %Y"))

            # ── Column summary ────────────────────────────────────────────────
            with st.expander("📋 Column Summary", expanded=False):
                col_info = []
                for col in fview.columns:
                    non_null = fview[col].notna().sum()
                    dtype    = str(fview[col].dtype)
                    sample   = str(fview[col].dropna().iloc[0])[:60] if non_null > 0 else "—"
                    col_info.append({"Column": col, "Type": dtype, "Non-null": non_null, "Sample Value": sample})
                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

            # ── Inline search & filter ────────────────────────────────────────
            sv1, sv2 = st.columns([3, 1])
            with sv1:
                fv_search = st.text_input("🔎 Search any column", "", key="fv_search")
            with sv2:
                fv_rows = st.selectbox("Rows to show", [50, 100, 250, 500, "All"], index=1, key="fv_rows")

            fview_display = fview.copy()
            if fv_search:
                mask = fview_display.apply(
                    lambda col: col.astype(str).str.contains(fv_search, case=False, na=False)
                ).any(axis=1)
                fview_display = fview_display[mask]

            if fv_rows != "All":
                fview_display = fview_display.head(int(fv_rows))

            st.caption(f"Showing {len(fview_display):,} of {len(fview):,} rows · {len(fview.columns)} columns")

            # Smart column config: make source_url clickable, text cols wider
            col_cfg = {}
            for col in fview_display.columns:
                if "url" in col.lower() or "link" in col.lower() or "permalink" in col.lower():
                    col_cfg[col] = st.column_config.LinkColumn(col)
                elif "text" in col.lower() or "review" in col.lower() or "content" in col.lower():
                    col_cfg[col] = st.column_config.TextColumn(col, width="large")
                elif "rating" in col.lower() or "score" in col.lower():
                    col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f")
                elif "date" in col.lower():
                    col_cfg[col] = st.column_config.TextColumn(col, width="small")

            st.dataframe(
                fview_display.reset_index(drop=True),
                use_container_width=True,
                height=480,
                column_config=col_cfg,
            )

            # ── Download this file ────────────────────────────────────────────
            st.divider()
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    f"📥 Download full file ({len(fview):,} rows)",
                    data=fview.to_csv(index=False),
                    file_name=Path(selected_path).name,
                    mime="text/csv",
                    use_container_width=True,
                )
            with dcol2:
                if fv_search:
                    filtered_for_dl = fview[
                        fview.apply(lambda col: col.astype(str).str.contains(fv_search, case=False, na=False)).any(axis=1)
                    ]
                    st.download_button(
                        f"📥 Download filtered ({len(filtered_for_dl):,} rows)",
                        data=filtered_for_dl.to_csv(index=False),
                        file_name=f"filtered_{Path(selected_path).name}",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("Apply a search filter above to enable filtered download.")

        except Exception as e:
            st.error(f"Could not load file: {e}")
