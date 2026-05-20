# Porter Customer Intelligence Platform

Multi-agent LLM pipeline that scrapes reviews from 4 platforms, analyses them through 5 AI agents with different perspectives, and surfaces prioritised action items on a live Streamlit dashboard.

## Architecture

```
config/config.py        ← change PRODUCT here to run for any company
scrapers/scrape_all.py  ← Google Play (customer + driver), Reddit, Trustpilot
pipeline/analyse.py     ← 5-agent LLM analysis (PM, CS Lead, Growth, Skeptic, Synthesiser)
dashboard/app.py        ← Streamlit dashboard (Overview / Deep Dive / Action Items)
run.py                  ← single entry point
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
# Full pipeline
python run.py --all

# Test on 20 reviews first (saves API cost)
python run.py --scrape
python run.py --analyse --sample 20

# Launch dashboard
streamlit run dashboard/app.py
```

## Multi-Agent Design

Each review passes through 5 agents with different objectives:

| Agent | Perspective | Output |
|---|---|---|
| PM | Feature gaps, UX friction | Category + Severity |
| CS Lead | Urgent recurring pain | Support category + Urgency |
| Growth Analyst | Churn & acquisition blockers | Churn risk + Growth impact |
| Skeptic | Is this noise or real? | Actionable flag + Reasoning |
| Synthesiser | Reconciles all 4 | Priority + Confidence + Action item |

## To run for a different product

Edit `config/config.py`:
```python
PRODUCT = "Dunzo"
PLAYSTORE_CUSTOMER_APP_ID = "com.dunzo.user"
TRUSTPILOT_COMPANY_URL = "https://www.trustpilot.com/review/dunzo.com"
REDDIT_QUERIES = ["Dunzo app review", "Dunzo delivery experience"]
```
Then `python run.py --all`.
