"""
run.py — single entry point for the full pipeline
Usage:
    python run.py --scrape          # scrape only
    python run.py --analyse         # analyse only (uses existing raw CSV)
    python run.py --analyse --sample 50  # test on 50 reviews
    python run.py --all             # scrape + analyse
    streamlit run dashboard/app.py  # launch dashboard separately
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description=f"Porter Intelligence Pipeline")
    parser.add_argument("--scrape",  action="store_true", help="Run scrapers")
    parser.add_argument("--analyse", action="store_true", help="Run multi-agent analysis")
    parser.add_argument("--all",     action="store_true", help="Run full pipeline")
    parser.add_argument("--sample",  type=int, default=None, help="Analyse only N reviews")
    args = parser.parse_args()

    if not any([args.scrape, args.analyse, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.scrape or args.all:
        print("\n[1/2] Running scrapers...")
        from scrapers.scrape_all import run_scrapers
        run_scrapers()

    if args.analyse or args.all:
        print("\n[2/2] Running multi-agent analysis...")
        from pipeline.analyse import run_analysis
        run_analysis(sample=args.sample)

    print("\n✅ Done. Launch dashboard with:")
    print("   streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    main()
