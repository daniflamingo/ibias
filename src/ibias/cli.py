"""Command line entry point. After `pip install -e .`, run `ibias --help`.

The pipeline is four steps; each one reads what the previous step wrote,
so you can rerun any step on its own:

    ibias fetch --from 2026-10-01 --to 2026-10-07   # API -> data/raw/
    ibias load                                      # data/raw/ -> data/ibias.duckdb
    ibias count                                     # articles -> data/features/term_counts.parquet
    ibias explore --by pillar --group Opinion       # print top and distinctive terms

Stage 1b, partisan phrases from House of Commons speeches:

    ibias hansard-fetch --from 2023-07-01 --to 2025-06-30   # TheyWorkForYou -> data/raw/hansard/
    ibias hansard-load                                      # speeches table in DuckDB
    ibias phrases                                           # top Labour/Conservative phrases
    ibias slant --by pillar                                 # score Guardian articles
"""

import argparse
from datetime import date

import pandas as pd

from ibias import analyse, features, partisan, store
from ibias.sources import guardian, hansard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ibias")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download Guardian articles for a date range")
    p.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    p.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    p.add_argument("--force", action="store_true", help="re-download days already saved")

    sub.add_parser("load", help="load raw files into DuckDB")
    sub.add_parser("count", help="count words and phrases per article")

    p = sub.add_parser("explore", help="print top and distinctive terms")
    p.add_argument("--by", default="pillar", choices=["pillar", "section"])
    p.add_argument("--group", default="Opinion", help="the group to compare with the rest")
    p.add_argument("--n", type=int, default=2, help="1 = words, 2 = two-word phrases, 3 = three")

    p = sub.add_parser("hansard-fetch", help="download Commons debates and the MP list")
    p.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    p.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    p.add_argument("--force", action="store_true", help="re-download files already saved")

    sub.add_parser("hansard-load", help="parse saved debates into the speeches table")

    p = sub.add_parser("phrases", help="find the most partisan phrases")
    p.add_argument("--top", type=int, default=500, help="how many 2-word and 3-word phrases to keep")
    p.add_argument("--min-speakers", type=int, default=partisan.MIN_SPEAKERS, help="drop phrases used by fewer MPs")
    p.add_argument("--min-days", type=int, default=partisan.MIN_DAYS, help="drop phrases used on fewer sitting days")
    p.add_argument(
        "--min-ratio", type=float, default=partisan.MIN_RATIO, help="how many times more one party must use a phrase"
    )

    p.add_argument(
        "--no-era-check", action="store_true", help="keep phrases that flip party when the government changes"
    )

    p = sub.add_parser("slant", help="score Guardian articles with the partisan phrases")
    p.add_argument("--by", default="pillar", choices=["pillar", "section"])
    p.add_argument("--explain", action="store_true", help="show which phrases drive each group's score")

    args = parser.parse_args(argv)

    if args.command == "fetch":
        guardian.fetch_range(args.start, args.end, force=args.force)
    elif args.command == "load":
        print(f"loaded {store.load('guardian')} Guardian articles")
    elif args.command == "count":
        print(f"wrote {features.build_term_counts()}")
    elif args.command == "explore":
        counts = analyse.load_counts(args.by)
        pd.set_option("display.width", 120)
        print("Most frequent terms:\n", analyse.top_terms(counts).to_string(index=False))
        print(f"\nTerms most typical of {args.by} = {args.group!r} (n={args.n}):")
        print(analyse.distinctive_terms(counts, args.group, n=args.n).to_string())
    elif args.command == "hansard-fetch":
        hansard.fetch_range(args.start, args.end, force=args.force)
    elif args.command == "hansard-load":
        con = store.connect()
        print(f"loaded {store.load_speeches(con)} speeches")
        print(con.sql("SELECT party, count(*) AS speeches FROM speeches GROUP BY 1 ORDER BY 2 DESC LIMIT 8"))
    elif args.command == "phrases":
        path = partisan.build_partisan_phrases(
            top_k=args.top,
            min_speakers=args.min_speakers,
            min_days=args.min_days,
            min_ratio=args.min_ratio,
            check_eras=not args.no_era_check,
        )
        phrases = pd.read_parquet(path)
        print(f"wrote {path}")
        for party in partisan.PARTIES:
            top = phrases[phrases["leans"] == party].head(25)
            print(f"\nMost {party}-leaning phrases:\n", top[["term", "lab", "con", "ratio", "speakers", "days", "chi2"]].round({"ratio": 1}).to_string(index=False))
    elif args.command == "slant":
        scores = partisan.score_articles()
        pd.set_option("display.width", 120)
        print("Slant: +1 = only Labour-leaning phrases, -1 = only Conservative-leaning.")
        print("low_95 to high_95: the range the slant could plausibly take with a different sample of articles.")
        print("'articles' counts only articles that use at least one partisan phrase.\n")
        print(partisan.slant_by(scores, args.by).round(3).to_string())
        if args.explain:
            reasons = partisan.explain(args.by)
            for group, g in reasons.groupby("grp"):
                print(f"\n{args.by} = {group}")
                print(g.drop(columns="grp").to_string(index=False))


if __name__ == "__main__":
    main()
