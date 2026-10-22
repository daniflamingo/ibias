"""Command line entry point. After `pip install -e .`, run `ibias --help`.

The pipeline is four steps; each one reads what the previous step wrote,
so you can rerun any step on its own:

    ibias fetch --from 2026-10-01 --to 2026-10-07   # API -> data/raw/
    ibias load                                      # data/raw/ -> data/ibias.duckdb
    ibias count                                     # articles -> data/features/term_counts.parquet
    ibias explore --by pillar --group Opinion       # print top and distinctive terms
"""

import argparse
from datetime import date

import pandas as pd

from ibias import analyse, features, store
from ibias.sources import guardian


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


if __name__ == "__main__":
    main()
