"""Count words and phrases per article and save them as Parquet.

Output: data/features/term_counts.parquet with one row per (article, term):
    article_id | n | term            | count
    guardian:… | 2 | prime minister  | 4

This "long" format is easy to filter and aggregate with SQL or pandas,
and it is the input to every model we build later.
"""

from pathlib import Path

import duckdb
import pandas as pd

from ibias import config, store
from ibias.text import count_terms

# A term must appear in at least this many articles to be kept.
# Rare phrases are mostly noise and would make the table huge.
MIN_DOCS = 5


def build_term_counts(
    con: duckdb.DuckDBPyConnection | None = None,
    out_path: Path | None = None,
    min_docs: int = MIN_DOCS,
) -> Path:
    con = con or store.connect()
    out_path = out_path or config.FEATURES_DIR / "term_counts.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for article_id, body in con.execute("SELECT id, body FROM articles").fetchall():
        for term, count in count_terms(body).items():
            rows.append((article_id, term.count(" ") + 1, term, count))
    df = pd.DataFrame(rows, columns=["article_id", "n", "term", "count"])

    doc_freq = df.groupby("term")["article_id"].transform("nunique")
    df = df[doc_freq >= min_docs]
    df.to_parquet(out_path, index=False)
    return out_path
