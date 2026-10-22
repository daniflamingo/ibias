"""Load the raw files into a DuckDB database of clean articles.

DuckDB is an analytics database that lives in a single file (data/ibias.duckdb).
You query it with SQL from Python, the duckdb command line, or a notebook:

    import duckdb
    con = duckdb.connect("data/ibias.duckdb")
    con.sql("SELECT section, count(*) FROM articles GROUP BY 1 ORDER BY 2 DESC").show()
"""

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from ibias import config
from ibias.sources import guardian

# Each outlet module provides OUTLET and to_article(); add new outlets here.
SOURCES = {guardian.OUTLET: guardian}

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id           VARCHAR PRIMARY KEY,   -- '<outlet>:<outlet's own id>', unique across outlets
    outlet       VARCHAR NOT NULL,
    published_at TIMESTAMPTZ NOT NULL,
    section      VARCHAR,
    pillar       VARCHAR,
    headline     VARCHAR,
    byline       VARCHAR,
    authors      VARCHAR[],
    url          VARCHAR,
    body         VARCHAR NOT NULL,
    word_count   INTEGER
)
"""


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    return con


def iter_raw(outlet: str) -> Iterator[dict]:
    for path in sorted((config.RAW_DIR / outlet).rglob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def load(outlet: str = "guardian", con: duckdb.DuckDBPyConnection | None = None) -> int:
    """Rebuild this outlet's rows from its raw files. Returns the number of articles loaded.

    INSERT OR REPLACE uses the primary key, so loading the same day twice
    updates the rows instead of duplicating them.
    """
    con = con or connect()
    source = SOURCES[outlet]
    articles = [a for a in map(source.to_article, iter_raw(outlet)) if a is not None]
    if not articles:
        return 0
    df = pd.DataFrame(articles).drop_duplicates("id", keep="last")
    con.register("incoming", df)
    con.execute("""
        INSERT OR REPLACE INTO articles
        SELECT id, outlet, CAST(published_at AS TIMESTAMPTZ), section, pillar, headline,
               byline, authors, url, body, word_count
        FROM incoming
    """)
    con.unregister("incoming")
    return len(df)
