"""First look at the data: which terms are typical of one group of articles versus another.

This is the same idea Gentzkow & Shapiro use to find partisan phrases, applied
to any two groups. Here we compare one Guardian section (or pillar) with the rest.
Later, with Hansard, the two groups become Labour and Conservative MPs.

The score is a smoothed log-odds ratio:
    log( (count_in_group + 1) / (total_in_group + V) ) - log( (count_in_rest + 1) / (total_in_rest + V) )
where V is the number of distinct terms. Adding 1 stops rare terms from getting
infinite scores. Positive means "used more in the group", negative "used more elsewhere".
"""

import numpy as np
import pandas as pd

from ibias import config, store


def load_counts(column: str = "pillar") -> pd.DataFrame:
    """Term counts joined to an article attribute (section, pillar, ...)."""
    if column not in {"section", "pillar", "outlet"}:
        raise ValueError(f"unsupported column: {column}")
    con = store.connect()
    path = config.FEATURES_DIR / "term_counts.parquet"
    return con.execute(f"""
        SELECT a.{column} AS grp, t.n, t.term, t.count
        FROM read_parquet('{path.as_posix()}') t
        JOIN articles a ON a.id = t.article_id
    """).df()


def top_terms(counts: pd.DataFrame) -> pd.DataFrame:
    """The most frequent terms overall, by n-gram size."""
    totals = counts.groupby(["n", "term"])["count"].sum().reset_index()
    return totals.sort_values("count", ascending=False).groupby("n").head(15)


def distinctive_terms(counts: pd.DataFrame, group: str, n: int | None = None, k: int = 20) -> pd.DataFrame:
    """Terms most over-used in `group` compared with all other articles."""
    if n is not None:
        counts = counts[counts["n"] == n]
    in_group = counts[counts["grp"] == group].groupby("term")["count"].sum()
    rest = counts[counts["grp"] != group].groupby("term")["count"].sum()
    df = pd.DataFrame({"group": in_group, "rest": rest}).fillna(0)
    vocab = len(df)
    df["log_odds"] = np.log((df["group"] + 1) / (df["group"].sum() + vocab)) - np.log(
        (df["rest"] + 1) / (df["rest"].sum() + vocab)
    )
    return df.sort_values("log_odds", ascending=False).head(k)
