"""Find partisan phrases in Hansard and use them to score articles.

This follows Gentzkow & Shapiro (2010), "What Drives Media Slant?":

1. Count every two- and three-word phrase said by Labour MPs and by Conservative MPs.
2. For each phrase, ask: is the split between the parties further from what
   we'd expect by chance than for other phrases? Pearson's chi-squared statistic
   measures exactly that, for a 2x2 table:

                        this phrase    all other phrases
        Labour            f_lab           F_lab - f_lab
        Conservative      f_con           F_con - f_con

       chi2 = (f_con * not_lab - f_lab * not_con)^2
              / ((f_con + f_lab) * (f_con + not_con) * (f_lab + not_lab) * (not_con + not_lab))

3. Keep the top phrases by chi2. Each "leans" to the party that uses it more often
   relative to how much that party speaks overall.

Then an article's slant is the balance of Labour-leaning and Conservative-leaning
phrases it uses: +1 = only Labour-leaning phrases, -1 = only Conservative-leaning.

Two cautions to keep in mind when reading results:
- Government versus opposition: the governing party says "our plan" and the opposition
  "this government has failed", whoever is in power. Use a date range that covers both
  a Conservative and a Labour government (the switch was July 2024): we then keep only
  phrases that lean the same way under both governments (see ERA_SPLIT).
- Topic: a party talking more about one subject is not the same as slanted language.
"""

from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from ibias import config, store
from ibias.text import ngrams, tokenize

PARTIES = ("Labour", "Conservative")

# Phrases containing these words are parliamentary procedure, not politics:
# "hon friend", "right hon member", "secretary state", "madam deputy speaker".
PROCEDURAL = frozenset("""
hon honourable right friend friends gentleman lady member members house speaker deputy madam
minister ministers secretary state shadow parliamentary committee clause amendment amendments
bill order question questions debate motion division lords chamber constituency constituents
dispatch box vol col mc wh ws official report
""".split())

# Phrases about who is in power ("previous government", "this government has failed")
# flip sides at every election: they measure being in government, not ideology.
INCUMBENCY = frozenset("government governments government's opposition".split())

# Parties naming themselves or each other ("labour government", "conservative party")
# tells us who is speaking, not how they frame an issue, so those phrases are dropped too.
PARTY_WORDS = frozenset("""
labour conservative conservatives tory tories liberal democrat democrats snp reform
""".split())

# The general election that moved Labour into government. Phrases are checked separately
# before and after it, and kept only if they lean to the same party in both periods.
# That removes phrases that are really about being in government or opposition
# ("good news", "red sea" under the Conservatives) rather than about party.
ERA_SPLIT = "2024-07-05"

# Hand-picked phrases to ignore: everyday English that happens to be partisan in
# Parliament ("five minutes" is a speech time limit). Edit the file to add your own.
STOPLIST_PATH = Path(__file__).with_name("phrase_stoplist.txt")

# Ignore phrases said fewer times than this in total; rare phrases give noisy chi2.
MIN_COUNT = 20
# Ignore phrases used by fewer distinct MPs, or on fewer sitting days, than this.
# Without these, one bill debated over a few days ("victoria tower gardens") looks partisan.
MIN_SPEAKERS = 10
MIN_DAYS = 20
# Chi-squared rewards frequency: a common phrase with a mild lean ("make sure") can
# outrank a rarer, sharply partisan one. So we also require one party to use a phrase
# at least this many times as often as the other, per phrase spoken.
MIN_RATIO = 2.0


def phrases_path() -> Path:
    return config.FEATURES_DIR / "partisan_phrases.parquet"


def count_party_phrases(
    con: duckdb.DuckDBPyConnection, min_speakers: int = MIN_SPEAKERS, min_days: int = MIN_DAYS
) -> pd.DataFrame:
    """Total count of each 2- and 3-word phrase by party, and on how many days by how many MPs.

    A year of Commons debates has millions of distinct phrases, too many for a
    Python dict on a laptop. So we count one speaker-day at a time in Python,
    append to a DuckDB table, and let DuckDB do the big GROUP BY (it spills to disk).
    """
    con.execute(
        "CREATE OR REPLACE TEMP TABLE phrase_rows"
        " (party VARCHAR, speaker VARCHAR, day DATE, term VARCHAR, n TINYINT, count INTEGER)"
    )
    rows = con.execute(
        "SELECT party, speaker_id, date, text FROM speeches WHERE party IN ('Labour', 'Conservative') ORDER BY date"
    ).fetchall()
    buffer: Counter = Counter()
    # tqdm wraps any loop and draws a progress bar with the time remaining.
    for i, (party, speaker, day, text) in enumerate(tqdm(rows, desc="counting phrases", unit=" speeches"), 1):
        tokens = tokenize(text)
        for n in (2, 3):
            buffer.update((party, speaker, day, term, n) for term in ngrams(tokens, n))
        if i % 20_000 == 0 or i == len(rows):
            _flush(con, buffer)
            buffer = Counter()
    print("adding up counts across all speeches (no progress bar for this part, usually under a minute)")
    return con.execute(f"""
        SELECT term, any_value(n) AS n,
               sum(count) FILTER (WHERE party = 'Labour')       AS lab,
               sum(count) FILTER (WHERE party = 'Conservative') AS con,
               sum(count) FILTER (WHERE party = 'Labour' AND day < DATE '{ERA_SPLIT}')        AS lab_before,
               sum(count) FILTER (WHERE party = 'Conservative' AND day < DATE '{ERA_SPLIT}')  AS con_before,
               sum(count) FILTER (WHERE party = 'Labour' AND day >= DATE '{ERA_SPLIT}')       AS lab_after,
               sum(count) FILTER (WHERE party = 'Conservative' AND day >= DATE '{ERA_SPLIT}') AS con_after,
               count(DISTINCT speaker) AS speakers,
               count(DISTINCT day) AS days
        FROM phrase_rows
        GROUP BY term
        HAVING sum(count) >= {MIN_COUNT}
           AND count(DISTINCT speaker) >= {int(min_speakers)}
           AND count(DISTINCT day) >= {int(min_days)}
    """).df().fillna(0)


def _flush(con: duckdb.DuckDBPyConnection, buffer: Counter) -> None:
    if not buffer:
        return
    df = pd.DataFrame(
        [(*key, c) for key, c in buffer.items()], columns=["party", "speaker", "day", "term", "n", "count"]
    )
    con.register("buf", df)
    con.execute("INSERT INTO phrase_rows SELECT * FROM buf")
    con.unregister("buf")


def chi_squared(counts: pd.DataFrame) -> pd.DataFrame:
    """Add chi2 and the leaning party to a table with columns term, lab, con."""
    df = counts.copy()
    total_lab, total_con = df["lab"].sum(), df["con"].sum()
    f_lab, f_con = df["lab"], df["con"]
    not_lab, not_con = total_lab - f_lab, total_con - f_con
    num = (f_con * not_lab - f_lab * not_con) ** 2
    den = (f_con + f_lab) * (f_con + not_con) * (f_lab + not_lab) * (not_con + not_lab)
    df["chi2"] = num / den
    # Compare rates, not raw counts, since one party may simply speak more.
    # Adding 1 to each count keeps the ratio finite for phrases one party never says.
    rate_lab, rate_con = (f_lab + 1) / total_lab, (f_con + 1) / total_con
    df["leans"] = np.where(rate_lab > rate_con, "Labour", "Conservative")
    df["ratio"] = np.maximum(rate_lab, rate_con) / np.minimum(rate_lab, rate_con)
    return df


def load_stoplist(path: Path = STOPLIST_PATH) -> set[str]:
    if not path.exists():
        return set()
    lines = (line.split("#")[0].strip().lower() for line in path.read_text().splitlines())
    return {line for line in lines if line}


def consistent_across_eras(counts: pd.DataFrame) -> pd.Series:
    """True where a phrase leans to the same party before and after ERA_SPLIT.

    If the Hansard data covers only one of the two periods, the check cannot run,
    so every phrase passes and a warning is printed.
    """
    totals = counts[["lab_before", "con_before", "lab_after", "con_after"]].sum()
    if (totals == 0).any():
        print(f"warning: Hansard data does not cover both sides of {ERA_SPLIT}; skipping the government/opposition check")
        return pd.Series(True, index=counts.index)
    before = (counts["lab_before"] + 1) / totals["lab_before"] - (counts["con_before"] + 1) / totals["con_before"]
    after = (counts["lab_after"] + 1) / totals["lab_after"] - (counts["con_after"] + 1) / totals["con_after"]
    return np.sign(before) == np.sign(after)


def build_partisan_phrases(
    con: duckdb.DuckDBPyConnection | None = None,
    top_k: int = 500,
    min_speakers: int = MIN_SPEAKERS,
    min_days: int = MIN_DAYS,
    min_ratio: float = MIN_RATIO,
    check_eras: bool = True,
) -> Path:
    """Write the top_k most partisan 2-word and top_k 3-word phrases to Parquet."""
    con = con or store.connect()
    counts = count_party_phrases(con, min_speakers, min_days)
    if counts.empty:
        raise SystemExit("No phrase passed the filters. Fetch more days, or lower --min-speakers / --min-days.")
    excluded = PROCEDURAL | PARTY_WORDS | INCUMBENCY
    dropped = counts["term"].str.split().apply(lambda words: any(w in excluded for w in words))
    stoplist = load_stoplist()
    # Pad with spaces so "left us" matches whole words only, not "left usual".
    dropped |= counts["term"].apply(lambda t: any(f" {stop} " in f" {t} " for stop in stoplist))
    scored = chi_squared(counts[~dropped])
    scored = scored[scored["ratio"] >= min_ratio]
    if check_eras:
        scored = scored[consistent_across_eras(scored)]
    top = (
        scored.sort_values("chi2", ascending=False)
        .groupby("n")
        .head(top_k)
        .sort_values("chi2", ascending=False)
    )
    path = phrases_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    top.to_parquet(path, index=False)
    return path


def score_articles(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Slant per article from the term counts and the partisan phrase list.

    Needs `ibias count` (term_counts.parquet) and `ibias phrases` to have run.
    slant = (Labour-leaning phrase uses - Conservative-leaning) / (all partisan phrase uses)
    """
    con = con or store.connect()
    terms = (config.FEATURES_DIR / "term_counts.parquet").as_posix()
    phrases = phrases_path().as_posix()
    return con.execute(f"""
        SELECT a.id, a.published_at, a.pillar, a.section, a.headline,
               sum(t.count) FILTER (WHERE p.leans = 'Labour')       AS lab_phrases,
               sum(t.count) FILTER (WHERE p.leans = 'Conservative') AS con_phrases,
               (coalesce(lab_phrases, 0) - coalesce(con_phrases, 0)) / sum(t.count) AS slant
        FROM read_parquet('{terms}') t
        JOIN read_parquet('{phrases}') p USING (term)
        JOIN articles a ON a.id = t.article_id
        GROUP BY ALL
    """).df()


def slant_by(scores: pd.DataFrame, column: str = "pillar", n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Slant per group, with a 95% range from bootstrapping.

    Each article counts in proportion to how many partisan phrases it used.

    The bootstrap answers "how much could this number move if we had happened to
    get a different sample of articles?". We redraw the group's articles at random,
    with replacement, n_boot times, recompute the slant each time, and report the
    middle 95% of those values. If two groups' ranges overlap a lot, the difference
    between them may be chance.
    """
    df = scores.fillna({"lab_phrases": 0, "con_phrases": 0})
    rng = np.random.default_rng(seed)
    rows = {}
    for group, g in df.groupby(column):
        lab, con = g["lab_phrases"].to_numpy(), g["con_phrases"].to_numpy()
        boot = []
        for _ in range(0, n_boot, 100):  # 100 resamples at a time to keep memory small
            idx = rng.integers(0, len(g), size=(100, len(g)))
            l, c = lab[idx].sum(axis=1), con[idx].sum(axis=1)
            boot.append((l - c) / (l + c))
        low, high = np.percentile(np.concatenate(boot)[:n_boot], [2.5, 97.5])
        rows[group] = {
            "articles": len(g),
            "lab_phrases": lab.sum(),
            "con_phrases": con.sum(),
            "slant": (lab.sum() - con.sum()) / (lab.sum() + con.sum()),
            "low_95": low,
            "high_95": high,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(column).sort_values("slant")


def explain(column: str = "pillar", k: int = 5, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """The partisan phrases that contribute most to each group's slant."""
    if column not in {"pillar", "section"}:
        raise ValueError(f"unsupported column: {column}")
    con = con or store.connect()
    terms = (config.FEATURES_DIR / "term_counts.parquet").as_posix()
    phrases = phrases_path().as_posix()
    return con.execute(f"""
        SELECT a.{column} AS grp, p.leans, t.term, sum(t.count) AS uses, count(DISTINCT a.id) AS articles
        FROM read_parquet('{terms}') t
        JOIN read_parquet('{phrases}') p USING (term)
        JOIN articles a ON a.id = t.article_id
        GROUP BY grp, p.leans, t.term
        QUALIFY row_number() OVER (PARTITION BY grp, p.leans ORDER BY uses DESC) <= {int(k)}
        ORDER BY grp, p.leans, uses DESC
    """).df()
