from datetime import date

import pandas as pd
from conftest import make_item

from ibias import analyse, features, store
from ibias.sources import guardian
from ibias.text import count_terms, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Prime Minister’s plan isn't working") == ["prime", "minister's", "plan", "isn't", "working"]


def test_count_terms_includes_phrases():
    counts = count_terms("Public services need investment. Public services matter.")
    assert counts["public services"] == 2
    assert counts["public services need"] == 1


def test_load_is_idempotent():
    guardian.save_day(date(2026, 10, 21), [make_item(i) for i in range(3)])
    con = store.connect()
    store.load("guardian", con)
    store.load("guardian", con)  # loading twice must not duplicate rows
    assert con.execute("SELECT count(*) FROM articles").fetchone()[0] == 3


def test_end_to_end_distinctive_terms():
    news = [make_item(i, pillar="News") for i in range(6)]
    opinion = [
        make_item(100 + i, pillar="Opinion", body="Austerity hurt public services, and austerity must end.")
        for i in range(6)
    ]
    guardian.save_day(date(2026, 10, 21), news + opinion)
    store.load("guardian")
    path = features.build_term_counts()
    assert not pd.read_parquet(path).empty

    counts = analyse.load_counts("pillar")
    top = analyse.distinctive_terms(counts, "Opinion", n=1, k=3)
    assert top.index[0] == "austerity"
