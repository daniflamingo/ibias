import gzip
import json
from datetime import date

import pandas as pd
import pytest
from conftest import make_item

from ibias import features, partisan, store
from ibias.sources import guardian, hansard

PEOPLE = {
    "posts": [
        {"id": "uk.org.publicwhip/cons/1", "organization_id": "house-of-commons"},
        {"id": "uk.org.publicwhip/cons/2", "organization_id": "house-of-commons"},
        {"id": "scot/1", "organization_id": "scottish-parliament"},
    ],
    "memberships": [
        # person 1 switches from Conservative to Labour on 2025-01-01
        {"id": "uk.org.publicwhip/member/1", "person_id": "uk.org.publicwhip/person/1", "post_id": "uk.org.publicwhip/cons/1",
         "on_behalf_of_id": "conservative", "start_date": "2019-12-12", "end_date": "2024-12-31"},
        {"id": "uk.org.publicwhip/member/2", "person_id": "uk.org.publicwhip/person/1", "post_id": "uk.org.publicwhip/cons/1",
         "on_behalf_of_id": "labourco-operative", "start_date": "2025-01-01"},
        {"id": "uk.org.publicwhip/member/3", "person_id": "uk.org.publicwhip/person/2", "post_id": "uk.org.publicwhip/cons/2",
         "on_behalf_of_id": "labour", "start_date": "2019-12-12"},
        # a seat in another parliament must be ignored
        {"id": "uk.org.publicwhip/member/4", "person_id": "uk.org.publicwhip/person/2", "post_id": "scot/1",
         "on_behalf_of_id": "scottish-national-party", "start_date": "2019-12-12"},
    ],
}

LAB = "Working people deserve decent jobs and strong public services. Working people pay the price."
CON = "Hard-working families need lower taxes and sound money. Lower taxes help hard-working families."


def write_debate(day: str, speeches: list[tuple[str, str]]) -> None:
    body = "".join(
        f'<speech id="uk.org.publicwhip/debate/{day}a.{i}.0" speakername="MP {i}" person_id="{pid}"><p>{text}</p></speech>'
        for i, (pid, text) in enumerate(speeches)
    )
    body += f'<speech id="uk.org.publicwhip/debate/{day}a.99.0" speakerid="unknown"><p>Question put.</p></speech>'
    path = hansard.debate_path(f"debates{day}a.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(f"<publicwhip>{body}</publicwhip>")


def write_people() -> None:
    hansard.people_path().parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(hansard.people_path(), "wt") as f:
        json.dump(PEOPLE, f)


def test_latest_versions_picks_newest_letter_in_range():
    html = "debates2025-03-11a.xml debates2025-03-12a.xml debates2025-03-12b.xml debates2025-04-01a.xml"
    files = hansard.latest_versions(html, date(2025, 3, 1), date(2025, 3, 31))
    assert files == {"2025-03-11": "debates2025-03-11a.xml", "2025-03-12": "debates2025-03-12b.xml"}


def test_party_lookup_follows_dates_and_ignores_other_parliaments():
    parties = hansard.PartyLookup(PEOPLE)
    assert parties.party("uk.org.publicwhip/person/1", "2024-06-01") == "conservative"
    assert parties.party("uk.org.publicwhip/person/1", "2025-06-01") == "labourco-operative"
    assert parties.party("uk.org.publicwhip/member/1", "2024-06-01") == "conservative"
    assert parties.party("uk.org.publicwhip/person/2", "2025-06-01") == "labour"
    assert parties.party("uk.org.publicwhip/person/999", "2025-06-01") is None


def test_partisan_phrases_and_slant_end_to_end():
    write_people()
    lab, con = "uk.org.publicwhip/person/2", "uk.org.publicwhip/person/1"
    for d in range(1, 21):
        # person 1 is Conservative before 2025, so these are Conservative speeches
        write_debate(f"2024-03-{d:02d}", [(lab, LAB), (con, CON)])
    db = store.connect()
    assert store.load_speeches(db) == 40  # the "unknown" procedural line is skipped
    assert set(db.execute("SELECT DISTINCT party FROM speeches").fetchnumpy()["party"]) == {"Labour", "Conservative"}

    phrases = pd.read_parquet(partisan.build_partisan_phrases(db, top_k=50, min_speakers=1, min_days=1)).set_index("term")
    assert phrases.loc["working people", "leans"] == "Labour"
    assert phrases.loc["lower taxes", "leans"] == "Conservative"

    left = [make_item(i, body=LAB) for i in range(6)]
    right = [make_item(100 + i, pillar="Opinion", body=CON) for i in range(6)]
    guardian.save_day(date(2026, 10, 21), left + right)
    store.load("guardian", db)
    features.build_term_counts(db)
    by_pillar = partisan.slant_by(partisan.score_articles(db), "pillar")
    assert by_pillar.loc["News", "slant"] == 1.0
    assert by_pillar.loc["Opinion", "slant"] == -1.0
    assert by_pillar.loc["News", "low_95"] <= 1.0 <= by_pillar.loc["News", "high_95"]

    reasons = partisan.explain("pillar", k=2, con=db)
    assert set(reasons.loc[reasons["grp"] == "Opinion", "leans"]) == {"Conservative"}


def test_ratio_compares_rates_not_raw_counts():
    # Labour speaks twice as much overall, so 20 vs 10 is no lean at all.
    df = partisan.chi_squared(pd.DataFrame({"term": ["a", "b"], "lab": [19, 181], "con": [9, 91]}))
    assert (df["ratio"] < 1.05).all()


def test_chi_squared_is_zero_for_evenly_split_phrase():
    df = partisan.chi_squared(pd.DataFrame({"term": ["a", "b"], "lab": [10, 30], "con": [10, 30]}))
    assert df["chi2"].tolist() == [0.0, 0.0]


def test_phrases_from_too_few_days_are_dropped():
    write_people()
    write_debate("2024-03-01", [("uk.org.publicwhip/person/2", LAB * 20), ("uk.org.publicwhip/person/1", CON * 20)])
    db = store.connect()
    store.load_speeches(db)
    with pytest.raises(SystemExit):
        partisan.build_partisan_phrases(db, min_speakers=1, min_days=2)


def test_bootstrap_range_is_wide_for_few_articles_and_narrow_for_many():
    few = pd.DataFrame({"pillar": "A", "lab_phrases": [3, 0, 1], "con_phrases": [0, 2, 1]})
    many = pd.DataFrame({"pillar": "B", "lab_phrases": [3, 0, 1] * 300, "con_phrases": [0, 2, 1] * 300})
    out = partisan.slant_by(pd.concat([few, many]), "pillar")
    width = out["high_95"] - out["low_95"]
    assert out.loc["A", "slant"] == out.loc["B", "slant"]
    assert width["A"] > 5 * width["B"]


def test_era_check_drops_phrases_that_flip_with_the_government():
    counts = pd.DataFrame({
        "term": ["working people", "good news", "filler"],
        # "good news" is Conservative while in government, Labour once Labour governs
        "lab_before": [50, 5, 500], "con_before": [5, 50, 500],
        "lab_after": [50, 50, 500], "con_after": [5, 5, 500],
    })
    assert partisan.consistent_across_eras(counts).tolist()[:2] == [True, False]


def test_stoplist_matches_whole_words(tmp_path):
    path = tmp_path / "stop.txt"
    path.write_text("# comment\nleft us   # tributes\n\n")
    stop = partisan.load_stoplist(path)
    assert stop == {"left us"}
