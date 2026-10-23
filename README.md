# iBias

Can we infer the political leaning of a newspaper from its word choice?

The idea comes from Gentzkow & Shapiro, "What Drives Media Slant?" (Econometrica, 2010).
They found the phrases that US Democrats and Republicans in Congress use at
different rates ("death tax" against "estate tax"), then scored newspapers by
which party's language they use more. iBias does the same for UK outlets,
starting with the Guardian and using UK parliamentary speeches (Hansard) to learn
which phrases are partisan.

## Setup

You need Python 3.10 or newer.

```bash
cd ~/TestProjects/iBias
python3 -m venv .venv             # a private Python just for this project
source .venv/bin/activate         # run this again in each new terminal
pip install -e ".[dev]"           # installs iBias and its dependencies
cp .env.example .env              # then paste your Guardian key into .env
```

`pip install -e` installs the project in "editable" mode: code changes take
effect without reinstalling.

## Running the pipeline

Each step reads what the previous step wrote, so any step can be rerun on its own.

```bash
ibias fetch --from 2026-10-01 --to 2026-10-07   # 1. download raw articles
ibias load                                      # 2. build the DuckDB database
ibias count                                     # 3. count words and phrases
ibias explore --by pillar --group Opinion       # 4. print top and distinctive terms
```

### Stage 1b: partisan phrases from Parliament

```bash
ibias hansard-fetch --from 2023-07-01 --to 2025-06-30   # Commons debates + MP list (~500 sitting days)
ibias hansard-load                                      # speeches with each MP's party, into DuckDB
ibias phrases                                           # top 500 two-word and 500 three-word partisan phrases
ibias slant --by pillar                                 # Guardian slant per pillar (needs `ibias count` first)
```

Debates come from [TheyWorkForYou](https://www.theyworkforyou.com/) (mySociety's ParlParse XML,
one file per sitting day) and parties from ParlParse's `people.json`. `ibias phrases` follows
Gentzkow & Shapiro: count every two- and three-word phrase said by Labour and by Conservative
MPs, rank by Pearson's chi-squared, and drop procedural phrases like "hon friend".
An article's slant is (Labour-leaning phrase uses - Conservative-leaning) / all partisan
phrase uses, from +1 (all Labour-leaning) to -1 (all Conservative-leaning).

Two cautions: the governing party talks differently from the opposition whoever is in power,
so pick a date range covering both a Conservative and a Labour government (the switch was
4 July 2024); and a party talking about a topic more is not the same as slanted language.
The scores are relative: compare sections, authors or outlets with each other, not with zero.

Then open `notebooks/01_explore.ipynb` (`jupyter lab`) to poke at the data yourself.

## Where the data goes

Everything is written under `data/`, which git ignores:

| Path | What | Why |
| --- | --- | --- |
| `data/raw/guardian/YYYY/MM/DD.jsonl.gz` | API responses, untouched, one JSON object per line | Source of truth; we can always redo cleaning from here |
| `data/ibias.duckdb` | Table `articles`, one row per article | Fast SQL over millions of rows, in a single file |
| `data/features/term_counts.parquet` | Word and phrase counts per article | Input to every model |

A year of Guardian articles is roughly 300 MB compressed, so this fits on a laptop for a long time.

## Code layout

```
src/ibias/
  config.py            paths and the API key
  sources/guardian.py  fetch + convert Guardian items to the common article shape
  store.py             load raw files into DuckDB
  text.py              tokenising, stopwords, n-grams
  features.py          per-article term counts -> Parquet
  sources/hansard.py   fetch + parse Commons debates, MP parties
  analyse.py           top terms, distinctive terms (log-odds)
  partisan.py          partisan phrases (chi-squared) and article slant
  cli.py               the `ibias` command
tests/                 run with `pytest`
```

To add another outlet, write `sources/<outlet>.py` with `OUTLET` and
`to_article()`, and register it in `store.SOURCES`.

## Roadmap

1. Stage 1 (this): Guardian fetch, storage, word counts, first exploration.
2. Stage 1b (done): UK partisan phrases from Commons speeches, used to score Guardian articles.
3. Stage 2: more outlets, better models (TF-IDF + logistic regression, embeddings, LLMs).
4. Stage 3: a FastAPI endpoint that estimates the slant of a pasted text.
