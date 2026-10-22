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
  analyse.py           top terms, distinctive terms (log-odds)
  cli.py               the `ibias` command
tests/                 run with `pytest`
```

To add another outlet, write `sources/<outlet>.py` with `OUTLET` and
`to_article()`, and register it in `store.SOURCES`.

## Roadmap

1. Stage 1 (this): Guardian fetch, storage, word counts, first exploration.
2. Stage 1b: UK partisan phrases. Take Commons speeches with party labels
   (ParlParse/TheyWorkForYou XML, or ParlSpeech V2), find phrases Labour and
   Conservative MPs use at different rates, and score Guardian articles with them.
3. Stage 2: more outlets, better models (TF-IDF + logistic regression, embeddings, LLMs).
4. Stage 3: a FastAPI endpoint that estimates the slant of a pasted text.
