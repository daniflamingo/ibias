"""Fetch articles from the Guardian Open Platform and save them untouched.

API docs: https://open-platform.theguardian.com/documentation/search

We ask the /search endpoint for one day at a time, 200 results per page
(the maximum), and write every result as one JSON line in a gzipped file:
    data/raw/guardian/2026/10/21.jsonl.gz

Saving the raw response, before any cleaning, means a bug in a later step
never forces us to download everything again.
"""

import gzip
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from ibias import config

OUTLET = "guardian"
SEARCH_URL = "https://content.guardianapis.com/search"
PAGE_SIZE = 200
# The free developer key allows about 1 call per second and 500 calls per day.
SECONDS_BETWEEN_CALLS = 1.0


def raw_path(day: date) -> Path:
    return config.RAW_DIR / OUTLET / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl.gz"


def fetch_day(day: date, api_key: str, session: requests.Session | None = None) -> list[dict]:
    """Return every Guardian item published on `day`, following all pages."""
    session = session or requests.Session()
    params = {
        "from-date": day.isoformat(),
        "to-date": day.isoformat(),
        "page-size": PAGE_SIZE,
        "order-by": "oldest",
        # Ask for the article text and metadata, not just headlines.
        "show-fields": "headline,byline,bodyText,wordcount,trailText",
        # Contributor tags tell us who wrote the piece, with a stable id.
        "show-tags": "contributor",
        "api-key": api_key,
    }
    results: list[dict] = []
    page, pages = 1, 1
    while page <= pages:
        data = _get(session, {**params, "page": page})
        results.extend(data["results"])
        pages = data["pages"]
        page += 1
        time.sleep(SECONDS_BETWEEN_CALLS)
    return results


def _get(session: requests.Session, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries + 1):
        resp = session.get(SEARCH_URL, params=params, timeout=30)
        # 429 means "too many requests": wait longer each time, then retry.
        if resp.status_code == 429 and attempt < retries:
            time.sleep(5 * 2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()["response"]
    raise RuntimeError("unreachable")


def save_day(day: date, results: list[dict]) -> Path:
    path = raw_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first, then rename: an interrupted run never
    # leaves a half-written file that looks complete.
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def fetch_range(start: date, end: date, force: bool = False) -> None:
    """Fetch and save each day from start to end inclusive, skipping days already saved."""
    api_key = config.guardian_api_key()
    session = requests.Session()
    day = start
    while day <= end:
        if raw_path(day).exists() and not force:
            print(f"{day}: already saved, skipping")
        else:
            results = fetch_day(day, api_key, session)
            save_day(day, results)
            print(f"{day}: saved {len(results)} items")
        day += timedelta(days=1)


def to_article(item: dict) -> dict | None:
    """Turn one raw API result into the common article shape every outlet shares.

    Returns None for things that are not plain articles (live blogs, galleries,
    crosswords, ...), which we skip for now.
    """
    if item.get("type") != "article":
        return None
    fields = item.get("fields", {})
    body = (fields.get("bodyText") or "").strip()
    if not body:
        return None
    contributors = [t["webTitle"] for t in item.get("tags", []) if t.get("type") == "contributor"]
    return {
        "id": f"{OUTLET}:{item['id']}",
        "outlet": OUTLET,
        "published_at": item["webPublicationDate"],
        "section": item.get("sectionName"),
        "pillar": item.get("pillarName"),  # News / Opinion / Sport / Arts / Lifestyle
        "headline": fields.get("headline") or item.get("webTitle"),
        "byline": fields.get("byline"),
        "authors": contributors,
        "url": item.get("webUrl"),
        "body": body,
        "word_count": int(fields.get("wordcount") or 0),
    }
