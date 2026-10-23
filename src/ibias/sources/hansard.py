"""House of Commons speeches with each speaker's party, from TheyWorkForYou.

mySociety's ParlParse project turns Hansard into clean XML, one file per sitting day:
    https://www.theyworkforyou.com/pwdata/scrapedxml/debates/debates2025-03-12a.xml
When Hansard corrects a day, a new version appears with the next letter (b, c, ...),
so for each date we take the highest letter.

Each speech looks like this (simplified):
    <speech id="uk.org.publicwhip/debate/2025-03-12a.301.2"
            speakername="Rachel Reeves" person_id="uk.org.publicwhip/person/24822">
        <p>The Government are delivering ...</p>
    </speech>

The XML does not say which party the speaker belongs to. That comes from
people.json, ParlParse's list of every MP and their party over time.
"""

import gzip
import json
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import requests
from tqdm import tqdm

from ibias import config

OUTLET = "hansard"
DEBATES_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
PEOPLE_URL = "https://raw.githubusercontent.com/mysociety/parlparse/master/members/people.json"
# Be polite to a small charity's server.
SECONDS_BETWEEN_CALLS = 1.0

FILE_RE = re.compile(r"debates(\d{4}-\d{2}-\d{2})([a-z])\.xml")

# Labour and Labour/Co-operative MPs sit as one party.
PARTY_NAMES = {"labour": "Labour", "labourco-operative": "Labour", "conservative": "Conservative"}


def raw_dir() -> Path:
    return config.RAW_DIR / OUTLET


def people_path() -> Path:
    return raw_dir() / "people.json.gz"


# ---------------------------------------------------------------- fetching


def latest_versions(index_html: str, start: date, end: date) -> dict[str, str]:
    """From the directory listing, pick the newest file for each date in range."""
    latest: dict[str, str] = {}
    for day, letter in set(FILE_RE.findall(index_html)):
        if start.isoformat() <= day <= end.isoformat() and letter > latest.get(day, ""):
            latest[day] = letter
    return {day: f"debates{day}{letter}.xml" for day, letter in sorted(latest.items())}


def debate_path(filename: str) -> Path:
    year = filename[len("debates") : len("debates") + 4]
    return raw_dir() / "debates" / year / f"{filename}.gz"


def _download(session: requests.Session, url: str, dest: Path) -> None:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with gzip.open(tmp, "wb") as f:
        f.write(resp.content)
    tmp.replace(dest)


def fetch_people(session: requests.Session | None = None) -> Path:
    """Download the MP list. Rerun now and then to pick up new MPs and party switches."""
    _download(session or requests.Session(), PEOPLE_URL, people_path())
    return people_path()


def fetch_range(start: date, end: date, force: bool = False) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "iBias research project (personal, non-commercial)"
    if force or not people_path().exists():
        fetch_people(session)
        print("saved people.json")
    index = session.get(DEBATES_URL, timeout=60)
    index.raise_for_status()
    files = latest_versions(index.text, start, end)
    print(f"{len(files)} sitting days between {start} and {end}")
    for filename in tqdm(files.values(), desc="downloading debates", unit=" days"):
        dest = debate_path(filename)
        if dest.exists() and not force:
            continue
        _download(session, DEBATES_URL + filename, dest)
        time.sleep(SECONDS_BETWEEN_CALLS)


# ---------------------------------------------------------------- parties


class PartyLookup:
    """Answer "which party was this MP in on this date?" from people.json.

    people.json follows the Popolo standard: a list of `memberships`, each one
    a period (start_date to end_date) in which a person held a seat (`post_id`)
    for a party (`on_behalf_of_id`). Seats in other parliaments (Holyrood,
    the Senedd, ...) are memberships too, so we keep only House of Commons seats.
    """

    def __init__(self, people: dict):
        commons_posts = {p["id"] for p in people["posts"] if p.get("organization_id") == "house-of-commons"}
        self.by_person: dict[str, list[tuple[str, str, str]]] = {}
        self.by_member: dict[str, tuple[str, str, str]] = {}
        for m in people["memberships"]:
            if m.get("post_id") not in commons_posts or "person_id" not in m:
                continue
            period = (m.get("start_date", "0000"), m.get("end_date", "9999-12-31"), m.get("on_behalf_of_id", ""))
            self.by_person.setdefault(m["person_id"], []).append(period)
            self.by_member[m["id"]] = period
        self.member_person = {m["id"]: m.get("person_id") for m in people["memberships"]}

    @classmethod
    def load(cls, path: Path | None = None) -> "PartyLookup":
        with gzip.open(path or people_path(), "rt", encoding="utf-8") as f:
            return cls(json.load(f))

    def party(self, speaker_id: str | None, day: str) -> str | None:
        """Party id on `day` (YYYY-MM-DD) for a person id or an older-style member id."""
        if not speaker_id:
            return None
        if "/member/" in speaker_id:
            speaker_id = self.member_person.get(speaker_id) or ""
        for start, end, party in self.by_person.get(speaker_id, []):
            if start <= day <= end:
                return party
        return None


# ---------------------------------------------------------------- parsing


def parse_debate(path: Path, parties: PartyLookup) -> Iterator[dict]:
    """Yield one dict per speech in a day's debates file."""
    day = FILE_RE.search(path.name).group(1)
    with gzip.open(path, "rb") as f:
        root = ET.parse(f).getroot()
    for speech in root.iter("speech"):
        # Newer files use person_id; older ones a member id in speakerid.
        speaker = speech.get("person_id") or speech.get("speakerid")
        if not speaker or speaker == "unknown":
            continue  # procedural lines, e.g. "Question put and agreed to."
        text = " ".join("".join(p.itertext()).strip() for p in speech.iter("p")).strip()
        if not text:
            continue
        party_id = parties.party(speaker, day)
        yield {
            "id": speech.get("id"),
            "date": day,
            "speaker_id": speaker,
            "speaker_name": speech.get("speakername"),
            "party": PARTY_NAMES.get(party_id, party_id),
            "text": text,
        }


def iter_speeches(parties: PartyLookup | None = None) -> Iterator[dict]:
    parties = parties or PartyLookup.load()
    paths = sorted((raw_dir() / "debates").rglob("*.xml.gz"))
    for path in tqdm(paths, desc="parsing debates", unit=" days"):
        yield from parse_debate(path, parties)
