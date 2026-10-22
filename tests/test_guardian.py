from datetime import date

from conftest import make_item

from ibias.sources import guardian


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.payload}


class FakeSession:
    """Serves two pages of results, like the real API would for a busy day."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        page = params["page"]
        return FakeResponse({"pages": 2, "results": [make_item(page * 10 + j) for j in range(3)]})


def test_fetch_day_follows_every_page(monkeypatch):
    monkeypatch.setattr(guardian, "SECONDS_BETWEEN_CALLS", 0)
    session = FakeSession()
    results = guardian.fetch_day(date(2026, 10, 21), "test-key", session)
    assert len(results) == 6
    assert [c["page"] for c in session.calls] == [1, 2]
    assert session.calls[0]["from-date"] == session.calls[0]["to-date"] == "2026-10-21"


def test_save_day_writes_gzipped_jsonl():
    path = guardian.save_day(date(2026, 10, 21), [make_item(1), make_item(2)])
    assert path.parts[-4:] == ("guardian", "2026", "10", "21.jsonl.gz")
    assert not path.with_name(path.name + ".part").exists()


def test_to_article_skips_non_articles_and_empty_bodies():
    assert guardian.to_article(make_item(1, type_="liveblog")) is None
    assert guardian.to_article(make_item(1, body="")) is None
    article = guardian.to_article(make_item(1))
    assert article["id"] == "guardian:politics/2026/oct/21/story-1"
    assert article["authors"] == ["A Reporter"]
    assert article["word_count"] == 12
