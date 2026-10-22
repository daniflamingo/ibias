import pytest

from ibias import config


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Point every test at an empty temporary data folder, never your real data/."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "ibias.duckdb")
    monkeypatch.setattr(config, "FEATURES_DIR", tmp_path / "features")
    return tmp_path


def make_item(i, pillar="News", section="Politics", type_="article", body=None):
    """A made-up item shaped like a Guardian /search result."""
    return {
        "id": f"politics/2026/oct/21/story-{i}",
        "type": type_,
        "sectionName": section,
        "pillarName": pillar,
        "webPublicationDate": "2026-10-21T09:00:00Z",
        "webTitle": f"Story {i}",
        "webUrl": f"https://www.theguardian.com/politics/2026/oct/21/story-{i}",
        "fields": {
            "headline": f"Story {i}",
            "byline": "A Reporter",
            "bodyText": body if body is not None else f"The prime minister met union leaders about public services, story {i}.",
            "wordcount": "12",
        },
        "tags": [{"type": "contributor", "webTitle": "A Reporter"}],
    }
