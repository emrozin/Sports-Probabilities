"""The site-wide features: Today as the home page, search, favorite teams, player pages, and the
disk cache that keeps slow pages fast."""

import pickle

import pytest
from fastapi.testclient import TestClient

import app as site
import overview
from web import disk_cached


@pytest.fixture
def client(league_db, monkeypatch):
    monkeypatch.setattr(site, "DB_PATH", league_db)
    monkeypatch.setattr(overview, "DB_PATH", league_db)
    site._cache.clear()
    overview._cache.clear()
    return TestClient(site.app)


def test_today_is_the_home_page_and_the_premier_league_moved(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "<h1>Today</h1>" in home.text
    epl = client.get("/premier-league/")
    assert epl.status_code == 200
    assert "Premier League probabilities" in epl.text
    assert 'href="/premier-league/"' in home.text       # the sport tab points at the new address
    assert '<a href="/premier-league/" aria-current="page">Soccer</a>' in epl.text      # under the Soccer tab
    assert 'class="cup-switch"' in epl.text and 'href="/la-liga/"' in epl.text and 'href="/champions-league/"' in epl.text


def test_search_folds_accents_and_case():
    assert overview.fold("Nikola Jokić") == "nikola jokic"
    assert overview.fold("Nott'm Forest") == "nott m forest"


def test_search_finds_teams_in_every_sport(client):
    results = overview.search("team a")
    assert results[0]["name"] == "Team A" and results[0]["sport"] == "Premier League"
    eagles = overview.search("eagles")
    assert any(r["sport"] == "NFL" and r["kind"] == "Team" for r in eagles)
    assert overview.search("x") == []                    # one letter is too short to search
    assert overview.search("zzzz nobody") == []


def test_search_ranks_exact_matches_first(client):
    assert overview.search("team c")[0]["name"] == "Team C"
    assert all("team" in r["name"].lower() for r in overview.search("team"))


def test_search_page(client):
    page = client.get("/search/", params={"q": "team b"})
    assert page.status_code == 200
    assert "Team B" in page.text and 'href="/team/team-b/"' in page.text
    empty = client.get("/search/", params={"q": "nobody here"})
    assert "Nothing found" in empty.text
    assert client.get("/search/").status_code == 200     # no query: just the box


def test_team_pages_have_a_follow_button_and_cards_name_their_teams(client):
    team = client.get("/team/team-a/").text
    assert 'data-follow="epl:Team A"' in team
    assert "/static/site.js" in team                     # the script that remembers favorites


def test_unknown_players_are_not_found(client):
    assert client.get("/nba/player/999999/").status_code == 404
    assert client.get("/nfl/player/00-0000000/").status_code == 404


def test_disk_cache_computes_once_per_key(tmp_path):
    calls = []
    make = lambda: calls.append(1) or len(calls)
    assert disk_cached(tmp_path, "thing", "v1", make) == 1
    assert disk_cached(tmp_path, "thing", "v1", make) == 1     # read back from the file
    assert disk_cached(tmp_path, "thing", "v2", make) == 2     # a new key (a new update) computes again
    assert pickle.loads((tmp_path / "thing.pkl").read_bytes())["key"] == "v2"


def test_disk_cache_survives_a_broken_file(tmp_path):
    (tmp_path / "thing.pkl").write_bytes(b"not a pickle")
    assert disk_cached(tmp_path, "thing", "k", lambda: 42) == 42
    assert disk_cached(tmp_path, "thing", "k", lambda: 0) == 42
