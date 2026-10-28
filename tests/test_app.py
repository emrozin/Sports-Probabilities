"""The website: every page loads, missing pages 404, and every link leads somewhere real."""

import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as site
from epl import run_backtest


@pytest.fixture
def client(league_db, monkeypatch):
    import nfl.site
    monkeypatch.setattr(site, "DB_PATH", league_db)
    monkeypatch.setattr(nfl.site, "DB_PATH", league_db)   # a Premier League-only database: NFL pages are empty
    site._cache.clear()
    nfl.site._cache.clear()
    run_backtest.main()  # writes backtest predictions into the test database
    return TestClient(site.app)


def test_every_page_loads(client):
    for path in ["/premier-league/", "/results/", "/table/", "/table/2020/", "/team/team-a/", "/team/team-a/all/",
                 "/team/team-a/2019/", "/backtest/", "/ratings/"]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "Premier League probabilities" in response.text


def test_missing_pages_show_the_not_found_page(client):
    for path in ["/team/nobody/", "/team/team-a/1900/", "/table/1900/", "/results/1900-01-01/", "/no-such-page/"]:
        response = client.get(path)
        assert response.status_code == 404, path
        assert "Page not found" in response.text


def test_every_link_on_the_site_works(client):
    """Follow every internal link from the home page, exactly as the static site build does."""
    queue, seen = ["/"], {"/"}
    while queue:
        path = queue.pop()
        response = client.get(path)
        assert response.status_code == 200, f"broken link to {path}"
        for link in re.findall(r'(?:href|value)="(/[^"#?]*)"', response.text):
            if not link.startswith("/static/") and link not in seen:
                seen.add(link)
                queue.append(link)
    assert len(seen) > 50  # it really did crawl the site


def test_results_page_highlights_the_winner(client):
    html = client.get("/results/").text
    assert 'is-finished' in html                       # finished matches are drawn as result cards
    assert "side-home winner" in html or "side-home level" in html or "side-home loser" in html


def test_premier_league_pages_match_the_other_soccer_pages(client):
    results = client.get("/results/").text
    assert 'class="season-picker"' in results and 'class="matches"' in results and "Previous week" not in results
    for label in ("Table and odds", "Club ratings", "Track record"):
        assert label in results                          # the same menu as the other soccer competitions
    assert 'href="/forecast/" ' not in results and 'href="/backtest/" ' not in results   # no longer menu items
    seasons = [int(v.strip("/").split("/")[-1]) for v in __import__("re").findall(r'option value="(/results/\d+/)"', results)]
    assert seasons and client.get(f"/results/{min(seasons)}/").status_code == 200
    old = client.get(f"/results/{min(seasons)}-10-01/", follow_redirects=False)  # an old weekly address
    assert old.status_code == 301 and old.headers["location"] == f"/results/{min(seasons)}/"
    table = client.get("/table/").text
    assert "Favorites for the title" in table and ">Title<" in table and ">Bottom 3<" in table


def test_percentages_always_add_up_to_100():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        p = rng.dirichlet([1, 1, 1])
        shown = site.percentages(*p)
        assert sum(shown.values()) == 100
        assert all(abs(shown[o] - q * 100) < 1 for o, q in zip(("home", "draw", "away"), p))


def test_team_slugs():
    assert site.slugify("Nott'm Forest") == "nottm-forest"
    assert site.slugify("Man United") == "man-united"
    assert site.slugify("Brighton & Hove Albion") == "brighton-and-hove-albion"


def test_league_table_ranks_by_points_then_goal_difference_then_goals():
    games = pd.DataFrame([
        # A: win + loss (3 pts, GD 0, GF 4).  B: loss + win (3 pts, GD +1, GF 3).  C: 2 losses... etc.
        {"season": 2024, "date": "2024-08-10", "home": "A", "away": "B", "home_score": 3, "away_score": 1},
        {"season": 2024, "date": "2024-08-17", "home": "B", "away": "A", "home_score": 2, "away_score": 1},
        {"season": 2024, "date": "2024-08-24", "home": "B", "away": "C", "home_score": 0, "away_score": 0},
        {"season": 2024, "date": "2024-08-24", "home": "C", "away": "A", "home_score": 0, "away_score": 0},
    ]).assign(home_xg=np.nan, away_xg=np.nan)
    table = site.league_table(games, 2024).set_index("team")
    assert table.loc["A", "points"] == 4 and table.loc["B", "points"] == 4 and table.loc["C", "points"] == 2
    assert table.loc["A", "gd"] == 1 and table.loc["B", "gd"] == -1
    assert list(table.sort_values("position").index) == ["A", "B", "C"]  # level on points: A's better GD wins
    assert table.loc["A", "form"] == ["W", "L", "D"]


def test_goals_markets_appear_on_the_site(client):
    assert "Over 2.5 goals" in client.get("/results/").text
    assert "Chance of a" in client.get("/team/team-a/").text
    backtest = client.get("/backtest/").text
    assert "Over/under 2.5 goals and exact scores" in backtest


def test_match_pages_for_finished_matches(client):
    results = client.get("/results/").text
    match_links = re.findall(r'href="(/match/[^"]+)"', results)
    assert match_links, "results page should link each score to its match page"
    page = client.get(match_links[0])
    assert page.status_code == 200
    html = page.text
    assert "Every possible score" in html
    assert html.count('class="cell ') == 36          # a 6 x 6 heat map
    assert html.count("cell-actual") == 1 or "falls in that group" in html
    assert "Form going into the match" in html and "Head to head" in html


def test_unknown_matches_show_the_not_found_page(client):
    assert client.get("/match/2026-10-10/team-a-v-nobody/").status_code == 404
    assert client.get("/match/1999-01-01/team-a-v-team-b/").status_code == 404


def test_heatmap_marks_only_the_actual_score():
    from epl.model import scoreline_grid
    grid = scoreline_grid(1.5, 1.1)
    hm = site.heatmap(grid, actual=(2, 1))
    marked = [(r["goals"], j) for r in hm["rows"] for j, c in enumerate(r["cells"]) if c["actual"]]
    assert marked == [(2, 1)]
    assert hm["rows"][2]["cells"][1]["outcome"] == "home"
    assert hm["rows"][1]["cells"][1]["outcome"] == "draw"
    assert site.heatmap(grid, actual=(7, 0))["actual_off_grid"]


def test_season_forecast_page(client):
    html = client.get("/forecast/").text
    assert "Season forecast" in html and "Chance of finishing in each position" in html
    assert html.count('class="pos-cell') == 10 * 10     # 10 simulated teams x 10 positions
    assert "/forecast/" in client.get("/table/").text    # linked from the table
