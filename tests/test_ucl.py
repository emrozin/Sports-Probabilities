"""Champions League: reading ESPN's soccer feed (from payloads shaped like its real answers), the
database, the model, the season simulation and the pages."""

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ucl import import_espn


def team(team_id, name, color="112233"):
    return {"id": team_id, "displayName": name, "shortDisplayName": name[:10], "abbreviation": name[:3].upper(),
            "color": color, "alternateColor": "ffffff"}


def event(event_id, when, home, away, hs=None, aw=None, stage="league-phase", season=2026, state="post",
          pens=None, leg=None, neutral=False, odds=None, country="Spain"):
    completed = state == "post"
    status = {"type": {"name": "STATUS_FULL_TIME" if completed else "STATUS_SCHEDULED", "state": state, "completed": completed}}
    competitors = []
    for side, t, s, p in (("home", home, hs, pens[0] if pens else None), ("away", away, aw, pens[1] if pens else None)):
        c = {"id": t["id"], "homeAway": side, "score": str(s if s is not None else 0), "team": t}
        if p is not None:
            c["shootoutScore"] = p
        competitors.append(c)
    comp = {"id": event_id, "date": when, "neutralSite": neutral, "venue": {"address": {"country": country}},
            "competitors": competitors, "odds": [odds] if odds else []}
    if leg:
        comp["leg"] = {"value": leg, "displayValue": f"{leg} Leg"}
    return {"id": event_id, "date": when, "season": {"year": season, "slug": stage}, "status": status, "competitions": [comp]}


DRAFTKINGS = {"provider": {"id": "100", "name": "DraftKings"},
              "moneyline": {"home": {"open": {"odds": "+650"}, "close": {"odds": "+650"}},
                            "draw": {"open": {"odds": "+475"}, "close": {"odds": "+475"}},
                            "away": {"open": {"odds": "-320"}, "close": {"odds": "-320"}}}}
REAL, AEK, PSG, ARS = team("86", "Real Madrid"), team("887", "AEK Athens"), team("160", "Paris Saint-Germain"), team("359", "Arsenal")


def test_parse_events_reads_results_fixtures_shootouts_and_odds():
    payload = {"events": [
        event("1", "2026-10-21T19:00Z", REAL, ARS, 2, 1),
        event("2", "2026-11-04T17:45Z", AEK, REAL, state="pre", odds=DRAFTKINGS, country="Greece"),
        event("3", "2026-05-30T16:00Z", PSG, ARS, 1, 1, stage="final", season=2025, pens=(4, 3), neutral=True, country="Hungary"),
    ]}
    games, teams = import_espn.parse_events(payload, "uefa.champions")
    played, upcoming, final = games
    assert (played["status"], played["home_score"], played["away_score"], played["date"], played["kickoff"]) == ("final", 2, 1, "2026-10-21", "15:00")
    assert upcoming["status"] == "scheduled" and upcoming["home_score"] is None
    assert upcoming["home_ml"] == pytest.approx(7.5) and upcoming["draw_ml"] == pytest.approx(5.75)
    assert upcoming["away_ml"] == pytest.approx(1 + 100 / 320)
    assert (final["home_pens"], final["away_pens"], final["neutral"], final["stage"], final["season"]) == (4, 3, 1, "final", 2025)
    assert {t["team_id"] for t in teams} == {"86", "359", "887", "160"}


class FakeFeed:
    """Serves payloads by (league, day); anything else is an empty day."""

    def __init__(self, days: dict):
        self.days = days
        self.asked = []

    def __call__(self, url, params=None, timeout=None):
        slug = url.split("/soccer/")[1].split("/")[0]
        self.asked.append((slug, params["dates"]))
        payload = self.days.get((slug, params["dates"]), {"events": []})
        return type("Response", (), {"status_code": 200, "json": staticmethod(lambda: payload)})()


def test_import_saves_games_teams_and_draftkings_odds(project, monkeypatch):
    monkeypatch.setattr(import_espn, "CACHE_DIR", project / "data" / "raw" / "espn_soccer")
    monkeypatch.setattr(import_espn, "DOMESTIC", {"esp.1": "Spain"})
    monkeypatch.setattr(import_espn, "EUROPE", {"uefa.champions": "Champions League"})
    calendar = {"leagues": [{"season": {"year": 2026}, "calendar": ["2026-10-18T07:00Z"]}], "events": []}
    feed = FakeFeed({
        ("uefa.champions", "20261021"): {"events": [event("1", "2026-10-21T19:00Z", REAL, ARS, 2, 1)]},
        ("uefa.champions", "20261104"): {"events": [event("2", "2026-11-04T17:45Z", AEK, REAL, state="pre", odds=DRAFTKINGS)]},
        ("esp.1", "20261015"): calendar,
        ("esp.1", "20261018"): {"events": [event("9", "2026-10-18T19:00Z", REAL, team("83", "Barcelona"), 1, 1, stage="2026-27-laliga")]},
    })
    import_espn.main([], today=date(2026, 10, 27), get=feed)
    conn = sqlite3.connect(project / "data" / "sports.db")
    soccer = pd.read_sql("SELECT * FROM soccer_games ORDER BY date", conn)
    assert list(soccer["event_id"]) == ["9", "1", "2"]
    ucl = pd.read_sql("""SELECT g.status, h.name AS home, a.name AS away, g.home_score, i.espn_id FROM games g
                         JOIN leagues l ON l.id = g.league_id AND l.code = 'UCL' JOIN teams h ON h.id = g.home_team_id
                         JOIN teams a ON a.id = g.away_team_id JOIN cup_game_info i ON i.game_id = g.id ORDER BY g.date""", conn)
    assert ucl.fillna(-1).values.tolist() == [["final", "Real Madrid", "Arsenal", 2, "1"], ["scheduled", "AEK Athens", "Real Madrid", -1, "2"]]
    prices = dict(conn.execute("SELECT outcome, price FROM odds WHERE bookmaker = 'DraftKings' AND is_closing = 1").fetchall())
    assert prices["draw"] == pytest.approx(5.75)
    assert conn.execute("SELECT color FROM soccer_teams WHERE team_id = '86'").fetchone()[0] == "112233"
    # a second run updates rather than duplicates
    import_espn.main([], today=date(2026, 10, 27), get=feed)
    assert conn.execute("SELECT COUNT(*) FROM cup_game_info").fetchone()[0] == 3        # 2 Champions League, 1 La Liga
    assert conn.execute("SELECT COUNT(*) FROM games g JOIN leagues l ON l.id = g.league_id WHERE l.code = 'ESP'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM soccer_games").fetchone()[0] == 3
    conn.close()


def test_a_league_season_asks_only_for_its_calendar_days():
    feed = FakeFeed({("esp.1", "20241015"): {"leagues": [{"season": {"year": 2024}, "calendar": ["2024-08-15T07:00Z", "2024-08-16T07:00Z"]}]}})
    days = import_espn.season_days(import_espn.Feed(feed), "esp.1", 2024)
    assert days == [date(2024, 8, 15), date(2024, 8, 16)]
    cup = FakeFeed({("uefa.champions", "20241015"): {"leagues": [{"season": {"year": 2024, "startDate": "2024-07-01T04:00Z",
                                                                             "endDate": "2025-07-01T03:59Z"}, "calendar": [{"label": "phases"}]}]}})
    days = import_espn.season_days(import_espn.Feed(cup), "uefa.champions", 2024)
    assert days[0] == date(2024, 7, 1) and days[-1] == date(2025, 7, 1) and len(days) == 366


# ---------------------------------------------------------------- the model

def simulated_europe(seed=0, seasons=(2023, 2024, 2025), euro_games=150):
    """Three leagues of known strength (England best, Scotland worst) and European games between
    their clubs."""
    rng = np.random.default_rng(seed)
    level = {"eng.1": 0.35, "por.1": 0.0, "sco.1": -0.35}
    clubs = {f"{lg}-{i}": (lg, v + rng.normal(0, 0.15)) for lg, v in level.items() for i in range(8)}
    rows, eid = [], 0

    def game(comp, season, day, h, a, stage):
        nonlocal eid
        s = clubs[h][1] - clubs[a][1]
        hg, ag = rng.poisson(np.exp(0.3 + 0.12 + s / 2)), rng.poisson(np.exp(0.3 - s / 2))
        eid += 1
        rows.append({"event_id": str(eid), "competition": comp, "season": season, "stage": stage, "date": day, "kickoff": "15:00",
                     "status": "final", "neutral": 0, "leg": None, "home_id": h, "away_id": a, "home_score": hg, "away_score": ag,
                     "home_pens": None, "away_pens": None, "extra_time": 0})
    for season in seasons:
        start = pd.Timestamp(f"{season}-08-10")
        for lg in level:
            members = [c for c, (l, _) in clubs.items() if l == lg]
            k = 0
            for h in members:
                for a in members:
                    if h != a:
                        k += 1
                        game(lg, season, (start + pd.Timedelta(days=2 * k)).strftime("%Y-%m-%d"), h, a, f"{season}-league")
        for j in range(euro_games):
            h, a = rng.choice(list(clubs), 2, replace=False)
            game("uefa.champions", season, (start + pd.Timedelta(days=30 + j)).strftime("%Y-%m-%d"), h, a, "league-phase")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True), clubs


def test_model_finds_each_leagues_strength_from_the_european_games():
    from ucl import model as m
    games, clubs = simulated_europe()
    fit = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    league = fit["leagues"].sum(axis=1)
    assert league["eng.1"] > league["por.1"] > league["sco.1"]
    rated = pd.Series({c: m.rating(fit, c) for c in clubs})
    true = pd.Series({c: v for c, (_, v) in clubs.items()})
    assert np.corrcoef(rated, true)[0, 1] > 0.8
    assert 0.05 < fit["home_europe"] < 0.3
    p = m.predict(fit, "eng.1-0", "sco.1-0")
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1)
    assert m.predict(fit, "nobody-new", "eng.1-0")["away"] > 0.5         # an unknown club gets the weakest league's level


def test_ties_and_finals():
    from ucl import model as m
    games, _ = simulated_europe()
    fit = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    strong, weak = "eng.1-0", "sco.1-0"
    p = m.tie_chance(fit, strong, weak)
    assert 0.5 < p < 1 and 0 < m.tie_chance(fit, weak, strong) < 0.5
    assert abs(m.tie_chance(fit, weak, strong) - (1 - p)) < 0.01             # who hosts extra time matters only a little
    assert m.tie_chance(fit, weak, strong, first_leg=(4, 0)) > 0.8      # a four-goal first-leg lead turns it around
    assert m.tie_chance(fit, strong, weak, first_leg=(0, 5)) < 0.1
    f = m.single_game_winner(fit, strong, weak)
    assert m.single_game_winner(fit, weak, strong) == pytest.approx(1 - f, abs=1e-6) and f > 0.5


def test_result_after_90_minutes_counts_extra_time_as_a_draw():
    from ucl.model import result_of
    assert result_of(2, 1) == "home" and result_of(1, 1) == "draw" and result_of(2, 1, extra_time=1) == "draw"


def test_season_simulation_adds_up_and_follows_strength():
    from ucl import model as m, season as s
    games, clubs = simulated_europe()
    fit = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    rng = np.random.default_rng(2)
    teams = list(clubs) + [f"x-{i}" for i in range(12)]
    rows = []
    for md in range(8):
        order = rng.permutation(teams)
        for k in range(18):
            played = md < 3
            rows.append({"event_id": f"lp{md}-{k}", "competition": "uefa.champions", "season": 2026, "stage": "league-phase",
                         "date": f"2026-{9 + md // 4:02d}-{10 + (md % 4) * 5:02d}", "kickoff": "15:00",
                         "status": "final" if played else "scheduled", "neutral": 0, "leg": None,
                         "home_id": order[2 * k], "away_id": order[2 * k + 1],
                         "home_score": int(rng.integers(0, 4)) if played else None, "away_score": int(rng.integers(0, 3)) if played else None,
                         "home_pens": None, "away_pens": None, "extra_time": 0})
    allg = pd.concat([games, pd.DataFrame(rows)], ignore_index=True)
    odds = s.simulate(allg, 2026, fit, sims=2000)
    totals = odds[s.COLUMNS].sum()
    assert totals.to_dict() == pytest.approx({"top8": 8, "top24": 24, "round_of_16": 16, "quarterfinals": 8,
                                              "semifinals": 4, "final": 2, "champion": 1})
    english = odds.loc[[c for c in odds.index if c.startswith("eng")], "champion"].sum()
    scottish = odds.loc[[c for c in odds.index if c.startswith("sco")], "champion"].sum()
    assert english > scottish
    assert (odds["points"] >= 0).all() and odds["position"].between(1, 36).all()


def test_elo_blend_for_league_games():
    from ucl import elo, model as m
    games, clubs = simulated_europe()
    diffs, ratings = elo.run(games)
    assert len(diffs) == len(games) and set(ratings) == set(clubs)
    coef = elo.coefficients(games, diffs, competitions=("eng.1", "por.1", "sco.1"), tuning=(2023, 2025))
    strong = max(clubs, key=lambda c: clubs[c][1])
    weak = min(clubs, key=lambda c: clubs[c][1])
    q = elo.chances(ratings[strong] - ratings[weak] + elo.HOME, coef)
    assert q["home"] > q["away"] and sum(q.values()) == pytest.approx(1)
    fit = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    plain = m.predict(fit, strong, weak, europe=False)
    fit["elo"] = {"ratings": ratings, "coef": coef}
    mixed = m.predict(fit, strong, weak, europe=False)
    assert mixed["home"] == pytest.approx((1 - elo.LEAGUE_BLEND) * plain["home"] + elo.LEAGUE_BLEND * q["home"])
    assert mixed["home"] + mixed["draw"] + mixed["away"] == pytest.approx(1)
    assert m.predict(fit, strong, weak, europe=True)["home"] == pytest.approx(m.predict({k: v for k, v in fit.items() if k != "elo"},
                                                                                         strong, weak, europe=True)["home"])
    assert mixed["exp_home"] == plain["exp_home"]                   # the goals stay the model's


def test_league_season_simulation_adds_up():
    from ucl import model as m, season as s
    games, clubs = simulated_europe()
    fit = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    english = [c for c in clubs if c.startswith("eng")]
    rows = []
    for k, (h, a) in enumerate((h, a) for h in english for a in english if h != a):
        played = k < 10
        rows.append({"event_id": f"x{k}", "competition": "eng.1", "season": 2026, "stage": "2026-27-league",
                     "date": (pd.Timestamp("2026-08-15") + pd.Timedelta(days=3 * k)).strftime("%Y-%m-%d"), "kickoff": "15:00",
                     "status": "final" if played else "scheduled", "neutral": 0, "leg": None, "home_id": h, "away_id": a,
                     "home_score": 3 if played and h == english[0] else 0 if played else None,
                     "away_score": 0 if played else None, "home_pens": None, "away_pens": None, "extra_time": 0})
    allg = pd.concat([games, pd.DataFrame(rows)], ignore_index=True)
    odds = s.simulate_league(allg, 2026, fit, "eng.1", sims=2000)
    assert odds["champion"].sum() == pytest.approx(1) and odds["top4"].sum() == pytest.approx(4)
    assert odds["bottom3"].sum() == pytest.approx(3)
    assert odds["position"].between(1, 8).all() and len(odds) == 8
    assert (odds["points"] >= 0).all()
    assert odds.loc[english[0], "champion"] > 0.5                  # won all its games so far


# ---------------------------------------------------------------- the pages

@pytest.fixture
def ucl_project(project, monkeypatch):
    """A database with simulated league and European games, this season's league phase (two rounds
    played, the rest to come, with odds) and the predictions and odds the update makes."""
    from datetime import datetime
    import ucl.site
    from ucl import predict_upcoming, season
    games, clubs = simulated_europe()
    rng = np.random.default_rng(3)
    names = {c: f"Club {c.upper()}" for c in clubs}
    teams = list(clubs)[:24] + [f"x-{i}" for i in range(12)]
    rows = []
    for md in range(8):
        order = rng.permutation(teams)
        for k in range(18):
            played = md < 2
            day = (pd.Timestamp("2026-10-20") + pd.Timedelta(days=14 * (md - 1))).strftime("%Y-%m-%d")
            rows.append({"event_id": f"lp{md}-{k}", "competition": "uefa.champions", "season": 2026, "stage": "league-phase",
                         "date": day, "kickoff": "15:00", "status": "final" if played else "scheduled", "neutral": 0, "leg": None,
                         "home_id": order[2 * k], "away_id": order[2 * k + 1],
                         "home_score": int(rng.integers(0, 4)) if played else None, "away_score": int(rng.integers(0, 3)) if played else None,
                         "home_pens": None, "away_pens": None, "extra_time": 0, "country": "Spain"})
    # a La Liga season among eight of the clubs: 20 games played, the rest after the next eight days
    spanish = [c for c in clubs if c.startswith("por")]
    pairs = [(h, a) for h in spanish for a in spanish if h != a]
    for k, (h, a) in enumerate(rng.permutation(pairs)):
        played = k < 20
        day = (pd.Timestamp("2026-08-20") + pd.Timedelta(days=3 * k if played else 3 * k + 40)).strftime("%Y-%m-%d")
        rows.append({"event_id": f"es{k}", "competition": "esp.1", "season": 2026, "stage": "2026-27-laliga",
                     "date": day, "kickoff": "20:00", "status": "final" if played else "scheduled", "neutral": 0, "leg": None,
                     "home_id": h, "away_id": a, "home_score": int(rng.integers(0, 4)) if played else None,
                     "away_score": int(rng.integers(0, 3)) if played else None, "home_pens": None, "away_pens": None,
                     "extra_time": 0, "country": "Spain"})
    allg = pd.concat([games, pd.DataFrame(rows)], ignore_index=True)
    for col in import_espn.COLUMNS:
        if col not in allg:
            allg[col] = None
    upcoming = allg["event_id"] == "lp2-0"
    allg.loc[upcoming, ["home_ml_open", "draw_ml_open", "away_ml_open", "home_ml", "draw_ml", "away_ml"]] = [2.1, 3.4, 3.5, 2.0, 3.5, 3.7]
    folder = project / "data" / "raw" / "espn_soccer"
    folder.mkdir(parents=True)
    for (comp, s), part in allg.groupby(["competition", "season"]):
        part[import_espn.COLUMNS].to_csv(folder / f"{comp}_{s}.csv", index=False)
    team_rows = [{"team_id": t, "name": names.get(t, f"Club {t.upper()}"), "short_name": t, "abbreviation": t[:3],
                  "color": "c8102e" if t.startswith("eng") else "004d98", "alternate_color": "ffffff"} for t in set(allg["home_id"]) | set(allg["away_id"])]
    pd.DataFrame(team_rows).to_csv(folder / "teams.csv", index=False)
    conn = sqlite3.connect(project / "data" / "sports.db")
    conn.executescript((project / "schema.sql").read_text())
    import_espn.load(conn, folder)
    conn.close()
    predict_upcoming.main(["--record"], now=datetime(2026, 10, 27, 9, 0, tzinfo=predict_upcoming.EASTERN))
    season.SIMULATIONS = 500
    season.main(["--record"], today=date(2026, 10, 27))
    monkeypatch.setattr(ucl.site, "DB_PATH", project / "data" / "sports.db")
    ucl.site._cache.clear()
    return project


def test_live_predictions_are_saved_and_recorded(ucl_project):
    conn = sqlite3.connect(ucl_project / "data" / "sports.db")
    n = conn.execute("SELECT COUNT(DISTINCT game_id) FROM predictions WHERE model_name = 'europe_goals'").fetchone()[0]
    assert n == 18                                    # the next round, within eight days
    record = pd.read_csv(ucl_project / "records" / "europe_live_predictions.csv")
    assert len(record) == 18 and (record[["home_win", "draw", "away_win"]].sum(axis=1).round(6) == 1).all()
    odds = pd.read_csv(ucl_project / "records" / "europe_season_odds.csv")
    assert odds.groupby("competition")["champion"].sum().to_dict() == pytest.approx({"UCL": 1, "ESP": 1})
    conn.close()


def test_champions_league_pages_load(ucl_project):
    from fastapi.testclient import TestClient
    import app
    client = TestClient(app.app)
    for path in ["/champions-league/", "/champions-league/results/", "/champions-league/table/", "/champions-league/ratings/",
                 "/champions-league/track-record/", "/champions-league/match/lp2-0/", "/champions-league/match/lp0-0/",
                 "/champions-league/team/club-eng-1-0/", "/methodology/", "/europa-league/", "/europa-league/table/",
                 "/conference-league/", "/conference-league/track-record/"]:
        response = client.get(path)
        assert response.status_code == 200, path
    upcoming = client.get("/champions-league/").text
    assert "Club ENG.1-0" in upcoming or "Club " in upcoming
    assert "Market" in upcoming                         # DraftKings' odds on the one game that has them
    table = client.get("/champions-league/table/").text
    assert "Favorites to win it" in table and "zone-line" in table
    assert 'class="cup-switch"' in upcoming and "/conference-league/" in upcoming     # the competition switcher
    assert client.get("/champions-league/match/nope/").status_code == 404
    assert '<a href="/premier-league/" aria-current="page">Soccer</a>' in upcoming      # one Soccer tab for every competition
    for path in ["/la-liga/", "/la-liga/results/", "/la-liga/table/", "/la-liga/ratings/", "/la-liga/track-record/",
                 "/la-liga/team/club-por-1-0/", "/la-liga/match/es0/", "/bundesliga/", "/serie-a/table/", "/ligue-1/results/"]:
        assert client.get(path).status_code == 200, path
    liga = client.get("/la-liga/table/").text
    assert "Favorites for the title" in liga and "Bottom 3" in liga and "Club POR.1-0" in liga
    assert 'href="/la-liga/table/" aria-current="page"' in liga and 'href="/table/"' in liga   # the switcher keeps the page type
    team_page = client.get("/la-liga/team/club-por-1-0/").text
    assert "Win the league" in team_page
    assert client.get("/champions-league/team/nobody/").status_code == 404


def test_transfermarkt_clubs_are_linked_by_matching_games():
    from ucl.squad_values import club_mapping
    rng = np.random.default_rng(0)
    rows_tm, rows_ours = [], []
    for k in range(40):
        day = (pd.Timestamp("2024-08-01") + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        h, a = rng.integers(0, 4, 2)
        rows_tm.append({"date": day, "home_club_id": 11 + k % 2, "away_club_id": 13, "home_club_goals": h, "away_club_goals": a})
        rows_ours.append({"date": day, "home_id": ["A", "B"][k % 2], "away_id": "C", "home_score": h, "away_score": a})
    rows_tm.append({"date": "2024-08-01", "home_club_id": 99, "away_club_id": 98, "home_club_goals": 9, "away_club_goals": 9})
    mapping = club_mapping(pd.DataFrame(rows_tm), pd.DataFrame(rows_ours))
    assert mapping == {11: "A", 12: "B", 13: "C"}            # clubs with too few matched games aren't linked


def test_squad_values_inform_ratings():
    from ucl import model as m
    games, clubs = simulated_europe(euro_games=20)            # few European games: leagues are hard to compare
    true = pd.Series({c: v for c, (_, v) in clubs.items()})
    values = pd.DataFrame([{"espn_id": c, "month": month, "value_eur": float(np.exp(18 + 2 * v))}
                           for c, v in true.items() for month in ("2023-07-01", "2024-07-01", "2025-07-01", "2026-07-01")])
    without = m.fit(m.window(games, "2026-08-01"), "2026-08-01")
    with_values = m.fit(m.window(games, "2026-08-01"), "2026-08-01", values=values)
    assert with_values["value_attack"] > 0 and with_values["value_defense"] > 0
    corr = lambda f: np.corrcoef(pd.Series({c: m.rating(f, c) for c in clubs}), true)[0, 1]
    assert corr(with_values) > corr(without)
    hv, av, _ = m.value_columns(games.head(5), values)
    assert len(hv) == 5 and np.isfinite(hv).all()


def test_understat_clubs_and_games_are_linked_to_espn():
    from ucl.import_understat import club_mapping, link, understat_games
    rng = np.random.default_rng(4)
    names = {"Real Madrid": "86", "Barcelona": "83", "Sevilla": "243", "Valencia": "94"}
    us_rows, espn_rows, k = [], [], 0
    for day in range(30):
        for h, a in [("Real Madrid", "Barcelona"), ("Sevilla", "Valencia")] if day % 2 else [("Barcelona", "Sevilla"), ("Valencia", "Real Madrid")]:
            k += 1
            hs, as_ = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            date = (pd.Timestamp("2024-08-15") + pd.Timedelta(days=7 * day)).strftime("%Y-%m-%d")
            espn_date = (pd.Timestamp(date) + pd.Timedelta(days=k % 2)).strftime("%Y-%m-%d")   # a day apart sometimes (time zones)
            us_rows.append({"id": str(k), "isResult": True, "datetime": f"{date} 20:00:00", "h": {"title": h}, "a": {"title": a},
                            "goals": {"h": str(hs), "a": str(as_)}, "xG": {"h": "1.4", "a": "0.6"}})
            espn_rows.append({"event_id": f"e{k}", "season": 2024, "date": espn_date, "home_id": names[h], "away_id": names[a],
                              "home_score": hs, "away_score": as_})
    us_rows.append({"id": "x", "isResult": False, "datetime": "2025-05-01 20:00:00", "h": {"title": "Sevilla"},
                    "a": {"title": "Barcelona"}, "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}})
    us = understat_games("La_liga", [2024], fetch=lambda season, league: us_rows)
    assert len(us) == 60                                     # only played games
    espn = pd.DataFrame(espn_rows)
    mapping = club_mapping(us, espn)
    assert mapping == names
    linked = link(us, espn, mapping)
    assert len(linked) == 60 and linked["score_agrees"].all() and set(linked["event_id"]) == set(espn["event_id"])


def test_the_model_can_learn_from_xg():
    from ucl import model as m
    games, clubs = simulated_europe(seasons=(2024, 2025))
    games["home_xg"], games["away_xg"] = np.nan, np.nan
    plain = m.fit(m.window(games, "2026-08-01"), "2026-08-01", xg_weight=0.5)
    games.loc[games["competition"] == "eng.1", ["home_xg", "away_xg"]] = [3.0, 0.2]   # every English home side dominates on xG
    with_xg = m.fit(m.window(games, "2026-08-01"), "2026-08-01", xg_weight=0.5)
    assert with_xg["home_domestic"] > plain["home_domestic"]
    same = m.fit(m.window(games, "2026-08-01"), "2026-08-01", xg_weight=0.0)    # weight 0: xG ignored
    assert same["home_domestic"] == pytest.approx(plain["home_domestic"], rel=1e-6)


def test_backtest_fits_the_leagues_with_xg_and_the_cups_without(monkeypatch):
    from ucl import model as m
    games, _ = simulated_europe(seasons=(2024, 2025))
    games["home_xg"], games["away_xg"] = 1.0, 1.0
    used = []
    real_fit = m.fit
    monkeypatch.setattr(m, "fit", lambda *a, **k: used.append(k["xg_weight"]) or real_fit(*a, **k))
    last = games[games["season"] == 2025]
    days = {last[last["competition"] == "eng.1"]["date"].max(), last[last["competition"] == "uefa.champions"]["date"].max()}
    out = m.backtest(games, 2025, dates=days, competitions=["eng.1", m.UCL])
    assert set(out["competition"]) == {"eng.1", m.UCL}
    assert set(used) == {m.XG_WEIGHT, m.LEAGUE_XG_WEIGHT}
