"""NBA: the importer (ESPN's schedule and box score files), the player model and lineups, ESPN's
live feeds (rosters, injuries, betting lines), player props, playoff odds and the website pages.
Everything runs on a simulated league; nothing touches the internet."""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import espn
import nba.site as nba_site
from nba import import_nba, live, players as nba_players, predict_players, predict_upcoming, records, season as nba_season
from nba.import_nba import BOX_COLUMNS, DIVISIONS, SCHEDULE_COLUMNS
from nba.model import backtest, fit, load_games, load_players, predict

ABBRS = [a for _, (_, teams) in DIVISIONS.items() for a in teams]           # the 30 teams
NOW = datetime(2025, 11, 20, 12, 0, tzinfo=predict_upcoming.EASTERN)
STAR = 0                                                                    # each team's player 0 is its star


def simulated_league(seed=0):
    """Three seasons (2023-24 to 2025-26) in ESPN's formats: schedules and player box scores.
    Each team has 9 players with known ratings; the 2025-26 season is played up to NOW."""
    rng = np.random.default_rng(seed)
    rating = {(t, k): (8.0 if k == STAR else rng.normal(0, 2)) * (1 + 0.1 * (t % 3)) for t in range(30) for k in range(9)}
    usual = np.array([34, 32, 30, 30, 28, 24, 20, 14, 8], float)
    schedule, box, gid = [], [], 401000000
    for season in (2023, 2024, 2025):
        start = pd.Timestamp(f"{season}-10-22T23:30Z")
        for rnd in range(20):
            order = rng.permutation(30)
            for k in range(15):
                home, away = int(order[2 * k]), int(order[2 * k + 1])
                gid += 1
                tip = start + pd.Timedelta(days=3 * rnd)
                played = tip < pd.Timestamp("2025-11-20T12:00Z")
                lines, strength = {}, {}
                for team in (home, away):
                    out = rng.random(9) < np.where(np.arange(9) == STAR, 0.2, 0.08)
                    mins = np.where(out, 0, usual)
                    mins = mins * 240 / mins.sum()
                    lines[team] = (mins, out)
                    strength[team] = sum(5 * mins[p] / 240 * rating[(team, p)] for p in range(9))
                margin = 3 + strength[home] - strength[away] + rng.normal(0, 11)
                total = 225 + rng.normal(0, 10)
                hs, as_ = int(round((total + margin) / 2)), int(round((total - margin) / 2))
                if hs == as_:
                    hs += 1
                notes = "NBA Cup Championship" if (season == 2024 and rnd == 10 and k == 0) else np.nan
                row = {"game_id": gid, "season": season + 1, "season_type": 2, "date": tip.strftime("%Y-%m-%dT%H:%MZ"),
                       "neutral_site": False, "status_type_completed": played,
                       "status_type_name": "STATUS_FINAL" if played else "STATUS_SCHEDULED", "notes_headline": notes}
                for side, team, score in (("home", home, hs), ("away", away, as_)):
                    row.update({f"{side}_id": team + 1, f"{side}_display_name": f"Club {ABBRS[team]}",
                                f"{side}_abbreviation": ABBRS[team], f"{side}_score": score if played else 0,
                                f"{side}_color": "112233", f"{side}_alternate_color": "FFFFFF"})
                schedule.append(row)
                if not played:
                    continue
                for team in (home, away):
                    mins, out = lines[team]
                    for p in range(9):
                        m = mins[p]
                        box.append({"game_id": gid, "athlete_id": (team + 1) * 100 + p, "athlete_display_name": f"{ABBRS[team]} Player {p}",
                                    "athlete_position_abbreviation": ["PG", "SG", "SF", "PF", "C"][p % 5], "team_id": team + 1,
                                    "starter": p < 5, "did_not_play": bool(out[p]), "reason": "INJURY" if out[p] else np.nan,
                                    "minutes": m if m > 0 else np.nan, "points": rng.poisson(m * (0.4 + 0.04 * rating[(team, p)])) if m else np.nan,
                                    "rebounds": rng.poisson(m * 0.18) if m else np.nan, "assists": rng.poisson(m * 0.1) if m else np.nan,
                                    "three_point_field_goals_made": rng.poisson(m * 0.04) if m else np.nan,
                                    "steals": 1, "blocks": 0, "turnovers": 2, "plus_minus": 0})
    # a game postponed (listed again later as game 401999999)
    postponed = dict(schedule[5], game_id=401999998, status_type_name="STATUS_POSTPONED", status_type_completed=False,
                     date="2025-11-01T23:30Z")
    schedule.append(postponed)
    return pd.DataFrame(schedule, columns=SCHEDULE_COLUMNS), pd.DataFrame(box, columns=BOX_COLUMNS), rating


@pytest.fixture(scope="module")
def league():
    return simulated_league()


@pytest.fixture
def nba_project(project, monkeypatch, league):
    schedule, box, _ = league
    db = project / "data" / "sports.db"
    conn = sqlite3.connect(db)
    import_nba.import_games(conn, schedule, box)
    games, lines = load_games(conn), load_players(conn)
    from nba.run_backtest import save_seasons
    save_seasons(conn, backtest(games, lines, 2024, 2025), [2024, 2025])
    conn.close()
    predict_upcoming.main(["--record"], now=NOW)
    predict_players.main(["--record"], now=NOW)
    monkeypatch.setattr(nba_site, "DB_PATH", db)
    monkeypatch.setattr(nba_site, "date", type("FrozenDate", (), {"today": staticmethod(lambda: NOW.date())}))
    nba_site._cache.clear()
    return project


# ---------------------------------------------------------------- importing

def test_import_keeps_teams_seasons_and_box_scores(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    assert [r[0] for r in conn.execute("SELECT DISTINCT season FROM games ORDER BY season")] == [2023, 2024, 2025]
    teams = pd.read_sql("SELECT * FROM nba_teams", conn)
    assert len(teams) == 30 and teams.groupby("conference").size().to_dict() == {"East": 15, "West": 15}
    assert conn.execute("SELECT COUNT(*) FROM nba_game_info WHERE source_game_id = '401999998'").fetchone()[0] == 0   # postponed
    assert conn.execute("SELECT COUNT(*) FROM nba_game_info WHERE season_type = 'cup_final'").fetchone()[0] == 1
    dnp = conn.execute("SELECT COUNT(*) FROM nba_player_games WHERE did_not_play = 1").fetchone()[0]
    assert dnp > 0
    assert conn.execute("SELECT COUNT(*) FROM games WHERE status = 'scheduled' AND home_score IS NOT NULL").fetchone()[0] == 0
    conn.close()


def test_reimporting_updates_instead_of_duplicating(project, league):
    schedule, box, _ = league
    conn = sqlite3.connect(project / "data" / "sports.db")
    first = import_nba.import_games(conn, schedule, box)
    second = import_nba.import_games(conn, schedule, box)
    assert second["new"] == 0 and second["updated"] == first["new"]
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == first["new"]
    conn.close()


# ---------------------------------------------------------------- the model

def test_model_finds_the_stars_and_a_missing_star_lowers_the_chance(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    games, lines = load_games(conn), load_players(conn)
    conn.close()
    model = fit(games, lines, "2025-11-20")
    stars = [str((t + 1) * 100 + STAR) for t in range(30)]
    others = [p for p in model["ratings"].index if p not in stars]
    assert model["ratings"][stars].mean() > model["ratings"][others].mean() + 1.5   # a small league: ratings stay near average
    team = "Club BOS"
    lineup = live.expected_lineup(team, live.latest_lines(lines), pd.DataFrame(columns=["team"]), set(), "2025-11-21")
    without = live.expected_lineup(team, live.latest_lines(lines), pd.DataFrame(columns=["team"]), {str(100 * 1 + STAR)}, "2025-11-21")
    other = live.expected_lineup("Club NY", live.latest_lines(lines), pd.DataFrame(columns=["team"]), set(), "2025-11-21")
    full = predict(model, team, "Club NY", lineup, other)["home_win"]
    missing = predict(model, team, "Club NY", without, other)["home_win"]
    assert missing < full - 0.02
    assert without.loc[without["player_id"] == "100", "minutes"].iloc[0] == 0


def test_backtest_modes_give_valid_chances(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    games, lines = load_games(conn), load_players(conn)
    conn.close()
    for mode in ("played", "actual", "usual", "none"):
        r = backtest(games, lines, 2025, 2025, mode=mode, with_points=False)
        assert len(r) > 100 and r["model_home"].between(0, 1).all(), mode


# ---------------------------------------------------------------- ESPN's feeds

def espn_event(game_id, home="Club BOS", away="Club NY", state="pre"):
    """A scoreboard event shaped like ESPN's (only the parts read here)."""
    return {"id": str(game_id), "status": {"type": {"state": state}},
            "competitions": [{"competitors": [{"homeAway": "home", "team": {"displayName": home}},
                                              {"homeAway": "away", "team": {"displayName": away}}],
                              "odds": [{"provider": {"name": "DraftKings"}, "details": "BOS -3.5", "spread": -3.5, "overUnder": 225.5,
                                        "moneyline": {"home": {"close": {"odds": "-155"}, "open": {"odds": "-175"}},
                                                      "away": {"close": {"odds": "+130"}, "open": {"odds": "+145"}}},
                                        "pointSpread": {"home": {"close": {"line": "-3.5", "odds": "-105"}}},
                                        "total": {"over": {"close": {"line": "o225.5", "odds": "-112"}}}}]}]}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP error")


def test_american_odds_and_lines_are_read():
    assert espn.decimal_odds("+130") == pytest.approx(2.30)
    assert espn.decimal_odds("-155") == pytest.approx(1 + 100 / 155)
    assert espn.decimal_odds("EVEN") == 2.0 and espn.decimal_odds("OFF") is None
    row = espn.parse_event(espn_event(1))
    assert row["spread"] == -3.5 and row["total"] == 225.5 and row["home_ml_open"] == pytest.approx(1 + 100 / 175)
    assert espn.parse_event({"id": "2", "competitions": [{"odds": []}]}) is None


def test_lines_are_saved_matched_by_espn_id_and_closing_lines_kept(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    gid, home, away = conn.execute("""SELECT i.source_game_id, h.name, a.name FROM nba_game_info i JOIN games g ON g.id = i.game_id
                                      JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                                      WHERE g.status = 'scheduled' ORDER BY g.date LIMIT 1""").fetchone()
    payload = {"events": [espn_event(gid, home, away, "post"), espn_event(999, "X", "Y")]}
    conn.close()
    espn.main(["--record"], today=NOW.date(), get=lambda url, params, timeout: FakeResponse(payload))
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    market = espn.market_lines(conn, "NBA")
    assert len(market) == 1 and market["market_home"].iloc[0] == pytest.approx((1 / (1 + 100 / 155)) / (1 / (1 + 100 / 155) + 1 / 2.3))
    later = espn.parse_event(espn_event(gid, home, away, "pre")) | {"home_ml": 3.0}
    espn.save(conn, "NBA", [later])                                   # an earlier line never replaces the closing one
    assert conn.execute("SELECT home_ml FROM espn_lines").fetchone()[0] == pytest.approx(1 + 100 / 155)
    conn.close()
    assert espn.RECORD_PATH.exists() and len(pd.read_csv(espn.RECORD_PATH)) == 1


def test_espn_down_is_not_an_error(nba_project, capsys):
    def broken(*args, **kwargs):
        raise ConnectionError("no network")
    espn.main([], today=NOW.date(), get=broken)
    live.main([], get=broken)
    out = capsys.readouterr().out
    assert "couldn't get lines" in out and "Couldn't get NBA rosters" in out


def test_rosters_and_injuries_are_read_and_leave_out_injured_players(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    teams = conn.execute("SELECT espn_id, name, abbreviation FROM nba_teams").fetchall()
    rosters = {str(i): {"athletes": [{"id": str(i * 100 + p), "displayName": f"{abbr} Player {p}", "position": {"abbreviation": "G"},
                                      "injuries": [{"status": "Day-To-Day"}] if p == 3 else []} for p in range(9)]}
               for i, _, abbr in teams}
    injuries = {"injuries": [{"displayName": "Club BOS", "injuries": [
        {"status": "Out", "athlete": {"displayName": "BOS Player 0", "team": {"abbreviation": "BOS"},
                                      "links": [{"href": "https://www.espn.com/nba/player/_/id/100/bos-player-0"}]},
         "details": {"returnDate": "2025-12-01", "fantasyStatus": {"abbreviation": "OUT"}}, "shortComment": "Ankle"}]}]}

    def get(url, timeout):
        return FakeResponse(injuries if url == live.INJURIES_URL else rosters[url.split("/teams/")[1].split("/")[0]])
    conn.close()
    live.main(["--record"], get=get)
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    ros, inj = live.load(conn)
    conn.close()
    assert len(ros) == 270 and set(inj["status"]) == {"Out", "Day-To-Day"}
    assert live.unavailable(inj) == {"100"}
    assert live.unavailable(inj, on="2025-12-05") == set()               # back by then
    lineup = live.expected_lineup("Club BOS", pd.DataFrame(columns=["team", "name", "position", "date", "minutes_after"]),
                                  ros, live.unavailable(inj), "2025-11-21")
    assert lineup.loc[lineup["player_id"] == "100", "out"].iloc[0] and lineup["minutes"].gt(0).sum() == 8


# ---------------------------------------------------------------- predictions and props

def test_predictions_are_recorded_and_not_changed_after_tip_off(nba_project):
    record = records.read_record()
    assert len(record) > 0 and np.allclose(record["home_win"] + record["away_win"], 1)
    assert record["date"].min() >= "2025-11-20" and record["date"].max() <= "2025-11-28"
    later = datetime(2025, 11, 26, 12, 0, tzinfo=predict_upcoming.EASTERN)
    predict_upcoming.main(["--record"], now=later)
    after = records.read_record()
    started = record[record["date"] < "2025-11-26"]
    kept = after[after["game"].isin(started["game"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, started.reset_index(drop=True))


def test_player_projections_follow_the_lineup(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    proj = pd.read_sql("SELECT * FROM nba_player_projections", conn)
    conn.close()
    totals = proj.groupby(["game_id", "team"])["minutes"].sum()
    assert not proj.empty and totals.between(235, 240.01).all()        # players under a minute aren't saved
    assert (proj[["points", "rebounds", "assists", "threes"]] >= 0).all().all()
    stars = proj[proj["player_id"].astype(int) % 100 == STAR]
    assert stars["points"].mean() > proj["points"].mean()
    assert predict_players.RECORD_PATH.exists()


def test_chances_and_likeliest_props():
    chance = nba_players.chance_at_least("points", [10, 20, 30], 20)
    assert chance[0] < chance[1] < chance[2] and 0.4 < chance[1] < 0.6
    proj = pd.DataFrame({"game_id": [1, 1, 2], "team": ["A", "A", "B"], "player_id": ["1", "2", "3"], "name": ["x", "y", "z"],
                         "proj_points": [28.0, 12.0, 22.0], "proj_rebounds": [5.0, 11.0, 4.0], "proj_assists": [8.0, 2.0, 3.0],
                         "proj_threes": [3.0, 0.5, 2.0]})
    picks = nba_players.likeliest_props(nba_players.prop_chances(proj), "game_id", 2)
    assert picks["player_id"].is_unique and set(picks["game_id"]) <= {1, 2}
    assert (picks["line"] >= picks["stat"].map(nba_players.LIKELIEST_MIN_LINE)).all()


# ---------------------------------------------------------------- playoff odds

def test_a_best_of_seven():
    assert nba_season.series_win(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(0.5)
    assert nba_season.series_win(np.array([0.6]), np.array([0.6]))[0] == pytest.approx(0.710208, abs=1e-6)
    assert nba_season.series_win(np.array([0.7]), np.array([0.5]))[0] > nba_season.series_win(np.array([0.6]), np.array([0.6]))[0] - 0.05


def test_playoff_simulation_adds_up(nba_project):
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    games, lines, teams = load_games(conn), load_players(conn), nba_season.team_info(conn)
    conn.close()
    odds = nba_season.simulate(nba_season.setup(games, lines, teams, 2025, "2025-11-20"), n=400)
    assert odds["top_six"].sum() == pytest.approx(12) and odds["playoffs"].sum() == pytest.approx(16)
    assert odds["conference_finals"].sum() == pytest.approx(4) and odds["finals"].sum() == pytest.approx(2)
    assert odds["champion"].sum() == pytest.approx(1)
    assert (odds["playoffs"] >= odds["top_six"]).all() and (odds["top_six"] >= odds["top_seed"]).all()


def test_standings_order_breaks_ties_by_head_to_head():
    members = np.array([0, 1, 2])
    wins = np.array([10, 10, 5.0])
    h2h = np.zeros((3, 3))
    h2h[1, 0] = 2                       # team 1 beat team 0 twice
    order = nba_season.order(members, wins, np.zeros(3), h2h, np.array([0.9, 0.1, 0.5]))
    assert list(order) == [1, 0, 2]


# ---------------------------------------------------------------- pages

@pytest.mark.parametrize("path", ["/nba/", "/nba/props/", "/nba/results/", "/nba/standings/", "/nba/standings/?view=division",
                                  "/nba/standings/2024/", "/nba/playoffs/", "/nba/ratings/", "/nba/team/club-bos/",
                                  "/nba/team/club-bos/2024/", "/nba/track-record/", "/nba/backtest/"])
def test_nba_pages_load(nba_project, path):
    import app
    response = TestClient(app.app).get(path)
    assert response.status_code == 200, path
    assert "NBA probabilities" in response.text


def test_game_pages_show_lineups_and_props(nba_project):
    import app
    client = TestClient(app.app)
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    upcoming_id = conn.execute("""SELECT i.source_game_id FROM nba_game_info i JOIN games g ON g.id = i.game_id
                                  JOIN nba_player_projections p ON p.game_id = g.id LIMIT 1""").fetchone()[0]
    finished_id = conn.execute("""SELECT i.source_game_id FROM nba_game_info i JOIN games g ON g.id = i.game_id
                                  WHERE g.status = 'final' LIMIT 1""").fetchone()[0]
    conn.close()
    page = client.get(f"/nba/game/{upcoming_id}/").text
    assert "Expected lineups" in page and "Player 0" in page
    assert "Box score" in client.get(f"/nba/game/{finished_id}/").text
    assert client.get("/nba/game/1/").status_code == 404
    assert client.get("/nba/team/not-a-team/").status_code == 404
    assert client.get("/nba/standings/1999/").status_code == 404
    assert "Upcoming games" in client.get("/nba/").text and "Game details" in client.get("/nba/").text


def test_nba_pages_work_with_no_data(project, monkeypatch):
    import app
    monkeypatch.setattr(nba_site, "DB_PATH", project / "data" / "sports.db")
    nba_site._cache.clear()
    client = TestClient(app.app)
    for path in ("/nba/", "/nba/props/", "/nba/results/", "/nba/standings/", "/nba/playoffs/", "/nba/ratings/",
                 "/nba/track-record/", "/nba/backtest/"):
        assert client.get(path).status_code == 200, path


def test_rerunning_the_backtest_replaces_a_season(nba_project):
    from nba import run_backtest
    run_backtest.main(["--full"])
    run_backtest.main([])            # the current season again: replaced, not duplicated
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    n, games = conn.execute("""SELECT COUNT(*), COUNT(DISTINCT game_id) FROM predictions
                               WHERE model_name = 'nba_players_backtest'""").fetchone()
    conn.close()
    assert n == 2 * games and games > 0


def test_lines_are_fetched_one_day_at_a_time(nba_project):
    asked = []

    def get(url, params, timeout):
        asked.append(params["dates"])
        if "-" in params["dates"]:
            return FakeResponse({"code": 400}, status=400)   # the real feed refuses ranges
        return FakeResponse({"events": [espn_event(1)]})     # the same game every day: kept once
    events = espn.fetch_events("NBA", datetime(2026, 10, 19).date(), datetime(2026, 10, 22).date(), get)
    assert asked == ["20261019", "20261020", "20261021", "20261022"] and len(events) == 1


def test_suggested_bets_are_logged_once_settled_and_shown(nba_project, monkeypatch):
    import app
    import betting
    import overview
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    gid, game_id = conn.execute("""SELECT i.source_game_id, g.id FROM nba_game_info i JOIN games g ON g.id = i.game_id
                                   JOIN predictions p ON p.game_id = g.id AND p.model_name = 'nba_players'
                                   WHERE g.status = 'scheduled' ORDER BY g.date LIMIT 1""").fetchone()
    p = conn.execute("SELECT probability FROM predictions WHERE game_id = ? AND outcome = 'home' AND model_name = 'nba_players'",
                     (game_id,)).fetchone()[0]
    price = 1 / max(p - 0.15, 0.05)                      # a price well above the model's chance: an edge on the home side
    espn.save(conn, "NBA", [{"game": gid, "provider": "test", "home_ml_open": None, "away_ml_open": None,
                             "home_ml": price, "away_ml": 1.3, "spread": -2.5, "total": 220.5, "final": 0}])
    conn.close()
    betting.main(["--record"], now=NOW)
    betting.main(["--record"], now=NOW)                   # logged once, not twice
    conn = sqlite3.connect(nba_project / "data" / "sports.db")
    bets = pd.read_sql("SELECT * FROM bets", conn)
    assert len(bets) == 1 and bets.at[0, "outcome"] == "home" and bets.at[0, "edge"] >= betting.MIN_EDGE
    conn.execute("UPDATE games SET status = 'final', home_score = 110, away_score = 100 WHERE id = ?", (game_id,))
    conn.commit()
    log = betting.logged(conn)
    conn.close()
    assert log.at[0, "result"] == "won" and log.at[0, "profit"] == pytest.approx(price - 1)
    assert betting.RECORD_PATH.exists()
    monkeypatch.setattr(overview, "DB_PATH", nba_project / "data" / "sports.db")
    overview._cache.clear()
    page = TestClient(app.app).get("/betting/").text
    assert "Live record" in page, page[page.find("<main"):page.find("<main") + 3000]
    assert "1–0" in page and "NBA:</strong> Not tested yet" in page
    assert TestClient(app.app).get("/today/").status_code == 200
