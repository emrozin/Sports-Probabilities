"""NFL predictions, the prediction record, the website's NFL pages, and the multi-sport update."""

import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import nfl.site as nfl_site
import update
from nfl import predict_upcoming, records
from nfl.elo import load_nfl_games
from nfl.import_nfl import import_games, read_games
from nfl.model import add_qb_changes, backtest
from nfl.run_backtest import save

TEAMS = ["KC", "BUF", "SEA", "DEN", "SF", "DAL", "GB", "MIA"]
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=predict_upcoming.EASTERN)
HEADER = ("game_id,season,game_type,week,gameday,gametime,away_team,away_score,home_team,home_score,location,"
          "overtime,home_rest,away_rest,away_moneyline,home_moneyline,spread_line,away_spread_odds,home_spread_odds,"
          "total_line,under_odds,over_odds,div_game,roof,home_qb_name,away_qb_name,stadium")


def simulated_csv(seed=0) -> str:
    """Six seasons of an 8-team league in nflverse's format; 2026 is played through week 5."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 5, len(TEAMS))))
    rows = []
    for season in range(2021, 2027):
        start = pd.Timestamp(f"{season}-09-10")
        for week in range(1, 18):
            order = rng.permutation(TEAMS)
            day = (start + pd.Timedelta(days=7 * (week - 1))).strftime("%Y-%m-%d")
            for k in range(4):
                away, home = order[2 * k], order[2 * k + 1]
                gid = f"{season}_{week:02d}_{away}_{home}"
                played = season < 2026 or week <= 5
                margin = 2 + strength[home] - strength[away] + rng.normal(0, 13)
                hs, as_ = (max(0, round(22 + margin / 2)), max(0, round(22 - margin / 2))) if played else ("", "")
                qbs = (f"{home} Starter", f"{away} Starter") if (played or week == 6) else ("", "")
                if week == 6 and season == 2026 and home == order[1]:
                    qbs = (f"{home} Backup", qbs[1])          # one listed backup this week
                ml = (-150, 130) if played or week == 6 else ("", "")
                rows.append(f"{gid},{season},REG,{week},{day},13:00,{away},{as_},{home},{hs},Home,0,7,7,"
                            f"{ml[1]},{ml[0]},2.5,-110,-110,44.5,-110,-110,0,outdoors,{qbs[0]},{qbs[1]},Stadium")
    return "\n".join([HEADER, *rows]) + "\n"


@pytest.fixture
def nfl_project(project, monkeypatch):
    """A temporary project with simulated NFL games, backtest predictions and this week's live predictions."""
    db = project / "data" / "sports.db"
    conn = sqlite3.connect(db)
    import_games(conn, read_games(simulated_csv()))
    games = add_qb_changes(load_nfl_games(conn))
    save(conn, backtest(games, 2023))
    conn.close()
    predict_upcoming.main(["--record"], now=NOW)
    monkeypatch.setattr(nfl_site, "DB_PATH", db)
    monkeypatch.setattr(nfl_site, "date", type("FrozenDate", (), {"today": staticmethod(lambda: date(2026, 10, 14))}))
    nfl_site._cache.clear()
    return project


@pytest.fixture
def client(nfl_project):
    import app
    return TestClient(app.app)


def test_live_predictions_cover_the_next_eight_days_and_are_recorded(nfl_project):
    record = records.read_record()
    assert set(record["week"]) == {6, 7}                        # Oct 15 and Oct 22
    assert np.allclose(record["home_win"] + record["away_win"], 1)
    assert (record["exp_home_points"] > 0).all()


def test_listed_backup_quarterback_lowers_a_teams_chances(nfl_project):
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    games = add_qb_changes(load_nfl_games(conn))
    conn.close()
    week6 = games[(games["season"] == 2026) & (games["week"] == 6)]
    assert week6["home_qb_change"].sum() == 1 and week6["away_qb_change"].sum() == 0


def test_recorded_predictions_are_restored_not_remade_locally(nfl_project):
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    before = conn.execute("SELECT COUNT(*) FROM predictions WHERE model_name = 'nfl_points_qb'").fetchone()[0]
    conn.execute("DELETE FROM predictions WHERE model_name = 'nfl_points_qb'")
    conn.commit()
    conn.close()
    recorded = records.read_record()
    predict_upcoming.main([], now=NOW)          # a local run: restores the record, changes nothing in it
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    after = conn.execute("SELECT COUNT(*) FROM predictions WHERE model_name = 'nfl_points_qb'").fetchone()[0]
    conn.close()
    assert after == before
    pd.testing.assert_frame_equal(records.read_record(), recorded)


def test_games_that_have_kicked_off_are_never_re_predicted(nfl_project):
    recorded = records.read_record().set_index("game")
    later = datetime(2026, 10, 15, 14, 0, tzinfo=predict_upcoming.EASTERN)   # after the 13:00 kickoffs
    predict_upcoming.main(["--record"], now=later)
    after = records.read_record().set_index("game")
    started = [g for g in recorded.index if recorded.loc[g, "date"] == "2026-10-15"]
    assert started
    pd.testing.assert_frame_equal(after.loc[started], recorded.loc[started])


@pytest.mark.parametrize("path", ["/nfl/", "/nfl/results/", "/nfl/results/2025/17/", "/nfl/standings/",
                                  "/nfl/standings/2022/", "/nfl/team/kansas-city-chiefs/",
                                  "/nfl/team/kansas-city-chiefs/2024/", "/nfl/track-record/", "/nfl/backtest/",
                                  "/nfl/ratings/"])
def test_nfl_pages_load(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "NFL probabilities" in response.text
    assert 'href="/static/style.css"' in response.text   # no page may override the site's base path


def test_upcoming_page_shows_this_weeks_predictions(client):
    page = client.get("/nfl/").text
    assert "Week 6" in page and "Saved before kickoff" in page and "Game details" in page


def test_game_page_shows_forecast_form_and_every_earlier_meeting(client, nfl_project):
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    game, away, home = conn.execute("""SELECT i.source_game_id, a.name, h.name FROM nfl_game_info i
                                       JOIN games g ON g.id = i.game_id JOIN teams h ON h.id = g.home_team_id
                                       JOIN teams a ON a.id = g.away_team_id
                                       WHERE g.season = 2026 AND i.week = 6 LIMIT 1""").fetchone()
    meetings = conn.execute("""SELECT COUNT(*) FROM games g JOIN teams h ON h.id = g.home_team_id
                               JOIN teams a ON a.id = g.away_team_id
                               WHERE g.status = 'final' AND ((h.name = ? AND a.name = ?) OR (h.name = ? AND a.name = ?))""",
                            (home, away, away, home)).fetchone()[0]
    conn.close()
    page = client.get(f"/nfl/game/{game.lower().replace('_', '-')}/").text
    assert "Forecast" in page and "Form going into the game" in page
    assert f"{meetings} meeting" in page
    assert "Expected starting quarterbacks" in page
    assert "Why the model picks" in page and "Quarterbacks" in page and "Home field" in page


def test_standings_are_split_into_divisions(client):
    page = client.get("/nfl/standings/").text
    assert "AFC West" in page and "Kansas City Chiefs" in page


def test_track_record_shows_live_predictions_once_games_are_played(client, nfl_project):
    assert "No finished live predictions yet" in client.get("/nfl/track-record/").text
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    conn.execute("""UPDATE games SET status = 'final', home_score = 27, away_score = 20
                    WHERE id IN (SELECT game_id FROM nfl_game_info WHERE week = 6) AND season = 2026""")
    conn.commit()
    conn.close()
    nfl_site._cache.clear()
    page = client.get("/nfl/track-record/").text
    assert "So far" in page and "Week 6" in page


def test_unknown_nfl_pages_are_404_with_nfl_navigation(client):
    response = client.get("/nfl/team/not-a-team/")
    assert response.status_code == 404
    assert "Go to upcoming games" in response.text


def test_sport_menu_links_both_sports(client):
    page = client.get("/nfl/").text
    assert 'href="/"' in page and 'href="/nfl/"' in page and 'class="sport-tabs"' in page


def test_update_keeps_going_when_one_sport_fails(monkeypatch, capsys):
    ran = []

    class Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command):
        module = command[2]
        ran.append(module)
        return Result(1 if module == "epl.import_epl" else 0)

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    monkeypatch.setattr(update.sys, "argv", ["update.py"])
    with pytest.raises(SystemExit) as exit_info:
        update.main()
    assert exit_info.value.code == 1
    assert ran == ["epl.import_epl", "nfl.import_nfl", "nfl.import_nfl_stats", "nfl.run_backtest",
                   "nfl.predict_upcoming", "nfl.predict_players", "nfl.season",
                   "nba.import_nba", "nba.live", "nba.run_backtest", "nba.predict_upcoming", "nba.predict_players", "nba.prop_lines", "nba.season",
                   "cfb.import_cfb", "cfb.run_backtest", "cfb.predict_upcoming", "cfb.conference",
                   "cbb.import_cbb", "cbb.run_backtest", "cbb.predict_upcoming",
                   "ucl.import_espn", "ucl.import_understat", "ucl.squad_values", "ucl.run_backtest", "ucl.predict_upcoming", "ucl.season", "espn", "betting", "warm_cache"]
    assert "Premier League: Results, stats and odds" in capsys.readouterr().out


def test_update_steps_can_carry_their_own_arguments(monkeypatch):
    commands = []

    class Result:
        returncode = 0

    monkeypatch.setattr(update.subprocess, "run", lambda command: commands.append(command[2:]) or Result())
    monkeypatch.setattr(update.sys, "argv", ["update.py", "--record"])
    update.main()
    assert ["epl.import_epl", "--league", "CHAMP"] in commands          # the Championship, for promoted teams
    assert ["epl.predict_upcoming", "--record"] in commands
