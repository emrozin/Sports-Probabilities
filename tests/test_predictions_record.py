"""Live predictions: made before kickoff, never changed after it, and kept safe in the official record."""

import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from epl import predict_upcoming
from epl import records
from conftest import build_database, simulate_league
from epl.predict_upcoming import UK, has_started

FRIDAY_EVENING = datetime(2026, 10, 9, 20, 0, tzinfo=UK)
SATURDAY_1PM = datetime(2026, 10, 10, 13, 0, tzinfo=UK)  # after the 12:30 kickoff, before the 15:00 one


def add_fixtures(db_path, rows):
    conn = sqlite3.connect(db_path)
    ids = dict(conn.execute("SELECT name, id FROM teams").fetchall())
    for date, kickoff, home, away in rows:
        conn.execute("""INSERT INTO games (league_id, season, date, home_team_id, away_team_id, status, kickoff)
                        VALUES (1, 2026, ?, ?, ?, 'scheduled', ?)""", (date, ids[home], ids[away], kickoff))
    conn.commit()
    conn.close()


@pytest.fixture
def fixtures(league_db):
    add_fixtures(league_db, [("2026-10-10", "12:30", "Team A", "Team B"),
                             ("2026-10-10", "15:00", "Team C", "Team D")])
    return league_db


def stored(db_path):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT h.name, p.outcome, p.probability FROM predictions p
        JOIN games g ON g.id = p.game_id JOIN teams h ON h.id = g.home_team_id
        WHERE p.model_name = 'poisson_xg' ORDER BY h.name, p.outcome""").fetchall()
    conn.close()
    return {(team, outcome): prob for team, outcome, prob in rows}


def test_has_started_uses_uk_kickoff_times():
    assert not has_started("2026-10-10", "12:30", FRIDAY_EVENING)
    assert has_started("2026-10-10", "12:30", SATURDAY_1PM)
    assert not has_started("2026-10-10", "15:00", SATURDAY_1PM)
    # 12:30 in London is 7:30 in New York: a run from the US at 7:45 am is already too late.
    assert has_started("2026-10-10", "12:30", datetime(2026, 10, 10, 7, 45, tzinfo=ZoneInfo("America/New_York")))
    # Without a kickoff time, the whole matchday counts as started.
    assert has_started("2026-10-09", None, FRIDAY_EVENING)
    assert not has_started("2026-10-10", None, FRIDAY_EVENING)


def test_recorded_run_saves_predictions_to_the_record(fixtures):
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    record = pd.read_csv(records.RECORD_PATH)
    assert list(record["home"]) == ["Team A", "Team C"]
    assert record[["home_win", "draw", "away_win"]].sum(axis=1).tolist() == pytest.approx([1, 1])
    assert len(stored(fixtures)) == 6


def test_predictions_are_never_changed_after_kickoff(fixtures):
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    # Pretend the Friday predictions were different, so any later overwrite would show.
    conn = sqlite3.connect(fixtures)
    conn.execute("UPDATE predictions SET probability = 0.3 WHERE model_name = 'poisson_xg'")
    conn.commit()
    conn.close()
    records.export(sqlite3.connect(fixtures), "poisson_xg", "v2")

    predict_upcoming.main(["--record"], now=SATURDAY_1PM)
    after = stored(fixtures)
    assert after[("Team A", "home")] == 0.3      # kicked off at 12:30: untouched
    assert after[("Team C", "home")] != 0.3      # kicks off at 15:00: refreshed
    record = pd.read_csv(records.RECORD_PATH).set_index("home")
    assert record.loc["Team A", "home_win"] == 0.3


def test_local_runs_keep_the_official_record(fixtures):
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    official = records.RECORD_PATH.read_text()
    official_predictions = stored(fixtures)

    # Someone experiments locally and changes a stored prediction...
    conn = sqlite3.connect(fixtures)
    conn.execute("UPDATE predictions SET probability = 0.9 WHERE model_name = 'poisson_xg'")
    conn.commit()
    conn.close()

    # ...then runs the normal local update: the official predictions come back, the file is untouched.
    predict_upcoming.main([], now=FRIDAY_EVENING)
    assert stored(fixtures) == official_predictions  # exactly
    assert records.RECORD_PATH.read_text() == official


def test_the_record_survives_a_database_rebuild(project, fixtures):
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    before = stored(fixtures)

    # Rebuild the database from scratch with an extra team first, so every id shifts.
    Path(fixtures).unlink()
    games, _ = simulate_league()
    extra = games.iloc[[0]].assign(home="Aardvarks", id=0)
    build_database(fixtures, pd.concat([extra, games], ignore_index=True))
    add_fixtures(fixtures, [("2026-10-10", "12:30", "Team A", "Team B"),
                            ("2026-10-10", "15:00", "Team C", "Team D")])

    restored = records.restore(sqlite3.connect(fixtures))
    assert restored == 2
    assert stored(fixtures) == before  # exactly


def test_expected_goals_are_recorded_and_restored(project, fixtures):
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    record = pd.read_csv(records.RECORD_PATH)
    assert record[["exp_home_goals", "exp_away_goals"]].notna().all().all()

    conn = sqlite3.connect(fixtures)
    before = conn.execute("SELECT exp_home, exp_away FROM prediction_goals ORDER BY game_id").fetchall()
    conn.execute("DELETE FROM prediction_goals")
    conn.commit()
    records.restore(conn)
    after = conn.execute("SELECT exp_home, exp_away FROM prediction_goals ORDER BY game_id").fetchall()
    assert after == before  # exactly


def test_older_records_without_expected_goals_still_restore(project, fixtures):
    records.RECORD_PATH.parent.mkdir(exist_ok=True)
    records.RECORD_PATH.write_text(
        "season,date,home,away,model_name,model_version,home_win,draw,away_win,created_at\n"
        "2026,2026-10-10,Team A,Team B,poisson_xg,v2,0.5,0.3,0.2,2026-10-09 18:00:00\n")
    conn = sqlite3.connect(fixtures)
    assert records.restore(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM prediction_goals").fetchone()[0] == 0


def test_upcoming_matches_have_a_match_page(fixtures, monkeypatch):
    from fastapi.testclient import TestClient
    import app as site
    from datetime import date
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    monkeypatch.setattr(site, "DB_PATH", fixtures)
    monkeypatch.setattr(site, "date", type("FrozenDate", (), {"today": staticmethod(lambda: date(2026, 10, 9))}))
    site._cache.clear()
    client = TestClient(site.app)
    home = client.get("/premier-league/").text
    assert 'href="/match/2026-10-10/team-a-v-team-b/"' in home
    page = client.get("/match/2026-10-10/team-a-v-team-b/")
    assert page.status_code == 200
    assert "12:30 UK time" in page.text and "Likeliest" not in page.text and "Every possible score" in page.text


def test_track_record_shows_only_live_predictions_once_played(fixtures, monkeypatch):
    from fastapi.testclient import TestClient
    import app as site
    from epl import run_backtest
    run_backtest.main()
    predict_upcoming.main(["--record"], now=FRIDAY_EVENING)
    monkeypatch.setattr(site, "DB_PATH", fixtures)
    site._cache.clear()
    client = TestClient(site.app)
    assert "No finished live predictions yet" in client.get("/track-record/").text

    # One of the two predicted matches is played (result filled in, no closing odds yet).
    conn = sqlite3.connect(fixtures)
    conn.execute("""UPDATE games SET home_score = 2, away_score = 1, status = 'final'
                    WHERE date = '2026-10-10' AND kickoff = '12:30'""")
    conn.commit()
    site._cache.clear()
    html = client.get("/track-record/").text
    assert "1 match since" in html
    assert html.count('class="match card') == 1        # only the live prediction, not the backtest
    assert "Saved before kickoff" in html
    assert "fewer matches until then" in html          # no closing odds for it yet
