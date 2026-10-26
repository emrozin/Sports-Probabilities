"""Men's college basketball: the importer (ESPN's schedule files), the model, the prediction record
and the website pages, including standings filtered by conference."""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import cbb.site as cbb_site
from cbb import import_cbb, predict_upcoming, records
from cbb.model import NON_D1, backtest, elo_benchmark, fit, load_games, predict
from cbb.run_backtest import BACKTEST_NAME, BACKTEST_VERSION
from nfl.run_backtest import save

TEAMS = [("Duke", 2), ("North Carolina", 2), ("Virginia", 2), ("Louisville", 2),       # ACC
         ("Kentucky", 23), ("Auburn", 23), ("Tennessee", 23), ("Florida", 23)]        # SEC
NOW = datetime(2026, 1, 14, 12, 0, tzinfo=predict_upcoming.EASTERN)


def simulated_schedule(seed=0) -> pd.DataFrame:
    """Three seasons (2023-24 to 2025-26) in ESPN's format: eight Division I teams and one small
    school outside Division I (no conference). The 2025-26 season is played up to 14 January."""
    rng = np.random.default_rng(seed)
    strength = dict(zip([t for t, _ in TEAMS], rng.normal(0, 8, len(TEAMS))))
    strength["Small College"] = -30
    conf = dict(TEAMS)
    ids = {t: 100 + i for i, t in enumerate([t for t, _ in TEAMS] + ["Small College"])}
    rows, gid = [], 401000000
    for season in range(2023, 2026):
        for rnd in range(40):
            order = list(rng.permutation([t for t, _ in TEAMS]))
            pairs = [(order[2 * k], order[2 * k + 1]) for k in range(4)]
            if rnd == 0:
                pairs[0] = (order[0], "Small College")
            tip = pd.Timestamp(f"{season}-11-06T00:30Z") + pd.Timedelta(days=3 * rnd)   # 19:30 Eastern the day before
            for home, away in pairs:
                gid += 1
                played = tip < pd.Timestamp("2026-01-14T12:00Z")
                margin = 3.5 + strength[home] - strength[away] + rng.normal(0, 10)
                hs, as_ = round(72 + margin / 2), round(72 - margin / 2)
                if hs == as_:
                    hs += 1
                rows.append({"game_id": gid, "season": season + 1, "season_type": 2, "date": tip.strftime("%Y-%m-%dT%H:%MZ"),
                             "neutral_site": False, "conference_competition": conf.get(home) == conf.get(away),
                             "status_type_completed": played, "home_id": ids[home], "home_location": home,
                             "home_score": hs if played else 0, "home_conference_id": conf.get(home, np.nan),
                             "home_color": "001A57" if home == "Duke" else "333333", "home_alternate_color": "ffffff",
                             "away_id": ids[away], "away_location": away, "away_score": as_ if played else 0,
                             "away_conference_id": conf.get(away, np.nan), "away_color": "0033A0", "away_alternate_color": np.nan,
                             "notes_headline": np.nan})
    return pd.DataFrame(rows, columns=import_cbb.COLUMNS)


@pytest.fixture
def cbb_project(project, monkeypatch):
    db = project / "data" / "sports.db"
    conn = sqlite3.connect(db)
    import_cbb.import_games(conn, simulated_schedule())
    save(conn, backtest(load_games(conn), 2024), BACKTEST_NAME, BACKTEST_VERSION)
    conn.close()
    predict_upcoming.main(["--record"], now=NOW)
    monkeypatch.setattr(cbb_site, "DB_PATH", db)
    monkeypatch.setattr(cbb_site, "date", type("FrozenDate", (), {"today": staticmethod(lambda: NOW.date())}))
    cbb_site._cache.clear()
    return project


def test_import_uses_starting_year_seasons_eastern_dates_and_conference_names(cbb_project):
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM games ORDER BY season")]
    assert seasons == [2023, 2024, 2025]                                  # ESPN's 2024 is 2023-24
    first = conn.execute("SELECT date, kickoff FROM games ORDER BY date, id LIMIT 1").fetchone()
    assert first == ("2023-11-05", "19:30")
    names = {r[0] for r in conn.execute("SELECT DISTINCT home_conference FROM cbb_game_info")}
    assert {"ACC", "SEC"} <= names
    assert conn.execute("SELECT color FROM cbb_teams WHERE name = 'Duke'").fetchone()[0] == "#001A57"
    # unplayed games come in as 0-0 and must stay unscored
    assert conn.execute("SELECT COUNT(*) FROM games WHERE status = 'scheduled' AND home_score IS NOT NULL").fetchone()[0] == 0
    conn.close()


def test_reimporting_updates_games_instead_of_duplicating_them(project):
    schedule = simulated_schedule()
    conn = sqlite3.connect(project / "data" / "sports.db")
    first = import_cbb.import_games(conn, schedule)
    second = import_cbb.import_games(conn, schedule)
    assert second["new"] == 0 and second["updated"] == first["new"]
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == len(schedule)
    conn.close()


def test_current_season_starts_in_august():
    assert import_cbb.current_season(datetime(2026, 3, 1).date()) == 2025
    assert import_cbb.current_season(datetime(2026, 10, 16).date()) == 2026


def test_schools_outside_division_one_share_a_low_rating(cbb_project):
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    games = load_games(conn)
    conn.close()
    assert (games["home"] == NON_D1).sum() + (games["away"] == NON_D1).sum() == 3
    model = fit(games[games["status"] == "final"], "2026-01-14", 2025)
    overall = model["ratings"]["offense"] + model["ratings"]["defense"]
    assert overall[NON_D1] == overall.min()
    best = overall.drop(NON_D1).idxmax()
    assert predict(model, best, NON_D1)["home_win"] > 0.9


def test_weeks_are_named_by_their_monday(cbb_project):
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    games = load_games(conn)
    conn.close()
    assert (pd.to_datetime(games["week"]).dt.dayofweek == 0).all()
    assert "/" not in "".join(games["week"])


def test_elo_benchmark_gives_a_chance_for_every_finished_game(cbb_project):
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    games = load_games(conn)
    conn.close()
    elo = elo_benchmark(games)
    finished = games[games["status"] == "final"]
    assert elo.reindex(finished["id"]).notna().all() if elo.index.name == "id" else len(elo) >= len(finished)
    assert elo.between(0, 1).all()


def test_live_predictions_are_recorded_and_not_changed_after_tip_off(cbb_project):
    record = records.read_record()
    assert len(record) > 0 and np.allclose(record["home_win"] + record["away_win"], 1)
    assert record["date"].min() >= "2026-01-14" and record["date"].max() <= "2026-01-22"
    later = datetime(2026, 1, 20, 22, 0, tzinfo=predict_upcoming.EASTERN)
    predict_upcoming.main(["--record"], now=later)
    after = records.read_record()
    assert len(after) > len(record)
    started = record[record["date"] < "2026-01-20"]
    kept = after[after["game"].isin(started["game"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, started.reset_index(drop=True))


@pytest.mark.parametrize("path", ["/cbb/", "/cbb/?conference=SEC", "/cbb/results/", "/cbb/ratings/",
                                  "/cbb/ratings/?conference=ACC", "/cbb/standings/", "/cbb/standings/2024/",
                                  "/cbb/team/duke/", "/cbb/team/duke/2024/", "/cbb/track-record/", "/cbb/backtest/"])
def test_cbb_pages_load(cbb_project, path):
    import app
    response = TestClient(app.app).get(path)
    assert response.status_code == 200, path
    assert "College basketball" in response.text


def test_standings_filter_by_conference(cbb_project):
    import app
    client = TestClient(app.app)
    page = client.get("/cbb/standings/").text
    assert "ACC" in page and "SEC" in page and "Duke" in page and "Kentucky" in page
    sec = client.get("/cbb/standings/?conference=SEC").text
    assert "Kentucky" in sec and "Duke" not in sec.split("<h2>SEC</h2>")[-1]
    assert NON_D1 not in page
    assert client.get("/cbb/standings/1999/").status_code == 404
    assert client.get("/cbb/team/not-a-team/").status_code == 404


def test_cbb_pages_work_with_no_data(project, monkeypatch):
    import app
    monkeypatch.setattr(cbb_site, "DB_PATH", project / "data" / "sports.db")
    cbb_site._cache.clear()
    client = TestClient(app.app)
    for path in ("/cbb/", "/cbb/results/", "/cbb/ratings/", "/cbb/standings/", "/cbb/track-record/", "/cbb/backtest/"):
        assert client.get(path).status_code in (200, 404), path


def test_track_record_chart_appears_once_two_weeks_have_finished(cbb_project):
    import app
    client = TestClient(app.app)
    assert "chart-record" not in client.get("/cbb/track-record/").text     # nothing finished yet
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    conn.execute("""UPDATE games SET status = 'final', home_score = 70, away_score = 65
                    WHERE id IN (SELECT game_id FROM predictions WHERE model_name = 'cbb_points')""")
    conn.commit()
    conn.close()
    cbb_site._cache.clear()
    page = client.get("/cbb/track-record/").text
    assert "chart-record" in page and "A simple Elo rating" in page


def test_a_renamed_school_keeps_one_name_and_one_history(project):
    schedule = simulated_schedule()
    duke = (schedule["home_location"] == "Duke") & (schedule["season"] <= 2024)   # called something else before 2024-25
    schedule.loc[duke, "home_location"] = "Duke Blue Devils"
    schedule.loc[(schedule["away_location"] == "Duke") & (schedule["season"] <= 2024), "away_location"] = "Duke Blue Devils"
    conn = sqlite3.connect(project / "data" / "sports.db")
    import_cbb.import_games(conn, schedule)
    games = load_games(conn)
    assert "Duke Blue Devils" not in set(games["home_team"]) | set(games["away_team"])
    assert games[(games["home_team"] == "Duke") | (games["away_team"] == "Duke")]["season"].nunique() == 3
    conn.close()


def test_reimporting_after_a_rename_moves_old_games_to_the_new_name(project):
    schedule = simulated_schedule()
    old = schedule.copy()
    for side in ("home", "away"):
        old.loc[old[f"{side}_location"] == "Duke", f"{side}_location"] = "Duke Blue Devils"
    conn = sqlite3.connect(project / "data" / "sports.db")
    import_cbb.import_games(conn, old)         # an earlier import, before the rename
    import_cbb.import_games(conn, schedule)    # ESPN now calls every Duke game "Duke"
    names = set(load_games(conn)[["home_team", "away_team"]].stack())
    assert "Duke" in names and "Duke Blue Devils" not in names
    conn.close()


def test_espn_lines_show_on_the_cards(cbb_project):
    import app
    import espn
    conn = sqlite3.connect(cbb_project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    game, home, away = conn.execute("""SELECT i.source_game_id, h.name, a.name FROM cbb_game_info i JOIN games g ON g.id = i.game_id
                                       JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                                       JOIN predictions p ON p.game_id = g.id AND p.model_name = 'cbb_points'
                                       WHERE g.status = 'scheduled' LIMIT 1""").fetchone()
    espn.save(conn, "CBB", [{"game": game, "provider": "DraftKings", "home_ml_open": None, "away_ml_open": None,
                             "home_ml": 1.5, "away_ml": 2.7, "spread": -4.5, "total": 141.5, "final": 0}])
    conn.close()
    cbb_site._cache.clear()
    page = TestClient(app.app).get("/cbb/").text
    assert "Market" in page and f"{home} by 4.5" in page and "141.5" in page
