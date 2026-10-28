"""The error analysis and the player availability collector."""

import pandas as pd
import pytest
import requests

from epl import error_analysis
from epl import fetch_availability
from epl import run_backtest


def test_error_analysis_runs_and_shares_add_up(league_db, capsys):
    run_backtest.main()
    import sqlite3
    df = error_analysis.load_backtest_frame(sqlite3.connect(league_db))
    assert len(df) > 0 and df["round"].min() == 1
    stages = pd.cut(df["round"], [0, 5, 10, 19, 29, 38])
    table = error_analysis.breakdown(df, stages, df["gap"].sum())
    assert table["share"].sum() == pytest.approx(1.0)
    error_analysis.main()
    out = capsys.readouterr().out
    assert "By stage of the season" in out and "Calibration" in out and "Teams the model forecasts worst" in out


FEED = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Spurs"}],
    "elements": [
        {"id": 10, "first_name": "Bukayo", "second_name": "Saka", "web_name": "Saka", "team": 1, "element_type": 3,
         "status": "d", "chance_of_playing_next_round": 75, "news": "Hamstring - 75% chance of playing",
         "minutes": 540, "starts": 6, "expected_goals": "2.10", "now_cost": 100},
        {"id": 11, "first_name": "Guglielmo", "second_name": "Vicario", "web_name": "Vicario", "team": 2,
         "element_type": 1, "status": "a", "chance_of_playing_next_round": None, "news": "",
         "minutes": 630, "starts": 7, "expected_goals": "0.00", "now_cost": 50},
    ],
}


def test_availability_snapshot_keeps_the_useful_fields():
    players = fetch_availability.snapshot(FEED)
    saka = players.set_index("name").loc["Bukayo Saka"]
    assert saka["team"] == "Arsenal" and saka["position"] == "MID" and saka["status"] == "doubtful"
    assert saka["chance_next_round"] == 75 and saka["price"] == 10.0
    assert players.set_index("name").loc["Guglielmo Vicario", "position"] == "GK"


def test_availability_saves_a_dated_file(project, monkeypatch, capsys):
    class Response:
        def raise_for_status(self): pass
        def json(self): return FEED
    monkeypatch.setattr(fetch_availability.requests, "get", lambda *a, **k: Response())
    fetch_availability.main()
    files = list((project / "records" / "availability").glob("*.csv"))
    assert len(files) == 1 and len(pd.read_csv(files[0])) == 2
    assert "1 not fully available" in capsys.readouterr().out


def test_availability_failure_never_stops_the_update(project, monkeypatch, capsys):
    monkeypatch.setattr(fetch_availability.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")))
    fetch_availability.main()  # must not raise
    assert "Skipping this snapshot" in capsys.readouterr().out
    assert not (project / "records" / "availability").exists()


def test_team_news_lists_who_is_out_regulars_first(tmp_path):
    import pandas as pd
    from epl.team_news import latest, team_news
    folder = tmp_path / "availability"
    folder.mkdir()
    rows = [(1, "Bukayo Saka", "Arsenal", "MID", "injured", 0, 0, "Hamstring injury", 9, 800),
            (2, "Backup Keeper", "Arsenal", "GK", "doubtful", 50, 50, "Knock", 0, 0),
            (3, "Declan Rice", "Arsenal", "MID", "available", 100, 100, "", 10, 900),
            (4, "James Maddison", "Spurs", "MID", "suspended", 0, 0, "Suspended until 2 Nov", 8, 700)]
    pd.DataFrame(rows, columns=["player_id", "name", "team", "position", "status", "chance_this_round",
                                "chance_next_round", "news", "starts", "minutes"]).to_csv(folder / "2026-10-20.csv", index=False)
    pd.DataFrame(columns=["team"]).to_csv(folder / "2026-10-01.csv", index=False)
    snap, day = latest(folder)
    assert day == "2026-10-20"                              # the newest snapshot
    arsenal = team_news("Arsenal", snap)
    assert [p["name"] for p in arsenal] == ["Bukayo Saka", "Backup Keeper"]     # available players left out
    assert arsenal[0]["out"] and arsenal[0]["regular"] and arsenal[1]["chance"] == 50 and not arsenal[1]["regular"]
    assert team_news("Tottenham", snap)[0]["name"] == "James Maddison"         # FPL's "Spurs" is our "Tottenham"
