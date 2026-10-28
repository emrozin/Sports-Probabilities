"""Missing starters (nfl/starters.py): regulars from the previous game ruled out this week, and
their effect on the prediction."""

import sqlite3
from pathlib import Path

import pytest

from nfl.model import INJURY_DEFENSE, INJURY_OFFENSE, injury_effect
from nfl.starters import add_to_games, missing_starters

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def conn(tmp_path):
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.executescript((ROOT / "schema.sql").read_text())
    conn.execute("INSERT INTO leagues (id, code, name, sport) VALUES (1, 'NFL', 'NFL', 'american_football')")
    conn.executemany("INSERT INTO teams (id, league_id, name) VALUES (?, 1, ?)", [(1, "PHI"), (2, "DAL"), (3, "NYG")])
    games = [(1, "2024-09-08", 1, 2, 1), (2, "2024-09-15", 1, 3, 2), (3, "2024-09-22", 3, 1, 3)]
    conn.executemany("""INSERT INTO games (id, league_id, season, date, home_team_id, away_team_id, status)
                        VALUES (?, 1, 2024, ?, ?, ?, 'final')""", [(g, d, h, a) for g, d, h, a, _ in games])
    conn.executemany("""INSERT INTO nfl_game_info (game_id, source_game_id, game_type, week, neutral_site)
                        VALUES (?, ?, 'REG', ?, 0)""", [(g, f"2024_0{w}_X", w) for g, _, _, _, w in games])
    snaps = [(1, "PHI", "Saquon Barkley", "RB", 0.8, 0.0), (1, "PHI", "Jalen Hurts", "QB", 1.0, 0.0),
             (1, "PHI", "Lane Johnson", "T", 1.0, 0.0), (1, "PHI", "Zack Baun", "LB", 0.0, 0.95),
             (1, "PHI", "Backup Guy", "WR", 0.2, 0.0), (2, "PHI", "Saquon Barkley", "RB", 0.75, 0.0)]
    conn.executemany("""INSERT INTO nfl_snap_counts (game_id, team, name, position, offense_pct, defense_pct)
                        VALUES (?, ?, ?, ?, ?, ?)""", snaps)
    conn.executemany("INSERT INTO nfl_injuries (season, week, team, player_id, name, position, status) VALUES (2024, ?, 'PHI', ?, ?, ?, ?)",
                     [(2, "a", "Lane Johnson", "T", "Out"), (2, "b", "Jalen Hurts", "QB", "Out"),
                      (2, "c", "Backup Guy", "WR", "Out"), (2, "d", "Saquon Barkley", "RB", "Questionable")])
    conn.execute("INSERT INTO nfl_rosters (season, week, team, player_id, name, position, status) VALUES (2024, 2, 'PHI', 'e', 'Zack Baun', 'LB', 'RES')")
    conn.commit()
    return conn


def test_only_regulars_ruled_out_count_and_not_the_quarterback(conn):
    m = missing_starters(conn).set_index(["game_id", "team"])
    row = m.loc[(2, "PHI")]
    assert row["offense"] == pytest.approx(1.0)          # Lane Johnson; not Hurts (QB), the backup, or a questionable player
    assert row["defense"] == pytest.approx(0.95)         # Zack Baun, on injured reserve
    assert "Lane Johnson" in row["names"] and "Hurts" not in row["names"]
    assert (1, "PHI") not in m.index                     # no previous game to know the regulars from


def test_missing_starters_lower_the_teams_expected_margin(conn):
    import pandas as pd
    games = add_to_games(pd.DataFrame({"id": [2], "home": ["PHI"], "away": ["NYG"]}), missing_starters(conn))
    g = next(games.itertuples())
    assert g.away_missing_off == 0 and g.home_missing_off == pytest.approx(1.0)
    effect = injury_effect((g.home_missing_off, g.home_missing_def), (g.away_missing_off, g.away_missing_def))
    assert effect == pytest.approx(INJURY_OFFENSE * 1.0 + INJURY_DEFENSE * 0.95) and effect < 0
