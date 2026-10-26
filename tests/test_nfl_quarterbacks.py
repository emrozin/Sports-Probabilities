"""Starting quarterbacks: nflverse's listing checked against the box scores and the injury report."""

import sqlite3

import pandas as pd
import pytest

from nfl.quarterbacks import correct_quarterbacks, person, read_overrides

SCHEMA = open("schema.sql").read()


def game(gid, week, status, home, away, home_qb, away_qb, season=2026):
    return {"id": gid, "season": season, "week": week, "date": f"2026-09-{week + 9:02d}", "status": status,
            "home": home, "away": away, "home_qb": home_qb, "away_qb": away_qb}


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    yield c
    c.close()


def passes(conn, rows):
    conn.executemany("INSERT INTO nfl_player_stats (game_id, player_id, name, position, team, pass_attempts) "
                     "VALUES (?, ?, ?, 'QB', ?, ?)", [(g, name.lower(), name, team, att) for g, team, name, att in rows])


def test_a_stale_listing_is_corrected_from_the_box_scores_and_for_the_next_game(conn):
    # Week 1: Starter hurt early, Backup finishes. Week 2: Backup starts. Weeks 3-4: Starter back, but
    # the schedule still lists Backup, including for upcoming week 5.
    games = pd.DataFrame([game(1, 1, "final", "SEA", "NE", "Starter", "Other"),
                          game(2, 2, "final", "SEA", "NE", "Backup", "Other"),
                          game(3, 3, "final", "SEA", "NE", "Backup", "Other"),
                          game(4, 4, "final", "SEA", "NE", "Backup", "Other"),
                          game(5, 5, "scheduled", "SEA", "NE", "Backup", "Other")])
    passes(conn, [(1, "SEA", "Starter", 5), (1, "SEA", "Backup", 30), (2, "SEA", "Backup", 32),
                  (3, "SEA", "Starter", 35), (4, "SEA", "Starter", 31)]
           + [(g, "NE", "Other", 30) for g in range(1, 5)])
    fixed = correct_quarterbacks(conn, games, overrides={}).set_index("id")
    assert list(fixed["home_qb"]) == ["Starter", "Backup", "Starter", "Starter", "Starter"]
    assert fixed.at[5, "home_qb_listed"] == "Backup"
    assert (fixed["away_qb"] == "Other").all()


def test_a_ruled_out_starter_or_a_returning_usual_starter_keeps_the_listing(conn):
    games = pd.DataFrame([game(1, 1, "final", "BAL", "NE", "Star", "Other"),
                          game(2, 2, "final", "BAL", "NE", "Star", "Other"),
                          game(3, 3, "final", "BAL", "NE", "Fill In", "Other"),
                          game(4, 4, "scheduled", "BAL", "NE", "Star", "Other"),       # the star returns
                          game(5, 5, "scheduled", "NE", "BAL", "Other", "Third")])    # Fill In ruled out
    passes(conn, [(1, "BAL", "Star", 30), (2, "BAL", "Star", 30), (3, "BAL", "Fill In", 30)])
    conn.execute("INSERT INTO nfl_injuries (season, week, team, player_id, name, status) "
                 "VALUES (2026, 5, 'BAL', 'f', 'Fill In', 'Out')")
    fixed = correct_quarterbacks(conn, games, overrides={}).set_index("id")
    assert fixed.at[4, "home_qb"] == "Star"
    assert fixed.at[5, "away_qb"] == "Third"


def test_the_same_quarterback_spelled_differently_is_not_a_correction(conn):
    games = pd.DataFrame([game(1, 1, "final", "PHI", "NE", "Michael Vick", "Gardner Minshew")])
    passes(conn, [(1, "PHI", "Mike Vick", 30), (1, "NE", "Gardner Minshew II", 30)])
    fixed = correct_quarterbacks(conn, games, overrides={})
    assert fixed.at[0, "home_qb"] == "Michael Vick" and fixed.at[0, "away_qb"] == "Gardner Minshew"
    assert person("Mitch Trubisky") == person("Mitchell Trubisky") == person("Mitchell Trubisky Jr.")
    assert person("Derek Carr") != person("Jason Carr")


def test_overrides_file_names_the_starter(conn, tmp_path):
    path = tmp_path / "qb_overrides.csv"
    path.write_text("# a comment\ngame,team,qb\n2026-05-ne-sea,sea,Third\n")
    overrides = read_overrides(path)
    assert overrides == {("2026_05_NE_SEA", "SEA"): "Third"}
    conn.execute("INSERT INTO nfl_game_info (game_id, source_game_id, game_type, week, neutral_site) "
                 "VALUES (5, '2026_05_NE_SEA', 'REG', 5, 0)")
    games = pd.DataFrame([game(5, 5, "scheduled", "SEA", "NE", "Backup", "Other")])
    assert correct_quarterbacks(conn, games, overrides=overrides).at[0, "home_qb"] == "Third"


def roster(conn, rows):
    conn.executemany("INSERT INTO nfl_rosters (season, week, team, player_id, name, position, status) "
                     "VALUES (2026, ?, ?, ?, ?, 'QB', ?)", [(w, team, name.lower(), name, status) for w, team, name, status in rows])


def test_a_listed_starter_on_injured_reserve_is_replaced_by_the_backup_who_started_before(conn):
    # Like the Eagles in October 2026: the schedule still lists Hurts, who went on injured reserve;
    # McKee started for the team before (last season), Payton never has; Dalton was released.
    games = pd.DataFrame([game(1, 1, "final", "PHI", "NE", "Tanner McKee", "Other", season=2025),
                          game(2, 1, "final", "PHI", "NE", "Jalen Hurts", "Other"),
                          game(3, 2, "scheduled", "PHI", "NE", "Jalen Hurts", "Other")])
    passes(conn, [(1, "PHI", "Tanner McKee", 40), (2, "PHI", "Jalen Hurts", 30), (2, "PHI", "Andy Dalton", 1)])
    roster(conn, [(2, "PHI", "Jalen Hurts", "RES"), (2, "PHI", "Tanner McKee", "ACT"), (2, "PHI", "Cole Payton", "ACT"),
                  (2, "PHI", "Andy Dalton", "CUT")])
    fixed = correct_quarterbacks(conn, games, overrides={}).set_index("id")
    assert fixed.at[3, "home_qb"] == "Tanner McKee" and fixed.at[3, "home_qb_listed"] == "Jalen Hurts"
    assert fixed.at[2, "home_qb"] == "Jalen Hurts"           # finished games go by the box score


def test_the_latest_roster_counts_until_the_next_one_is_out(conn):
    games = pd.DataFrame([game(1, 1, "final", "PHI", "NE", "Jalen Hurts", "Other"),
                          game(2, 3, "scheduled", "PHI", "NE", "Jalen Hurts", "Other")])
    passes(conn, [(1, "PHI", "Jalen Hurts", 30)])
    roster(conn, [(2, "PHI", "Jalen Hurts", "RES"), (2, "PHI", "Tanner McKee", "ACT")])   # no week 3 roster yet
    assert correct_quarterbacks(conn, games, overrides={}).at[1, "home_qb"] == "Tanner McKee"


def test_overrides_only_apply_to_games_not_yet_played(conn):
    conn.execute("INSERT INTO nfl_game_info (game_id, source_game_id, game_type, week, neutral_site) "
                 "VALUES (1, '2026_01_NE_PHI', 'REG', 1, 0)")
    games = pd.DataFrame([game(1, 1, "final", "PHI", "NE", "Jalen Hurts", "Other")])
    passes(conn, [(1, "PHI", "Tanner McKee", 45)])
    fixed = correct_quarterbacks(conn, games, overrides={("2026_01_NE_PHI", "PHI"): "Andy Dalton"})
    assert fixed.at[0, "home_qb"] == "Tanner McKee"          # the wrong guess doesn't reach the ratings


def test_players_on_injured_reserve_count_as_out_for_projections(conn):
    from nfl.players import load_injuries
    conn.execute("INSERT INTO nfl_rosters VALUES (2026, 6, 'PHI', 'hurts', 'Jalen Hurts', 'QB', 'RES')")
    conn.execute("INSERT INTO nfl_rosters VALUES (2026, 6, 'PHI', 'brown', 'A.J. Brown', 'WR', 'ACT')")
    conn.execute("INSERT INTO nfl_injuries VALUES (2026, 6, 'PHI', 'brown', 'A.J. Brown', 'WR', 'Questionable')")
    reports = load_injuries(conn)
    assert reports[(2026, 6, "PHI")] == {"brown": "Questionable", "hurts": "Out"}
    assert reports[(2026, 7, "PHI")] == {"hurts": "Out"}     # carried to the next week
