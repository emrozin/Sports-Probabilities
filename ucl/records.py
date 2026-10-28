"""The official record of live European cup predictions: records/europe_live_predictions.csv.

Like every sport's: each prediction is committed before kickoff and never changed after it, so the
Git history shows when it was made. Games are identified by ESPN's game id.
"""

import sqlite3
from pathlib import Path

import pandas as pd

RECORD_PATH = Path("records/europe_live_predictions.csv")
COLUMNS = ["game", "competition", "season", "stage", "date", "home", "away", "model_name", "model_version",
           "home_win", "draw", "away_win", "exp_home_goals", "exp_away_goals", "created_at"]
KEY = ["game", "model_name", "model_version"]


def read_record() -> pd.DataFrame:
    if not RECORD_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RECORD_PATH, dtype={"game": str, "date": str, "created_at": str}, float_precision="round_trip")


def restore(conn: sqlite3.Connection) -> int:
    restored = 0
    for r in read_record().itertuples():
        found = conn.execute("SELECT game_id FROM cup_game_info WHERE espn_id = ?", (str(r.game),)).fetchone()
        if found is None:
            continue
        for outcome, probability in (("home", r.home_win), ("draw", r.draw), ("away", r.away_win)):
            conn.execute("""INSERT OR REPLACE INTO predictions (game_id, model_name, model_version, outcome, probability, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                         (found[0], r.model_name, r.model_version, outcome, float(probability), r.created_at))
        conn.execute("""INSERT OR REPLACE INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     (found[0], r.model_name, r.model_version, float(r.exp_home_goals), float(r.exp_away_goals), r.created_at))
        restored += 1
    conn.commit()
    return restored


def export(conn: sqlite3.Connection, model_name: str, model_version: str) -> int:
    current = pd.read_sql("""
        SELECT i.espn_id AS game, l.code AS competition, g.season, i.stage, g.date, h.name AS home, a.name AS away,
               p.model_name, p.model_version,
               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS home_win,
               MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS draw,
               MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS away_win,
               MAX(pg.exp_home) AS exp_home_goals, MAX(pg.exp_away) AS exp_away_goals,
               MAX(p.created_at) AS created_at
        FROM predictions p
        JOIN prediction_goals pg ON pg.game_id = p.game_id
                                AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        JOIN games g ON g.id = p.game_id
        JOIN leagues l ON l.id = g.league_id
        JOIN cup_game_info i ON i.game_id = g.id
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        WHERE p.model_name = ? AND p.model_version = ?
        GROUP BY p.game_id
    """, conn, params=(model_name, model_version))
    current["game"] = current["game"].astype(str)
    combined = (pd.concat([read_record().reindex(columns=COLUMNS), current], ignore_index=True)
                  .drop_duplicates(subset=KEY, keep="last").sort_values(["date", "game"]))
    RECORD_PATH.parent.mkdir(exist_ok=True)
    combined[COLUMNS].to_csv(RECORD_PATH, index=False)
    return len(combined)
