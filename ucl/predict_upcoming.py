"""Predict upcoming European cup games (Champions League, Europa League, Conference League) and
save the predictions.

Usage:
    python -m ucl.predict_upcoming            # on your computer
    python -m ucl.predict_upcoming --record   # the automated run: also updates the official record

Fits the model on every finished game (domestic and European) and predicts the next 8 days' cup
games that haven't kicked off. A prediction can be refreshed until kickoff and is never changed
after it (records/europe_live_predictions.csv, written only by the automated run).
"""

import sqlite3
import sys
from datetime import datetime, timedelta

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from nfl.predict_upcoming import EASTERN, has_started
from ucl import elo, records
from ucl.cups import CUPS
from ucl.model import LEAGUE_XG_WEIGHT, fit, load_games, load_values, predict, window

MODEL_NAME = "europe_goals"
MODEL_VERSION = "v1"
DAYS_AHEAD = 8


def upcoming_games(conn) -> pd.DataFrame:
    """Cup games still to play, with ESPN's team ids."""
    return pd.read_sql("""SELECT g.id, l.code AS competition, g.season, g.date, g.kickoff, i.espn_id, i.stage, i.neutral,
                                 hs.espn_id AS home_espn, aw.espn_id AS away_espn, h.name AS home, a.name AS away
                          FROM games g JOIN leagues l ON l.id = g.league_id AND l.code IN ('UCL', 'UEL', 'UECL', 'ESP', 'GER', 'ITA', 'FRA')
                          JOIN cup_game_info i ON i.game_id = g.id
                          JOIN cup_teams hs ON hs.team_id = g.home_team_id AND hs.league_id = g.league_id
                          JOIN cup_teams aw ON aw.team_id = g.away_team_id AND aw.league_id = g.league_id
                          JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                          WHERE g.status = 'scheduled' AND i.cancelled = 0
                          ORDER BY g.date, g.kickoff""", conn)


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    recording = "--record" in argv
    now = now or datetime.now(EASTERN)
    today = now.astimezone(EASTERN).date()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    restored = records.restore(conn)
    recorded = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM predictions WHERE model_name = ? AND model_version = ?",
                                           (MODEL_NAME, MODEL_VERSION))}
    games = upcoming_games(conn)
    games = games[(games["date"] >= today.isoformat()) & (games["date"] <= (today + timedelta(days=DAYS_AHEAD)).isoformat())]
    to_predict = [g for g in games.itertuples() if not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)]
    if not recording:
        to_predict = [g for g in to_predict if g.id not in recorded]
    if restored:
        print(f"Loaded {restored} recorded European cup predictions from {records.RECORD_PATH}.")
    if not to_predict:
        print(f"No European cup games to predict in the next {DAYS_AHEAD} days.")
        if recording:
            records.export(conn, MODEL_NAME, MODEL_VERSION)
        conn.close()
        return
    soccer = load_games(conn)
    values, models = load_values(conn), {}
    for g in to_predict:
        kind = CUPS[g.competition]["kind"]
        if kind not in models:                   # the cups' model and the leagues' (which learns from xG too)
            models[kind] = fit(window(soccer, today.isoformat()), today.isoformat(), values=values,
                               xg_weight=LEAGUE_XG_WEIGHT if kind == "league" else None)
            elo.attach(models[kind], soccer)     # league games blend in Elo (ucl/elo.py)
        p = predict(models[kind], g.home_espn, g.away_espn, bool(g.neutral), kind == "cup")
        for outcome in ("home", "draw", "away"):
            conn.execute("""INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (game_id, model_name, model_version, outcome)
                            DO UPDATE SET probability = excluded.probability, created_at = datetime('now')""",
                         (g.id, MODEL_NAME, MODEL_VERSION, outcome, float(p[outcome])))
        conn.execute("""INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (game_id, model_name, model_version)
                        DO UPDATE SET exp_home = excluded.exp_home, exp_away = excluded.exp_away, created_at = datetime('now')""",
                     (g.id, MODEL_NAME, MODEL_VERSION, float(p["exp_home"]), float(p["exp_away"])))
    conn.commit()
    if recording:
        records.export(conn, MODEL_NAME, MODEL_VERSION)
    conn.close()
    print(f"Saved European cup predictions for {len(to_predict)} games.")


if __name__ == "__main__":
    main()
