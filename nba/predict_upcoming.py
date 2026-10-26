"""Predict upcoming NBA games and save the predictions.

Usage:
    python -m nba.predict_upcoming            # on your computer
    python -m nba.predict_upcoming --record   # the automated run: also updates the official record

Predicts the next 8 days' games that haven't tipped off, with each team's expected lineup: its
current roster minus players listed Out or Doubtful (run python -m nba.live first for the latest
injury report). A prediction can be refreshed until tip-off and is never changed after it
(records/nba_live_predictions.csv).
"""

import sqlite3
import sys
from datetime import datetime, timedelta

from db import DB_PATH, SCHEMA_PATH
from nba import live, records
from nba.model import fit, load_games, load_players, predict
from nfl.predict_upcoming import EASTERN, has_started

MODEL_NAME = "nba_players"
MODEL_VERSION = "v2"     # v2: back-to-backs
DAYS_AHEAD = 8


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    recording = "--record" in argv
    now = now or datetime.now(EASTERN)
    today = now.astimezone(EASTERN).date()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    restored = records.restore(conn)
    recorded = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM predictions WHERE model_name = ?", (MODEL_NAME,))}
    games = load_games(conn)
    upcoming = games[(games["status"] == "scheduled") & (games["date"] >= today.isoformat())
                     & (games["date"] <= (today + timedelta(days=DAYS_AHEAD)).isoformat())]
    to_predict = [g for g in upcoming.itertuples()
                  if not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)]
    if not recording:
        to_predict = [g for g in to_predict if g.id not in recorded]
    if restored:
        print(f"Loaded {restored} recorded NBA predictions from {records.RECORD_PATH}.")
    if not to_predict:
        print(f"No NBA games to predict in the next {DAYS_AHEAD} days.")
        if recording:
            records.export(conn, MODEL_NAME, MODEL_VERSION)
        conn.close()
        return
    players = load_players(conn)
    model = fit(games, players, today.isoformat(), int(to_predict[0].season))
    latest = live.latest_lines(players)
    rosters, injuries = live.load(conn)
    out = live.unavailable(injuries)
    for g in to_predict:
        home = live.expected_lineup(g.home, latest, rosters, out, g.date)
        away = live.expected_lineup(g.away, latest, rosters, out, g.date)
        p = predict(model, g.home, g.away, home, away, bool(g.neutral_site),
                    home_b2b=int(g.home_b2b), away_b2b=int(g.away_b2b))
        for outcome, prob in (("home", p["home_win"]), ("away", 1 - p["home_win"])):
            conn.execute("""INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (game_id, model_name, model_version, outcome)
                            DO UPDATE SET probability = excluded.probability, created_at = datetime('now')""",
                         (g.id, MODEL_NAME, MODEL_VERSION, outcome, float(prob)))
        conn.execute("""INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (game_id, model_name, model_version)
                        DO UPDATE SET exp_home = excluded.exp_home, exp_away = excluded.exp_away, created_at = datetime('now')""",
                     (g.id, MODEL_NAME, MODEL_VERSION, float(p["home_points"]), float(p["away_points"])))
    conn.commit()
    if recording:
        records.export(conn, MODEL_NAME, MODEL_VERSION)
    conn.close()
    print(f"Saved NBA predictions for {len(to_predict)} games.")


if __name__ == "__main__":
    main()
