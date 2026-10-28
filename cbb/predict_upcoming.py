"""Predict upcoming men's college basketball games and save the predictions.

Usage:
    python -m cbb.predict_upcoming            # on your computer
    python -m cbb.predict_upcoming --record   # the automated run: also updates the official record

Predicts the next 8 days' games between Division I teams that haven't tipped off. A prediction can
be refreshed until tip-off and is never changed after it (records/cbb_live_predictions.csv).
"""

import sqlite3
import sys
from datetime import datetime, timedelta

from cbb import records
from cbb.model import NON_D1, current_fit, load_games, predict
from db import DB_PATH, SCHEMA_PATH
from nfl.predict_upcoming import EASTERN, has_started

MODEL_NAME = "cbb_points"
MODEL_VERSION = "v3"     # v3: preseason ratings until mid-December
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
                     & (games["date"] <= (today + timedelta(days=DAYS_AHEAD)).isoformat())
                     & (games["home"] != NON_D1) & (games["away"] != NON_D1)]
    to_predict = [g for g in upcoming.itertuples()
                  if not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)]
    if not recording:
        to_predict = [g for g in to_predict if g.id not in recorded]
    if restored:
        print(f"Loaded {restored} recorded basketball predictions from {records.RECORD_PATH}.")
    if not to_predict:
        print(f"No basketball games to predict in the next {DAYS_AHEAD} days.")
        if recording:
            records.export(conn, MODEL_NAME, MODEL_VERSION)
        conn.close()
        return
    model = current_fit(games, today.isoformat(), int(to_predict[0].season))
    for g in to_predict:
        p = predict(model, g.home, g.away, bool(g.neutral_site))
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
    print(f"Saved basketball predictions for {len(to_predict)} games.")


if __name__ == "__main__":
    main()
