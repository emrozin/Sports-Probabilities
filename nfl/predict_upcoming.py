"""Predict upcoming NFL games and save the predictions.

Usage:
    python -m nfl.predict_upcoming            # on your computer
    python -m nfl.predict_upcoming --record   # the automated run: also updates the official record

Fits the points model on every finished game and predicts the games of the next 8 days that
haven't kicked off. A prediction can be refreshed until kickoff and is never changed after it,
so stored predictions form a genuine pre-game track record (records/nfl_live_predictions.csv,
written only by the automated run, like the Premier League's).

Quarterbacks: nflverse lists each team's expected starter for the current week's games, and the
model uses that quarterback's rating. Further ahead, no starter is listed yet, so the team's usual
starter is assumed; the daily runs refresh the predictions as starters are named.
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from nfl import records
from nfl.elo import load_nfl_games
from nfl.import_nfl import ensure_columns
from nfl.model import TRAINING_YEARS, add_qb_changes, fit, missing, predict
from nfl.starters import add_to_games, missing_starters

MODEL_NAME = "nfl_points_qb"
MODEL_VERSION = "v3"   # v3: missing starters; v2: quarterback ratings, shrinkage 5, margin SD 12 (v1, until 14 Oct 2026: a usual-starter flag)
EASTERN = ZoneInfo("America/New_York")  # nflverse dates and kickoff times are US Eastern
DAYS_AHEAD = 8


def has_started(game_date: str, kickoff: str | None, now: datetime) -> bool:
    """True once a game has kicked off. Without a kickoff time, the whole day counts as started."""
    if kickoff:
        return now >= datetime.fromisoformat(f"{game_date} {kickoff}").replace(tzinfo=EASTERN)
    return game_date <= now.astimezone(EASTERN).date().isoformat()


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    recording = "--record" in argv
    now = now or datetime.now(EASTERN)
    today = now.astimezone(EASTERN).date()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    ensure_columns(conn)
    restored = records.restore(conn)
    recorded = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM predictions "
                                           "WHERE model_name = ? AND model_version = ?", (MODEL_NAME, MODEL_VERSION))}

    games = add_qb_changes(load_nfl_games(conn))
    games = add_to_games(games, missing_starters(conn, since=int(games["season"].max()) if len(games) else 0))
    kickoffs = dict(conn.execute("SELECT id, kickoff FROM games WHERE id IN (SELECT game_id FROM nfl_game_info)"))
    window = games[(games["status"] == "scheduled") & (games["date"] >= today.isoformat())
                   & (games["date"] <= (today + timedelta(days=DAYS_AHEAD)).isoformat())]
    to_predict = [g for g in window.itertuples() if not has_started(g.date, kickoffs.get(g.id), now)]
    if not recording:
        to_predict = [g for g in to_predict if g.id not in recorded]   # locally, the official record stands

    if restored:
        print(f"Loaded {restored} recorded NFL predictions from {records.RECORD_PATH}.")
    if not to_predict:
        print(f"No NFL games to predict in the next {DAYS_AHEAD} days.")
        if recording:
            records.export(conn, MODEL_NAME, MODEL_VERSION)
        conn.close()
        return

    played = games[games["status"] == "final"]
    start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    model = fit(played[played["date"] >= start], today.isoformat())

    lines = []
    for g in to_predict:
        # Listed starters where nflverse has them; otherwise each team's usual starter.
        p = predict(model, g.home, g.away, bool(g.neutral_site), home_qb=g.home_qb_expected,
                    away_qb=g.away_qb_expected, home_qb_starts=g.home_qb_starts, away_qb_starts=g.away_qb_starts,
                    home_missing=missing(g, "home"), away_missing=missing(g, "away"))
        for outcome, prob in (("home", p["home_win"]), ("away", 1 - p["home_win"])):
            conn.execute("""INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (game_id, model_name, model_version, outcome)
                            DO UPDATE SET probability = excluded.probability, created_at = datetime('now')""",
                         (g.id, MODEL_NAME, MODEL_VERSION, outcome, float(prob)))
        conn.execute("""INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (game_id, model_name, model_version)
                        DO UPDATE SET exp_home = excluded.exp_home, exp_away = excluded.exp_away,
                                      created_at = datetime('now')""",
                     (g.id, MODEL_NAME, MODEL_VERSION, float(p["home_points"]), float(p["away_points"])))
        favorite = g.home if p["margin"] >= 0 else g.away
        qb_note = "".join(f"  [{team}: {qb} starting]" for team, qb, changed in
                          ((g.away, g.away_qb, g.away_qb_change), (g.home, g.home_qb, g.home_qb_change)) if changed)
        lines.append(f"  {g.date} {kickoffs.get(g.id) or '--:--'}  {g.away:>3} at {g.home:<3}  "
                     f"{g.home} {p['home_win']:.0%}  ({favorite} by {abs(p['margin']):.1f}, total {p['total']:.1f}){qb_note}")
    conn.commit()
    if recording:
        records.export(conn, MODEL_NAME, MODEL_VERSION)
    conn.close()
    print(f"Saved NFL predictions for {len(lines)} games:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
