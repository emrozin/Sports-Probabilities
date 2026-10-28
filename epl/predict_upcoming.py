"""Predict upcoming EPL fixtures and save the predictions.

Usage:
    python -m epl.predict_upcoming            # on your computer
    python -m epl.predict_upcoming --record   # the automated weekly run: also updates the official record

Fits the model on finished matches before today and predicts every scheduled match
that hasn't kicked off yet. A match's prediction can be refreshed until kickoff and
is never changed after it, so stored predictions form a genuine pre-match track record.

The official record is records/live_predictions.csv, written by the automated weekly
run (--record). Without --record, recorded predictions are loaded into your local
database and left as they are; only matches missing from the record are predicted
locally, and the record file itself isn't changed.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from epl import records
from epl.import_fixtures import ensure_kickoff_column
from epl.model import TRAINING_YEARS, fit_poisson, load_market, load_totals, load_with_championship, predict

DB_PATH = "data/sports.db"
MODEL_NAME = "poisson_xg"
MODEL_VERSION = "v2"  # 180-day half-life, shrinkage 0.5, xG weight 0.5
UK = ZoneInfo("Europe/London")  # fixture dates and kickoff times are UK local time


def has_started(match_date: str, kickoff: str | None, now: datetime) -> bool:
    """True once a match has kicked off. Without a kickoff time, the whole matchday counts as started."""
    if kickoff:
        start = datetime.fromisoformat(f"{match_date} {kickoff}").replace(tzinfo=UK)
        return now >= start
    return match_date <= now.astimezone(UK).date().isoformat()


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    recording = "--record" in argv
    now = now or datetime.now(UK)
    today = now.astimezone(UK).date().isoformat()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(Path("schema.sql").read_text())  # adds any tables an older database is missing
    ensure_kickoff_column(conn)
    restored = records.restore(conn)
    recorded_ids = {row[0] for row in conn.execute(
        "SELECT DISTINCT game_id FROM predictions WHERE model_name = ? AND model_version = ?",
        (MODEL_NAME, MODEL_VERSION))}

    upcoming = pd.read_sql("""
        SELECT g.id, g.date, g.kickoff, h.name AS home, a.name AS away
        FROM games g
        JOIN leagues l ON l.id = g.league_id
        JOIN teams h   ON h.id = g.home_team_id
        JOIN teams a   ON a.id = g.away_team_id
        WHERE l.code = 'EPL' AND g.status = 'scheduled' AND g.date >= ?
        ORDER BY g.date, g.kickoff, h.name
    """, conn, params=(today,))

    started = [m for m in upcoming.itertuples() if has_started(m.date, m.kickoff, now)]
    to_predict = [m for m in upcoming.itertuples() if not has_started(m.date, m.kickoff, now)]
    if not recording:
        # Locally, the official record stands: only predict matches it doesn't have yet.
        to_predict = [m for m in to_predict if m.id not in recorded_ids]

    if restored:
        print(f"Loaded {restored} recorded predictions from {records.RECORD_PATH}.")
    for m in started:
        print(f"Not predicting {m.home} v {m.away}: it has already kicked off.")
    if not to_predict:
        print("No upcoming matches to predict. Fixtures are posted on Tuesday (midweek) "
              "and Friday (weekend) afternoons.")
        if recording:
            records.export(conn, MODEL_NAME, MODEL_VERSION)
        conn.close()
        return

    # Train only on matches finished before today, exactly like the backtest.
    games = load_with_championship(conn)          # the Championship rates newly promoted teams
    window_start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    train = games[(games["date"] >= window_start) & (games["date"] < today)]
    model = fit_poisson(train, today)
    known = set(model["ratings"].index)
    market = load_market(conn, closing=False).set_index("id")
    totals = load_totals(conn, closing=False).set_index("id")

    rows = []
    for m in to_predict:
        if m.home not in known or m.away not in known:
            print(f"Skipped {m.home} v {m.away}: no recent matches to rate a team on")
            continue
        p = predict(model, m.home, m.away)
        for outcome in ("home", "draw", "away"):
            conn.execute(
                """
                INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (game_id, model_name, model_version, outcome)
                DO UPDATE SET probability = excluded.probability,
                              created_at  = datetime('now')
                """,
                (m.id, MODEL_NAME, MODEL_VERSION, outcome, float(p[outcome])),
            )
        conn.execute(
            """
            INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (game_id, model_name, model_version)
            DO UPDATE SET exp_home = excluded.exp_home, exp_away = excluded.exp_away,
                          created_at = datetime('now')
            """,
            (m.id, MODEL_NAME, MODEL_VERSION, float(p["exp_home"]), float(p["exp_away"])),
        )

        row = {
            "date": m.date,
            "kickoff": m.kickoff or "-",
            "match": f"{m.home} v {m.away}",
            "exp goals": f"{p['exp_home']:.2f}-{p['exp_away']:.2f}",
            "model H/D/A": f"{p['home']:.0%} / {p['draw']:.0%} / {p['away']:.0%}",
        }
        if m.id in market.index:
            mk = market.loc[m.id]
            row["market H/D/A"] = f"{mk['mkt_home']:.0%} / {mk['mkt_draw']:.0%} / {mk['mkt_away']:.0%}"
        row["over 2.5"] = f"{p['over_2.5']:.0%}"
        if m.id in totals.index:
            row["over 2.5"] += f" (mkt {totals.loc[m.id, 'mkt_over']:.0%})"
        i, j, prob = p["top_scores"][0]
        row["likeliest"] = f"{i}-{j} ({prob:.0%})"
        rows.append(row)
    conn.commit()

    print(f"Model trained on {len(train)} matches before {today}. Saved predictions for {len(rows)} matches:\n")
    print(pd.DataFrame(rows).fillna("-").to_string(index=False))
    if recording:
        total = records.export(conn, MODEL_NAME, MODEL_VERSION)
        print(f"\nOfficial record updated: {records.RECORD_PATH} now holds {total} predictions.")
    conn.close()


if __name__ == "__main__":
    main()
