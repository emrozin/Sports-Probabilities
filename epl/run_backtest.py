"""Run the walk-forward backtest and save its predictions to the database.

Usage:
    python -m epl.run_backtest

For every week since 2016/17, the model is trained only on matches played before
that week (Premier League and Championship) and predicts that week's games. The predictions are saved under their
own model name, separate from the live predictions made by predict_upcoming.py.
The website's backtest and results pages read them. Re-run after importing new
results so the backtest covers the latest matches (it takes a few seconds).
"""

import sqlite3
from pathlib import Path

from epl.model import backtest, load_with_championship

DB_PATH = "data/sports.db"
BACKTEST_NAME = "poisson_xg_backtest"
BACKTEST_VERSION = "v3"           # v3: Championship matches help rate promoted teams
START_SEASON = 2016
HELD_OUT_FROM = 2021  # settings were tuned on 2016/17-2020/21; 2021/22 onward is the honest test


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(Path("schema.sql").read_text())  # adds any tables an older database is missing
    games = load_with_championship(conn)
    results = backtest(games, start_season=START_SEASON)

    rows = [
        (r.id, BACKTEST_NAME, BACKTEST_VERSION, outcome, float(getattr(r, f"model_{outcome}")))
        for r in results.itertuples()
        for outcome in ("home", "draw", "away")
    ]
    goals = [(r.id, BACKTEST_NAME, BACKTEST_VERSION, float(r.exp_home), float(r.exp_away))
             for r in results.itertuples()]
    # Replace the previous backtest run entirely, in one transaction.
    with conn:
        for table in ("predictions", "prediction_goals"):
            conn.execute(f"DELETE FROM {table} WHERE model_name = ? AND model_version = ?",
                         (BACKTEST_NAME, BACKTEST_VERSION))
        conn.executemany(
            """
            INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.executemany(
            """
            INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away)
            VALUES (?, ?, ?, ?, ?)
            """,
            goals,
        )
    conn.close()
    print(f"Saved backtest predictions for {len(results)} matches since {START_SEASON}/{(START_SEASON + 1) % 100:02d}.")


if __name__ == "__main__":
    main()
