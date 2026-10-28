"""Run the NFL walk-forward backtest and save its predictions to the database.

Usage:
    python -m nfl.run_backtest

For every week since 2010, the points model (with quarterback ratings) is trained only on
games played before that week and predicts that week's games. The predictions are saved under
their own model name, separate from live predictions, and the website's results, team and
backtest pages read them. Re-run after importing new results (it takes a few seconds).
"""

import sqlite3

from db import DB_PATH, SCHEMA_PATH
from nfl.elo import load_nfl_games
from nfl.model import add_qb_changes, backtest
from nfl.starters import add_to_games, missing_starters

BACKTEST_NAME = "nfl_points_qb_backtest"
BACKTEST_VERSION = "v3"   # v2: quarterback ratings; v3: missing starters
START_SEASON = 2010
HELD_OUT_FROM = 2018  # settings were tuned on 2010-2017; 2018 onward is the honest test


def save(conn, results, name=BACKTEST_NAME, version=BACKTEST_VERSION) -> None:
    """Replace a model's saved predictions: home/away win chances and expected points for each side."""
    rows = [(r.id, name, version, outcome, float(p))
            for r in results.itertuples() for outcome, p in (("home", r.model_home), ("away", 1 - r.model_home))]
    # prediction_goals holds each prediction's expected score: goals for soccer, points for the NFL.
    points = [(r.id, name, version, float(r.home_points), float(r.away_points)) for r in results.itertuples()]
    with conn:
        for table in ("predictions", "prediction_goals"):
            conn.execute(f"DELETE FROM {table} WHERE model_name = ? AND model_version = ?", (name, version))
        conn.executemany("INSERT INTO predictions (game_id, model_name, model_version, outcome, probability) "
                         "VALUES (?, ?, ?, ?, ?)", rows)
        conn.executemany("INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away) "
                         "VALUES (?, ?, ?, ?, ?)", points)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = add_qb_changes(load_nfl_games(conn))
    if games.empty:
        raise SystemExit("No NFL games found. Run python -m nfl.import_nfl first.")
    games = add_to_games(games, missing_starters(conn))
    results = backtest(games, START_SEASON)
    save(conn, results)
    conn.close()
    print(f"Saved NFL backtest predictions for {len(results)} games since {START_SEASON}.")


if __name__ == "__main__":
    main()
