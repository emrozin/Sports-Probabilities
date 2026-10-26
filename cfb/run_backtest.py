"""Run the college football walk-forward backtest and save its predictions: python -m cfb.run_backtest

For every week since 2010, the model is trained only on games played before that week and
predicts that week's games involving a top-division (FBS) team. Settings were tuned on 2010-2017
(cfb/tune_model.py), so 2018 onward is the honest test. Takes about 15 seconds.
"""

import sqlite3

from cfb.model import backtest, load_games
from db import DB_PATH, SCHEMA_PATH
from nfl.run_backtest import save

BACKTEST_NAME = "cfb_points_backtest"
BACKTEST_VERSION = "v2"   # v2: quarterback ratings and the new-starter penalty
START_SEASON = 2010
HELD_OUT_FROM = 2018


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    if games.empty:
        raise SystemExit("No college football games found. Run python -m cfb.import_cfb first.")
    results = backtest(games, START_SEASON)
    save(conn, results, BACKTEST_NAME, BACKTEST_VERSION)
    conn.close()
    print(f"Saved college football backtest predictions for {len(results)} games since {START_SEASON}.")


if __name__ == "__main__":
    main()
