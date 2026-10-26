"""Run the basketball walk-forward backtest and save its predictions: python -m cbb.run_backtest

Every week since 2010-11, the model is trained only on games before that week and predicts that
week's games between Division I teams. Settings were tuned on 2010-11 to 2017-18
(cbb/tune_model.py), so 2018-19 onward is the honest test (log loss 0.5382, a simple Elo 0.5491). Takes about a minute.
"""

import sqlite3

from cbb.model import backtest, load_games
from db import DB_PATH, SCHEMA_PATH
from nfl.run_backtest import save

BACKTEST_NAME = "cbb_points_backtest"
BACKTEST_VERSION = "v2"
START_SEASON = 2010
HELD_OUT_FROM = 2018


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    if games.empty:
        raise SystemExit("No basketball games found. Run python -m cbb.import_cbb first.")
    results = backtest(games, START_SEASON)
    save(conn, results, BACKTEST_NAME, BACKTEST_VERSION)
    conn.close()
    print(f"Saved basketball backtest predictions for {len(results)} games since {START_SEASON}-{(START_SEASON + 1) % 100:02d}.")


if __name__ == "__main__":
    main()
