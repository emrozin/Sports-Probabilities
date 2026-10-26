"""Run the NBA walk-forward backtest and save its predictions: python -m nba.run_backtest

Every week since 2005-06, the model is fitted only on games before that week and predicts that
week's games the way a live prediction does: each team's lineup is every player available for the
game (he played, or sat by the coach's decision; injured players are what the injury report would
have ruled out), with his usual minutes. Settings were tuned on 2010-11 to 2013-14
(nba/tune_model.py), so 2014-15 onward is the honest test: log loss 0.6124 on 15,476 games,
against 0.6265 for a simple Elo rating. Version 2 adds back-to-backs (0.6135 without them; tested
with python -m nba.rest_test).

The full run takes several minutes, so by default only the seasons not saved yet and the current
season are (re)done; python -m nba.run_backtest --full redoes everything.
"""

import sqlite3
import sys

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from nba.model import backtest, load_games, load_players

BACKTEST_NAME = "nba_players_backtest"
BACKTEST_VERSION = "v2"   # v2: back-to-backs
START_SEASON = 2005
HELD_OUT_FROM = 2014


def save_seasons(conn, results: pd.DataFrame, seasons: list[int]) -> None:
    """Replace the saved predictions for these seasons only."""
    seasons = [int(s) for s in seasons]           # plain ints: SQLite doesn't match numpy integers
    ids = pd.read_sql(f"SELECT id FROM games WHERE season IN ({', '.join('?' * len(seasons))}) "
                      f"AND league_id = (SELECT id FROM leagues WHERE code = 'NBA')", conn, params=seasons)["id"].tolist()
    with conn:
        for table in ("predictions", "prediction_goals"):
            conn.executemany(f"DELETE FROM {table} WHERE model_name = ? AND model_version = ? AND game_id = ?",
                             [(BACKTEST_NAME, BACKTEST_VERSION, i) for i in ids])
        conn.executemany("INSERT INTO predictions (game_id, model_name, model_version, outcome, probability) VALUES (?, ?, ?, ?, ?)",
                         [(r.id, BACKTEST_NAME, BACKTEST_VERSION, o, float(p)) for r in results.itertuples()
                          for o, p in (("home", r.model_home), ("away", 1 - r.model_home))])
        conn.executemany("INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away) VALUES (?, ?, ?, ?, ?)",
                         [(r.id, BACKTEST_NAME, BACKTEST_VERSION, float(r.home_points), float(r.away_points))
                          for r in results.itertuples()])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    if games.empty:
        raise SystemExit("No NBA games found. Run python -m nba.import_nba first.")
    finished = games[games["status"] == "final"]
    seasons = sorted(int(s) for s in finished["season"].unique() if s >= START_SEASON)
    saved = set(pd.read_sql("""SELECT DISTINCT g.season FROM predictions p JOIN games g ON g.id = p.game_id
                               WHERE p.model_name = ? AND p.model_version = ?""", conn,
                            params=(BACKTEST_NAME, BACKTEST_VERSION))["season"])
    todo = seasons if "--full" in argv else [s for s in seasons if s not in saved or s == seasons[-1]]
    if not todo:
        print("NBA backtest is up to date.")
        conn.close()
        return
    players = load_players(conn)
    done = 0
    for season in todo:          # one season at a time, saved as it goes
        results = backtest(games, players, season, season)
        save_seasons(conn, results, [season])
        done += len(results)
    conn.close()
    span = f"{todo[0]}-{(todo[0] + 1) % 100:02d}" + (f" to {todo[-1]}-{(todo[-1] + 1) % 100:02d}" if len(todo) > 1 else "")
    print(f"Saved NBA backtest predictions for {done} games ({span}).")


if __name__ == "__main__":
    main()
