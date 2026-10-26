"""Does knowing about back-to-backs help? python -m nba.rest_test [tuning|held_out]

Walk-forward backtests (live-like lineups), with and without the back-to-back term, scored by log
loss: on the tuning seasons (2010-11 to 2013-14) for several strengths of the term's shrinkage,
then on the held-out seasons (2014-15 on) for the chosen one.

Results (log loss): tuning seasons 0.5950 without, 0.5939 with (shrinkage 1, 5 or 20 the same);
held-out 2014-15 to 2025-26 0.6135 without, 0.6124 with (picked the winner 65.7% -> 66.0%), on
15,476 games. The fitted effect is about 2.5 points against the team on a back-to-back.
"""

import sqlite3
import sys

import numpy as np

from db import DB_PATH
from nba.model import REST_SHRINKAGE, backtest, load_games, load_players, log_loss
from nba.run_backtest import HELD_OUT_FROM

TUNING = (2010, 2013)


def score(games, players, first, last, rest_shrinkage) -> tuple[float, float, int]:
    r = backtest(games, players, first, last, with_points=False, rest_shrinkage=rest_shrinkage)
    g = games.set_index("id").loc[r["id"]]
    won = (g["home_score"] > g["away_score"]).to_numpy()
    p = r["model_home"].to_numpy()
    return log_loss(p, won), float(np.mean((p > 0.5) == won)), len(r)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games, players = load_games(conn), load_players(conn)
    current = int(games["season"].max())
    b2b = games[(games["status"] == "final") & ((games["home_b2b"] == 1) | (games["away_b2b"] == 1))]
    print(f"Games with a team on a back-to-back: {len(b2b)} of {int((games['status'] == 'final').sum())}")
    if "held_out" in argv:
        settings = [None, REST_SHRINKAGE]
        first, last = HELD_OUT_FROM, current - 1
    else:
        settings = [None, 1.0, 5.0, 20.0]
        first, last = TUNING
    for s in settings:
        loss, acc, n = score(games, players, first, last, s)
        label = "without back-to-backs" if s is None else f"back-to-back term, shrinkage {s:g}"
        print(f"{first}-{last}  {label:<40} log loss {loss:.4f}  picked the winner {acc:.1%}  ({n} games)", flush=True)


if __name__ == "__main__":
    main()
