"""Better ratings at the start of the season? python -m cbb.preseason_test [tuning|held_out]

The original model rates teams on every game of the last three seasons by recency, so in November
a team is rated almost exactly as it finished in March, although rosters turn over a lot. The
alternative: preseason ratings (the March ratings pulled part of the way toward the team's new
conference's average: `carry` is the share kept) count as `prior_weight` games, and the season's
own games take over as they're played.

Scored by log loss on the early season (games before 15 December) and the whole season: on the
tuning seasons (2010-11 to 2017-18) for a grid of settings, then once on the held-out seasons
(2018-19 on) for the chosen ones.
"""

import itertools
import sqlite3
import sys

import numpy as np
import pandas as pd

from cbb.model import backtest, load_games
from db import DB_PATH

TUNING = (2010, 2017)
HELD_OUT = (2018, 2025)
GRID = {"carry": [0.5, 0.65, 0.8, 1.0], "prior_weight": [4, 8, 16, 32]}
CHOSEN = {"carry": 0.65, "prior_weight": 8, "switch": "12-15"}   # set from the tuning run


def early(date: pd.Series) -> pd.Series:
    """Games before 15 December of the season's first calendar year."""
    d = pd.to_datetime(date)
    return (d.dt.month >= 10) & ((d.dt.month < 12) | (d.dt.day < 15))


def score(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    d = results.merge(games[["id", "date", "home_score", "away_score"]], on="id")
    won = (d["home_score"] > d["away_score"]).to_numpy()
    p = np.clip(d["model_home"].to_numpy(), 1e-6, 1 - 1e-6)
    loss = -np.where(won, np.log(p), np.log(1 - p))
    first = early(d["date"]).to_numpy()
    return {"early": float(loss[first].mean()), "season": float(loss.mean()), "early_games": int(first.sum()),
            "games": len(d), "early_picked": float(((p > 0.5) == won)[first].mean())}


def show(label: str, s: dict) -> None:
    print(f"{label:<44} early season {s['early']:.4f} ({s['early_games']} games, picked {s['early_picked']:.1%})  "
          f"whole season {s['season']:.4f} ({s['games']} games)", flush=True)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = load_games(sqlite3.connect(DB_PATH))
    span = HELD_OUT if "held_out" in argv else TUNING
    show(f"{span[0]}-{span[1]} original", score(backtest(games, *span), games))
    settings = [CHOSEN] if "held_out" in argv else [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())] + [CHOSEN]
    for s in settings:
        show(f"{span[0]}-{span[1]} carry {s['carry']:.2f}, worth {s['prior_weight']} games"
             + (f", until {s['switch']}" if s.get("switch") else ""), score(backtest(games, *span, **s), games))


if __name__ == "__main__":
    main()
