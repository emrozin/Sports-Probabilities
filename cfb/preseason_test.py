"""Better college football ratings at the start of the season? python -m cfb.preseason_test [tuning|held_out]

The usual model rates teams on the last four seasons' games, earlier seasons counting less, so in
September a team is rated much as it finished last year, although rosters turn over a lot. The
alternative, like college basketball's (cbb/preseason_test.py): preseason ratings (last season's
ratings pulled part of the way toward the team's conference average: `carry` is the share kept)
count as `prior_weight` games, and this season's games take over as they're played, until the
`switch` date, after which the usual fit takes over again.

Scored by log loss on games involving a top-division (FBS) team: the early season (August and
September) and the whole season. Tuning seasons 2010-2017 for a grid of settings, then once on the
held-out seasons (2018 on) for the chosen one.
"""

import itertools
import sqlite3
import sys

import numpy as np
import pandas as pd

from cfb.model import backtest, load_games
from db import DB_PATH

TUNING = (2010, 2017)
HELD_OUT = (2018, 2025)
GRID = {"carry": [0.5, 0.7, 0.8, 0.85, 1.0], "prior_weight": [2, 3, 4, 5, 8], "switch": ["10-01", "11-01", "12-31"]}
CHOSEN = {"carry": 0.8, "prior_weight": 3, "switch": "11-01"}   # set from the tuning run


def early(date: pd.Series) -> pd.Series:
    """Games in August and September."""
    return pd.to_datetime(date).dt.month.isin([8, 9]).reset_index(drop=True)


def score(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    d = results.merge(games[["id", "date", "home_score", "away_score"]], on="id")
    d = d[d["home_score"] != d["away_score"]]
    won = (d["home_score"] > d["away_score"]).to_numpy()
    p = np.clip(d["model_home"].to_numpy(), 1e-6, 1 - 1e-6)
    loss = -np.where(won, np.log(p), np.log(1 - p))
    first = early(d["date"]).to_numpy()
    return {"early": float(loss[first].mean()), "season": float(loss.mean()), "early_games": int(first.sum()),
            "games": len(d), "early_picked": float(((p > 0.5) == won)[first].mean())}


def show(label: str, s: dict) -> None:
    print(f"{label:<52} early season {s['early']:.4f} ({s['early_games']} games, picked {s['early_picked']:.1%})  "
          f"whole season {s['season']:.4f} ({s['games']} games)", flush=True)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = load_games(sqlite3.connect(DB_PATH))
    held_out = "held_out" in argv
    span = HELD_OUT if held_out else TUNING
    show(f"{span[0]}-{span[1]} usual", score(backtest(games, *span), games))
    settings = [CHOSEN] if held_out else [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())]
    for s in settings:
        show(f"{span[0]}-{span[1]} carry {s['carry']:.2f}, worth {s['prior_weight']} games, until {s['switch']}",
             score(backtest(games, *span, **s), games))


if __name__ == "__main__":
    main()
