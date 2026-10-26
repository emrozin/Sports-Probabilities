"""Tune the basketball model on 2010-11 to 2017-18: python -m cbb.tune_model

A grid of recency (half-life), the weight of earlier seasons, shrinkage and training years; for each,
the walk-forward backtest's log loss at several margin spreads. The held-out seasons (2018-19 on)
are scored once in cbb/run_backtest.py.
"""

import itertools
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm

from cbb.model import backtest, load_games
from db import DB_PATH

TUNING = (2010, 2017)
MARGIN_SDS = [9, 10, 10.5, 11, 11.5, 12, 13]
# Best: 120, 0.8, 0.25, 3 (0.5186); then checked past the edges: season weight 1.0 gave 0.5183,
# shrinkage 0.1 or 0.03 no better.
GRID = {"half_life": [60, 120, 240], "season_weight": [0.3, 0.5, 0.8, 1.0], "shrinkage": [0.1, 0.25, 1, 4], "years": [2, 3]}


def score(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    d = results.merge(games[["id", "home_score", "away_score"]], on="id")
    won = (d["home_score"] > d["away_score"]).to_numpy()
    out = {}
    for sd in MARGIN_SDS:
        p = np.clip(norm.cdf(d["model_margin"] / sd), 1e-6, 1 - 1e-6)
        out[sd] = float(-np.mean(np.where(won, np.log(p), np.log(1 - p))))
    return out


def main() -> None:
    games = load_games(sqlite3.connect(DB_PATH))
    rows = []
    for half_life, season_weight, shrinkage, years in itertools.product(*GRID.values()):
        results = backtest(games, *TUNING, half_life_days=half_life, season_weight=season_weight,
                           shrinkage=shrinkage, training_years=years)
        for sd, loss in score(results, games).items():
            rows.append({"half_life": half_life, "season_weight": season_weight, "shrinkage": shrinkage,
                         "years": years, "margin_sd": sd, "log_loss": loss})
        print(f"{half_life}, {season_weight}, {shrinkage}, {years}: best so far {min(r['log_loss'] for r in rows):.4f}", flush=True)
    print(pd.DataFrame(rows).sort_values("log_loss").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
