"""Ways to make the cups model more accurate, tried on the tuning seasons: python -m ucl.improve_test

Scored on the Champions League's and the Europa League's 2016-17 to 2020-21 games (about 1,600), the
result after 90 minutes. Each change is tried on its own against the current model:

- season_weight: games from earlier seasons count less again (squads change in the summer)
- defense_shrinkage: defense ratings pulled harder toward the league than attack ratings (a run of
  clean sheets may be partly luck)
- europe_weight and team_shrinkage, re-checked with the Europa League's games in the score
- squad values: each club's Transfermarkt squad value as part of its rating (ucl/squad_values.py),
  with the clubs' own ratings pulled toward what their squads predict (team_shrinkage)
- rho: the Dixon-Coles adjustment (more 0-0s and 1-1s than independent Poisson counts give),
  applied to the chances only, so it needs no refit
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from db import DB_PATH
from epl.model import scoreline_grid
from ucl import model as m

TUNING = (2016, 2020)
COMPETITIONS = ("uefa.champions", "uefa.europa")
VARIANTS = {"current": {}, "season_weight 0.8": {"season_weight": 0.8}, "season_weight 0.6": {"season_weight": 0.6},
            "defense_shrinkage 2": {"defense_shrinkage": 2.0}, "defense_shrinkage 4": {"defense_shrinkage": 4.0},
            "europe_weight 8": {"europe_weight": 8.0}, "europe_weight 2": {"europe_weight": 2.0},
            "team_shrinkage 10": {"team_shrinkage": 10.0},
            "squad values": {"values": True}, "squad values, team_shrinkage 10": {"values": True, "team_shrinkage": 10.0},
            "squad values, team_shrinkage 20": {"values": True, "team_shrinkage": 20.0},
            "squad values, team_shrinkage 40": {"values": True, "team_shrinkage": 40.0},
            "squad values, shrinkage 20, half-life 365": {"values": True, "team_shrinkage": 20.0, "half_life_days": 365},
            "squad values, shrinkage 20, season_weight 0.8": {"values": True, "team_shrinkage": 20.0, "season_weight": 0.8},
            "squad values, shrinkage 20, league_shrinkage 3": {"values": True, "team_shrinkage": 20.0, "league_shrinkage": 3.0},
            "squad values, shrinkage 20, europe_weight 2": {"values": True, "team_shrinkage": 20.0, "europe_weight": 2.0}}
RHOS = [0.0, -0.05, -0.1, -0.15]


def with_rho(results: pd.DataFrame, rho: float) -> pd.DataFrame:
    out = results.copy()
    p = [(lambda g: (np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()))(scoreline_grid(h, a, rho=rho))
         for h, a in zip(results["exp_home"], results["exp_away"])]
    out[["home", "draw", "away"]] = p
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games = m.load_games(conn)
    values = m.load_values(conn)
    names = argv or list(VARIANTS)
    for name in names:
        settings = {k: (values if k == "values" else v) for k, v in VARIANTS[name].items()}
        results = m.backtest(games, *TUNING, competitions=COMPETITIONS, **settings)
        scores = {rho: m.log_loss(with_rho(results, rho), games) for rho in (RHOS if name == "current" else [0.0])}
        by_cup = {c: m.log_loss(results[results["competition"] == c], games) for c in COMPETITIONS}
        print(f"{name:<22} " + "  ".join(f"rho {r:+.2f}: {v:.4f}" for r, v in scores.items())
              + f"   (Champions League {by_cup['uefa.champions']:.4f}, Europa League {by_cup['uefa.europa']:.4f}, {len(results)} games)",
              flush=True)


if __name__ == "__main__":
    main()
