"""Tune the Champions League model: python -m ucl.tune_model [held_out]

Scored by log loss on the Champions League's games (league or group phase and knockouts, the
result after 90 minutes), walk-forward. Settings are chosen on the tuning seasons, 2016-17 to
2020-21, one at a time around the current values; then the chosen model is checked once on the
held-out seasons (2021-22 on) against two benchmarks:

- base rates: every game gets the tuning seasons' share of home wins, draws and away wins
- Elo: a rating for every club from every game (domestic and European), updated after each game
  as Elo does (K = 20, scaled up by the goal difference), the way clubelo.com rates clubs; a game's
  chances come from the rating difference through an ordered logistic model fitted on the tuning
  seasons.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from db import DB_PATH
from ucl import model as m
from ucl.elo import ordered_logit, run as run_elo

TUNING = (2016, 2020)
HELD_OUT = (2021, 2025)
GRID = {"half_life_days": [180, 365, 730], "team_shrinkage": [2.0, 5.0, 10.0, 20.0],
        "league_shrinkage": [0.3, 1.0, 3.0], "europe_weight": [1.0, 2.0, 4.0], "training_years": [2, 3, 4]}


def elo_ratings(games: pd.DataFrame) -> pd.DataFrame:
    """Each game's pre-game Elo difference (home minus away, home advantage included): ucl/elo.py."""
    return run_elo(games)[0]


def benchmark_chances(games: pd.DataFrame, tuning: tuple = TUNING, competitions=(m.UCL,)) -> pd.DataFrame:
    """Every scored game's chances from the two benchmarks, fitted on the tuning seasons:
    event_id, base_home/draw/away, elo_home/draw/away."""
    scored = m.scored_games(games, competitions)
    elo = elo_ratings(games).merge(scored[["event_id", "season", "home_score", "away_score", "extra_time"]], on="event_id")
    outcome = np.array([{"away": 0, "draw": 1, "home": 2}[m.result_of(h, a, e)]
                        for h, a, e in zip(elo["home_score"], elo["away_score"], elo["extra_time"].fillna(0))])
    tune = elo["season"].between(*tuning).to_numpy()
    base = np.bincount(outcome[tune], minlength=3) / max(tune.sum(), 1)
    b, c1, c2 = ordered_logit(elo["elo_diff"].to_numpy()[tune], outcome[tune])
    s = lambda z: 1 / (1 + np.exp(-z))
    d = elo["elo_diff"].to_numpy()
    p_away, p_not_home = s(c1 - b * d), s(c2 - b * d)
    return pd.DataFrame({"event_id": elo["event_id"], "base_home": base[2], "base_draw": base[1], "base_away": base[0],
                         "elo_home": 1 - p_not_home, "elo_draw": p_not_home - p_away, "elo_away": p_away})


def benchmarks(games: pd.DataFrame, held: tuple) -> dict:
    scored = m.scored_games(games)
    elo = elo_ratings(games).merge(scored[["event_id", "season", "home_score", "away_score", "extra_time"]], on="event_id")
    outcome = np.array([{"away": 0, "draw": 1, "home": 2}[m.result_of(h, a, e)]
                        for h, a, e in zip(elo["home_score"], elo["away_score"], elo["extra_time"].fillna(0))])
    tune = elo["season"].between(*TUNING).to_numpy()
    test = elo["season"].between(*held).to_numpy()
    base = np.bincount(outcome[tune], minlength=3) / tune.sum()
    b, c1, c2 = ordered_logit(elo["elo_diff"].to_numpy()[tune], outcome[tune])
    s = lambda z: 1 / (1 + np.exp(-z))
    d = elo["elo_diff"].to_numpy()[test]
    p_away, p_not_home = s(c1 - b * d), s(c2 - b * d)
    p = np.choose(outcome[test], [p_away, p_not_home - p_away, 1 - p_not_home])
    return {"base_rates": float(-np.mean(np.log(base[outcome[test]]))), "elo": float(-np.mean(np.log(p))),
            "games": int(test.sum())}


def score(games: pd.DataFrame, span: tuple, **settings) -> float:
    return m.log_loss(m.backtest(games, *span, **settings), games)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    games = m.load_games(sqlite3.connect(DB_PATH))
    if "held_out" in argv:
        chosen = {k: getattr(m, k.upper()) for k in GRID}
        print(f"held out {HELD_OUT}: model {score(games, HELD_OUT, **chosen):.4f}", benchmarks(games, HELD_OUT), flush=True)
        return
    current = {k: getattr(m, k.upper()) for k in GRID}
    best = score(games, TUNING, **current)
    print(f"start {current}: {best:.4f}  benchmarks {benchmarks(games, TUNING)}", flush=True)
    for _ in range(2):                                   # two passes, one setting at a time
        for name, values in GRID.items():
            for v in values:
                if v == current[name]:
                    continue
                trial = {**current, name: v}
                loss = score(games, TUNING, **trial)
                print(f"  {name} = {v}: {loss:.4f}", flush=True)
                if loss < best - 1e-5:
                    best, current = loss, trial
            print(f"best so far {current}: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
