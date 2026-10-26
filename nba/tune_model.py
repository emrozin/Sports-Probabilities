"""Tune the NBA model on 2010-11 to 2013-14: python -m nba.tune_model

Walk-forward backtests (with the players who played; see nba/model.py's team_lineups) for each combination of player
and team shrinkage, half-life and training years; for each, the margin spread (MARGIN_SD) that
fits best. Prints the best combinations by log loss. Takes about an hour (each run refits the
model every week for four seasons).

Results (log loss): player shrinkage 10, team 40, half-life 365 days, 3 years, spread 11: 0.5928
(15: 0.5929; 5: 0.5941; 3: 0.5962; 1: 0.6029); a 730-day half-life was 0.0007-0.0014 worse. For
comparison on the same seasons: team ratings alone 0.6164 (untuned), a simple Elo 0.5992.
"""

import itertools
import sqlite3

import pandas as pd
from scipy.stats import norm

from db import DB_PATH
from nba.model import backtest, load_games, load_players, log_loss

TUNING = (2010, 2013)
MARGIN_SDS = [10, 10.5, 11, 11.5, 12, 12.5, 13, 14]
GRID = {"player_shrinkage": [3, 5, 10, 15], "team_shrinkage": [10, 40], "half_life_days": [240, 365, 730], "training_years": [3]}


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games, players = load_games(conn), load_players(conn)
    won = games.set_index("id")["home_score"] > games.set_index("id")["away_score"]
    rows = []
    for combo in itertools.product(*GRID.values()):
        settings = dict(zip(GRID, combo))
        r = backtest(games, players, *TUNING, mode="played", with_points=False, **settings)
        y = won.loc[r["id"]].to_numpy()
        for sd in MARGIN_SDS:
            rows.append({**settings, "margin_sd": sd, "log_loss": log_loss(norm.cdf(r["model_margin"] / sd), y)})
        print(f"{combo}: best so far {min(x['log_loss'] for x in rows):.4f}", flush=True)
    print(pd.DataFrame(rows).sort_values("log_loss").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
