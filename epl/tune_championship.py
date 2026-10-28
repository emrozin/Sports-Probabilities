"""Does Championship data help rate newly promoted teams? python -m epl.tune_championship [held_out]

Needs the Championship in the database: python -m epl.import_epl --league CHAMP

The model is fitted on Premier League and Championship matches together (epl.model.fit_with_championship),
with Championship matches counting champ_weight as much (0 is today's model, Premier League only).
Walk-forward backtests on the tuning seasons (2016/17-2020/21) pick the weight; the chosen one is
then compared with today's model once, on the held-out seasons (2021/22 on). Scored by log loss on
every Premier League match and on newly promoted teams' matches (where the help should show),
with the closing market for comparison.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from epl.model import add_scores, backtest, load_market, load_with_championship
from epl.promoted_teams import promoted_teams

DB_PATH = "data/sports.db"
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0]
TUNE = (2016, 2020)
TEST = (2021, 2025)


def scored(games, seasons, weight, market) -> pd.DataFrame:
    first, last = seasons
    results = backtest(games[games["season"] <= last], start_season=first, champ_weight=weight)
    results = results.merge(market, on="id", how="left")
    add_scores(results, "model")
    return results


def summary(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    epl = games[games["division"] == "EPL"]
    promoted = promoted_teams(epl)
    teams = epl.set_index("id")[["home", "away"]]
    r = results.join(teams, on="id")
    has_promoted = [(s, h) in promoted or (s, a) in promoted for s, h, a in zip(r["season"], r["home"], r["away"])]
    r["promoted"] = has_promoted
    with_market = r.dropna(subset=["mkt_home"]).copy()
    if not with_market.empty:
        add_scores(with_market, "mkt")
    return {"all": (len(r), r["model_logloss"].mean()), "promoted": (int(r["promoted"].sum()), r.loc[r["promoted"], "model_logloss"].mean()),
            "market_all": with_market["mkt_logloss"].mean() if not with_market.empty else np.nan,
            "market_promoted": with_market.loc[with_market["promoted"], "mkt_logloss"].mean() if not with_market.empty else np.nan}


def show(label: str, s: dict) -> None:
    print(f"  {label:<22} all {s['all'][1]:.4f} ({s['all'][0]})   promoted teams' matches {s['promoted'][1]:.4f} ({s['promoted'][0]})"
          f"   market {s['market_all']:.4f} / {s['market_promoted']:.4f}", flush=True)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games = load_with_championship(conn)
    market = load_market(conn, closing=True)
    conn.close()
    if not (games["division"] == "CHAMP").any():
        raise SystemExit("No Championship matches found. Run python -m epl.import_epl --league CHAMP first.")
    if "held_out" in argv:
        weight = float(argv[argv.index("held_out") + 1]) if len(argv) > argv.index("held_out") + 1 else 0.1
        print(f"Held out {TEST[0]}/{(TEST[0] + 1) % 100:02d} to {TEST[1]}/{(TEST[1] + 1) % 100:02d}:")
        runs = {w: scored(games, TEST, w, market) for w in (0.0, weight)}
        common = set.intersection(*(set(r["id"]) for r in runs.values()))
        for w, r in runs.items():
            show(f"weight {w}", summary(r, games))
            show("  same games as 0.0", summary(r[r["id"].isin(common)], games))
        return
    print(f"Tuning seasons {TUNE[0]}/{(TUNE[0] + 1) % 100:02d} to {TUNE[1]}/{(TUNE[1] + 1) % 100:02d}:")
    for w in WEIGHTS:
        show(f"weight {w}", summary(scored(games, TUNE, w, market), games))


if __name__ == "__main__":
    main()
