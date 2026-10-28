"""March Madness: every team's chance of reaching each round of the NCAA tournament.

The bracket is a tree: each game is played by the winners of the two games (or the two teams) below
it. Going up from the first round, each team's chance of winning a game is the sum, over every
opponent who could be there, of the chance that opponent gets there times the chance of beating him
(the model's win chance on a neutral court). That gives exact chances, with no simulation noise.

Ratings are the college basketball model's on the morning the tournament starts (cbb/model.py).

The bracket:
- For past tournaments it's rebuilt from the games themselves (each game's two teams came from their
  previous tournament games), which is how the backtest below works.
- For this season, it's read from cbb/brackets/bracket_<season>.csv once the field is announced on
  Selection Sunday (region, seed, team; two teams on one seed for the First Four; the four regions
  listed in Final Four order: the first two regions' winners meet, then the last two). The page
  says so until the file exists.

Usage:
    python -m cbb.tournament            # the backtest: every tournament since 2010-11
    python -m cbb.tournament --record   # ...and save its results for the site (records/cbb_tournament_backtest.json)
"""

import json
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cbb.model import MARGIN_SD, fit, predict, window

TOURNAMENT = re.compile(r"Men's Basketball Championship - ", re.I)
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]   # the bracket's top-to-bottom order in a region
ROUNDS = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Final", "Champion"]
BRACKET_DIR = Path(__file__).resolve().parent / "brackets"
RECORD_PATH = Path("records/cbb_tournament_backtest.json")
CALIBRATION_BINS = [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0]


# ---------------------------------------------------------------- the bracket as a tree

def tournament_games(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """A season's NCAA tournament games (First Four to the final), in date order."""
    notes = games["notes"].fillna("")
    part = games[(games["season"] == season) & notes.str.contains(TOURNAMENT) & ~notes.str.contains("NIT|CBI|CIT|Crown", regex=True)]
    return part.sort_values(["date", "id"])


def tree_from_games(tour: pd.DataFrame, home: str = "home", away: str = "away"):
    """The bracket rebuilt from played games: a node is (left, right), a leaf a team name. Each
    game's two sides are the trees of the games that brought each team there."""
    last_game: dict = {}          # team -> the subtree that produced it so far
    tree = None
    for g in tour.itertuples():
        a, b = getattr(g, home), getattr(g, away)
        tree = (last_game.get(a, a), last_game.get(b, b))
        last_game[a] = last_game[b] = tree
    return tree


def tree_from_bracket(bracket: pd.DataFrame):
    """The bracket from a field file: region, seed, team (two rows on a seed for a First Four)."""
    regions = list(dict.fromkeys(bracket["region"]))
    if len(regions) != 4:
        raise ValueError(f"A bracket needs 4 regions, found {regions}")

    def region_tree(region):
        part = bracket[bracket["region"] == region]
        slots = []
        for seed in SEED_ORDER:
            teams = list(part.loc[part["seed"] == seed, "team"])
            if not teams:
                raise ValueError(f"{region}: no team on seed {seed}")
            slots.append(teams[0] if len(teams) == 1 else (teams[0], teams[1]))
        while len(slots) > 1:
            slots = [(slots[i], slots[i + 1]) for i in range(0, len(slots), 2)]
        return slots[0]
    r = [region_tree(x) for x in regions]
    return ((r[0], r[1]), (r[2], r[3]))


def leaves(tree) -> list[str]:
    return [tree] if isinstance(tree, str) else leaves(tree[0]) + leaves(tree[1])


def chances(tree, win) -> dict:
    """{team: [chance of winning its 1st, 2nd, ... game in the tree]} from win(a, b) = P(a beats b).
    Exact: at each game, sum over the possible opponents."""
    out: dict = {t: [] for t in leaves(tree)}

    def walk(node) -> dict:
        if isinstance(node, str):
            return {node: 1.0}
        left, right = walk(node[0]), walk(node[1])
        result = {}
        for a, pa in left.items():
            result[a] = pa * sum(pb * win(a, b) for b, pb in right.items())
        for b, pb in right.items():
            result[b] = pb * sum(pa * win(b, a) for a, pa in left.items())
        for t, p in result.items():
            out[t].append(p)
        return result
    walk(tree)
    return out


def by_round(tree, win) -> pd.DataFrame:
    """One row per team: its chance of reaching each round (ROUNDS), aligned so that a First Four
    team's first win puts it in the Round of 64."""
    c = chances(tree, win)
    depth = max(len(v) for v in c.values())       # 6 without a First Four game, 7 with
    rows = []
    for team, wins in c.items():
        first_four = len(wins) == depth and depth == 7
        steps = wins[1:] if first_four else wins
        steps = [1.0] * (6 - len(steps)) + steps     # a team that advanced without playing (a forfeit)
        reach = [wins[0] if first_four else 1.0] + steps    # reaching the round of 64, then each later round
        rows.append({"team": team, "first_four": first_four, **dict(zip(ROUNDS, reach[:len(ROUNDS)]))})
    return pd.DataFrame(rows).sort_values("Champion", ascending=False).reset_index(drop=True)


def model_win(model: dict, margin_sd: float = MARGIN_SD):
    @lru_cache(maxsize=None)
    def win(a: str, b: str) -> float:
        return predict(model, a, b, neutral=True, margin_sd=margin_sd)["home_win"]
    return win


# ---------------------------------------------------------------- this season's field

def bracket_path(season: int) -> Path:
    return BRACKET_DIR / f"bracket_{season}.csv"


def current_odds(games: pd.DataFrame, season: int, model: dict) -> pd.DataFrame | None:
    """This season's odds, from the bracket file (games already played count as decided), or None
    before it exists. games: load_games's (home_team/away_team are the names in the bracket)."""
    path = bracket_path(season)
    if not path.exists():
        return None
    bracket = pd.read_csv(path)
    model_chance = model_win(model)
    played = tournament_games(games[games["status"] == "final"], season)
    results = {}
    for g in played.itertuples():          # games already played count as decided
        winner, loser = (g.home_team, g.away_team) if g.home_score > g.away_score else (g.away_team, g.home_team)
        results[(winner, loser)], results[(loser, winner)] = 1.0, 0.0
    win = lambda a, b: results.get((a, b), model_chance(a, b))
    table = by_round(tree_from_bracket(bracket), win)
    info = bracket.drop_duplicates("team").set_index("team")
    return table.assign(region=table["team"].map(info["region"]), seed=table["team"].map(info["seed"]))


# ---------------------------------------------------------------- the backtest

def reached(tour: pd.DataFrame, team: str) -> int:
    """How many rounds past the round of 64 a team reached (0: lost its first game there or in the
    First Four; 6: champion)."""
    played = tour[(tour["home"] == team) | (tour["away"] == team)]
    wins = int(((played["home"] == team) & (played["home_score"] > played["away_score"])).sum()
               + ((played["away"] == team) & (played["away_score"] > played["home_score"])).sum())
    first_four = played["notes"].fillna("").str.contains("First Four|Opening Round", regex=True).any()
    return max(wins - int(first_four), 0)


def backtest(games: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """For each tournament, every team's predicted chance of reaching each round (with ratings from
    the morning it started) and how far it got."""
    played = games[games["status"] == "final"]
    rows = []
    for season in range(first, last + 1):
        tour = tournament_games(played, season)
        if len(tour) < 60:          # no tournament (2019-20) or incomplete data
            continue
        start = tour["date"].min()
        model = fit(window(played, start), start, season)
        tree = tree_from_games(tour)
        if not 64 <= len(leaves(tree)) <= 68:       # games missing from the data: the bracket can't be rebuilt
            continue
        table = by_round(tree, model_win(model))
        for r in table.to_dict("records"):
            got = reached(tour, r["team"])
            for k, name in enumerate(ROUNDS[1:], 1):
                rows.append({"season": season, "team": r["team"], "round": name, "p": r[name], "reached": int(got >= k)})
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> dict:
    """Calibration of the 'reaches round' chances, and the champions' predicted title chances."""
    p = results["p"].clip(1e-6, 1 - 1e-6)
    loss = float(-np.mean(np.where(results["reached"] == 1, np.log(p), np.log(1 - p))))
    base = results.groupby("round")["reached"].transform("mean").clip(1e-6, 1 - 1e-6)   # knowing only how many teams reach each round
    base_loss = float(-np.mean(np.where(results["reached"] == 1, np.log(base), np.log(1 - base))))
    champs = results[(results["round"] == "Champion") & (results["reached"] == 1)]
    title = results[results["round"] == "Champion"].copy()
    title["rank"] = title.groupby("season")["p"].rank(ascending=False)
    champ_rank = title[title["reached"] == 1]
    return {"loss": loss, "base_loss": base_loss, "tournaments": int(results["season"].nunique()),
            "champion_chance": float(champs["p"].mean()) if len(champs) else None,
            "champion_was_favorite": int((champ_rank["rank"] == 1).sum()),
            "champion_top4": int((champ_rank["rank"] <= 4).sum())}


def main(argv=None) -> None:
    from cbb.model import load_games
    from db import DB_PATH
    argv = sys.argv[1:] if argv is None else argv
    games = load_games(sqlite3.connect(DB_PATH))
    results = backtest(games, 2010, int(games["season"].max()))
    s = summary(results)
    bins = pd.cut(results["p"], CALIBRATION_BINS)
    s["calibration"] = [{"range": f"{b.left * 100:.0f}–{b.right * 100:.0f}%", "n": len(part),
                         "predicted": float(part["p"].mean()), "actual": float(part["reached"].mean())}
                        for b, part in results.groupby(bins, observed=True)]
    champions = results[(results["round"] == "Champion")]
    s["seasons"] = []
    for season, part in champions.groupby("season"):
        winner = part[part["reached"] == 1]
        favorite = part.sort_values("p", ascending=False).iloc[0]
        s["seasons"].append({"season": int(season), "champion": winner["team"].iloc[0] if len(winner) else None,
                             "champion_chance": float(winner["p"].iloc[0]) if len(winner) else None,
                             "favorite": favorite["team"], "favorite_chance": float(favorite["p"])})
    if "--record" in argv:
        RECORD_PATH.parent.mkdir(exist_ok=True)
        RECORD_PATH.write_text(json.dumps(s, indent=1))
    print(f"{s['tournaments']} tournaments: log loss {s['loss']:.4f} for 'reaches round' chances "
          f"(knowing only how many teams reach each round: {s['base_loss']:.4f}). The champion was the model's "
          f"title favorite {s['champion_was_favorite']} times and in its top 4 {s['champion_top4']} times; on average "
          f"the model gave the champion {s['champion_chance']:.0%} before the tournament.")
    for c in s["calibration"]:
        print(f"  given {c['range']}: {c['n']} chances, predicted {c['predicted']:.0%}, happened {c['actual']:.0%}")


if __name__ == "__main__":
    main()
