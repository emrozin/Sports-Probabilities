"""NBA player projections and props: points, rebounds, assists and threes.

For each player in a team's expected lineup (nba/live.py, or in the backtest the players who
really played):

1. A rate per minute for each stat: his recent games (the most recent counting most), pulled
   toward the average for his position (guard, forward, center) when he has played few minutes.
2. His expected minutes, from his recent role, with the team's minutes adding up to 240, so when
   a starter sits the others' minutes go up.
3. His share of the team's total: rate x minutes against the rest of the lineup. The team's
   expected points come from the game model; its rebounds, assists and threes from its recent
   games. So when a star sits, part of his share goes to his teammates (halfway for points,
   rebounds and assists; not for threes: a missing shooter's threes mostly aren't replaced).
4. The chance of reaching each line (e.g. 20+ points) from a negative binomial spread around the
   projection (the variance a fixed multiple of the projection), fitted on 2014-15 to 2016-17.

Tested against the simple benchmark of the player's average so far this season, projecting the
whole available roster the way a live projection does: on 2017-18 to 2025-26 (239,683 player-games)
the average miss was smaller by 2.3% for points, 0.7% rebounds, 3.1% assists and 2.4% threes, and
the chances were calibrated (40-60% hit 49% of the time, 60-75% 67%, 75-90% 81%). The five
likeliest props each day were given 93.7% and hit 91.8%.

Usage:
    python -m nba.players        # the backtest: accuracy, calibration and the likeliest props
"""

import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import nbinom

STATS = ["points", "rebounds", "assists", "threes"]
STAT_NAMES = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "threes"}
HALF_LIFE_GAMES = 15          # how fast a player's rates follow his recent games (tuned on 2008-09 to 2013-14)
TEAM_HALF_LIFE_GAMES = 20     # the same for a team's rebounds, assists and threes per game
PRIOR_MINUTES = {"points": 5, "rebounds": 5, "assists": 5, "threes": 5}   # pull toward the position average (minutes' worth)
SHARE_POWER = {"points": 0.5, "rebounds": 0.5, "assists": 0.5, "threes": 0.0}       # 1: the team total is shared out in full, 0: not at all
# The spread around a projection: negative binomial with size = DISPERSION x projection, so the
# variance is a fixed multiple of the projection (1 + 1/DISPERSION): a star's points vary less,
# relative to his average, than a bench player's. Chosen on 2014-15 to 2016-17 by how well the
# chances at the prop lines scored; a fixed size fitted worse (points 0.195 against 0.189).
DISPERSION = {"points": 0.5, "rebounds": 1.0, "assists": 4.0, "threes": 4.0}
SPREAD_SEASONS = (2014, 2016)
PROPS = {"points": [10, 15, 20, 25, 30, 35, 40], "rebounds": [5, 8, 10, 12, 15], "assists": [5, 8, 10, 12],
         "threes": [1, 2, 3, 4, 5]}
TEAM_MINUTES = 240.0
LIKELIEST_MIN_LINE = {"points": 15, "rebounds": 8, "assists": 5, "threes": 2}   # leave out the obvious (e.g. 5+ points)


def position_group(position) -> str:
    text = str(position or "").upper()
    if text in ("C", "FC", "CF"):
        return "C"
    if text in ("PG", "SG", "G", "GF"):
        return "G"
    return "F"


# ---------------------------------------------------------------- rates

def add_rates(players: pd.DataFrame, half_life: float = HALF_LIFE_GAMES) -> pd.DataFrame:
    """For every player line: before the game (pre_*) and after it (post_*), the player's
    recent-games sums of each stat and of minutes (most recent counting most), for the rates."""
    players = players.copy()
    players["group"] = players["position"].map(position_group)
    played = players[players["minutes"] > 0].sort_values(["player_id", "date", "game_id"])
    by = played.groupby("player_id", sort=False)
    cols = {}
    for col in [*STATS, "minutes"]:
        after = by[col].transform(lambda s: s.fillna(0).ewm(halflife=half_life).sum())
        cols[f"post_{col}"] = after
        cols[f"pre_{col}"] = after.groupby(played["player_id"]).shift(1).fillna(0.0) * 0.5 ** (1 / half_life)
    played = played.assign(**cols)
    keep = ["game_id", "player_id", *cols]
    return players.merge(played[keep], on=["game_id", "player_id"], how="left")


def position_rates(players: pd.DataFrame) -> dict:
    """League-wide rate per minute of each stat for each position group."""
    played = players[players["minutes"] > 0]
    totals = played.groupby("group")[[*STATS, "minutes"]].sum()
    return {g: {s: float(totals.at[g, s] / totals.at[g, "minutes"]) for s in STATS} for g in totals.index}


def team_averages(players: pd.DataFrame, half_life: float = TEAM_HALF_LIFE_GAMES) -> pd.DataFrame:
    """Each team's recent per-game totals of each stat before every game (pre_team_*) and after
    its latest one (post_team_*), by (game_id, team)."""
    team = (players[players["minutes"] > 0].groupby(["game_id", "team", "date"])[STATS].sum().reset_index()
            .sort_values(["team", "date", "game_id"]))
    by = team.groupby("team", sort=False)
    out = team[["game_id", "team"]].copy()
    for s in STATS:
        after = by[s].transform(lambda x: x.ewm(halflife=half_life).mean())
        out[f"post_team_{s}"] = after
        out[f"pre_team_{s}"] = after.groupby(team["team"]).shift(1)
    return out


# ---------------------------------------------------------------- projecting

def project(lineup: pd.DataFrame, team_totals: pd.DataFrame, prior: dict, prefix: str = "pre_",
            prior_minutes: dict = PRIOR_MINUTES, share_power: dict = SHARE_POWER) -> pd.DataFrame:
    """Projections for lineup rows (game_id, team, player_id, group, minutes = expected minutes,
    and the {prefix}stat / {prefix}minutes sums). team_totals: (game_id, team) with team_<stat>
    expected for the game (points from the game model). Adds proj_<stat> and proj_minutes."""
    df = lineup.copy()
    key = [df["game_id"], df["team"]]
    total_minutes = df.groupby(key)["minutes"].transform("sum")
    df["proj_minutes"] = df["minutes"] * TEAM_MINUTES / total_minutes.where(total_minutes > 0, 1)
    df = df.merge(team_totals, on=["game_id", "team"], how="left")
    key = [df["game_id"], df["team"]]           # again: the merge renumbers the rows
    for s in STATS:
        base = df["group"].map(lambda g: prior.get(g, prior.get("F"))[s])
        rate = ((df[f"{prefix}{s}"].fillna(0) + base * prior_minutes[s])
                / (df[f"{prefix}minutes"].fillna(0) + prior_minutes[s]))
        raw = rate * df["proj_minutes"]
        team_raw = raw.groupby(key).transform("sum")
        scale = (df[f"team_{s}"] / team_raw.where(team_raw > 0, np.nan)).fillna(1.0) ** share_power[s]
        df[f"proj_{s}"] = raw * scale
    return df


def chance_at_least(stat: str, projected, line, dispersion: dict = DISPERSION):
    """Chance of at least `line` (a whole number) with this projection: negative binomial with
    size proportional to the projection."""
    mu = np.maximum(np.asarray(projected, float), 1e-6)
    k = np.maximum(dispersion[stat] * mu, 0.3)
    return 1 - nbinom.cdf(np.asarray(line) - 1, k, k / (k + mu))


def prop_chances(proj: pd.DataFrame, dispersion: dict = DISPERSION) -> pd.DataFrame:
    """One row per player and prop line: stat, line, chance."""
    rows = []
    for s, lines in PROPS.items():
        for line in lines:
            rows.append(proj[["game_id", "team", "player_id", "name", f"proj_{s}"]]
                        .rename(columns={f"proj_{s}": "projected"})
                        .assign(stat=s, line=line, chance=chance_at_least(s, proj[f"proj_{s}"], line, dispersion)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def likeliest_props(chances: pd.DataFrame, by: str, n: int, min_chance: float = 0.5) -> pd.DataFrame:
    """The n likeliest props in each group (a game or a day), one per player (his likeliest at or
    above each stat's notable line), likeliest first."""
    notable = chances[chances["line"] >= chances["stat"].map(LIKELIEST_MIN_LINE)]
    notable = notable[notable["chance"] >= min_chance]
    best = notable.sort_values(["chance", "line"], ascending=False).drop_duplicates(["game_id", "player_id"])
    return best.groupby(by, sort=False).head(n)


# ---------------------------------------------------------------- testing

def season_average(players: pd.DataFrame) -> pd.DataFrame:
    """The benchmark: each player's per-game average of each stat so far this season, before the
    game (his games played)."""
    played = players[players["minutes"] > 0].sort_values(["player_id", "date", "game_id"])
    by = played.groupby(["player_id", "season"], sort=False)
    out = played[["game_id", "player_id"]].copy()
    for s in STATS:
        out[f"avg_{s}"] = by[s].transform(lambda x: x.fillna(0).expanding().mean().shift(1))
    return out


def backtest(players: pd.DataFrame, team_points: pd.DataFrame, start_season: int, end_season: int | None = None,
             prior_minutes: dict = PRIOR_MINUTES, share_power: dict = SHARE_POWER) -> pd.DataFrame:
    """Projections for every player who played in a game of the test seasons, made before it the
    way a live projection is, next to what happened and the season-average benchmark. team_points: (game_id, team, points) from the game model's
    backtest."""
    test = players[(players["minutes"] > 0) & (players["season"] >= start_season)]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    history = players[players["season"] < start_season]
    prior = position_rates(history if not history.empty else players)
    teams = team_averages(players)
    totals = teams[["game_id", "team"]].copy()
    for s in STATS:
        totals[f"team_{s}"] = teams[f"pre_team_{s}"]
    totals = totals.merge(team_points.rename(columns={"points": "model_points"}), on=["game_id", "team"], how="left")
    totals["team_points"] = totals["model_points"].fillna(totals["team_points"])
    # like a live projection: the whole available roster (players who sat by the coach's decision
    # too), the game's minutes filled in order of usual minutes; scored on the players who played
    from nba.model import team_lineups
    roster = team_lineups(players[players["game_id"].isin(test["game_id"])], "roster_fill")
    lineup = (roster.merge(players.drop(columns=["minutes"]), on=["game_id", "team", "player_id"], how="left")
              .merge(test[["game_id", "player_id", "minutes"]].rename(columns={"minutes": "actual_minutes"}),
                     on=["game_id", "player_id"], how="left"))
    keep = ["game_id", "team", "player_id", "name", "group", "date", "season", "minutes", "actual_minutes",
            *(f"pre_{c}" for c in [*STATS, "minutes"]), *STATS]
    proj = project(lineup[keep].rename(columns={s: f"actual_{s}" for s in STATS}),
                   totals.drop(columns=["model_points"]), prior, "pre_", prior_minutes, share_power)
    proj = proj[proj["actual_minutes"] > 0]
    out = proj.merge(season_average(players), on=["game_id", "player_id"], how="left")
    numeric = [c for c in out.columns if c.startswith(("proj_", "actual_", "avg_"))]
    out[numeric] = out[numeric].astype(float)
    return out


def accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Average miss of the projections and of the season average, on games where both exist."""
    rows = []
    for s in STATS:
        both = df.dropna(subset=[f"avg_{s}", f"proj_{s}", f"actual_{s}"])
        rows.append({"stat": s, "games": len(both),
                     "model_miss": (both[f"proj_{s}"] - both[f"actual_{s}"]).abs().mean(),
                     "season_average_miss": (both[f"avg_{s}"] - both[f"actual_{s}"]).abs().mean()})
    out = pd.DataFrame(rows)
    out["better_by"] = 1 - out["model_miss"] / out["season_average_miss"]
    return out


def fit_dispersion(df: pd.DataFrame, grid=(0.2, 0.3, 0.5, 0.8, 1, 2, 4, 8)) -> dict:
    """For each stat, the spread factor whose chances at the prop lines score best (log loss)."""
    out = {}
    for s in STATS:
        d = df.dropna(subset=[f"proj_{s}", f"actual_{s}"])
        scores = {}
        for c in grid:
            losses = []
            for line in PROPS[s]:
                p = np.clip(chance_at_least(s, d[f"proj_{s}"], line, {s: c}), 1e-6, 1 - 1e-6)
                hit = d[f"actual_{s}"] >= line
                losses.append(-np.mean(np.where(hit, np.log(p), np.log(1 - p))))
            scores[c] = float(np.mean(losses))
        out[s] = min(scores, key=scores.get)
    return out


def calibration(df: pd.DataFrame, dispersion: dict = DISPERSION) -> pd.DataFrame:
    """Predicted chance of reaching each prop line against how often it happened, in bands."""
    rows = []
    for s, lines in PROPS.items():
        d = df.dropna(subset=[f"proj_{s}", f"actual_{s}"])
        for line in lines:
            rows.append(pd.DataFrame({"stat": s, "chance": chance_at_least(s, d[f"proj_{s}"], line, dispersion),
                                      "hit": d[f"actual_{s}"] >= line}))
    all_rows = pd.concat(rows)
    bands = pd.cut(all_rows["chance"], [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0])
    return all_rows.groupby(bands, observed=True).agg(predicted=("chance", "mean"), happened=("hit", "mean"),
                                                      count=("hit", "size")).reset_index()


def main(argv: list[str] | None = None) -> None:
    from db import DB_PATH
    from nba.model import load_players
    from nba.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
    conn = sqlite3.connect(DB_PATH)
    players = add_rates(load_players(conn))
    points = team_points_from_backtest(conn, BACKTEST_NAME, BACKTEST_VERSION)
    conn.close()
    periods = (("Tuning (rates)", START_SEASON, HELD_OUT_FROM - 1), ("Spread fitting", *SPREAD_SEASONS),
               ("Held-out", SPREAD_SEASONS[1] + 1, None))
    for label, first, last in periods:
        df = backtest(players, points, first, last)
        print(f"{label} seasons:\n{accuracy(df).round(3).to_string(index=False)}")
        if label == "Spread fitting":
            print(f"  best spread factors: {fit_dispersion(df)}")
        elif label == "Held-out":
            print(calibration(df).round(3).to_string(index=False))
            chances = prop_chances(df.assign(name=df["name"])).merge(df[["game_id", "player_id", "date", *(f"actual_{s}" for s in STATS)]],
                                                                    on=["game_id", "player_id"])
            picks = likeliest_props(chances, "date", 5)
            hit = picks.apply(lambda r: r[f"actual_{r['stat']}"] >= r["line"], axis=1)
            print(f"  the 5 likeliest props each day: predicted {picks['chance'].mean():.1%}, hit {hit.mean():.1%} ({len(picks)} props)")


def team_points_from_backtest(conn, name: str, version: str) -> pd.DataFrame:
    """Each team's expected points in each game from the saved game-model backtest predictions."""
    df = pd.read_sql("""SELECT pg.game_id, h.name AS home, a.name AS away, pg.exp_home, pg.exp_away
                        FROM prediction_goals pg JOIN games g ON g.id = pg.game_id
                        JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                        WHERE pg.model_name = ? AND pg.model_version = ?""", conn, params=(name, version))
    return pd.concat([df[["game_id", "home", "exp_home"]].set_axis(["game_id", "team", "points"], axis=1),
                      df[["game_id", "away", "exp_away"]].set_axis(["game_id", "team", "points"], axis=1)], ignore_index=True)


if __name__ == "__main__":
    main(sys.argv[1:])
