"""Missing starters: for every NFL game (played or to come), how much of each team's regular lineup
is ruled out, apart from the quarterback (who has his own rating).

A team's regulars for a game are the players who played at least REGULAR of its offensive or
defensive snaps in its previous game that season. One is missing when that week's injury report
lists him Out or Doubtful, or the weekly roster has him off the active roster (injured reserve,
released, traded...). The measure is the sum of the missing regulars' snap shares, separately
for offense and defense: two full-time starters out is about 2.

Why only regulars from the previous game: a player who has been out for weeks is already reflected
in the team's recent results, and so in its ratings; the news is the starter who just got hurt.
Game-day inactive lists aren't used: they come out 90 minutes before kickoff, after the forecast.

The effect on the expected margin (nfl/model.py, INJURY_OFFENSE and INJURY_DEFENSE) was fitted on
2013-2017 and checked once on 2018-2025 (python -m nfl.starters): see the README.
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from nfl.import_nfl_stats import name_key
from nfl.quarterbacks import OFF_ROSTER

REGULAR = 0.6
OUT = {"Out", "Doubtful"}
FIRST_SEASON = 2013     # snap counts start in 2013


def missing_starters(conn, since: int = FIRST_SEASON) -> pd.DataFrame:
    """game_id, team, offense, defense (the snap shares of the regulars ruled out), names; for games
    from season `since` on (the website only needs this season's)."""
    columns = ["game_id", "team", "offense", "defense", "names"]
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"nfl_snap_counts", "nfl_injuries", "nfl_rosters", "nfl_game_info"} <= have:
        return pd.DataFrame(columns=columns)
    snap_cols = {r[1] for r in conn.execute("PRAGMA table_info(nfl_snap_counts)")}
    if "defense_pct" not in snap_cols:
        return pd.DataFrame(columns=columns)
    games = pd.read_sql("""SELECT g.id AS game_id, g.season, i.week, h.name AS home, a.name AS away
                           FROM games g JOIN nfl_game_info i ON i.game_id = g.id
                           JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                           WHERE g.season >= ?""", conn, params=(max(since, FIRST_SEASON),))
    if games.empty:
        return pd.DataFrame(columns=columns)
    sides = pd.concat([games[["game_id", "season", "week", "home"]].rename(columns={"home": "team"}),
                       games[["game_id", "season", "week", "away"]].rename(columns={"away": "team"})])
    sides = sides.sort_values(["team", "season", "week"])
    sides["previous"] = sides.groupby(["team", "season"])["game_id"].shift(1)

    snaps = pd.read_sql("""SELECT s.game_id, s.team, s.name, s.position, s.offense_pct, s.defense_pct FROM nfl_snap_counts s
                           JOIN games g ON g.id = s.game_id WHERE g.season >= ?""", conn, params=(max(since, FIRST_SEASON),))
    snaps["key"] = snaps["name"].map(name_key)
    snaps["off"] = np.where((snaps["offense_pct"].fillna(0) >= REGULAR) & (snaps["position"] != "QB"), snaps["offense_pct"], 0.0)
    snaps["def"] = np.where(snaps["defense_pct"].fillna(0) >= REGULAR, snaps["defense_pct"], 0.0)
    regulars = snaps[(snaps["off"] > 0) | (snaps["def"] > 0)]

    out = pd.read_sql(f"""SELECT season, week, team, name FROM nfl_injuries WHERE season >= ? AND status IN ({','.join('?' * len(OUT))})
                          UNION SELECT season, week, team, name FROM nfl_rosters WHERE season >= ?
                          AND status IN ({','.join('?' * len(OFF_ROSTER))})""",
                      conn, params=[since, *OUT, since, *OFF_ROSTER])
    out["key"] = out["name"].map(name_key)
    out = out.drop_duplicates(["season", "week", "team", "key"])

    merged = (sides.dropna(subset=["previous"]).astype({"previous": int})
              .merge(regulars.rename(columns={"game_id": "previous"}), on=["previous", "team"])
              .merge(out[["season", "week", "team", "key"]], on=["season", "week", "team", "key"]))
    if merged.empty:
        return pd.DataFrame(columns=columns)
    result = merged.groupby(["game_id", "team"]).agg(offense=("off", "sum"), defense=("def", "sum"),
                                                     names=("name", lambda n: ", ".join(sorted(n)))).reset_index()
    return result[columns]


def add_to_games(games: pd.DataFrame, missing: pd.DataFrame, id_column: str = "id") -> pd.DataFrame:
    """Adds home_missing_off, home_missing_def, away_missing_off, away_missing_def (0 when none)."""
    games = games.copy()
    for side in ("home", "away"):
        part = missing.rename(columns={"game_id": id_column, "team": side, "offense": f"{side}_missing_off",
                                       "defense": f"{side}_missing_def", "names": f"{side}_missing_names"})
        games = games.merge(part, on=[id_column, side], how="left")
        games[[f"{side}_missing_off", f"{side}_missing_def"]] = games[[f"{side}_missing_off", f"{side}_missing_def"]].fillna(0.0)
    return games


def main(argv=None) -> None:
    """Fit the effect of missing starters on the backtest's errors (2013-2017), check it on 2018-2025."""
    from scipy.stats import norm
    from db import DB_PATH
    from nfl.model import MARGIN_SD
    from nfl.run_backtest import HELD_OUT_FROM
    conn = sqlite3.connect(DB_PATH)
    g = pd.read_sql("""SELECT g.id, g.season, h.name AS home, a.name AS away, g.home_score, g.away_score,
                              pg.exp_home, pg.exp_away
                       FROM games g JOIN nfl_game_info i ON i.game_id = g.id AND i.game_type = 'REG'
                       JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                       JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = 'nfl_points_qb_backtest'
                                               AND pg.model_version = 'v2'   -- the version without missing starters
                       WHERE g.status = 'final' AND g.season >= ?""", conn, params=(FIRST_SEASON,))
    g = add_to_games(g, missing_starters(conn))
    g["error"] = (g["home_score"] - g["away_score"]) - (g["exp_home"] - g["exp_away"])
    x = np.column_stack([g["home_missing_off"] - g["away_missing_off"], g["home_missing_def"] - g["away_missing_def"]])
    tune = (g["season"] < HELD_OUT_FROM).to_numpy()
    beta, *_ = np.linalg.lstsq(x[tune], g.loc[tune, "error"], rcond=None)
    print(f"Fitted on {g.loc[tune, 'season'].min()}-{HELD_OUT_FROM - 1}: {beta[0]:+.2f} points per offensive starter's share, "
          f"{beta[1]:+.2f} per defensive")
    held = g[~tune]
    decided = (held["home_score"] != held["away_score"]).to_numpy()
    won = (held["home_score"] > held["away_score"]).to_numpy()[decided]

    def loss(margin):
        p = np.clip(norm.cdf(np.asarray(margin)[decided] / MARGIN_SD), 1e-6, 1 - 1e-6)
        return float(-np.mean(np.where(won, np.log(p), np.log(1 - p))))
    base = (held["exp_home"] - held["exp_away"]).to_numpy()
    print(f"Held out {HELD_OUT_FROM}-{held['season'].max()} ({decided.sum()} games): log loss {loss(base):.4f} without, "
          f"{loss(base + x[~tune] @ beta):.4f} with missing starters")


if __name__ == "__main__":
    main()
