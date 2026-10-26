"""Men's college basketball points model: offense and defense ratings for every Division I team,
the same method as college football's (cfb/model.py), fitted by weighted least squares on recent
seasons. Schools outside Division I play only a game or two a year, so they share one rating
("Non-Division I").
"""

import numpy as np
import pandas as pd

from cfb.model import fit as _fit, predict as _predict

# Tuned on 2010-11 to 2017-18 (python -m cbb.tune_model): log loss 0.5183. Held out (2018-19 to
# 2025-26, 43,330 games): 0.5382, against 0.5426 with the first guesses and 0.5491 for a simple Elo. Like college football, ratings need little shrinkage (teams are far apart); unlike it,
# earlier seasons need no extra discount beyond the 120-day half-life.
HALF_LIFE_DAYS = 120
SEASON_WEIGHT = 1.0
SHRINKAGE = 0.25
MARGIN_SD = 11.0
TRAINING_YEARS = 3
NON_D1 = "Non-Division I"


def load_games(conn) -> pd.DataFrame:
    """Every game (played and scheduled), in date order. home_team/away_team are the real names;
    home/away are the names the model rates (schools outside Division I pooled)."""
    games = pd.read_sql("""
        SELECT g.id, g.season, g.date, g.kickoff, g.status, h.name AS home_team, a.name AS away_team,
               g.home_score, g.away_score, i.neutral_site, i.season_type, i.source_game_id, i.conference_game,
               i.home_conference, i.away_conference, i.notes
        FROM games g
        JOIN leagues l ON l.id = g.league_id AND l.code = 'CBB'
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        JOIN cbb_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.id
    """, conn)
    games["home"] = np.where(games["home_conference"].isna(), NON_D1, games["home_team"])
    games["away"] = np.where(games["away_conference"].isna(), NON_D1, games["away_team"])
    # each Monday-Sunday week, named by its Monday ('2026-11-02')
    games["week"] = pd.to_datetime(games["date"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    return games


def fit(train: pd.DataFrame, as_of: str, season: int | None = None, half_life_days=HALF_LIFE_DAYS,
        season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE) -> dict:
    return _fit(train, as_of, season, half_life_days, season_weight, shrinkage, qb_shrinkage=None, inexperience_starts=0)


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD) -> dict:
    return _predict(model, home, away, neutral, margin_sd)


def window(played: pd.DataFrame, cutoff: str, training_years=TRAINING_YEARS) -> pd.DataFrame:
    start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
    return played[(played["date"] >= start) & (played["date"] < cutoff)]


def backtest(games: pd.DataFrame, start_season: int, end_season: int | None = None, half_life_days=HALF_LIFE_DAYS,
             season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, margin_sd=MARGIN_SD,
             training_years=TRAINING_YEARS) -> pd.DataFrame:
    """Walk-forward, a week at a time: fit on games before the week, predict the week's games
    between two Division I teams."""
    played = games[games["status"] == "final"]
    test = played[(played["season"] >= start_season) & (played["home"] != NON_D1) & (played["away"] != NON_D1)]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=False):
        cutoff = week_games["date"].min()
        train = window(played, cutoff, training_years)
        if train.empty:
            continue
        model = fit(train, cutoff, season, half_life_days, season_weight, shrinkage)
        for g in week_games.itertuples():
            p = predict(model, g.home, g.away, bool(g.neutral_site), margin_sd)
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "home_points": p["home_points"], "away_points": p["away_points"]})
    return pd.DataFrame(rows)


def elo_benchmark(games: pd.DataFrame, k: float = 25.0, home: float = 70.0, revert: float = 0.25) -> pd.Series:
    """A simple Elo rating's pre-game home win chance for every finished game (a benchmark: there
    are no free historical betting odds). Wins move ratings by k, more for bigger margins; each new
    season pulls ratings `revert` of the way back to average."""
    ratings: dict = {}
    out = {}
    season = None
    for g in games[games["status"] == "final"].itertuples():
        if g.season != season:
            ratings = {t: 1500 + (r - 1500) * (1 - revert) for t, r in ratings.items()}
            season = g.season
        rh, ra = ratings.get(g.home, 1500.0), ratings.get(g.away, 1500.0)
        diff = rh - ra + (0 if g.neutral_site else home)
        p = 1 / (1 + 10 ** (-diff / 400))
        out[g.id] = p
        margin = g.home_score - g.away_score
        result = 1.0 if margin > 0 else 0.0
        mult = np.log(abs(margin) + 1) * 2.2 / ((diff if result else -diff) * 0.001 + 2.2)
        change = k * mult * (result - p)
        ratings[g.home], ratings[g.away] = rh + change, ra - change
    return pd.Series(out, name="elo_home")
