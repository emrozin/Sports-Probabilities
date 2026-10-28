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
# Preseason ratings (version 3; python -m cbb.preseason_test): until 15 December, last season's
# ratings pulled 35% of the way toward each team's conference average count as 8 games, and the
# season's own games are added to them. Early-season log loss on the held-out seasons 0.4901 -> 0.4858.
PRESEASON = {"carry": 0.65, "prior_weight": 8, "switch": "12-15"}


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
        season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, prior: pd.DataFrame | None = None, prior_weight: float = 0.0) -> dict:
    return _fit(train, as_of, season, half_life_days, season_weight, shrinkage, qb_shrinkage=None, inexperience_starts=0,
                prior=prior, prior_weight=prior_weight)


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD) -> dict:
    return _predict(model, home, away, neutral, margin_sd)


def conference_of(games: pd.DataFrame, season: int) -> pd.Series:
    """Each Division I team's conference in a season (from its schedule, known in advance)."""
    part = games[games["season"] == season]
    sides = pd.concat([part[["home", "home_conference"]].set_axis(["team", "conf"], axis=1),
                       part[["away", "away_conference"]].set_axis(["team", "conf"], axis=1)]).dropna()
    return sides.drop_duplicates("team", keep="last").set_index("team")["conf"]


def season_prior(games: pd.DataFrame, season: int, carry: float, half_life_days=HALF_LIFE_DAYS,
                 season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, training_years=TRAINING_YEARS) -> tuple[pd.DataFrame, dict]:
    """Preseason ratings for a season: the ratings at its start (fitted on earlier games), with each
    team pulled (1 - carry) of the way toward its new conference's average. Rosters turn over a lot
    between seasons, so last season's rating is only partly this season's. (Scaling the pull by
    how much of last season's minutes return, from player box scores, didn't help on 2010-2017.)
    Returns (prior, the fit)."""
    played = games[games["status"] == "final"]
    start = games.loc[games["season"] == season, "date"].min()
    train = window(played, start, training_years)
    model = fit(train, start, season, half_life_days, season_weight, shrinkage)
    ratings = model["ratings"].copy()
    conf = ratings.index.map(conference_of(games, season))
    for col in ("offense", "defense"):
        mean = ratings[col].groupby(conf).transform("mean")
        ratings[col] = np.where(pd.isna(conf), ratings[col], carry * ratings[col] + (1 - carry) * mean)
    return ratings, model


def window(played: pd.DataFrame, cutoff: str, training_years=TRAINING_YEARS) -> pd.DataFrame:
    start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
    return played[(played["date"] >= start) & (played["date"] < cutoff)]


def fit_season(games: pd.DataFrame, played: pd.DataFrame, cutoff: str, season: int, priors: dict,
               carry: float, prior_weight: float, half_life_days=HALF_LIFE_DAYS, season_weight=SEASON_WEIGHT,
               shrinkage=SHRINKAGE, training_years=TRAINING_YEARS) -> dict:
    """The preseason-prior version: this season's games only, ratings pulled toward the preseason
    ratings (worth prior_weight games). priors: a cache, {season: (prior, fit)}."""
    if season not in priors:
        priors[season] = season_prior(games, season, carry, half_life_days, season_weight, shrinkage, training_years)
    prior, start_fit = priors[season]
    current = played[(played["season"] == season) & (played["date"] < cutoff)]
    if current.empty:
        return {**start_fit, "ratings": prior}
    model = fit(current, cutoff, season, half_life_days, season_weight, shrinkage, prior=prior, prior_weight=prior_weight)
    missing = prior.index.difference(model["ratings"].index)          # teams that haven't played yet
    model["ratings"] = pd.concat([model["ratings"], prior.loc[missing]])
    return model


def early_season(as_of: str, switch: str = PRESEASON["switch"]) -> bool:
    """Before the switch date (and after the summer): preseason ratings still lead."""
    return "07-01" <= as_of[5:] < switch


def current_fit(games: pd.DataFrame, as_of: str, season: int, priors: dict | None = None) -> dict:
    """The model as live predictions use it on a date: preseason ratings early in the season, the
    usual fit after. games: every game (played and scheduled)."""
    played = games[games["status"] == "final"]
    if early_season(as_of):
        return fit_season(games, played, as_of, season, priors if priors is not None else {},
                          PRESEASON["carry"], PRESEASON["prior_weight"])
    return fit(window(played, as_of), as_of, season)


def backtest(games: pd.DataFrame, start_season: int, end_season: int | None = None, half_life_days=HALF_LIFE_DAYS,
             season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, margin_sd=MARGIN_SD,
             training_years=TRAINING_YEARS, carry: float | None = None, prior_weight: float = 0.0,
             switch: str | None = None) -> pd.DataFrame:
    """Walk-forward, a week at a time: fit on games before the week, predict the week's games
    between two Division I teams. carry None: the original method (every game of the last three
    seasons, by recency); otherwise preseason ratings (season_prior) plus this season's games, until
    `switch` ('MM-DD'; None: all season), then the original method."""
    priors: dict = {}
    played = games[games["status"] == "final"]
    test = played[(played["season"] >= start_season) & (played["home"] != NON_D1) & (played["away"] != NON_D1)]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=False):
        cutoff = week_games["date"].min()
        if carry is not None and (switch is None or cutoff[5:] >= "07-01" and cutoff[5:] < switch):
            model = fit_season(games, played, cutoff, season, priors, carry, prior_weight, half_life_days,
                               season_weight, shrinkage, training_years)
        else:
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
