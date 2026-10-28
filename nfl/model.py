"""Points model for the NFL: offense and defense ratings, fitted like the Premier League model.

Each team has an offense rating (points it scores above average) and a defense rating
(points it stops). A team's expected points in a game:

    home points = base + home advantage / 2 + home offense - away defense
    away points = base - home advantage / 2 + away offense - home defense

so the expected home margin is  home advantage + (home offense + home defense) - (away offense + away defense).

Ratings are fitted by weighted least squares on recent games (each game's weight halves every
`half_life_days`, so last season counts less than this one), with shrinkage pulling ratings
toward average so a few games can't produce extreme ratings. Home advantage is fitted from
the same recent games, so it follows the league's shrinking home edge on its own.

The actual margin lands around the expected one with a typical spread of `margin_sd` points,
so the home win probability is P(margin > 0) from a normal distribution. Unlike Elo, this also
gives an expected score for each team, so its own spread and total.

Quarterbacks: every starting quarterback has his own rating, in points per game compared with
an average starter, fitted alongside the team ratings (so a team's offense rating is what it
does apart from its quarterback). Ratings are pulled toward average, more strongly for
quarterbacks with few games, and an inexperience term lowers the expected points of a
quarterback with few career starts (fading to nothing by 32 starts). So a backup, a rookie or a
starter returning from injury each gets an appropriate value, learned only from earlier games.

Version 1 used a simpler yes/no flag instead (is this the team's usual starter?); quarterback
ratings replaced it after scoring better on both the tuning and the held-out seasons.
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm

# Tuned on 2010-2017 (nfl/tune_model.py)
HALF_LIFE_DAYS = 150  # a game's weight halves every 150 days, so last season counts about half as much
SHRINKAGE = 5.0       # pull team ratings toward average (L2 regularization)
QB_SHRINKAGE = 10.0   # pull quarterback ratings toward average (stronger: a QB has fewer games than a team)
INEXPERIENCE_STARTS = 32  # the inexperience term fades to nothing by this many career starts
MARGIN_SD = 12.0      # typical gap between the actual and expected margin, in points
# Missing starters (nfl/starters.py): points of margin per regular starter ruled out (the sum of
# their snap shares), fitted on 2013-2017 by python -m nfl.starters; held-out log loss 0.6298 -> 0.6290.
INJURY_OFFENSE = -0.78
INJURY_DEFENSE = -0.91
TRAINING_YEARS = 4

USUAL_STARTER_GAMES = 8


def add_qb_changes(games: pd.DataFrame) -> pd.DataFrame:
    """Add quarterback columns for each side (home_/away_), all looking only backward:

        qb_change   1 if the listed starter isn't the team's usual starter (the most frequent starter
                    in its previous 8 games); 0 if he is, or if no starter is listed yet
        qb_expected the listed starter, or the usual starter when none is listed (upcoming games)
        qb_starts   the expected starter's career starts before this game (since 1999)
    """
    games = games.copy()
    history: dict[str, list] = {}
    starts: Counter = Counter()
    out = {f"{side}_{col}": {} for side in ("home", "away") for col in ("qb_change", "qb_expected", "qb_starts")}
    for g in games.sort_values(["date", "id"]).itertuples():
        for side in ("home", "away"):
            team, qb = getattr(g, side), getattr(g, f"{side}_qb")
            qb = qb if isinstance(qb, str) and qb else None   # missing can arrive as None or NaN
            recent = [q for q in history.get(team, [])[-USUAL_STARTER_GAMES:] if q]
            usual = Counter(recent).most_common(1)[0][0] if recent else None
            expected = qb or usual
            out[f"{side}_qb_change"][g.id] = int(qb is not None and usual is not None and qb != usual)
            out[f"{side}_qb_expected"][g.id] = expected
            out[f"{side}_qb_starts"][g.id] = starts[expected] if expected else np.nan
        if g.status == "final":
            for side in ("home", "away"):
                qb = getattr(g, f"{side}_qb")
                qb = qb if isinstance(qb, str) and qb else None
                history.setdefault(getattr(g, side), []).append(qb)
                if qb:
                    starts[qb] += 1
    for column, values in out.items():
        games[column] = games["id"].map(values)
    return games


def inexperience(starts) -> np.ndarray:
    """1 for a quarterback with no career starts, falling to 0 at INEXPERIENCE_STARTS; 0 if unknown."""
    starts = np.asarray(starts, dtype=float)
    return np.where(np.isnan(starts), 0.0, np.clip(1 - starts / INEXPERIENCE_STARTS, 0, 1))


def fit(train: pd.DataFrame, as_of, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE,
        qb_shrinkage=QB_SHRINKAGE) -> dict:
    """Fit team offense/defense ratings, quarterback ratings, base points, home advantage and the
    inexperience effect from finished games before `as_of` (`train` needs add_qb_changes columns)."""
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {t: i for i, t in enumerate(teams)}
    qbs = sorted(set(train["home_qb"].dropna()) | set(train["away_qb"].dropna()))
    qb_index = {q: i for i, q in enumerate(qbs)}
    n, games, nq = len(teams), len(train), len(qbs)

    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    w = np.tile(0.5 ** (days_ago / half_life_days), 2)

    # Two rows per game: the home team's points and the away team's points.
    # Columns: offense (n), defense (n), base, home advantage, inexperience, quarterbacks (nq).
    x = np.zeros((2 * games, 2 * n + 3 + nq))
    h = train["home"].map(index).to_numpy()
    a = train["away"].map(index).to_numpy()
    rows = np.arange(games)
    home_side = np.where(train["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    x[rows, h] = 1                 # home offense
    x[rows, n + a] = -1            # away defense
    x[rows, 2 * n] = 1
    x[rows, 2 * n + 1] = home_side
    x[games + rows, a] = 1         # away offense
    x[games + rows, n + h] = -1    # home defense
    x[games + rows, 2 * n] = 1
    x[games + rows, 2 * n + 1] = -home_side
    x[rows, 2 * n + 2] = inexperience(train["home_qb_starts"])
    x[games + rows, 2 * n + 2] = inexperience(train["away_qb_starts"])
    for side, side_rows in (("home", rows), ("away", games + rows)):
        qb_col = train[f"{side}_qb"].map(qb_index).to_numpy(float)
        known = ~np.isnan(qb_col)
        x[side_rows[known], 2 * n + 3 + qb_col[known].astype(int)] = 1
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])

    penalty = np.zeros(2 * n + 3 + nq)
    penalty[:2 * n] = shrinkage        # shrink team ratings toward 0 (average), not base or home advantage
    penalty[2 * n + 2] = 1e-6          # keeps the inexperience effect solvable when no game needs it
    penalty[2 * n + 3:] = qb_shrinkage  # shrink quarterback ratings toward 0 (an average starter)
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x + np.diag(penalty), xtw @ y)
    offense, defense = beta[:n] - beta[:n].mean(), beta[n:2 * n] - beta[n:2 * n].mean()
    qb_ratings = pd.Series(beta[2 * n + 3:], index=qbs, dtype=float)
    return {"ratings": pd.DataFrame({"offense": offense, "defense": defense}, index=teams),
            "base": beta[2 * n] + beta[:n].mean() - beta[n:2 * n].mean(),
            "home_adv": beta[2 * n + 1],
            "inexperience": beta[2 * n + 2],   # points for a quarterback with no career starts (negative)
            "qb_ratings": qb_ratings}          # points per game vs an average starter


def qb_value(model: dict, qb, starts) -> float:
    """A quarterback's effect on his team's points: his rating (0 if unknown) plus the inexperience term."""
    rating = model["qb_ratings"].get(qb, 0.0) if isinstance(qb, str) else 0.0
    return float(rating + model["inexperience"] * inexperience([starts])[0])


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD,
            home_qb=None, away_qb=None, home_qb_starts=np.nan, away_qb_starts=np.nan,
            home_missing=(0.0, 0.0), away_missing=(0.0, 0.0)) -> dict:
    """Expected points for each side, the expected home margin and total, and the home win probability.
    home_qb / away_qb: the expected starters (None: no quarterback adjustment). home_missing /
    away_missing: (offense, defense) snap shares of regular starters ruled out (nfl/starters.py)."""
    r = model["ratings"]
    edge = 0.0 if neutral else model["home_adv"] / 2
    home_pts = (model["base"] + edge + r.at[home, "offense"] - r.at[away, "defense"]
                + qb_value(model, home_qb, home_qb_starts))
    away_pts = (model["base"] - edge + r.at[away, "offense"] - r.at[home, "defense"]
                + qb_value(model, away_qb, away_qb_starts))
    injuries = injury_effect(home_missing, away_missing)       # split evenly between the two scores
    home_pts, away_pts = home_pts + injuries / 2, away_pts - injuries / 2
    margin = home_pts - away_pts
    return {"home_points": home_pts, "away_points": away_pts, "margin": margin, "total": home_pts + away_pts,
            "home_win": float(norm.cdf(margin / margin_sd))}


def injury_effect(home_missing=(0.0, 0.0), away_missing=(0.0, 0.0)) -> float:
    """Points of home margin from missing starters: (offense, defense) shares for each side."""
    return float(INJURY_OFFENSE * (home_missing[0] - away_missing[0]) + INJURY_DEFENSE * (home_missing[1] - away_missing[1]))


def missing(g, side: str) -> tuple[float, float]:
    """A game row's (offense, defense) missing starters for one side (0 if not added)."""
    return (float(getattr(g, f"{side}_missing_off", 0.0) or 0.0), float(getattr(g, f"{side}_missing_def", 0.0) or 0.0))


def backtest(games: pd.DataFrame, start_season: int, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE,
             margin_sd=MARGIN_SD, training_years=TRAINING_YEARS, qb_shrinkage=QB_SHRINKAGE) -> pd.DataFrame:
    """Walk-forward: for each week of each season, fit on earlier games only, then predict that week.
    `games` needs the QB change columns (add_qb_changes)."""
    played = games[games["status"] == "final"]
    test = played[played["season"] >= start_season]
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=True):
        cutoff = week_games["date"].min()
        start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
        train = played[(played["date"] >= start) & (played["date"] < cutoff)]
        if train.empty:
            continue
        model = fit(train, cutoff, half_life_days, shrinkage, qb_shrinkage)
        known = set(model["ratings"].index)
        for g in week_games.itertuples():
            if g.home not in known or g.away not in known:
                continue
            p = predict(model, g.home, g.away, bool(g.neutral_site), margin_sd,
                        g.home_qb_expected, g.away_qb_expected, g.home_qb_starts, g.away_qb_starts,
                        missing(g, "home"), missing(g, "away"))
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "model_total": p["total"], "home_points": p["home_points"],
                         "away_points": p["away_points"]})
    return pd.DataFrame(rows)
