"""An Elo rating for every club, blended into the league predictions.

The goals model (ucl/model.py) is built for the cups: it rates every club in Europe from its league
and European games together. In the big leagues' own games it does about as well as a plain Elo
rating of every club (K = 20, scaled up by the goal difference, the way clubelo.com rates clubs),
and the two make different mistakes: Elo follows a club's form within a season faster. So a league
game's chances are a blend: LEAGUE_BLEND of Elo's chances and the rest the goals model's. A game's
Elo chances come from the rating difference through an ordered logistic model fitted on the
tuning seasons' league games.

Tested on the walk-forward backtest's league games (La Liga, Bundesliga, Serie A and Ligue 1),
log loss:

    Elo's share          0.0     0.2     0.4     0.5     0.6     1.0
    2016-17 to 2020-21   0.9874  0.9850  0.9834  0.9830  0.9827  0.9838   (tuning)
    2021-22 to 2025-26   0.9908  0.9885  0.9871  0.9868  0.9866  0.9883   (held out)

0.6 was best on the tuning seasons and is best on the held-out ones too, and stayed best once the
leagues' model learned from xG (ucl/tune_xg.py). The cups keep the goals
model alone (Elo was clearly worse there: 0.933 against 0.910 in the Champions League), and the
season simulations and scoreline chances use the goals model, since Elo has no goals.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

K = 20.0
HOME = 65.0                   # home advantage in rating points
TUNING = (2016, 2020)
LEAGUES = ("esp.1", "ger.1", "ita.1", "fra.1")
LEAGUE_BLEND = 0.6            # Elo's share of a league game's chances


def run(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every finished game's pre-game rating difference (home minus away, home advantage
    included), and every club's rating after the last game."""
    played = games[(games["status"] == "final") & games["home_score"].notna()].sort_values(["date", "kickoff"])
    rating: dict = {}
    out = []
    for g in played.itertuples():
        rh, ra = rating.get(g.home_id, 1500.0), rating.get(g.away_id, 1500.0)
        diff = rh - ra + (0 if g.neutral == 1 else HOME)
        out.append((g.event_id, diff))
        expected = 1 / (1 + 10 ** (-diff / 400))
        actual = 1.0 if g.home_score > g.away_score else 0.0 if g.home_score < g.away_score else 0.5
        margin = abs(g.home_score - g.away_score)
        change = K * (1 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8) * (actual - expected)
        rating[g.home_id], rating[g.away_id] = rh + change, ra - change
    return pd.DataFrame(out, columns=["event_id", "elo_diff"]), rating


def ordered_logit(diff: np.ndarray, outcome: np.ndarray) -> tuple[float, float, float]:
    """Fit P(away) = s(c1 - b d), P(not home) = s(c2 - b d) on (diff, outcome 0 away / 1 draw / 2 home)."""
    s = lambda z: 1 / (1 + np.exp(-z))

    def nll(theta):
        b, c1, gap = theta
        c2 = c1 + np.exp(gap)
        p_away, p_not_home = s(c1 - b * diff), s(c2 - b * diff)
        p = np.choose(outcome, [p_away, p_not_home - p_away, 1 - p_not_home])
        return -np.sum(np.log(np.clip(p, 1e-9, 1)))
    theta = minimize(nll, [0.005, -0.5, 0.0], method="Nelder-Mead", options={"maxiter": 4000}).x
    return theta[0], theta[1], theta[1] + np.exp(theta[2])


def coefficients(games: pd.DataFrame, diffs: pd.DataFrame, competitions=LEAGUES, tuning=TUNING) -> tuple:
    """The ordered logistic model's coefficients, fitted on the tuning seasons' games of these
    competitions (all of them if there are none in the tuning seasons, as in small test data)."""
    from ucl.model import result_of
    part = games[games["competition"].isin(competitions) & (games["status"] == "final")]
    d = diffs.merge(part[["event_id", "season", "home_score", "away_score", "extra_time"]], on="event_id")
    tuned = d[d["season"].between(*tuning)]
    d = tuned if len(tuned) >= 100 else d
    if len(d) < 30:
        return None
    outcome = np.array([{"away": 0, "draw": 1, "home": 2}[result_of(h, a, e)]
                        for h, a, e in zip(d["home_score"], d["away_score"], d["extra_time"].fillna(0))])
    return ordered_logit(d["elo_diff"].to_numpy(float), outcome)


def chances(diff, coef) -> dict:
    """Home, draw and away chances from a rating difference."""
    b, c1, c2 = coef
    s = lambda z: 1 / (1 + np.exp(-z))
    p_away, p_not_home = s(c1 - b * np.asarray(diff, float)), s(c2 - b * np.asarray(diff, float))
    return {"home": 1 - p_not_home, "draw": p_not_home - p_away, "away": p_away}


def blend(p: dict, q: dict, share: float = LEAGUE_BLEND) -> dict:
    """The goals model's chances p with Elo's q mixed in (the expected goals and grid stay the model's)."""
    return {**p, **{o: (1 - share) * p[o] + share * q[o] for o in ("home", "draw", "away")}}


def attach(model: dict, games: pd.DataFrame) -> dict:
    """Add today's Elo ratings to a fitted model, so its league predictions blend them in."""
    if model is None or games.empty:
        return model
    diffs, ratings = run(games)
    coef = coefficients(games, diffs)
    if coef is not None:
        model["elo"] = {"ratings": ratings, "coef": coef}
    return model


def league_chances(model: dict, home: str, away: str, neutral: bool) -> dict | None:
    elo = model.get("elo")
    if not elo:
        return None
    r = elo["ratings"]
    return chances(r.get(home, 1500.0) - r.get(away, 1500.0) + (0 if neutral else HOME), elo["coef"])
