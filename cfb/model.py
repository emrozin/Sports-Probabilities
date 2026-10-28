"""College football points model: offense and defense ratings for every team, plus a rating for
every starting quarterback (since 2014) and a penalty for new starters, like the NFL's (nfl/model.py).
College depth charts aren't published reliably, so a forecast expects each team's starter from its
previous game.

Each game gives two equations: home points = base + home edge + home offense - away defense,
and the same for the away team. Ratings are fitted by weighted least squares on recent seasons:
recent games count more (HALF_LIFE_DAYS), and games from earlier seasons count less again
(SEASON_WEIGHT per season back), because college rosters turn over far more than NFL ones.
Ratings are pulled toward average (SHRINKAGE), which matters most for teams with few games
against the rest, like second-division (FCS) schools. The expected margin becomes a win
probability through a normal distribution (MARGIN_SD).
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

# Tuned on 2010-2017 (python -m cfb.tune_model): log loss 0.536 with first guesses (half-life 200,
# season weight 0.6, shrinkage 4, 3 years) down to 0.479. Much weaker shrinkage than the NFL's
# mattered most: college teams really are far apart, and with old games down-weighted, strong
# shrinkage pulled every rating toward average.
HALF_LIFE_DAYS = 400
SEASON_WEIGHT = 0.4       # each earlier season's games count this much less again
SHRINKAGE = 0.05
MARGIN_SD = 16.0
TRAINING_YEARS = 4
# Quarterbacks (version 2), tuned on 2015-2017 (the tuning seasons with quarterback data): log loss
# 0.4811 without them, 0.4800 with ratings, 0.4796 adding the new-starter penalty. Held-out 2018-2025:
# 0.4887 -> 0.4877, better in 6 of 8 seasons. A small gain: with team ratings this flexible, most of a
# quarterback's effect is already in his team's rating; it shows when a new or backup starter plays.
QB_SHRINKAGE = 10.0       # pull quarterback ratings toward average (None = no quarterback ratings)
INEXPERIENCE_STARTS = 12  # a quarterback's new-starter penalty fades away over his first 12 starts
# Preseason ratings (version 3; python -m cfb.preseason_test): until 1 November, last season's
# ratings pulled 20% of the way toward each team's conference average count as 3 games, and the
# season's own games are added to them. Tuned on 2010-2017 (early-season log loss 0.4165 -> 0.4139);
# held out (2018-2025): August-September 0.3957 -> 0.3940, whole season 0.4877 -> 0.4861.
PRESEASON = {"carry": 0.8, "prior_weight": 3, "switch": "11-01"}


def load_games(conn) -> pd.DataFrame:
    """Every college game (played and scheduled), in date order, with each side's starting
    quarterback where known (2014 on; ESPN player ids, so a transfer keeps his identity)."""
    has_starters = conn.execute("SELECT name FROM sqlite_master WHERE name = 'cfb_starters'").fetchone() is not None
    qb = ("hs.qb_id AS home_qb, hs.qb_name AS home_qb_name, aw.qb_id AS away_qb, aw.qb_name AS away_qb_name"
          if has_starters else "NULL AS home_qb, NULL AS home_qb_name, NULL AS away_qb, NULL AS away_qb_name")
    joins = ("LEFT JOIN cfb_starters hs ON hs.game_id = g.id AND hs.side = 'home' "
             "LEFT JOIN cfb_starters aw ON aw.game_id = g.id AND aw.side = 'away'") if has_starters else ""
    games = pd.read_sql(f"""
        SELECT g.id, g.season, g.date, g.kickoff, g.status, h.name AS home, a.name AS away,
               g.home_score, g.away_score, i.neutral_site, i.week, i.season_type, i.source_game_id,
               i.home_division, i.away_division, i.home_conference, i.away_conference,
               i.home_elo, i.away_elo, i.spread_line, i.total_line, i.notes, i.venue, i.conference_game, {qb}
        FROM games g
        JOIN leagues l ON l.id = g.league_id AND l.code = 'CFB'
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        JOIN cfb_game_info i ON i.game_id = g.id
        {joins}
        ORDER BY g.date, g.id
    """, conn)
    return add_expected_starters(games)


def add_expected_starters(games: pd.DataFrame) -> pd.DataFrame:
    """home_qb_expected / away_qb_expected: who a forecast made before the game would expect to start,
    i.e. the team's starter in its previous game (college depth charts aren't published reliably).
    Unknown (None) for a team's first game with quarterback data."""
    games = games.copy()
    last: dict[str, str] = {}
    starts: dict[str, int] = {}
    out = {f"{side}_{k}": {} for side in ("home", "away") for k in ("qb_expected", "qb_starts", "qb_expected_starts")}
    for g in games.itertuples():
        for side in ("home", "away"):
            qb, expected = getattr(g, f"{side}_qb"), last.get(getattr(g, side))
            out[f"{side}_qb_expected"][g.id] = expected
            out[f"{side}_qb_starts"][g.id] = starts.get(qb, 0) if isinstance(qb, str) else np.nan
            out[f"{side}_qb_expected_starts"][g.id] = starts.get(expected, 0) if expected else np.nan
        if g.status == "final":
            for side in ("home", "away"):
                qb = getattr(g, f"{side}_qb")
                if isinstance(qb, str) and qb:
                    last[getattr(g, side)] = qb
                    starts[qb] = starts.get(qb, 0) + 1
    for column, values in out.items():
        games[column] = games["id"].map(values)
    return games


def inexperience(starts, full: float) -> np.ndarray:
    """1 for a quarterback with no starts (since 2014), falling to 0 at `full` starts; 0 if unknown."""
    starts = np.asarray(starts, dtype=float)
    return np.where(np.isnan(starts), 0.0, np.clip(1 - starts / full, 0, 1))


def load_market(conn) -> pd.DataFrame:
    """The consensus closing moneyline (margin removed) for finished games: market_home."""
    return pd.read_sql("""SELECT game_id AS id, 1 / price AS market_home FROM odds
                          WHERE bookmaker = 'consensus' AND market = 'moneyline' AND outcome = 'home'""", conn)


def fit(train: pd.DataFrame, as_of: str, season: int | None = None, half_life_days=HALF_LIFE_DAYS,
        season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, qb_shrinkage=QB_SHRINKAGE,
        inexperience_starts=INEXPERIENCE_STARTS, prior: pd.DataFrame | None = None, prior_weight: float = 0.0,
        qb_prior: pd.Series | None = None, fixed_inexperience: float | None = None) -> dict:
    """Offense/defense ratings, quarterback ratings, base points and home advantage from finished
    games before as_of. season: the season being predicted (earlier seasons count less).
    qb_shrinkage None leaves quarterbacks out. Built as a sparse matrix: with ~1,500 quarterbacks
    in four seasons, a dense one would need gigabytes. prior (offense/defense by team) with
    prior_weight > 0: ratings are pulled toward the prior instead of toward average, as if it were
    worth prior_weight games (preseason ratings). qb_prior: quarterback ratings are pulled toward
    these instead of toward average; fixed_inexperience: the new-starter penalty is held at this
    value instead of fitted (both for the preseason version, fitted on a few weeks of games)."""
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {t: i for i, t in enumerate(teams)}
    use_qbs = qb_shrinkage is not None and "home_qb" in train
    qbs = sorted(set(train["home_qb"].dropna()) | set(train["away_qb"].dropna())) if use_qbs else []
    qb_index = {q: i for i, q in enumerate(qbs)}
    n, games, nq = len(teams), len(train), len(qbs)
    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    seasons_back = (season - train["season"].to_numpy()) if season is not None else 0
    w = np.tile(0.5 ** (days_ago / half_life_days) * season_weight ** np.maximum(seasons_back, 0), 2)

    h = train["home"].map(index).to_numpy()
    a = train["away"].map(index).to_numpy()
    rows = np.arange(games)
    home_side = np.where(train["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    r, c, v = [], [], []

    def put(row_idx, col_idx, values):
        r.append(np.asarray(row_idx)); c.append(np.asarray(col_idx)); v.append(np.broadcast_to(values, np.shape(row_idx)))

    put(rows, h, 1.0); put(rows, n + a, -1.0); put(rows, np.full(games, 2 * n), 1.0); put(rows, np.full(games, 2 * n + 1), home_side)
    put(games + rows, a, 1.0); put(games + rows, n + h, -1.0); put(games + rows, np.full(games, 2 * n), 1.0)
    put(games + rows, np.full(games, 2 * n + 1), -home_side)
    for side, side_rows in (("home", rows), ("away", games + rows)):
        if not use_qbs:
            break
        col = train[f"{side}_qb"].map(qb_index).to_numpy(float)
        known = ~np.isnan(col)
        put(side_rows[known], 2 * n + 2 + col[known].astype(int), 1.0)
    use_new = bool(inexperience_starts) and "home_qb_starts" in train
    if use_new:   # one more column: the new-starter penalty
        put(rows, np.full(games, 2 * n + 2 + nq), inexperience(train["home_qb_starts"], inexperience_starts))
        put(games + rows, np.full(games, 2 * n + 2 + nq), inexperience(train["away_qb_starts"], inexperience_starts))
    width = 2 * n + 2 + nq + int(use_new)
    x = sparse.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(2 * games, width))
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])
    penalty = np.zeros(width)
    penalty[:2 * n] = shrinkage
    penalty[2 * n + 2:2 * n + 2 + nq] = qb_shrinkage if use_qbs else 0
    if use_new:
        penalty[-1] = 1e-6
    xtw = x.T.multiply(w).tocsr()
    rhs = xtw @ y
    pull = np.zeros(width)             # (weight x value) of each pull toward a prior value
    if prior is not None and prior_weight > 0:
        pull[:n] = prior_weight * np.array([prior["offense"].get(t, 0.0) for t in teams])
        pull[n:2 * n] = prior_weight * np.array([prior["defense"].get(t, 0.0) for t in teams])
        penalty[:2 * n] += prior_weight
        penalty[2 * n:2 * n + 2] = 1e-6      # a few weeks of games may all be on neutral fields
    if qb_prior is not None and use_qbs:
        pull[2 * n + 2:2 * n + 2 + nq] = qb_shrinkage * np.array([qb_prior.get(q, 0.0) for q in qbs])
    if use_new and fixed_inexperience is not None:
        penalty[-1] = 1e6
        pull[-1] = 1e6 * fixed_inexperience
    rhs = rhs + pull
    beta = np.linalg.solve((xtw @ x).toarray() + np.diag(penalty), rhs)
    offense, defense = beta[:n] - beta[:n].mean(), beta[n:2 * n] - beta[n:2 * n].mean()
    return {"ratings": pd.DataFrame({"offense": offense, "defense": defense}, index=teams),
            "base": beta[2 * n] + beta[:n].mean() - beta[n:2 * n].mean(), "home_adv": beta[2 * n + 1],
            "qb_ratings": pd.Series(beta[2 * n + 2:2 * n + 2 + nq], index=qbs, dtype=float),
            "inexperience": float(beta[-1]) if use_new else 0.0, "inexperience_starts": inexperience_starts}


def qb_value(model: dict, qb) -> float:
    ratings = model.get("qb_ratings")
    return float(ratings.get(qb, 0.0)) if ratings is not None and isinstance(qb, str) else 0.0


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD,
            home_qb=None, away_qb=None, home_qb_starts=np.nan, away_qb_starts=np.nan) -> dict:
    r = model["ratings"]
    known = lambda t, c: r.at[t, c] if t in r.index else 0.0   # a team never seen: average
    edge = 0.0 if neutral else model["home_adv"] / 2
    new = lambda starts: (model.get("inexperience", 0.0) * float(inexperience([starts], model["inexperience_starts"])[0])
                          if model.get("inexperience_starts") else 0.0)
    home_pts = (model["base"] + edge + known(home, "offense") - known(away, "defense") + qb_value(model, home_qb)
                + new(home_qb_starts))
    away_pts = (model["base"] - edge + known(away, "offense") - known(home, "defense") + qb_value(model, away_qb)
                + new(away_qb_starts))
    margin = home_pts - away_pts
    return {"home_points": home_pts, "away_points": away_pts, "margin": margin, "total": home_pts + away_pts,
            "home_win": float(norm.cdf(margin / margin_sd))}


def window(played: pd.DataFrame, cutoff: str, training_years=TRAINING_YEARS) -> pd.DataFrame:
    start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
    return played[(played["date"] >= start) & (played["date"] < cutoff)]


def conference_of(games: pd.DataFrame, season: int) -> pd.Series:
    """Each team's conference in a season (from its schedule, known in advance)."""
    part = games[games["season"] == season]
    sides = pd.concat([part[["home", "home_conference"]].set_axis(["team", "conf"], axis=1),
                       part[["away", "away_conference"]].set_axis(["team", "conf"], axis=1)]).dropna()
    return sides.drop_duplicates("team", keep="last").set_index("team")["conf"]


def season_prior(games: pd.DataFrame, season: int, carry: float, **settings) -> tuple[pd.DataFrame, dict]:
    """Preseason ratings: the ratings when the season starts (the usual fit on earlier games), each
    team pulled (1 - carry) of the way toward its conference's average. Returns (prior, that fit)."""
    played = games[games["status"] == "final"]
    start = games.loc[games["season"] == season, "date"].min()
    years = settings.pop("training_years", TRAINING_YEARS)
    model = fit(window(played, start, years), start, season, **settings)
    ratings = model["ratings"].copy()
    conf = ratings.index.map(conference_of(games, season))
    for col in ("offense", "defense"):
        mean = ratings[col].groupby(conf).transform("mean")
        ratings[col] = np.where(pd.isna(conf), ratings[col], carry * ratings[col] + (1 - carry) * mean)
    return ratings, model


def fit_season(games: pd.DataFrame, played: pd.DataFrame, cutoff: str, season: int, priors: dict,
               carry: float, prior_weight: float, **settings) -> dict:
    """The preseason version: this season's games only, with team ratings pulled toward the
    preseason ratings (worth prior_weight games), quarterbacks toward their ratings when the season
    started, and the new-starter penalty as it was then. priors: a cache, {season: (prior, fit)}."""
    if season not in priors:
        priors[season] = season_prior(games, season, carry, **settings)
    prior, start_fit = priors[season]
    current = played[(played["season"] == season) & (played["date"] < cutoff)]
    if current.empty:
        return {**start_fit, "ratings": prior}
    settings = {k: v for k, v in settings.items() if k != "training_years"}
    model = fit(current, cutoff, season, **settings, prior=prior, prior_weight=prior_weight,
                qb_prior=start_fit["qb_ratings"], fixed_inexperience=start_fit["inexperience"])
    missing = prior.index.difference(model["ratings"].index)          # teams that haven't played yet
    model["ratings"] = pd.concat([model["ratings"], prior.loc[missing]])
    qbs = start_fit["qb_ratings"]
    model["qb_ratings"] = pd.concat([model["qb_ratings"], qbs[qbs.index.difference(model["qb_ratings"].index)]])
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
             training_years=TRAINING_YEARS, qb_shrinkage=QB_SHRINKAGE, starters="expected",
             inexperience_starts=INEXPERIENCE_STARTS, carry: float | None = None, prior_weight: float = 0.0,
             switch: str | None = None) -> pd.DataFrame:
    """Walk-forward: for each week, fit on earlier games only, then predict that week's games
    that involve at least one top-division (FBS) team. starters: 'expected' (each team's previous
    starter, what a live forecast knows) or 'actual' (who really started: an upper bound).
    carry None: the usual fit (the last four seasons by recency); otherwise preseason ratings
    (season_prior) plus this season's games until `switch` ('MM-DD'), then the usual fit."""
    priors: dict = {}
    settings = dict(half_life_days=half_life_days, season_weight=season_weight, shrinkage=shrinkage,
                    qb_shrinkage=qb_shrinkage, inexperience_starts=inexperience_starts, training_years=training_years)
    played = games[games["status"] == "final"]
    test = played[(played["season"] >= start_season) & ((played["home_division"] == "fbs") | (played["away_division"] == "fbs"))]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    suffix = "_qb_expected" if starters == "expected" else "_qb"
    rows = []
    for (season, _, week), week_games in test.groupby(["season", "season_type", "week"], sort=False):
        cutoff = week_games["date"].min()
        if carry is not None and (switch is None or "07-01" <= cutoff[5:] < switch):
            model = fit_season(games, played, cutoff, season, priors, carry, prior_weight, **settings)
        else:
            train = window(played, cutoff, training_years)
            if train.empty:
                continue
            model = fit(train, cutoff, season, half_life_days, season_weight, shrinkage, qb_shrinkage, inexperience_starts)
        for g in week_games.itertuples():
            starts = "_qb_expected_starts" if starters == "expected" else "_qb_starts"
            p = predict(model, g.home, g.away, bool(g.neutral_site), margin_sd,
                        getattr(g, "home" + suffix, None), getattr(g, "away" + suffix, None),
                        getattr(g, "home" + starts, np.nan), getattr(g, "away" + starts, np.nan))
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "home_points": p["home_points"], "away_points": p["away_points"]})
    return pd.DataFrame(rows)


def elo_probability(home_elo, away_elo, neutral, home_field: float = 55.0):
    """CollegeFootballData's pre-game Elo as a win probability (a benchmark), with a home bonus."""
    diff = np.asarray(home_elo, float) - np.asarray(away_elo, float) + np.where(np.asarray(neutral) == 1, 0, home_field)
    return 1 / (1 + 10 ** (-diff / 400))


def team_strength(model: dict, team: str, qb=None, qb_starts=np.nan) -> float:
    """Points per game better than an average team on a neutral field, with this quarterback."""
    r = model["ratings"]
    if team not in r.index:
        return float("nan")
    new = (model.get("inexperience", 0.0) * float(inexperience([qb_starts], model["inexperience_starts"])[0])
           if model.get("inexperience_starts") else 0.0)
    return float(r.at[team, "offense"] + r.at[team, "defense"] + qb_value(model, qb) + new)
