"""Champions League model: an attack and a defense rating for every club in Europe, from its
domestic league games and its European games together.

The problem: in the Champions League most clubs meet once, from a dozen countries, so the
European games alone can't rate them. Domestic games rate a club well against its own league. So
each club's rating is its league's rating plus its own difference from that league:

    goals the home side scores ~ Poisson( exp(base + home + attack[home] + league_attack[home's league]
                                               - defense[away] - league_defense[away's league]) )

and the same for the away side, without the home term. Two home advantages are fitted: one for
domestic games and one for European games. Within a league the league terms mostly cancel out, so
domestic games fix the clubs' differences; European games between clubs from different leagues fix
how strong each league is.

- Clubs are pulled toward their league's average (TEAM_SHRINKAGE), leagues more gently toward
  Europe's average (LEAGUE_SHRINKAGE). Both are worth that many games.
- Recent games count more (a match's weight halves every HALF_LIFE_DAYS), and European games count
  EUROPE_WEIGHT times a league game, since they're the only ones that compare leagues.
- A club's league is the domestic league it last played in. Clubs from leagues the feed doesn't
  cover (Croatia, Serbia, Ukraine, Czechia, ...) share one "other" league, rated from their European
  games only.
- Fitted by weighted maximum likelihood (L-BFGS on the exact gradient), on TRAINING_YEARS of games.

A match's chances come from the two expected goal counts and the Poisson scoreline grid, as in the
Premier League model (epl/model.py). Two-legged ties add extra time (a third of a game's goals) and,
if still level, a penalty shootout at even odds.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from epl.model import goal_markets, scoreline_grid

UCL = "uefa.champions"
EUROPE = ("uefa.champions", "uefa.europa", "uefa.europa.conf")
OTHER = "other"
# Tuned on the Champions League's 2016-17 to 2020-21 seasons (python -m ucl.tune_model, one setting at a
# time, two passes): log loss 0.9217 with the first guesses (365-day half-life, 3 years, European games
# worth 2) down to 0.9186, mostly from a longer memory (a 730-day half-life over 4 years). Elo scored
# 0.9629 and base rates 1.0634 on the same games.
# Version 2 (python -m ucl.improve_test, scored on the Champions and Europa Leagues' 2016-17 to 2020-21
# games): Transfermarkt squad values as part of each rating, with clubs pulled harder toward what their
# squads predict (team shrinkage 5 -> 20): log loss 0.9624 -> 0.9598. Weighting earlier seasons less,
# shrinking defense more, the Dixon-Coles adjustment and other weights all stayed within 0.0004.
HALF_LIFE_DAYS = 730
TEAM_SHRINKAGE = 20.0
LEAGUE_SHRINKAGE = 0.3
EUROPE_WEIGHT = 4.0
TRAINING_YEARS = 4
SEASON_WEIGHT = 1.0
DEFENSE_SHRINKAGE = 1.0
XG_WEIGHT = 0.0               # share of xG in what the ratings learn from, for cup games (ucl/tune_xg.py: xG made them worse)
LEAGUE_XG_WEIGHT = 0.75       # ... and for the big leagues' games, where it helped
EXTRA_TIME = 1 / 3            # extra time is 30 minutes: a third of a game's expected goals


def load_games(conn) -> pd.DataFrame:
    """Every soccer game (domestic and European), in date order, with Understat's xG where there is
    one (the big five leagues since 2014-15: ucl/import_understat.py)."""
    has_xg = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'soccer_xg'").fetchone()
    xg = ("x.home_xg, x.away_xg", "LEFT JOIN soccer_xg x ON x.event_id = g.event_id") if has_xg else ("NULL AS home_xg, NULL AS away_xg", "")
    return pd.read_sql(f"""SELECT g.event_id, g.competition, g.season, g.stage, g.date, g.kickoff, g.status, g.neutral, g.leg,
                                  g.home_id, g.away_id, g.home_score, g.away_score, g.home_pens, g.away_pens, g.extra_time, {xg[0]}
                           FROM soccer_games g {xg[1]} ORDER BY g.date, g.kickoff, g.event_id""", conn)


def window(games: pd.DataFrame, as_of: str, training_years: float = TRAINING_YEARS) -> pd.DataFrame:
    start = (pd.Timestamp(as_of) - pd.DateOffset(days=int(365.25 * training_years))).strftime("%Y-%m-%d")
    return games[(games["status"] == "final") & (games["date"] >= start) & (games["date"] < as_of)
                 & games["home_score"].notna() & games["away_score"].notna()]


def leagues_of(train: pd.DataFrame) -> dict:
    """Each club's league: the domestic competition of its latest domestic game."""
    domestic = train[~train["competition"].isin(EUROPE)]
    sides = pd.concat([domestic[["date", "competition", "home_id"]].rename(columns={"home_id": "team"}),
                       domestic[["date", "competition", "away_id"]].rename(columns={"away_id": "team"})])
    return sides.sort_values("date").drop_duplicates("team", keep="last").set_index("team")["competition"].to_dict()


def fit(train: pd.DataFrame, as_of: str, half_life_days: float = HALF_LIFE_DAYS, team_shrinkage: float = TEAM_SHRINKAGE,
        league_shrinkage: float = LEAGUE_SHRINKAGE, europe_weight: float = EUROPE_WEIGHT, start: dict | None = None,
        season_weight: float = SEASON_WEIGHT, defense_shrinkage: float = DEFENSE_SHRINKAGE,
        values: pd.DataFrame | None = None, xg_weight: float | None = None) -> dict:
    """The ratings from finished games before as_of. xg_weight: in games with Understat's xG, learn
    from this blend of xG and goals (XG_WEIGHT by default; games without xG use goals). start: an earlier fit to start from (faster).
    season_weight: each earlier season's games count this much less again (seasons start on 1 July);
    defense_shrinkage: the defense ratings' pull toward the league, as a multiple of the attack's.
    values: squad values (espn_id, month, value_eur; ucl/squad_values.py): each side's log squad
    value (as it was that month) adds to its attack and defense with fitted weights, so ratings
    start from what a squad is worth on paper and results move them from there."""
    teams = sorted(set(train["home_id"]) | set(train["away_id"]))
    league = leagues_of(train)
    groups = sorted({league.get(t, OTHER) for t in teams})
    ti = {t: i for i, t in enumerate(teams)}
    gi = {g: i for i, g in enumerate(groups)}
    nt, ng, n = len(teams), len(groups), len(train)
    # columns: attack[nt], defense[nt], league attack[ng], league defense[ng], base, home (domestic), home (Europe)
    A, D, LA, LD = 0, nt, 2 * nt, 2 * nt + ng
    BASE, HOME_DOM, HOME_EUR = 2 * nt + 2 * ng, 2 * nt + 2 * ng + 1, 2 * nt + 2 * ng + 2
    use_values = values is not None and not values.empty
    VAL_A, VAL_D = HOME_EUR + 1, HOME_EUR + 2
    width = HOME_EUR + 1 + (2 if use_values else 0)
    if use_values:
        hv, av, center = value_columns(train, values)
    h = train["home_id"].map(ti).to_numpy()
    a = train["away_id"].map(ti).to_numpy()
    hg = np.array([gi[league.get(t, OTHER)] for t in train["home_id"]])
    ag = np.array([gi[league.get(t, OTHER)] for t in train["away_id"]])
    europe = train["competition"].isin(EUROPE).to_numpy()
    at_home = (train["neutral"].fillna(0).to_numpy() == 0).astype(float)
    rows = np.arange(n)

    def side(attacker, attacker_group, defender, defender_group, home_term):
        r = np.concatenate([rows] * 5 + [rows, rows])
        c = np.concatenate([A + attacker, LA + attacker_group, D + defender, LD + defender_group, np.full(n, BASE),
                            np.full(n, HOME_DOM), np.full(n, HOME_EUR)])
        v = np.concatenate([np.ones(n), np.ones(n), -np.ones(n), -np.ones(n), np.ones(n),
                            home_term * ~europe, home_term * europe])
        return sparse.csr_matrix((v, (r, c)), shape=(n, HOME_EUR + 1))

    x = sparse.vstack([side(h, hg, a, ag, at_home), side(a, ag, h, hg, np.zeros(n))]).tocsr()
    if use_values:   # attacker's value adds to its attack; defender's value to its defense (so it subtracts goals)
        extra = sparse.csr_matrix(np.column_stack([np.concatenate([hv, av]), -np.concatenate([av, hv])]))
        x = sparse.hstack([x, extra]).tocsr()
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])
    xg_weight = XG_WEIGHT if xg_weight is None else xg_weight
    if xg_weight and "home_xg" in train:
        xg = np.concatenate([train["home_xg"].to_numpy(float), train["away_xg"].to_numpy(float)])
        y = np.where(np.isnan(xg), y, (1 - xg_weight) * y + xg_weight * np.nan_to_num(xg))
    days = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    w = 0.5 ** (days / half_life_days) * np.where(europe, europe_weight, 1.0)
    if season_weight != 1.0:
        start_of = lambda d: d.dt.year - (d.dt.month < 7)
        back = (start_of(pd.Series([pd.Timestamp(as_of)])).iloc[0] - start_of(pd.to_datetime(train["date"]))).to_numpy()
        w = w * season_weight ** np.maximum(back, 0)
    w = np.concatenate([w, w])
    penalty = np.zeros(width)
    penalty[A:D] = team_shrinkage
    penalty[D:LA] = team_shrinkage * defense_shrinkage
    penalty[LA:BASE] = league_shrinkage
    xt = x.T.tocsr()

    def objective(beta):
        eta = x @ beta
        mu = np.exp(np.clip(eta, -10, 10))
        value = float(np.sum(w * (mu - y * eta)) + 0.5 * np.sum(penalty * beta ** 2))
        grad = xt @ (w * (mu - y)) + penalty * beta
        return value, grad

    beta0 = np.zeros(width)
    beta0[BASE] = np.log(max(y.mean(), 0.1))
    if start is not None:
        for t, i in ti.items():
            if t in start["ratings"].index:
                beta0[A + i] = start["ratings"].at[t, "attack"]
                beta0[D + i] = start["ratings"].at[t, "defense"]
        for g, i in gi.items():
            if g in start["leagues"].index:
                beta0[LA + i] = start["leagues"].at[g, "attack"]
                beta0[LD + i] = start["leagues"].at[g, "defense"]
        beta0[[BASE, HOME_DOM, HOME_EUR]] = start["base"], start["home_domestic"], start["home_europe"]
    result = minimize(objective, beta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    beta = result.x
    ratings = pd.DataFrame({"attack": beta[A:D], "defense": beta[D:LA],
                            "league": [league.get(t, OTHER) for t in teams]}, index=teams)
    leagues = pd.DataFrame({"attack": beta[LA:LD], "defense": beta[LD:BASE]}, index=groups)
    counts = pd.concat([train["home_id"], train["away_id"]]).value_counts()
    ratings["games"] = counts.reindex(teams).fillna(0).astype(int)
    model = {"ratings": ratings, "leagues": leagues, "base": float(beta[BASE]), "home_domestic": float(beta[HOME_DOM]),
             "home_europe": float(beta[HOME_EUR]), "as_of": as_of, "converged": bool(result.success)}
    if use_values:
        model["value_attack"], model["value_defense"] = float(beta[VAL_A]), float(beta[VAL_D])
        model["values"] = current_values(values, as_of, center)
        model["value_center"] = center
    return model


def value_columns(train: pd.DataFrame, values: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Each game's home and away log squad value (the latest within six months before the game),
    centered on the average; missing values get the average (0)."""
    v = values.assign(month=pd.to_datetime(values["month"]), log_value=np.log(values["value_eur"].clip(lower=1e5)))
    v = v.sort_values("month")
    center = float(v["log_value"].mean())
    out = []
    for side in ("home_id", "away_id"):
        rows = train[[side, "date"]].assign(when=pd.to_datetime(train["date"]), order=np.arange(len(train)))
        m = pd.merge_asof(rows.sort_values("when"), v[["espn_id", "month", "log_value"]], left_on="when", right_on="month",
                          left_by=side, right_by="espn_id", direction="backward", tolerance=pd.Timedelta(days=190))
        out.append(m.sort_values("order")["log_value"].fillna(center).to_numpy() - center)
    return out[0], out[1], center


def current_values(values: pd.DataFrame, as_of: str, center: float) -> dict:
    """Each club's latest log squad value before as_of (within a year), centered."""
    v = values[values["month"] <= as_of]
    v = v[v["month"] >= (pd.Timestamp(as_of) - pd.DateOffset(days=365)).strftime("%Y-%m-%d")]
    latest = v.sort_values("month").drop_duplicates("espn_id", keep="last")
    return dict(zip(latest["espn_id"], np.log(latest["value_eur"].clip(lower=1e5)) - center))


def load_values(conn) -> pd.DataFrame:
    if conn.execute("SELECT name FROM sqlite_master WHERE name = 'soccer_squad_values'").fetchone() is None:
        return pd.DataFrame(columns=["espn_id", "month", "value_eur"])
    return pd.read_sql("SELECT espn_id, month, value_eur FROM soccer_squad_values", conn)


def strength(model: dict, team: str) -> tuple[float, float]:
    """(attack, defense) on the log scale, league included; an unknown club gets the 'other' league's."""
    r, lg = model["ratings"], model["leagues"]
    other = lg.loc[OTHER] if OTHER in lg.index else pd.Series({"attack": lg["attack"].min(), "defense": lg["defense"].min()})
    v = model.get("values", {}).get(team, 0.0)
    va, vd = model.get("value_attack", 0.0) * v, model.get("value_defense", 0.0) * v
    if team in r.index:
        group = lg.loc[r.at[team, "league"]] if r.at[team, "league"] in lg.index else other
        return r.at[team, "attack"] + group["attack"] + va, r.at[team, "defense"] + group["defense"] + vd
    return float(other["attack"]) + va, float(other["defense"]) + vd


def expected_goals(model: dict, home: str, away: str, neutral: bool = False, europe: bool = True) -> tuple[float, float]:
    ha, hd = strength(model, home)
    aa, ad = strength(model, away)
    edge = 0.0 if neutral else model["home_europe" if europe else "home_domestic"]
    return float(np.exp(model["base"] + edge + ha - ad)), float(np.exp(model["base"] + aa - hd))


def predict(model: dict, home: str, away: str, neutral: bool = False, europe: bool = True) -> dict:
    """europe False: a league game (the domestic home advantage). In a league game, if the model
    has Elo ratings attached (ucl/elo.py), the result chances blend them in."""
    exp_home, exp_away = expected_goals(model, home, away, neutral, europe)
    grid = scoreline_grid(exp_home, exp_away)
    p = {"exp_home": exp_home, "exp_away": exp_away, "grid": grid, **goal_markets(grid)}
    if not europe:
        from ucl import elo
        q = elo.league_chances(model, home, away, neutral)
        if q is not None:
            p = elo.blend(p, q)
    return p


def rating(model: dict, team: str) -> float:
    """Goal difference a game against an average club in Europe's big five leagues, neutral field."""
    if "reference" not in model:           # the average big-five club, worked out once per fit
        big = [g for g in ("eng.1", "esp.1", "ger.1", "ita.1", "fra.1") if g in model["ratings"]["league"].values]
        reference = model["ratings"][model["ratings"]["league"].isin(big)] if big else model["ratings"]
        model["reference"] = (float(np.mean([strength(model, t)[0] for t in reference.index])),
                              float(np.mean([strength(model, t)[1] for t in reference.index])))
    ref_attack, ref_defense = model["reference"]
    a, d = strength(model, team)
    scored = np.exp(model["base"] + a - ref_defense)
    conceded = np.exp(model["base"] + ref_attack - d)
    return float(scored - conceded)


def tie_chance(model: dict, first_home: str, second_home: str, first_leg: tuple | None = None) -> float:
    """Chance that first_home (at home in the first leg) wins a two-legged tie: both legs' scorelines,
    then extra time at the second leg's ground, then a shootout at even odds. first_leg: its score
    (first_home's goals, second_home's goals) if already played."""
    g1 = scoreline_grid(*expected_goals(model, first_home, second_home)) if first_leg is None else None
    eh2, ea2 = expected_goals(model, second_home, first_home)
    g2 = scoreline_grid(eh2, ea2)
    n = g2.shape[0]
    # margin for first_home over the two legs
    if first_leg is None:
        m1 = np.zeros(2 * n - 1)
        for i in range(n):
            for j in range(n):
                m1[i - j + n - 1] += g1[i, j]
    else:
        m1 = np.zeros(2 * n - 1)
        m1[first_leg[0] - first_leg[1] + n - 1] = 1.0
    m2 = np.zeros(2 * n - 1)                       # second leg: first_home is away
    for i in range(n):
        for j in range(n):
            m2[j - i + n - 1] += g2[i, j]
    total = np.convolve(m1, m2)                    # index k -> margin k - 2(n - 1)
    zero = 2 * (n - 1)
    win = total[zero + 1:].sum()
    level = total[zero]
    et = scoreline_grid(ea2 * EXTRA_TIME, eh2 * EXTRA_TIME)   # first_home's extra-time goals first
    et_win = np.tril(et, -1).sum()
    et_level = np.trace(et)
    return float(win + level * (et_win + 0.5 * et_level))


def single_game_winner(model: dict, home: str, away: str, neutral: bool = True) -> float:
    """Chance home wins a one-off knockout game (the final): 90 minutes, extra time, penalties."""
    eh, ea = expected_goals(model, home, away, neutral)
    g = scoreline_grid(eh, ea)
    et = scoreline_grid(eh * EXTRA_TIME, ea * EXTRA_TIME)
    return float(np.tril(g, -1).sum() + np.trace(g) * (np.tril(et, -1).sum() + 0.5 * np.trace(et)))


# ---------------------------------------------------------------- the backtest

MAIN_STAGES = ("group-stage", "201617-group-stage", "league-phase", "knockout-round-playoffs", "round-of-32", "round-of-16", "quarterfinals",
               "semifinals", "final")


def scored_games(games: pd.DataFrame, competitions=(UCL,)) -> pd.DataFrame:
    """The games a backtest scores: a cup's main competition (no qualifying rounds), every league game."""
    part = games[games["competition"].isin(competitions) & (games["status"] == "final")]
    return part[part["stage"].isin(MAIN_STAGES) | ~part["competition"].isin(EUROPE)]


def backtest(games: pd.DataFrame, first_season: int, last_season: int | None = None, dates: set | None = None,
             competitions=(UCL,), **settings) -> pd.DataFrame:
    """Walk-forward: before each cup match day, fit on every game played before it, then predict
    that day's games. dates: only these match days (for adding new ones)."""
    test = scored_games(games, competitions)
    test = test[(test["season"] >= first_season) & ((test["season"] <= last_season) if last_season is not None else True)]
    if dates is not None:
        test = test[test["date"].isin(dates)]
    years = settings.pop("training_years", TRAINING_YEARS)
    rows, previous = [], {}
    elo_diff, coef = {}, None
    fixed_xg = settings.pop("xg_weight", None)            # tuning: one xG weight for every game
    if not test["competition"].isin(EUROPE).all():        # league games blend in Elo (ucl/elo.py), as the site does
        from ucl import elo
        diffs, _ = elo.run(games)
        coef = elo.coefficients(games, diffs)
        elo_diff = dict(zip(diffs["event_id"], diffs["elo_diff"]))
    for day, day_games in test.groupby("date"):
        models = {}
        for europe in sorted(set(day_games["competition"].isin(EUROPE))):   # the cups' model and the leagues' model
            xg_weight = fixed_xg if fixed_xg is not None else (XG_WEIGHT if europe else LEAGUE_XG_WEIGHT)
            models[europe] = fit(window(games, day, years), day, start=previous.get(europe), xg_weight=xg_weight, **settings)
            previous[europe] = {k: models[europe][k] for k in ("ratings", "leagues", "base", "home_domestic", "home_europe")}
        for g in day_games.itertuples():
            model = models[g.competition in EUROPE]
            p = predict(model, g.home_id, g.away_id, bool(g.neutral), g.competition in EUROPE)
            if g.competition not in EUROPE and coef is not None and g.event_id in elo_diff:
                p = elo.blend(p, elo.chances(elo_diff[g.event_id], coef))
            rows.append({"event_id": g.event_id, "competition": g.competition, "date": day, "season": g.season, "stage": g.stage,
                         "home": p["home"], "draw": p["draw"], "away": p["away"],
                         "exp_home": p["exp_home"], "exp_away": p["exp_away"]})
    return pd.DataFrame(rows)


def result_of(home_score, away_score, extra_time=0) -> str:
    """The result after 90 minutes: a game that went to extra time was level then."""
    if extra_time == 1:
        return "draw"
    return "home" if home_score > away_score else "away" if away_score > home_score else "draw"


def log_loss(results: pd.DataFrame, games: pd.DataFrame) -> float:
    """Average log loss of the chances given to the result after 90 minutes."""
    d = results.merge(games[["event_id", "home_score", "away_score", "extra_time"]], on="event_id")
    outcome = [result_of(h, a, e) for h, a, e in zip(d["home_score"], d["away_score"], d["extra_time"].fillna(0))]
    p = np.array([row[o] for row, o in zip(d.to_dict("records"), outcome)])
    return float(-np.mean(np.log(np.clip(p, 1e-9, 1))))
