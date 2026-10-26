"""NBA model: player ratings and who's available, plus team ratings.

The winning margin of a game is modelled as

    home court  +  sum over the home team's players of (share of the game on the floor x rating)
                -  the same for the away team  +  a team term for each side

A player's rating is how many points per game his team gains (or loses) with him on the floor for
a whole game, compared with an average player, after allowing for everyone else on both teams.
It is estimated from every game of the last few seasons at once (ridge regression, recent games
counting more), so a player is separated from his teammates as lineups change through injuries,
rest, trades and rotations. The team term (pulled hard toward zero) picks up what the players
don't: coaching, system, how the roster fits.

For a game still to play, the available players are the team's roster minus those the injury
report rules out (nba/live.py). Each one's usual minutes are his recent average over the games he
was available for (a coach's decision not to play him counts as 0), and the game's 240 minutes
are filled in order of usual minutes (the end of the bench gets what's left). The backtest does
the same with each game's box score: every player available for it (he played, or sat by the
coach's decision), with his usual minutes before it.

A team playing the second night of a back-to-back is worse: a fitted term for the difference
in back-to-backs (one side's minus the other's) is part of the margin. Who sits out to rest is
already handled by availability.

Points (for the expected score and total) come from the team offense/defense model used for
college basketball (cfb/model.py's fit); the margin above splits that total between the teams.
"""

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.stats import norm

from cfb.model import fit as team_fit

# Tuned on 2010-11 to 2013-14 (python -m nba.tune_model): log loss 0.5928, from 0.5945 with first
# guesses; weaker player shrinkage (3) or a two-year half-life were worse, stronger (15) the same.
HALF_LIFE_DAYS = 365
SEASON_WEIGHT = 1.0
PLAYER_SHRINKAGE = 10.0     # pulls player ratings toward average (in games' worth of evidence)
TEAM_SHRINKAGE = 40.0       # pulls the team terms toward zero
TRAINING_YEARS = 3
MARGIN_SD = 11.0
MINUTES_HALF_LIFE = 8       # games: how fast expected minutes follow a player's recent role
DEBUT_MINUTES = 12.0        # expected minutes for a player with no NBA games yet
REST_SHRINKAGE = 5.0        # pulls the back-to-back effect toward zero
POINTS = {"half_life_days": 120, "season_weight": 0.5, "shrinkage": 2.0}   # the team points model


# ---------------------------------------------------------------- data

def load_games(conn) -> pd.DataFrame:
    """Every game (played and scheduled), in date order, with its week (named by its Monday)."""
    games = pd.read_sql("""
        SELECT g.id, g.season, g.date, g.kickoff, g.status, h.name AS home, a.name AS away,
               g.home_score, g.away_score, i.season_type, i.neutral_site, i.notes, i.source_game_id
        FROM games g
        JOIN leagues l ON l.id = g.league_id AND l.code = 'NBA'
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        JOIN nba_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.kickoff, g.id
    """, conn)
    games["week"] = pd.to_datetime(games["date"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    return add_rest(games)


def add_rest(games: pd.DataFrame) -> pd.DataFrame:
    """home_rest / away_rest: days since the team's previous game (NaN for its first of the
    season); home_b2b / away_b2b: 1 on the second night of a back-to-back. The schedule is known in
    advance, so scheduled games count too."""
    games = games.copy()
    sides = pd.concat([games[["id", "season", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
                       games[["id", "season", "date", "away"]].rename(columns={"away": "team"}).assign(side="away")])
    sides["day"] = pd.to_datetime(sides["date"])
    sides = sides.sort_values(["team", "day", "id"])
    sides["rest"] = (sides["day"] - sides.groupby(["team", "season"])["day"].shift(1)).dt.days
    for side in ("home", "away"):
        part = sides[sides["side"] == side].set_index("id")["rest"]
        games[f"{side}_rest"] = games["id"].map(part)
        games[f"{side}_b2b"] = (games[f"{side}_rest"] == 1).astype(int)
    return games


def load_players(conn) -> pd.DataFrame:
    """Every player line in a finished game, with the game's date, season and the player's side."""
    players = pd.read_sql("""
        SELECT p.game_id, p.team, p.player_id, p.name, p.position, p.starter, p.minutes, p.did_not_play, p.reason,
               p.points, p.rebounds, p.assists, p.threes, p.plus_minus, g.date, g.season,
               CASE WHEN h.name = p.team THEN 'home' ELSE 'away' END AS side
        FROM nba_player_games p
        JOIN games g ON g.id = p.game_id
        JOIN teams h ON h.id = g.home_team_id
        ORDER BY g.date, p.game_id
    """, conn)
    return add_expected_minutes(players)


def available(players: pd.DataFrame) -> pd.Series:
    """Whether the player was available for the game: he played, or the box score lists him as not
    playing by the coach's decision (rather than injured, resting, suspended...)."""
    reason = players["reason"].fillna("").str.upper()
    return (players["minutes"] > 0) | ((players["did_not_play"] == 1) & (reason.str.startswith("COACH") | (reason == "")))


def add_expected_minutes(players: pd.DataFrame, half_life: float = MINUTES_HALF_LIFE) -> pd.DataFrame:
    """expected_minutes: before each game, the player's average minutes in the games he played,
    the most recent counting most (DEBUT_MINUTES before his first); minutes_after: the same
    including this game. rotation_after: his average minutes in the games he was available for
    (a coach's-decision DNP counts as 0), for live lineups, where the whole roster is listed."""
    players = players.copy()
    active = players[available(players)].sort_values(["player_id", "date", "game_id"])
    rotation = active.groupby("player_id")["minutes"].transform(lambda m: m.fillna(0).ewm(halflife=half_life).mean())
    before = rotation.groupby(active["player_id"]).shift(1)
    players = players.merge(active[["game_id", "player_id"]].assign(rotation_after=rotation.to_numpy(),
                                                                     rotation_before=before.to_numpy()),
                            on=["game_id", "player_id"], how="left")
    played = players[players["minutes"] > 0].sort_values(["player_id", "date", "game_id"])
    after = played.groupby("player_id")["minutes"].transform(lambda m: m.ewm(halflife=half_life).mean())
    played = played.assign(minutes_after=after)
    played["expected_minutes"] = played.groupby("player_id")["minutes_after"].shift(1).fillna(DEBUT_MINUTES)
    players = players.merge(played[["game_id", "player_id", "expected_minutes", "minutes_after"]],
                            on=["game_id", "player_id"], how="left")
    return players


def fill_minutes(usual: np.ndarray, total: float = 240.0) -> np.ndarray:
    """Share a game's minutes among the available players: in order of their usual minutes, each
    gets his usual minutes until the game's are used up (the end of the bench gets what's left);
    if they add up to less than a game (injuries), everyone's go up in proportion."""
    usual = np.maximum(np.asarray(usual, float), 0)
    if usual.sum() <= total:
        return usual * (total / usual.sum()) if usual.sum() > 0 else usual
    order = np.argsort(-usual, kind="stable")
    before = np.concatenate([[0.0], np.cumsum(usual[order])[:-1]])
    out = np.zeros_like(usual)
    out[order] = np.clip(np.minimum(usual[order], total - before), 0, None)
    return out


def floor_shares(minutes: pd.Series, team_key: pd.Series) -> pd.Series:
    """Minutes as a share of a game on the floor: a team's shares add up to 5 (five players on the
    floor), whatever the overtime."""
    total = minutes.groupby(team_key).transform("sum")
    return 5 * minutes / total.where(total > 0, 1)


# ---------------------------------------------------------------- fitting

def fit(games: pd.DataFrame, players: pd.DataFrame, as_of: str, season: int | None = None,
        half_life_days=HALF_LIFE_DAYS, season_weight=SEASON_WEIGHT, player_shrinkage=PLAYER_SHRINKAGE,
        team_shrinkage=TEAM_SHRINKAGE, training_years=TRAINING_YEARS, with_points: bool = True,
        rest_shrinkage=REST_SHRINKAGE) -> dict:
    """Player ratings, team terms, home court and the back-to-back effect from finished games
    before as_of (plus the team points model, for expected scores). games/players: everything; this
    picks the window. rest_shrinkage None: no back-to-back term."""
    start = (pd.Timestamp(as_of) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
    train = games[(games["status"] == "final") & (games["date"] >= start) & (games["date"] < as_of)]
    lines = players[players["game_id"].isin(train["id"]) & (players["minutes"] > 0)]
    train = train[train["id"].isin(lines["game_id"])]          # games with a box score
    season = season if season is not None else int(train["season"].max()) if len(train) else 0
    if train.empty:
        return {"ratings": pd.Series(dtype=float), "teams": pd.Series(dtype=float), "home_adv": 2.5, "b2b": 0.0,
                "points": None, "as_of": as_of, "season": season}

    row = {g: i for i, g in enumerate(train["id"])}
    player_ids = sorted(lines["player_id"].unique())
    teams = sorted(set(train["home"]) | set(train["away"]))
    p_col = {p: i for i, p in enumerate(player_ids)}
    t_col = {t: len(player_ids) + i for i, t in enumerate(teams)}
    rest = rest_shrinkage is not None and "home_b2b" in train
    width = len(player_ids) + len(teams) + 1 + int(rest)
    home_col = len(player_ids) + len(teams)

    share = floor_shares(lines["minutes"], lines["game_id"].astype(str) + lines["team"])
    sign = np.where(lines["side"] == "home", 1.0, -1.0)
    n = np.arange(len(train))
    r = np.concatenate([lines["game_id"].map(row).to_numpy(), n, n, n])
    c = np.concatenate([lines["player_id"].map(p_col).to_numpy(), train["home"].map(t_col).to_numpy(),
                        train["away"].map(t_col).to_numpy(), np.full(len(train), home_col)])
    v = np.concatenate([sign * share.to_numpy(), np.ones(len(train)), -np.ones(len(train)),
                        1.0 - train["neutral_site"].to_numpy(float)])
    if rest:   # positive when the away team is the one on a back-to-back
        r = np.concatenate([r, n])
        c = np.concatenate([c, np.full(len(train), home_col + 1)])
        v = np.concatenate([v, (train["away_b2b"] - train["home_b2b"]).to_numpy(float)])
    x = sparse.csr_matrix((v, (r, c)), shape=(len(train), width))
    y = (train["home_score"] - train["away_score"]).to_numpy(float)
    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    w = 0.5 ** (days_ago / half_life_days) * season_weight ** np.maximum(season - train["season"].to_numpy(), 0)

    penalty = np.zeros(width)
    penalty[:len(player_ids)] = player_shrinkage
    penalty[len(player_ids):home_col] = team_shrinkage
    penalty[home_col] = 1e-6
    if rest:
        penalty[home_col + 1] = rest_shrinkage
    xtw = x.T.multiply(w).tocsr()
    a = (xtw @ x).toarray() + np.diag(penalty)
    beta = linalg.solve(a, xtw @ y, assume_a="pos")
    model = {"ratings": pd.Series(beta[:len(player_ids)], index=player_ids),
             "teams": pd.Series(beta[len(player_ids):home_col], index=teams), "home_adv": float(beta[home_col]),
             "b2b": float(beta[home_col + 1]) if rest else 0.0, "points": None, "as_of": as_of, "season": season}
    if with_points:
        model["points"] = team_fit(train, as_of, season, qb_shrinkage=None, inexperience_starts=0, **POINTS)
    return model


# ---------------------------------------------------------------- predicting

def lineup_strength(model: dict, lineup: pd.DataFrame) -> float:
    """Sum of floor share x rating for one team's expected lineup (columns player_id, minutes)."""
    if lineup.empty:
        return 0.0
    shares = 5 * lineup["minutes"] / max(lineup["minutes"].sum(), 1e-9)
    return float((shares * lineup["player_id"].map(model["ratings"]).fillna(0.0)).sum())


def predict(model: dict, home: str, away: str, home_lineup: pd.DataFrame, away_lineup: pd.DataFrame,
            neutral: bool = False, margin_sd: float = MARGIN_SD, home_b2b: int = 0, away_b2b: int = 0) -> dict:
    """Win chance, margin and expected points. Lineups: each team's available players with their
    expected minutes (columns player_id, minutes); an empty lineup counts as an average team.
    home_b2b / away_b2b: 1 if the team is on the second night of a back-to-back."""
    parts = breakdown(model, home, away, home_lineup, away_lineup, neutral, home_b2b, away_b2b)
    margin = sum(parts.values())
    total = 2 * 112.0
    if model.get("points") is not None:
        pts = model["points"]
        known = lambda team, col: pts["ratings"].at[team, col] if team in pts["ratings"].index else 0.0
        edge = 0.0 if neutral else pts["home_adv"] / 2
        home_pts = pts["base"] + edge + known(home, "offense") - known(away, "defense")
        away_pts = pts["base"] - edge + known(away, "offense") - known(home, "defense")
        total = home_pts + away_pts
    return {"home_win": float(norm.cdf(margin / margin_sd)), "margin": margin, "total": total,
            "home_points": (total + margin) / 2, "away_points": (total - margin) / 2}


def breakdown(model: dict, home: str, away: str, home_lineup: pd.DataFrame, away_lineup: pd.DataFrame,
              neutral: bool = False, home_b2b: int = 0, away_b2b: int = 0) -> dict:
    """The predicted home margin, split into its parts (points, + favours the home team): the
    players expected to play, the team terms, home court and back-to-backs."""
    t = model["teams"]
    return {"players": lineup_strength(model, home_lineup) - lineup_strength(model, away_lineup),
            "team": float(t.get(home, 0.0)) - float(t.get(away, 0.0)),
            "home_court": 0.0 if neutral else model["home_adv"],
            "rest": model.get("b2b", 0.0) * (int(away_b2b) - int(home_b2b))}


def team_lineups(players: pd.DataFrame, mode: str = "played") -> pd.DataFrame:
    """For every finished game and team, the lineup a prediction made before it would use:
    player_id and minutes, by (game_id, team).

    mode 'roster_fill': every player available for the game (he played, or sat by the coach's
        decision), with his usual minutes, the game's 240 filled in order: what a live
        prediction does (the default). 'roster': the same, scaled down evenly instead.
    mode 'played': the players who really played, with their expected minutes (this knows a
        little too much: late scratches and who the coach left out). Used for tuning.
    mode 'actual': the minutes they really played (an upper bound: nobody knows that beforehand).
    mode 'usual': ignoring who's available: the team's players from its previous 5 games, with
        their expected minutes (a player hurt in the meantime still counts)."""
    played = players[players["minutes"] > 0]
    if mode == "actual":
        return played[["game_id", "team", "player_id", "minutes"]]
    if mode == "played":
        return played[["game_id", "team", "player_id"]].assign(minutes=played["expected_minutes"])
    if mode in ("roster", "roster_fill"):
        # like a live prediction: every player available for the game (he played, or sat by the
        # coach's decision), with his usual minutes counting such games as 0
        avail = players[available(players)].dropna(subset=["rotation_before"])
        lineups = avail[["game_id", "team", "player_id"]].assign(minutes=avail["rotation_before"].to_numpy())
        if mode == "roster_fill":
            lineups["minutes"] = lineups.groupby(["game_id", "team"])["minutes"].transform(lambda m: fill_minutes(m.to_numpy()))
        return lineups
    if mode != "usual":
        raise ValueError(mode)
    order = played[["game_id", "date", "team"]].drop_duplicates().sort_values(["team", "date", "game_id"])
    order["n"] = order.groupby("team").cumcount()
    by_n = played.merge(order[["game_id", "team", "n"]], on=["game_id", "team"])
    rows = []
    for k in range(1, 6):   # the five games before
        prev = by_n[["team", "n", "player_id", "minutes_after", "date"]].assign(n=by_n["n"] + k)
        rows.append(order[["game_id", "team", "n"]].merge(prev, on=["team", "n"]))
    usual = pd.concat(rows).sort_values("date").drop_duplicates(["game_id", "team", "player_id"], keep="last")
    return usual[["game_id", "team", "player_id"]].assign(minutes=usual["minutes_after"])


# ---------------------------------------------------------------- testing

def backtest(games: pd.DataFrame, players: pd.DataFrame, start_season: int, end_season: int | None = None,
             mode: str = "roster_fill", half_life_days=HALF_LIFE_DAYS, season_weight=SEASON_WEIGHT,
             player_shrinkage=PLAYER_SHRINKAGE, team_shrinkage=TEAM_SHRINKAGE, training_years=TRAINING_YEARS,
             margin_sd=MARGIN_SD, with_points: bool = True, players_only: bool = False,
             rest_shrinkage=REST_SHRINKAGE) -> pd.DataFrame:
    """Walk-forward, a week at a time: fit on games before the week, predict the week's games
    (regular season, play-in and playoffs). mode: see team_lineups. players_only False with mode
    'none': the team terms alone (no player ratings), for comparison."""
    test = games[(games["status"] == "final") & (games["season"] >= start_season) & (games["season_type"] != "cup_final")]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    lineups = team_lineups(players, mode) if mode != "none" else pd.DataFrame(columns=["game_id", "team", "player_id", "minutes"])
    by_game = {k: v for k, v in lineups.groupby(["game_id", "team"])}
    empty = pd.DataFrame(columns=["player_id", "minutes"])
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=False):
        cutoff = week_games["date"].min()
        model = fit(games, players, cutoff, season, half_life_days, season_weight,
                    player_shrinkage if mode != "none" else 1e9, team_shrinkage if mode != "none" else 1.0,
                    training_years, with_points, rest_shrinkage)
        if model["ratings"].empty and model["teams"].empty:
            continue
        for g in week_games.itertuples():
            p = predict(model, g.home, g.away, by_game.get((g.id, g.home), empty), by_game.get((g.id, g.away), empty),
                        bool(g.neutral_site), margin_sd, getattr(g, "home_b2b", 0), getattr(g, "away_b2b", 0))
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "home_points": p["home_points"], "away_points": p["away_points"]})
    return pd.DataFrame(rows)


def elo_benchmark(games: pd.DataFrame, k: float = 20.0, home: float = 100.0, revert: float = 0.25) -> pd.Series:
    """A simple Elo rating's pre-game home win chance for every finished game (the benchmark: there
    are no free historical betting lines). The settings FiveThirtyEight used for the NBA."""
    ratings: dict = {}
    out = {}
    season = None
    for g in games[(games["status"] == "final") & (games["season_type"] != "cup_final")].itertuples():
        if g.season != season:
            ratings = {t: 1505 + (r - 1505) * (1 - revert) for t, r in ratings.items()}
            season = g.season
        rh, ra = ratings.get(g.home, 1505.0), ratings.get(g.away, 1505.0)
        diff = rh - ra + (0 if g.neutral_site else home)
        p = 1 / (1 + 10 ** (-diff / 400))
        out[g.id] = p
        margin = g.home_score - g.away_score
        result = 1.0 if margin > 0 else 0.0
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * (diff if result else -diff))
        change = k * mult * (result - p)
        ratings[g.home], ratings[g.away] = rh + change, ra - change
    return pd.Series(out, name="elo_home")


def log_loss(p, won) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    won = np.asarray(won, bool)
    return float(-np.mean(np.where(won, np.log(p), np.log(1 - p))))
