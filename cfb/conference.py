"""Conference title odds: the rest of a college football season played out thousands of times.

For each simulation:

1. Every remaining regular-season game gets a random result from the model (team ratings and home
   field, plus the usual game-to-game spread of MARGIN_SD points). Each team also gets one random
   "how good are they really" shift for the whole simulation (RATING_SD points), because ratings
   are estimates; without it the odds come out too sure of themselves (the same idea as the NFL
   playoff odds, nfl/season.py).
2. Each conference's top two teams by conference winning percentage meet in its title game. The
   Sun Belt still has divisions, so there it's the East winner against the West winner.
3. Ties are settled roughly the way conferences do it: head-to-head record among the tied teams,
   then overall record, then the team's rating (most conferences end with a computer ranking or
   the playoff committee's ranking), then a coin flip. The real rules have more steps (common
   opponents, opponents' records), but these settle most ties the same way.
4. The title game is played on a neutral field for the power conferences and the MAC, and at the
   higher seed's stadium for the others.

Tested on past seasons, but only from when each conference stopped using divisions (so the
top-two rule applies): odds made before weeks 1, 5, 9 and 12, against who really won the title game.
RATING_SD was chosen on 2022-2023 (44 conference races; champion log loss 1.70 with no shift, 1.58
at 8 points, 1.64 at 15). Held-out 2024-2025 (72 races): champion log loss 2.01 (2.22 with no
shift; 2.66 picking a team at random), so the eventual champion got about a 13% chance on average
(7% at random): 8% before the season, 23% before week 12. Reaching the title game: 0.261 (0.399 at
random). The chances are calibrated: teams given 5-15% won 8% of the time, 15-30% 20%, 30-50% 34%.

Usage:
    python -m cfb.conference              # today's odds, saved to the database (and shown on the website)
    python -m cfb.conference --record     # ...and to the public record, records/cfb_conference_odds.csv
    python -m cfb.conference --history    # also fill in this season's odds before each week played so far
    python -m cfb.conference --test       # how well these odds did on past seasons
"""

import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb.model import MARGIN_SD, current_fit, load_games, predict, team_strength
from db import DB_PATH, SCHEMA_PATH

SIMULATIONS = 10_000
RATING_SD = 8.0             # points; chosen on 2022-2023 (python -m cfb.conference --test)
INDEPENDENTS = "FBS Independents"
MIN_TEAMS = 6               # smaller "conferences" (the two-team Pac-12 of 2024-25) have no title game
NEUTRAL_TITLE_GAMES = {"ACC", "Big 12", "Big Ten", "SEC", "Pac-12", "Mid-American"}
SUN_BELT_EAST = {"App State", "Coastal Carolina", "Georgia Southern", "Georgia State", "James Madison", "Marshall",
                 "Old Dominion"}
# The last season each conference picked its title game teams by division (None: still does)
DIVISIONS_UNTIL = {"ACC": 2022, "American Athletic": 2019, "Big 12": 2010, "Big Ten": 2023, "Conference USA": 2022,
                   "Mid-American": 2023, "Mountain West": 2022, "Pac-12": 2021, "SEC": 2023, "Sun Belt": None}
SUN_BELT_DIVISIONS_FROM = 2022   # the East/West split above matches the membership from 2022 on
OUTCOMES = ["title_game", "champion"]


def has_title_game_rule(conference: str, season: int) -> bool:
    """Whether the simulation's rule (top two, or the Sun Belt's divisions) applies that season."""
    if conference == "Sun Belt":
        return season >= SUN_BELT_DIVISIONS_FROM
    last = DIVISIONS_UNTIL.get(conference)
    return last is not None and season > last


def is_title_game(games: pd.DataFrame) -> pd.Series:
    return (games["notes"].fillna("").str.contains("Championship") & (games["season_type"] == "regular")
            & (games["home_conference"] == games["away_conference"]))


def members(games: pd.DataFrame, season: int) -> dict[str, list[str]]:
    """Each conference's teams that season (a team's conference as listed in its games)."""
    reg = games[(games["season"] == season) & (games["season_type"] == "regular")]
    sides = pd.concat([reg[["home", "home_conference", "home_division"]].set_axis(["team", "conference", "division"], axis=1),
                       reg[["away", "away_conference", "away_division"]].set_axis(["team", "conference", "division"], axis=1)])
    sides = sides[(sides["division"] == "fbs") & sides["conference"].notna() & (sides["conference"] != INDEPENDENTS)]
    listed = sides.groupby("team")["conference"].agg(lambda c: c.mode().iloc[0])
    out = {}
    for conf, teams in listed.groupby(listed):
        if len(teams) >= MIN_TEAMS:
            out[conf] = sorted(teams.index)
    return out


# ---------------------------------------------------------------- setting up a season

def setup(games: pd.DataFrame, season: int, as_of: str) -> dict:
    """Everything the simulation needs: the conferences, the results known before as_of, and the
    model's expected margin for every game left."""
    confs = members(games, season)
    conf_of = {t: c for c, ts in confs.items() for t in ts}
    reg = games[(games["season"] == season) & (games["season_type"] == "regular")]
    titles = reg[is_title_game(reg)]
    reg = reg[~is_title_game(reg) & (reg["home"].isin(conf_of) | reg["away"].isin(conf_of))]
    known = reg[(reg["status"] == "final") & (reg["date"] < as_of)]
    left = reg[~reg.index.isin(known.index)]
    model = current_fit(games, as_of, season)
    margins = np.array([predict(model, g.home, g.away, bool(g.neutral_site))["margin"] for g in left.itertuples()])
    teams = sorted(set(reg["home"]) | set(reg["away"]))
    index = {t: i for i, t in enumerate(teams)}
    rating = np.array([team_strength(model, t) for t in teams])
    rating = np.where(np.isnan(rating), np.nanmin(rating) if np.isfinite(rating).any() else 0.0, rating)
    decided = titles[(titles["status"] == "final") & (titles["date"] < as_of)]
    winners = {}
    for g in decided.itertuples():
        winners[g.home_conference] = (g.home if g.home_score > g.away_score else g.away, {g.home, g.away})
    return {"season": season, "as_of": as_of, "confs": confs, "conf_of": conf_of, "teams": teams, "index": index,
            "rating": rating, "home_adv": model["home_adv"],
            "known_home": np.array([index[t] for t in known["home"]], int),
            "known_away": np.array([index[t] for t in known["away"]], int),
            "known_home_won": (known["home_score"] > known["away_score"]).to_numpy(),
            "left_home": np.array([index[t] for t in left["home"]], int),
            "left_away": np.array([index[t] for t in left["away"]], int),
            "left_margin": margins.astype(float), "decided": winners}


# ---------------------------------------------------------------- simulating

def simulate(s: dict, n: int = SIMULATIONS, rating_sd: float = RATING_SD, seed: int = 0) -> pd.DataFrame:
    """Each conference team's average conference and overall wins, and its chance of reaching and
    winning the title game."""
    rng = np.random.default_rng(seed)
    n_teams = len(s["teams"])
    shift = rng.normal(0, rating_sd, (n, n_teams))
    lh, la = s["left_home"], s["left_away"]
    sim_margin = s["left_margin"] + shift[:, lh] - shift[:, la] + rng.normal(0, MARGIN_SD, (n, len(lh)))
    left_home_won = sim_margin > 0                                                        # n x games

    conf_id = np.array([-1] * n_teams)
    names = list(s["confs"])
    for k, c in enumerate(names):
        for t in s["confs"][c]:
            conf_id[s["index"][t]] = k
    home = np.concatenate([s["known_home"], lh])
    away = np.concatenate([s["known_away"], la])
    won = np.hstack([np.broadcast_to(s["known_home_won"], (n, len(s["known_home"]))), left_home_won])   # n x all games
    same_conf = (conf_id[home] >= 0) & (conf_id[home] == conf_id[away])

    def tally(mask):
        wins, games = np.zeros((n, n_teams)), np.zeros(n_teams)
        np.add.at(games, home[mask], 1)
        np.add.at(games, away[mask], 1)
        hw = won[:, mask]
        for col, (h, a) in enumerate(zip(home[mask], away[mask])):
            wins[:, h] += hw[:, col]
            wins[:, a] += ~hw[:, col]
        return wins, games

    conf_wins, conf_games = tally(same_conf)
    all_wins, all_games = tally(np.ones(len(home), bool))
    conf_pct = conf_wins / np.maximum(conf_games, 1)
    all_pct = all_wins / np.maximum(all_games, 1)

    title_game = np.zeros(n_teams)
    champion = np.zeros(n_teams)
    conf_games_idx = {k: np.where(same_conf & (conf_id[home] == k))[0] for k in range(len(names))}
    tiebreak_noise = rng.random((n, n_teams))
    true_rating = s["rating"] + shift

    def best(sim: int, group: np.ndarray) -> int:
        """The team that comes first among `group` (indexes) in simulation `sim`."""
        pct = conf_pct[sim, group]
        tied = group[pct == pct.max()]
        if len(tied) == 1:
            return int(tied[0])
        k = conf_id[tied[0]]
        g = conf_games_idx[k]
        in_group = np.isin(home[g], tied) & np.isin(away[g], tied)
        h, a, w = home[g][in_group], away[g][in_group], won[sim, g][in_group]
        h2h_w = np.array([((h == t) & w).sum() + ((a == t) & ~w).sum() for t in tied], float)
        h2h_g = np.array([((h == t) | (a == t)).sum() for t in tied], float)
        h2h = np.where(h2h_g > 0, h2h_w / np.maximum(h2h_g, 1), 0.5)
        key = np.lexsort((tiebreak_noise[sim, tied], s["rating"][tied], all_pct[sim, tied], h2h))
        return int(tied[key[-1]])

    for k, c in enumerate(names):
        group = np.array([s["index"][t] for t in s["confs"][c]])
        if c in s["decided"]:
            winner, pair = s["decided"][c]
            for t in pair:
                title_game[s["index"][t]] += n
            champion[s["index"][winner]] += n
            continue
        sun_belt = c == "Sun Belt"
        east = np.array([s["index"][t] for t in s["confs"][c] if t in SUN_BELT_EAST]) if sun_belt else None
        west = np.array([s["index"][t] for t in s["confs"][c] if t not in SUN_BELT_EAST]) if sun_belt else None
        for sim in range(n):
            if sun_belt and len(east) and len(west):
                first, second = best(sim, east), best(sim, west)
                if conf_pct[sim, second] > conf_pct[sim, first]:
                    first, second = second, first
            else:
                first = best(sim, group)
                second = best(sim, group[group != first])
            title_game[[first, second]] += 1
            edge = 0.0 if c in NEUTRAL_TITLE_GAMES else s["home_adv"]
            margin = true_rating[sim, first] - true_rating[sim, second] + edge
            champion[first if rng.random() < norm.cdf(margin / MARGIN_SD) else second] += 1

    rows = []
    for c in names:
        for t in s["confs"][c]:
            i = s["index"][t]
            rows.append({"team": t, "conference": c, "conf_wins": conf_wins[:, i].mean(), "conf_games": conf_games[i],
                         "wins": all_wins[:, i].mean(), "games": all_games[i],
                         "title_game": title_game[i] / n, "champion": champion[i] / n})
    return pd.DataFrame(rows).sort_values(["conference", "champion", "conf_wins"], ascending=[True, False, False],
                                          ignore_index=True)


# ---------------------------------------------------------------- testing on past seasons

def actual_outcomes(games: pd.DataFrame, season: int) -> dict[str, tuple[str, set]]:
    """Each conference's real champion and title game teams that season."""
    reg = games[(games["season"] == season) & (games["status"] == "final")]
    out = {}
    for g in reg[is_title_game(reg)].itertuples():
        out[g.home_conference] = (g.home if g.home_score > g.away_score else g.away, {g.home, g.away})
    return out


def checkpoints(games: pd.DataFrame, season: int, weeks=(1, 5, 9, 12)) -> list[str]:
    """The first day of each of these regular-season weeks: odds made before that week's games."""
    reg = games[(games["season"] == season) & (games["season_type"] == "regular")]
    firsts = reg.groupby("week")["date"].min()
    return [firsts[w] for w in weeks if w in firsts.index]


def season_test(games: pd.DataFrame, seasons, n: int = 2000, rating_sd: float = RATING_SD) -> pd.DataFrame:
    rows = []
    for season in seasons:
        truth = actual_outcomes(games, season)
        for as_of in checkpoints(games, season):
            odds = simulate(setup(games, season, as_of), n=n, rating_sd=rating_sd)
            for c, group in odds.groupby("conference"):
                if c not in truth or not has_title_game_rule(c, season):
                    continue
                winner, pair = truth[c]
                for r in group.itertuples():
                    rows.append({"season": season, "as_of": as_of, "conference": c, "team": r.team, "teams": len(group),
                                 "champion": r.champion, "won": r.team == winner,
                                 "title_game": r.title_game, "played": r.team in pair})
    return pd.DataFrame(rows)


def score(test: pd.DataFrame) -> dict:
    """Log loss of the champion (the chance given to the team that won, per conference and date,
    against picking a team at random) and of reaching the title game (yes/no per team)."""
    winners = test[test["won"]]
    champ = float(-np.log(np.clip(winners["champion"], 1e-4, 1)).mean())
    uniform = float(np.log(winners["teams"]).mean())
    p = np.clip(test["title_game"], 1e-4, 1 - 1e-4)
    game = float(-(np.where(test["played"], np.log(p), np.log(1 - p))).mean())
    base_rate = (2 / test["teams"]).clip(1e-4, 1 - 1e-4)
    game_base = float(-(np.where(test["played"], np.log(base_rate), np.log(1 - base_rate))).mean())
    return {"races": len(winners), "champion": round(champ, 3), "champion_at_random": round(uniform, 3),
            "title_game": round(game, 3), "title_game_at_random": round(game_base, 3)}


# ---------------------------------------------------------------- saving

RECORD_PATH = Path("records/cfb_conference_odds.csv")
COLUMNS = ["season", "as_of", "conference", "team", "conf_wins", "conf_games", "wins", "games", *OUTCOMES]


def save(conn, odds: pd.DataFrame, season: int, as_of: str) -> None:
    conn.execute("DELETE FROM cfb_conference_odds WHERE season = ? AND as_of = ?", (season, as_of))
    conn.executemany(f"INSERT INTO cfb_conference_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     [(season, as_of, r.conference, r.team, float(r.conf_wins), int(r.conf_games), float(r.wins),
                       int(r.games), float(r.title_game), float(r.champion)) for r in odds.itertuples()])
    conn.commit()


def restore(conn) -> int:
    """Copy the recorded odds into the database (the database isn't in the repository)."""
    if not RECORD_PATH.exists():
        return 0
    rows = pd.read_csv(RECORD_PATH, dtype={"as_of": str})
    conn.executemany(f"INSERT OR REPLACE INTO cfb_conference_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     rows[COLUMNS].itertuples(index=False, name=None))
    conn.commit()
    return len(rows)


def export(conn) -> None:
    RECORD_PATH.parent.mkdir(exist_ok=True)
    pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM cfb_conference_odds ORDER BY season, as_of, conference, champion DESC",
                conn).round(4).to_csv(RECORD_PATH, index=False)


def current_season(games: pd.DataFrame, today: str) -> int:
    started = games[(games["season_type"] == "regular") & (games["date"] <= today)]
    return int(started["season"].max()) if not started.empty else int(games["season"].max())


def main(argv: list[str] | None = None, today: str | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    if "--test" in argv:
        for label, seasons in (("Tuning seasons 2022-2023", range(2022, 2024)), ("Held-out seasons 2024-2025", range(2024, 2026))):
            print(f"{label}: {score(season_test(games, seasons))}")
        return
    restored = restore(conn)
    if restored:
        print(f"Loaded {restored} recorded rows from {RECORD_PATH}.")
    today = today or date.today().isoformat()
    season = current_season(games, today)
    reg = games[(games["season"] == season) & (games["season_type"] == "regular")]
    if reg[is_title_game(reg) & (reg["status"] == "scheduled")].empty and reg[reg["status"] == "scheduled"].empty:
        print(f"The {season} regular season is over; no conference odds to update.")
        conn.close()
        return
    if "--history" in argv:
        for as_of in [d for d in reg.groupby("week")["date"].min().sort_values() if d < today]:
            save(conn, simulate(setup(games, season, as_of)), season, as_of)
            print(f"  odds as of {as_of} saved")
    odds = simulate(setup(games, season, today))
    save(conn, odds, season, today)
    if "--record" in argv:
        export(conn)
    conn.close()
    print(f"Conference title odds for {season} as of {today} ({SIMULATIONS:,} simulations):")
    print(odds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
