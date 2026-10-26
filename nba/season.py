"""NBA playoff odds: the rest of the season played out thousands of times with the model.

For each simulation:

1. Every remaining regular-season game gets a random result: the model's expected margin (each
   team's players and their expected minutes, its team term, home court) plus the usual
   game-to-game spread (MARGIN_SD). Each team also gets one random "how good are they really"
   shift for the whole simulation (RATING_SD points), because ratings are estimates and rosters
   change. For the current season, players on the injury report are left out of games before
   their expected return date.
2. Conference standings, with ties broken by head-to-head record among the tied teams, then
   conference record, then a coin flip (the NBA's rules have a few more steps in between).
3. The play-in (7th v 8th for the 7th seed; the loser v the winner of 9th v 10th for the 8th),
   then four best-of-seven rounds: 1 v 8, 4 v 5, 3 v 6, 2 v 7, the better seed at home in games
   1, 2, 5 and 7 (in the Finals, the team with more wins).

The result is each team's average wins and its chances of a top-six place (straight into the
playoffs), the playoffs (directly or through the play-in), the top seed, the conference finals,
the Finals and the title. Tested on past seasons with the play-in (from 2020-21): odds made on
1 December, 15 January and 1 March, against what happened. RATING_SD was chosen on 2020-21 and
2021-22 (playoff log loss 0.482 with no shift, 0.402 at 3 points, 0.401 at 4.5). Held-out 2022-23
to 2025-26 (360 team-dates): making the playoffs 0.318 (0.343 with no shift; 0.691 for the
league-wide rate), a top-six place 0.337, the Finals 0.357; teams given 10-30% made the playoffs
18% of the time, 30-50% 38%, 70-90% 76%, over 90% 98%. Average miss on final wins: 3.8.

Usage:
    python -m nba.season              # today's odds, saved to the database (and shown on the website)
    python -m nba.season --record     # ...and to the public record, records/nba_season_odds.csv
    python -m nba.season --test       # how well these odds did on past seasons
"""

import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from db import DB_PATH, SCHEMA_PATH
from nba import live
from nba.model import MARGIN_SD, fit, lineup_strength, load_games, load_players

SIMULATIONS = 10_000
RATING_SD = 3.0             # points per game; chosen on 2020-21 and 2021-22 (python -m nba.season --test)
RECENT_DAYS = 45            # past seasons: a team's lineup is its players from the last 45 days
OUTCOMES = ["top_six", "playoffs", "top_seed", "conference_finals", "finals", "champion"]
PLAY_IN_FROM = 2020         # 2020-21: the play-in's first full season


# ---------------------------------------------------------------- setting up a season

def team_info(conn) -> pd.DataFrame:
    return pd.read_sql("SELECT name, conference, division FROM nba_teams", conn).set_index("name")


def setup(games: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame, season: int, as_of: str,
          rosters: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None) -> dict:
    """Everything the simulation needs: results before as_of, and the model's expected margin for
    every regular-season game from as_of on. rosters/injuries: the live ones (current season);
    without them each team's lineup is its players from its recent games."""
    reg = games[(games["season"] == season) & (games["season_type"] == "regular")]
    known = reg[(reg["status"] == "final") & (reg["date"] < as_of)]
    left = reg[~reg.index.isin(known.index)]
    model = fit(games, players, as_of, season, with_points=False)
    names = sorted(teams.index)
    index = {t: i for i, t in enumerate(names)}
    before = players[players["date"] < as_of]
    latest = live.latest_lines(before) if not before.empty else pd.DataFrame(
        columns=["team", "name", "position", "date", "minutes_after"])
    term = model["teams"]

    def strength(team, on, out) -> float:
        if rosters is not None and not rosters.empty:
            lineup = live.expected_lineup(team, latest, rosters, out, on)
        else:
            recent = latest[(latest["team"] == team) &
                            (pd.to_datetime(latest["date"]) >= pd.Timestamp(as_of) - pd.Timedelta(days=RECENT_DAYS))]
            lineup = pd.DataFrame({"player_id": recent.index, "minutes": recent["minutes_after"].to_numpy()})
        return lineup_strength(model, lineup) + float(term.get(team, 0.0))

    if injuries is not None and not injuries.empty:      # live: who's out depends on the game's date
        cache = {}
        def game_strength(team, on):
            out = frozenset(live.unavailable(injuries, on))
            if (team, out) not in cache:
                cache[(team, out)] = strength(team, on, set(out))
            return cache[(team, out)]
    else:
        fixed = {t: strength(t, as_of, set()) for t in names}
        game_strength = lambda team, on: fixed[team]
    margins = np.array([game_strength(g.home, g.date) - game_strength(g.away, g.date)
                        + (0.0 if g.neutral_site else model["home_adv"]) for g in left.itertuples()])
    base = np.array([game_strength(t, as_of) for t in names])
    return {"season": season, "as_of": as_of, "teams": names, "index": index, "home_adv": model["home_adv"],
            "conference": np.array([teams.at[t, "conference"] for t in names]),
            "strength": base, "known_home": known["home"].map(index).to_numpy(int),
            "known_away": known["away"].map(index).to_numpy(int),
            "known_home_won": (known["home_score"] > known["away_score"]).to_numpy(),
            "left_home": left["home"].map(index).to_numpy(int), "left_away": left["away"].map(index).to_numpy(int),
            "left_margin": margins.astype(float)}


# ---------------------------------------------------------------- simulating

def series_win(p_home: np.ndarray, p_away: np.ndarray) -> np.ndarray:
    """Chance the team with home court wins a best-of-seven (home in games 1, 2, 5, 7), given its
    chance of winning a game at home and away. Works on arrays (one entry per simulation)."""
    home_games = [True, True, False, False, True, False, True]
    states = {(0, 0): np.ones_like(p_home)}
    won = np.zeros_like(p_home)
    for game in range(7):
        p = p_home if home_games[game] else p_away
        new = {}
        for (a, b), prob in states.items():
            if a + b != game:
                continue
            for (da, db, q) in ((1, 0, p), (0, 1, 1 - p)):
                key = (a + da, b + db)
                if key[0] == 4:
                    won = won + prob * q
                elif key[1] < 4:
                    new[key] = new.get(key, 0) + prob * q
        states = new
    return won


def simulate(s: dict, n: int = SIMULATIONS, rating_sd: float = RATING_SD, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = len(s["teams"])
    shift = rng.normal(0, rating_sd, (n, t))
    lh, la = s["left_home"], s["left_away"]
    margin = s["left_margin"] + shift[:, lh] - shift[:, la] + rng.normal(0, MARGIN_SD, (n, len(lh)))
    home = np.concatenate([s["known_home"], lh])
    away = np.concatenate([s["known_away"], la])
    home_won = np.hstack([np.broadcast_to(s["known_home_won"], (n, len(s["known_home"]))), margin > 0])
    wins = np.zeros((n, t))
    conf_wins = np.zeros((n, t))
    h2h = np.zeros((n, t, t))
    same_conf = s["conference"][home] == s["conference"][away]
    for col, (h, a) in enumerate(zip(home, away)):
        w = home_won[:, col]
        wins[:, h] += w
        wins[:, a] += ~w
        h2h[:, h, a] += w
        h2h[:, a, h] += ~w
        if same_conf[col]:
            conf_wins[:, h] += w
            conf_wins[:, a] += ~w
    strength = s["strength"] + shift
    noise = rng.random((n, t))
    out = {k: np.zeros(t) for k in OUTCOMES}

    def game(a, b, neutral=False):
        """a at home (unless neutral): True where a wins, per simulation."""
        diff = strength[np.arange(n), a] - strength[np.arange(n), b] + (0 if neutral else s["home_adv"])
        return rng.random(n) < norm.cdf(diff / MARGIN_SD)

    def series(a, b):
        """a has home court: the winner of each simulation's series."""
        diff = strength[np.arange(n), a] - strength[np.arange(n), b]
        p = series_win(norm.cdf((diff + s["home_adv"]) / MARGIN_SD), norm.cdf((diff - s["home_adv"]) / MARGIN_SD))
        return np.where(rng.random(n) < p, a, b)

    champions = {}
    for conf in ("East", "West"):
        members = np.where(s["conference"] == conf)[0]
        seeds = np.zeros((n, len(members)), int)
        for sim in range(n):
            seeds[sim] = order(members, wins[sim], conf_wins[sim], h2h[sim], noise[sim])
        for k in range(6):
            np.add.at(out["top_six"], seeds[:, k], 1)
        np.add.at(out["top_seed"], seeds[:, 0], 1)
        if s["season"] >= PLAY_IN_FROM:
            win78 = game(seeds[:, 6], seeds[:, 7])
            seed7 = np.where(win78, seeds[:, 6], seeds[:, 7])
            loser78 = np.where(win78, seeds[:, 7], seeds[:, 6])
            seed9_10 = np.where(game(seeds[:, 8], seeds[:, 9]), seeds[:, 8], seeds[:, 9])
            seed8 = np.where(game(loser78, seed9_10), loser78, seed9_10)
        else:
            seed7, seed8 = seeds[:, 6], seeds[:, 7]
        bracket = np.column_stack([seeds[:, :6], seed7, seed8])
        for k in range(8):
            np.add.at(out["playoffs"], bracket[:, k], 1)
        r1 = [series(bracket[:, i], bracket[:, j]) for i, j in ((0, 7), (3, 4), (2, 5), (1, 6))]
        seed_of = lambda team: np.argmax(bracket == team[:, None], axis=1)
        def better_first(a, b):
            swap = seed_of(a) > seed_of(b)
            return np.where(swap, b, a), np.where(swap, a, b)
        semis = [series(*better_first(r1[0], r1[1])), series(*better_first(r1[2], r1[3]))]
        for team in semis:
            np.add.at(out["conference_finals"], team, 1)
        finalist = series(*better_first(semis[0], semis[1]))
        np.add.at(out["finals"], finalist, 1)
        champions[conf] = finalist
    east, west = champions["East"], champions["West"]
    more = wins[np.arange(n), east] + noise[np.arange(n), east] >= wins[np.arange(n), west] + noise[np.arange(n), west]
    winner = series(np.where(more, east, west), np.where(more, west, east))
    np.add.at(out["champion"], winner, 1)
    df = pd.DataFrame({"team": s["teams"], "conference": s["conference"], "wins": wins.mean(axis=0),
                       **{k: v / n for k, v in out.items()}})
    return df.sort_values(["conference", "wins"], ascending=[True, False], ignore_index=True)


def order(members: np.ndarray, wins: np.ndarray, conf_wins: np.ndarray, h2h: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """A conference's teams in standings order for one simulation: wins, then (among teams tied on
    wins) head-to-head record among them, then conference wins, then a coin flip."""
    ranked = members[np.argsort(-wins[members], kind="stable")]
    out = []
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and wins[ranked[j + 1]] == wins[ranked[i]]:
            j += 1
        tied = ranked[i:j + 1]
        if len(tied) > 1:
            sub = h2h[np.ix_(tied, tied)]
            played = sub + sub.T
            h2h_pct = np.where(played.sum(axis=1) > 0, sub.sum(axis=1) / np.maximum(played.sum(axis=1), 1), 0.5)
            tied = tied[np.lexsort((noise[tied], conf_wins[tied], h2h_pct))[::-1]]
        out.extend(tied)
        i = j + 1
    return np.array(out)


# ---------------------------------------------------------------- testing on past seasons

def actual_outcomes(games: pd.DataFrame, season: int, teams: list[str]) -> pd.DataFrame:
    """What really happened: each team's outcomes (1/0), from the play-in and playoff games."""
    post = games[(games["season"] == season) & (games["status"] == "final")]
    playin = post[post["season_type"] == "playin"]
    playoffs = post[post["season_type"] == "playoffs"]
    rounds = {}
    for team in teams:
        mine = playoffs[(playoffs["home"] == team) | (playoffs["away"] == team)]
        rounds[team] = len(set(mine["home"]) | set(mine["away"]) - {team})
    in_playin = set(playin["home"]) | set(playin["away"])
    last = playoffs.sort_values(["date", "id"]).iloc[-1] if not playoffs.empty else None
    champion = (last["home"] if last["home_score"] > last["away_score"] else last["away"]) if last is not None else None
    reg = games[(games["season"] == season) & (games["season_type"] == "regular") & (games["status"] == "final")]
    rows = []
    for team in teams:
        rows.append({"team": team, "top_six": int(rounds[team] > 0 and team not in in_playin),
                     "playoffs": int(rounds[team] > 0), "conference_finals": int(rounds[team] >= 3),
                     "finals": int(rounds[team] >= 4), "champion": int(team == champion)})
    out = pd.DataFrame(rows)
    out["wins"] = [int(((reg["home"] == t) & (reg["home_score"] > reg["away_score"])).sum()
                       + ((reg["away"] == t) & (reg["away_score"] > reg["home_score"])).sum()) for t in teams]
    return out


def checkpoints(season: int) -> list[str]:
    return [f"{season}-12-01", f"{season + 1}-01-15", f"{season + 1}-03-01"]


def season_test(games, players, teams, seasons, n: int = 2000, rating_sd: float = RATING_SD) -> pd.DataFrame:
    rows = []
    for season in seasons:
        truth = actual_outcomes(games, season, sorted(teams.index)).set_index("team")
        played = games[(games["season"] == season) & (games["season_type"] == "regular") & (games["status"] == "final")]
        for as_of in checkpoints(season):
            if (played["date"] < as_of).sum() < 30:
                continue
            odds = simulate(setup(games, players, teams, season, as_of), n=n, rating_sd=rating_sd).set_index("team")
            for team in odds.index:
                rows.append({"season": season, "as_of": as_of, "team": team, "pred_wins": odds.at[team, "wins"],
                             "wins": truth.at[team, "wins"],
                             **{f"p_{k}": odds.at[team, k] for k in OUTCOMES if k != "top_seed"},
                             **{k: truth.at[team, k] for k in OUTCOMES if k != "top_seed"}})
    return pd.DataFrame(rows)


def score(test: pd.DataFrame) -> dict:
    out = {"team_seasons": len(test)}
    for k in ("top_six", "playoffs", "conference_finals", "finals"):
        p = np.clip(test[f"p_{k}"], 1e-4, 1 - 1e-4)
        out[k] = round(float(-np.mean(np.where(test[k] == 1, np.log(p), np.log(1 - p)))), 3)
    champs = test[test["champion"] == 1]
    out["champion"] = round(float(-np.log(np.clip(champs["p_champion"], 1e-4, 1)).mean()), 3)
    out["wins_miss"] = round(float((test["pred_wins"] - test["wins"]).abs().mean()), 2)
    return out


# ---------------------------------------------------------------- saving

RECORD_PATH = Path("records/nba_season_odds.csv")
COLUMNS = ["season", "as_of", "team", "conference", "wins", *OUTCOMES]


def save(conn, odds: pd.DataFrame, season: int, as_of: str) -> None:
    conn.execute("DELETE FROM nba_season_odds WHERE season = ? AND as_of = ?", (season, as_of))
    conn.executemany(f"INSERT INTO nba_season_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     [(season, as_of, r.team, r.conference, float(r.wins), *(float(getattr(r, k)) for k in OUTCOMES))
                      for r in odds.itertuples()])
    conn.commit()


def restore(conn) -> int:
    if not RECORD_PATH.exists():
        return 0
    rows = pd.read_csv(RECORD_PATH, dtype={"as_of": str})
    conn.executemany(f"INSERT OR REPLACE INTO nba_season_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     rows[COLUMNS].itertuples(index=False, name=None))
    conn.commit()
    return len(rows)


def export(conn) -> None:
    RECORD_PATH.parent.mkdir(exist_ok=True)
    pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM nba_season_odds ORDER BY season, as_of, conference, wins DESC", conn) \
      .round(4).to_csv(RECORD_PATH, index=False)


def main(argv: list[str] | None = None, today: str | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games, players, teams = load_games(conn), load_players(conn), team_info(conn)
    if "--test" in argv:
        for label, seasons in (("Tuning seasons 2020-21 and 2021-22", range(2020, 2022)),
                               ("Held-out seasons 2022-23 to 2025-26", range(2022, 2026))):
            print(f"{label}: {score(season_test(games, players, teams, seasons))}")
        return
    restored = restore(conn)
    if restored:
        print(f"Loaded {restored} recorded rows from {RECORD_PATH}.")
    today = today or date.today().isoformat()
    reg = games[(games["season_type"] == "regular")]
    season = int(reg[reg["date"] <= today]["season"].max()) if (reg["date"] <= today).any() else int(reg["season"].max())
    if reg[(reg["season"] == season) & (reg["status"] == "scheduled")].empty:
        print(f"The {season}-{(season + 1) % 100:02d} regular season is over; no playoff odds to update.")
        conn.close()
        return
    rosters, injuries = live.load(conn)
    odds = simulate(setup(games, players, teams, season, today, rosters, injuries))
    save(conn, odds, season, today)
    if "--record" in argv:
        export(conn)
    conn.close()
    print(f"NBA playoff odds for {season}-{(season + 1) % 100:02d} as of {today} ({SIMULATIONS:,} simulations):")
    print(odds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
