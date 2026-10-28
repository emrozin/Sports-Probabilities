"""Playoff odds: the rest of an NFL season played out thousands of times with the model.

For each simulation:

1. Every remaining regular-season game gets a random result from the model: the expected margin
   (team ratings, home advantage and the expected quarterbacks) plus the usual game-to-game
   spread (MARGIN_SD points). On top of that, each team gets one random "how good are they
   really" shift for the whole simulation (RATING_SD points), because ratings are estimates:
   without it, the odds come out too sure of themselves.
2. Standings, division winners and playoff seeds are worked out with the main NFL tiebreakers:
   head-to-head among the tied teams, then division record (for division ties) or conference
   record, then point differential, then a coin flip. (The full NFL rules have more steps, like
   common opponents and strength of victory; these few settle almost every real tie.)
3. The playoffs are played out the same way: 7 teams per conference (6 before 2020), the top
   seed gets a bye, higher seeds play at home, and the Super Bowl is on a neutral field.

The result is each team's average wins and its chance of winning its division, making the
playoffs, getting the top seed, reaching the Super Bowl and winning it.

Tested on past seasons: odds made after weeks 4, 8 and 12 of each season, against who really
made the playoffs. RATING_SD was chosen on 2010-2017 (playoff log loss 0.397 with no shift,
0.386 at 3-4 points, 0.394 at 7.5). Held-out 2018-2025 (768 team-seasons): playoff log loss 0.403
(0.416 with no shift; 0.681 for the league-wide rate), division 0.347, and the chances are
calibrated: teams given 20-40% made it 35% of the time, 40-60% 52%, 60-80% 67%, 80-95% 83%.

Usage:
    python -m nfl.season              # today's odds, saved to the database (and shown on the website)
    python -m nfl.season --record     # ...and to the public record, records/nfl_season_odds.csv
    python -m nfl.season --history    # also fill in this season's odds after each week played so far
    python -m nfl.season --test       # how well these odds did on past seasons (tuning and held-out)
"""

import sqlite3
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from db import DB_PATH, SCHEMA_PATH
from nfl.elo import load_nfl_games
from nfl.model import MARGIN_SD, TRAINING_YEARS, USUAL_STARTER_GAMES, add_qb_changes, fit, predict, qb_value
from nfl.teams import TEAMS, conference, division

SIMULATIONS = 10_000
RATING_SD = 3.5            # points; tuned on 2010-2017 (python -m nfl.season --test), see the docstring
FOURTEEN_TEAMS_FROM = 2020  # 7 playoff teams per conference from 2020; 6 before
OUTCOMES = ["division", "playoffs", "top_seed", "conference", "champion"]


# ---------------------------------------------------------------- setting up a season

def starters_as_of(games: pd.DataFrame, as_of: str) -> tuple[dict, Counter]:
    """Each team's usual starting quarterback before a date (most frequent of its last 8 starts)
    and every quarterback's career starts before it."""
    done = games[(games["status"] == "final") & (games["date"] < as_of)].sort_values("date")
    history, starts = {}, Counter()
    for side in ("home", "away"):
        for team, qb in zip(done[side], done[f"{side}_qb"]):
            if isinstance(qb, str) and qb:
                starts[qb] += 1
    for g in done.itertuples():
        for side in ("home", "away"):
            qb = getattr(g, f"{side}_qb")
            if isinstance(qb, str) and qb:
                history.setdefault(getattr(g, side), []).append(qb)
    usual = {team: Counter(qbs[-USUAL_STARTER_GAMES:]).most_common(1)[0][0] for team, qbs in history.items()}
    return usual, starts


def setup(games: pd.DataFrame, season: int, as_of: str) -> dict:
    """Everything a simulation needs, using only what was known before `as_of`.

    games: load_nfl_games + add_qb_changes. Games before as_of count with their real results;
    everything later is simulated. For a past season (a test), later games are simulated even
    though their results are known, and with each team's usual starter as of as_of rather than
    whoever really started, so the test knows no more than a live forecast would."""
    played = games[games["status"] == "final"]
    start = (pd.Timestamp(as_of) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    model = fit(played[(played["date"] >= start) & (played["date"] < as_of)], as_of)
    usual, starts = starters_as_of(games, as_of)

    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    teams = sorted(set(reg["home"]) | set(reg["away"]))
    index = {t: i for i, t in enumerate(teams)}
    known = reg[(reg["status"] == "final") & (reg["date"] < as_of)]
    remaining = reg.drop(known.index)

    def margin(g) -> float:
        if g.status == "scheduled":   # live: the listed starter this week, the usual one after that
            hq, aq = g.home_qb_expected, g.away_qb_expected
        else:                         # a test: only what was known at as_of
            hq, aq = usual.get(g.home), usual.get(g.away)
        hq = hq if isinstance(hq, str) else usual.get(g.home)
        aq = aq if isinstance(aq, str) else usual.get(g.away)
        return predict(model, g.home, g.away, bool(g.neutral_site), home_qb=hq, away_qb=aq,
                       home_qb_starts=starts.get(hq, np.nan), away_qb_starts=starts.get(aq, np.nan))["margin"]

    ratings = model["ratings"]
    strength = np.array([
        (ratings.at[t, "offense"] + ratings.at[t, "defense"] if t in ratings.index else 0.0)
        + qb_value(model, usual.get(t), starts.get(usual.get(t), np.nan)) for t in teams])
    return {
        "season": season, "as_of": as_of, "teams": teams,
        "conference": np.array([conference(t) for t in teams]),
        "division": np.array([division(t) for t in teams]),
        # team positions as integers, even when there are no games (an empty column has no type of its own)
        "known": (known["home"].map(index).to_numpy(dtype=int), known["away"].map(index).to_numpy(dtype=int),
                  (known["home_score"] - known["away_score"]).to_numpy(dtype=float)),
        "remaining": (remaining["home"].map(index).to_numpy(dtype=int), remaining["away"].map(index).to_numpy(dtype=int),
                      np.array([margin(g) for g in remaining.itertuples()], dtype=float)),
        "strength": strength, "home_adv": float(model["home_adv"]),
        "playoff_teams": 7 if season >= FOURTEEN_TEAMS_FROM else 6,
    }


# ---------------------------------------------------------------- simulating

def simulate(s: dict, n: int = SIMULATIONS, rating_sd: float = RATING_SD, seed: int = 0) -> pd.DataFrame:
    """Play out the rest of the season n times. One row per team: average wins and the chance of
    each outcome in OUTCOMES."""
    rng = np.random.default_rng(seed)
    teams, T = s["teams"], len(s["teams"])
    shift = rng.normal(0, rating_sd, (n, T))          # how good each team "really" is, per simulation

    # regular season: known results plus simulated ones
    kh, ka, km = s["known"]
    rh, ra, rm = s["remaining"]
    margins = np.concatenate([np.broadcast_to(km, (n, len(km))),
                              rm + shift[:, rh] - shift[:, ra] + rng.normal(0, MARGIN_SD, (n, len(rm)))], axis=1)
    home, away = np.concatenate([kh, rh]), np.concatenate([ka, ra])
    result = np.where(margins > 0, 1.0, np.where(margins < 0, 0.0, 0.5))   # home team's share of a win

    wins = np.zeros((n, T))
    point_diff = np.zeros((n, T))
    games_played = np.zeros(T)
    h2h = np.zeros((n, T, T))                         # h2h[sim, a, b]: a's wins over b
    same_div = s["division"][home] == s["division"][away]
    same_conf = s["conference"][home] == s["conference"][away]
    div_w, div_g = np.zeros((n, T)), np.zeros(T)
    conf_w, conf_g = np.zeros((n, T)), np.zeros(T)
    for g in range(len(home)):
        h, a, r = home[g], away[g], result[:, g]
        wins[:, h] += r
        wins[:, a] += 1 - r
        point_diff[:, h] += margins[:, g]
        point_diff[:, a] -= margins[:, g]
        games_played[[h, a]] += 1
        h2h[:, h, a] += r
        h2h[:, a, h] += 1 - r
        if same_div[g]:
            div_w[:, h] += r
            div_w[:, a] += 1 - r
            div_g[[h, a]] += 1
        if same_conf[g]:
            conf_w[:, h] += r
            conf_w[:, a] += 1 - r
            conf_g[[h, a]] += 1
    pct = wins / np.maximum(games_played, 1)
    div_pct = div_w / np.maximum(div_g, 1)
    conf_pct = conf_w / np.maximum(conf_g, 1)
    coin = rng.random((n, T))

    counts = {k: np.zeros(T) for k in OUTCOMES}
    playoff_draws = rng.random((n, 16))
    seeds_n = s["playoff_teams"]
    divisions = sorted(set(s["division"]))
    for i in range(n):
        def order(group: list[int], within_division: bool) -> list[int]:
            group = sorted(group, key=lambda t: -pct[i, t])
            out, k = [], 0
            while k < len(group):
                block = [t for t in group if pct[i, t] == pct[i, group[k]]]
                if len(block) > 1:
                    def h2h_pct(t):
                        others = [o for o in block if o != t]
                        w = h2h[i, t, others].sum()
                        played = w + h2h[i, others, t].sum()
                        return w / played if played else 0.5
                    record = div_pct if within_division else conf_pct
                    block.sort(key=lambda t: (-h2h_pct(t), -record[i, t], -conf_pct[i, t], -point_diff[i, t], coin[i, t]))
                out += block
                k += len(block)
            return out

        playoff_field = {}
        for conf in ("AFC", "NFC"):
            winners, others = [], []
            for d in divisions:
                if not d.startswith(conf):
                    continue
                ranked = order([t for t in range(T) if s["division"][t] == d], True)
                winners.append(ranked[0])
                others += ranked[1:]
                counts["division"][ranked[0]] += 1
            seeds = order(winners, False) + order(others, False)[:seeds_n - 4]
            for t in seeds:
                counts["playoffs"][t] += 1
            counts["top_seed"][seeds[0]] += 1
            playoff_field[conf] = seeds

        # the playoffs
        draw = iter(playoff_draws[i])

        def play(a: int, b: int, neutral: bool = False) -> int:
            """a hosts b; returns the winner."""
            m = s["strength"][a] + shift[i, a] - s["strength"][b] - shift[i, b] + (0 if neutral else s["home_adv"])
            return a if next(draw) < norm.cdf(m / MARGIN_SD) else b

        champs = {}
        for conf, seeds in playoff_field.items():
            byes = 1 if seeds_n == 7 else 2
            alive = seeds[:byes]
            for k in range(byes, byes + (seeds_n - byes) // 2):
                alive.append(play(seeds[k], seeds[seeds_n - 1 - (k - byes)]))
            alive.sort(key=seeds.index)                       # reseed: best seed plays the lowest
            semi = [play(alive[0], alive[3]), play(alive[1], alive[2])]
            semi.sort(key=seeds.index)
            champs[conf] = play(semi[0], semi[1])
            counts["conference"][champs[conf]] += 1
        counts["champion"][play(champs["AFC"], champs["NFC"], neutral=True)] += 1

    out = pd.DataFrame({"team": teams, "wins": wins.mean(axis=0),
                        **{k: v / n for k, v in counts.items()}})
    return out.sort_values(["playoffs", "wins"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- testing on past seasons

def actual_outcomes(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """What really happened: wins, and who won a division, made the playoffs, reached and won the
    Super Bowl (read from the playoff games: division winners host wild card games or have a bye)."""
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    post = games[(games["season"] == season) & (games["game_type"] != "REG") & (games["status"] == "final")]
    teams = sorted(set(reg["home"]) | set(reg["away"]))
    wins = {t: 0.0 for t in teams}
    for g in reg.itertuples():
        share = 1.0 if g.home_score > g.away_score else 0.0 if g.home_score < g.away_score else 0.5
        wins[g.home] += share
        wins[g.away] += 1 - share
    wild_card = post[post["game_type"] == "WC"]
    divisional = post[post["game_type"] == "DIV"]
    byes = (set(divisional["home"]) | set(divisional["away"])) - (set(wild_card["home"]) | set(wild_card["away"]))
    division_winners = set(wild_card["home"]) | byes
    winner = lambda g: g.home if g.home_score > g.away_score else g.away
    in_super_bowl = set(post.loc[post["game_type"] == "SB", "home"]) | set(post.loc[post["game_type"] == "SB", "away"])
    champion = {winner(g) for g in post[post["game_type"] == "SB"].itertuples()}
    in_playoffs = set(post["home"]) | set(post["away"])
    return pd.DataFrame({
        "team": teams, "wins": [wins[t] for t in teams],
        "division": [t in division_winners for t in teams], "playoffs": [t in in_playoffs for t in teams],
        "conference": [t in in_super_bowl for t in teams], "champion": [t in champion for t in teams],
    })


def season_test(games: pd.DataFrame, seasons, weeks=(4, 8, 12), n: int = 2000, rating_sd: float = RATING_SD) -> pd.DataFrame:
    """Playoff and division odds after `weeks` of each season, next to what happened."""
    rows = []
    for season in seasons:
        reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
        for week in weeks:
            upcoming = reg[reg["week"] > week]
            if upcoming.empty:
                continue
            as_of = upcoming["date"].min()
            odds = simulate(setup(games, season, as_of), n=n, rating_sd=rating_sd, seed=season * 100 + week)
            actual = actual_outcomes(games, season)
            merged = odds.merge(actual, on="team", suffixes=("", "_actual"))
            merged["season"], merged["week"] = season, week
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def log_loss(p, happened) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.005, 0.995)
    y = np.asarray(happened, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(test: pd.DataFrame) -> dict:
    """Log loss of the playoff and division chances, and wins' average miss."""
    return {"playoffs": log_loss(test["playoffs"], test["playoffs_actual"]),
            "division": log_loss(test["division"], test["division_actual"]),
            "wins_miss": float((test["wins"] - test["wins_actual"]).abs().mean())}


# ---------------------------------------------------------------- saving the odds

RECORD_PATH = Path("records/nfl_season_odds.csv")
COLUMNS = ["season", "as_of", "team", "wins", *OUTCOMES]


def save(conn, odds: pd.DataFrame, season: int, as_of: str) -> None:
    conn.execute("DELETE FROM nfl_season_odds WHERE season = ? AND as_of = ?", (season, as_of))
    conn.executemany(
        f"INSERT INTO nfl_season_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
        [(season, as_of, r.team, float(r.wins), *(float(getattr(r, k)) for k in OUTCOMES)) for r in odds.itertuples()])
    conn.commit()


def restore(conn) -> int:
    """Copy the recorded odds into the database (the database isn't in the repository)."""
    if not RECORD_PATH.exists():
        return 0
    rows = pd.read_csv(RECORD_PATH, dtype={"as_of": str})
    conn.executemany(f"INSERT OR REPLACE INTO nfl_season_odds ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     rows[COLUMNS].itertuples(index=False, name=None))
    conn.commit()
    return len(rows)


def export(conn) -> None:
    RECORD_PATH.parent.mkdir(exist_ok=True)
    pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM nfl_season_odds ORDER BY season, as_of, playoffs DESC", conn) \
      .round(4).to_csv(RECORD_PATH, index=False)


def current_season(games: pd.DataFrame, today: str) -> int:
    reg = games[games["game_type"] == "REG"]
    started = reg[reg["date"] <= today]
    return int(started["season"].max()) if not started.empty else int(reg["season"].max())


def week_starts(games: pd.DataFrame, season: int, before: str) -> list[str]:
    """The first day of each regular-season week of a season that has started before a date: the
    days the odds would have been made after each week."""
    reg = games[(games["season"] == season) & (games["game_type"] == "REG")]
    firsts = reg.groupby("week")["date"].min().sort_values()
    return [d for d in firsts if d < before]


def main(argv: list[str] | None = None, today: str | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = add_qb_changes(load_nfl_games(conn))
    if "--test" in argv:
        for label, seasons in (("Tuning seasons 2010-2017", range(2010, 2018)), ("Held-out seasons 2018-2025", range(2018, 2026))):
            print(f"{label}: {score(season_test(games, seasons))}")
        return
    restored = restore(conn)
    if restored:
        print(f"Loaded {restored} recorded rows from {RECORD_PATH}.")
    today = today or date.today().isoformat()
    season = current_season(games, today)
    if games[(games["season"] == season) & (games["game_type"] == "REG") & (games["status"] == "scheduled")].empty:
        print(f"The {season} regular season is over; no playoff odds to update.")
        conn.close()
        return
    if "--history" in argv:
        for as_of in week_starts(games, season, today):
            save(conn, simulate(setup(games, season, as_of)), season, as_of)
            print(f"  odds as of {as_of} saved")
    odds = simulate(setup(games, season, today))
    save(conn, odds, season, today)
    if "--record" in argv:
        export(conn)
    conn.close()
    print(f"Playoff odds for {season} as of {today} ({SIMULATIONS:,} simulations):")
    print(odds.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
