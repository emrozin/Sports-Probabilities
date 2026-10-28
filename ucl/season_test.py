"""How much uncertainty should the season simulation add? python -m ucl.season_test [leagues] [held_out]

The simulation (ucl/season.py) can shift every club's ratings by a random amount in each simulated
season (SPREAD), because the ratings are estimates; with no shift it treats them as exact and may be
too sure of the favorites. Two tests, scored by log loss on what happened:

1. Knockout stages: from the morning of each season's round of 16 (with the real pairings), the rest
   is played out 10,000 times; each club's chances of reaching the quarter-finals, semi-finals and
   final and of winning are scored. The quarter- and semi-final pairings are drawn at random in each
   simulation (the open draw of the seasons before 2024-25). Champions League and Europa League
   2016-17 to 2020-21 for tuning; all three cups 2021-22 to 2025-26 held out.
2. League phases (2024-25 and 2025-26, all three cups): from before the first match day and from
   halfway, the whole season is simulated with ucl/season.py; each club's chances of the top 8, the
   top 24, the quarter-finals and the trophy are scored. These are held-out only (the format began
   in 2024-25).

History: the first version of this test (Champions League knockouts only) found a big shift (0.4)
best on 2016-17 to 2020-21 and no shift best on 2021-22 to 2025-26. This version, with twice the
knockout data, the league phases and the squad-value model, found (log loss):

    spread                          0.0     0.1     0.2     0.3     0.4     0.6
    knockouts 2016-2020 (tuning)    0.4061  0.4057  0.4057  0.4073  0.4095  0.4177
    knockouts 2021-2025             0.3709  0.3716  0.3755  0.3796  0.3867  0.3996
    league phases 2024-2025         0.3032  0.3030  0.3028  0.3052  0.3095  0.3210

Small shifts make almost no difference and big ones hurt: the simulation's chances are about as
confident as they should be, so no shift is used for the cups.

3. The big leagues (La Liga, Bundesliga, Serie A, Ligue 1): from a quarter of the way and halfway
   through each season, the rest is simulated and each club's chances of the title, the top four and
   the bottom three are scored on how the season ended (python -m ucl.season_test leagues [held_out]).
   A league season has 20 to 30 games left per club, so an error in a rating adds up more than in a
   cup. Log loss:

    spread                          0.0     0.05    0.10    0.15    0.20    0.30
    leagues 2016-2020 (tuning)      0.1430  0.1427  0.1417  0.1428  0.1452  0.1534
    leagues 2021-2025 (held out)    0.1690  0.1684  0.1665  0.1661  0.1671  0.1738

   0.10 was best on the tuning seasons and beat no shift on the held-out ones, so the league
   simulations use it (ucl/season.py LEAGUE_SPREAD).
"""

import sqlite3
import sys

import numpy as np
import pandas as pd

from db import DB_PATH
from ucl.model import fit, load_games, load_values, window
from ucl.season import Rates, play_final, play_tie, simulate, simulate_league

SPREADS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
TUNING, HELD_OUT = (2016, 2020), (2021, 2025)
ROUNDS = ["quarterfinals", "semifinals", "final", "champion"]
CUPS = {"UCL": "uefa.champions", "UEL": "uefa.europa", "UECL": "uefa.europa.conf"}


def round_of_16(games: pd.DataFrame, season: int, competition: str) -> tuple[list[tuple[str, str]], str] | None:
    """The real round-of-16 ties (first leg's home club first) and the first leg's date."""
    r16 = games[(games["competition"] == competition) & (games["season"] == season) & (games["stage"] == "round-of-16")]
    if r16.empty:
        return None
    ties = {}
    for g in r16.sort_values("date").itertuples():
        ties.setdefault(frozenset((g.home_id, g.away_id)), (g.home_id, g.away_id))
    return (list(ties.values()), r16["date"].min()) if len(ties) == 8 else None


def winner_of_final(part: pd.DataFrame):
    final = part[part["stage"] == "final"]
    if final.empty:
        return None
    f = final.iloc[0]
    hp = f.home_pens if pd.notna(f.home_pens) else 0
    ap = f.away_pens if pd.notna(f.away_pens) else 0
    return f.home_id if (f.home_score, hp) > (f.away_score, ap) else f.away_id


def reached(games: pd.DataFrame, season: int, team: str, competition: str) -> dict:
    """How far a club got: 1 for each stage it reached."""
    part = games[(games["competition"] == competition) & (games["season"] == season)]
    plays = lambda stage: bool(((part["stage"] == stage) & ((part["home_id"] == team) | (part["away_id"] == team))).any())
    out = {s: plays(s) for s in ("knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final")}
    out["champion"] = winner_of_final(part) == team
    out["top8"] = out["round-of-16"] and not out["knockout-round-playoffs"]
    out["top24"] = out["round-of-16"] or out["knockout-round-playoffs"]
    return out


def knockout_odds(model: dict, ties: list, spread: float, sims: int = 10_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = sorted({t for tie in ties for t in tie})
    index = {t: i for i, t in enumerate(teams)}
    rates = Rates(model, teams, sims, rng, spread)
    reach = {r: np.zeros((sims, len(teams)), bool) for r in ROUNDS}
    rows = np.arange(sims)
    alive = np.column_stack([play_tie(np.full(sims, index[a]), np.full(sims, index[b]), rates, rng) for a, b in ties])
    for r in ("quarterfinals", "semifinals", "final"):
        reach[r][rows[:, None], alive] = True
        if r == "final":
            break
        order = np.argsort(rng.random(alive.shape), axis=1)          # a random draw for the next round
        alive = np.take_along_axis(alive, order, axis=1)
        alive = np.column_stack([play_tie(alive[:, k], alive[:, k + 1], rates, rng) for k in range(0, alive.shape[1], 2)])
    champion = play_final(alive[:, 0], alive[:, 1], rates, rng)
    reach["champion"][rows, champion] = True
    return pd.DataFrame({r: v.mean(axis=0) for r, v in reach.items()}, index=teams)


def loss(p: float, happened: bool) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(-np.log(p if happened else 1 - p))


def knockout_scores(games: pd.DataFrame, span: tuple, cups: list[str], spreads=SPREADS, values=None) -> dict:
    losses = {s: [] for s in spreads}
    for code in cups:
        comp = CUPS[code]
        for season in range(span[0], span[1] + 1):
            found = round_of_16(games, season, comp)
            if found is None:
                continue
            ties, day = found
            model = fit(window(games, day), day, values=values)
            for s in spreads:
                for team, row in knockout_odds(model, ties, s).iterrows():
                    got = reached(games, season, team, comp)
                    losses[s] += [loss(row[r], got["champion" if r == "champion" else r]) for r in ROUNDS]
    return {s: (float(np.mean(v)), len(v)) for s, v in losses.items()}


def league_phase_scores(games: pd.DataFrame, seasons=(2024, 2025), spreads=SPREADS, sims: int = 4000, values=None) -> dict:
    """From before match day 1 and from halfway through the league phase."""
    losses = {s: [] for s in spreads}
    for comp in CUPS.values():
        for season in seasons:
            lp = games[(games["competition"] == comp) & (games["season"] == season) & (games["stage"] == "league-phase")]
            if lp.empty:
                continue
            days = sorted(lp["date"].unique())
            for cutoff in (days[0], days[len(days) // 2]):
                past = games[games["date"] < cutoff]
                model = fit(window(games, cutoff), cutoff, values=values)
                known = pd.concat([past, lp[lp["date"] >= cutoff].assign(status="scheduled", home_score=np.nan, away_score=np.nan)])
                for s in spreads:
                    odds = simulate(known, season, model, sims=sims, spread=s, competition=comp)
                    for team, row in odds.iterrows():
                        got = reached(games, season, team, comp)
                        losses[s] += [loss(row["top8"], got["top8"]), loss(row["top24"], got["top24"]),
                                      loss(row["quarterfinals"], got["quarterfinals"]), loss(row["champion"], got["champion"])]
    return {s: (float(np.mean(v)), len(v)) for s, v in losses.items()}


LEAGUES = {"ESP": "esp.1", "GER": "ger.1", "ITA": "ita.1", "FRA": "fra.1"}
LEAGUE_SPREADS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def final_table(part: pd.DataFrame) -> list:
    """A finished league season's clubs in order: points, goal difference, goals scored."""
    done = part[part["status"] == "final"]
    rows = {}
    for g in done.itertuples():
        for t, f, a in ((g.home_id, g.home_score, g.away_score), (g.away_id, g.away_score, g.home_score)):
            r = rows.setdefault(t, [0, 0, 0])
            r[0] += 3 * (f > a) + (f == a)
            r[1] += f - a
            r[2] += f
    return sorted(rows, key=lambda t: tuple(-v for v in rows[t]))


def league_scores(games: pd.DataFrame, span: tuple, spreads=LEAGUE_SPREADS, sims: int = 4000, values=None,
                  fractions=(0.25, 0.5)) -> dict:
    """From a quarter and halfway through each season of the four big leagues: the title, top four
    and bottom three chances, scored on how the season ended."""
    losses = {s: [] for s in spreads}
    for comp in LEAGUES.values():
        for season in range(span[0], span[1] + 1):
            part = games[(games["competition"] == comp) & (games["season"] == season) & (games["status"] != "other")]
            if len(part) < 200:
                continue
            order = final_table(part)
            n = len(order)
            got = {t: {"champion": k == 0, "top4": k < 4, "bottom3": k >= n - 3} for k, t in enumerate(order)}
            days = sorted(part["date"].unique())
            for f in fractions:
                cutoff = days[int(len(days) * f)]
                model = fit(window(games, cutoff), cutoff, values=values)
                known = part.copy()
                later = known["date"] >= cutoff
                known.loc[later, ["status", "home_score", "away_score"]] = ["scheduled", np.nan, np.nan]
                for s in spreads:
                    odds = simulate_league(known, season, model, comp, sims=sims, spread=s)
                    for team, row in odds.iterrows():
                        if team in got:
                            losses[s] += [loss(row[c], got[team][c]) for c in ("champion", "top4", "bottom3")]
    return {s: (float(np.mean(v)), len(v)) for s, v in losses.items()}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games, values = load_games(conn), load_values(conn)
    if "leagues" in argv:
        span, label = (HELD_OUT, "2021-2025") if "held_out" in argv else (TUNING, "2016-2020")
        for s, (l, n) in league_scores(games, span, values=values).items():
            print(f"big leagues {label}: spread {s:.2f}: log loss {l:.4f} ({n} chances)", flush=True)
        return
    if "held_out" in argv:
        for s, (l, n) in knockout_scores(games, HELD_OUT, ["UCL", "UEL", "UECL"], values=values).items():
            print(f"knockouts 2021-2025, all cups: spread {s:.2f}: log loss {l:.4f} ({n} chances)", flush=True)
        for s, (l, n) in league_phase_scores(games, values=values).items():
            print(f"league phases 2024-2025, all cups: spread {s:.2f}: log loss {l:.4f} ({n} chances)", flush=True)
        return
    for s, (l, n) in knockout_scores(games, TUNING, ["UCL", "UEL"], values=values).items():
        print(f"knockouts 2016-2020, Champions and Europa League: spread {s:.2f}: log loss {l:.4f} ({n} chances)", flush=True)


if __name__ == "__main__":
    main()
