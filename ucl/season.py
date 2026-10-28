"""Season simulations: for the European cups, each club's chances of finishing in the top 8 of the
league phase, reaching each knockout round and winning it (the Champions League, the Europa League,
the Conference League); for La Liga, the Bundesliga, Serie A and Ligue 1, the title, the top four
and the bottom three (simulate_league). python -m ucl.season [--record]

The format since 2024-25: 36 clubs play 8 league-phase games each (6 in the Conference League), in one table. 1st-8th go
straight to the round of 16; 9th-24th play two-legged knockout play-offs; 25th-36th are out.
Then two-legged ties to the semi-finals and a one-off final on a neutral ground.

Every game still to play is simulated SIMULATIONS times from the model's expected goals (ucl/model.py),
with the results already in kept as they are:

- League phase: 3 points a win, 1 a draw; ties on points split by goal difference, goals scored,
  away goals scored, wins, away wins, then at random (UEFA's later tie-breakers, such as the
  opponents' results, aren't modelled).
- The knockout bracket follows the seeding rules: play-off ties are 9th/10th against 23rd/24th,
  11th/12th against 21st/22nd, 13th/14th against 19th/20th, 15th/16th against 17th/18th (which of
  each pair at random, as the draw does), and their winners meet 7th/8th, 5th/6th, 3rd/4th and
  1st/2nd in the round of 16. 1st and 2nd are kept in opposite halves of the bracket. The
  better-placed club plays the second leg at home. Once the real pairings of a round are known (in
  the fixtures), they're used instead, with any legs already played.
- A level tie goes to extra time at the second leg's ground (a third of a game's goals), then
  penalties at even odds.
"""

import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from ucl.model import EXTRA_TIME, LEAGUE_XG_WEIGHT, UCL, expected_goals, fit, load_games, load_values, window

SIMULATIONS = 10_000
# Each simulated season's random shift in every club's ratings (log scale). Tested on past knockout
# stages and league phases (python -m ucl.season_test, with the squad-value model): a small shift
# (0.1-0.2) was marginally better on the tuning knockouts (0.4061 -> 0.4057) and the league phases
# (0.3032 -> 0.3028), no shift was best on the held-out knockouts (0.3709 against 0.3716 at 0.1),
# and bigger shifts were worse everywhere. The simulation's confidence is about right, so none is used.
SPREAD = 0.0
RECORD_PATH = Path("records/europe_season_odds.csv")
ROUNDS = ["knockout-round-playoffs", "round-of-16", "quarterfinals", "semifinals", "final"]
COLUMNS = ["top8", "top24", "round_of_16", "quarterfinals", "semifinals", "final", "champion"]
PLAYOFF_PAIRS = [((8, 9), (22, 23)), ((10, 11), (20, 21)), ((12, 13), (18, 19)), ((14, 15), (16, 17))]   # 0-based positions
R16_SEEDS = [(6, 7), (4, 5), (2, 3), (0, 1)]          # meet the winners of the play-off pairs above, in order


def league_phase(games: pd.DataFrame, season: int, competition: str = UCL) -> pd.DataFrame:
    part = games[(games["competition"] == competition) & (games["season"] == season) & (games["stage"] == "league-phase")]
    return part[part["status"] != "other"]


def goal_rates(model: dict, teams: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """home[i, j], away[i, j]: expected goals for i at home to j, and for j away at i."""
    n = len(teams)
    home, away = np.zeros((n, n)), np.zeros((n, n))
    for i, a in enumerate(teams):
        for j, b in enumerate(teams):
            if i != j:
                home[i, j], away[i, j] = expected_goals(model, a, b)
    return home, away


def table_order(points, gd, gf, away_gf, wins, away_wins, rng) -> np.ndarray:
    """(sims, teams) stats -> (sims, teams) team indices from 1st to last."""
    noise = rng.random(points.shape)
    keys = (noise, away_wins, wins, away_gf, gf, gd, points)        # lexsort: last key first
    return np.lexsort(tuple(-k for k in keys), axis=-1)


class Rates:
    """Expected goals for any pairing in any simulation: the model's, times each simulated season's
    random shift in every club's attack and defense (SPREAD on the log scale). The shifts stand for
    the model's ratings being estimates: without them the simulation would treat today's ratings as
    exact and be too sure of the favorites over a whole season."""

    def __init__(self, model: dict, teams: list[str], sims: int, rng, spread: float = 0.0):
        self.home, self.away = goal_rates(model, teams)
        self.neutral = np.where(self.home > 0, self.home * np.exp(-model["home_europe"]), 0.0)
        n = len(teams)
        self.attack = rng.normal(0, spread, (sims, n)) if spread else np.zeros((sims, n))
        self.defense = rng.normal(0, spread, (sims, n)) if spread else np.zeros((sims, n))
        self.rows = np.arange(sims)

    def at_home(self, i, j) -> np.ndarray:
        """Goals for i at home to j (arrays over simulations)."""
        return self.home[i, j] * np.exp(self.attack[self.rows, i] - self.defense[self.rows, j])

    def away_side(self, i, j) -> np.ndarray:
        """Goals for j away at i."""
        return self.away[i, j] * np.exp(self.attack[self.rows, j] - self.defense[self.rows, i])

    def on_neutral(self, i, j) -> np.ndarray:
        """Goals for i against j on a neutral ground."""
        return self.neutral[i, j] * np.exp(self.attack[self.rows, i] - self.defense[self.rows, j])


def play_tie(first_home, second_home, rates: Rates, rng, leg1=None, leg2=None) -> np.ndarray:
    """Winners of two-legged ties (arrays over simulations): first_home hosts the first leg.
    leg1/leg2: (goals of that leg's home side, of its away side) if already played."""
    sims = len(first_home)
    g1h = np.full(sims, leg1[0]) if leg1 else rng.poisson(rates.at_home(first_home, second_home))
    g1a = np.full(sims, leg1[1]) if leg1 else rng.poisson(rates.away_side(first_home, second_home))
    second_home_rate, first_away_rate = rates.at_home(second_home, first_home), rates.away_side(second_home, first_home)
    g2h = np.full(sims, leg2[0]) if leg2 else rng.poisson(second_home_rate)
    g2a = np.full(sims, leg2[1]) if leg2 else rng.poisson(first_away_rate)
    margin = (g1h + g2a) - (g1a + g2h)                               # for first_home
    et_first = rng.poisson(first_away_rate * EXTRA_TIME)
    et_second = rng.poisson(second_home_rate * EXTRA_TIME)
    margin = np.where(margin == 0, et_first - et_second, margin)
    coin = rng.random(sims) < 0.5
    first_wins = (margin > 0) | ((margin == 0) & coin)
    return np.where(first_wins, first_home, second_home)


def play_final(a, b, rates: Rates, rng) -> np.ndarray:
    """One game on a neutral ground: 90 minutes, extra time, penalties."""
    ra, rb = rates.on_neutral(a, b), rates.on_neutral(b, a)
    ga, gb = rng.poisson(ra), rng.poisson(rb)
    eta, etb = rng.poisson(ra * EXTRA_TIME), rng.poisson(rb * EXTRA_TIME)
    margin = np.where(ga == gb, eta - etb, ga - gb)
    return np.where((margin > 0) | ((margin == 0) & (rng.random(len(a)) < 0.5)), a, b)


def known_ties(games: pd.DataFrame, season: int, stage: str, index: dict, competition: str = UCL) -> list[dict]:
    """A knockout round's real pairings from the fixtures: [{first, second, leg1, leg2}], first
    hosting the first leg; the final as one game."""
    part = games[(games["competition"] == competition) & (games["season"] == season) & (games["stage"] == stage) & (games["status"] != "other")]
    ties = {}
    for g in part.sort_values(["date", "kickoff"]).itertuples():
        key = frozenset((g.home_id, g.away_id))
        tie = ties.setdefault(key, {"first": g.home_id, "second": g.away_id, "leg1": None, "leg2": None, "games": 0})
        tie["games"] += 1
        if g.status == "final":
            score = (int(g.home_score), int(g.away_score))
            if g.home_id == tie["first"] and tie["games"] == 1:
                tie["leg1"] = score
            elif g.home_id == tie["second"]:
                tie["leg2"] = score
                if getattr(g, "extra_time", 0) == 1 or (pd.notna(g.home_pens)):
                    tie["decided"] = g.home_id if (g.home_score, g.home_pens or 0) > (g.away_score, g.away_pens or 0) else g.away_id
    return [t for t in ties.values() if t["first"] in index and t["second"] in index]


def simulate(games: pd.DataFrame, season: int, model: dict, sims: int = SIMULATIONS, seed: int = 0,
             spread: float | None = None, competition: str = UCL) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lp = league_phase(games, season, competition)
    teams = sorted(set(lp["home_id"]) | set(lp["away_id"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rates = Rates(model, teams, sims, rng, SPREAD if spread is None else spread)
    stats = {k: np.zeros((sims, n)) for k in ("points", "gd", "gf", "away_gf", "wins", "away_wins")}
    for g in lp.itertuples():
        i, j = index[g.home_id], index[g.away_id]
        if g.status == "final":
            hg, ag = np.full(sims, g.home_score), np.full(sims, g.away_score)
        else:
            hg, ag = rng.poisson(rates.at_home(i, j)), rng.poisson(rates.away_side(i, j))
        stats["points"][:, i] += 3 * (hg > ag) + (hg == ag)
        stats["points"][:, j] += 3 * (ag > hg) + (hg == ag)
        stats["gd"][:, i] += hg - ag
        stats["gd"][:, j] += ag - hg
        stats["gf"][:, i] += hg
        stats["gf"][:, j] += ag
        stats["away_gf"][:, j] += ag
        stats["wins"][:, i] += hg > ag
        stats["wins"][:, j] += ag > hg
        stats["away_wins"][:, j] += ag > hg
    order = table_order(stats["points"], stats["gd"], stats["gf"], stats["away_gf"], stats["wins"], stats["away_wins"], rng)
    position = np.argsort(order, axis=1)                    # (sims, teams): 0 = 1st
    reached = {k: np.zeros((sims, n), bool) for k in COLUMNS}
    rows = np.arange(sims)[:, None]
    reached["top8"][rows, order[:, :8]] = True
    reached["top24"][rows, order[:, :24]] = True
    flip = lambda: rng.random(sims) < 0.5

    # knockout play-offs
    real = known_ties(games, season, ROUNDS[0], index, competition)
    if len(real) == 8:
        real = sorted(real, key=lambda t: _seed_slot(t, index, position))     # 9th/10th's ties first, as PLAYOFF_PAIRS
        winners = [_real_tie(t, index, rates, rng, sims) for t in real]
        po_winners = [(winners[k], winners[k + 1]) for k in (0, 2, 4, 6)]
    else:
        po_winners = []
        for (s1, s2), (u1, u2) in PLAYOFF_PAIRS:
            f = flip()
            seeded = np.where(f, order[:, s1], order[:, s2])
            unseeded = np.where(f, order[:, u1], order[:, u2])
            seeded_b = np.where(f, order[:, s2], order[:, s1])
            unseeded_b = np.where(f, order[:, u2], order[:, u1])
            po_winners.append((play_tie(unseeded, seeded, rates, rng), play_tie(unseeded_b, seeded_b, rates, rng)))
    # round of 16: (seed, play-off winner) slots, then the bracket
    real16 = known_ties(games, season, ROUNDS[1], index, competition)
    if len(real16) == 8:
        r16 = sorted(real16, key=lambda t: _seed_slot(t, index, position), reverse=True)   # 7th/8th's ties first
        r16_winners = [_real_tie(t, index, rates, rng, sims) for t in r16]
        r16_winners = [r16_winners[k] for k in (6, 0, 4, 2, 7, 1, 5, 3)]
        entrants = [np.full(sims, index[t[k]]) for t in r16 for k in ("first", "second")]
    else:
        r16_winners, entrants = [], []
        for (sa, sb), (wa, wb) in zip(R16_SEEDS, po_winners):
            f = flip()
            s_a, s_b = np.where(f, order[:, sa], order[:, sb]), np.where(f, order[:, sb], order[:, sa])
            r16_winners += [play_tie(wa, s_a, rates, rng), play_tie(wb, s_b, rates, rng)]
            entrants += [wa, s_a, wb, s_b]
        # slots: [7/8 a, 7/8 b, 5/6 a, 5/6 b, 3/4 a, 3/4 b, 1/2 a, 1/2 b] -> halves keep 1st and 2nd apart
        r16_winners = [r16_winners[k] for k in (6, 0, 4, 2, 7, 1, 5, 3)]
    for e in entrants:
        reached["round_of_16"][np.arange(sims), e] = True
    # quarter-finals, semi-finals, final
    real_qf = known_ties(games, season, ROUNDS[2], index, competition)
    if len(real_qf) == 4:
        qf_entrants = [np.full(sims, index[t[k]]) for t in real_qf for k in ("first", "second")]
        qf_winners = [_real_tie(t, index, rates, rng, sims) for t in real_qf]
    else:
        qf_entrants = r16_winners
        qf_winners = [_host_by_position(r16_winners[k], r16_winners[k + 1], position, rates, rng) for k in (0, 2, 4, 6)]
    for e in qf_entrants:
        reached["quarterfinals"][np.arange(sims), e] = True
    real_sf = known_ties(games, season, ROUNDS[3], index, competition)
    if len(real_sf) == 2:
        sf_entrants = [np.full(sims, index[t[k]]) for t in real_sf for k in ("first", "second")]
        sf_winners = [_real_tie(t, index, rates, rng, sims) for t in real_sf]
    else:
        sf_entrants = qf_winners
        sf_winners = [_host_by_position(qf_winners[k], qf_winners[k + 1], position, rates, rng) for k in (0, 2)]
    for e in sf_entrants:
        reached["semifinals"][np.arange(sims), e] = True
    for e in sf_winners:
        reached["final"][np.arange(sims), e] = True
    champion = play_final(sf_winners[0], sf_winners[1], rates, rng)
    played_final = games[(games["competition"] == competition) & (games["season"] == season) & (games["stage"] == "final") & (games["status"] == "final")]
    if len(played_final):
        g = played_final.iloc[0]
        won = g.home_id if (g.home_score, g.home_pens if pd.notna(g.home_pens) else 0) > (g.away_score, g.away_pens if pd.notna(g.away_pens) else 0) else g.away_id
        if won in index:
            champion = np.full(sims, index[won])
    reached["champion"][np.arange(sims), champion] = True
    out = pd.DataFrame({k: v.mean(axis=0) for k, v in reached.items()}, index=teams)
    out["points"] = stats["points"].mean(axis=0)
    out["position"] = position.mean(axis=0) + 1
    return out.sort_values(["champion", "top8", "points"], ascending=False)


def _host_by_position(a, b, position, rates, rng):
    """A tie where the better-placed club in the league phase hosts the second leg."""
    rows = np.arange(len(a))
    a_better = position[rows, a] < position[rows, b]
    first = np.where(a_better, b, a)
    second = np.where(a_better, a, b)
    return play_tie(first, second, rates, rng)


def _real_tie(t, index, rates, rng, sims):
    if t.get("decided"):
        return np.full(sims, index[t["decided"]])
    first, second = np.full(sims, index[t["first"]]), np.full(sims, index[t["second"]])
    return play_tie(first, second, rates, rng, t["leg1"], t["leg2"])


def _seed_slot(t, index, position) -> float:
    """Where a real tie sits in the bracket: by the better league-phase position of its two clubs."""
    return float(np.minimum(position[:, index[t["first"]]], position[:, index[t["second"]]]).mean())


LEAGUE_COLUMNS = ["champion", "top4", "bottom3"]
LEAGUE_SPREAD = 0.1       # each simulated season's random shift in every club's ratings (ucl/season_test.py: it helps in the leagues)


def simulate_league(games: pd.DataFrame, season: int, model: dict, competition: str, sims: int = SIMULATIONS,
                    seed: int = 0, spread: float | None = None) -> pd.DataFrame:
    """A league's season played out: each club's chance of the title, the top four (the Champions
    League places in Spain, Germany, Italy and France) and the bottom three (relegation, or in
    Germany and France the bottom two plus a relegation play-off), its expected points and position.
    Ties on points are split by goal difference, then goals scored, then at random (the leagues'
    own head-to-head rules aren't modelled)."""
    rng = np.random.default_rng(seed)
    part = games[(games["competition"] == competition) & (games["season"] == season) & (games["status"] != "other")]
    teams = sorted(set(part["home_id"]) | set(part["away_id"]))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rates = Rates(model, teams, sims, rng, LEAGUE_SPREAD if spread is None else spread)
    rates.home = np.zeros((n, n))
    rates.away = np.zeros((n, n))
    for i, a in enumerate(teams):           # league games: the domestic home advantage
        for j, b in enumerate(teams):
            if i != j:
                rates.home[i, j], rates.away[i, j] = expected_goals(model, a, b, europe=False)
    points, gd, gf = (np.zeros((sims, n)) for _ in range(3))
    for g in part.itertuples():
        i, j = index[g.home_id], index[g.away_id]
        if g.status == "final":
            hg, ag = np.full(sims, g.home_score), np.full(sims, g.away_score)
        else:
            hg, ag = rng.poisson(rates.at_home(i, j)), rng.poisson(rates.away_side(i, j))
        points[:, i] += 3 * (hg > ag) + (hg == ag)
        points[:, j] += 3 * (ag > hg) + (hg == ag)
        gd[:, i] += hg - ag
        gd[:, j] += ag - hg
        gf[:, i] += hg
        gf[:, j] += ag
    order = np.lexsort((-rng.random((sims, n)), -gf, -gd, -points), axis=-1)
    position = np.argsort(order, axis=1)
    out = pd.DataFrame({"champion": (position == 0).mean(axis=0), "top4": (position < 4).mean(axis=0),
                        "bottom3": (position >= n - 3).mean(axis=0), "points": points.mean(axis=0),
                        "position": position.mean(axis=0) + 1}, index=teams)
    return out.sort_values(["champion", "points"], ascending=False)


def main(argv: list[str] | None = None, today: date | None = None) -> None:
    from ucl.cups import CUPS
    argv = sys.argv[1:] if argv is None else argv
    today = today or date.today()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    names = dict(conn.execute("SELECT team_id, name FROM soccer_teams").fetchall())
    values = load_values(conn)
    conn.close()
    model, league_model, frames = None, None, []
    for cup in CUPS.values():
        part = games[games["competition"] == cup["slug"]]
        if part.empty:
            continue
        season = int(part["season"].max())
        if cup["kind"] == "league":
            league_model = league_model or fit(window(games, today.isoformat()), today.isoformat(), values=values,
                                               xg_weight=LEAGUE_XG_WEIGHT)
            odds = simulate_league(games, season, league_model, cup["slug"])
            odds.insert(0, "team", [names.get(t, t) for t in odds.index])
            odds.insert(0, "team_id", odds.index)
            odds.insert(0, "date", today.isoformat())
            odds.insert(0, "season", season)
            odds.insert(0, "competition", cup["code"])
            frames.append(odds)
            print(cup["name"])
            print(odds.head(5)[["team", "champion", "top4", "bottom3", "points"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            continue
        if league_phase(games, season, cup["slug"]).empty or season < 2024:
            print(f"{cup['name']}: no league phase for {season}, nothing to simulate.")
            continue
        model = model or fit(window(games, today.isoformat()), today.isoformat(), values=values)
        odds = simulate(games, season, model, competition=cup["slug"])
        odds.insert(0, "team", [names.get(t, t) for t in odds.index])
        odds.insert(0, "team_id", odds.index)
        odds.insert(0, "date", today.isoformat())
        odds.insert(0, "season", season)
        odds.insert(0, "competition", cup["code"])
        frames.append(odds)
        print(cup["name"])
        print(odds.head(8)[["team", "top8", "round_of_16", "quarterfinals", "champion"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "--record" in argv and frames:
        new = pd.concat(frames, ignore_index=True)
        RECORD_PATH.parent.mkdir(exist_ok=True)
        old = pd.read_csv(RECORD_PATH, dtype={"team_id": str, "date": str}) if RECORD_PATH.exists() else pd.DataFrame()
        if not old.empty:
            old = old[old["date"] != today.isoformat()]
        pd.concat([old, new], ignore_index=True).round(5).to_csv(RECORD_PATH, index=False)
        print(f"Saved {RECORD_PATH}.")


if __name__ == "__main__":
    main()
