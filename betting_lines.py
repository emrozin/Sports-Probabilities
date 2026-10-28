"""Spreads and totals: would the betting rule do better on point spreads and over/unders than on
moneylines? python -m betting_lines [--record]

The models already predict each side's points (or goals), so they give a chance for every spread
and total, not just for who wins:

- NFL and college football: the actual margin (and total) is the expected one plus a normal error
  whose size (standard deviation) is measured on the tuning seasons, along with any average
  bias in the totals. A line on a whole number can push (the stake is refunded): the chance of
  landing exactly on it is the chance of the error putting the score within half a point of it.
- Premier League: the model's full scoreline grid (epl/model.py) gives the over/under 2.5 goals
  and Asian handicap chances exactly. Quarter-goal handicaps (-0.75, say) split the stake over the
  two neighbouring lines, as bookmakers settle them.

The rule is the moneyline rule (betting.py): bet when chance x price - 1 is at least 3%, at most one
bet per game in each market (the better side), a quarter of the Kelly stake, at most 3%.

Prices: the NFL's closing spread and total prices (nflverse); for college football only the lines
are known, so the usual -110 (1.909) is assumed on both sides; the Premier League's best price
across bookmakers before the match for over/under 2.5 (as for its moneyline bets), and Pinnacle's
for Asian handicaps (HANDICAP_BOOKMAKER).

Everything is scored on the held-out seasons of each sport's backtest only (NFL and college
football 2018 on, Premier League 2021/22 on); the error sizes come from the tuning seasons before.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import betting
from db import DB_PATH

RECORD_PATH = Path("records/spreads_totals.csv")
# Asian handicaps at Pinnacle's price: football-data.co.uk's best price across bookmakers can come
# from a bookmaker offering a different handicap from the listed one, which made 2025/26 look like
# +50% a bet. Pinnacle's price always goes with its own line.
HANDICAP_BOOKMAKER = "Pinnacle"
STANDARD_PRICE = 1 / 0.5238      # -110
MARKETS = {"NFL": ("spread", "total"), "CFB": ("spread", "total"), "EPL": ("asian_handicap", "total")}
NAMES = {"spread": "Point spread", "total": "Over/under", "asian_handicap": "Asian handicap"}


# ---------------------------------------------------------------- the data

def points_games(conn, league: str) -> pd.DataFrame:
    """Finished games with the backtest's expected points and the lines (home handicap, total)."""
    info = betting.sports()[league]
    name, version = info["backtest"]
    table = {"NFL": "nfl_game_info", "CFB": "cfb_game_info"}[league]
    g = pd.read_sql(f"""SELECT g.id AS game_id, g.season, g.date, g.home_score, g.away_score, pg.exp_home, pg.exp_away,
                               i.spread_line, i.total_line
                        FROM games g JOIN {table} i ON i.game_id = g.id
                        JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = ? AND pg.model_version = ?
                        WHERE g.status = 'final'""", conn, params=(name, version))
    if league == "NFL":     # nflverse's spread_line is how many points the home team is favoured by
        g["handicap"] = -g["spread_line"]
        o = pd.read_sql("""SELECT game_id, market, outcome, price FROM odds WHERE bookmaker = 'nflverse'
                           AND market IN ('spread', 'total')""", conn)
        wide = o.pivot_table(index="game_id", columns=["market", "outcome"], values="price")
        wide.columns = [f"{m}_{o}" for m, o in wide.columns]
        g = g.merge(wide.reset_index(), on="game_id", how="left")
    else:                   # college football's spread_line is already the home team's handicap
        g["handicap"] = g["spread_line"]
        for col in ("spread_home", "spread_away", "total_over", "total_under"):
            g[col] = STANDARD_PRICE
    return g


def soccer_games(conn) -> pd.DataFrame:
    """Premier League matches with expected goals and the best pre-match over/under 2.5 and Asian
    handicap prices."""
    name, version = betting.sports()["EPL"]["backtest"]
    g = pd.read_sql("""SELECT g.id AS game_id, g.season, g.date, g.home_score, g.away_score, pg.exp_home, pg.exp_away
                       FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = 'EPL'
                       JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = ? AND pg.model_version = ?
                       WHERE g.status = 'final'""", conn, params=(name, version))
    o = pd.read_sql("""SELECT game_id, market, outcome, bookmaker, price, line FROM odds
                       WHERE market IN ('total', 'asian_handicap') AND is_closing = 0""", conn)
    o = o[((o["market"] == "total") & (o["line"] == 2.5) & o["bookmaker"].isin(["Market average", betting.EPL_PRICE]))
          | ((o["market"] == "asian_handicap") & (o["bookmaker"] == HANDICAP_BOOKMAKER))]
    o = o.sort_values("bookmaker", key=lambda b: b == betting.EPL_PRICE).drop_duplicates(["game_id", "market", "outcome"], keep="last")
    wide = o.pivot_table(index="game_id", columns=["market", "outcome"], values="price")
    wide.columns = [f"{'spread' if m == 'asian_handicap' else m}_{x}" for m, x in wide.columns]
    lines = o[o["market"] == "asian_handicap"].drop_duplicates("game_id").set_index("game_id")["line"].rename("handicap")
    g = g.merge(wide.reset_index(), on="game_id", how="left").merge(lines.reset_index(), on="game_id", how="left")
    g["total_line"] = 2.5
    return g


# ---------------------------------------------------------------- chances and results

def halves(line: float) -> list[float]:
    """A quarter line (-0.75) is half a bet on each neighbour (-0.5 and -1.0); others are one line."""
    if abs(line * 4 - round(line * 4)) < 1e-9 and abs(line * 2 - round(line * 2)) > 1e-9:
        return [line - 0.25, line + 0.25]
    return [line]


def normal_outcome(mean: float, sd: float, line: float) -> tuple[float, float, float]:
    """(win, push, lose) for 'the quantity beats -line', i.e. quantity + line > 0, with a normal error;
    whole-number lines push within half a point."""
    if abs(line - round(line)) < 1e-9:
        lose = norm.cdf((-line - 0.5 - mean) / sd)
        win = 1 - norm.cdf((-line + 0.5 - mean) / sd)
        return win, 1 - win - lose, lose
    win = 1 - norm.cdf((-line - mean) / sd)
    return win, 0.0, 1 - win


def grid_outcome(grid: np.ndarray, value: np.ndarray, line: float) -> tuple[float, float, float]:
    """(win, push, lose) for value + line > 0 over a scoreline grid."""
    x = value + line
    return float(grid[x > 1e-9].sum()), float(grid[np.abs(x) <= 1e-9].sum()), float(grid[x < -1e-9].sum())


def expected_return(chances: list[tuple[float, float, float]], price: float) -> float:
    """Expected profit per unit staked, the stake split evenly over the lines."""
    return float(np.mean([w * (price - 1) - l for w, _, l in chances]))


def settle_line(value: float, line: float, price: float) -> float:
    """Profit per unit for 'value + line > 0' at this price, quarter lines split."""
    out = []
    for part in halves(line):
        x = value + part
        out.append(price - 1 if x > 1e-9 else 0.0 if abs(x) <= 1e-9 else -1.0)
    return float(np.mean(out))


def bets_for(games: pd.DataFrame, chance, market: str) -> pd.DataFrame:
    """Every side of every game in the market with its expected return and settled profit.
    chance(row, side_line, quantity) -> (win, push, lose); market 'spread' or 'total'."""
    rows = []
    for g in games.itertuples():
        margin, total = g.home_score - g.away_score, g.home_score + g.away_score
        if market == "spread":
            if pd.isna(g.handicap):
                continue
            sides = [("home", g.handicap, getattr(g, "spread_home", np.nan), "margin", margin),
                     ("away", -g.handicap, getattr(g, "spread_away", np.nan), "-margin", -margin)]
        else:
            if pd.isna(g.total_line):
                continue
            sides = [("over", -g.total_line, getattr(g, "total_over", np.nan), "total", total),
                     ("under", g.total_line, getattr(g, "total_under", np.nan), "-total", -total)]
        for side, line, price, quantity, actual in sides:
            if pd.isna(price) or price <= 1:
                continue
            parts = [chance(g, part, quantity) for part in halves(line)]
            ev = expected_return(parts, price)
            win = np.mean([p[0] for p in parts]); lose = np.mean([p[2] for p in parts])
            rows.append({"game_id": g.game_id, "season": g.season, "date": g.date, "market": market, "outcome": side,
                         "line": line, "price": price, "p": win, "p_lose": lose, "edge": ev,
                         "kelly": max(ev, 0) / (price - 1), "profit": settle_line(actual, line, price)})
    return pd.DataFrame(rows)


def errors(games: pd.DataFrame, seasons) -> dict:
    """The margin and total errors' standard deviations and the totals' average bias on these seasons."""
    part = games[games["season"].between(*seasons)]
    margin = (part["home_score"] - part["away_score"]) - (part["exp_home"] - part["exp_away"])
    total = (part["home_score"] + part["away_score"]) - (part["exp_home"] + part["exp_away"])
    return {"margin_sd": float(margin.std()), "total_sd": float(total.std()), "total_bias": float(total.mean())}


def candidates(chances: pd.DataFrame, min_edge: float = betting.MIN_EDGE) -> pd.DataFrame:
    best = chances.sort_values("edge", ascending=False).drop_duplicates(["game_id", "market"])
    return best[best["edge"] >= min_edge]


def summary(league: str, market: str, picks: pd.DataFrame, all_sides: pd.DataFrame) -> dict:
    profit = picks["profit"].to_numpy()
    low, high = betting.roi_range(profit)
    sim = betting.simulate(picks.assign(profit=picks["profit"]))
    # accuracy against the market: log loss of the first side's chance (home covers, or over) in
    # games that weren't a push, the model's against the price's (margin removed)
    first = all_sides[all_sides["outcome"].isin(["home", "over"]) & (all_sides["profit"] != 0)
                      & (all_sides["line"] * 4 % 2 == 0)].copy()      # whole and half lines only
    other = all_sides[~all_sides["outcome"].isin(["home", "over"])].set_index("game_id")["price"]
    first["q"] = (1 / first["price"]) / (1 / first["price"] + 1 / first["game_id"].map(other))
    first["p2"] = first["p"] / (first["p"] + first["p_lose"])          # leaving pushes out
    won = (first["profit"] > 0).to_numpy()
    loss = lambda p: float(-np.mean(np.where(won, np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(1 - p, 1e-6, 1)))))
    return {"league": league, "market": market, "name": f"{betting.sports()[league]['name']}: {NAMES[market].lower()}",
            "seasons": f"{int(all_sides['season'].min())}-{int(all_sides['season'].max())}",
            "games": int(all_sides["game_id"].nunique()), "bets": len(picks), "expected": float(picks["edge"].mean()) if len(picks) else np.nan,
            "roi": float(profit.mean()) if len(picks) else np.nan, "roi_low": low, "roi_high": high,
            "bankroll": sim["final"], "drawdown": sim["drawdown"],
            "model_loss": loss(first["p2"].to_numpy()) if len(first) else np.nan,
            "market_loss": loss(first["q"].to_numpy()) if len(first) else np.nan}


def run(conn) -> pd.DataFrame:
    out = []
    for league in ("NFL", "CFB"):
        info = betting.sports()[league]
        g = points_games(conn, league)
        if g.empty:
            continue
        current = int(pd.read_sql(f"SELECT MAX(season) FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = '{league}'", conn).iloc[0, 0])
        e = errors(g, (g["season"].min(), info["held_out"] - 1))
        held = g[(g["season"] >= info["held_out"]) & (g["season"] < current)]

        def chance(row, line, quantity):
            if quantity in ("margin", "-margin"):
                mean = row.exp_home - row.exp_away
                return normal_outcome(mean if quantity == "margin" else -mean, e["margin_sd"], line)
            mean = row.exp_home + row.exp_away + e["total_bias"]
            return normal_outcome(mean if quantity == "total" else -mean, e["total_sd"], line)
        for market in ("spread", "total"):
            sides = bets_for(held, chance, market)
            if not sides.empty:
                out.append(summary(league, market, candidates(sides), sides))
    from epl.model import scoreline_grid
    g = soccer_games(conn)
    if not g.empty:
        held = g[(g["season"] >= betting.sports()["EPL"]["held_out"]) & (g["season"] < int(g["season"].max()))]
        grids = {}

        def chance(row, line, quantity):
            if row.game_id not in grids:
                grid = scoreline_grid(row.exp_home, row.exp_away)
                h, a = np.indices(grid.shape)
                grids[row.game_id] = (grid / grid.sum(), h - a, h + a)
            grid, margin, total = grids[row.game_id]
            value = {"margin": margin, "-margin": -margin, "total": total, "-total": -total}[quantity]
            return grid_outcome(grid, value, line)
        for market in ("spread", "total"):
            sides = bets_for(held, chance, market)
            if not sides.empty:
                name = "asian_handicap" if market == "spread" else market
                out.append(summary("EPL", name, candidates(sides), sides))
    return pd.DataFrame(out)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    results = run(conn)
    for r in results.itertuples():
        print(f"{r.name:<42} {r.seasons}: {r.bets:>5} bets of {r.games} games, expected {r.expected:+.1%}, "
              f"actual {r.roi:+.1%} (95% range {r.roi_low:+.1%} to {r.roi_high:+.1%}), $1,000 -> ${r.bankroll:,.0f}; "
              f"log loss {r.model_loss:.4f} (market {r.market_loss:.4f})")
    if "--record" in argv:
        RECORD_PATH.parent.mkdir(exist_ok=True)
        results.to_csv(RECORD_PATH, index=False)
        print(f"Saved {RECORD_PATH}.")


if __name__ == "__main__":
    main()
