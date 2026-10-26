"""Experiments with the betting rule (betting.py): what was tried to make it pick better bets, and how
each change did on the held-out seasons. The results are shown on the Betting page.

The changes:

1. Best price: bet at the best price across bookmakers ('Market max') instead of the average.
   Premier League only (the one sport with more than one price). Kept: about 6 points better per bet.
2. Skip big disagreements: no bet in games where the model and the market differ by more than
   10 percentage points on any result (in the past tests, the biggest "edges" lost the most).
3. Blend: mix the model's chance with the market's (log-linear pooling, weight w on the model)
   before working out the edge. w is the weight with the best log loss on the tuning seasons, so it's
   chosen for accuracy, not fitted to the bets.
4. Tuned rule: the blend, plus the minimum edge and the disagreement cap that grew a quarter-Kelly
   bankroll most on the tuning seasons, then scored once on the held-out seasons.

The market's chance is the price with the margin taken out: Pinnacle's before the game where known
(the sharpest bookmaker), otherwise the average's. Only the Premier League, the NFL and college
football have past prices; NBA and college basketball lines are only being collected now.

Usage:
    python -m betting_experiments            # print the results
    python -m betting_experiments --record   # ...and save them for the site, records/betting_experiments.csv
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import betting
from betting import games, kelly, model_chances, roi_range, simulate, sports
from db import DB_PATH

LEAGUES = ("EPL", "NFL", "CFB")
W_GRID = np.round(np.arange(0, 1.001, 0.05), 2)
MIN_EDGES = [0.0, 0.01, 0.02, 0.03, 0.05]
CAPS = [0.03, 0.05, 0.08, 0.10, 0.15, 1.0]   # 1.0: no cap
SKIP_OVER = 0.10
MIN_BETS = 50
RECORD_PATH = Path("records/betting_experiments.csv")


def fair(game_id: pd.Series, price: pd.Series) -> pd.Series:
    """Each result's chance from the prices, the margin taken out proportionally."""
    implied = 1 / price
    return implied / implied.groupby(game_id).transform("sum")


def price_table(conn, league: str) -> pd.DataFrame:
    """game_id, outcome, avg (the average price), best (the best price: what the rule bets at now)
    and q (the market's chance)."""
    if league == "EPL":
        o = pd.read_sql("""SELECT game_id, outcome, bookmaker, price FROM odds WHERE market = '1x2' AND is_closing = 0
                           AND bookmaker IN ('Market average', ?, 'Pinnacle')""", conn, params=(betting.EPL_PRICE,))
        w = o.pivot_table(index=["game_id", "outcome"], columns="bookmaker", values="price").reset_index()
        w = w.rename(columns={"Market average": "avg", betting.EPL_PRICE: "best", "Pinnacle": "pin"})
        for col in ("avg", "best", "pin"):
            if col not in w:
                w[col] = np.nan
        w = w.dropna(subset=["avg"])
        w = w[w.groupby("game_id")["outcome"].transform("count") == 3].copy()
        pin_ok = w["pin"].notna().groupby(w["game_id"]).transform("all")
        w["q"] = np.where(pin_ok, fair(w["game_id"], w["pin"].fillna(w["avg"])), fair(w["game_id"], w["avg"]))
        w["best"] = w["best"].fillna(w["avg"]).clip(lower=w["avg"])
        return w[["game_id", "outcome", "avg", "best", "q"]]
    w = betting.prices(conn, league).rename(columns={"price": "avg"})
    w = w[w.groupby("game_id")["outcome"].transform("count") == 2].copy()
    w["q"] = fair(w["game_id"], w["avg"])
    w["best"] = w["avg"]
    return w[["game_id", "outcome", "avg", "best", "q"]]


def table(conn, league: str) -> pd.DataFrame:
    """Every finished backtest game before this season, one row per result: the model's chance p,
    the prices, the market's chance q, whether it won, and the game's biggest disagreement (gap)."""
    info = sports()[league]
    rows = games(conn, league)
    if rows.empty:
        return pd.DataFrame()
    current = int(rows["season"].max())
    rows = rows[(rows["status"] == "final") & (rows["season"] < current)].copy()
    rows["winner"] = [betting.winner(r) for _, r in rows.iterrows()]
    df = model_chances(conn, *info["backtest"]).merge(price_table(conn, league), on=["game_id", "outcome"])
    df = df[df.groupby("game_id")["outcome"].transform("count") == (3 if league == "EPL" else 2)]
    df = df.merge(rows[["game_id", "season", "date", "winner"]], on="game_id")
    if df.empty:
        return df
    df["p"] = df["p"] / df.groupby("game_id")["p"].transform("sum")
    df["won"] = (df["outcome"] == df["winner"]).astype(int)
    df["push"] = ((df["winner"] == "draw") & (league != "EPL")).astype(int)
    df["gap"] = (df["p"] - df["q"]).abs().groupby(df["game_id"]).transform("max")
    df["held_out"] = df["season"] >= info["held_out"]
    return df


def blend(df: pd.DataFrame, w: float) -> pd.Series:
    """Log-linear pool: chance proportional to p^w * q^(1-w). w = 1 is the model, w = 0 the market."""
    x = np.exp(w * np.log(df["p"].clip(1e-6)) + (1 - w) * np.log(df["q"].clip(1e-6)))
    return x / x.groupby(df["game_id"]).transform("sum")


def log_loss(df: pd.DataFrame, chance: pd.Series) -> float:
    return float(-np.log(chance[df["won"] == 1].clip(1e-9)).mean())


def bets(df: pd.DataFrame, w: float = 1.0, min_edge: float = betting.MIN_EDGE, cap: float = 1.0,
         price: str = "best") -> pd.DataFrame:
    """The rule's bets: in games within `cap` of the market, the best result by blended edge,
    if it's at least min_edge; with Kelly stakes and the profit of $1."""
    d = df[df["gap"] <= cap].copy()
    if d.empty:
        return d.assign(price=[], edge=[], kelly=[], profit=[])
    d["pb"] = blend(d, w)
    d["price"] = d[price]
    d["edge"] = d["pb"] * d["price"] - 1
    best = d.sort_values("edge", ascending=False).drop_duplicates("game_id")
    best = best[best["edge"] >= min_edge].copy()
    best["kelly"] = kelly(best["pb"], best["price"])
    best["profit"] = np.where(best["push"] == 1, 0.0, np.where(best["won"] == 1, best["price"] - 1, -1.0))
    return best


def score(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0, "won": np.nan, "roi": np.nan, "roi_low": np.nan, "roi_high": np.nan, "kelly_final": 1000.0}
    low, high = roi_range(b["profit"].to_numpy())
    return {"bets": len(b), "won": float((b["profit"] > 0).mean()), "roi": float(b["profit"].mean()),
            "roi_low": low, "roi_high": high, "kelly_final": simulate(b)["final"]}


def tune(df: pd.DataFrame, w: float) -> tuple[float, float, dict] | None:
    """The minimum edge and cap that grew a quarter-Kelly bankroll most (at least MIN_BETS bets)."""
    best = None
    for min_edge in MIN_EDGES:
        for cap in CAPS:
            s = score(bets(df, w, min_edge, cap))
            if s["bets"] >= MIN_BETS and (best is None or s["kelly_final"] > best[2]["kelly_final"]):
                best = (min_edge, cap, s)
    return best


def experiments(conn, league: str) -> list[dict]:
    df = table(conn, league)
    if df.empty:
        return []
    tuning, test = df[~df["held_out"]], df[df["held_out"]]
    if tuning.empty or test.empty:
        return []
    name = sports()[league]["name"]
    seasons = f"{int(test['season'].min())}–{int(test['season'].max())}"
    tuned_seasons = f"{int(tuning['season'].min())}–{int(tuning['season'].max())}"
    losses = {w: log_loss(tuning, blend(tuning, w)) for w in W_GRID}
    w = min(losses, key=losses.get)
    model, market, blended = log_loss(test, test["p"]), log_loss(test, test["q"]), log_loss(test, blend(test, w))
    out = []

    def add(rule: str, s: dict, kept: bool = False, note: str = "", on: str = seasons, in_sample: bool = False):
        out.append({"league": league, "sport": name, "rule": rule, "seasons": on, **s, "kept": kept,
                    "in_sample": in_sample, "note": note})

    if league == "EPL":
        add("Original rule, at the average price", score(bets(test, price="avg")))
        add("Bet at the best price", score(bets(test)), kept=True,
            note="Now used: the best price across bookmakers, not the average.")
    else:
        add("Current rule", score(bets(test)), kept=True, note="Only one price is available.")
    add(f"Skip games more than {SKIP_OVER * 100:.0f} pts off the market", score(bets(test, cap=SKIP_OVER)))
    add(f"Blend: {w * 100:.0f}% model, {(1 - w) * 100:.0f}% market", score(bets(test, w)),
        note=f"Held-out log loss: model {model:.4f}, market {market:.4f}, blend {blended:.4f} (lower is better).")
    found = tune(tuning, w)
    if found:
        min_edge, cap, s = found
        label = f"Tuned: {w * 100:.0f}% blend, edges {min_edge * 100:.0f}%+" + ("" if cap >= 1 else f", skip >{cap * 100:.0f} pts")
        add(label, s, on=tuned_seasons, in_sample=True, note="The seasons it was tuned on: not a fair test.")
        add(label, score(bets(test, w, min_edge, cap)), note="The same rule on new seasons.")
    return out


def run(conn) -> pd.DataFrame:
    return pd.DataFrame([row for league in LEAGUES for row in experiments(conn, league)])


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    df = run(conn)
    conn.close()
    for _, r in df.iterrows():
        print(f"{r['sport']:<17} {r['rule']:<45} {r['seasons']:>9}  {r['bets']:>5} bets  "
              + (f"{r['roi']:+.1%} per bet ({r['roi_low']:+.1%} to {r['roi_high']:+.1%}), $1,000 -> ${r['kelly_final']:,.0f}"
                 if r["bets"] else "no bets"))
    if "--record" in argv and not df.empty:
        RECORD_PATH.parent.mkdir(exist_ok=True)
        df.round(4).to_csv(RECORD_PATH, index=False)
        print(f"Saved {RECORD_PATH}.")


if __name__ == "__main__":
    main()
