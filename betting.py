"""Betting: where the model's chance and the bookmaker's price give a positive expected return,
how much to stake, and how that strategy has done.

The rule, for every sport:

1. For each possible result, the expected return of a $1 bet is   model chance x decimal price - 1.
   A bet is suggested when that's at least MIN_EDGE (3%), at most one per game (the best one).
2. The stake is a fraction of the Kelly criterion: the full Kelly stake, the share of a bankroll
   that maximises its long-run growth if the chances are right, is  edge / (price - 1). Full Kelly
   is far too aggressive when the chances are estimates, so the site uses a quarter of it
   (KELLY_FRACTION), and never more than MAX_STAKE (3%) of the bankroll on one bet.
3. Prices: the Premier League's best bookmaker price before the match ('Market max' on
   football-data.co.uk: shopping around is worth about 6 points per bet in the held-out test);
   NFL moneylines (nflverse); NBA, college basketball and this season's college football from
   ESPN (DraftKings). For past college football seasons only a consensus with the margin taken
   out is available, so a typical margin (ASSUMED_MARGIN) is put back.

How it has done: the same rule on the held-out seasons of each sport's backtest (predictions made
from earlier games only), against the prices of the time. Live: every bet is logged the first time
it's suggested, at that moment's price, and never changed (records/bets.csv); it's settled when the
game ends. Closing-line value (did the price move toward the model before kickoff?) is measured
where a closing price is known: Pinnacle's for the Premier League, ESPN's for its sports.

Usage:
    python -m betting             # log today's suggested bets and the lines' movement (after the predictions)
    python -m betting --record    # ...and the public records, records/bets.csv and records/line_history.csv
    python -m betting --history   # how the rule did on past seasons, per sport

This is a statistics project, not betting advice: the model hasn't been shown to beat the market.
Other changes to the rule (blending with the market, skipping big disagreements, tuning the minimum
edge) were tested and didn't hold up on the held-out seasons: betting_experiments.py.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from db import DB_PATH, SCHEMA_PATH

MIN_EDGE = 0.03
KELLY_FRACTION = 0.25
MAX_STAKE = 0.03
ASSUMED_MARGIN = 0.045       # a typical bookmaker margin on a two-way moneyline
EPL_PRICE = "Market max"     # the best price across bookmakers (the average if it's missing)
START_BANKROLL = 1000.0
EDGE_BUCKETS = [0.03, 0.06, 0.10, 0.20, np.inf]
RECORD_PATH = Path("records/bets.csv")
HISTORY_PATH = Path("records/line_history.csv")
MOVE = 0.002                 # a new snapshot only when a chance moved by at least this much


def sports() -> dict:
    """Each sport's league code, name, live model, backtest and held-out seasons."""
    from cbb.predict_upcoming import MODEL_NAME as CBB_LIVE
    from cbb.run_backtest import BACKTEST_NAME as CBB_BT, BACKTEST_VERSION as CBB_BV, HELD_OUT_FROM as CBB_HO
    from cfb.predict_upcoming import MODEL_NAME as CFB_LIVE
    from cfb.run_backtest import BACKTEST_NAME as CFB_BT, BACKTEST_VERSION as CFB_BV, HELD_OUT_FROM as CFB_HO
    from epl.predict_upcoming import MODEL_NAME as EPL_LIVE
    from epl.run_backtest import BACKTEST_NAME as EPL_BT, BACKTEST_VERSION as EPL_BV, HELD_OUT_FROM as EPL_HO
    from nba.predict_upcoming import MODEL_NAME as NBA_LIVE
    from nba.run_backtest import BACKTEST_NAME as NBA_BT, BACKTEST_VERSION as NBA_BV, HELD_OUT_FROM as NBA_HO
    from nfl.predict_upcoming import MODEL_NAME as NFL_LIVE
    from nfl.run_backtest import BACKTEST_NAME as NFL_BT, BACKTEST_VERSION as NFL_BV, HELD_OUT_FROM as NFL_HO
    return {
        "EPL": {"name": "Premier League", "live": EPL_LIVE, "backtest": (EPL_BT, EPL_BV), "held_out": EPL_HO, "url": "/"},
        "NFL": {"name": "NFL", "live": NFL_LIVE, "backtest": (NFL_BT, NFL_BV), "held_out": NFL_HO, "url": "/nfl/"},
        "NBA": {"name": "NBA", "live": NBA_LIVE, "backtest": (NBA_BT, NBA_BV), "held_out": NBA_HO, "url": "/nba/"},
        "CFB": {"name": "College football", "live": CFB_LIVE, "backtest": (CFB_BT, CFB_BV), "held_out": CFB_HO, "url": "/cfb/"},
        "CBB": {"name": "College basketball", "live": CBB_LIVE, "backtest": (CBB_BT, CBB_BV), "held_out": CBB_HO, "url": "/cbb/"},
    }


# ---------------------------------------------------------------- reading

def _has(conn, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE name = ?", (table,)).fetchone() is not None


def american_to_decimal(value) -> float | None:
    if value is None or pd.isna(value) or value == 0:
        return None
    value = float(value)
    return 1 + value / 100 if value > 0 else 1 + 100 / -value


def decimal_to_american(price: float) -> str:
    if price >= 2:
        return f"+{(price - 1) * 100:.0f}"
    return f"{-100 / (price - 1):.0f}"


def games(conn, league: str) -> pd.DataFrame:
    """The league's games: id, season, date, kickoff, status, home, away, scores."""
    return pd.read_sql("""SELECT g.id AS game_id, g.season, g.date, g.kickoff, g.status, h.name AS home, a.name AS away,
                                 g.home_score, g.away_score
                          FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = ?
                          JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id""", conn, params=(league,))


def model_chances(conn, model_name: str, version: str | None = None) -> pd.DataFrame:
    """game_id, outcome, p: the model's chances (any version: each game's latest)."""
    df = pd.read_sql("""SELECT game_id, outcome, probability AS p, model_version, created_at FROM predictions
                        WHERE model_name = ? AND (? IS NULL OR model_version = ?)""", conn, params=(model_name, version, version))
    if df.empty:
        return df[["game_id", "outcome", "p"]]
    latest = df.sort_values(["created_at", "model_version"]).groupby("game_id")["model_version"].last()
    df = df[df["model_version"] == df["game_id"].map(latest)]
    return df.drop_duplicates(["game_id", "outcome"], keep="last")[["game_id", "outcome", "p"]]


def espn_prices(conn, league: str, closing: bool = False) -> pd.DataFrame:
    """ESPN's moneylines. closing: only games that have started, whose saved line is the last one
    before the game (ESPN drops the lines once a game is over, so the last one saved is the close)."""
    if not _has(conn, "espn_lines"):
        return pd.DataFrame(columns=["game_id", "outcome", "price"])
    df = pd.read_sql("""SELECT e.game_id, e.home_ml, e.away_ml, (e.final = 1 OR g.status = 'final') AS closed
                        FROM espn_lines e JOIN games g ON g.id = e.game_id WHERE e.league = ?""", conn, params=(league,))
    if closing:
        df = df[df["closed"] == 1]
    out = pd.concat([df[["game_id", "home_ml"]].set_axis(["game_id", "price"], axis=1).assign(outcome="home"),
                     df[["game_id", "away_ml"]].set_axis(["game_id", "price"], axis=1).assign(outcome="away")])
    return out.dropna(subset=["price"])


def prices(conn, league: str) -> pd.DataFrame:
    """game_id, outcome, price (decimal, bookmaker's margin included): what a bettor could get
    before the game."""
    if league == "EPL":
        df = pd.read_sql("""SELECT game_id, outcome, bookmaker, price FROM odds WHERE market = '1x2' AND is_closing = 0
                            AND bookmaker IN ('Market average', ?)""", conn, params=(EPL_PRICE,))
        df = df.sort_values("bookmaker", key=lambda b: b == EPL_PRICE).drop_duplicates(["game_id", "outcome"], keep="last")
        return df[["game_id", "outcome", "price"]].reset_index(drop=True)
    if league == "NFL":
        if not _has(conn, "nfl_game_info"):
            return pd.DataFrame(columns=["game_id", "outcome", "price"])
        df = pd.read_sql("SELECT game_id, home_moneyline, away_moneyline FROM nfl_game_info", conn)
        out = pd.concat([pd.DataFrame({"game_id": df["game_id"], "outcome": side,
                                       "price": df[f"{side}_moneyline"].map(american_to_decimal)}) for side in ("home", "away")])
        return out.dropna(subset=["price"])
    espn = espn_prices(conn, league)
    if league == "CFB":
        consensus = pd.read_sql("""SELECT game_id, outcome, price FROM odds WHERE bookmaker = 'consensus'
                                   AND market = 'moneyline'""", conn)
        # the consensus has the margin taken out (price = 1 / chance): put a typical one back
        consensus["price"] = 1 / ((1 / consensus["price"]) * (1 + ASSUMED_MARGIN))
        consensus = consensus[~consensus["game_id"].isin(espn["game_id"])]
        return pd.concat([espn, consensus], ignore_index=True)
    return espn


def fair_closing(conn, league: str, live: bool = False) -> pd.DataFrame:
    """game_id, outcome, q_close: the closing price's chance, margin removed (for closing-line value).
    live: for logged bets, where the NFL's moneylines at kickoff (nflverse) are the close; the past
    seasons' NFL test already bets at those, so it has no closing-line value."""
    if league == "NFL" and live and _has(conn, "nfl_game_info"):
        df = prices(conn, "NFL")
        done = pd.read_sql("SELECT id AS game_id FROM games WHERE status = 'final'", conn)
        df = df[df["game_id"].isin(done["game_id"])]
    elif league == "EPL":
        df = pd.read_sql("""SELECT game_id, outcome, price FROM odds WHERE bookmaker = 'Pinnacle'
                            AND market = '1x2' AND is_closing = 1""", conn)
    elif league in ("NBA", "CBB", "CFB"):
        df = espn_prices(conn, league, closing=True)
    else:
        return pd.DataFrame(columns=["game_id", "outcome", "q_close"])
    if df.empty:
        return pd.DataFrame(columns=["game_id", "outcome", "q_close"])
    implied = 1 / df["price"]
    return df.assign(q_close=implied / implied.groupby(df["game_id"]).transform("sum"))[["game_id", "outcome", "q_close"]]


# ---------------------------------------------------------------- the rule

def kelly(p, price) -> np.ndarray:
    """The full Kelly stake as a share of the bankroll (0 when there's no edge)."""
    p, price = np.asarray(p, float), np.asarray(price, float)
    return np.clip((p * price - 1) / (price - 1), 0, None)


def stake_share(full_kelly, fraction: float = KELLY_FRACTION, cap: float = MAX_STAKE) -> np.ndarray:
    return np.minimum(np.asarray(full_kelly, float) * fraction, cap)


def candidates(chances: pd.DataFrame, offered: pd.DataFrame, min_edge: float = MIN_EDGE) -> pd.DataFrame:
    """The best bet in each game with an edge of at least min_edge: game_id, outcome, p, price,
    market (the price's chance with the margin removed), edge, kelly."""
    df = chances.merge(offered, on=["game_id", "outcome"])
    if df.empty:
        return pd.DataFrame(columns=["game_id", "outcome", "p", "price", "market", "edge", "kelly"])
    implied = 1 / df["price"]
    df["market"] = implied / implied.groupby(df["game_id"]).transform("sum")
    df["edge"] = df["p"] * df["price"] - 1
    df["kelly"] = kelly(df["p"], df["price"])
    best = df.sort_values("edge", ascending=False).drop_duplicates("game_id")
    return best[best["edge"] >= min_edge].reset_index(drop=True)


def winner(row) -> str | None:
    if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
        return None
    if row["home_score"] > row["away_score"]:
        return "home"
    return "away" if row["away_score"] > row["home_score"] else "draw"


def settle(bets: pd.DataFrame, game_rows: pd.DataFrame, three_way: bool = False) -> pd.DataFrame:
    """Adds result ('won'/'lost'/'push', or None if not played yet) and profit per unit staked.
    three_way: the market has a draw (the Premier League); otherwise a tie refunds the bet."""
    keep = [c for c in bets.columns if c not in ("status", "home_score", "away_score")]
    df = bets[keep].merge(game_rows[["game_id", "status", "home_score", "away_score"]], on="game_id", how="left")
    results = []
    for _, r in df.iterrows():
        if r["status"] != "final":
            results.append(None)
            continue
        w = winner(r)
        results.append("push" if w == "draw" and not three_way else "won" if w == r["outcome"] else "lost")
    df["result"] = results
    df["profit"] = np.select([df["result"] == "won", df["result"] == "lost"], [df["price"] - 1, -1.0], 0.0)
    df.loc[df["result"].isna(), "profit"] = np.nan
    return df


# ---------------------------------------------------------------- past seasons

def simulate(settled: pd.DataFrame, fraction: float = KELLY_FRACTION, cap: float = MAX_STAKE,
             bankroll: float = START_BANKROLL) -> dict:
    """Bets day by day: each day's stakes are set from the bankroll at the start of the day."""
    path, peak, worst = [], bankroll, 0.0
    for day, group in settled.sort_values("date").groupby("date"):
        stakes = stake_share(group["kelly"], fraction, cap) * bankroll
        bankroll += float((stakes * group["profit"]).sum())
        peak = max(peak, bankroll)
        worst = max(worst, 1 - bankroll / peak)
        path.append({"date": day, "bankroll": bankroll})
    return {"final": bankroll, "drawdown": worst, "path": path}


def roi_range(profit: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% range of the return per unit staked, from resampling the bets (how much luck could do)."""
    if len(profit) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(profit, size=(draws, len(profit)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def history(conn, league: str, min_edge: float = MIN_EDGE) -> dict | None:
    """The rule on the sport's held-out backtest seasons (before the current one)."""
    info = sports()[league]
    rows = games(conn, league)
    if rows.empty:
        return None
    current = int(rows["season"].max())
    chances = model_chances(conn, *info["backtest"])
    picks = candidates(chances, prices(conn, league), min_edge)
    picks = picks.merge(rows, on="game_id")
    picks = picks[(picks["season"] >= info["held_out"]) & (picks["season"] < current) & (picks["status"] == "final")]
    if picks.empty:
        return None
    settled = settle(picks, rows, three_way=league == "EPL").dropna(subset=["profit"])
    settled = settled.merge(fair_closing(conn, league), on=["game_id", "outcome"], how="left")
    profit = settled["profit"].to_numpy()
    low, high = roi_range(profit)
    buckets = []
    for lo, hi in zip(EDGE_BUCKETS[:-1], EDGE_BUCKETS[1:]):
        part = settled[(settled["edge"] >= lo) & (settled["edge"] < hi)]
        if len(part):
            buckets.append({"edges": f"{lo * 100:.0f}%+" if np.isinf(hi) else f"{lo * 100:.0f}-{hi * 100:.0f}%",
                            "bets": len(part), "expected": float(part["edge"].mean()), "roi": float(part["profit"].mean())})
    sim = simulate(settled)
    clv = settled.dropna(subset=["q_close"])
    held = rows[(rows["season"] >= info["held_out"]) & (rows["season"] < current) & (rows["status"] == "final")]
    games_priced = chances[chances["game_id"].isin(held["game_id"])].merge(prices(conn, league), on=["game_id", "outcome"])["game_id"].nunique()
    return {"league": league, "name": info["name"], "seasons": (info["held_out"], current - 1), "bets": len(settled),
            "games": games_priced, "won": int((settled["result"] == "won").sum()), "expected": float(settled["edge"].mean()),
            "roi": float(profit.mean()), "roi_low": low, "roi_high": high, "average_price": float(settled["price"].mean()),
            "kelly_final": sim["final"], "drawdown": sim["drawdown"], "path": sim["path"], "buckets": buckets,
            "clv": float((clv["price"] * clv["q_close"] - 1).mean()) if len(clv) else None, "clv_bets": len(clv),
            "assumed_margin": league == "CFB"}


# ---------------------------------------------------------------- live

COLUMNS = ["league", "date", "home", "away", "outcome", "p", "market", "price", "edge", "kelly", "suggested_at"]


def suggestions(conn, now: datetime | None = None) -> pd.DataFrame:
    """Every bet the rule suggests right now, for games that haven't started, across sports."""
    from nfl.predict_upcoming import EASTERN, has_started
    now = now or datetime.now(EASTERN)
    out = []
    for league, info in sports().items():
        rows = games(conn, league)
        if rows.empty:
            continue
        upcoming = rows[(rows["status"] == "scheduled") & (rows["date"] >= now.astimezone(EASTERN).date().isoformat())]
        upcoming = upcoming[[not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)
                             for g in upcoming.itertuples()]]
        if upcoming.empty:
            continue
        chances = model_chances(conn, info["live"])
        chances = chances[chances["game_id"].isin(upcoming["game_id"])]
        picks = candidates(chances, prices(conn, league))
        if not picks.empty:
            out.append(picks.merge(upcoming, on="game_id").assign(league=league))
    if not out:
        return pd.DataFrame(columns=["league", "game_id", "date", "kickoff", "home", "away", "outcome", "p", "market",
                                     "price", "edge", "kelly"])
    return pd.concat(out, ignore_index=True).sort_values(["date", "kickoff", "edge"], ascending=[True, True, False])


def record(conn, now: datetime | None = None) -> int:
    """Log each suggested bet the first time it's suggested (at that moment's price); never changed."""
    from nfl.predict_upcoming import EASTERN
    now = now or datetime.now(EASTERN)
    picks = suggestions(conn, now)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    before = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
    conn.executemany("""INSERT OR IGNORE INTO bets (league, game_id, outcome, p, market, price, edge, kelly, suggested_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     [(r.league, int(r.game_id), r.outcome, float(r.p), float(r.market), float(r.price), float(r.edge),
                       float(r.kelly), stamp) for r in picks.itertuples()])
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0] - before


def logged(conn) -> pd.DataFrame:
    """Every logged bet with its game, settled where the game has finished, and its closing-line value."""
    if not _has(conn, "bets"):
        return pd.DataFrame()
    bets = pd.read_sql("SELECT * FROM bets", conn)
    if bets.empty:
        return bets
    out = []
    for league, group in bets.groupby("league"):
        rows = games(conn, league)
        settled = settle(group, rows, three_way=league == "EPL").merge(rows[["game_id", "date", "kickoff", "home", "away"]], on="game_id")
        out.append(settled.merge(fair_closing(conn, league, live=True), on=["game_id", "outcome"], how="left"))
    df = pd.concat(out, ignore_index=True)
    df["clv"] = df["price"] * df["q_close"] - 1
    return df.sort_values(["date", "kickoff"])


# ---------------------------------------------------------------- line movement

def fair_chances(offered: pd.DataFrame) -> pd.DataFrame:
    """game_id, home, draw (NaN for two-way), away: the price's chances, margin removed."""
    if offered.empty:
        return pd.DataFrame(columns=["game_id", "home", "draw", "away"])
    implied = 1 / offered["price"]
    df = offered.assign(q=implied / implied.groupby(offered["game_id"]).transform("sum"))
    wide = df.pivot_table(index="game_id", columns="outcome", values="q").reset_index()
    for col in ("home", "draw", "away"):
        if col not in wide:
            wide[col] = np.nan
    return wide[["game_id", "home", "draw", "away"]]


def snapshot_lines(conn, now: datetime | None = None) -> int:
    """Save the market's and the model's current chances for every game that hasn't started, when
    either moved since the last snapshot (and ESPN's opening line, the first time). Returns rows saved."""
    from nfl.predict_upcoming import EASTERN, has_started
    now = now or datetime.now(EASTERN)
    stamp = now.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M")
    saved = 0
    last = pd.read_sql("""SELECT h.* FROM line_history h JOIN (SELECT game_id, MAX(taken_at) AS t FROM line_history
                          WHERE taken_at != 'open' GROUP BY game_id) m ON m.game_id = h.game_id AND m.t = h.taken_at""",
                       conn).set_index("game_id")
    opened = {r[0] for r in conn.execute("SELECT game_id FROM line_history WHERE taken_at = 'open'")}
    for league, info in sports().items():
        rows = games(conn, league)
        upcoming = rows[(rows["status"] == "scheduled") & (rows["date"] >= now.astimezone(EASTERN).date().isoformat())]
        upcoming = upcoming[[not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)
                             for g in upcoming.itertuples()]]
        if upcoming.empty:
            continue
        market = fair_chances(prices(conn, league)).set_index("game_id")
        model = model_chances(conn, info["live"]).pivot_table(index="game_id", columns="outcome", values="p")
        if league in ("NBA", "CBB", "CFB") and _has(conn, "espn_lines"):
            opens = pd.read_sql("SELECT game_id, home_ml_open, away_ml_open FROM espn_lines WHERE league = ?",
                                conn, params=(league,)).dropna()
            for r in opens[opens["game_id"].isin(upcoming["game_id"]) & ~opens["game_id"].isin(opened)].itertuples():
                h, a = 1 / r.home_ml_open, 1 / r.away_ml_open
                conn.execute("INSERT OR IGNORE INTO line_history (league, game_id, taken_at, market_home) VALUES (?, ?, 'open', ?)",
                             (league, int(r.game_id), h / (h + a)))
                saved += 1
        for gid in upcoming["game_id"]:
            mh = float(market.at[gid, "home"]) if gid in market.index else None
            md = float(market.at[gid, "draw"]) if gid in market.index and pd.notna(market.at[gid, "draw"]) else None
            ph = float(model.at[gid, "home"]) if gid in model.index and "home" in model and pd.notna(model.at[gid, "home"]) else None
            pd_ = float(model.at[gid, "draw"]) if gid in model.index and "draw" in model and pd.notna(model.at[gid, "draw"]) else None
            if mh is None and ph is None:
                continue
            if gid in last.index:
                before = last.loc[gid]
                moved = lambda new, old: (new is None) != pd.isna(old) or (new is not None and abs(new - old) >= MOVE)
                if not any(moved(n, o) for n, o in ((mh, before["market_home"]), (md, before["market_draw"]),
                                                     (ph, before["model_home"]), (pd_, before["model_draw"]))):
                    continue
            conn.execute("""INSERT OR REPLACE INTO line_history (league, game_id, taken_at, market_home, market_draw,
                                                                 model_home, model_draw) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                         (league, int(gid), stamp, mh, md, ph, pd_))
            saved += 1
    conn.commit()
    return saved


HISTORY_COLUMNS = ["league", "date", "home", "away", "taken_at", "market_home", "market_draw", "model_home", "model_draw"]


def restore_history(conn) -> int:
    if not HISTORY_PATH.exists():
        return 0
    rec = pd.read_csv(HISTORY_PATH)
    ids = {}
    for league in rec["league"].unique():
        g = games(conn, league)
        ids.update({(league, r.date, r.home, r.away): r.game_id for r in g.itertuples()})
    rows = [(r.league, int(ids[(r.league, r.date, r.home, r.away)]), r.taken_at,
             *[None if pd.isna(v) else float(v) for v in (r.market_home, r.market_draw, r.model_home, r.model_draw)])
            for r in rec.itertuples() if (r.league, r.date, r.home, r.away) in ids]
    conn.executemany("""INSERT OR IGNORE INTO line_history (league, game_id, taken_at, market_home, market_draw,
                                                            model_home, model_draw) VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
    conn.commit()
    return len(rows)


def export_history(conn) -> None:
    df = pd.read_sql("""SELECT l.league, g.date, h.name AS home, a.name AS away, l.taken_at, l.market_home, l.market_draw,
                               l.model_home, l.model_draw
                        FROM line_history l JOIN games g ON g.id = l.game_id
                        JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id""", conn)
    if df.empty:
        return
    old = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame(columns=HISTORY_COLUMNS)
    combined = (pd.concat([old.reindex(columns=HISTORY_COLUMNS), df[HISTORY_COLUMNS]], ignore_index=True)
                  .drop_duplicates(["league", "date", "home", "away", "taken_at"], keep="last")
                  .sort_values(["date", "league", "home", "taken_at"]))
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    combined.round(4).to_csv(HISTORY_PATH, index=False)


def movement(conn, game_id: int) -> pd.DataFrame:
    """A game's saved snapshots, oldest first (the opening line first)."""
    if not _has(conn, "line_history"):
        return pd.DataFrame(columns=["taken_at", "market_home", "market_draw", "model_home", "model_draw"])
    df = pd.read_sql("SELECT taken_at, market_home, market_draw, model_home, model_draw FROM line_history WHERE game_id = ?",
                     conn, params=(int(game_id),))
    return df.sort_values("taken_at", key=lambda t: t.where(t != "open", "0")).reset_index(drop=True)


def movement_chart(conn, game_id: int, home: str, away: str) -> dict | None:
    """The 'How the line moved' chart for a game page (None until there are two snapshots)."""
    from charts import movement_chart as chart
    df = movement(conn, game_id)
    return chart(df.astype(object).where(df.notna(), float("nan")).to_dict("records"), home, away) if len(df) >= 2 else None


# ---------------------------------------------------------------- the bet log

def restore(conn) -> int:
    if not RECORD_PATH.exists():
        return 0
    rec = pd.read_csv(RECORD_PATH)
    restored = 0
    for r in rec.itertuples():
        found = conn.execute("""SELECT g.id FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = ?
                                JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                                WHERE g.date = ? AND h.name = ? AND a.name = ?""", (r.league, r.date, r.home, r.away)).fetchone()
        if found:
            conn.execute("""INSERT OR IGNORE INTO bets (league, game_id, outcome, p, market, price, edge, kelly, suggested_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (r.league, found[0], r.outcome, r.p, r.market, r.price, r.edge, r.kelly, r.suggested_at))
            restored += 1
    conn.commit()
    return restored


def export(conn) -> None:
    df = logged(conn)
    if df.empty:
        return
    RECORD_PATH.parent.mkdir(exist_ok=True)
    current = df[COLUMNS]
    old = pd.read_csv(RECORD_PATH) if RECORD_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    combined = (pd.concat([old.reindex(columns=COLUMNS), current], ignore_index=True)
                  .drop_duplicates(["league", "date", "home", "away"], keep="first").sort_values(["date", "league"]))
    combined.round(4).to_csv(RECORD_PATH, index=False)


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    if "--history" in argv:
        for league in sports():
            h = history(conn, league)
            if h is None:
                print(f"{sports()[league]['name']}: no priced backtest games.")
                continue
            print(f"{h['name']} {h['seasons'][0]}-{h['seasons'][1]}: {h['bets']} bets of {h['games']} games, "
                  f"won {h['won']}, expected {h['expected']:+.1%}, actual {h['roi']:+.1%} per bet "
                  f"(95% range {h['roi_low']:+.1%} to {h['roi_high']:+.1%}); quarter-Kelly $1,000 -> ${h['kelly_final']:,.0f}, "
                  f"worst drawdown {h['drawdown']:.0%}; closing-line value "
                  + ("n/a" if h["clv"] is None else f"{h['clv']:+.1%}"))
            for b in h["buckets"]:
                print(f"    edges {b['edges']}: {b['bets']} bets, expected {b['expected']:+.1%}, actual {b['roi']:+.1%}")
        conn.close()
        return
    restore(conn)
    restore_history(conn)
    moved = snapshot_lines(conn, now)
    added = record(conn, now)
    if "--record" in argv:
        export(conn)
        export_history(conn)
    conn.close()
    print(f"Betting: {added} new suggested bets logged; line movement saved for {moved} games.")


if __name__ == "__main__":
    main()
