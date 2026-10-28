"""NBA player prop lines from the sportsbook, to test the model's props against: python -m nba.prop_lines [--record]

ESPN's feed shows DraftKings' player props: an over/under line and both prices for each player's
points, rebounds, assists and threes. There's no free history of these, so the test runs forward:
each update saves the latest line for every upcoming game until tip-off (the last one saved is the
closing line), records/nba_prop_lines.csv keeps them, and once a game is played each line is scored:

- accuracy: the log loss of the chance the over hit, the model's (its projection and spread,
  nba/players.py) against the price's with DraftKings' margin taken out;
- the betting rule (betting.py): bet the over or the under when chance x price - 1 is at least 3%.

Lines are half-points (22.5), so there are no pushes; a player who doesn't play voids the line.
The feed's two items for one line are the over then the under (checked against the matching
"22+" milestone price).
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from db import DB_PATH, SCHEMA_PATH
from espn import dig
from nfl.predict_upcoming import EASTERN, has_started

PROPS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game}/competitions/{game}"
             "/odds/{provider}/propBets")
PROVIDER = "100"          # DraftKings
STAT_TYPES = {"Total Points": "points", "Total Rebounds": "rebounds", "Total Assists": "assists",
              "Total 3-Point Field Goals": "threes"}
DAYS_AHEAD = 1
RECORD_PATH = Path("records/nba_prop_lines.csv")
COLUMNS = ["game", "player_id", "stat", "line", "over_price", "under_price", "line_open", "over_open", "under_open", "updated_at"]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse(payload: dict, game: str) -> list[dict]:
    """One row per player and stat: the line and the over and under prices (decimal), now and at the open."""
    pairs: dict = {}
    for item in payload.get("items") or []:
        stat = STAT_TYPES.get(dig(item, "type", "name"))
        ref = dig(item, "athlete", "$ref")
        if stat is None or not ref:
            continue
        player = ref.rstrip("/").split("/athletes/")[1].split("?")[0]
        pairs.setdefault((player, stat), []).append(item)
    rows = []
    for (player, stat), items in pairs.items():
        if len(items) != 2:
            continue
        over, under = items
        line = number(dig(over, "current", "target", "value"))
        if line is None or line != number(dig(under, "current", "target", "value")):
            continue
        rows.append({"game": str(game), "player_id": player, "stat": stat, "line": line,
                     "over_price": number(dig(over, "odds", "decimal", "value")), "under_price": number(dig(under, "odds", "decimal", "value")),
                     "line_open": number(dig(over, "open", "target", "value")),
                     "over_open": number(dig(over, "odds", "decimal", "open")), "under_open": number(dig(under, "odds", "decimal", "open"))})
    return [r for r in rows if r["over_price"] and r["under_price"]]


def fetch(game: str, get=requests.get) -> list[dict]:
    response = get(PROPS_URL.format(game=game, provider=PROVIDER), params={"limit": "1000", "lang": "en", "region": "us"}, timeout=60)
    if response.status_code != 200:
        return []
    return parse(response.json(), game)


def upcoming_games(conn, now: datetime) -> pd.DataFrame:
    today = now.astimezone(EASTERN).date()
    games = pd.read_sql("""SELECT i.source_game_id AS game, g.date, g.kickoff FROM games g
                           JOIN nba_game_info i ON i.game_id = g.id
                           WHERE g.status = 'scheduled' AND g.date BETWEEN ? AND ?""", conn,
                        params=(today.isoformat(), (today + timedelta(days=DAYS_AHEAD)).isoformat()))
    return games[[not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now) for g in games.itertuples()]]


def save(conn, rows: list[dict], stamp: str) -> int:
    conn.executemany(f"""INSERT OR REPLACE INTO nba_prop_lines ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})""",
                     [tuple({**r, "updated_at": stamp}[c] for c in COLUMNS) for r in rows])
    conn.commit()
    return len(rows)


def restore(conn) -> int:
    if not RECORD_PATH.exists():
        return 0
    df = pd.read_csv(RECORD_PATH, dtype={"game": str, "player_id": str})
    conn.executemany(f"INSERT OR IGNORE INTO nba_prop_lines ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                     [tuple(None if pd.isna(v) else v for v in r) for r in df[COLUMNS].itertuples(index=False)])
    conn.commit()
    return len(df)


def export(conn) -> int:
    df = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM nba_prop_lines ORDER BY game, player_id, stat", conn)
    RECORD_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RECORD_PATH, index=False)
    return len(df)


# ---------------------------------------------------------------- scoring

def compare(conn) -> pd.DataFrame:
    """Every saved line with the model's projection (the last before tip-off) and, once played,
    what happened: model and market chances of the over, and the result."""
    from nba.players import chance_at_least
    lines = pd.read_sql("SELECT * FROM nba_prop_lines", conn)
    if lines.empty:
        return lines
    proj = pd.read_sql("""SELECT i.source_game_id AS game, p.player_id, p.name, p.team, p.minutes, p.points, p.rebounds,
                                 p.assists, p.threes, g.status, g.date
                          FROM nba_player_projections p JOIN games g ON g.id = p.game_id
                          JOIN nba_game_info i ON i.game_id = g.id""", conn)
    proj = proj.melt(id_vars=["game", "player_id", "name", "team", "minutes", "status", "date"],
                     value_vars=["points", "rebounds", "assists", "threes"], var_name="stat", value_name="projected")
    df = lines.merge(proj, on=["game", "player_id", "stat"])
    if df.empty:
        return df
    df["model_over"] = np.nan
    for stat in df["stat"].unique():
        part = df["stat"] == stat
        df.loc[part, "model_over"] = chance_at_least(stat, df.loc[part, "projected"], np.ceil(df.loc[part, "line"]))
    implied_over, implied_under = 1 / df["over_price"], 1 / df["under_price"]
    df["market_over"] = implied_over / (implied_over + implied_under)
    box = pd.read_sql("""SELECT i.source_game_id AS game, b.player_id, b.minutes AS played, b.points, b.rebounds, b.assists, b.threes
                         FROM nba_player_games b JOIN nba_game_info i ON i.game_id = b.game_id""", conn)
    box = box.melt(id_vars=["game", "player_id", "played"], value_vars=["points", "rebounds", "assists", "threes"],
                   var_name="stat", value_name="actual")
    df = df.merge(box, on=["game", "player_id", "stat"], how="left")
    done = (df["status"] == "final") & (df["played"].fillna(0) > 0)
    df["over_hit"] = np.where(done, (df["actual"] > df["line"]).astype(float), np.nan)
    return df


def scores(df: pd.DataFrame, min_edge: float = 0.03) -> dict | None:
    """Accuracy and the betting rule on the settled lines."""
    done = df.dropna(subset=["over_hit"]) if not df.empty else df
    if done.empty:
        return None
    hit = done["over_hit"].to_numpy()
    ll = lambda p: float(-np.mean(np.where(hit == 1, np.log(np.clip(p, 1e-6, 1)), np.log(np.clip(1 - p, 1e-6, 1)))))
    ev_over = done["model_over"] * done["over_price"] - 1
    ev_under = (1 - done["model_over"]) * done["under_price"] - 1
    side_over = ev_over >= ev_under
    ev = np.where(side_over, ev_over, ev_under)
    bet = ev >= min_edge
    won = np.where(side_over, hit == 1, hit == 0)
    profit = np.where(won, np.where(side_over, done["over_price"], done["under_price"]) - 1, -1.0)[bet]
    from betting import roi_range
    low, high = roi_range(profit) if len(profit) else (np.nan, np.nan)
    by_stat = {s: {"lines": int((done["stat"] == s).sum()), "model": ll_part(done[done["stat"] == s], "model_over"),
                   "market": ll_part(done[done["stat"] == s], "market_over")} for s in done["stat"].unique()}
    return {"lines": len(done), "games": int(done["game"].nunique()), "model": ll(done["model_over"]), "market": ll(done["market_over"]),
            "bets": int(bet.sum()), "roi": float(profit.mean()) if len(profit) else None, "roi_low": low, "roi_high": high,
            "expected": float(ev[bet].mean()) if bet.any() else None, "by_stat": by_stat}


def ll_part(df: pd.DataFrame, col: str) -> float:
    hit = df["over_hit"].to_numpy()
    p = np.clip(df[col].to_numpy(float), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(hit == 1, np.log(p), np.log(1 - p)))) if len(df) else float("nan")


def main(argv: list[str] | None = None, now: datetime | None = None, get=requests.get) -> None:
    argv = sys.argv[1:] if argv is None else argv
    now = now or datetime.now(EASTERN)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    restore(conn)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    saved = 0
    for g in upcoming_games(conn, now).itertuples():
        try:
            saved += save(conn, fetch(g.game, get), stamp)
        except requests.RequestException as error:
            print(f"  game {g.game}: couldn't reach ESPN ({error}); carrying on.")
    if "--record" in argv:
        export(conn)
    s = scores(compare(conn))
    conn.close()
    print(f"Saved {saved} NBA prop lines (DraftKings via ESPN) for upcoming games.")
    if s:
        print(f"Settled so far: {s['lines']} lines in {s['games']} games. Log loss of the over: model {s['model']:.4f}, "
              f"market {s['market']:.4f}; the rule's {s['bets']} bets returned "
              + (f"{s['roi']:+.1%} each." if s["roi"] is not None else "nothing yet."))


if __name__ == "__main__":
    main()
