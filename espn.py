"""Betting lines from ESPN's public scoreboard feed, saved as the season goes (there are no free
historical lines for the NBA or college basketball, and the college football source has no
current-season lines).

Usage:
    python -m espn             # fetch the lines for recent and upcoming games
    python -m espn --record    # ...and update the public record, records/espn_lines.csv

ESPN shows one sportsbook's lines (DraftKings at the moment): the moneyline, spread and total, as
they opened and their latest (the closing line, once the game has started). Each run saves the
latest for every game from 3 days ago to 8 days ahead, matched to our games by ESPN's game id
(the id every source here uses for college and NBA games). The record keeps them, so the market
comparison builds up over the season, and survives rebuilding the database.

ESPN drops a game's lines once it's over, so the closing line is the last one saved before the game:
when a game is final, its saved line is marked as closing (mark_closed). With updates at 10:07 and
17:37 that's within a few hours of the start for most games.

The feed is ESPN's public API (site.api.espn.com), the one its own website uses. It has no
documented terms, so it's used lightly: a few requests per update. If it's unreachable, the update
carries on without lines.
"""

import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from db import DB_PATH, SCHEMA_PATH

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
LEAGUES = {   # our league code: ESPN's path, extra parameters, our table with ESPN game ids
    "NBA": ("basketball/nba", {"limit": "500"}, "nba_game_info"),
    "CBB": ("basketball/mens-college-basketball", {"groups": "50", "limit": "1000"}, "cbb_game_info"),
    "CFB": ("football/college-football", {"groups": "80", "limit": "500"}, "cfb_game_info"),
}
DAYS_BACK, DAYS_AHEAD = 3, 8
RECORD_PATH = Path("records/espn_lines.csv")
COLUMNS = ["league", "game", "date", "home", "away", "provider", "home_ml_open", "away_ml_open", "home_ml", "away_ml",
           "spread", "total", "final", "updated_at"]


# ---------------------------------------------------------------- reading the feed

def decimal_odds(american) -> float | None:
    """'+130' -> 2.30, '-155' -> 1.645, 'EVEN' -> 2.0; None for anything else ('OFF', missing)."""
    text = str(american).strip().upper() if american is not None else ""
    if text in ("EVEN", "EV", "PK"):
        return 2.0
    try:
        value = float(text.replace("+", ""))
    except ValueError:
        return None
    if value >= 100:
        return 1 + value / 100
    if value <= -100:
        return 1 + 100 / -value
    return None


def number(text) -> float | None:
    """'-3.5' -> -3.5, 'o235.5' -> 235.5, 'PK' -> 0.0."""
    value = str(text).strip().lower().lstrip("ou") if text is not None else ""
    if value in ("pk", "even"):
        return 0.0
    try:
        return float(value.replace("+", ""))
    except ValueError:
        return None


def dig(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def parse_event(event: dict) -> dict | None:
    """One game's lines from a scoreboard event; None if it has none."""
    comp = (event.get("competitions") or [{}])[0]
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None
    odds = odds_list[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    home_ml = dig(odds, "moneyline", "home", "close", "odds") or dig(odds, "homeTeamOdds", "moneyLine")
    away_ml = dig(odds, "moneyline", "away", "close", "odds") or dig(odds, "awayTeamOdds", "moneyLine")
    spread = number(dig(odds, "pointSpread", "home", "close", "line"))
    total = number(dig(odds, "total", "over", "close", "line"))
    row = {"game": str(event.get("id")), "provider": dig(odds, "provider", "name"),
           "home": dig(sides.get("home", {}), "team", "displayName"), "away": dig(sides.get("away", {}), "team", "displayName"),
           "home_ml_open": decimal_odds(dig(odds, "moneyline", "home", "open", "odds")),
           "away_ml_open": decimal_odds(dig(odds, "moneyline", "away", "open", "odds")),
           "home_ml": decimal_odds(home_ml), "away_ml": decimal_odds(away_ml),
           "spread": spread if spread is not None else odds.get("spread"),
           "total": total if total is not None else odds.get("overUnder"),
           "final": int(dig(event, "status", "type", "state") in ("in", "post"))}
    if row["home_ml"] is None and row["spread"] is None and row["total"] is None:
        return None
    return row


def fetch_events(league: str, start: date, end: date, get=requests.get) -> list[dict]:
    """Every event from start to end, one day at a time (the feed refuses date ranges)."""
    path, params, _ = LEAGUES[league]
    events = {}
    day = start
    while day <= end:
        response = get(SCOREBOARD.format(path=path), params={**params, "dates": f"{day:%Y%m%d}"}, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"ESPN answered {response.status_code} for {day:%Y-%m-%d}")
        for event in response.json().get("events", []):
            events[str(event.get("id"))] = event
        day += timedelta(days=1)
    return list(events.values())


# ---------------------------------------------------------------- saving

def save(conn, league: str, rows: list[dict]) -> int:
    """Store each game's latest lines (matched to our games by ESPN id). Returns games saved."""
    info = LEAGUES[league][2]
    if conn.execute("SELECT name FROM sqlite_master WHERE name = ?", (info,)).fetchone() is None:
        return 0
    saved = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for r in rows:
        found = conn.execute(f"SELECT i.game_id, g.date FROM {info} i JOIN games g ON g.id = i.game_id "
                             f"WHERE i.source_game_id = ?", (r["game"],)).fetchone()
        if found is None:
            continue
        old = conn.execute("SELECT final FROM espn_lines WHERE league = ? AND source_game_id = ?", (league, r["game"])).fetchone()
        if old is not None and old[0] == 1 and not r["final"]:
            continue          # never replace a closing line with an earlier one
        conn.execute("""INSERT OR REPLACE INTO espn_lines (league, source_game_id, game_id, date, provider, home_ml_open,
                                                          away_ml_open, home_ml, away_ml, spread, total, final, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (league, r["game"], found[0], found[1], r["provider"], r["home_ml_open"], r["away_ml_open"],
                      r["home_ml"], r["away_ml"], r["spread"], r["total"], r["final"], r.get("updated_at") or now))
        saved += 1
    conn.commit()
    return saved


def mark_closed(conn) -> int:
    """Mark the saved line of every finished game as its closing line (the last one before the game)."""
    count = conn.execute("""UPDATE espn_lines SET final = 1 WHERE final = 0
                            AND game_id IN (SELECT id FROM games WHERE status = 'final')""").rowcount
    conn.commit()
    return count


def market_lines(conn, league: str) -> pd.DataFrame:
    """For every game with ESPN lines: market_home (the moneyline's home win chance, margin
    removed), the home spread and the total. Empty if none."""
    try:
        df = pd.read_sql("SELECT game_id AS id, home_ml, away_ml, spread, total, final FROM espn_lines WHERE league = ?",
                         conn, params=(league,))
    except Exception:
        return pd.DataFrame(columns=["id", "market_home", "market_spread", "market_total", "market_final"])
    home, away = 1 / df["home_ml"], 1 / df["away_ml"]
    return pd.DataFrame({"id": df["id"], "market_home": home / (home + away), "market_spread": df["spread"],
                         "market_total": df["total"], "market_final": df["final"]})


def restore(conn) -> int:
    """Copy the recorded lines into the database (the database isn't in the repository)."""
    if not RECORD_PATH.exists():
        return 0
    rows = pd.read_csv(RECORD_PATH, dtype={"game": str})
    count = 0
    for league, group in rows.groupby("league"):
        if league in LEAGUES:
            count += save(conn, league, group.astype(object).where(group.notna(), None).to_dict("records"))
    return count


def export(conn) -> int:
    current = pd.read_sql("""SELECT l.league, l.source_game_id AS game, l.date, h.name AS home, a.name AS away, l.provider,
                                    l.home_ml_open, l.away_ml_open, l.home_ml, l.away_ml, l.spread, l.total, l.final, l.updated_at
                             FROM espn_lines l JOIN games g ON g.id = l.game_id
                             JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id""", conn)
    old = pd.read_csv(RECORD_PATH, dtype={"game": str}) if RECORD_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    combined = (pd.concat([old.reindex(columns=COLUMNS), current[COLUMNS]], ignore_index=True)
                  .drop_duplicates(["league", "game"], keep="last").sort_values(["league", "date", "game"]))
    RECORD_PATH.parent.mkdir(exist_ok=True)
    combined.round(4).to_csv(RECORD_PATH, index=False)
    return len(combined)


def main(argv: list[str] | None = None, today: date | None = None, get=requests.get) -> None:
    argv = sys.argv[1:] if argv is None else argv
    today = today or date.today()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    restore(conn)
    for league in LEAGUES:
        info = LEAGUES[league][2]
        has_games = conn.execute(f"SELECT name FROM sqlite_master WHERE name = '{info}'").fetchone() and \
            conn.execute(f"SELECT 1 FROM {info} i JOIN games g ON g.id = i.game_id WHERE g.date BETWEEN ? AND ? LIMIT 1",
                         ((today - timedelta(days=DAYS_BACK)).isoformat(), (today + timedelta(days=DAYS_AHEAD)).isoformat())).fetchone()
        if not has_games:
            continue
        try:
            events = fetch_events(league, today - timedelta(days=DAYS_BACK), today + timedelta(days=DAYS_AHEAD), get)
        except Exception as error:        # ESPN down or unreachable: carry on without lines
            print(f"{league}: couldn't get lines from ESPN ({error}); skipped.")
            continue
        rows = [r for r in (parse_event(e) for e in events) if r]
        print(f"{league}: lines for {save(conn, league, rows)} games.")
    closed = mark_closed(conn)
    if closed:
        print(f"Closing lines for {closed} finished games.")
    if "--record" in argv:
        export(conn)
    conn.close()


if __name__ == "__main__":
    main()
