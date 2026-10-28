"""European soccer from ESPN's public soccer feed: the three European cups and the domestic
leagues of the clubs that play in them, all with ESPN's own team ids, so a club is the same club in
every competition.

Usage:
    python -m ucl.import_espn            # update: recent and upcoming days, and this season's fixtures
    python -m ucl.import_espn --history  # the first time: every season since FIRST_SEASON (15-30 minutes)

Why the domestic leagues: in the Champions League most teams meet once or twice, from many
countries, so the European games alone say little about how strong a club is. Its league games say
a lot about it relative to its league; the European games then tell how strong each league is.

The feed only answers one day at a time. Each domestic league's season comes with a calendar of
the days it has games, so only those are asked for; the European cups are asked for every day of
their season. Finished seasons are saved to data/raw/espn_soccer/ (one CSV per competition and
season) and never asked for again; the current season is refreshed around today on every update,
and in full once a week (for the fixtures further ahead).

What's saved:
- soccer_games: every game, domestic and European (the model's input)
- teams/games (leagues 'UCL', 'UEL', 'UECL', and La Liga, Bundesliga, Serie A and Ligue 1 as
  'ESP', 'GER', 'ITA', 'FRA') and cup_game_info: the games the website shows (ucl/cups.py), for
  predictions and the website, with ESPN's team colors in soccer_teams (the Premier League's own
  pages use football-data.co.uk instead, app.py)
- odds: DraftKings' three-way moneyline as ESPN shows it, for those upcoming games
  (bookmaker 'DraftKings'; is_closing 0 = when it opened, 1 = the latest, the closing price once
  the game has started)
"""

import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from db import DB_PATH, SCHEMA_PATH, insert_price
from espn import decimal_odds, dig

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard"
CACHE_DIR = Path("data/raw/espn_soccer")
FIRST_SEASON = 2013
EUROPE = {"uefa.champions": "Champions League", "uefa.europa": "Europa League",
          "uefa.europa.conf": "Conference League"}
DOMESTIC = {   # ESPN's league slug: country
    "eng.1": "England", "esp.1": "Spain", "ger.1": "Germany", "ita.1": "Italy", "fra.1": "France",
    "por.1": "Portugal", "ned.1": "Netherlands", "bel.1": "Belgium", "tur.1": "Turkey", "sco.1": "Scotland",
    "gre.1": "Greece", "aut.1": "Austria", "sui.1": "Switzerland", "den.1": "Denmark", "nor.1": "Norway",
    "swe.1": "Sweden", "isr.1": "Israel", "rou.1": "Romania", "rus.1": "Russia",
}
CALENDAR_YEAR = {"nor.1", "swe.1"}     # seasons run March to November
DAYS_BACK, DAYS_AHEAD = 4, 10
FULL_REFRESH_DAYS = 7
WORKERS = 8
COLUMNS = ["event_id", "competition", "season", "stage", "date", "kickoff", "status", "neutral", "leg",
           "home_id", "away_id", "home_score", "away_score", "home_pens", "away_pens", "country",
           "extra_time", "home_ml_open", "draw_ml_open", "away_ml_open", "home_ml", "draw_ml", "away_ml", "odds_provider"]
TEAM_COLUMNS = ["team_id", "name", "short_name", "abbreviation", "color", "alternate_color"]


# ---------------------------------------------------------------- reading the feed

def eastern(utc_text: str) -> tuple[str, str]:
    """'2026-10-21T16:45Z' -> ('2026-10-21', '12:45') in US Eastern time, like the other sports."""
    from nfl.predict_upcoming import EASTERN
    moment = datetime.strptime(utc_text.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    local = moment.astimezone(EASTERN)
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M")


def status_of(event: dict) -> str:
    """'final', 'scheduled', or 'other' (postponed, cancelled, abandoned, in progress)."""
    kind = dig(event, "status", "type") or {}
    if kind.get("completed") and kind.get("state") == "post":
        return "final"
    if kind.get("state") == "pre" and kind.get("name") in ("STATUS_SCHEDULED", "STATUS_TBD"):
        return "scheduled"
    return "other"


def score(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_events(payload: dict, competition: str) -> tuple[list[dict], list[dict]]:
    """(games, teams) from one day's scoreboard."""
    games, teams = [], []
    for event in payload.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if "home" not in sides or "away" not in sides or not event.get("date"):
            continue
        day, kickoff = eastern(event["date"])
        status = status_of(event)
        odds = next((o for o in comp.get("odds") or [] if o), {})
        ml = lambda side, when: decimal_odds(dig(odds, "moneyline", side, when, "odds"))
        row = {"event_id": str(event.get("id")), "competition": competition,
               "season": dig(event, "season", "year"), "stage": dig(event, "season", "slug"),
               "date": day, "kickoff": kickoff, "status": status, "neutral": int(bool(comp.get("neutralSite"))),
               "leg": dig(comp, "leg", "value"), "country": dig(comp, "venue", "address", "country"),
               "extra_time": int(dig(event, "status", "type", "name") in ("STATUS_FINAL_AET", "STATUS_FINAL_PEN")),
               "odds_provider": dig(odds, "provider", "name")}
        for side in ("home", "away"):
            c = sides[side]
            team = c.get("team") or {}
            row[f"{side}_id"] = str(team.get("id"))
            row[f"{side}_score"] = score(c.get("score")) if status == "final" else None
            row[f"{side}_pens"] = score(c.get("shootoutScore")) if status == "final" else None
            teams.append({"team_id": str(team.get("id")), "name": team.get("displayName"),
                          "short_name": team.get("shortDisplayName"), "abbreviation": team.get("abbreviation"),
                          "color": team.get("color"), "alternate_color": team.get("alternateColor")})
        for side in ("home", "draw", "away"):
            row[f"{side}_ml_open"] = ml(side, "open")
            row[f"{side}_ml"] = ml(side, "close")
        if row["draw_ml"] is None:
            row["draw_ml"] = decimal_odds(dig(odds, "drawOdds", "moneyLine"))
        games.append(row)
    return games, teams


class Feed:
    """The scoreboard, one day at a time, with retries."""

    def __init__(self, get=None):
        self.session = requests.Session() if get is None else None
        self.get = get or self.session.get

    def day(self, slug: str, day: date) -> dict:
        for attempt in range(4):
            try:
                response = self.get(SCOREBOARD.format(slug=slug), params={"dates": f"{day:%Y%m%d}", "limit": "200"}, timeout=30)
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (400, 404):
                    return {}
            except requests.RequestException:
                pass
            time.sleep(1 + 2 * attempt)
        raise RuntimeError(f"ESPN didn't answer for {slug} on {day}")

    def days(self, slug: str, days: list[date], competition: str) -> tuple[list[dict], list[dict]]:
        games, teams = [], []
        with ThreadPoolExecutor(WORKERS) as pool:
            for payload in pool.map(lambda d: self.day(slug, d), days):
                g, t = parse_events(payload, competition)
                games += g
                teams += t
        return games, teams


def every_day(slug: str, season: int) -> list[date]:
    first = date(season, 1, 1) if slug in CALENDAR_YEAR else date(season, 7, 1)
    last = date(season, 12, 31) if slug in CALENDAR_YEAR else date(season + 1, 6, 30)
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def season_days(feed: Feed, slug: str, season: int) -> list[date]:
    """The days a competition's season has games: the league's calendar where the feed gives one,
    otherwise every day of the season."""
    probe = date(season, 6, 15) if slug in CALENDAR_YEAR else date(season, 10, 15)
    league = (feed.day(slug, probe).get("leagues") or [{}])[0]
    if dig(league, "season", "year") not in (season, None):
        return every_day(slug, season)                     # the feed's calendar is for another season
    calendar = league.get("calendar") or []
    if calendar and isinstance(calendar[0], str):
        return sorted({datetime.strptime(d[:10], "%Y-%m-%d").date() for d in calendar})
    start = dig(league, "season", "startDate") or (f"{season}-01-01" if slug in CALENDAR_YEAR else f"{season}-07-01")
    end = dig(league, "season", "endDate") or (f"{season}-12-31" if slug in CALENDAR_YEAR else f"{season + 1}-06-30")
    first, last = (datetime.strptime(x[:10], "%Y-%m-%d").date() for x in (start, end))
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


# ---------------------------------------------------------------- the cache

def cache_path(slug: str, season: int) -> Path:
    return CACHE_DIR / f"{slug}_{season}.csv"


def current_season(slug: str, today: date) -> int:
    if slug in CALENDAR_YEAR:
        return today.year
    return today.year if today.month >= 7 else today.year - 1


def merge(old: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(games, columns=COLUMNS)
    both = pd.concat([old, new], ignore_index=True) if not old.empty else new
    return both.drop_duplicates("event_id", keep="last").sort_values(["date", "kickoff", "event_id"]).reset_index(drop=True)


def read_cache(slug: str, season: int) -> pd.DataFrame:
    path = cache_path(slug, season)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"event_id": str, "home_id": str, "away_id": str})


def update_season(feed: Feed, slug: str, season: int, today: date, full: bool) -> tuple[int, list[dict]]:
    """Fetch a season (all of it, or the days around today) into its cache file. Returns (games, teams)."""
    name = EUROPE.get(slug) or f"{DOMESTIC.get(slug, slug)} ({slug})"
    if full:
        days = season_days(feed, slug, season)
    else:
        days = [today + timedelta(days=k) for k in range(-DAYS_BACK, DAYS_AHEAD + 1)]
    games, teams = feed.days(slug, days, slug)
    games = [g for g in games if g["season"] in (season, None)]
    table = merge(read_cache(slug, season), games)
    if full and slug in DOMESTIC and short(slug, season, len(table)):
        # the feed's calendar of game days is sometimes incomplete: ask for every day instead
        more, found = feed.days(slug, every_day(slug, season), slug)
        teams += found
        table = merge(table, [g for g in more if g["season"] in (season, None)])
    if table.empty:
        return 0, teams
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_path(slug, season), index=False)
    print(f"  {name} {season}: {len(table)} games ({len(games)} read)", flush=True)
    return len(table), teams


def short(slug: str, season: int, count: int) -> bool:
    """Fewer games than three-quarters of a typical season of this league (from the other seasons saved)."""
    others = [len(read_cache(slug, s)) for s in range(FIRST_SEASON, season + 3) if s != season and cache_path(slug, s).exists()]
    others = [n for n in others if n > 0]
    return len(others) >= 3 and count < 0.75 * float(np.median(others))


def all_competitions() -> list[str]:
    return list(EUROPE) + list(DOMESTIC)


# ---------------------------------------------------------------- the database

def load(conn, folder: Path = CACHE_DIR, teams: list[dict] | None = None) -> int:
    """Every cached game into soccer_games, and the European cups' into games/cup_game_info/odds."""
    frames = [pd.read_csv(p, dtype={"event_id": str, "home_id": str, "away_id": str}) for p in sorted(folder.glob("*.csv"))
              if not p.name.startswith("teams")]
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True).drop_duplicates("event_id", keep="last")
    df = df[df["home_id"].notna() & df["away_id"].notna() & (df["home_id"] != "None")]
    known = pd.read_csv(folder / "teams.csv", dtype={"team_id": str}) if (folder / "teams.csv").exists() else pd.DataFrame(columns=TEAM_COLUMNS)
    if teams:
        known = pd.concat([known, pd.DataFrame(teams, columns=TEAM_COLUMNS)]).drop_duplicates("team_id", keep="last")
        known.to_csv(folder / "teams.csv", index=False)
    conn.execute("DELETE FROM soccer_games")
    if "extra_time" not in df:
        df["extra_time"] = 0
    conn.executemany(f"INSERT INTO soccer_games ({', '.join(COLUMNS[:17])}) VALUES ({', '.join('?' * 17)})",
                     [tuple(None if pd.isna(v) else v for v in r) for r in df[COLUMNS[:17]].itertuples(index=False)])
    conn.executemany("INSERT OR REPLACE INTO soccer_teams (team_id, name, short_name, abbreviation, color, alternate_color) VALUES (?, ?, ?, ?, ?, ?)",
                     [tuple(None if pd.isna(v) else v for v in r) for r in known[TEAM_COLUMNS].itertuples(index=False)])
    from ucl.cups import CUPS
    for cup in CUPS.values():
        load_cup(conn, df[df["competition"] == cup["slug"]], known, cup)
    conn.commit()
    return len(df)


def load_cup(conn, df: pd.DataFrame, teams: pd.DataFrame, cup: dict) -> None:
    conn.execute("INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, 'soccer')", (cup["code"], cup["name"]))
    league_id = conn.execute("SELECT id FROM leagues WHERE code = ?", (cup["code"],)).fetchone()[0]
    names = dict(zip(teams["team_id"], teams["name"]))
    team_ids = {}

    def team(espn_id: str) -> int:
        if espn_id not in team_ids:
            name = names.get(espn_id) or f"Team {espn_id}"
            row = conn.execute("SELECT team_id FROM cup_teams WHERE league_id = ? AND espn_id = ?", (league_id, espn_id)).fetchone()
            if row:
                team_ids[espn_id] = row[0]
                if not conn.execute("SELECT 1 FROM teams WHERE league_id = ? AND name = ? AND id != ?", (league_id, name, row[0])).fetchone():
                    conn.execute("UPDATE teams SET name = ? WHERE id = ?", (name, row[0]))
            else:
                found = conn.execute("""SELECT t.id FROM teams t WHERE t.league_id = ? AND t.name = ?
                                        AND t.id NOT IN (SELECT team_id FROM cup_teams WHERE league_id = ?)""",
                                     (league_id, name, league_id)).fetchone()
                if found is None:
                    taken = conn.execute("SELECT 1 FROM teams WHERE league_id = ? AND name = ?", (league_id, name)).fetchone()
                    name = f"{name} ({espn_id})" if taken else name      # two clubs with one name: keep both
                    conn.execute("INSERT INTO teams (league_id, name) VALUES (?, ?)", (league_id, name))
                    found = conn.execute("SELECT id FROM teams WHERE league_id = ? AND name = ?", (league_id, name)).fetchone()
                team_ids[espn_id] = found[0]
                conn.execute("INSERT INTO cup_teams (league_id, espn_id, team_id) VALUES (?, ?, ?)", (league_id, espn_id, found[0]))
        return team_ids[espn_id]

    for g in df.itertuples():
        if g.status == "other" and pd.isna(g.home_score):
            status = "scheduled"
        else:
            status = "final" if g.status == "final" else "scheduled"
        home, away = team(g.home_id), team(g.away_id)
        existing = conn.execute("SELECT game_id FROM cup_game_info WHERE espn_id = ?", (g.event_id,)).fetchone()
        values = (int(g.season), g.date, home, away, None if pd.isna(g.home_score) else int(g.home_score),
                  None if pd.isna(g.away_score) else int(g.away_score), status, g.kickoff)
        if existing:
            game_id = existing[0]
            conn.execute("""UPDATE games SET season = ?, date = ?, home_team_id = ?, away_team_id = ?, home_score = ?,
                            away_score = ?, status = ?, kickoff = ? WHERE id = ?""", (*values, game_id))
        else:
            conn.execute("""INSERT OR IGNORE INTO games (league_id, season, date, home_team_id, away_team_id, home_score, away_score, status, kickoff)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (league_id, *values))
            game_id = conn.execute("SELECT id FROM games WHERE league_id = ? AND season = ? AND date = ? AND home_team_id = ? AND away_team_id = ?",
                                   (league_id, int(g.season), g.date, home, away)).fetchone()[0]
        conn.execute("""INSERT OR REPLACE INTO cup_game_info (game_id, espn_id, stage, leg, neutral, home_pens, away_pens, cancelled)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (game_id, g.event_id, g.stage, None if pd.isna(g.leg) else int(g.leg), int(g.neutral),
                      None if pd.isna(g.home_pens) else int(g.home_pens), None if pd.isna(g.away_pens) else int(g.away_pens),
                      int(g.status == "other" and pd.isna(g.home_score) and g.date < date.today().isoformat())))
        if g.status != "final" and not pd.isna(g.home_ml):
            for side in ("home", "draw", "away"):
                for closing, column in ((0, f"{side}_ml_open"), (1, f"{side}_ml")):
                    price = getattr(g, column)
                    if not pd.isna(price):
                        conn.execute("""DELETE FROM odds WHERE game_id = ? AND bookmaker = 'DraftKings' AND market = '1x2'
                                        AND outcome = ? AND is_closing = ?""", (game_id, side, closing))
                        insert_price(conn, game_id, "DraftKings", "1x2", side, float(price), None, closing)


# ---------------------------------------------------------------- running it

def season_over(slug: str, season: int) -> date:
    """The day after a season is surely over."""
    return date(season + 1, 1, 15) if slug in CALENDAR_YEAR else date(season + 1, 7, 15)


def main(argv: list[str] | None = None, today: date | None = None, get=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    today = today or date.today()
    history = "--history" in argv
    feed = Feed(get)
    teams: list[dict] = []
    for slug in all_competitions():
        now = current_season(slug, today)
        for season in range(FIRST_SEASON if history else now - 1, now + 1):
            path = cache_path(slug, season)
            if season < now:
                # a finished season is read in full once, after it's over, and never again
                if (path.exists() and date.fromtimestamp(path.stat().st_mtime) >= season_over(slug, season)
                        and not (history and slug in DOMESTIC and short(slug, season, len(read_cache(slug, season))))):
                    continue
                if not history and not path.exists():
                    continue
                full = today >= season_over(slug, season)
            else:
                full = history or not path.exists() or (time.time() - path.stat().st_mtime) > FULL_REFRESH_DAYS * 86400
            try:
                _, found = update_season(feed, slug, season, today, full)
            except RuntimeError as error:
                print(f"  {EUROPE.get(slug, slug)} {season}: {error}; carrying on.")
                continue
            teams += found
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    n = load(conn, teams=teams)
    conn.close()
    print(f"Saved {n:,} soccer games (domestic and European) with the cups' and big leagues' fixtures.")


if __name__ == "__main__":
    main()
