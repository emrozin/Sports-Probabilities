"""Import men's college basketball games and results into data/sports.db.

Usage:
    python -m cbb.import_cbb

Source: sportsdataverse-data's ESPN schedule files (github.com/sportsdataverse/sportsdataverse-data),
one per season since 2002-03, with scores, neutral sites, conference games and each team's
conference, plus team colors. The current season's file includes the full schedule; games not yet
played are stored as 'scheduled'. Past seasons are cached in data/raw/cbb/. Safe to re-run.

Seasons are stored by their starting year (2025 means 2025-26); ESPN names them by the ending year.
Schools outside Division I appear only as opponents (a game or two each) and have no conference.
"""

import io
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH, as_int, get_team_id

URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "espn_mens_college_basketball_schedules/mbb_schedule_{espn_season}.csv")
RAW = RAW_DIR / "cbb"
FIRST_SEASON = 2002            # 2002-03
LEAGUE = ("CBB", "Men's college basketball", "basketball")
EASTERN = ZoneInfo("America/New_York")
COLUMNS = ["game_id", "season", "season_type", "date", "neutral_site", "conference_competition", "status_type_completed",
           "home_id", "home_location", "home_score", "home_conference_id", "home_color", "home_alternate_color",
           "away_id", "away_location", "away_score", "away_conference_id", "away_color", "away_alternate_color",
           "notes_headline"]

# ESPN's conference ids (checked against member schools in 2009-10, 2022-23 and 2025-26)
CONFERENCES = {
    1: "America East", 2: "ACC", 3: "Atlantic 10", 4: "Big East", 5: "Big Sky", 6: "Big South", 7: "Big Ten",
    8: "Big 12", 9: "Big West", 10: "CAA", 11: "Conference USA", 12: "Ivy League", 13: "MAAC", 14: "MAC",
    16: "MEAC", 18: "Missouri Valley", 19: "NEC", 20: "Ohio Valley", 21: "Pac-12", 22: "Patriot League",
    23: "SEC", 24: "SoCon", 25: "Southland", 26: "SWAC", 27: "Sun Belt", 29: "WCC", 30: "WAC",
    43: "Independents", 44: "Mountain West", 45: "Horizon League", 46: "ASUN", 49: "Summit League",
    62: "American",
}


def current_season(today=None) -> int:
    """The season (by starting year) that's under way or next: from August on, the new one."""
    today = today or datetime.now(EASTERN).date()
    return today.year if today.month >= 8 else today.year - 1


def fetch(season: int, refresh: bool) -> pd.DataFrame | None:
    path = RAW / f"mbb_schedule_{season + 1}.csv"
    if refresh or not path.exists():
        response = requests.get(URL.format(espn_season=season + 1), timeout=300)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return read_schedule(path.read_bytes().decode("utf-8"))


def read_schedule(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"The basketball schedule file has changed: missing columns {missing}")
    return df[COLUMNS]


def conference(value):
    if value is None or pd.isna(value):
        return None
    return CONFERENCES.get(int(value), f"Conference {int(value)}")


def hex_color(value):
    text = str(value).strip().lstrip("#") if value is not None and not pd.isna(value) else ""
    return f"#{text.upper()}" if len(text) == 6 else None


def truthy(value) -> bool:
    return str(value).strip().upper() in ("TRUE", "1", "T")


def current_names(df: pd.DataFrame) -> dict:
    """ESPN team id -> the name in that team's latest game. ESPN renames schools now and then
    ('Maryland-Eastern Shore' became 'Maryland Eastern Shore' in 2024); the id stays the same, so
    each school keeps one name and one rating history."""
    sides = pd.concat([df[[f"{side}_id", f"{side}_location", "date"]].set_axis(["id", "name", "date"], axis=1)
                       for side in ("home", "away")])
    sides = sides.dropna(subset=["id", "name"])
    sides = sides[pd.to_numeric(sides["id"], errors="coerce") > 0]
    sides = sides.assign(id=pd.to_numeric(sides["id"]).astype(int)).sort_values("date", kind="stable")
    return sides.groupby("id")["name"].last().to_dict()


def team_name(row: dict, side: str, names: dict) -> str:
    team_id = pd.to_numeric(row[f"{side}_id"], errors="coerce")
    return names.get(int(team_id), row[f"{side}_location"]) if pd.notna(team_id) else row[f"{side}_location"]


def get_league_id(conn) -> int:
    conn.execute("INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, ?)", LEAGUE)
    return conn.execute("SELECT id FROM leagues WHERE code = ?", (LEAGUE[0],)).fetchone()[0]


def import_games(conn, df: pd.DataFrame) -> dict:
    conn.executescript(SCHEMA_PATH.read_text())
    league_id = get_league_id(conn)
    cache: dict = {}
    counts = {"new": 0, "updated": 0, "final": 0, "scheduled": 0}
    colors = {}
    names = current_names(df)
    for row in df.dropna(subset=["home_location", "away_location"]).to_dict("records"):
        start = pd.Timestamp(row["date"])
        start = (start.tz_localize("UTC") if start.tzinfo is None else start).tz_convert(EASTERN)
        day = start.strftime("%Y-%m-%d")
        kickoff = start.strftime("%H:%M") if start.strftime("%H:%M") != "00:00" else None   # midnight: time unknown
        played = truthy(row["status_type_completed"]) and pd.notna(row["home_score"]) and pd.notna(row["away_score"]) \
            and not (row["home_score"] == 0 and row["away_score"] == 0)
        values = (day, kickoff, as_int(row["home_score"]) if played else None, as_int(row["away_score"]) if played else None,
                  "final" if played else "scheduled")
        home_name, away_name = team_name(row, "home", names), team_name(row, "away", names)
        home = get_team_id(conn, league_id, home_name, cache)
        away = get_team_id(conn, league_id, away_name, cache)
        for side, name in (("home", home_name), ("away", away_name)):
            colors[name] = (hex_color(row[f"{side}_color"]), hex_color(row[f"{side}_alternate_color"]))
        source_id = str(int(row["game_id"]))
        existing = conn.execute("SELECT game_id FROM cbb_game_info WHERE source_game_id = ?", (source_id,)).fetchone()
        if existing:
            game_id = existing[0]
            conn.execute("""UPDATE OR IGNORE games SET date = ?, kickoff = ?, home_score = ?, away_score = ?, status = ?,
                                                     home_team_id = ?, away_team_id = ? WHERE id = ?""",
                         (*values, home, away, game_id))
            counts["updated"] += 1
        else:
            cursor = conn.execute("""INSERT OR IGNORE INTO games (league_id, season, date, kickoff, home_score, away_score, status,
                                                                  home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                  (league_id, int(row["season"]) - 1, *values, home, away))
            if cursor.rowcount == 0:
                continue   # the same teams on the same day twice: a duplicate in the source
            game_id = cursor.lastrowid
            counts["new"] += 1
        conn.execute("""INSERT OR REPLACE INTO cbb_game_info (game_id, source_game_id, season_type, neutral_site, conference_game,
                                                             home_conference, away_conference, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (game_id, source_id, "postseason" if int(row["season_type"]) == 3 else "regular",
                      int(truthy(row["neutral_site"])), int(truthy(row["conference_competition"])),
                      conference(row["home_conference_id"]), conference(row["away_conference_id"]),
                      row["notes_headline"] if isinstance(row["notes_headline"], str) else None))
        counts["final" if played else "scheduled"] += 1
    conn.executemany("INSERT OR REPLACE INTO cbb_teams (name, color, alt_color) VALUES (?, ?, ?)",
                     [(name, c, a) for name, (c, a) in colors.items()])
    conn.commit()
    return counts


def main() -> None:
    latest = current_season()
    frames = []
    for season in range(FIRST_SEASON, latest + 1):
        df = fetch(season, refresh=season >= latest - 1)
        if df is not None:
            frames.append(df)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    counts = import_games(conn, pd.concat(frames, ignore_index=True))
    conn.close()
    print(f"Men's college basketball {FIRST_SEASON}-{latest + 1}: {counts['final']} games played, "
          f"{counts['scheduled']} still to play; {counts['new']} new, {counts['updated']} updated.")


if __name__ == "__main__":
    main()
