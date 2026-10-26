"""Import NBA games, results and player box scores into data/sports.db.

Usage:
    python -m nba.import_nba

Source: sportsdataverse-data's ESPN files (github.com/sportsdataverse/sportsdataverse-data), one
schedule and one player box score file per season since 2002-03. The current season's schedule
includes every game still to play (stored as 'scheduled'). Files are trimmed to the columns used
here and cached in data/raw/nba/; the last two seasons are downloaded again each run. Safe to re-run.

Seasons are stored by their starting year (2026 means 2026-27); ESPN names them by the ending year.
Teams are identified by ESPN's franchise id and always shown under their current name, so a move
or a rename (Seattle to Oklahoma City, Charlotte Bobcats to Hornets) keeps one history. All-Star
games and preseason games are left out.
"""

import io
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH, as_int, get_team_id

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
SCHEDULE_URL = BASE + "espn_nba_schedules/nba_schedule_{espn_season}.csv"
BOX_URL = BASE + "espn_nba_player_boxscores/player_box_{espn_season}.csv"
RAW = RAW_DIR / "nba"
FIRST_SEASON = 2002            # 2002-03
LEAGUE = ("NBA", "NBA", "basketball")
EASTERN = ZoneInfo("America/New_York")
SCHEDULE_COLUMNS = ["game_id", "season", "season_type", "date", "neutral_site", "status_type_completed", "status_type_name",
                    "notes_headline",
                    "home_id", "home_display_name", "home_abbreviation", "home_score", "home_color", "home_alternate_color",
                    "away_id", "away_display_name", "away_abbreviation", "away_score", "away_color", "away_alternate_color"]
BOX_COLUMNS = ["game_id", "athlete_id", "athlete_display_name", "athlete_position_abbreviation", "team_id", "starter",
               "did_not_play", "reason", "minutes", "points", "rebounds", "assists", "three_point_field_goals_made",
               "steals", "blocks", "turnovers", "plus_minus"]
OPTIONAL = {"reason"}           # not in the files before 2012-13
SEASON_TYPES = {2: "regular", 3: "playoffs", 5: "playin"}

# Conferences and divisions (the current alignment, unchanged since 2004-05), by ESPN abbreviation
DIVISIONS = {
    "Atlantic": ("East", ["BOS", "BKN", "NY", "PHI", "TOR"]),
    "Central": ("East", ["CHI", "CLE", "DET", "IND", "MIL"]),
    "Southeast": ("East", ["ATL", "CHA", "MIA", "ORL", "WSH"]),
    "Northwest": ("West", ["DEN", "MIN", "OKC", "POR", "UTAH"]),
    "Pacific": ("West", ["GS", "LAC", "LAL", "PHX", "SAC"]),
    "Southwest": ("West", ["DAL", "HOU", "MEM", "NO", "SA"]),
}
DIVISION_OF = {abbr: (conf, div) for div, (conf, teams) in DIVISIONS.items() for abbr in teams}


def current_season(today=None) -> int:
    """The season (by starting year) under way or next: from August on, the new one."""
    today = today or datetime.now(EASTERN).date()
    return today.year if today.month >= 8 else today.year - 1


def download(url: str, path, columns: list[str], refresh: bool) -> pd.DataFrame | None:
    """A season file, trimmed to `columns` and cached; None if it doesn't exist (yet)."""
    if refresh or not path.exists():
        response = requests.get(url, timeout=600)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content), low_memory=False)
        missing = [c for c in columns if c not in df.columns and c not in OPTIONAL]
        if missing:
            raise SystemExit(f"An NBA file has changed: {url} is missing columns {missing}")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.reindex(columns=columns).to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def fetch(season: int, refresh: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    espn = season + 1
    schedule = download(SCHEDULE_URL.format(espn_season=espn), RAW / f"schedule_{espn}.csv", SCHEDULE_COLUMNS, refresh)
    box = download(BOX_URL.format(espn_season=espn), RAW / f"player_box_{espn}.csv", BOX_COLUMNS, refresh)
    return schedule, box


def truthy(value) -> bool:
    return str(value).strip().upper() in ("TRUE", "1", "T")


def hex_color(value):
    text = str(value).strip().lstrip("#") if value is not None and not pd.isna(value) else ""
    return f"#{text.upper()}" if len(text) == 6 else None


def franchises(schedule: pd.DataFrame) -> dict:
    """ESPN team id -> (current name, abbreviation, color, alternate color), from each team's latest game."""
    sides = pd.concat([schedule[[f"{s}_id", f"{s}_display_name", f"{s}_abbreviation", f"{s}_color", f"{s}_alternate_color", "date"]]
                       .set_axis(["id", "name", "abbr", "color", "alt", "date"], axis=1) for s in ("home", "away")])
    sides = sides[pd.to_numeric(sides["id"], errors="coerce").between(1, 30)]   # the 30 franchises; All-Star teams have other ids
    sides = sides.assign(id=sides["id"].astype(int)).sort_values("date", kind="stable")
    latest = sides.groupby("id").last()
    return {i: (r["name"], r["abbr"], hex_color(r["color"]), hex_color(r["alt"])) for i, r in latest.iterrows()}


def season_type(row) -> str | None:
    kind = SEASON_TYPES.get(as_int(row["season_type"]))
    notes = row["notes_headline"] if isinstance(row["notes_headline"], str) else ""
    tournament = "NBA Cup" in notes or "In-Season Tournament" in notes
    if kind == "regular" and tournament and "Championship" in notes:
        return "cup_final"     # played in December, but doesn't count in the standings
    return kind


def get_league_id(conn) -> int:
    conn.execute("INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, ?)", LEAGUE)
    return conn.execute("SELECT id FROM leagues WHERE code = ?", (LEAGUE[0],)).fetchone()[0]


def import_games(conn, schedule: pd.DataFrame, box: pd.DataFrame | None = None) -> dict:
    conn.executescript(SCHEMA_PATH.read_text())
    league_id = get_league_id(conn)
    teams = franchises(schedule)
    for espn_id, (name, abbr, color, alt) in teams.items():
        conf, div = DIVISION_OF.get(abbr, (None, None))
        conn.execute("INSERT OR REPLACE INTO nba_teams (espn_id, name, abbreviation, conference, division, color, alt_color) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)", (espn_id, name, abbr, conf, div, color, alt))
    cache: dict = {}
    counts = {"new": 0, "updated": 0, "final": 0, "scheduled": 0, "players": 0}
    source_to_game = {}
    postponed = []
    for row in schedule.to_dict("records"):
        home_id, away_id = as_int(row["home_id"]), as_int(row["away_id"])
        kind = season_type(row)
        if home_id not in teams or away_id not in teams or kind is None:
            continue
        if row.get("status_type_name") in ("STATUS_POSTPONED", "STATUS_CANCELED"):
            postponed.append(str(as_int(row["game_id"])))   # the game is listed again on its new date
            continue
        start = pd.Timestamp(row["date"])
        start = (start.tz_localize("UTC") if start.tzinfo is None else start).tz_convert(EASTERN)
        played = truthy(row["status_type_completed"]) and pd.notna(row["home_score"]) and pd.notna(row["away_score"]) \
            and not (row["home_score"] == 0 and row["away_score"] == 0)
        values = (start.strftime("%Y-%m-%d"), start.strftime("%H:%M"),
                  as_int(row["home_score"]) if played else None, as_int(row["away_score"]) if played else None,
                  "final" if played else "scheduled")
        home = get_team_id(conn, league_id, teams[home_id][0], cache)
        away = get_team_id(conn, league_id, teams[away_id][0], cache)
        source_id = str(as_int(row["game_id"]))
        existing = conn.execute("SELECT game_id FROM nba_game_info WHERE source_game_id = ?", (source_id,)).fetchone()
        if existing:
            game_id = existing[0]
            conn.execute("""UPDATE OR IGNORE games SET date = ?, kickoff = ?, home_score = ?, away_score = ?, status = ?,
                                                     home_team_id = ?, away_team_id = ? WHERE id = ?""", (*values, home, away, game_id))
            counts["updated"] += 1
        else:
            cursor = conn.execute("""INSERT OR IGNORE INTO games (league_id, season, date, kickoff, home_score, away_score, status,
                                                                  home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                  (league_id, as_int(row["season"]) - 1, *values, home, away))
            if cursor.rowcount == 0:
                continue   # the same teams on the same day twice: a duplicate in the source
            game_id = cursor.lastrowid
            counts["new"] += 1
        conn.execute("""INSERT OR REPLACE INTO nba_game_info (game_id, source_game_id, season_type, neutral_site, notes)
                        VALUES (?, ?, ?, ?, ?)""",
                     (game_id, source_id, kind, int(truthy(row["neutral_site"])),
                      row["notes_headline"] if isinstance(row["notes_headline"], str) else None))
        source_to_game[source_id] = (game_id, played)
        counts["final" if played else "scheduled"] += 1
    for source_id in postponed:   # a game imported before it was postponed
        found = conn.execute("SELECT game_id FROM nba_game_info WHERE source_game_id = ?", (source_id,)).fetchone()
        if found:
            conn.execute("DELETE FROM nba_game_info WHERE game_id = ?", found)
            conn.execute("DELETE FROM predictions WHERE game_id = ?", found)
            conn.execute("DELETE FROM prediction_goals WHERE game_id = ?", found)
            conn.execute("DELETE FROM games WHERE id = ?", found)
    if box is not None and not box.empty:
        counts["players"] = import_box(conn, box, source_to_game, teams)
    conn.commit()
    return counts


def import_box(conn, box: pd.DataFrame, source_to_game: dict, teams: dict) -> int:
    """Each player's line in each finished game (including players who didn't play, with the reason)."""
    box = box[box["game_id"].astype(str).isin(source_to_game.keys()) & pd.to_numeric(box["athlete_id"], errors="coerce").notna()]
    rows = []
    def num(v):
        value = pd.to_numeric(v, errors="coerce")     # '--' and blanks: unknown
        return float(value) if pd.notna(value) else None
    for r in box.to_dict("records"):
        game_id, played = source_to_game[str(as_int(r["game_id"]))]
        team = teams.get(as_int(r["team_id"]))
        if not played or team is None:
            continue
        minutes = num(r["minutes"]) or 0.0
        dnp = truthy(r["did_not_play"]) or minutes == 0
        rows.append((game_id, team[0], str(as_int(r["athlete_id"])), r["athlete_display_name"],
                     r["athlete_position_abbreviation"] if isinstance(r["athlete_position_abbreviation"], str) else None,
                     int(truthy(r["starter"])), minutes, int(dnp), r["reason"] if isinstance(r["reason"], str) else None,
                     num(r["points"]), num(r["rebounds"]), num(r["assists"]), num(r["three_point_field_goals_made"]),
                     num(r["steals"]), num(r["blocks"]), num(r["turnovers"]), num(r["plus_minus"])))
    conn.executemany("""INSERT OR REPLACE INTO nba_player_games (game_id, team, player_id, name, position, starter, minutes,
                                                                  did_not_play, reason, points, rebounds, assists, threes,
                                                                  steals, blocks, turnovers, plus_minus)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
    return len(rows)


def main() -> None:
    latest = current_season()
    schedules, boxes = [], []
    for season in range(FIRST_SEASON, latest + 1):
        schedule, box = fetch(season, refresh=season >= latest - 1)
        if schedule is not None:
            schedules.append(schedule)
        if box is not None:
            boxes.append(box)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    counts = import_games(conn, pd.concat(schedules, ignore_index=True),
                         pd.concat(boxes, ignore_index=True) if boxes else None)
    conn.close()
    print(f"NBA {FIRST_SEASON}-{latest + 1}: {counts['final']} games played, {counts['scheduled']} still to play "
          f"({counts['new']} new, {counts['updated']} updated); {counts['players']} player box score lines.")


if __name__ == "__main__":
    main()
