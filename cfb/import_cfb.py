"""Import college football games, results and betting lines into data/sports.db.

Usage:
    python -m cfb.import_cfb

Source: cfbfastR-data (github.com/sportsdataverse/cfbfastR-data), which republishes
CollegeFootballData's schedules: every game since 2001 with scores, divisions, conferences and
CollegeFootballData's own pre-game Elo ratings (kept as a benchmark), plus betting lines from
several sportsbooks (2006-2025 so far). Updated weekly during the season.

Only games between two top-division schools (FBS) or second-division ones (FCS) are kept;
FCS games are there to rate the FCS teams that FBS teams play. Betting lines are turned into one
consensus per game: each sportsbook's moneyline with its margin removed, averaged, plus the
median spread and total. Safe to re-run: existing games are updated.

Starting quarterbacks (2014 on) come from the play-by-play player stats: the player with the most
dropbacks for each team. Those files are large (30-60 MB a season), so each season is boiled down to
a small file of starters (data/raw/cfb/starters_<season>.csv) and the big file isn't kept.
"""

import gzip
import io
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH, as_int, get_team_id, insert_price

BASE_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main"
SCHEDULE_URL = BASE_URL + "/schedules/csv/cfb_schedules_{season}.csv"
LINES_URL = BASE_URL + "/betting/csv/cfb_line_odds.csv.gz"
PLAYERS_URL = BASE_URL + "/player_stats/csv/player_stats_{season}.csv"
PLAYERS_FROM = 2014   # the first season with player stats
RAW = RAW_DIR / "cfb"
FIRST_SEASON = 2001
LEAGUE = ("CFB", "College football (FBS)", "american_football")
BOOKMAKER = "consensus"
DIVISIONS = {"fbs", "fcs"}
EASTERN = ZoneInfo("America/New_York")
REQUIRED = ["game_id", "season", "week", "season_type", "start_date", "completed", "neutral_site",
            "home_id", "home_team", "home_division", "home_points", "away_id", "away_team", "away_division",
            "away_points"]


def fetch(url: str, path, refresh: bool) -> bytes:
    """Download a file, or reuse the cached copy (past seasons don't change)."""
    if path.exists() and not refresh:
        return path.read_bytes()
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def current_season(today=None) -> int:
    today = today or datetime.now(EASTERN).date()
    return today.year if today.month >= 7 else today.year - 1


def load_schedules(seasons) -> pd.DataFrame:
    latest = max(seasons)
    frames = []
    for season in seasons:
        content = fetch(SCHEDULE_URL.format(season=season), RAW / f"cfb_schedules_{season}.csv", season >= latest - 1)
        frames.append(read_schedule(content.decode("utf-8")))
    return pd.concat(frames, ignore_index=True)


def read_schedule(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"The cfbfastR-data schedule file has changed: missing columns {missing}")
    return df


def kickoff_eastern(start: str) -> tuple[str, str | None]:
    """'2026-10-17T23:30:00.000Z' -> ('2026-10-17', '19:30'), US Eastern time."""
    t = pd.Timestamp(start)
    t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(EASTERN)
    return t.strftime("%Y-%m-%d"), t.strftime("%H:%M")


def truthy(value) -> bool:
    return str(value).strip().upper() in ("TRUE", "1", "T")


# ---------------------------------------------------------------- betting lines

def american_to_probability(odds: float) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def consensus_lines(lines: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game: the consensus home win probability (each book's moneyline with its margin
    removed, averaged), and the median home spread and total.

    Lines name teams by an abbreviation ('UTH') in early seasons and by school name later; each
    name is matched to the team it appears with in every one of its games."""
    lines = lines.dropna(subset=["game_id", "abbr"]).copy()
    lines["game_id"] = lines["game_id"].astype("int64")
    teams = games.set_index("game_id")[["home_id", "away_id"]]
    lines = lines[lines["game_id"].isin(teams.index)]
    lines = lines.join(teams, on="game_id")
    candidates = pd.concat([lines[["abbr", "home_id"]].rename(columns={"home_id": "team"}),
                            lines[["abbr", "away_id"]].rename(columns={"away_id": "team"})])
    best = candidates.groupby(["abbr", "team"]).size().reset_index(name="n").sort_values("n", ascending=False)
    team_of = best.drop_duplicates("abbr").set_index("abbr")["team"]
    lines["team"] = lines["abbr"].map(team_of)
    lines["side"] = np.where(lines["team"] == lines["home_id"], "home",
                             np.where(lines["team"] == lines["away_id"], "away", None))

    ml = lines[(lines["market_type"] == "money_line") & lines["odds"].notna() & lines["side"].notna()]
    ml = ml.assign(p=[american_to_probability(o) for o in ml["odds"]])
    wide = ml.pivot_table(index=["game_id", "book"], columns="side", values="p", aggfunc="first").dropna()
    wide = wide[(wide["home"] + wide["away"]).between(0.95, 1.2)]   # skip broken pairs
    wide["home_win"] = wide["home"] / (wide["home"] + wide["away"])
    market = wide.groupby("game_id")["home_win"].mean().rename("market_home")

    spread = lines[(lines["market_type"] == "spread") & (lines["side"] == "home")].groupby("game_id")["lines"].median()
    total = lines[(lines["market_type"] == "total")].groupby("game_id")["lines"].median()
    return pd.concat([market, spread.rename("spread_line"), total.rename("total_line")], axis=1)


def load_lines(games: pd.DataFrame) -> pd.DataFrame:
    content = fetch(LINES_URL, RAW / "cfb_line_odds.csv.gz", True)
    lines = pd.read_csv(io.BytesIO(gzip.decompress(content)) if content[:2] == b"\x1f\x8b" else io.BytesIO(content),
                        low_memory=False)
    return consensus_lines(lines, games)


# ---------------------------------------------------------------- starting quarterbacks

PASSER_COLUMNS = [("completion_player_id", "completion_player"), ("incompletion_player_id", "incompletion_player"),
                  ("interception_thrown_player_id", "interception_thrown_player"),
                  ("sack_taken_player_id", "sack_taken_player")]


def starters_from_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Each team's main quarterback in each game: the player with the most dropbacks (completions,
    incompletions, interceptions and sacks), and how many he had."""
    ids = names = None
    for id_col, name_col in PASSER_COLUMNS:
        ids = plays[id_col] if ids is None else ids.fillna(plays[id_col])
        names = plays[name_col] if names is None else names.fillna(plays[name_col])
    passes = plays.assign(qb_id=ids, qb_name=names, order=pd.to_numeric(plays["play_id"], errors="coerce"))
    passes = passes.dropna(subset=["qb_id", "qb_name", "team"])
    passes["qb_id"] = passes["qb_id"].astype("int64").astype(str)
    attempts = (passes.groupby(["game_id", "team", "qb_id"]).agg(qb_name=("qb_name", "first"), attempts=("qb_id", "size"))
                .reset_index())
    # the player with the most dropbacks: the first passer is sometimes a trick play or a wildcat snap
    return (attempts.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
            .sort_values(["game_id", "team"]).reset_index(drop=True))


def load_starters(seasons, latest: int) -> pd.DataFrame:
    frames = []
    for season in seasons:
        if season < PLAYERS_FROM:
            continue
        path = RAW / f"starters_{season}.csv"
        if not path.exists() or season >= latest:
            content = requests.get(PLAYERS_URL.format(season=season), timeout=600)
            if content.status_code == 404:
                continue
            content.raise_for_status()
            columns = ["game_id", "team", "play_id"] + [c for pair in PASSER_COLUMNS for c in pair]
            plays = pd.read_csv(io.BytesIO(content.content), usecols=columns, low_memory=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            starters_from_plays(plays).to_csv(path, index=False)
        frames.append(pd.read_csv(path, dtype={"qb_id": str}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_id", "team", "qb_id", "qb_name", "attempts"])


def import_starters(conn, starters: pd.DataFrame) -> int:
    """Match each starter to his game and side (home/away) and store him."""
    games = pd.read_sql("""SELECT i.game_id, i.source_game_id, h.name AS home, a.name AS away FROM cfb_game_info i
                           JOIN games g ON g.id = i.game_id JOIN teams h ON h.id = g.home_team_id
                           JOIN teams a ON a.id = g.away_team_id""", conn)
    games["source_game_id"] = games["source_game_id"].astype(str)
    starters = starters.assign(source_game_id=starters["game_id"].astype("int64").astype(str)).drop(columns="game_id")
    merged = starters.merge(games, on="source_game_id")
    merged["side"] = np.where(merged["team"] == merged["home"], "home", np.where(merged["team"] == merged["away"], "away", None))
    merged = merged.dropna(subset=["side"])
    conn.executemany("INSERT OR REPLACE INTO cfb_starters (game_id, side, qb_id, qb_name, attempts) VALUES (?, ?, ?, ?, ?)",
                     merged[["game_id", "side", "qb_id", "qb_name", "attempts"]].itertuples(index=False, name=None))
    conn.commit()
    return len(merged)


# ---------------------------------------------------------------- the database

def get_league_id(conn) -> int:
    conn.execute("INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, ?)", LEAGUE)
    return conn.execute("SELECT id FROM leagues WHERE code = ?", (LEAGUE[0],)).fetchone()[0]


def text(value):
    return None if value is None or pd.isna(value) else str(value)


def import_games(conn, schedule: pd.DataFrame, lines: pd.DataFrame | None = None) -> dict:
    conn.executescript(SCHEMA_PATH.read_text())
    league_id = get_league_id(conn)
    cache: dict = {}
    keep = schedule[schedule["home_division"].isin(DIVISIONS) & schedule["away_division"].isin(DIVISIONS)
                    & schedule["home_team"].notna() & schedule["away_team"].notna()]
    lines = lines if lines is not None else pd.DataFrame(columns=["market_home", "spread_line", "total_line"])
    counts = {"new": 0, "updated": 0, "final": 0, "scheduled": 0, "odds": 0}
    for row in keep.to_dict("records"):
        source_id = str(int(row["game_id"]))
        day, kickoff = kickoff_eastern(row["start_date"])
        if truthy(row.get("start_time_tbd")):
            kickoff = None
        played = truthy(row["completed"]) and pd.notna(row["home_points"]) and pd.notna(row["away_points"])
        values = (day, kickoff, as_int(row["home_points"]) if played else None,
                  as_int(row["away_points"]) if played else None, "final" if played else "scheduled")
        home = get_team_id(conn, league_id, row["home_team"], cache)
        away = get_team_id(conn, league_id, row["away_team"], cache)
        existing = conn.execute("SELECT game_id FROM cfb_game_info WHERE source_game_id = ?", (source_id,)).fetchone()
        if existing:
            game_id = existing[0]
            # OR IGNORE: if the source moves a game onto a date where the same teams already have a
            # (duplicate) game, keep the old date rather than stop the import
            conn.execute("UPDATE OR IGNORE games SET date = ?, kickoff = ?, home_score = ?, away_score = ?, status = ?, "
                         "home_team_id = ?, away_team_id = ? WHERE id = ?", (*values, home, away, game_id))
            counts["updated"] += 1
        else:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO games (league_id, season, date, kickoff, home_score, away_score, status,
                                                home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (league_id, int(row["season"]), *values, home, away))
            if cursor.rowcount == 0:
                continue   # the same teams on the same day twice: a duplicate row in the source
            game_id = cursor.lastrowid   # (after an ignored insert, lastrowid would be the previous game's)
            counts["new"] += 1
        line = lines.loc[int(row["game_id"])] if int(row["game_id"]) in lines.index else None
        conn.execute("""INSERT OR REPLACE INTO cfb_game_info
                        (game_id, source_game_id, week, season_type, neutral_site, conference_game,
                         home_division, away_division, home_conference, away_conference, home_elo, away_elo,
                         venue, notes, spread_line, total_line)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (game_id, source_id, as_int(row["week"]), row["season_type"], int(truthy(row["neutral_site"])),
                      int(truthy(row.get("conference_game"))), row["home_division"], row["away_division"],
                      text(row.get("home_conference")), text(row.get("away_conference")),
                      None if pd.isna(row.get("home_pregame_elo")) else float(row["home_pregame_elo"]),
                      None if pd.isna(row.get("away_pregame_elo")) else float(row["away_pregame_elo"]),
                      text(row.get("venue")), text(row.get("notes")),
                      None if line is None or pd.isna(line["spread_line"]) else float(line["spread_line"]),
                      None if line is None or pd.isna(line["total_line"]) else float(line["total_line"])))
        if played:
            counts["final"] += 1
            if line is not None and pd.notna(line["market_home"]):
                p = float(line["market_home"])
                counts["odds"] += insert_price(conn, game_id, BOOKMAKER, "moneyline", "home", 1 / p, None, 1)
                counts["odds"] += insert_price(conn, game_id, BOOKMAKER, "moneyline", "away", 1 / (1 - p), None, 1)
        else:
            counts["scheduled"] += 1
    conn.commit()
    return counts


def main() -> None:
    seasons = range(FIRST_SEASON, current_season() + 1)
    schedule = load_schedules(seasons)
    lines = load_lines(schedule)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    counts = import_games(conn, schedule, lines)
    starters = import_starters(conn, load_starters(seasons, max(seasons)))
    teams = conn.execute("SELECT COUNT(*) FROM teams t JOIN leagues l ON l.id = t.league_id WHERE l.code = 'CFB'").fetchone()[0]
    conn.close()
    print(f"College football {min(seasons)}-{max(seasons)}: {counts['final']} games played, "
          f"{counts['scheduled']} still to play; {counts['new']} new, {counts['updated']} updated, "
          f"{counts['odds']} new odds rows, {teams} teams; {starters} starting quarterbacks.")


if __name__ == "__main__":
    main()
