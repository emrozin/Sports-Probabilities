"""Import NFL weekly team and player stats from nflverse into data/sports.db.

Usage:
    python -m nfl.import_nfl_stats            # every season since 1999 (first run: ~60 downloads)

Source: nflverse's stats releases, one file per season for teams and one for players (passing,
rushing and receiving numbers, with expected points added), plus the weekly injury reports (from 2009)
and each player's share of his team's offensive snaps (from 2013). Past seasons are downloaded once and
cached in data/raw/nfl_stats/; the current season is downloaded again every run. Rows are linked
to games by nflverse's game id, so run python -m nfl.import_nfl first. Safe to re-run.
"""

import io
import sqlite3
from datetime import date

import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH
from nfl.import_nfl import FRANCHISES

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {"team": "stats_team/stats_team_week_{season}.csv", "player": "stats_player/stats_player_week_{season}.csv",
         "injuries": "injuries/injuries_{season}.csv", "snaps": "snap_counts/snap_counts_{season}.csv",
         "rosters": "weekly_rosters/roster_weekly_{season}.csv"}
FIRST = {"team": 1999, "player": 1999, "injuries": 2009, "snaps": 2013, "rosters": 2009}
CACHE_DIR = RAW_DIR / "nfl_stats"
FIRST_SEASON = 1999

TEAM_COLUMNS = {"attempts": "pass_attempts", "sacks_suffered": "sacks", "carries": "carries",
                "passing_yards": "passing_yards", "rushing_yards": "rushing_yards",
                "passing_epa": "passing_epa", "rushing_epa": "rushing_epa"}
PLAYER_COLUMNS = {"completions": "completions", "attempts": "pass_attempts", "passing_yards": "passing_yards",
                  "passing_tds": "passing_tds", "passing_interceptions": "interceptions",
                  "carries": "carries", "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds",
                  "targets": "targets", "receptions": "receptions", "receiving_yards": "receiving_yards",
                  "receiving_tds": "receiving_tds", "receiving_air_yards": "receiving_air_yards"}
REQUIRED = {"team": ["game_id", "team", *TEAM_COLUMNS], "player": ["game_id", "team", "player_id",
                                                                    "player_display_name", "position", *PLAYER_COLUMNS],
            "injuries": ["season", "week", "team", "gsis_id", "full_name", "position", "report_status"],
            "snaps": ["game_id", "team", "player", "position", "offense_snaps", "offense_pct"],
            "rosters": ["season", "week", "team", "gsis_id", "full_name", "position", "status"]}
TRIMMED = {"rosters"}   # cached with only the columns used (the full files are large)


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def fetch(kind: str, season: int, refresh: bool) -> pd.DataFrame | None:
    """One season's file, from the cache unless it's missing or `refresh`. None if nflverse has no file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{kind}_{season}.csv"
    if refresh or not path.exists():
        response = requests.get(f"{BASE_URL}/{FILES[kind].format(season=season)}", timeout=120)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if kind in TRIMMED:
            full = pd.read_csv(io.BytesIO(response.content), low_memory=False)
            full.reindex(columns=REQUIRED[kind]).to_csv(path, index=False)
        else:
            path.write_bytes(response.content)
    df = pd.read_csv(io.BytesIO(path.read_bytes()), low_memory=False)
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise SystemExit(f"The nflverse {kind} stats file for {season} has changed: missing columns {missing}")
    return df


def store_injuries(conn, df: pd.DataFrame) -> int:
    """Write one season's injury reports, replacing that season's stored reports. Returns rows written."""
    df = df.dropna(subset=["gsis_id", "team", "week"]).copy()
    df["team"] = df["team"].replace(FRANCHISES)
    rows = [(int(r.season), int(r.week), r.team, r.gsis_id, r.full_name, r.position,
             r.report_status if isinstance(r.report_status, str) else None) for r in df.itertuples()]
    conn.executemany("DELETE FROM nfl_injuries WHERE season = ?", [(s,) for s in set(df["season"].astype(int))])
    conn.executemany("INSERT OR REPLACE INTO nfl_injuries VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def store_rosters(conn, df: pd.DataFrame) -> int:
    """Write one season's weekly rosters (each player's status every week: active, injured reserve,
    practice squad...), replacing that season's stored rosters. Returns rows written."""
    df = df.dropna(subset=["gsis_id", "team", "week"]).copy()
    df["team"] = df["team"].replace(FRANCHISES)
    rows = [(int(r.season), int(r.week), r.team, r.gsis_id, r.full_name, r.position,
             r.status if isinstance(r.status, str) else None) for r in df.itertuples()]
    conn.executemany("DELETE FROM nfl_rosters WHERE season = ?", [(s,) for s in set(df["season"].astype(int))])
    conn.executemany("INSERT OR REPLACE INTO nfl_rosters VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def name_key(name) -> str:
    """'D.K. Metcalf' and 'DK Metcalf' -> 'dkmetcalf': for matching names between nflverse files."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def store_snaps(conn, df: pd.DataFrame, game_ids: dict) -> int:
    """Write one season's snap counts, matched to player ids by game, team and name."""
    df = df[df["offense_snaps"].fillna(0) > 0].dropna(subset=["game_id", "team", "player"]).copy()
    df["team"] = df["team"].replace(FRANCHISES)
    df["game"] = df["game_id"].map(game_ids)
    df = df.dropna(subset=["game"])
    games = sorted(set(df["game"].astype(int)))
    known = {}
    for chunk in range(0, len(games), 500):
        part = games[chunk:chunk + 500]
        for game, team, pid, name in conn.execute(
                f"SELECT game_id, team, player_id, name FROM nfl_player_stats WHERE game_id IN ({','.join('?' * len(part))})", part):
            known[(game, team, name_key(name))] = pid
    rows = [(int(r.game), r.team, r.player, r.position, known.get((int(r.game), r.team, name_key(r.player))),
             int(r.offense_snaps), float(r.offense_pct)) for r in df.itertuples()]
    conn.executemany("DELETE FROM nfl_snap_counts WHERE game_id = ?", [(g,) for g in games])
    conn.executemany("INSERT OR REPLACE INTO nfl_snap_counts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def ensure_columns(conn) -> None:
    """Databases created before receiving_air_yards was added don't have it yet; add it."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(nfl_player_stats)")}
    if "receiving_air_yards" not in existing:
        conn.execute("ALTER TABLE nfl_player_stats ADD COLUMN receiving_air_yards INTEGER")


def store(conn, kind: str, df: pd.DataFrame, game_ids: dict) -> int:
    """Write one season's rows, replacing any already stored for those games. Returns rows written."""
    if kind == "injuries":
        return store_injuries(conn, df)
    if kind == "rosters":
        return store_rosters(conn, df)
    if kind == "snaps":
        return store_snaps(conn, df, game_ids)
    columns = TEAM_COLUMNS if kind == "team" else PLAYER_COLUMNS
    keys = ["game_id", "team"] + (["player_id", "player_display_name", "position"] if kind == "player" else [])
    df = df[keys + list(columns)].dropna(subset=["game_id", "team"]).copy()
    df["team"] = df["team"].replace(FRANCHISES)
    df["game_id"] = df["game_id"].map(game_ids)
    df = df.dropna(subset=["game_id"]).rename(columns={**columns, "player_display_name": "name"})
    if kind == "player":
        df = df.dropna(subset=["player_id"])
        df["name"] = df["name"].fillna(df["player_id"])   # a few old rows have no name
        involved = df[["pass_attempts", "carries", "targets"]].fillna(0).sum(axis=1) > 0
        df = df[involved]                                   # offensive players only
    epa = {"passing_epa", "rushing_epa"}
    rows = [tuple(None if pd.isna(v) else (v if c in epa or isinstance(v, str) else int(v))
                  for c, v in zip(df.columns, r)) for r in df.itertuples(index=False)]
    table = f"nfl_{kind}_stats"
    conn.executemany(f"DELETE FROM {table} WHERE game_id = ?", [(int(g),) for g in set(df["game_id"])])
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(df.columns)}) VALUES "
                     f"({', '.join('?' for _ in df.columns)})", rows)
    return len(rows)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    ensure_columns(conn)
    game_ids = dict(conn.execute("SELECT source_game_id, game_id FROM nfl_game_info"))
    if not game_ids:
        raise SystemExit("No NFL games found. Run python -m nfl.import_nfl first.")
    latest = current_season()
    totals = {"team": 0, "player": 0, "injuries": 0, "snaps": 0, "rosters": 0}
    for season in range(FIRST_SEASON, latest + 1):
        for kind in ("team", "player", "injuries", "snaps", "rosters"):   # snaps after players: they're matched to them
            if season < FIRST[kind]:
                continue
            df = fetch(kind, season, refresh=season == latest)
            if df is not None:
                totals[kind] += store(conn, kind, df, game_ids)
        conn.commit()
    conn.close()
    print(f"Stored {totals['team']:,} team-game and {totals['player']:,} player-game stat rows ({FIRST_SEASON}-{latest}) "
          f"{totals['injuries']:,} injury report entries ({FIRST['injuries']}-{latest}) "
          f"{totals['snaps']:,} snap counts ({FIRST['snaps']}-{latest}) "
          f"and {totals['rosters']:,} weekly roster entries ({FIRST['rosters']}-{latest}).")


if __name__ == "__main__":
    main()
