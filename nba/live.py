"""Who's on each NBA team and who's hurt, from ESPN's public feeds, for upcoming games.

Usage:
    python -m nba.live             # current rosters and injury report, saved to the database
    python -m nba.live --record    # ...and a dated copy of the injury report in records/nba_injuries/

Rosters come from each team's roster page (site.api.espn.com/.../teams/{id}/roster) and the injury
report from ESPN's NBA injuries page (Out, Doubtful, Day-To-Day, with expected return dates). Player
ids are ESPN's, the same as in the box scores. Players listed Out or Doubtful don't count in a
team's expected lineup (nba/model.py); Day-To-Day players do. If the feeds are unreachable, the
last saved rosters and report are used (or, with none, each team's players from its recent games).
"""

import re
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from db import DB_PATH, SCHEMA_PATH
from nba.model import fill_minutes

ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
UNAVAILABLE = {"Out", "Doubtful", "Suspension", "Suspended"}
RECORD_DIR = Path("records/nba_injuries")


def player_id(athlete: dict) -> str | None:
    """ESPN's athlete id: the 'id' field, or the number in his player page link."""
    if athlete.get("id"):
        return str(athlete["id"])
    for link in athlete.get("links") or []:
        found = re.search(r"/id/(\d+)", str(link.get("href", "")))
        if found:
            return found.group(1)
    return None


def parse_roster(data: dict, team: str) -> list[dict]:
    rows = []
    for a in data.get("athletes") or []:
        if isinstance(a, dict) and "items" in a:          # some feeds group athletes by position
            rows += parse_roster({"athletes": a["items"]}, team)
            continue
        pid = player_id(a)
        if pid is None:
            continue
        injuries = a.get("injuries") or []
        rows.append({"team": team, "player_id": pid, "name": a.get("displayName") or a.get("fullName"),
                     "position": (a.get("position") or {}).get("abbreviation"),
                     "status": injuries[0].get("status") if injuries and isinstance(injuries[0], dict) else None})
    return rows


def parse_injuries(data: dict, abbreviations: dict) -> list[dict]:
    """abbreviations: ESPN team abbreviation -> our team name."""
    rows = []
    for entry in data.get("injuries") or []:
        for item in entry.get("injuries") or []:
            athlete = item.get("athlete") or {}
            abbr = (athlete.get("team") or {}).get("abbreviation") or entry.get("abbreviation")
            team = abbreviations.get(abbr) or abbreviations.get(entry.get("displayName"))
            pid = player_id(athlete)
            if team is None or pid is None:
                continue
            status = item.get("status")
            fantasy = ((item.get("details") or {}).get("fantasyStatus") or {}).get("abbreviation")
            if fantasy in ("OUT", "OFS"):
                status = "Out"
            rows.append({"team": team, "player_id": pid, "name": athlete.get("displayName"), "status": status,
                         "return_date": (item.get("details") or {}).get("returnDate"),
                         "comment": item.get("shortComment"), "updated": item.get("date")})
    return rows


def fetch(conn, get=requests.get) -> tuple[list[dict], list[dict]]:
    teams = conn.execute("SELECT espn_id, name, abbreviation FROM nba_teams").fetchall()
    rosters = []
    for espn_id, name, _ in teams:
        response = get(ROSTER_URL.format(team_id=espn_id), timeout=60)
        response.raise_for_status()
        rosters += parse_roster(response.json(), name)
    response = get(INJURIES_URL, timeout=60)
    response.raise_for_status()
    lookup = {abbr: name for _, name, abbr in teams} | {name: name for _, name, _ in teams}
    return rosters, parse_injuries(response.json(), lookup)


def save(conn, rosters: list[dict], injuries: list[dict]) -> None:
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if rosters:
        conn.execute("DELETE FROM nba_rosters_now")
        conn.executemany("INSERT OR REPLACE INTO nba_rosters_now (team, player_id, name, position, fetched_at) VALUES (?, ?, ?, ?, ?)",
                         [(r["team"], r["player_id"], r["name"], r["position"], fetched) for r in rosters])
    # the injury report: the dedicated page, plus any roster entry it doesn't have
    report = {(r["team"], r["player_id"]): r for r in injuries}
    for r in rosters:
        if r["status"] and (r["team"], r["player_id"]) not in report:
            report[(r["team"], r["player_id"])] = {**r, "return_date": None, "comment": None, "updated": None}
    if injuries or rosters:
        conn.execute("DELETE FROM nba_injuries_now")
        conn.executemany("""INSERT OR REPLACE INTO nba_injuries_now (team, player_id, name, status, return_date, comment, updated, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                         [(r["team"], r["player_id"], r["name"], r["status"], r.get("return_date"), r.get("comment"),
                           r.get("updated"), fetched) for r in report.values()])
    conn.commit()


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The saved rosters and injury report (empty frames if none)."""
    try:
        rosters = pd.read_sql("SELECT team, player_id, name, position FROM nba_rosters_now", conn)
        injuries = pd.read_sql("SELECT team, player_id, name, status, return_date, comment FROM nba_injuries_now", conn)
    except Exception:
        rosters = pd.DataFrame(columns=["team", "player_id", "name", "position"])
        injuries = pd.DataFrame(columns=["team", "player_id", "name", "status", "return_date", "comment"])
    return rosters, injuries


def unavailable(injuries: pd.DataFrame, on: str | None = None) -> set:
    """Player ids ruled out (Out or Doubtful). on: a game date; a player whose expected return date
    is on or before it counts as back (used for games further ahead, e.g. the season simulation)."""
    out = injuries[injuries["status"].isin(UNAVAILABLE)]
    if on is not None:
        back = pd.to_datetime(out["return_date"], errors="coerce") <= pd.Timestamp(on)
        out = out[~back.fillna(False)]
    return set(out["player_id"].astype(str))


def main(argv: list[str] | None = None, get=requests.get) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    try:
        rosters, injuries = fetch(conn, get)
    except Exception as error:
        print(f"Couldn't get NBA rosters and injuries from ESPN ({error.__class__.__name__}); using the last saved ones.")
        conn.close()
        return
    save(conn, rosters, injuries)
    if "--record" in argv:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        _, report = load(conn)
        report.to_csv(RECORD_DIR / f"{date.today().isoformat()}.csv", index=False)
    conn.close()
    out = sum(r["status"] in UNAVAILABLE for r in injuries)
    print(f"NBA rosters: {len(rosters)} players on {len({r['team'] for r in rosters})} teams; "
          f"injury report: {len(injuries)} players ({out} out).")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- expected lineups

NO_HISTORY_MINUTES = 5.0    # a player with no NBA minutes yet (a rookie, a two-way player): a small role
RECENT_DAYS = 400           # without a roster from ESPN: players whose latest game for the team is this recent


def latest_lines(players: pd.DataFrame) -> pd.DataFrame:
    """Each player's latest game he was available for: his team then, name, position, and his
    usual minutes after it (minutes_after: counting coach's-decision games as 0, so a deep-bench
    player who rarely gets in counts for little)."""
    rows = players[players["rotation_after"].notna()] if "rotation_after" in players else players[players["minutes"] > 0]
    latest = rows.sort_values(["date", "game_id"]).drop_duplicates("player_id", keep="last").set_index("player_id")
    if "rotation_after" in latest:
        latest = latest.assign(minutes_after=latest["rotation_after"])
    return latest[["team", "name", "position", "date", "minutes_after"]]


def expected_lineup(team: str, latest: pd.DataFrame, rosters: pd.DataFrame, out: set, on: str) -> pd.DataFrame:
    """A team's available players and expected minutes for a game on date `on`: player_id, name,
    minutes, and the players ruled out (out=True, minutes 0) for showing on the site."""
    if not rosters.empty and team in set(rosters["team"]):
        members = rosters[rosters["team"] == team][["player_id", "name"]].astype({"player_id": str})
    else:
        recent = latest[(latest["team"] == team) &
                        (pd.to_datetime(latest["date"]) >= pd.Timestamp(on) - pd.Timedelta(days=RECENT_DAYS))]
        members = pd.DataFrame({"player_id": recent.index.astype(str), "name": recent["name"].to_numpy()})
    minutes = members["player_id"].map(latest["minutes_after"]).fillna(NO_HISTORY_MINUTES)
    lineup = members.assign(usual_minutes=minutes.to_numpy(), out=members["player_id"].isin(out).to_numpy())
    lineup["minutes"] = 0.0
    playing = ~lineup["out"]
    lineup.loc[playing, "minutes"] = fill_minutes(lineup.loc[playing, "usual_minutes"].to_numpy())
    return lineup.sort_values("usual_minutes", ascending=False).reset_index(drop=True)
