"""Who actually starts at quarterback: nflverse's schedule, checked against the box scores.

nflverse's schedule names each team's starting quarterback, but the name can go stale: in 2026
it kept listing Drew Lock for Seattle for four weeks after Sam Darnold came back. So:

1. Finished games: if the listed quarterback didn't throw a pass, the start goes to the
   quarterback who threw the most (from the player stats).
2. Upcoming games: if the listed quarterback is neither the one who started the team's last game
   nor its usual starter (the most frequent in its last 8 games), and the last starter finished
   that game (threw most of the passes) and isn't Out or Doubtful on this week's injury report,
   the last starter is assumed to start again. (A listed usual starter is trusted: that's a
   starter coming back from an injury.)
3. Upcoming games: if the quarterback chosen above is ruled out (Out or Doubtful on the injury
   report, or on injured reserve or another reserve list, or released, on nflverse's weekly
   roster), the starter is the team's other quarterback on the roster who most recently started
   for it (then the one who has thrown the most passes for it, then anywhere). Players on injured
   reserve aren't on the weekly injury report, so the roster is what catches them (Jalen Hurts
   in October 2026: the schedule kept listing him, and Tanner McKee started).
4. nfl/qb_overrides.csv: a line "game,team,qb" (e.g. 2026_06_SEA_DEN,SEA,Sam Darnold, or the game
   as in its page address, 2026-06-sea-den) names the starter for one game still to be played and
   beats everything above. For benchings the data can't see. Finished games always go by the box
   score, so a wrong guess there never reaches the ratings.
"""

import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd

OVERRIDES_PATH = Path(__file__).with_name("qb_overrides.csv")
RULED_OUT = {"Out", "Doubtful"}
OFF_ROSTER = {"RES", "CUT", "RET", "EXE", "TRD", "TRT"}   # reserve lists, released, retired, traded
FINISHED_SHARE = 0.5   # threw at least this share of his team's passes: he finished the game
USUAL_STARTER_GAMES = 8
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def name_key(name) -> str:
    """'D.K. Metcalf' and 'DK Metcalf' -> 'dkmetcalf': for matching names between nflverse files."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


@lru_cache(maxsize=None)
def person(name) -> str:
    """A looser key for the same person spelled differently: first initial + last name, without
    Jr./II. 'Mike Vick' = 'Michael Vick', 'Mitch Trubisky' = 'Mitchell Trubisky', 'PJ Walker' = 'Phillip Walker'."""
    words = [name_key(w) for w in str(name).replace(".", " ").split()]
    words = [w for w in words if w and w not in SUFFIXES]
    if not words:
        return ""
    return f"{words[0][0]}:{words[-1]}" if len(words) > 1 else words[0]


def read_overrides(path: Path = OVERRIDES_PATH) -> dict:
    """{(nflverse game id, team): quarterback} from the overrides file, if there is one."""
    if not path.exists():
        return {}
    with path.open() as f:
        return {(r["game"].strip().upper().replace("-", "_"), r["team"].strip().upper()): r["qb"].strip()
                for r in csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
                if r.get("game") and r.get("team") and r.get("qb")}


def _table(conn, name: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def correct_quarterbacks(conn, games: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """games (with id, season, week, date, status, home, away, home_qb, away_qb) with home_qb and
    away_qb corrected as described above. Adds home_qb_listed/away_qb_listed (nflverse's names)."""
    games = games.copy()
    for side in ("home", "away"):
        games[f"{side}_qb_listed"] = games[f"{side}_qb"]
    overrides = read_overrides() if overrides is None else overrides

    passers = pd.DataFrame()
    if _table(conn, "nfl_player_stats"):
        passers = pd.read_sql("SELECT game_id, team, name, pass_attempts FROM nfl_player_stats "
                              "WHERE pass_attempts > 0", conn)
    injuries = {}
    if _table(conn, "nfl_injuries"):
        for season, week, team, name, status in conn.execute("SELECT season, week, team, name, status FROM nfl_injuries"):
            injuries[(season, week, team, person(name))] = status

    rosters: dict = {}   # (season, team) -> {week: {person: (status, name, position)}}
    if _table(conn, "nfl_rosters"):
        for season, week, team, name, position, status in conn.execute(
                "SELECT season, week, team, name, position, status FROM nfl_rosters WHERE position = 'QB'"):
            rosters.setdefault((season, team), {}).setdefault(week, {})[person(name)] = (status, name, position)

    def roster_for(season, week, team) -> dict:
        """The team's quarterbacks that week (or its latest roster before it that season)."""
        weeks = rosters.get((season, team), {})
        known = [w for w in weeks if w <= week]
        return weeks[max(known)] if known else {}

    def ruled_out(season, week, team, qb) -> bool:
        if injuries.get((season, week, team, person(qb))) in RULED_OUT:
            return True
        entry = roster_for(season, week, team).get(person(qb))
        return entry is not None and entry[0] in OFF_ROSTER

    # the schedule's spelling of each team's quarterbacks, so a corrected name matches the listed ones
    spelling = {}
    for side in ("home", "away"):
        for team, name in zip(games[side], games[f"{side}_qb"]):
            if isinstance(name, str) and name:
                spelling.setdefault((team, person(name)), name)

    box = {}   # (game id, team) -> (top passer, his share of the team's passes, every passer)
    if not passers.empty:
        passers = passers.assign(person=passers["name"].map(person))
        passers = passers.sort_values(["game_id", "team", "pass_attempts"], ascending=[True, True, False])
        keys = ["game_id", "team"]
        totals = passers.groupby(keys)["pass_attempts"].sum()
        people = passers.groupby(keys)["person"].agg(set)
        for r in passers.drop_duplicates(keys).itertuples(index=False):
            key = (r.game_id, r.team)
            box[key] = (spelling.get((r.team, r.person), r.name), r.pass_attempts / totals[key], people[key])

    thrown: dict = {}   # (team or None for anywhere, person) -> passes thrown
    if not passers.empty:
        for (team, who), n in passers.groupby(["team", "person"])["pass_attempts"].sum().items():
            thrown[(team, who)] = n
            thrown[(None, who)] = thrown.get((None, who), 0) + n

    def replacement(season, week, team, out_qb):
        """The team's healthy quarterback most likely to start instead of out_qb (None if unknown)."""
        options = [(key, name) for key, (status, name, _) in roster_for(season, week, team).items()
                   if status not in OFF_ROSTER and key != person(out_qb)]
        if not options:
            return None
        starts = history.get(team, [])

        def rank(option):
            key = option[0]
            last = max((i for i, q in enumerate(starts) if q and person(q) == key), default=-1)
            return (last, thrown.get((team, key), 0), thrown.get((None, key), 0))
        key, name = max(options, key=rank)
        return spelling.get((team, key), name)

    source_ids = {}
    if _table(conn, "nfl_game_info"):
        source_ids = dict(conn.execute("SELECT game_id, source_game_id FROM nfl_game_info"))
    last_start: dict[str, tuple] = {}   # team -> (starter, finished the game?)
    history: dict[str, list] = {}         # team -> its starters, oldest first
    corrected = {"home": {}, "away": {}}
    ordered = games.sort_values(["date", "id"])
    for g in ordered[["id", "season", "week", "status", "home", "away", "home_qb", "away_qb"]].itertuples(index=False):
        for side in ("home", "away"):
            team, listed = getattr(g, side), getattr(g, f"{side}_qb")
            qb = listed if isinstance(listed, str) and listed else None
            if g.status == "final":
                if (g.id, team) in box:
                    top, share, people = box[(g.id, team)]
                    if qb is None or person(qb) not in people:
                        qb = top                      # the listed quarterback didn't throw a pass
                    last_start[team] = (qb, person(qb) == person(top) and share >= FINISHED_SHARE)
                history.setdefault(team, []).append(qb)
            elif qb is not None and team in last_start:
                previous, finished = last_start[team]
                recent = [q for q in history.get(team, [])[-USUAL_STARTER_GAMES:] if q]
                usual = Counter(recent).most_common(1)[0][0] if recent else None
                out = injuries.get((g.season, g.week, team, person(previous))) in RULED_OUT
                if previous and person(qb) not in (person(previous), person(usual)) and finished and not out:
                    qb = previous                     # a stale listing: the last starter is healthy
            if g.status != "final":
                if qb is not None and ruled_out(g.season, g.week, team, qb):
                    qb = replacement(g.season, g.week, team, qb) or qb
                qb = overrides.get((source_ids.get(g.id), team), qb)
            corrected[side][g.id] = qb
    for side in ("home", "away"):
        games[f"{side}_qb"] = games["id"].map(corrected[side])
    return games
