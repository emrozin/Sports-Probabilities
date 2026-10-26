"""NFL player projections: passing, rushing and receiving yards, touchdowns and receptions.

Usage:
    python -m nfl.import_nfl_stats     # once, then weekly (update.py does it)
    python -m nfl.players              # test the projections on past seasons

For each team in a game, built only from games played before it:

  1. Volume. The team's expected pass attempts and carries: its recent average, with recent games
     counting more. (Adjusting for the expected score, since trailing teams throw more, was tested
     and made the projections worse, so it's switched off.)
  2. Shares. Each player's share of the team's targets and carries in his recent games with the
     team, nudged up or down when his share of the offensive snaps in his latest game differs from
     usual (an early sign of a bigger or smaller role). The expected starting quarterback throws
     the passes.
  3. Efficiency. Catch rate, yards per target, yards per carry, yards per pass attempt and
     touchdown and interception rates, each pulled toward the average for the player's position
     when he has few opportunities, and adjusted for how much the opponent's defense allows.
  4. Touchdowns are scaled by how many points the team model expects the team to score. The
     chance of scoring at least one touchdown comes from the expected number (a Poisson count).

Injury reports: players listed Out or Doubtful aren't projected; players listed Questionable get
85% of their usual share; and a player who missed recent games injured is projected again when
he's back on the report without being ruled out. Otherwise, players who didn't record a stat in
their team's latest game are left out (usually inactive). On the held-out seasons, injury reports
cut the average miss by about 4-5% for rushing yards, receptions and receiving yards.

Comparisons between versions (total_error) count every player who was projected or actually
played, with a missing projection or a player who didn't play counted as zero, so a version can't
look better just by projecting fewer players.
"""

import json
import math
import sqlite3
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from db import DB_PATH
from nfl.quarterbacks import correct_quarterbacks

POSITIONS = ("QB", "RB", "WR", "TE")
LEAGUE_DEFAULTS = {  # starting points for the running league averages (roughly 2010s NFL levels)
    "catch": 0.64, "yds_per_target": 7.4, "rec_td_per_target": 0.045,
    "yds_per_carry": 4.2, "rush_td_per_carry": 0.03,
    "yds_per_att": 6.9, "pass_td_per_att": 0.043, "int_per_att": 0.025,
    "pass_att": 34.0, "carries": 26.0, "points": 22.0,
}


@dataclass
class Settings:
    """Tuned on 2012-2017 (nfl/players.py main)."""
    decay: float = 0.9           # each older game counts this much as the next one, for shares and volume
    efficiency_decay: float = 0.95   # efficiency changes more slowly, so it remembers more games
    shrink_targets: float = 40   # pseudo-opportunities pulling a receiver's rates toward his position's
    shrink_carries: float = 60
    shrink_attempts: float = 150
    shrink_defense: float = 200  # pseudo-plays pulling a defense's allowed rates toward the league's
    script: float = 0.0          # extra pass attempts per point the team is expected to trail by:
                                 # tested at 0.1-0.5 and made projections worse, so it's off
    injuries: bool = True        # use the weekly injury reports (see project_team)
    questionable: float = 0.85   # share multiplier for players listed Questionable
    redistribute: float = 0.0    # how much of a ruled-out player's share goes to his teammates (0-1):
                                 # tested at 0.5 and 1 and made projections slightly worse
    returner_games: int = 6      # a player who missed games injured is projected again if he played
                                 # within this many of his team's games
    snaps: float = 0.5           # target/carry share scaled by (latest snap share / usual snap share) ** this
    # Tested and switched off (nfl/player_experiments.py): neither improved the projections.
    defense_by_position: bool = False   # receiving yards adjusted by what the opponent allows to WRs / TEs / RBs
    air_yards: float = 0.0       # yards-per-target prior scaled by (player's depth of target / position's) ** this


SETTINGS = Settings()


def load(conn) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(games, team stats, player stats), with dates and teams, in date order."""
    games = pd.read_sql("""
        SELECT g.id, g.season, g.date, g.status, h.name AS home, a.name AS away, g.home_score, g.away_score,
               i.week, i.game_type, i.home_qb, i.away_qb
        FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
        JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
        JOIN nfl_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.id""", conn)
    games = correct_quarterbacks(conn, games)   # the listed starter, checked against the box scores
    teams = pd.read_sql("SELECT * FROM nfl_team_stats", conn)
    players = pd.read_sql("SELECT * FROM nfl_player_stats", conn)
    return games, teams, players


OUT_STATUSES = {"Out", "Doubtful"}   # players listed Doubtful almost never play (27 of 3,405 since 2009)


def load_snaps(conn) -> dict:
    """{(game id, player id): share of the team's offensive snaps}; empty if not imported."""
    try:
        rows = conn.execute("SELECT game_id, player_id, offense_pct FROM nfl_snap_counts WHERE player_id IS NOT NULL")
        return {(g, p): pct for g, p, pct in rows}
    except Exception:
        return {}


def load_injuries(conn) -> dict:
    """{(season, week, team): {player id: status}} from the weekly injury reports, plus players on
    injured reserve or another reserve list (or released) on that week's roster, as 'Out': they
    aren't on the injury report. The latest week's roster also counts for the week after it, in case
    the next roster isn't out yet. Empty if neither has been imported."""
    reports: dict = defaultdict(dict)
    try:
        df = pd.read_sql("SELECT season, week, team, player_id, status FROM nfl_injuries", conn)
        for r in df.itertuples():
            reports[(r.season, r.week, r.team)][r.player_id] = r.status
    except Exception:
        pass
    try:
        from nfl.quarterbacks import OFF_ROSTER
        off = pd.read_sql(f"SELECT season, week, team, player_id FROM nfl_rosters WHERE status IN "
                          f"({', '.join('?' * len(OFF_ROSTER))})", conn, params=sorted(OFF_ROSTER))
    except Exception:
        off = pd.DataFrame(columns=["season", "week", "team", "player_id"])
    if not off.empty:
        latest = off[off["season"] == off["season"].max()]
        latest = latest[latest["week"] == latest["week"].max()]
        for r in pd.concat([off, latest.assign(week=latest["week"] + 1)]).itertuples():
            reports[(r.season, r.week, r.team)].setdefault(r.player_id, "Out")
    return dict(reports)


class Ewma:
    """Exponentially weighted sums: each update first multiplies everything held so far by `decay`."""

    def __init__(self, decay: float):
        self.decay = decay
        self.sums: dict = defaultdict(float)

    def add(self, **values):
        for k in list(self.sums):
            self.sums[k] *= self.decay
        for k, v in values.items():
            self.sums[k] += v

    def get(self, key, default=0.0):
        return self.sums.get(key, default)


def rate(num: float, den: float, prior: float, strength: float) -> float:
    """A rate pulled toward `prior`, as if `strength` extra opportunities at the prior rate had been seen."""
    return (num + prior * strength) / (den + strength)


class Projector:
    """Plays through games in date order, keeping running team, defense and player numbers, and
    projects any game asked for using only games before its date."""

    def __init__(self, games, teams, players, settings: Settings = SETTINGS, injuries: dict | None = None,
                 snaps: dict | None = None):
        self.s = settings
        self.injuries = injuries or {}
        self.snaps = snaps or {}
        self.last_snap: dict[tuple, float] = {}                         # (team, player) -> latest snap share
        self.allowed_pos = defaultdict(lambda: Ewma(settings.efficiency_decay))   # (team, position) defense
        self.league_pos = defaultdict(lambda: Ewma(0.995))
        self.injured_out: dict[str, bool] = {}                         # player -> missed his latest team game injured
        self.team_games: dict[str, int] = defaultdict(int)             # team -> games played so far
        self.last_played: dict[tuple, int] = {}                        # (team, player) -> team game number
        self.games = games.set_index("id")
        self.team_stats = teams.set_index(["game_id", "team"])
        self.player_rows = {gid: grp for gid, grp in players.groupby("game_id")}
        self.volume = defaultdict(lambda: Ewma(settings.decay))        # team -> pass_att, carries, points, n
        self.allowed = defaultdict(lambda: Ewma(settings.efficiency_decay))   # team's defense
        self.usage = defaultdict(lambda: Ewma(settings.decay))         # (team, player) -> targets/carries shares
        self.skill = defaultdict(lambda: Ewma(settings.efficiency_decay))   # player -> efficiency numbers
        self.league = Ewma(0.995)
        self.last_game_players: dict[str, set] = {}
        self.roster: dict[str, set] = defaultdict(set)                # team -> every player seen with it
        self.info: dict[str, dict] = {}                                # player -> name, position

    # ---- learning from a finished game
    def update(self, game_id: int) -> None:
        g = self.games.loc[game_id]
        rows = self.player_rows.get(game_id)
        for team, opp, pts in ((g.home, g.away, g.home_score), (g.away, g.home, g.away_score)):
            if (game_id, team) not in self.team_stats.index:
                continue
            t = self.team_stats.loc[(game_id, team)]
            mine = rows[rows["team"] == team] if rows is not None else pd.DataFrame()
            targets = float(mine["targets"].fillna(0).sum()) if len(mine) else 0.0
            carries = float(t["carries"] or 0)
            pass_att = float(t["pass_attempts"] or 0)
            self.volume[team].add(pass_att=pass_att, carries=carries, points=float(pts), n=1.0)
            self.allowed[opp].add(pass_yds=float(t["passing_yards"] or 0), pass_att=pass_att,
                                  rush_yds=float(t["rushing_yards"] or 0), carries=carries)
            self.league.add(pass_yds=float(t["passing_yards"] or 0), pass_att=pass_att,
                            rush_yds=float(t["rushing_yards"] or 0), carries=carries)
            present = set()
            by_position = defaultdict(lambda: [0.0, 0.0])           # receiving yards and targets allowed
            for p in mine.itertuples():
                if p.position in ("WR", "TE", "RB"):
                    by_position[p.position][0] += 0.0 if pd.isna(p.receiving_yards) else float(p.receiving_yards)
                    by_position[p.position][1] += 0.0 if pd.isna(p.targets) else float(p.targets)
                present.add(p.player_id)
                self.info[p.player_id] = {"name": p.name, "position": p.position}
                v = lambda x: 0.0 if pd.isna(x) else float(x)
                snap = self.snaps.get((game_id, p.player_id))
                self.usage[(team, p.player_id)].add(
                    target_share=v(p.targets) / targets if targets else 0.0,
                    carry_share=v(p.carries) / carries if carries else 0.0, n=1.0,
                    snap=snap or 0.0, snap_n=1.0 if snap is not None else 0.0)
                if snap is not None:
                    self.last_snap[(team, p.player_id)] = snap
                self.skill[p.player_id].add(
                    targets=v(p.targets), receptions=v(p.receptions), rec_yds=v(p.receiving_yards),
                    rec_tds=v(p.receiving_tds), carries=v(p.carries), rush_yds=v(p.rushing_yards),
                    rush_tds=v(p.rushing_tds), att=v(p.pass_attempts), pass_yds=v(p.passing_yards),
                    pass_tds=v(p.passing_tds), ints=v(p.interceptions),
                    air=v(getattr(p, "receiving_air_yards", 0)), air_targets=v(p.targets)
                    if not pd.isna(getattr(p, "receiving_air_yards", np.nan)) else 0.0)
            for pos in ("WR", "TE", "RB"):
                yds, tgt = by_position[pos]
                self.allowed_pos[(opp, pos)].add(yds=yds, targets=tgt)
                self.league_pos[pos].add(yds=yds, targets=tgt)
            # players who didn't appear this game: their usage decays as if they had 0 share
            report = self.injuries.get((g.season, g.week, team), {})
            for pid in self.roster[team] - present:
                self.usage[(team, pid)].add(n=0.0)
                if pid in report:
                    self.injured_out[pid] = True          # missed it while on the injury report
            self.team_games[team] += 1
            for pid in present:
                self.injured_out.pop(pid, None)
                self.last_played[(team, pid)] = self.team_games[team]
            self.roster[team] |= present
            self.last_game_players[team] = present

    # ---- projecting a game
    def defense_factor(self, opp: str, kind: str) -> float:
        a, lg, k = self.allowed[opp], self.league, self.s.shrink_defense
        if kind == "pass":
            league = lg.get("pass_yds") / max(lg.get("pass_att"), 1) or LEAGUE_DEFAULTS["yds_per_att"]
            allowed = rate(a.get("pass_yds"), a.get("pass_att"), league, k)
        else:
            league = lg.get("rush_yds") / max(lg.get("carries"), 1) or LEAGUE_DEFAULTS["yds_per_carry"]
            allowed = rate(a.get("rush_yds"), a.get("carries"), league, k)
        return allowed / league

    def receiving_factor(self, opp: str, pos: str) -> float:
        """How much the opponent allows to this position's receivers, per target, relative to the league."""
        a, lg = self.allowed_pos[(opp, pos)], self.league_pos[pos]
        league = lg.get("yds") / max(lg.get("targets"), 1) or LEAGUE_DEFAULTS["yds_per_target"]
        return rate(a.get("yds"), a.get("targets"), league, self.s.shrink_defense / 3) / league

    def position_rates(self) -> dict:
        """League-wide efficiency by position, from every player's running numbers (cached per date)."""
        totals = defaultdict(lambda: defaultdict(float))
        for pid, sk in self.skill.items():
            pos = self.info.get(pid, {}).get("position")
            if pos in POSITIONS:
                for k, v in sk.sums.items():
                    totals[pos][k] += v
        d = LEAGUE_DEFAULTS
        out = {}
        for pos in POSITIONS:
            t = totals[pos]
            out[pos] = {
                "catch": t["receptions"] / t["targets"] if t["targets"] > 50 else d["catch"],
                "yds_per_target": t["rec_yds"] / t["targets"] if t["targets"] > 50 else d["yds_per_target"],
                "rec_td_per_target": t["rec_tds"] / t["targets"] if t["targets"] > 50 else d["rec_td_per_target"],
                "yds_per_carry": t["rush_yds"] / t["carries"] if t["carries"] > 50 else d["yds_per_carry"],
                "rush_td_per_carry": t["rush_tds"] / t["carries"] if t["carries"] > 50 else d["rush_td_per_carry"],
                "yds_per_att": t["pass_yds"] / t["att"] if t["att"] > 50 else d["yds_per_att"],
                "pass_td_per_att": t["pass_tds"] / t["att"] if t["att"] > 50 else d["pass_td_per_att"],
                "int_per_att": t["ints"] / t["att"] if t["att"] > 50 else d["int_per_att"],
                "adot": t["air"] / t["air_targets"] if t["air_targets"] > 50 else 8.0,
            }
        return out

    def project_team(self, team: str, opp: str, expected_points: float | None, expected_margin: float | None,
                     qb_name: str | None, rates: dict, season: int | None = None, week: int | None = None) -> list[dict]:
        s, vol = self.s, self.volume[team]
        n = vol.get("n")
        if n <= 0:
            return []
        pass_att = vol.get("pass_att") / n
        carries = vol.get("carries") / n
        avg_points = vol.get("points") / n or LEAGUE_DEFAULTS["points"]
        if expected_margin is not None:                      # game script: trailing teams throw more
            pass_att -= s.script * expected_margin
            carries += s.script * expected_margin * 0.6
        td_scale = float(np.clip((expected_points or avg_points) / avg_points, 0.6, 1.6))
        pass_factor, rush_factor = self.defense_factor(opp, "pass"), self.defense_factor(opp, "rush")

        candidates = set(self.last_game_players.get(team, set()))
        report = self.injuries.get((season, week, team), {}) if s.injuries else {}
        if report:
            # Players who missed recent games injured and aren't ruled out this week come back.
            for pid in self.roster[team]:
                recent = self.team_games[team] - self.last_played.get((team, pid), -99) <= s.returner_games
                # Still on this week's report but not ruled out: likely to play. (Players dropped from the
                # report entirely are often on injured reserve, so they don't count.)
                if self.injured_out.get(pid) and recent and pid in report and report[pid] not in OUT_STATUSES:
                    candidates.add(pid)
        # The listed starting quarterback plays even if he sat out the team's latest game.
        listed = None
        if qb_name:
            named = [pid for pid, i in self.info.items() if i["name"] == qb_name and i["position"] == "QB"]
            listed = next((pid for pid in named if pid in self.roster[team]), named[0] if named else None)
            if listed is None:
                # A quarterback with no stats yet (a debut): projected at his position's average rates.
                listed = f"new:{team}:{qb_name}"
                self.info[listed] = {"name": qb_name, "position": "QB"}
            candidates.add(listed)
        usage = {pid: self.usage[(team, pid)] for pid in candidates}

        def share(pid, k):
            u = usage[pid]
            value = u.get(k) / max(u.get("n"), 1e-9) * min(1.0, u.get("n"))
            if s.snaps and u.get("snap_n") > 0.5 and (team, pid) in self.last_snap:
                usual = u.get("snap") / u.get("snap_n")       # role change: latest snaps vs usual snaps
                if usual > 0.05:
                    value *= float(np.clip(self.last_snap[(team, pid)] / usual, 0.5, 1.5)) ** s.snaps
            return value
        target_shares = {pid: share(pid, "target_share") for pid in candidates}
        carry_shares = {pid: share(pid, "carry_share") for pid in candidates}
        out_now = {pid for pid in candidates if report.get(pid) in OUT_STATUSES and pid != listed}
        for shares in (target_shares, carry_shares):
            before = min(1.0, sum(shares.values()))
            for pid in out_now:                          # ruled out: no projection
                shares.pop(pid, None)
            for pid in shares:
                if report.get(pid) == "Questionable":
                    shares[pid] *= s.questionable
            remaining = sum(shares.values())
            if remaining > 0 and before > remaining:     # his teammates absorb (part of) his share
                scale = 1 + s.redistribute * (before - remaining) / remaining
                for pid in shares:
                    shares[pid] *= scale
            total = sum(shares.values())
            if total > 1:
                for pid in shares:
                    shares[pid] /= total
        candidates -= out_now

        # the expected starting quarterback throws the passes
        qbs = [pid for pid in candidates if self.info.get(pid, {}).get("position") == "QB"]
        starter = listed
        if starter is None and qbs:
            starter = max(qbs, key=lambda pid: self.skill[pid].get("att"))

        out = []
        for pid in candidates:
            pos = self.info.get(pid, {}).get("position")
            if pos not in POSITIONS:
                continue
            pr, sk = rates[pos], self.skill[pid]
            row = {"player_id": pid, "name": self.info[pid]["name"], "position": pos, "team": team,
                   "pass_att": 0.0, "pass_yds": 0.0, "pass_tds": 0.0, "ints": 0.0}
            if pid == starter:
                ypa = rate(sk.get("pass_yds"), sk.get("att"), pr["yds_per_att"], s.shrink_attempts)
                row.update(pass_att=pass_att, pass_yds=pass_att * ypa * pass_factor,
                           pass_tds=pass_att * rate(sk.get("pass_tds"), sk.get("att"), pr["pass_td_per_att"],
                                                    s.shrink_attempts) * td_scale,
                           ints=pass_att * rate(sk.get("ints"), sk.get("att"), pr["int_per_att"], s.shrink_attempts))
            elif pos == "QB":
                continue                                     # backup quarterbacks: not projected
            targets = target_shares.get(pid, 0.0) * pass_att
            car = carry_shares.get(pid, 0.0) * carries
            ypt_prior = pr["yds_per_target"]
            if s.air_yards and sk.get("air_targets") > 0:      # deeper targets: more yards per target
                adot = rate(sk.get("air"), sk.get("air_targets"), pr["adot"], s.shrink_targets)
                ypt_prior *= max(adot / pr["adot"], 0.2) ** s.air_yards
            rec_factor = self.receiving_factor(opp, pos) if s.defense_by_position and pos in ("WR", "TE", "RB") \
                else pass_factor
            row.update(
                targets=targets,
                receptions=targets * rate(sk.get("receptions"), sk.get("targets"), pr["catch"], s.shrink_targets),
                rec_yds=targets * rate(sk.get("rec_yds"), sk.get("targets"), ypt_prior, s.shrink_targets)
                * rec_factor,
                rec_tds=targets * rate(sk.get("rec_tds"), sk.get("targets"), pr["rec_td_per_target"],
                                       s.shrink_targets) * td_scale,
                carries=car,
                rush_yds=car * rate(sk.get("rush_yds"), sk.get("carries"), pr["yds_per_carry"], s.shrink_carries)
                * rush_factor,
                rush_tds=car * rate(sk.get("rush_tds"), sk.get("carries"), pr["rush_td_per_carry"],
                                    s.shrink_carries) * td_scale)
            row["anytime_td"] = 1 - math.exp(-(row["rec_tds"] + row["rush_tds"]))
            if max(row["pass_att"], targets, car) >= 0.5:
                out.append(row)
        return out

    def run(self, targets: dict) -> pd.DataFrame:
        """Project the games in `targets` ({game id: (home points, away points) or None}), playing
        through every finished game before each one's date."""
        rows = []
        by_date = self.games.groupby("date")
        wanted = defaultdict(list)
        for gid in targets:
            wanted[self.games.at[gid, "date"]].append(gid)
        for day, group in by_date:
            if day in wanted:
                rates = self.position_rates()
                for gid in wanted[day]:
                    g = self.games.loc[gid]
                    pts = targets[gid]
                    for side, team, opp, qb in (("home", g.home, g.away, g.home_qb), ("away", g.away, g.home, g.away_qb)):
                        exp_pts = margin = None
                        if pts is not None:
                            exp_pts = pts[0] if side == "home" else pts[1]
                            margin = (pts[0] - pts[1]) * (1 if side == "home" else -1)
                        for r in self.project_team(team, opp, exp_pts, margin, qb if isinstance(qb, str) else None,
                                                   rates, int(g.season), int(g.week)):
                            rows.append({"game_id": gid, **r})
            for gid in group.index[group["status"] == "final"]:
                self.update(gid)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- testing on past seasons

STATS = {  # projection column -> actual column
    "pass_yds": "passing_yards", "pass_tds": "passing_tds", "rush_yds": "rushing_yards",
    "receptions": "receptions", "rec_yds": "receiving_yards",
}


def benchmarks(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Every player-game with two simple benchmarks for each stat, using only earlier games: the player's
    average in his previous 3 games (last3_*), and his average so far that season (season_*, falling
    back to last3_* in his first game of a season)."""
    played = players.merge(games[["id", "season", "date"]], left_on="game_id", right_on="id")
    played = played.sort_values(["player_id", "date"]).reset_index(drop=True)
    for proj, actual in STATS.items():
        values = played[actual].fillna(0)
        previous = values.groupby(played["player_id"]).shift(1)               # each game sees only earlier ones
        played[f"last3_{proj}"] = (previous.groupby(played["player_id"]).rolling(3, min_periods=1).mean()
                                   .reset_index(level=0, drop=True))
        in_season = values.groupby([played["player_id"], played["season"]]).shift(1)
        keys = [played["player_id"], played["season"]]
        season_avg = in_season.fillna(0).groupby(keys).cumsum() / in_season.notna().groupby(keys).cumsum()
        played[f"season_{proj}"] = season_avg.where(in_season.notna().groupby(keys).cumsum() > 0).fillna(
            played[f"last3_{proj}"])
    return played


def compare(projections: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Projections (proj_* columns) next to what happened and the benchmarks, for players who played.
    Players projected but absent from the game's stats (didn't play, or recorded nothing) are left out."""
    projections = projections.rename(columns={c: f"proj_{c}" for c in projections.columns
                                              if c not in ("game_id", "player_id")})
    history = players[players["player_id"].isin(set(projections["player_id"]))]   # only these players' games
    return projections.merge(benchmarks(history, games), on=["game_id", "player_id"])


def accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Average miss of the projections and the benchmarks for each stat, from compare()'s output."""
    rows = []
    for proj, actual in STATS.items():
        sub = df[df[f"proj_{proj}"] > 0.5] if proj in ("pass_yds", "pass_tds") else df
        sub = sub.dropna(subset=[f"last3_{proj}"])
        truth = sub[actual].fillna(0)
        rows.append({"stat": proj, "players": len(sub),
                     "model": (sub[f"proj_{proj}"] - truth).abs().mean(),
                     "last 3 games": (sub[f"last3_{proj}"] - truth).abs().mean(),
                     "season average": (sub[f"season_{proj}"] - truth).abs().mean()})
    td = df.dropna(subset=["proj_anytime_td"])
    scored = ((td["rushing_tds"].fillna(0) + td["receiving_tds"].fillna(0)) > 0).astype(float)
    p = td["proj_anytime_td"].clip(0.01, 0.99)
    base = scored.mean() if len(td) else np.nan
    rows.append({"stat": "anytime TD (log loss)", "players": len(td),
                 "model": float(-(scored * np.log(p) + (1 - scored) * np.log(1 - p)).mean()) if len(td) else np.nan,
                 "last 3 games": np.nan,
                 "season average": float(-(scored * np.log(base) + (1 - scored) * np.log(1 - base)).mean())
                 if 0 < base < 1 else np.nan})
    return pd.DataFrame(rows)


def total_error(projections: pd.DataFrame, players: pd.DataFrame, game_ids) -> dict:
    """Average miss per player-game actually played, counting every player who was projected or played
    (missing projection or didn't play = 0). For comparing versions fairly."""
    columns = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "receptions": "receptions",
               "rec_yds": "receiving_yards"}
    actual = players[players["game_id"].isin(set(game_ids)) & players["position"].isin(POSITIONS)]
    p = projections[["game_id", "player_id", *columns]].rename(columns={k: f"p_{k}" for k in columns})
    a = actual[["game_id", "player_id", *columns.values()]].rename(columns={v: f"a_{k}" for k, v in columns.items()})
    m = p.merge(a, on=["game_id", "player_id"], how="outer").fillna(0)
    return {k: float((m[f"p_{k}"] - m[f"a_{k}"]).abs().sum() / max(len(actual), 1)) for k in columns}


def evaluate(projections: pd.DataFrame, players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Average miss (mean absolute error) of the projections and of the two benchmarks."""
    return accuracy(compare(projections, players, games))


# ---------------------------------------------------------------- ranges

RANGES_PATH = Path(__file__).with_name("ranges.json")
RANGE_BINS = {"pass_yds": [150, 200, 250], "rush_yds": [10, 25, 45, 70], "rec_yds": [10, 25, 45, 70],
              "receptions": [1.5, 3, 5]}
QUANTILES = [round(q, 2) for q in np.arange(0.05, 1.0, 0.05)]   # 5%, 10%, ... 95%


def fit_ranges(df: pd.DataFrame) -> dict:
    """How actual results spread around projections: for each stat and size of projection, quantiles of
    actual / projected (from compare()'s output, players who played)."""
    table = {}
    for stat, edges in RANGE_BINS.items():
        sub = df[df[f"proj_{stat}"] > 0.5]
        ratio = sub[STATS[stat]].fillna(0) / sub[f"proj_{stat}"]
        bins = np.digitize(sub[f"proj_{stat}"], edges)
        overall = np.quantile(ratio, QUANTILES)
        # a size band with few projections uses the spread of all of them
        table[stat] = {"edges": edges, "quantiles": [
            (np.quantile(ratio[bins == b], QUANTILES) if (bins == b).sum() >= 100 else overall).round(4).tolist()
            for b in range(len(edges) + 1)]}
    return table


def load_ranges() -> dict | None:
    return json.loads(RANGES_PATH.read_text()) if RANGES_PATH.exists() else None


def stat_range(table: dict, stat: str, projected: float, low: float = 0.1, high: float = 0.9) -> tuple[float, float]:
    """The range the actual result falls in with probability high - low (80% by default)."""
    t = table[stat]
    q = t["quantiles"][int(np.digitize(projected, t["edges"]))]
    return projected * float(np.interp(low, QUANTILES, q)), projected * float(np.interp(high, QUANTILES, q))


MILESTONES = {"pass_yds": list(range(150, 401, 25)), "rush_yds": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150],
              "rec_yds": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150], "receptions": list(range(1, 13))}


def milestone(stat: str, projected: float) -> float:
    """The round number closest to the projection, e.g. 90 receiving yards for a projection of 92."""
    options = MILESTONES[stat]
    return min(options, key=lambda m: (abs(m - projected), -m))


def chance_at_least(table: dict, stat: str, projected: float, threshold: float) -> float:
    """Chance the actual result is at least threshold, from the same spread as the ranges.
    Stats are whole numbers, so "at least 5 catches" is read as more than 4.5. Limited to 5-95%,
    the part of the spread the table covers."""
    t = table[stat]
    q = np.asarray(t["quantiles"][int(np.digitize(projected, t["edges"]))])
    needed = (threshold - 0.5) / projected
    # results equal to a quantile value (e.g. many players with 0 yards) count at the top of that tie
    below = float(np.interp(needed, q, QUANTILES, left=0.0, right=1.0))
    return float(np.clip(1 - below, 0.05, 0.95))


def chance_calibration(table: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted chance of reaching each player's milestone vs how often it happened, in bands."""
    rows = []
    for stat in RANGE_BINS:
        sub = df[df[f"proj_{stat}"] > 0.5]
        for proj, actual in zip(sub[f"proj_{stat}"], sub[STATS[stat]].fillna(0)):
            m = milestone(stat, proj)
            rows.append((stat, chance_at_least(table, stat, proj, m), actual >= m))
    out = pd.DataFrame(rows, columns=["stat", "chance", "hit"])
    out["band"] = pd.cut(out["chance"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0], include_lowest=True)
    return (out.groupby("band", observed=True)
            .agg(players=("hit", "size"), predicted=("chance", "mean"), actual=("hit", "mean")).reset_index())


# Sportsbook-style player props: "at least this many". The anytime touchdown comes from the projection itself.
PROPS = {"pass_yds": [225, 250, 275, 300], "rush_yds": [50, 75, 100], "rec_yds": [50, 75, 100],
         "receptions": [4, 5, 6, 7, 8]}
PROP_NAMES = {"pass_yds": "passing yards", "rush_yds": "rushing yards", "rec_yds": "receiving yards",
              "receptions": "receptions"}


def prop_chances(table: dict, proj: pd.DataFrame) -> pd.DataFrame:
    """Every prop's chance for every projected player: one row per (game_id, player_id, prop).
    Needs the projection columns pass_yds, rush_yds, rec_yds, receptions, anytime_td and position."""
    rows = []
    for r in proj.itertuples():
        qb = r.position == "QB"
        for stat, thresholds in PROPS.items():
            value = getattr(r, stat)
            if stat == "pass_yds" and not qb or stat in ("receptions", "rec_yds") and qb:
                continue   # passing props for quarterbacks only, receiving props for everyone else
            if pd.isna(value) or value < 0.5:
                continue
            for t in thresholds:
                rows.append((r.game_id, r.player_id, stat, t, f"{t}+ {PROP_NAMES[stat]}",
                             chance_at_least(table, stat, value, t)))
        if not pd.isna(r.anytime_td):
            rows.append((r.game_id, r.player_id, "td", 1, "scores a touchdown", float(r.anytime_td)))
    return pd.DataFrame(rows, columns=["game_id", "player_id", "stat", "threshold", "prop", "chance"])


def likeliest_props(chances: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """The n likeliest props in each group (e.g. per game or per day), at most one per player:
    his likeliest. Ties go to the bigger milestone."""
    best = (chances.sort_values(["chance", "threshold"], ascending=False)
            .drop_duplicates(["game_id", "player_id"]))
    return best.groupby(by, sort=False).head(n)


def prop_hit(row, actual) -> bool:
    """Whether a prop came in, given the player's actual stats row (nfl_player_stats columns)."""
    value = lambda c: 0 if c not in actual or pd.isna(actual[c]) else actual[c]
    if row["stat"] == "td":
        return value("rushing_tds") + value("receiving_tds") > 0
    return value(STATS[row["stat"]]) >= row["threshold"]


def props_backtest(table: dict, df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """The likeliest props from compare()'s output, with whether each one hit."""
    proj = df[["game_id", "player_id", "date", *(f"proj_{c}" for c in
                                                 ("position", "pass_yds", "rush_yds", "rec_yds", "receptions", "anytime_td"))]]
    proj = proj.rename(columns=lambda c: c.removeprefix("proj_"))
    chances = prop_chances(table, proj).merge(proj[["game_id", "player_id", "date"]], on=["game_id", "player_id"])
    picks = likeliest_props(chances, by, n)
    actual = df.set_index(["game_id", "player_id"])
    picks["hit"] = [prop_hit(r, actual.loc[(r["game_id"], r["player_id"])]) for _, r in picks.iterrows()]
    return picks


def coverage(table: dict, df: pd.DataFrame) -> dict:
    """Share of actual results inside the 80% range, per stat (should be close to 80%)."""
    out = {}
    for stat in RANGE_BINS:
        sub = df[df[f"proj_{stat}"] > 0.5]
        bounds = [stat_range(table, stat, v) for v in sub[f"proj_{stat}"]]
        actual = sub[STATS[stat]].fillna(0).to_numpy()
        inside = [lo <= a <= hi for (lo, hi), a in zip(bounds, actual)]
        out[stat] = float(np.mean(inside)) if inside else float("nan")
    return out


def backtest_targets(conn, first: int, last: int) -> dict:
    """Finished games in these seasons, with the team model's backtest expected points."""
    from nfl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION
    df = pd.read_sql("""SELECT g.id, pg.exp_home, pg.exp_away FROM games g
                        JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
                        LEFT JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = ? AND pg.model_version = ?
                        WHERE g.status = 'final' AND g.season BETWEEN ? AND ?""",
                     conn, params=(BACKTEST_NAME, BACKTEST_VERSION, first, last))
    return {r.id: (None if pd.isna(r.exp_home) else (r.exp_home, r.exp_away)) for r in df.itertuples()}


def main() -> None:
    """Test the projections on past seasons, and (re)build the ranges in nfl/ranges.json from the tuning seasons."""
    conn = sqlite3.connect(DB_PATH)
    games, teams, players = load(conn)
    if players.empty:
        raise SystemExit("No player stats found. Run python -m nfl.import_nfl_stats first.")
    injuries, snaps = load_injuries(conn), load_snaps(conn)
    table = None
    for label, (first, last) in (("Tuning seasons 2012-2017", (2012, 2017)), ("Held-out seasons 2018-2025", (2018, 2025))):
        projections = Projector(games, teams, players, injuries=injuries, snaps=snaps).run(backtest_targets(conn, first, last))
        compared = compare(projections, players, games)
        print(f"\n{label}: average miss per player-game (lower is better)")
        print(accuracy(compared).round(3).to_string(index=False))
        if table is None:
            table = fit_ranges(compared)   # ranges come from the tuning seasons only
            RANGES_PATH.write_text(json.dumps(table, indent=1))
        print("Share of results inside the 80% range: " +
              ", ".join(f"{k} {v:.0%}" for k, v in coverage(table, compared).items()))
        print("Chance of reaching the milestone shown on game pages, predicted vs actual:")
        print(chance_calibration(table, compared).round(3).to_string(index=False))
        print("Likeliest props (one per player), predicted vs how often they hit:")
        for label_, by, n in (("top 10 per day", "date", 10), ("top 5 per game", "game_id", 5)):
            picks = props_backtest(table, compared, by, n)
            print(f"  {label_}: {len(picks)} props, predicted {picks['chance'].mean():.1%}, hit {picks['hit'].mean():.1%}")
    conn.close()


if __name__ == "__main__":
    main()
