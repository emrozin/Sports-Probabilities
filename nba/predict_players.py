"""Project NBA players for upcoming games (points, rebounds, assists, threes) and save them.

Usage:
    python -m nba.predict_players            # on your computer
    python -m nba.predict_players --record   # ...and the public record, records/nba_player_projections.csv

Run after python -m nba.predict_upcoming: each team's expected points come from that game's
prediction. Lineups are each team's current roster minus players listed Out or Doubtful
(nba/live.py). Like the game predictions, a projection can be refreshed until tip-off and is never
changed after it.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from nba import live
from nba.model import load_players
from nba.players import STATS, add_rates, position_group, position_rates, project, team_averages
from nba.predict_upcoming import MODEL_NAME, MODEL_VERSION
from nfl.predict_upcoming import EASTERN, has_started

RECORD_PATH = Path("records/nba_player_projections.csv")
COLUMNS = ["game", "date", "team", "player_id", "name", "minutes", *STATS, "model_version", "created_at"]


def upcoming_lineups(conn, players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """For each game and team: the expected lineup with each player's latest rate sums."""
    latest = live.latest_lines(players)
    rosters, injuries = live.load(conn)
    out = live.unavailable(injuries)
    sums = (players[players["minutes"] > 0].sort_values(["date", "game_id"]).drop_duplicates("player_id", keep="last")
            .set_index("player_id")[[f"post_{c}" for c in [*STATS, "minutes"]]])
    rows = []
    for g in games.itertuples():
        for team in (g.home, g.away):
            lineup = live.expected_lineup(team, latest, rosters, out, g.date)
            lineup = lineup[~lineup["out"]]
            positions = lineup["player_id"].map(latest["position"])
            if not rosters.empty:
                positions = positions.fillna(lineup["player_id"].map(rosters.set_index("player_id")["position"]))
            rows.append(lineup.assign(game_id=g.id, team=team, group=positions.map(position_group).to_numpy())
                        .join(sums, on="player_id"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    now = now or datetime.now(EASTERN)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    predicted = pd.read_sql("""SELECT g.id, g.date, g.kickoff, i.source_game_id AS game, h.name AS home, a.name AS away,
                                      pg.exp_home, pg.exp_away
                               FROM prediction_goals pg JOIN games g ON g.id = pg.game_id
                               JOIN nba_game_info i ON i.game_id = g.id
                               JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                               WHERE pg.model_name = ? AND pg.model_version = ? AND g.status = 'scheduled'""",
                            conn, params=(MODEL_NAME, MODEL_VERSION))
    predicted = predicted[[not has_started(g.date, g.kickoff if isinstance(g.kickoff, str) else None, now)
                           for g in predicted.itertuples()]]
    if predicted.empty:
        print("No upcoming NBA games with predictions; run python -m nba.predict_upcoming first.")
        conn.close()
        return
    players = add_rates(load_players(conn))
    lineup = upcoming_lineups(conn, players, predicted)
    teams = team_averages(players).sort_values("game_id").drop_duplicates("team", keep="last").set_index("team")
    totals = pd.concat([pd.DataFrame({"game_id": predicted["id"], "team": predicted[side], "team_points": predicted[f"exp_{side}"]})
                        for side in ("home", "away")], ignore_index=True)
    for s in STATS:
        if s != "points":
            totals[f"team_{s}"] = totals["team"].map(teams[f"post_team_{s}"])
    proj = project(lineup, totals, position_rates(players[players["season"] >= players["season"].max() - 3]), "post_")
    created = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S")
    rows = [(int(r.game_id), r.team, str(r.player_id), r.name, float(r.proj_minutes), *(float(getattr(r, f"proj_{s}")) for s in STATS),
             MODEL_VERSION, created) for r in proj.itertuples() if r.proj_minutes >= 1]
    with conn:
        conn.executemany(f"DELETE FROM nba_player_projections WHERE game_id = ?", [(int(i),) for i in predicted["id"]])
        conn.executemany(f"""INSERT INTO nba_player_projections (game_id, team, player_id, name, minutes, {', '.join(STATS)},
                                                                  model_version, created_at)
                             VALUES ({', '.join('?' * (7 + len(STATS)))})""", rows)
    if "--record" in argv:
        export(conn)
    conn.close()
    print(f"Saved NBA player projections for {len(rows)} players in {len(predicted)} games.")


def restore(conn) -> int:
    if not RECORD_PATH.exists():
        return 0
    rec = pd.read_csv(RECORD_PATH, dtype={"game": str, "player_id": str})
    ids = dict(conn.execute("SELECT source_game_id, game_id FROM nba_game_info"))
    rec["game_id"] = rec["game"].map(ids)
    rec = rec.dropna(subset=["game_id"])
    conn.executemany(f"""INSERT OR REPLACE INTO nba_player_projections (game_id, team, player_id, name, minutes, {', '.join(STATS)},
                                                                        model_version, created_at)
                         VALUES ({', '.join('?' * (7 + len(STATS)))})""",
                     [(int(r.game_id), r.team, r.player_id, r.name, r.minutes, *(getattr(r, s) for s in STATS), r.model_version, r.created_at)
                      for r in rec.itertuples()])
    conn.commit()
    return len(rec)


def export(conn) -> None:
    current = pd.read_sql(f"""SELECT i.source_game_id AS game, g.date, p.team, p.player_id, p.name, p.minutes, {', '.join('p.' + s for s in STATS)},
                                     p.model_version, p.created_at
                              FROM nba_player_projections p JOIN games g ON g.id = p.game_id JOIN nba_game_info i ON i.game_id = g.id""", conn)
    old = pd.read_csv(RECORD_PATH, dtype={"game": str, "player_id": str}) if RECORD_PATH.exists() else pd.DataFrame(columns=COLUMNS)
    combined = (pd.concat([old.reindex(columns=COLUMNS), current.astype({"game": str, "player_id": str})[COLUMNS]], ignore_index=True)
                  .drop_duplicates(["game", "player_id"], keep="last").sort_values(["date", "game", "team", "minutes"],
                                                                                    ascending=[True, True, True, False]))
    RECORD_PATH.parent.mkdir(exist_ok=True)
    combined.round(2).to_csv(RECORD_PATH, index=False)


if __name__ == "__main__":
    main()
