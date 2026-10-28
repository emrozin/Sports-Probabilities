"""Run the European cups' walk-forward backtest and save its predictions: python -m ucl.run_backtest [--all]

Before every match day of the Champions League, the Europa League and the Conference League since
2016-17 (the Conference League began in 2021-22), the model is fitted on the games played before
that day only (domestic and European) and predicts the day's games. Settings were tuned on the
Champions League's 2016-17 to 2020-21 seasons (ucl/tune_model.py); for every cup, 2021-22 on is
reported as the honest test.

The first run takes several minutes (one fit per match day); after that each update only adds the
match days that have been played since (--all redoes everything). It also saves a summary for the
website (records/europe_backtest.json): each competition's and season's log loss for the model and
two benchmarks.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from ucl.cups import CUPS
from ucl.model import backtest, load_games, load_values, log_loss, scored_games

ALL = tuple(c["slug"] for c in CUPS.values())

BACKTEST_NAME = "europe_goals_backtest"
BACKTEST_VERSION = "v3"   # v2: squad values; v3: league games learn from xG and blend in Elo
START_SEASON = 2016
HELD_OUT_FROM = 2021


def save(conn, results, name=BACKTEST_NAME, version=BACKTEST_VERSION, replace: bool = True) -> int:
    ids = dict(conn.execute("SELECT espn_id, game_id FROM cup_game_info").fetchall())
    results = results[results["event_id"].isin(ids)]
    with conn:
        for table in ("predictions", "prediction_goals"):
            if replace:
                conn.execute(f"DELETE FROM {table} WHERE model_name = ? AND model_version = ?", (name, version))
            else:
                conn.executemany(f"DELETE FROM {table} WHERE model_name = ? AND model_version = ? AND game_id = ?",
                                 [(name, version, ids[e]) for e in results["event_id"]])
        conn.executemany("INSERT INTO predictions (game_id, model_name, model_version, outcome, probability) VALUES (?, ?, ?, ?, ?)",
                         [(ids[r.event_id], name, version, outcome, float(getattr(r, outcome)))
                          for r in results.itertuples() for outcome in ("home", "draw", "away")])
        conn.executemany("INSERT INTO prediction_goals (game_id, model_name, model_version, exp_home, exp_away) VALUES (?, ?, ?, ?, ?)",
                         [(ids[r.event_id], name, version, float(r.exp_home), float(r.exp_away)) for r in results.itertuples()])
    return len(results)


SUMMARY_PATH = Path("records/europe_backtest.json")


def saved_results(conn, name=BACKTEST_NAME, version=BACKTEST_VERSION) -> pd.DataFrame:
    return pd.read_sql("""SELECT i.espn_id AS event_id, l.code AS competition, g.date, g.season,
                                 MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS home,
                                 MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS draw,
                                 MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS away
                          FROM predictions p JOIN games g ON g.id = p.game_id JOIN leagues l ON l.id = g.league_id
                          JOIN cup_game_info i ON i.game_id = g.id
                          WHERE p.model_name = ? AND p.model_version = ? GROUP BY p.game_id""", conn, params=(name, version))


def summary(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Each competition's and season's log loss: the model's, Elo's and the base rates' (ucl/tune_model.py)."""
    from ucl.tune_model import TUNING, benchmark_chances
    bench = benchmark_chances(games, TUNING, ALL)
    as_chances = lambda b, k: b.rename(columns={f"{k}_home": "home", f"{k}_draw": "draw", f"{k}_away": "away"})

    def scores(part):
        b = bench[bench["event_id"].isin(part["event_id"])]
        return {"games": len(part), "model": log_loss(part, games) if len(part) else None,
                "elo": log_loss(as_chances(b, "elo"), games) if len(b) else None,
                "base": log_loss(as_chances(b, "base"), games) if len(b) else None}
    out = {"held_out_from": HELD_OUT_FROM, "competitions": {}}
    for code, cup in CUPS.items():
        mine = results[results["competition"] == code]
        if mine.empty:
            continue
        out["competitions"][code] = {
            "seasons": [{"season": int(season), **scores(part), "held_out": int(season) >= HELD_OUT_FROM}
                        for season, part in mine.groupby("season")],
            "held_out": scores(mine[mine["season"] >= HELD_OUT_FROM])}
    out["held_out"] = scores(results[results["season"] >= HELD_OUT_FROM])
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    games = load_games(conn)
    if games.empty:
        raise SystemExit("No soccer games found. Run python -m ucl.import_espn --history first.")
    everything = "--all" in argv
    done = set() if everything else set(saved_results(conn)["event_id"])
    todo = scored_games(games, ALL)
    todo = todo[(todo["season"] >= START_SEASON) & ~todo["event_id"].isin(done)]
    if not todo.empty:
        n = save(conn, backtest(games, START_SEASON, dates=set(todo["date"]), competitions=ALL,
                                 values=load_values(conn)), replace=everything)
        print(f"Saved European cup backtest predictions for {n} games.")
    results = saved_results(conn)
    current = int(games.loc[games["competition"] == "uefa.champions", "season"].max())
    results = results[results["season"] < current]                 # finished seasons only
    if not results.empty:
        SUMMARY_PATH.parent.mkdir(exist_ok=True)
        s = summary(results, games)
        SUMMARY_PATH.write_text(json.dumps(s, indent=1))
        for code, c in s["competitions"].items():
            h = c["held_out"]
            if h["model"] is not None:
                print(f"{CUPS[code]['name']}, held out ({HELD_OUT_FROM}-{(HELD_OUT_FROM + 1) % 100:02d} on, {h['games']} games): "
                      f"log loss {h['model']:.4f}, Elo {h['elo']:.4f}, base rates {h['base']:.4f}.")
    conn.close()


if __name__ == "__main__":
    main()
