"""Squad values for every club in the European cups model, from Transfermarkt: python -m ucl.squad_values

The match model rates clubs from their results. The betting market also knows how good a squad is
on paper, which matters most where results say little: early in a season, after a summer of
transfers, and between leagues that rarely meet. Transfermarkt's player values are a free, if
rough, measure of that.

Source: transfermarkt-datasets (CC0), already downloaded for the Premier League's squad values
(python -m epl.inspect_transfermarkt). Its updates paused in mid-July 2026, so values stop at June
2026: this season's transfers aren't in them.

Built as for the Premier League (epl/import_transfermarkt.py): on the first of every month, each
club's squad is the players whose latest transfer or appearance was for it, valued at their latest
valuation (if from the last 18 months), and its value is the total of its 18 most valuable
players. Transfermarkt clubs are linked to ESPN's by matching games (same date, give or take a day
for time zones, and the same score) in every competition both have. Saved to soccer_squad_values.
"""

import sqlite3
from collections import Counter

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from epl.import_transfermarkt import read, squad_values

FIRST_MONTH = "2013-07-01"
MIN_MATCHES = 6          # games two clubs must share before they're linked


def club_mapping(tm_games: pd.DataFrame, ours: pd.DataFrame) -> dict[int, str]:
    """Transfermarkt club id -> ESPN team id: for each Transfermarkt club, the ESPN club it matches
    most often (same-day matches count three times as much as a day either side), if at least
    MIN_MATCHES times and twice as often as any other (other games on the same day can share a
    score); one Transfermarkt club per ESPN club."""
    tm = tm_games.dropna(subset=["home_club_goals", "away_club_goals"]).copy()
    tm["date"] = pd.to_datetime(tm["date"])
    ours = ours.assign(date=pd.to_datetime(ours["date"]))
    votes: dict[int, Counter] = {}
    for shift, weight in ((0, 3), (-1, 1), (1, 1)):
        m = tm.merge(ours.assign(date=ours["date"] + pd.Timedelta(days=shift)),
                     left_on=["date", "home_club_goals", "away_club_goals"], right_on=["date", "home_score", "away_score"])
        for tm_col, our_col in (("home_club_id", "home_id"), ("away_club_id", "away_id")):
            for club, espn in zip(m[tm_col], m[our_col]):
                votes.setdefault(int(club), Counter())[espn] += weight
    best: dict[str, tuple[int, int]] = {}
    for club, counts in votes.items():
        top = counts.most_common(2)
        espn, n = top[0]
        second = top[1][1] if len(top) > 1 else 0
        if n >= 3 * MIN_MATCHES and n >= 2 * second and n > best.get(espn, (None, 0))[1]:
            best[espn] = (club, n)
    return {club: espn for espn, (club, _) in best.items()}


def main() -> None:
    print("Reading Transfermarkt tables...", flush=True)
    tm_games, transfers, appearances, valuations = (read(t) for t in ("games", "transfers", "appearances", "player_valuations"))
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    ours = pd.read_sql("SELECT date, home_id, away_id, home_score, away_score FROM soccer_games WHERE status = 'final'", conn)
    mapping = club_mapping(tm_games[tm_games["date"] >= FIRST_MONTH], ours)
    print(f"Linked {len(mapping)} Transfermarkt clubs to ESPN's.", flush=True)
    months = pd.date_range(FIRST_MONTH, valuations["date"].max(), freq="MS")
    values = squad_values(transfers, appearances, valuations, set(mapping), months)
    values["espn_id"] = values["club_id"].map(mapping)
    with conn:
        conn.execute("DELETE FROM soccer_squad_values")
        conn.executemany("INSERT INTO soccer_squad_values (espn_id, month, value_eur, players) VALUES (?, ?, ?, ?)",
                         [(r.espn_id, r.week.strftime("%Y-%m-%d"), float(r.value_eur), int(r.players)) for r in values.itertuples()])
    latest = values[values["week"] == values["week"].max()].sort_values("value_eur", ascending=False)
    names = dict(conn.execute("SELECT team_id, name FROM soccer_teams").fetchall())
    print(f"Saved {len(values):,} monthly squad values, {months[0]:%Y-%m} to {months[-1]:%Y-%m}. Most valuable in the last month:")
    for r in latest.head(8).itertuples():
        print(f"  {names.get(r.espn_id, r.espn_id):<28} €{r.value_eur / 1e6:,.0f}m")
    conn.close()


if __name__ == "__main__":
    main()
