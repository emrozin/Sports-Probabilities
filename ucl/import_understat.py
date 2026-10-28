"""Expected goals (xG) for Europe's big five leagues from Understat: python -m ucl.import_understat

Understat has xG for every game in the Premier League, La Liga, the Bundesliga, Serie A and Ligue 1
since 2014-15 (the same source as the Premier League's own model, epl/import_understat.py, and the
same cache, data/raw/understat/: finished seasons are downloaded once, about 50 files the first time,
a few seconds apart). This links each Understat game to ESPN's game (soccer_games) and saves both
sides' xG in soccer_xg, which the Europe-wide model (ucl/model.py) learns from alongside the goals.

Linking: Understat and ESPN name clubs differently, so each Understat club is matched to ESPN's by
their games: a game on the same day (give or take one, for time zones) with the same score votes for
the pairing, and a club is linked to the ESPN club with the most votes (at least 5, and twice the
runner-up; a club with few games, like one just promoted, is linked through opponents already
linked). Then each game is linked by the two clubs and the date.
"""

import sqlite3
import sys
from collections import Counter, defaultdict

import pandas as pd
import requests

from db import DB_PATH, SCHEMA_PATH
from epl.import_understat import LAST_SEASON, fetch_season

LEAGUES = {"EPL": "eng.1", "La_liga": "esp.1", "Bundesliga": "ger.1", "Serie_A": "ita.1", "Ligue_1": "fra.1"}
FIRST_SEASON = 2014
MIN_VOTES = 5


def understat_games(league: str, seasons, fetch=fetch_season) -> pd.DataFrame:
    """Every played game: season, date, Understat club names, score and xG."""
    rows = []
    for season in seasons:
        try:
            matches = fetch(season, league)
        except (requests.RequestException, RuntimeError) as err:
            print(f"  {league} {season}: skipped ({err})")
            continue
        for m in matches:
            if not m.get("isResult"):
                continue
            rows.append({"season": season, "date": str(m["datetime"])[:10], "home": m["h"]["title"], "away": m["a"]["title"],
                         "home_score": int(m["goals"]["h"]), "away_score": int(m["goals"]["a"]),
                         "home_xg": float(m["xG"]["h"]), "away_xg": float(m["xG"]["a"]), "understat_id": str(m["id"])})
    return pd.DataFrame(rows)


def near(espn: pd.DataFrame, days: int = 1) -> dict:
    """{(date, home score, away score): [ESPN games]}, each game listed under its date and the dates next to it."""
    out = defaultdict(list)
    for g in espn.itertuples():
        day = pd.Timestamp(g.date)
        for k in range(-days, days + 1):
            out[((day + pd.Timedelta(days=k)).strftime("%Y-%m-%d"), int(g.home_score), int(g.away_score))].append(g)
    return out


def club_mapping(us: pd.DataFrame, espn: pd.DataFrame) -> dict:
    """{Understat club name: ESPN id}, from games on (about) the same day with the same score.
    A second pass links clubs with few games (just promoted) through opponents already linked."""
    index = near(espn)

    def decide(votes, least):
        out = {}
        for club, counter in votes.items():
            (best, n), *rest = counter.most_common(2) + [(None, 0)]
            if n >= least and n >= 2 * rest[0][1]:
                out[club] = best
        return out

    votes = defaultdict(Counter)
    for g in us.itertuples():
        found = index.get((g.date, g.home_score, g.away_score), [])
        if len(found) == 1:
            votes[g.home][found[0].home_id] += 1
            votes[g.away][found[0].away_id] += 1
    mapping = decide(votes, MIN_VOTES)
    votes = defaultdict(Counter)                     # second pass: the opponent is known
    for g in us.itertuples():
        found = index.get((g.date, g.home_score, g.away_score), [])
        if g.home not in mapping and g.away in mapping:
            found = [e for e in found if e.away_id == mapping[g.away]]
            if len(found) == 1:
                votes[g.home][found[0].home_id] += 1
        elif g.away not in mapping and g.home in mapping:
            found = [e for e in found if e.home_id == mapping[g.home]]
            if len(found) == 1:
                votes[g.away][found[0].away_id] += 1
    taken = set(mapping.values())
    mapping.update({club: espn_id for club, espn_id in decide(votes, 3).items() if espn_id not in taken})
    return mapping


def link(us: pd.DataFrame, espn: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Understat games with ESPN's event_id: the same two clubs within two days."""
    by_pair = defaultdict(list)
    for g in espn.itertuples():
        by_pair[(g.home_id, g.away_id)].append(g)
    rows = []
    for g in us.itertuples():
        h, a = mapping.get(g.home), mapping.get(g.away)
        candidates = [e for e in by_pair.get((h, a), []) if abs((pd.Timestamp(e.date) - pd.Timestamp(g.date)).days) <= 2]
        if len(candidates) == 1:
            rows.append({"event_id": candidates[0].event_id, "home_xg": g.home_xg, "away_xg": g.away_xg,
                         "understat_id": g.understat_id, "score_agrees": (int(candidates[0].home_score), int(candidates[0].away_score))
                         == (g.home_score, g.away_score)})
    return pd.DataFrame(rows, columns=["event_id", "home_xg", "away_xg", "understat_id", "score_agrees"])


def save(conn, linked: pd.DataFrame) -> int:
    conn.executemany("""INSERT INTO soccer_xg (event_id, home_xg, away_xg, understat_id) VALUES (?, ?, ?, ?)
                        ON CONFLICT (event_id) DO UPDATE SET home_xg = excluded.home_xg, away_xg = excluded.away_xg,
                                                             understat_id = excluded.understat_id""",
                     [(r.event_id, float(r.home_xg), float(r.away_xg), r.understat_id) for r in linked.itertuples()])
    conn.commit()
    return len(linked)


def main(argv=None, fetch=fetch_season) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    seasons = range(FIRST_SEASON, LAST_SEASON + 1)
    total, problems = 0, False
    for league, slug in LEAGUES.items():
        espn = pd.read_sql("""SELECT event_id, season, date, home_id, away_id, home_score, away_score FROM soccer_games
                              WHERE competition = ? AND status = 'final' AND home_score IS NOT NULL""", conn, params=(slug,))
        if espn.empty:
            print(f"{league}: no ESPN games for {slug} yet (run python -m ucl.import_espn first)")
            problems = True
            continue
        us = understat_games(league, seasons, fetch)
        if us.empty:
            problems = True
            continue
        mapping = club_mapping(us, espn)
        linked = link(us, espn, mapping)
        unmapped = sorted(set(us["home"]) - set(mapping))
        total += save(conn, linked)
        print(f"{league}: {len(linked):,} of {len(us):,} played games linked; {len(mapping)} clubs matched"
              + (f"; not matched: {', '.join(unmapped)}" if unmapped else "")
              + (f"; {int((~linked['score_agrees']).sum())} scores disagree" if len(linked) and (~linked["score_agrees"]).any() else ""))
    conn.close()
    print(f"Saved xG for {total:,} games.")
    if problems and total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
