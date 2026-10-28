"""Shared pieces of the college football and basketball game pages (cfb/site.py, cbb/site.py):
each team's form going into the game, every earlier meeting, and why the model picks a team."""

import pandas as pd

from web import day_label, why

H2H_SHOWN = 10


def form(games: pd.DataFrame, team: str, before: str, season: int, url, n: int,
         home: str = "home", away: str = "away") -> dict:
    """A team's last n finished games before a date (oldest first), its record in them and this
    season, and days since its last game. home/away: the columns with the teams' names; url(row)
    gives a game's page."""
    own = games[((games[home] == team) | (games[away] == team)) & (games["status"] == "final")
                & (games["date"] < before)].sort_values(["date", "id"])
    if own.empty:
        return {"games": [], "last": None}
    rows = []
    for g in own.tail(n).itertuples():
        at_home = getattr(g, home) == team
        ts, os_ = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        rows.append({"outcome": "W" if ts > os_ else "L" if ts < os_ else "T", "score": f"{int(ts)}–{int(os_)}",
                     "venue": "vs" if at_home or bool(g.neutral_site) else "at",
                     "opponent": getattr(g, away) if at_home else getattr(g, home), "url": url(g), "date": day_label(g.date)})
    this = own[own["season"] == season]
    won = int(((this[home] == team) & (this["home_score"] > this["away_score"])).sum()
              + ((this[away] == team) & (this["away_score"] > this["home_score"])).sum())
    last = own.iloc[-1]
    return {"games": rows, "last": f"{sum(r['outcome'] == 'W' for r in rows)}–{sum(r['outcome'] == 'L' for r in rows)}",
            "season": f"{won}–{len(this) - won}" if len(this) else None,
            "rest": (pd.Timestamp(before) - pd.Timestamp(last["date"])).days, "opener": int(last["season"]) != season}


def head_to_head(games: pd.DataFrame, a: str, b: str, before: str, url, home: str = "home", away: str = "away",
                 stage=lambda g: "") -> dict:
    """Every earlier meeting of two teams in the data: the record and the latest H2H_SHOWN."""
    met = games[(((games[home] == a) & (games[away] == b)) | ((games[home] == b) & (games[away] == a)))
                & (games["status"] == "final") & (games["date"] < before)].sort_values(["date", "id"], ascending=False)
    wins = {a: 0, b: 0}
    meetings = []
    for g in met.itertuples():
        h, w = getattr(g, home), getattr(g, away)
        winner = h if g.home_score > g.away_score else w if g.away_score > g.home_score else None
        if winner:
            wins[winner] += 1
        if len(meetings) < H2H_SHOWN:
            meetings.append({"date": day_label(g.date, with_year=True), "home": h, "away": w, "neutral": bool(g.neutral_site),
                             "score": f"{int(g.away_score)}–{int(g.home_score)}", "winner": winner, "url": url(g),
                             "stage": stage(g)})
    return {"meetings": meetings, "count": len(met), "wins_a": wins[a], "wins_b": wins[b], "a": a, "b": b,
            "first": met["date"].min()[:4] if len(met) else None}


def points_why(model: dict, rated_home: str, rated_away: str, home: str, away: str, neutral: bool, home_win: float,
               market_home=None, qb: dict | None = None, note: str | None = None) -> dict | None:
    """'Why the model picks ...' for the offense/defense points model: offense, defense, home edge,
    and (college football) quarterbacks. rated_*: the names the model rates; home/away: shown.
    qb: {"value": points of home margin from the two quarterbacks, "note": ...}."""
    r = model["ratings"]
    get = lambda team, col: float(r.at[team, col]) if team in r.index else 0.0
    rows = [{"label": "Offense", "value": get(rated_home, "offense") - get(rated_away, "offense"),
             "note": "Points scored, against the defenses faced"},
            {"label": "Defense", "value": get(rated_home, "defense") - get(rated_away, "defense"),
             "note": "Points allowed, against the offenses faced"}]
    if qb is not None:
        rows.append({"label": "Quarterbacks", "value": qb["value"], "note": qb.get("note", "")})
    rows.append({"label": "Home field" if qb is not None else "Home court", "value": 0.0 if neutral else float(model["home_adv"]),
                 "note": "Neutral site" if neutral else ""})
    return why(rows, home, away, home_win, market_home, [note] if note else None)
