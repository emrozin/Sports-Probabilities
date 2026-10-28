"""Premier League team news for match pages: who's injured, suspended or doubtful, from the latest
Fantasy Premier League availability snapshot (epl/fetch_availability.py, records/availability/).

Not used by the model yet. The feed only shows the current situation, so there's no history of
pre-match team news to test an adjustment on: the snapshots saved since October 2026 are that
history, and once a season of them has built up, an adjustment can be fitted on one part and
checked on the rest, the way the NFL's missing starters were.
"""

from pathlib import Path

import pandas as pd

SNAPSHOTS = Path("records/availability")
FPL_NAMES = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
             "Nott'm Forest": "Nott'm Forest", "Leeds United": "Leeds"}      # FPL's name -> ours
OUT = {"injured", "suspended", "unavailable", "not eligible"}


def latest(folder: Path = SNAPSHOTS) -> tuple[pd.DataFrame, str | None]:
    """The newest snapshot and its date ('YYYY-MM-DD'), or an empty frame."""
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        return pd.DataFrame(), None
    df = pd.read_csv(files[-1])
    df["team"] = df["team"].replace(FPL_NAMES)
    return df, files[-1].stem


def team_news(team: str, snapshot: pd.DataFrame) -> list[dict]:
    """The team's players who are out or doubtful, regulars first: name, position, status, chance
    of playing this round, the news, and whether he's a regular (started at least half the team's
    games so far)."""
    if snapshot.empty:
        return []
    squad = snapshot[snapshot["team"] == team]
    if squad.empty:
        return []
    games = max(int(squad["starts"].max()), 1)
    chance = pd.to_numeric(squad["chance_this_round"], errors="coerce")
    news = squad[(squad["status"] != "available") | (chance.notna() & (chance < 100))].copy()
    news["chance"] = chance.reindex(news.index)
    news = news.sort_values(["starts", "minutes"], ascending=False)
    return [{"name": r.name, "position": r.position, "status": r.status,
             "out": r.status in OUT or (pd.notna(r.chance) and r.chance == 0),
             "chance": None if pd.isna(r.chance) else int(r.chance),
             "news": r.news if isinstance(r.news, str) else "", "starts": int(r.starts),
             "regular": r.starts >= games / 2} for r in news.itertuples()]
