"""Shared test helpers: a simulated league with known team strengths, and a database built from it.

Nothing here touches the internet or your real database in data/.
"""

import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]


def simulate_league(seed=0, n_teams=10, seasons=range(2013, 2027), current_season_games=30, home_boost=0.17):
    """Matches between teams with fixed, known strengths. Returns (games DataFrame, strengths dict)."""
    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    attack = dict(zip(teams, rng.normal(0, 0.3, n_teams)))
    defense = dict(zip(teams, rng.normal(0, 0.3, n_teams)))
    rows = []
    for season in seasons:
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        if season == max(seasons):
            fixtures = fixtures[:current_season_games]
        start = pd.Timestamp(f"{season}-08-16")
        for k, (h, a) in enumerate(fixtures):
            lam_h = np.exp(0.3 + home_boost + attack[h] - defense[a])
            lam_a = np.exp(0.3 + attack[a] - defense[h])
            rows.append({
                "id": len(rows) + 1,
                "season": season,
                "date": (start + pd.Timedelta(days=7 * (k // 5) + k % 2)).strftime("%Y-%m-%d"),
                "home": h, "away": a,
                "home_score": int(rng.poisson(lam_h)), "away_score": int(rng.poisson(lam_a)),
                "home_xg": lam_h * rng.uniform(0.8, 1.2) if season >= 2014 else np.nan,
                "away_xg": lam_a * rng.uniform(0.8, 1.2) if season >= 2014 else np.nan,
                "lam_h": lam_h, "lam_a": lam_a,
            })
    return pd.DataFrame(rows), {"attack": attack, "defense": defense, "home_boost": home_boost}


def build_database(path: Path, games: pd.DataFrame) -> None:
    """A database in the real schema, holding the simulated games, their xG and closing odds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript((ROOT / "schema.sql").read_text())
    conn.execute("INSERT INTO leagues (code, name, sport) VALUES ('EPL', 'Premier League', 'soccer')")
    teams = sorted(set(games["home"]) | set(games["away"]))
    ids = {t: conn.execute("INSERT INTO teams (league_id, name) VALUES (1, ?)", (t,)).lastrowid for t in teams}
    goals = np.arange(11)
    latest = games["season"].max()
    for g in games.itertuples():
        game_id = conn.execute(
            """INSERT INTO games (league_id, season, date, home_team_id, away_team_id, home_score, away_score, status)
               VALUES (1, ?, ?, ?, ?, ?, ?, 'final')""",
            (g.season, g.date, ids[g.home], ids[g.away], g.home_score, g.away_score)).lastrowid
        if not np.isnan(g.home_xg):
            conn.execute("INSERT INTO game_xg VALUES (?, 'understat', ?, ?, NULL)", (game_id, g.home_xg, g.away_xg))
        grid = np.outer(poisson.pmf(goals, g.lam_h), poisson.pmf(goals, g.lam_a))
        fair = (np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum())
        bookmaker = "Betfair Exchange" if g.season == latest else "Pinnacle"
        for outcome, p in zip(("home", "draw", "away"), fair):
            conn.execute("""INSERT INTO odds (game_id, bookmaker, market, outcome, price, line, is_closing)
                            VALUES (?, ?, '1x2', ?, ?, NULL, 1)""", (game_id, bookmaker, outcome, 1 / (p * 1.02)))
    conn.commit()
    conn.close()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project folder (schema and alias files, empty data/) set as the working directory."""
    for name in ("schema.sql", "epl/team_aliases.sql"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def league_db(project):
    """A temporary project with a simulated league already in data/sports.db."""
    games, _ = simulate_league()
    build_database(project / "data" / "sports.db", games)
    return project / "data" / "sports.db"


@pytest.fixture(autouse=True)
def never_the_real_database(tmp_path, monkeypatch):
    """Every website module reads its database from a fixed path in the project folder. Point them all
    at an empty temporary one, so no test ever reads your real data (tests that need data set their own)."""
    import importlib
    missing = tmp_path / "no-database" / "sports.db"
    for name in ("app", "nfl.site", "cfb.site", "cbb.site", "nba.site", "overview"):
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "DB_PATH", missing)
        module._cache.clear()
