"""The betting rule: expected return, Kelly stakes, settling bets (two- and three-way markets),
the past-season simulation, and the Today and Betting pages."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import betting


def test_prices_convert():
    assert betting.american_to_decimal(150) == pytest.approx(2.5)
    assert betting.american_to_decimal(-200) == pytest.approx(1.5)
    assert betting.decimal_to_american(2.5) == "+150" and betting.decimal_to_american(1.5) == "-200"


def test_kelly_and_capped_fractional_stakes():
    # 60% at even money: edge 20%, full Kelly 20% of the bankroll
    assert betting.kelly(0.6, 2.0) == pytest.approx(0.2)
    assert betting.kelly(0.4, 2.0) == 0                       # no edge, no bet
    assert betting.stake_share(0.2, 0.25) == pytest.approx(0.03)   # a quarter would be 5%: capped at 3%
    assert betting.stake_share(0.08, 0.25) == pytest.approx(0.02)


def test_one_bet_per_game_only_with_enough_edge():
    chances = pd.DataFrame({"game_id": [1, 1, 2, 2], "outcome": ["home", "away"] * 2, "p": [0.60, 0.40, 0.52, 0.48]})
    offered = pd.DataFrame({"game_id": [1, 1, 2, 2], "outcome": ["home", "away"] * 2, "price": [1.91, 1.91, 1.91, 1.91]})
    picks = betting.candidates(chances, offered)
    assert list(picks["game_id"]) == [1] and picks.at[0, "outcome"] == "home"
    assert picks.at[0, "edge"] == pytest.approx(0.6 * 1.91 - 1) and picks.at[0, "market"] == pytest.approx(0.5)


def test_settling_two_and_three_way_bets():
    games = pd.DataFrame({"game_id": [1, 2, 3], "status": ["final", "final", "scheduled"],
                          "home_score": [2, 1, None], "away_score": [1, 1, None]})
    bets = pd.DataFrame({"game_id": [1, 2, 3], "outcome": ["home", "away", "home"], "price": [2.0, 3.0, 2.0]})
    two = betting.settle(bets, games).set_index("game_id")
    assert two.at[1, "result"] == "won" and two.at[1, "profit"] == pytest.approx(1.0)
    assert two.at[2, "result"] == "push" and two.at[2, "profit"] == 0     # an NFL tie refunds
    assert pd.isna(two.at[3, "result"]) and np.isnan(two.at[3, "profit"])
    three = betting.settle(bets, games, three_way=True).set_index("game_id")
    assert three.at[2, "result"] == "lost" and three.at[2, "profit"] == -1  # a soccer draw beats the away bet


def test_bankroll_simulation_and_luck_range():
    settled = pd.DataFrame({"date": ["2025-01-01", "2025-01-01", "2025-01-02"], "kelly": [0.2, 0.2, 0.2],
                            "profit": [1.0, -1.0, -1.0]})
    sim = betting.simulate(settled, fraction=0.25, cap=0.03, bankroll=1000)
    assert sim["final"] == pytest.approx(1000 * 0.97)          # day 1 evens out; day 2 loses 3%
    assert sim["drawdown"] == pytest.approx(0.03)
    low, high = betting.roi_range(np.array([1.0, -1.0] * 50))
    assert low < 0 < high


def test_pages_work_with_no_data(project):
    import app
    client = TestClient(app.app)
    for path in ("/today/", "/betting/", "/betting/?bankroll=250&fraction=0.5"):
        response = client.get(path)
        assert response.status_code == 200, path
    assert "Read this first" in client.get("/betting/").text
    assert "No games today" in client.get("/today/").text


# ---------------------------------------------------------------- prices, closing lines, experiments

import sqlite3
from pathlib import Path

import betting_experiments as bx
import espn

ROOT = Path(__file__).resolve().parent.parent


def tiny_db(tmp_path) -> sqlite3.Connection:
    """A database with the real schema and two finished, one upcoming, Premier League games."""
    conn = sqlite3.connect(tmp_path / "t.db")
    conn.executescript((ROOT / "schema.sql").read_text())
    conn.execute("INSERT INTO leagues (id, code, name, sport) VALUES (1, 'EPL', 'Premier League', 'soccer')")
    conn.executemany("INSERT INTO teams (id, league_id, name) VALUES (?, 1, ?)", [(1, "A"), (2, "B")])
    conn.executemany("""INSERT INTO games (id, league_id, season, date, home_team_id, away_team_id, home_score, away_score, status)
                        VALUES (?, 1, 2025, ?, 1, 2, ?, ?, ?)""",
                     [(1, "2025-09-01", 2, 0, "final"), (2, "2025-09-08", 1, 1, "final"), (3, "2025-09-15", None, None, "scheduled")])
    return conn


def test_premier_league_bets_at_the_best_price(tmp_path):
    conn = tiny_db(tmp_path)
    rows = [(1, "Market average", o, p) for o, p in (("home", 2.0), ("draw", 3.4), ("away", 3.8))]
    rows += [(1, "Market max", o, p) for o, p in (("home", 2.15), ("draw", 3.6), ("away", 4.1))]
    rows += [(2, "Market average", o, p) for o, p in (("home", 1.5), ("draw", 4.0), ("away", 6.5))]   # no max: the average
    conn.executemany("""INSERT INTO odds (game_id, bookmaker, market, outcome, price, line, is_closing)
                        VALUES (?, ?, '1x2', ?, ?, NULL, 0)""", rows)
    got = betting.prices(conn, "EPL").set_index(["game_id", "outcome"])["price"]
    assert got[(1, "home")] == pytest.approx(2.15) and got[(1, "away")] == pytest.approx(4.1)
    assert got[(2, "home")] == pytest.approx(1.5) and len(got) == 6


def test_finished_games_keep_their_last_espn_line_as_the_close(tmp_path):
    conn = tiny_db(tmp_path)
    conn.executemany("""INSERT INTO espn_lines (league, source_game_id, game_id, date, home_ml, away_ml, final)
                        VALUES ('NBA', ?, ?, ?, 1.5, 2.6, 0)""", [("a", 1, "2025-09-01"), ("c", 3, "2025-09-15")])
    assert set(betting.espn_prices(conn, "NBA", closing=True)["game_id"]) == {1}   # finished, though never marked
    assert espn.mark_closed(conn) == 1
    assert conn.execute("SELECT game_id FROM espn_lines WHERE final = 1").fetchall() == [(1,)]


def test_blend_and_the_experiment_rule():
    df = pd.DataFrame({"game_id": [1, 1, 2, 2], "outcome": ["home", "away"] * 2, "p": [0.70, 0.30, 0.55, 0.45],
                       "q": [0.50, 0.50, 0.50, 0.50], "best": [1.95] * 4, "avg": [1.91] * 4,
                       "won": [1, 0, 0, 1], "push": [0] * 4, "date": ["2025-01-01"] * 4})
    df["gap"] = (df["p"] - df["q"]).abs().groupby(df["game_id"]).transform("max")
    assert bx.blend(df, 1.0).to_numpy() == pytest.approx(df["p"].to_numpy())     # all model
    assert bx.blend(df, 0.0).to_numpy() == pytest.approx(df["q"].to_numpy())     # all market
    half = bx.blend(df, 0.5)
    assert half.groupby(df["game_id"]).sum().to_numpy() == pytest.approx([1, 1]) and 0.5 < half[0] < 0.7
    both = bx.bets(df)                                    # edges 0.70*1.95-1 and 0.55*1.95-1: both bet
    assert set(both["game_id"]) == {1, 2} and both.set_index("game_id").at[1, "profit"] == pytest.approx(0.95)
    capped = bx.bets(df, cap=0.10)                        # game 1 is 20 points off the market: skipped
    assert list(capped["game_id"]) == [2] and capped.iloc[0]["profit"] == -1
    assert bx.score(capped)["bets"] == 1


def test_betting_page_shows_the_experiments(project):
    import app
    import overview
    (project / "records").mkdir(exist_ok=True)
    pd.DataFrame([{"league": "EPL", "sport": "Premier League", "rule": "Bet at the best price", "seasons": "2021–2025",
                   "bets": 1747, "won": 0.29, "roi": -0.034, "roi_low": -0.125, "roi_high": 0.055, "kelly_final": 102.0,
                   "kept": True, "note": "Now used."}]).to_csv(project / "records" / "betting_experiments.csv", index=False)
    text = TestClient(app.app).get("/betting/").text
    assert "What we tried" in text and "Bet at the best price" in text and "−3.4%" in text
    assert overview.experiment_rows()[0]["kept"]


def test_calibration_groups_games_from_the_favorites_side():
    from charts import calibration_chart
    p = np.array([0.65] * 40 + [0.30] * 40)            # the second group: the away team is the 70% favorite
    won = np.array([1] * 26 + [0] * 14 + [0] * 28 + [1] * 12, float)
    chart = calibration_chart((p, won), None)
    groups = {g["range"]: g for g in chart["live"]}
    assert groups["60–70%"]["actual"] == pytest.approx(65.0) and groups["60–70%"]["n"] == 40
    assert groups["70–80%"]["predicted"] == pytest.approx(70.0) and groups["70–80%"]["actual"] == pytest.approx(70.0)
    assert calibration_chart((np.array([0.6]), np.array([1.0])), None) is None       # too few to show


def test_spread_and_total_chances_and_settling():
    import betting_lines as bl
    assert bl.halves(-0.75) == [-1.0, -0.5] and bl.halves(-0.5) == [-0.5] and bl.halves(-1.0) == [-1.0]
    win, push, lose = bl.normal_outcome(3.0, 13.0, -3.0)              # expected margin 3, home -3: a whole-number line
    assert push > 0.02 and win == pytest.approx(lose, abs=1e-9) and win + push + lose == pytest.approx(1)
    win, push, lose = bl.normal_outcome(0.0, 13.0, 2.5)
    assert push == 0 and win > 0.5
    assert bl.settle_line(1, -0.75, 2.0) == pytest.approx(0.5)        # won by one at -0.75: half won, half refunded
    assert bl.settle_line(2, -0.75, 2.0) == pytest.approx(1.0)
    assert bl.settle_line(0, -0.25, 2.0) == pytest.approx(-0.5)       # a draw at -0.25: half refunded, half lost
    assert bl.settle_line(-3, 3.0, 1.9) == 0.0                        # a push
    import numpy as np
    from epl.model import scoreline_grid
    grid = scoreline_grid(1.5, 1.0)
    grid = grid / grid.sum()
    h, a = np.indices(grid.shape)
    over = bl.grid_outcome(grid, h + a, -2.5)
    under = bl.grid_outcome(grid, -(h + a), 2.5)
    assert over[0] == pytest.approx(under[2]) and over[1] == 0
    assert bl.expected_return([over], 1 / over[0]) == pytest.approx(0, abs=1e-9)   # a fair price returns nothing
