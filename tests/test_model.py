"""The model: valid probabilities, sensible ratings, and a backtest that never sees the future."""

import numpy as np
import pandas as pd
import pytest

from epl import model
from conftest import simulate_league
from epl.model import add_scores, backtest, fit_poisson, predict


@pytest.fixture(scope="module")
def fitted():
    games, truth = simulate_league(seed=1)
    return fit_poisson(games, games["date"].max()), truth


def test_probabilities_are_valid_and_sum_to_one(fitted):
    fit, _ = fitted
    teams = list(fit["ratings"].index)
    for home, away in [(teams[0], teams[1]), (teams[2], teams[5]), (teams[9], teams[3])]:
        p = predict(fit, home, away)
        assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-6)
        assert all(0 < p[o] < 1 for o in ("home", "draw", "away"))


def test_stronger_team_is_favoured(fitted):
    fit, _ = fitted
    strength = np.log(fit["ratings"]["attack"]) + np.log(fit["ratings"]["defense"])
    best, worst = strength.idxmax(), strength.idxmin()
    assert predict(fit, best, worst)["home"] > 0.5
    assert predict(fit, worst, best)["away"] > predict(fit, worst, best)["home"]


def test_fit_recovers_known_team_strengths():
    games, truth = simulate_league(seed=2, n_teams=16, seasons=range(2010, 2020))
    fit = fit_poisson(games, games["date"].max(), half_life_days=100_000, shrinkage=0.0, xg_weight=0.0)
    ratings = fit["ratings"]
    true_attack = pd.Series(truth["attack"])[ratings.index]
    true_defense = pd.Series(truth["defense"])[ratings.index]
    assert np.corrcoef(np.log(ratings["attack"]), true_attack)[0, 1] > 0.9
    assert np.corrcoef(np.log(ratings["defense"]), true_defense)[0, 1] > 0.9
    assert np.log(fit["home_adv"]) == pytest.approx(truth["home_boost"], abs=0.08)


def test_ratings_are_centred_on_an_average_team(fitted):
    fit, _ = fitted
    assert np.exp(np.log(fit["ratings"]["attack"]).mean()) == pytest.approx(1.0, abs=1e-6)
    assert np.exp(np.log(fit["ratings"]["defense"]).mean()) == pytest.approx(1.0, abs=1e-6)


def test_missing_xg_falls_back_to_goals():
    """Regression: with no xG at all, the columns arrive as objects rather than numbers."""
    games, _ = simulate_league(seed=3, seasons=range(2010, 2012))
    no_xg = games.assign(home_xg=None, away_xg=None).astype({"home_xg": object, "away_xg": object})
    as_of = games["date"].max()
    blended = fit_poisson(no_xg, as_of, xg_weight=0.5)
    goals_only = fit_poisson(games.assign(home_xg=np.nan, away_xg=np.nan), as_of, xg_weight=0.0)
    pd.testing.assert_frame_equal(blended["ratings"], goals_only["ratings"], atol=1e-6)


def test_backtest_never_trains_on_matches_from_the_week_it_predicts(monkeypatch):
    games, _ = simulate_league(seed=4, seasons=range(2014, 2018))
    real_fit = model.fit_poisson
    calls = []

    def recording_fit(train, as_of, **settings):
        calls.append((train["date"].max(), as_of))
        return real_fit(train, as_of, **settings)

    monkeypatch.setattr(model, "fit_poisson", recording_fit)
    results = backtest(games, start_season=2016)

    assert calls and len(results) > 0
    for latest_training_match, cutoff in calls:
        assert latest_training_match < cutoff

    # ...and every predicted match is played on or after the cutoff it was predicted from.
    dates = games.set_index("id")["date"]
    week_start = pd.to_datetime(dates).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    assert (dates[results["id"]] >= week_start[results["id"]]).all()
    assert set(week_start[results["id"]]) <= {cutoff for _, cutoff in calls}


def test_scores_for_perfect_and_uninformed_forecasts():
    df = pd.DataFrame({"result": ["home", "draw", "away"],
                       "a_home": [1.0, 0.0, 0.0], "a_draw": [0.0, 1.0, 0.0], "a_away": [0.0, 0.0, 1.0],
                       "b_home": [1 / 3] * 3, "b_draw": [1 / 3] * 3, "b_away": [1 / 3] * 3})
    add_scores(df, "a")
    add_scores(df, "b")
    assert df["a_logloss"].max() == pytest.approx(0.0)
    assert df["a_brier"].max() == pytest.approx(0.0)
    assert df["b_logloss"].mean() == pytest.approx(np.log(3))


def test_goal_markets_are_consistent():
    from scipy.stats import poisson
    from epl.model import goal_markets, scoreline_grid

    for exp_home, exp_away in [(1.4, 1.2), (2.8, 0.6), (0.7, 0.9)]:
        grid = scoreline_grid(exp_home, exp_away)
        m = goal_markets(grid)
        assert m["home"] + m["draw"] + m["away"] == pytest.approx(1.0)
        for line in (1.5, 2.5, 3.5):
            assert m[f"over_{line}"] + m[f"under_{line}"] == pytest.approx(1.0)
        assert m["over_1.5"] > m["over_2.5"] > m["over_3.5"]
        # With no draw adjustment, total goals follow a Poisson distribution with the combined average.
        assert m["over_2.5"] == pytest.approx(1 - poisson.cdf(2, exp_home + exp_away), abs=1e-5)
        scores = [p for _, _, p in m["top_scores"]]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(grid.max())


def test_predict_includes_goal_markets(fitted):
    fit, _ = fitted
    teams = list(fit["ratings"].index)
    p = predict(fit, teams[0], teams[1])
    assert {"over_2.5", "under_2.5", "btts", "top_scores", "exp_home", "exp_away"} <= set(p)


def test_backtest_skips_weeks_with_no_earlier_matches():
    """Regression: a league whose data starts after the backtest's first season used to crash."""
    games, _ = simulate_league(seed=6, seasons=range(2022, 2024))
    results = backtest(games, start_season=2016)
    assert len(results) > 0
    assert results["id"].isin(games.loc[games["season"] == 2022, "id"]).any()  # 2022 predicted once data exists


def test_value_weight_zero_is_todays_model():
    games, _ = simulate_league(seed=7, seasons=range(2020, 2022))
    values = {t: 100e6 * (i + 1) for i, t in enumerate(sorted(set(games["home"])))}
    as_of = games["date"].max()
    plain = fit_poisson(games, as_of)
    zero = fit_poisson(games, as_of, squad_values=values, value_weight=0)
    pd.testing.assert_frame_equal(plain["ratings"], zero["ratings"])


def test_squad_value_pulls_teams_with_little_data_toward_it():
    games, _ = simulate_league(seed=8, seasons=range(2020, 2022))
    as_of = games["date"].max()
    newcomer = pd.DataFrame([{"id": 99_999, "season": 2021, "date": as_of, "home": "Newcomer", "away": "Team A",
                              "home_score": 1, "away_score": 1, "home_xg": 1.0, "away_xg": 1.0,
                              "lam_h": 1.0, "lam_a": 1.0}])
    train = pd.concat([games, newcomer], ignore_index=True)
    others = {t: 100e6 for t in set(games["home"])}

    def strength(newcomer_value):
        fit = fit_poisson(train, as_of, shrinkage=4, value_weight=0.5,
                          squad_values={**others, "Newcomer": newcomer_value})
        r = fit["ratings"].loc["Newcomer"]
        return np.log(r["attack"]) + np.log(r["defense"])

    # One draw tells the model almost nothing; the squad value decides where it starts.
    assert strength(800e6) > 0.3 > -0.3 > strength(12e6)


def test_backtest_uses_each_weeks_squad_values():
    games, _ = simulate_league(seed=9, seasons=range(2014, 2017))
    teams = sorted(set(games["home"]))
    weeks = pd.to_datetime(games["date"]).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d").unique()
    values = {w: {t: 50e6 * (i + 1) for i, t in enumerate(teams)} for w in weeks}
    plain = backtest(games, start_season=2016)
    with_values = backtest(games, start_season=2016, squad_values_by_week=values, value_weight=0.3, shrinkage=2)
    assert len(plain) == len(with_values)
    assert not np.allclose(plain["model_home"], with_values["model_home"])


def test_fade_halves_the_squad_value_adjustment_and_returns_to_todays_model():
    assert model.faded_settings(4, 2, rounds_played=0, fade_rounds=6) == (4, 2)
    weight, shrink = model.faded_settings(4, 2, rounds_played=6, fade_rounds=6)
    assert weight == pytest.approx(2)
    assert shrink == pytest.approx(model.SHRINKAGE + (2 - model.SHRINKAGE) / 2)
    weight, shrink = model.faded_settings(4, 2, rounds_played=200, fade_rounds=6)
    assert weight == pytest.approx(0, abs=1e-6) and shrink == pytest.approx(model.SHRINKAGE, abs=1e-6)
    assert model.faded_settings(4, 2, rounds_played=30, fade_rounds=None) == (4, 2)   # no fade


def test_faded_squad_value_matters_early_in_a_season_but_not_late():
    games, _ = simulate_league(seed=10, seasons=range(2014, 2018))   # 2016 is a complete season
    teams = sorted(set(games["home"]))
    weeks = pd.to_datetime(games["date"]).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d").unique()
    values = {w: {t: 50e6 * (i + 1) for i, t in enumerate(teams)} for w in weeks}
    plain = backtest(games, start_season=2016).set_index("id")
    faded = backtest(games, start_season=2016, squad_values_by_week=values, value_weight=0.3, shrinkage=2,
                     value_fade_rounds=2).set_index("id")
    season_2016 = games[games["season"] == 2016].sort_values("date")["id"]
    early, late = season_2016.iloc[:20], season_2016.iloc[-20:]
    early_diff = (plain.loc[early, "model_home"] - faded.loc[early, "model_home"]).abs().mean()
    late_diff = (plain.loc[late, "model_home"] - faded.loc[late, "model_home"]).abs().mean()
    assert early_diff > 0.01
    assert late_diff < 1e-3


def test_championship_matches_rate_promoted_teams():
    rng = np.random.default_rng(5)
    top = {f"P{i}": 0.3 - 0.06 * i for i in range(8)}             # a top division
    second = {f"C{i}": -0.1 - 0.06 * i for i in range(8)}         # a weaker second division
    second["Riser"] = 0.25                                          # a strong second-division team
    rows, k = [], 0
    for division, teams in (("EPL", top), ("CHAMP", second)):
        for rep in range(4):
            for h in teams:
                for a in teams:
                    if h != a:
                        k += 1
                        hg, ag = rng.poisson(np.exp(0.25 + teams[h] - teams[a])), rng.poisson(np.exp(teams[a] - teams[h]))
                        rows.append({"id": k, "season": 2024, "date": f"2025-0{1 + rep}-{1 + k % 27:02d}", "home": h, "away": a,
                                     "home_score": hg, "away_score": ag, "home_xg": np.nan, "away_xg": np.nan, "division": division})
    games = pd.DataFrame(rows)
    plain = fit_poisson(games, "2025-06-01", champ_weight=0)
    assert "Riser" not in plain["ratings"].index                      # left out: Premier League only
    fit = fit_poisson(games, "2025-06-01", champ_weight=0.5)
    r = fit["ratings"]
    strength = np.log(r["attack"]) + np.log(r["defense"])
    assert strength["Riser"] > strength["C7"]                          # rated from its Championship games
    assert set(fit["championship"]) == {"scoring", "attack", "defense"}   # (its level is set by teams that move divisions)
    p = predict(fit, "P0", "Riser")
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-6)
