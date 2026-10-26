"""The redesign's logic: team colors, the model-vs-market tag and the team page charts."""

import colors
from charts import chance_bars, line_chart
from web import edge


def test_clashing_team_colors_switch_the_away_team_to_its_second_color():
    home, away = colors.pair(colors.EPL, "Liverpool", "Man United")       # both red
    assert home == colors.EPL["Liverpool"][0] and away == colors.EPL["Man United"][1]
    assert colors.pair(colors.NFL, "DEN", "SEA") == (colors.NFL["DEN"][0], colors.NFL["SEA"][0])   # orange v navy
    for table in (colors.NFL, colors.EPL):
        for h in table:
            for a in table:
                if h != a:
                    assert colors.difference(*colors.pair(table, h, a)) >= colors.MIN_DIFFERENCE


def test_dark_team_colors_are_lifted_for_the_dark_page_and_text_stays_readable():
    navy = colors.NFL["SEA"][0]
    lifted = colors.for_dark(navy)
    assert colors._lab(lifted)[0] >= colors.DARK_MIN_LIGHTNESS - 1
    assert colors.for_dark(colors.NFL["DEN"][0]) == colors.NFL["DEN"][0]   # already bright: unchanged
    assert colors.ink("#FFCD00") != "#ffffff" and colors.ink("#002244") == "#ffffff"
    assert "--hd:" in colors.style(colors.NFL, "DEN", "SEA") and "--al:" in colors.style(colors.NFL, "DEN", "SEA")


def test_model_market_tag_needs_a_five_point_gap():
    sides = {"home": "DEN", "away": "SEA"}
    assert edge({"home": 47, "away": 53}, {"home": 40, "away": 60}, sides) == {"team": "DEN", "points": 7}
    assert edge({"home": 42, "away": 58}, {"home": 40, "away": 60}, sides) is None
    assert edge({"home": 42, "away": 58}, None, sides) is None


def test_line_chart_places_points_around_zero():
    chart = line_chart([{"label": "Start", "value": 0.0, "tip": ""}, {"label": "1", "value": 3.0, "tip": ""},
                        {"label": "2", "value": -5.0, "tip": ""}])
    assert chart["limit"] == 6 and [p["x"] for p in chart["points"]] == [0, 50, 100]
    assert chart["points"][0]["y"] == 50 and chart["points"][1]["y"] < 50 < chart["points"][2]["y"]
    assert line_chart([{"label": "Start", "value": 1.0, "tip": ""}]) is None
    assert chance_bars([{"label": "1", "chance": 0.634, "outcome": "W", "tip": ""}])[0]["height"] == 63.4


def test_record_chart_keeps_a_running_typical_probability():
    import pytest
    from charts import record_chart
    weeks = [{"label": "Week 1", "model": [0.8, 0.5], "benchmark": [0.6, 0.6]},
             {"label": "Week 2", "model": [], "benchmark": []},                 # skipped
             {"label": "Week 3", "model": [0.25], "benchmark": [0.5]}]
    chart = record_chart(weeks, "the betting market")
    assert [p["label"] for p in chart["points"]] == ["Week 1", "Week 3"]
    assert chart["points"][0]["model"] == pytest.approx(100 * (0.8 * 0.5) ** 0.5)
    assert chart["last"]["model"] == pytest.approx(100 * (0.8 * 0.5 * 0.25) ** (1 / 3))
    assert chart["last"]["games"] == 3 and chart["model_path"].startswith("M 0.0 ")
    assert all(0 <= t["y"] <= 100 for t in chart["ticks"])
    assert record_chart(weeks[:1], "x") is None                                # one week isn't a line
