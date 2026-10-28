"""The college football pages of the website, under /cfb/ (included by app.py).

Games involving a top-division (FBS) school only. Each game shows the model's chance next to the
betting market's (closing consensus, for finished games) and CollegeFootballData's Elo rating,
a second benchmark that's available for upcoming games too.
"""

import sqlite3
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from cfb.model import MARGIN_SD, current_fit, elo_probability, load_games, load_market, team_strength, window
from cfb.predict_upcoming import MODEL_NAME, MODEL_VERSION
from cfb.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
import betting
import college
import espn
from web import BASE, ROOT, day_label, edge, ordinal, signed_number, slugify, templates, typical_probability

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter(prefix="/cfb")
RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/cfb_live_predictions.csv"
POWER = ["SEC", "Big Ten", "Big 12", "ACC"]   # listed first in the conference filter

_cache: dict = {}
GAME_COLUMNS = ["id", "season", "date", "kickoff", "status", "home", "away", "home_score", "away_score", "neutral_site",
                "week", "season_type", "source_game_id", "home_division", "away_division", "home_conference",
                "away_conference", "home_elo", "away_elo", "spread_line", "total_line", "notes", "venue",
                "conference_game", "home_qb", "home_qb_name", "away_qb", "away_qb_name", "home_qb_expected",
                "away_qb_expected", "home_qb_starts", "away_qb_starts", "home_qb_expected_starts",
                "away_qb_expected_starts", "elo_home"]


def connect():
    return closing(sqlite3.connect(DB_PATH))


def game_url(source_id) -> str | None:
    return f"{BASE}/cfb/game/{source_id}/" if isinstance(source_id, str) and source_id else None


def team_url(team: str, season: int | None = None) -> str:
    return f"{BASE}/cfb/team/{slugify(team)}/" + (f"{season}/" if season is not None else "")


templates.env.globals["cfb_team_url"] = team_url


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request, f"cfb/{template}", {"sport": "cfb", **context}, status_code=status_code)


def load_predictions(conn, name: str, version: str | None) -> pd.DataFrame:
    """One row per game. version None: every version, keeping each game's most recently saved one
    (live predictions recorded by an earlier model version still count)."""
    df = pd.read_sql("""SELECT p.game_id AS id, p.model_version, MAX(p.created_at) AS created_at,
                               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
                               MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
                        FROM predictions p LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                             AND pg.model_name = p.model_name AND pg.model_version = p.model_version
                        WHERE p.model_name = ? AND (? IS NULL OR p.model_version = ?)
                        GROUP BY p.game_id, p.model_version""", conn, params=(name, version, version))
    df = df.sort_values(["created_at", "model_version"]).drop_duplicates("id", keep="last")
    return df[["id", "model_home", "exp_home", "exp_away"]]


def data() -> dict:
    """Everything the pages read, loaded once and reused until the database file changes."""
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if _cache.get("stamp") == stamp and "games" in _cache:
        return _cache
    _cache.clear()
    _cache["stamp"] = stamp
    has = False
    if stamp is not None:
        with connect() as conn:
            has = conn.execute("SELECT name FROM sqlite_master WHERE name = 'cfb_game_info'").fetchone() is not None
            if has:
                games = load_games(conn)
                live = load_predictions(conn, MODEL_NAME, None).assign(source="Saved before kickoff")
                back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).assign(source="Backtest")
                market = load_market(conn)
                # this season's lines from ESPN (espn.py), for games the consensus file doesn't have
                lines = espn.market_lines(conn, "CFB")
                market = pd.concat([market, lines[~lines["id"].isin(market["id"])]], ignore_index=True)
    if not has or games.empty:
        empty = pd.DataFrame(columns=GAME_COLUMNS)
        _cache.update(games=empty, predictions=pd.DataFrame(columns=["id", "model_home", "exp_home", "exp_away", "source"]),
                      market=pd.DataFrame(columns=["id", "market_home"]), live=pd.DataFrame())
        return _cache
    games = games[(games["home_division"] == "fbs") | (games["away_division"] == "fbs")].copy()
    games["elo_home"] = elo_probability(games["home_elo"], games["away_elo"], games["neutral_site"])
    predictions = pd.concat([live, back[~back["id"].isin(live["id"])]], ignore_index=True)
    _cache.update(games=games, predictions=predictions, market=market, live=live)
    return _cache


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    d = data()
    return games.merge(d["predictions"], on="id", how="left").merge(d["market"], on="id", how="left")


def two_way(p: float) -> dict:
    home = int(round(p * 100))
    return {"home": home, "away": 100 - home}


def line_text(home_margin, home: str, away: str, decimals: int = 1) -> str | None:
    if home_margin is None or pd.isna(home_margin):
        return None
    if round(home_margin, decimals) == 0:
        return "Even"
    return f"{home if home_margin > 0 else away} by {abs(home_margin):.{decimals}f}"


def kickoff_label(kickoff) -> str | None:
    if not isinstance(kickoff, str) or not kickoff:
        return None
    t = pd.Timestamp(f"2000-01-01 {kickoff}")
    return f"{t:%-I:%M %p} ET"


def week_label(season_type: str, week: int) -> str:
    return "Bowls and playoff" if season_type == "postseason" else f"Week {week}"


def game_card(g, current_team: str | None = None) -> dict:
    finished = g.status == "final"
    margin = (g.exp_home - g.exp_away) if pd.notna(getattr(g, "exp_home", np.nan)) else None
    card = {"home": g.home, "away": g.away, "finished": finished, "neutral": bool(g.neutral_site),
            "date": day_label(g.date, with_year=True), "kickoff": kickoff_label(g.kickoff),
            "week": week_label(g.season_type, g.week), "notes": g.notes if isinstance(g.notes, str) else None,
            "home_conf": g.home_conference, "away_conf": g.away_conference,
            "model": two_way(g.model_home) if pd.notna(getattr(g, "model_home", np.nan)) else None,
            "market": two_way(g.market_home) if pd.notna(getattr(g, "market_home", np.nan)) else None,
            "elo": two_way(g.elo_home) if pd.notna(g.elo_home) else None,
            "source": getattr(g, "source", None) if isinstance(getattr(g, "source", None), str) else None,
            "model_line": line_text(margin, g.home, g.away),
            "market_line": line_text(-spread if pd.notna(spread := (g.spread_line if pd.notna(g.spread_line)
                                                                   else getattr(g, "market_spread", np.nan))) else None, g.home, g.away),
            "model_total": f"{g.exp_home + g.exp_away:.1f}" if margin is not None else None,
            "url": game_url(getattr(g, "source_game_id", None)),
            "market_total": f"{total:g}" if pd.notna(total := (g.total_line if pd.notna(g.total_line)
                                                               else getattr(g, "market_total", np.nan))) else None}
    card["edge"] = edge(card["model"], card["market"], {"home": g.home, "away": g.away})
    card["when"] = ", ".join(x for x in ([card["date"]] if current_team else []) + [card["kickoff"]]
                             + (["neutral site"] if card["neutral"] else []) + [card["notes"]] if x)
    if finished:
        hs, as_ = int(g.home_score), int(g.away_score)
        card.update(home_score=hs, away_score=as_, result="home" if hs > as_ else "away")
        if card["model"]:
            card["model_actual"] = card["model"][card["result"]]
            card["market_actual"] = card["market"][card["result"]] if card["market"] else None
        if current_team:
            own, other = (hs, as_) if g.home == current_team else (as_, hs)
            card["outcome"] = "W" if own > other else "L"
    return card


def conferences(games: pd.DataFrame) -> list[str]:
    confs = set(games.loc[games["home_division"] == "fbs", "home_conference"].dropna()) | \
            set(games.loc[games["away_division"] == "fbs", "away_conference"].dropna())
    return [c for c in POWER if c in confs] + sorted(c for c in confs if c not in POWER)


def filter_conference(games: pd.DataFrame, conference: str | None) -> pd.DataFrame:
    if not conference:
        return games
    return games[(games["home_conference"] == conference) | (games["away_conference"] == conference)]


# ---------------------------------------------------------------- pages

@router.get("/", response_class=HTMLResponse)
def upcoming(request: Request, conference: str | None = None):
    d = data()
    games = d["games"]
    if games.empty:
        return render(request, "upcoming.html", {"active": "upcoming", "days": [], "conferences": [], "conference": None})
    today = date.today().isoformat()
    live = d.get("live", pd.DataFrame())
    soon = games[(games["status"] == "scheduled") & (games["date"] >= today)]
    soon = soon.merge(live, on="id") if not live.empty else soon.iloc[0:0]
    shown = filter_conference(soon, conference).merge(d["market"], on="id", how="left") \
        .sort_values(["date", "kickoff", "home"], na_position="last")
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in shown.groupby("date", sort=True)]
    return render(request, "upcoming.html", {"active": "upcoming", "days": days, "conference": conference,
                                             "conferences": conferences(soon) if not soon.empty else [],
                                             "total": len(soon)})


@router.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request, conference: str | None = None):
    return results_page(request, None, None, conference)


@router.get("/results/{season}/{key}/", response_class=HTMLResponse)
def results_week(request: Request, season: int, key: str, conference: str | None = None):
    return results_page(request, season, key, conference)


def results_page(request: Request, season: int | None, key: str | None, conference: str | None):
    games = data()["games"]
    finished = games[games["status"] == "final"]
    if finished.empty:
        return render(request, "results.html", {"active": "results", "days": []})
    finished = finished.assign(key=np.where(finished["season_type"] == "postseason", "bowls", finished["week"].astype(str)))
    weeks = finished.groupby(["season", "key"])["date"].min().reset_index().sort_values("date")
    if season is None:
        season, key = weeks.iloc[-1]["season"], weeks.iloc[-1]["key"]
    pos = weeks[(weeks["season"] == season) & (weeks["key"] == key)]
    if pos.empty:
        raise HTTPException(status_code=404)
    i = weeks.index.get_loc(pos.index[0])
    url = lambda r: f"{BASE}/cfb/results/{r['season']}/{r['key']}/" + (f"?conference={conference}" if conference else "")
    week_games = finished[(finished["season"] == season) & (finished["key"] == key)]
    shown = filter_conference(with_predictions(week_games), conference).sort_values(["date", "kickoff", "home"])
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in shown.groupby("date", sort=True)]
    label = f"{season} season, " + ("bowls and playoff" if key == "bowls" else f"week {key}")
    return render(request, "results.html", {
        "active": "results", "days": days, "label": label, "conference": conference,
        "conferences": conferences(week_games), "summary": summary(shown),
        "previous_url": url(weeks.iloc[i - 1]) if i > 0 else None,
        "next_url": url(weeks.iloc[i + 1]) if i + 1 < len(weeks) else None,
        "base_url": f"{BASE}/cfb/results/{season}/{key}/"})


def summary(df: pd.DataFrame) -> dict | None:
    both = df.dropna(subset=["model_home"])
    both = both[both["home_score"] != both["away_score"]]
    if both.empty:
        return None
    won = (both["home_score"] > both["away_score"]).to_numpy()
    p = lambda col: np.where(won, both[col], 1 - both[col])
    out = {"games": len(both), "model_typical": typical_probability(-np.log(np.clip(p("model_home"), 1e-6, 1)).mean()),
           "model_favorite_won": int((p("model_home") > 0.5).sum())}
    priced = both.dropna(subset=["market_home"])
    if not priced.empty:
        pw = (priced["home_score"] > priced["away_score"]).to_numpy()
        mk = np.where(pw, priced["market_home"], 1 - priced["market_home"])
        out["market_typical"] = typical_probability(-np.log(mk).mean())
    return out


def starters_before(played: pd.DataFrame, cutoff: str) -> dict:
    """{team: (its latest starting quarterback before cutoff, his career starts by then)}."""
    done = played[played["date"] < cutoff]
    starts: dict = {}
    latest: dict = {}
    for side in ("home", "away"):
        for team, qb in zip(done[side], done[f"{side}_qb"]):
            if isinstance(qb, str) and qb:
                starts[qb] = starts.get(qb, 0) + 1
                latest[team] = qb   # games are in date order, so the last one wins
    return {team: (qb, starts.get(qb, 0)) for team, qb in latest.items()}


def qb_names() -> dict:
    d = data()
    if "qb_names" not in d:
        g = d["games"]
        d["qb_names"] = dict(zip(pd.concat([g["home_qb"], g["away_qb"]]), pd.concat([g["home_qb_name"], g["away_qb_name"]])))
    return d["qb_names"]


def strength(model: dict, team: str, starters: dict) -> float:
    qb, n = starters.get(team, (None, np.nan))
    return team_strength(model, team, qb, n)


def fbs_centered(model: dict, season: int, starters: dict) -> pd.DataFrame:
    """Offense, defense, quarterback and overall ratings for the season's FBS teams, measured from
    the average FBS team (the model itself is centered on every rated team, FCS schools included)."""
    r = model["ratings"]
    teams = [t for t in fbs_teams(season) if t in r.index]
    table = pd.DataFrame({"offense": r.loc[teams, "offense"], "defense": r.loc[teams, "defense"]})
    table["qb"] = [strength(model, t, starters) - r.at[t, "offense"] - r.at[t, "defense"] for t in teams]
    table = table - table.mean()
    table["overall"] = table.sum(axis=1)
    return table


def current_model() -> dict | None:
    d = data()
    today = date.today().isoformat()
    if d["games"].empty:
        return None
    if d.get("model_day") != today:
        played = d["games"][d["games"]["status"] == "final"] if not d["games"].empty else pd.DataFrame()
        with connect() as conn:
            everything = load_games(conn)      # FCS games too: they rate the FCS teams FBS schools play
        train = window(everything[everything["status"] == "final"], today)
        season = int(played["season"].max()) if not played.empty else None
        d["model"] = current_fit(everything, today, season) if not train.empty else None
        if d["model"] is not None:
            d["model"]["starters"] = starters_before(everything[everything["status"] == "final"], today)
        d["model_day"] = today
    return d["model"]


def fbs_teams(season: int) -> list[str]:
    games = data()["games"]
    s = games[games["season"] == season]
    return sorted(set(s.loc[s["home_division"] == "fbs", "home"]) | set(s.loc[s["away_division"] == "fbs", "away"]))


def team_conference(team: str, season: int) -> str | None:
    """The team's conference that season (its most common one), from a lookup built once per season."""
    d = data()
    key = ("conferences", season)
    if key not in d:
        games = d["games"]
        s = games[games["season"] == season]
        both = pd.concat([s[["home", "home_conference"]].set_axis(["team", "conf"], axis=1),
                          s[["away", "away_conference"]].set_axis(["team", "conf"], axis=1)]).dropna()
        d[key] = both.groupby("team")["conf"].agg(lambda c: c.mode().iloc[0]).to_dict() if len(both) else {}
    return d[key].get(team)


@router.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request, conference: str | None = None):
    model = current_model()
    games = data()["games"]
    if model is None or games.empty:
        return render(request, "ratings.html", {"active": "ratings", "rows": []})
    season = int(games.loc[games["status"] == "final", "season"].max())
    table = fbs_centered(model, season, model["starters"])
    rows = []
    for t, r in table.iterrows():
        qb = model["starters"].get(t, (None, 0))
        rows.append({"team": t, "conference": team_conference(t, season), "offense": r["offense"], "defense": r["defense"],
                     "qb_value": r["qb"], "qb_name": qb_names().get(qb[0]), "overall": r["overall"]})
    rows.sort(key=lambda x: -x["overall"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    confs = [c for c in POWER if any(r["conference"] == c for r in rows)] + \
        sorted({r["conference"] for r in rows if r["conference"] and r["conference"] not in POWER})
    shown = [r for r in rows if not conference or r["conference"] == conference]
    fmt = lambda v: {"text": signed_number(v, 1), "cls": "good" if round(v, 1) > 0 else "bad" if round(v, 1) < 0 else ""}
    return render(request, "ratings.html", {
        "active": "ratings", "conference": conference, "conferences": confs, "as_of": day_label(date.today().isoformat()),
        "home_adv": f"{model['home_adv']:.1f}",
        "rows": [{**r, "url": team_url(r["team"]), "o": fmt(r["offense"]), "d": fmt(r["defense"]), "all": fmt(r["overall"]),
                  "qb": fmt(r["qb_value"])} for r in shown]})


def find_team(slug: str) -> str:
    games = data()["games"]
    names = {slugify(t): t for t in pd.concat([games["home"], games["away"]]).unique()}
    if slug not in names:
        raise HTTPException(status_code=404)
    return names[slug]


def season_ratings(season: int) -> pd.DataFrame:
    """Every team's overall rating at the start of a season and after each week (cached)."""
    d = data()
    key = f"season_ratings_{season}"
    if key in d:
        return d[key]
    with connect() as conn:
        everything = load_games(conn)
    played = everything[everything["status"] == "final"]
    s = played[played["season"] == season]
    rows = []
    if not s.empty:
        weeks = s.groupby(["season_type", "week"])["date"].agg(["min", "max"]).reset_index().sort_values("min")
        cutoffs = [("Start", "before the season", weeks.iloc[0]["min"])]
        for w in weeks.itertuples():
            label = "Bowls" if w.season_type == "postseason" else str(w.week)
            when = "after the bowls" if w.season_type == "postseason" else f"after week {w.week}"
            cutoffs.append((label, when, (pd.Timestamp(w.max) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        priors: dict = {}
        for step, (label, when, cutoff) in enumerate(cutoffs):
            model = current_fit(everything, cutoff, season, priors)
            ranked = fbs_centered(model, season, starters_before(played, cutoff))["overall"].sort_values(ascending=False)
            for rank, (team, value) in enumerate(ranked.items(), 1):
                rows.append({"step": step, "label": label, "when": when, "team": team, "overall": float(value),
                             "rank": rank, "teams": len(ranked)})
    d[key] = pd.DataFrame(rows)
    return d[key]


@router.get("/team/{slug}/", response_class=HTMLResponse)
def team_latest(request: Request, slug: str):
    return team_page(request, slug, None)


@router.get("/team/{slug}/{season}/", response_class=HTMLResponse)
def team_season(request: Request, slug: str, season: int):
    return team_page(request, slug, season)


def team_page(request: Request, slug: str, season: int | None):
    from charts import chance_bars, line_chart
    team = find_team(slug)
    games = data()["games"]
    own = games[(games["home"] == team) | (games["away"] == team)]
    seasons = sorted(s for s in own["season"].unique() if s >= START_SEASON) or sorted(own["season"].unique())
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    shown = with_predictions(own[own["season"] == season]).sort_values("date")
    played = shown[shown["status"] == "final"]
    coming = shown[shown["status"] == "scheduled"]
    won = sum((g.home == team) == (g.home_score > g.away_score) for g in played.itertuples())
    conf = team_conference(team, season)
    conf_games = played[(played["home_conference"] == played["away_conference"]) & (played["home_conference"] == conf)]
    conf_won = sum((g.home == team) == (g.home_score > g.away_score) for g in conf_games.itertuples())

    ratings = season_ratings(season)
    line = None
    if not ratings.empty and team in set(ratings["team"]):
        mine = ratings[ratings["team"] == team].sort_values("step")
        line = line_chart([{"label": r.label, "value": r.overall,
                            "tip": f"{r.when.capitalize()}: {signed_number(r.overall, 1)} points per game, {ordinal(r.rank)} of {r.teams}"}
                           for r in mine.itertuples()], step=5, minimum=10)
    bars = []
    for g in played.itertuples():
        if pd.isna(g.model_home):
            continue
        at_home = g.home == team
        chance = g.model_home if at_home else 1 - g.model_home
        own_s, opp_s = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        outcome = "W" if own_s > opp_s else "L"
        bars.append({"label": "", "chance": chance, "outcome": outcome,
                     "tip": f"{week_label(g.season_type, g.week)}, {'vs' if at_home else 'at'} {g.away if at_home else g.home}: "
                            f"model {chance * 100:.0f}%, {'won' if outcome == 'W' else 'lost'} {int(own_s)}–{int(opp_s)}"})
    current = ratings[(ratings["team"] == team)].sort_values("step").iloc[-1] if line else None
    names = pd.concat([played.loc[played["home"] == team, "home_qb_name"], played.loc[played["away"] == team, "away_qb_name"]]).dropna()
    quarterbacks = [{"name": q, "starts": int(k)} for q, k in names.value_counts().items()]
    return render(request, "team.html", {
        "active": None, "team": team, "season": season, "conference": conf,
        "record": f"{won}–{len(played) - won}", "conf_record": f"{conf_won}–{len(conf_games) - conf_won}" if len(conf_games) else None,
        "rank": f"{ordinal(int(current['rank']))} of {int(current['teams'])}" if current is not None else None,
        "rating": signed_number(current["overall"], 1) if current is not None else None,
        "summary": summary(played), "rating_chart": line, "quarterbacks": quarterbacks, "chance_chart": chance_bars(bars) if bars else None,
        "played": [game_card(g, team) for g in played.sort_values("date", ascending=False).itertuples()],
        "coming": [game_card(g, team) for g in coming.dropna(subset=["model_home"]).itertuples()],
        "schedule": [{"date": day_label(g.date, with_year=True), "venue": "vs" if g.home == team else "at",
                      "opponent": g.away if g.home == team else g.home, "kickoff": kickoff_label(g.kickoff)}
                     for g in coming[coming["model_home"].isna()].itertuples()],
        "seasons": [{"label": str(s), "url": team_url(team, s), "selected": s == season} for s in sorted(seasons, reverse=True)],
    })


@router.get("/track-record/", response_class=HTMLResponse)
def track_record(request: Request):
    d = data()
    live = d.get("live", pd.DataFrame())
    games = d["games"]
    if live.empty or games.empty:
        return render(request, "track_record.html", {"active": "track", "weeks": [], "record_url": RECORD_URL})
    df = with_predictions(games[games["status"] == "final"].merge(live[["id"]], on="id")).sort_values(["date", "home"])
    weeks = []
    for (season, stype, week), group in sorted(df.groupby(["season", "season_type", "week"]), key=lambda kv: kv[1]["date"].min(), reverse=True):
        weeks.append({"label": f"{season} {week_label(stype, week)}", "summary": summary(group),
                      "games": [game_card(g) for g in group.itertuples()]})
    from charts import record_chart
    chart_weeks = []
    for (season, stype, week), group in sorted(df.groupby(["season", "season_type", "week"]), key=lambda kv: kv[1]["date"].min()):
        g = group.dropna(subset=["model_home", "elo_home"])
        won = (g["home_score"] > g["away_score"]).to_numpy()
        chart_weeks.append({"label": "Bowls" if stype == "postseason" else f"Wk {week}",
                            "model": np.where(won, g["model_home"], 1 - g["model_home"]),
                            "benchmark": np.where(won, g["elo_home"], 1 - g["elo_home"])})
    from charts import calibration_chart
    from web import calibration_pairs
    past = with_predictions(games[(games["status"] == "final") & (games["season"] >= HELD_OUT_FROM)])
    past = past[past["source"] == "Backtest"]
    calib = calibration_chart(calibration_pairs(df), calibration_pairs(past), past_from=f"{HELD_OUT_FROM}")
    return render(request, "track_record.html", {"active": "track", "weeks": weeks, "overall": summary(df),
                                                 "record_url": RECORD_URL, "calibration": calib,
                                                 "record_chart": record_chart(chart_weeks, "CollegeFootballData's Elo")})


@router.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    games = data()["games"]
    df = with_predictions(games[(games["status"] == "final") & (games["season"] >= START_SEASON)])
    df = df[df["source"] == "Backtest"]
    df = df[(df["home_score"] != df["away_score"]) & (df["season"] <= 2025)].dropna(subset=["model_home"])
    if df.empty:
        return render(request, "backtest.html", {"active": "backtest", "seasons": []})
    won = (df["home_score"] > df["away_score"]).to_numpy()
    loss = lambda p: -np.log(np.clip(np.where(won, p, 1 - p), 1e-6, 1))
    df = df.assign(model_ll=loss(df["model_home"]),
                   elo_ll=np.where(df["elo_home"].notna(), loss(df["elo_home"].fillna(0.5)), np.nan),
                   market_ll=np.where(df["market_home"].notna(), loss(df["market_home"].fillna(0.5)), np.nan))
    held = df[df["season"] >= HELD_OUT_FROM]
    priced = held.dropna(subset=["market_ll"])
    rated = held.dropna(subset=["elo_ll"])      # Elo is missing for some games: compare on the same games
    by_season = []
    for season, part in df.groupby("season"):
        p = part.dropna(subset=["market_ll"])
        r = part.dropna(subset=["elo_ll"])
        by_season.append({"label": str(season), "games": len(r), "model": f"{r['model_ll'].mean():.3f}",
                          "elo": f"{r['elo_ll'].mean():.3f}", "priced": len(p),
                          "model_priced": f"{p['model_ll'].mean():.3f}" if len(p) else "–",
                          "market": f"{p['market_ll'].mean():.3f}" if len(p) else "–", "held_out": season >= HELD_OUT_FROM})
    bins = pd.cut(held["model_home"], [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1])
    calibration = [{"range": f"{int(b.left * 100)}–{int(b.right * 100)}%", "games": len(part),
                    "predicted": f"{part['model_home'].mean() * 100:.0f}%",
                    "actual": f"{(part['home_score'] > part['away_score']).mean() * 100:.0f}%"}
                   for b, part in held.groupby(bins, observed=True)]
    return render(request, "backtest.html", {
        "active": "backtest", "seasons": by_season, "calibration": calibration, "from": HELD_OUT_FROM,
        "headline": {"games": len(held), "rated": len(rated), "model": f"{rated['model_ll'].mean():.4f}",
                     "elo": f"{rated['elo_ll'].mean():.4f}",
                     "model_typical": typical_probability(held["model_ll"].mean()),
                     "elo_typical": typical_probability(rated["elo_ll"].mean()),
                     "model_rated_typical": typical_probability(rated["model_ll"].mean()),
                     "priced": len(priced), "model_priced": f"{priced['model_ll'].mean():.4f}",
                     "market_priced": f"{priced['market_ll'].mean():.4f}",
                     "elo_priced": f"{priced['elo_ll'].mean():.4f}",
                     "favorite": f"{((held['model_home'] > 0.5) == won[df['season'].to_numpy() >= HELD_OUT_FROM]).mean() * 100:.1f}%"},
        "margin_sd": f"{MARGIN_SD:g}"})


# ---------------------------------------------------------------- standings

def standings_table(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Every top-division (FBS) team's record in a season: overall (regular season and bowls) and in
    conference games, points for and against. One row per team, with its conference."""
    s = games[(games["season"] == season) & (games["status"] == "final")]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = s[s[f"{side}_division"] == "fbs"]
        rows.append(pd.DataFrame({
            "team": part[side], "conference": part[f"{side}_conference"],
            "won": (part[f"{side}_score"] > part[f"{other}_score"]).astype(int),
            "lost": (part[f"{side}_score"] < part[f"{other}_score"]).astype(int),
            "conf_game": (part["home_conference"] == part["away_conference"]) & (part["season_type"] == "regular"),
            "pf": part[f"{side}_score"], "pa": part[f"{other}_score"]}))
    rows = pd.concat(rows, ignore_index=True)
    if rows.empty:
        return pd.DataFrame(columns=["team", "conference", "won", "lost", "conf_won", "conf_lost", "pf", "pa"])
    rows["conf_won"] = rows["won"] * rows["conf_game"]
    rows["conf_lost"] = rows["lost"] * rows["conf_game"]
    table = rows.groupby("team").agg(conference=("conference", lambda c: c.mode().iloc[0] if not c.mode().empty else None),
                                     won=("won", "sum"), lost=("lost", "sum"), conf_won=("conf_won", "sum"),
                                     conf_lost=("conf_lost", "sum"), pf=("pf", "sum"), pa=("pa", "sum")).reset_index()
    pct = lambda w, l: np.where(w + l > 0, w / np.maximum(w + l, 1), 0.0)
    table["conf_pct"] = pct(table["conf_won"], table["conf_lost"])
    table["pct"] = pct(table["won"], table["lost"])
    return table.sort_values(["conference", "conf_pct", "conf_won", "pct", "won"],
                             ascending=[True, False, False, False, False]).reset_index(drop=True)


@router.get("/standings/", response_class=HTMLResponse)
def standings_latest(request: Request, conference: str | None = None):
    return standings_page(request, None, conference)


@router.get("/standings/{season}/", response_class=HTMLResponse)
def standings_season(request: Request, season: int, conference: str | None = None):
    return standings_page(request, season, conference)


def standings_page(request: Request, season: int | None, conference: str | None):
    games = data()["games"]
    finished = games[games["status"] == "final"]
    seasons = sorted(s for s in finished["season"].unique() if s >= START_SEASON)
    if not seasons:
        return render(request, "standings.html", {"active": "standings", "groups": []})
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    table = standings_table(games, season)
    model = current_model() if season == seasons[-1] else None
    ratings = fbs_centered(model, season, model["starters"]) if model else None
    confs = [c for c in POWER if c in set(table["conference"])] + \
        sorted(c for c in set(table["conference"].dropna()) if c not in POWER)
    groups = []
    for conf in confs:
        if conference and conf != conference:
            continue
        members = table[table["conference"] == conf]
        has_conf_games = bool((members["conf_won"] + members["conf_lost"]).sum())
        groups.append({"label": conf, "conference_games": has_conf_games, "rows": [{
            "team": r.team, "url": team_url(r.team, season if season != seasons[-1] else None),
            "record": f"{r.won}–{r.lost}", "conf": f"{r.conf_won}–{r.conf_lost}" if has_conf_games else "",
            "pf": int(r.pf), "pa": int(r.pa), "diff": signed_number(r.pf - r.pa),
            "rating": signed_number(ratings.at[r.team, "overall"], 1)
            if ratings is not None and r.team in ratings.index else None} for r in members.itertuples()]})
    return render(request, "standings.html", {
        "active": "standings", "groups": groups, "season": season, "is_current": season == seasons[-1],
        "conference": conference, "conferences": confs, "base_url": f"{BASE}/cfb/standings/{season}/",
        "seasons": [{"label": str(s), "url": f"{BASE}/cfb/standings/{s}/", "selected": s == season} for s in sorted(seasons, reverse=True)]})


# ---------------------------------------------------------------- conference title odds

def load_conference_odds() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'cfb_conference_odds'").fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM cfb_conference_odds", conn)


def conference_odds() -> dict | None:
    """The latest title odds for the latest season that has them, and the ones from about a week
    earlier (for the change column). Cached with the rest of the data."""
    d = data()
    if "conference_odds" not in d:
        odds = load_conference_odds()
        result = None
        if not odds.empty:
            season = int(odds["season"].max())
            odds = odds[odds["season"] == season]
            dates = sorted(odds["as_of"].unique())
            week_ago = [x for x in dates if pd.Timestamp(x) <= pd.Timestamp(dates[-1]) - pd.Timedelta(days=6)]
            result = {"season": season, "as_of": dates[-1], "latest": odds[odds["as_of"] == dates[-1]],
                      "before": odds[odds["as_of"] == week_ago[-1]].set_index("team") if week_ago else None,
                      "before_as_of": week_ago[-1] if week_ago else None}
        d["conference_odds"] = result
    return d["conference_odds"]


@router.get("/conference-odds/", response_class=HTMLResponse)
def conference_odds_page(request: Request, conference: str | None = None):
    from cfb.conference import RATING_SD, SIMULATIONS
    from nfl.site import odds_text
    odds = conference_odds()
    if odds is None:
        return render(request, "conference_odds.html", {"active": "conference", "groups": [], "conferences": []})
    latest = odds["latest"]
    table = standings_table(data()["games"], odds["season"]).set_index("team")
    confs = [c for c in POWER if c in set(latest["conference"])] + \
        sorted(c for c in set(latest["conference"]) if c not in POWER)
    if conference and conference not in confs:
        raise HTTPException(status_code=404)
    cell = lambda p: {"text": odds_text(p), "value": float(p)}
    groups = []
    for conf in confs:
        if conference and conf != conference:
            continue
        rows = []
        for r in latest[latest["conference"] == conf].sort_values(["champion", "title_game", "conf_wins"], ascending=False).itertuples():
            record = table.loc[r.team] if r.team in table.index else None
            change = None
            if odds["before"] is not None and r.team in odds["before"].index:
                delta = (r.champion - odds["before"].at[r.team, "champion"]) * 100
                if round(delta) != 0:
                    change = {"text": f"{delta:+.0f}".replace("-", "−"), "dir": "up" if delta > 0 else "down"}
            rows.append({"team": r.team, "url": team_url(r.team),
                         "conf": f"{record.conf_won}–{record.conf_lost}" if record is not None else "0–0",
                         "record": f"{record.won}–{record.lost}" if record is not None else "0–0",
                         "conf_wins": f"{r.conf_wins:.1f}–{r.conf_games - r.conf_wins:.1f}",
                         "title_game": cell(r.title_game), "champion": cell(r.champion), "change": change})
        groups.append({"label": conf, "sun_belt": conf == "Sun Belt", "rows": rows})
    return render(request, "conference_odds.html", {
        "active": "conference", "groups": groups, "conferences": confs, "conference": conference,
        "base_url": f"{BASE}/cfb/conference-odds/", "season": odds["season"],
        "as_of": day_label(odds["as_of"]), "before_as_of": day_label(odds["before_as_of"]) if odds["before_as_of"] else None,
        "simulations": f"{SIMULATIONS:,}", "rating_sd": f"{RATING_SD:g}"})


# ---------------------------------------------------------------- game pages

def qb_label(qb, starts) -> str:
    name = qb_names().get(qb, qb) if isinstance(qb, str) else None
    if not name:
        return "not known"
    return f"{name} ({int(starts)} career start{'' if starts == 1 else 's'})" if pd.notna(starts) else name


@router.get("/game/{source_id}/", response_class=HTMLResponse)
def game_page(request: Request, source_id: str):
    d = data()
    games = d["games"]
    found = games[games["source_game_id"].astype(str) == source_id]
    if found.empty:
        raise HTTPException(status_code=404)
    g = next(with_predictions(found).itertuples())
    card = game_card(g)
    finished = g.status == "final"
    reasons = None
    model = current_model() if not finished else None
    if model is not None and pd.notna(getattr(g, "model_home", np.nan)):
        hq, aq = g.home_qb_expected, g.away_qb_expected
        hs, as_ = g.home_qb_expected_starts, g.away_qb_expected_starts
        qb_part = ((team_strength(model, g.home, hq, hs) - team_strength(model, g.home))
                   - (team_strength(model, g.away, aq, as_) - team_strength(model, g.away)))
        reasons = college.points_why(model, g.home, g.away, g.home, g.away, bool(g.neutral_site), float(g.model_home),
                                     getattr(g, "market_home", None),
                                     qb={"value": 0.0 if np.isnan(qb_part) else qb_part,
                                         "note": f"{qb_label(aq, as_)} vs {qb_label(hq, hs)}, each against an average starter"})
    with connect() as conn:
        moved = betting.movement_chart(conn, int(g.id), g.home, g.away)
    url = lambda row: game_url(row.source_game_id)
    stage = lambda row: row.notes if isinstance(row.notes, str) else ("Bowls and playoff" if row.season_type == "postseason" else "")
    return templates.TemplateResponse(request, "college_game.html", {"sport": "cfb",
        "active": None, "sport_name": "College football", "g": card, "finished": finished,
        "date": day_label(g.date, with_year=True), "why": reasons, "moved": moved, "first_season": 2001,
        "forms": {side: college.form(games, getattr(g, side), g.date, int(g.season), url, 5) for side in ("away", "home")},
        "h2h": college.head_to_head(games, g.away, g.home, g.date, url, stage=stage)})
