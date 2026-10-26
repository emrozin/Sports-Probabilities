"""The men's college basketball pages of the website, under /cbb/ (included by app.py).

Games between two Division I teams. There are no free betting odds for basketball, so the
backtest compares the model with a simple Elo rating instead (cbb/model.py).
"""

import sqlite3
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

import colors
from cbb.model import MARGIN_SD, NON_D1, elo_benchmark, fit, load_games, window
from cbb.predict_upcoming import MODEL_NAME
from cbb.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
from cfb.site import kickoff_label, line_text, two_way
from charts import chance_bars, line_chart
import espn
from web import BASE, ROOT, edge, day_label, ordinal, signed_number, slugify, templates, typical_probability

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter(prefix="/cbb")
RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/cbb_live_predictions.csv"
POWER = ["SEC", "Big Ten", "Big 12", "ACC", "Big East"]
GAME_COLUMNS = ["id", "season", "date", "kickoff", "status", "home_team", "away_team", "home_score", "away_score",
                "neutral_site", "season_type", "source_game_id", "conference_game", "home_conference", "away_conference",
                "notes", "home", "away", "week"]

_cache: dict = {}


def connect():
    return closing(sqlite3.connect(DB_PATH))


def season_label(season: int) -> str:
    return f"{season}–{(season + 1) % 100:02d}"


def team_url(team: str, season: int | None = None) -> str:
    return f"{BASE}/cbb/team/{slugify(team)}/" + (f"{season}/" if season is not None else "")


def team_colors() -> dict:
    return data().get("colors", {})


templates.env.globals["cbb_team_url"] = team_url
templates.env.globals["cbb_colors"] = lambda home, away: colors.style(team_colors(), home, away)
templates.env.globals["cbb_team_colors"] = lambda team: colors.team_style(team_colors(), team)
templates.env.globals["cbb_chip"] = lambda team: colors.chip(team_colors(), team)


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request, f"cbb/{template}", {"sport": "cbb", **context}, status_code=status_code)


def load_predictions(conn, name: str, version: str | None) -> pd.DataFrame:
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
    games = pd.DataFrame(columns=GAME_COLUMNS)
    if stamp is not None:
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'cbb_game_info'").fetchone():
                games = load_games(conn)
                live = load_predictions(conn, MODEL_NAME, None).assign(source="Saved before tip-off")
                back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).assign(source="Backtest")
                _cache["colors"] = {n: (c, a or "#7F7F7F") for n, c, a in conn.execute("SELECT name, color, alt_color FROM cbb_teams")
                                    if c}
                _cache["predictions"] = pd.concat([live, back[~back["id"].isin(live["id"])]], ignore_index=True)
                _cache["live"] = live
                _cache["market"] = espn.market_lines(conn, "CBB")
    _cache.setdefault("predictions", pd.DataFrame(columns=["id", "model_home", "exp_home", "exp_away", "source"]))
    _cache.setdefault("live", pd.DataFrame(columns=["id", "model_home", "exp_home", "exp_away", "source"]))
    _cache.setdefault("market", pd.DataFrame(columns=["id", "market_home", "market_spread", "market_total", "market_final"]))
    for key in ("predictions", "live"):
        for col in ("model_home", "exp_home", "exp_away"):
            _cache[key][col] = pd.to_numeric(_cache[key][col], errors="coerce")
    _cache["games"] = games
    return _cache


def d1(games: pd.DataFrame) -> pd.DataFrame:
    return games[(games["home"] != NON_D1) & (games["away"] != NON_D1)]


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    return games.merge(data()["predictions"], on="id", how="left").merge(data()["market"], on="id", how="left")


def game_card(g, current_team: str | None = None) -> dict:
    finished = g.status == "final"
    margin = (g.exp_home - g.exp_away) if pd.notna(getattr(g, "exp_home", np.nan)) else None
    home, away = g.home_team, g.away_team
    card = {"home": home, "away": away, "finished": finished, "neutral": bool(g.neutral_site),
            "home_conf": g.home_conference, "away_conf": g.away_conference,
            "model": two_way(g.model_home) if pd.notna(getattr(g, "model_home", np.nan)) else None,
            "source": g.source if isinstance(getattr(g, "source", None), str) else None,
            "model_line": line_text(margin, home, away),
            "model_total": f"{g.exp_home + g.exp_away:.0f}" if margin is not None else None}
    market_home = getattr(g, "market_home", np.nan)
    if pd.notna(market_home):
        card["market"] = two_way(market_home)
        spread, total = getattr(g, "market_spread", np.nan), getattr(g, "market_total", np.nan)
        card["market_line"] = line_text(-spread, home, away) if pd.notna(spread) else None
        card["market_total"] = f"{total:g}" if pd.notna(total) else None
        if card["model"] and not finished:
            card["edge"] = edge(card["model"], card["market"], {"home": home, "away": away})
    when = [day_label(g.date, with_year=True)] if current_team else []
    when += [kickoff_label(g.kickoff)] if isinstance(g.kickoff, str) else []
    when += ["neutral site"] if card["neutral"] else []
    when += [g.notes] if isinstance(g.notes, str) else []
    card["when"] = ", ".join(x for x in when if x)
    if finished:
        hs, as_ = int(g.home_score), int(g.away_score)
        card.update(home_score=hs, away_score=as_, result="home" if hs > as_ else "away")
        if card["model"]:
            card["model_actual"] = card["model"][card["result"]]
        if card.get("market"):
            card["market_actual"] = card["market"][card["result"]]
        if current_team:
            own, other = (hs, as_) if home == current_team else (as_, hs)
            card["outcome"] = "W" if own > other else "L"
    return card


def conferences(games: pd.DataFrame) -> list[str]:
    confs = set(games["home_conference"].dropna()) | set(games["away_conference"].dropna())
    return [c for c in POWER if c in confs] + sorted(c for c in confs if c not in POWER)


def filter_conference(games: pd.DataFrame, conference: str | None) -> pd.DataFrame:
    if not conference:
        return games
    return games[(games["home_conference"] == conference) | (games["away_conference"] == conference)]


def summary(df: pd.DataFrame) -> dict | None:
    df = df.dropna(subset=["model_home"])
    if df.empty:
        return None
    won = (df["home_score"] > df["away_score"]).to_numpy()
    p = np.where(won, df["model_home"].astype(float), 1 - df["model_home"].astype(float))
    return {"games": len(df), "model_typical": typical_probability(-np.log(np.clip(p, 1e-6, 1)).mean()),
            "model_favorite_won": int((p > 0.5).sum())}


def latest_season() -> int | None:
    games = data()["games"]
    return int(games["season"].max()) if not games.empty else None


# ---------------------------------------------------------------- pages

@router.get("/", response_class=HTMLResponse)
def upcoming(request: Request, conference: str | None = None):
    d = data()
    games = d1(d["games"])
    today = date.today().isoformat()
    soon = games[(games["status"] == "scheduled") & (games["date"] >= today)]
    predicted = soon.merge(d["live"], on="id").merge(d["market"], on="id", how="left")
    shown = filter_conference(predicted, conference).sort_values(["date", "kickoff", "home"], na_position="last")
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in shown.groupby("date", sort=True)]
    first = soon["date"].min() if not soon.empty else None
    return render(request, "upcoming.html", {"active": "upcoming", "days": days, "conference": conference,
                                             "conferences": conferences(predicted), "season_starts": day_label(first) if first else None})


@router.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request, conference: str | None = None):
    return results_page(request, None, conference)


@router.get("/results/{week}/", response_class=HTMLResponse)
def results_week(request: Request, week: str, conference: str | None = None):
    return results_page(request, week, conference)


def results_page(request: Request, week: str | None, conference: str | None):
    finished = d1(data()["games"])
    finished = finished[finished["status"] == "final"]
    if finished.empty:
        return render(request, "results.html", {"active": "results", "days": []})
    weeks = sorted(finished["week"].unique())
    if week is None:
        week = weeks[-1]
    elif week not in weeks:
        raise HTTPException(status_code=404)
    i = weeks.index(week)
    q = f"?conference={conference}" if conference else ""
    shown = filter_conference(with_predictions(finished[finished["week"] == week]), conference).sort_values(["date", "kickoff", "home"])
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]} for day, group in shown.groupby("date")]
    start = week
    return render(request, "results.html", {
        "active": "results", "days": days, "conference": conference, "summary": summary(shown),
        "conferences": conferences(finished[finished["week"] == week]),
        "label": f"Week of {day_label(start)}", "base_url": f"{BASE}/cbb/results/{week}/",
        "previous_url": f"{BASE}/cbb/results/{weeks[i - 1]}/{q}" if i > 0 else None,
        "next_url": f"{BASE}/cbb/results/{weeks[i + 1]}/{q}" if i + 1 < len(weeks) else None})


def current_model() -> dict | None:
    d = data()
    today = date.today().isoformat()
    if d["games"].empty:
        return None
    if d.get("model_day") != today:
        played = d["games"][d["games"]["status"] == "final"]
        train = window(played, today)
        d["model"] = fit(train, today, latest_season()) if not train.empty else None
        d["model_day"] = today
    return d["model"]


def d1_teams(season: int) -> list[str]:
    s = data()["games"]
    s = s[s["season"] == season]
    return sorted(set(s.loc[s["home_conference"].notna(), "home_team"]) | set(s.loc[s["away_conference"].notna(), "away_team"]))


def team_conference(team: str, season: int) -> str | None:
    s = data()["games"]
    s = s[s["season"] == season]
    both = pd.concat([s.loc[s["home_team"] == team, "home_conference"], s.loc[s["away_team"] == team, "away_conference"]]).dropna()
    return both.mode().iloc[0] if not both.empty else None


def centered(model: dict, season: int) -> pd.DataFrame:
    """Offense, defense and overall for the season's Division I teams, from the average D-I team."""
    r = model["ratings"]
    teams = [t for t in d1_teams(season) if t in r.index]
    table = r.loc[teams, ["offense", "defense"]]
    table = table - table.mean()
    table["overall"] = table["offense"] + table["defense"]
    return table


@router.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request, conference: str | None = None):
    model = current_model()
    season = latest_season()
    if model is None:
        return render(request, "ratings.html", {"active": "ratings", "rows": []})
    games = data()["games"]
    shown_season = season if ((games["season"] == season) & (games["status"] == "final")).any() else season - 1
    table = centered(model, shown_season).sort_values("overall", ascending=False)   # preseason: last season's teams
    fmt = lambda v: {"text": signed_number(v, 1), "cls": "good" if round(v, 1) > 0 else "bad" if round(v, 1) < 0 else ""}
    rows = [{"rank": i, "team": t, "url": team_url(t), "conference": team_conference(t, shown_season),
             "o": fmt(r.offense), "d": fmt(r.defense), "all": fmt(r.overall)}
            for i, (t, r) in enumerate(table.iterrows(), 1)]
    confs = [c for c in POWER if any(r["conference"] == c for r in rows)] + \
        sorted({r["conference"] for r in rows if r["conference"] and r["conference"] not in POWER})
    return render(request, "ratings.html", {
        "active": "ratings", "conference": conference, "conferences": confs, "as_of": day_label(date.today().isoformat()),
        "home_adv": f"{model['home_adv']:.1f}", "rows": [r for r in rows if not conference or r["conference"] == conference]})


def standings_table(season: int) -> pd.DataFrame:
    s = data()["games"]
    s = s[(s["season"] == season) & (s["status"] == "final")]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = s[s[f"{side}_conference"].notna()]
        rows.append(pd.DataFrame({"team": part[f"{side}_team"], "conference": part[f"{side}_conference"],
                                  "won": (part[f"{side}_score"] > part[f"{other}_score"]).astype(int),
                                  "lost": (part[f"{side}_score"] < part[f"{other}_score"]).astype(int),
                                  "conf_game": (part["conference_game"] == 1) & (part["season_type"] == "regular"),
                                  "pf": part[f"{side}_score"], "pa": part[f"{other}_score"]}))
    rows = pd.concat(rows, ignore_index=True)
    if rows.empty:
        return pd.DataFrame(columns=["team", "conference", "won", "lost", "conf_won", "conf_lost", "pf", "pa", "games"])
    rows["conf_won"], rows["conf_lost"] = rows["won"] * rows["conf_game"], rows["lost"] * rows["conf_game"]
    table = rows.groupby("team").agg(conference=("conference", lambda c: c.mode().iloc[0]), won=("won", "sum"), lost=("lost", "sum"),
                                     conf_won=("conf_won", "sum"), conf_lost=("conf_lost", "sum"), pf=("pf", "sum"),
                                     pa=("pa", "sum"), games=("won", "size")).reset_index()
    pct = lambda w, l: np.where(w + l > 0, w / np.maximum(w + l, 1), 0.0)
    table["conf_pct"], table["pct"] = pct(table["conf_won"], table["conf_lost"]), pct(table["won"], table["lost"])
    return table.sort_values(["conference", "conf_pct", "conf_won", "pct"], ascending=[True, False, False, False])


@router.get("/standings/", response_class=HTMLResponse)
def standings_latest(request: Request, conference: str | None = None):
    return standings_page(request, None, conference)


@router.get("/standings/{season}/", response_class=HTMLResponse)
def standings_season(request: Request, season: int, conference: str | None = None):
    return standings_page(request, season, conference)


def standings_page(request: Request, season: int | None, conference: str | None):
    games = data()["games"]
    seasons = sorted(s for s in games.loc[games["status"] == "final", "season"].unique() if s >= START_SEASON)
    if not seasons:
        return render(request, "standings.html", {"active": "standings", "groups": []})
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    table = standings_table(season)
    model = current_model() if season == seasons[-1] else None
    rated = centered(model, season) if model else None
    confs = [c for c in POWER if c in set(table["conference"])] + sorted(c for c in set(table["conference"]) if c not in POWER)
    groups = []
    for conf in confs:
        if conference and conf != conference:
            continue
        members = table[table["conference"] == conf]
        has_conf = bool((members["conf_won"] + members["conf_lost"]).sum())
        groups.append({"label": conf, "conference_games": has_conf, "rows": [{
            "team": r.team, "url": team_url(r.team, season if season != seasons[-1] else None),
            "record": f"{r.won}–{r.lost}", "conf": f"{r.conf_won}–{r.conf_lost}" if has_conf else "",
            "pf": f"{r.pf / r.games:.1f}", "pa": f"{r.pa / r.games:.1f}", "diff": signed_number((r.pf - r.pa) / r.games, 1),
            "rating": signed_number(rated.at[r.team, "overall"], 1) if rated is not None and r.team in rated.index else None}
            for r in members.itertuples()]})
    return render(request, "standings.html", {
        "active": "standings", "groups": groups, "season": season_label(season), "is_current": season == seasons[-1],
        "conference": conference, "conferences": confs, "base_url": f"{BASE}/cbb/standings/{season}/",
        "seasons": [{"label": season_label(s), "url": f"{BASE}/cbb/standings/{s}/", "selected": s == season}
                    for s in sorted(seasons, reverse=True)]})


def find_team(slug: str) -> tuple[str, set]:
    """The team behind a link: its name (from its latest game) and every name it appears under
    (a few small schools have two ESPN ids, e.g. 'Marian' and 'MARIAN')."""
    d = data()
    if "slugs" not in d:
        games = d["games"]
        last = pd.concat([games[["date", "home_team"]].set_axis(["date", "team"], axis=1),
                          games[["date", "away_team"]].set_axis(["date", "team"], axis=1)]).groupby("team")["date"].max()
        slugs: dict = {}
        for team, latest in last.sort_values().items():          # latest last, so its name wins
            name, names = slugs.get(slugify(team), (None, set()))
            slugs[slugify(team)] = (team, names | {team})
        d["slugs"] = slugs
    if slug not in d["slugs"]:
        raise HTTPException(status_code=404)
    return d["slugs"][slug]


def season_ratings(season: int) -> pd.DataFrame:
    """Every D-I team's overall rating at the start of a season and after each week (cached)."""
    d = data()
    key = f"season_ratings_{season}"
    if key in d:
        return d[key]
    played = d["games"][d["games"]["status"] == "final"]
    s = played[played["season"] == season]
    rows = []
    if not s.empty:
        weeks = s.groupby("week")["date"].agg(["min", "max"]).reset_index().sort_values("min")
        cutoffs = [("Start", "before the season", weeks.iloc[0]["min"])]
        for w in weeks.itertuples():
            after = (pd.Timestamp(w.max) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            cutoffs.append((f"{pd.Timestamp(w.max):%-d %b}", f"after {day_label(w.max)}", after))
        for step, (label, when, cutoff) in enumerate(cutoffs):
            model = fit(window(played, cutoff), cutoff, season)
            ranked = centered(model, season)["overall"].sort_values(ascending=False)
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
    team, aliases = find_team(slug)
    games = data()["games"]
    own = games[games["home_team"].isin(aliases) | games["away_team"].isin(aliases)].copy()
    for side in ("home_team", "away_team"):
        own.loc[own[side].isin(aliases), side] = team
    seasons = sorted(s for s in own["season"].unique() if s >= START_SEASON) or sorted(own["season"].unique())
    if season is None:
        played_seasons = sorted(own.loc[own["status"] == "final", "season"].unique())
        season = seasons[-1] if own[(own["season"] == seasons[-1]) & (own["status"] == "final")].shape[0] else (played_seasons or seasons)[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    shown = with_predictions(own[own["season"] == season]).sort_values("date")
    played, coming = shown[shown["status"] == "final"], shown[shown["status"] == "scheduled"]
    won = sum((g.home_team == team) == (g.home_score > g.away_score) for g in played.itertuples())
    conf = team_conference(team, season)
    conf_games = played[(played["conference_game"] == 1) & (played["season_type"] == "regular")]
    conf_won = sum((g.home_team == team) == (g.home_score > g.away_score) for g in conf_games.itertuples())
    ratings = season_ratings(season)
    mine = ratings[ratings["team"] == team].sort_values("step") if not ratings.empty else ratings
    line = line_chart([{"label": r.label, "value": r.overall,
                        "tip": f"{r.when.capitalize()}: {signed_number(r.overall, 1)} points per game, {ordinal(r.rank)} of {r.teams}"}
                       for r in mine.itertuples()], step=5, minimum=10) if len(mine) > 1 else None
    bars = []
    for g in played.dropna(subset=["model_home"]).itertuples():
        at_home = g.home_team == team
        chance = g.model_home if at_home else 1 - g.model_home
        own_s, opp_s = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        outcome = "W" if own_s > opp_s else "L"
        bars.append({"label": "", "chance": chance, "outcome": outcome,
                     "tip": f"{day_label(g.date, with_year=True)}, {'vs' if at_home else 'at'} {g.away_team if at_home else g.home_team}: "
                            f"model {chance * 100:.0f}%, {'won' if outcome == 'W' else 'lost'} {int(own_s)}–{int(opp_s)}"})
    current = mine.iloc[-1] if len(mine) else None
    return render(request, "team.html", {
        "active": None, "team": team, "season": season_label(season), "conference": conf,
        "record": f"{won}–{len(played) - won}", "conf_record": f"{conf_won}–{len(conf_games) - conf_won}" if len(conf_games) else None,
        "rating": signed_number(current["overall"], 1) if current is not None else None,
        "rank": f"{ordinal(int(current['rank']))} of {int(current['teams'])}" if current is not None else None,
        "summary": summary(played), "rating_chart": line, "chance_chart": chance_bars(bars) if bars else None,
        "played": [game_card(g, team) for g in played.sort_values("date", ascending=False).itertuples()],
        "coming": [game_card(g, team) for g in coming.dropna(subset=["model_home"]).itertuples()],
        "schedule": [{"date": day_label(g.date, with_year=True), "venue": "vs" if g.home_team == team else "at",
                      "opponent": g.away_team if g.home_team == team else g.home_team} for g in coming[coming["model_home"].isna()].itertuples()],
        "seasons": [{"label": season_label(s), "url": team_url(team, s), "selected": s == season} for s in sorted(seasons, reverse=True)]})


@router.get("/track-record/", response_class=HTMLResponse)
def track_record(request: Request):
    d = data()
    df = d1(d["games"])
    df = df[df["status"] == "final"].merge(d["live"], on="id").merge(d["market"], on="id", how="left").sort_values(["date", "home"])
    weeks = [{"label": f"Week of {day_label(w)}", "summary": summary(group),
              "games": [game_card(g) for g in group.itertuples()]}
             for w, group in sorted(df.groupby("week"), key=lambda kv: kv[0], reverse=True)]
    chart = None
    if not df.empty:
        from charts import record_chart
        if "elo" not in d:
            d["elo"] = elo_benchmark(d["games"])
        df = df.merge(d["elo"].rename("elo_home"), left_on="id", right_index=True, how="left")
        chart_weeks = []
        for w, group in sorted(df.groupby("week"), key=lambda kv: kv[0]):
            g = group.dropna(subset=["model_home", "elo_home"])
            won = (g["home_score"] > g["away_score"]).to_numpy()
            chart_weeks.append({"label": f"{pd.Timestamp(w):%-d %b}", "model": np.where(won, g["model_home"], 1 - g["model_home"]),
                                "benchmark": np.where(won, g["elo_home"], 1 - g["elo_home"])})
        chart = record_chart(chart_weeks, "a simple Elo rating")
    return render(request, "track_record.html", {"active": "track", "weeks": weeks, "overall": summary(df) if not df.empty else None,
                                                 "record_url": RECORD_URL, "record_chart": chart})


@router.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    d = data()
    games = d["games"]
    if "elo" not in d:
        d["elo"] = elo_benchmark(games) if not games.empty else pd.Series(dtype=float, name="elo_home")
    df = with_predictions(d1(games[(games["status"] == "final") & (games["season"] >= START_SEASON)]))
    df = df[df["source"] == "Backtest"].dropna(subset=["model_home"])
    df = df[df["season"] < latest_season()] if latest_season() is not None else df
    if df.empty:
        return render(request, "backtest.html", {"active": "backtest", "seasons": []})
    df = df.merge(d["elo"].rename("elo_home"), left_on="id", right_index=True, how="left")
    won = (df["home_score"] > df["away_score"]).to_numpy()
    loss = lambda p: -np.log(np.clip(np.where(won, p.astype(float), 1 - p.astype(float)), 1e-6, 1))
    df = df.assign(model_ll=loss(df["model_home"]), elo_ll=loss(df["elo_home"].fillna(0.5)))
    held = df[df["season"] >= HELD_OUT_FROM]
    seasons = [{"label": season_label(s), "games": len(part), "model": f"{part['model_ll'].mean():.3f}",
                "elo": f"{part['elo_ll'].mean():.3f}", "held_out": s >= HELD_OUT_FROM} for s, part in df.groupby("season")]
    bins = pd.cut(held["model_home"], [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1])
    calibration = [{"range": f"{int(b.left * 100)}–{int(b.right * 100)}%", "games": len(part),
                    "predicted": f"{part['model_home'].mean() * 100:.0f}%",
                    "actual": f"{(part['home_score'] > part['away_score']).mean() * 100:.0f}%"}
                   for b, part in held.groupby(bins, observed=True)]
    won_held = (held["home_score"] > held["away_score"]).to_numpy()
    return render(request, "backtest.html", {
        "active": "backtest", "seasons": seasons, "calibration": calibration, "from": season_label(HELD_OUT_FROM),
        "to": season_label(int(held["season"].max())), "margin_sd": f"{MARGIN_SD:g}",
        "headline": {"games": len(held), "model": f"{held['model_ll'].mean():.4f}", "elo": f"{held['elo_ll'].mean():.4f}",
                     "model_typical": typical_probability(held["model_ll"].mean()),
                     "elo_typical": typical_probability(held["elo_ll"].mean()),
                     "favorite": f"{((held['model_home'] > 0.5) == won_held).mean() * 100:.1f}%"}})
