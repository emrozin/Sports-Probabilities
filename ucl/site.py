"""The European cups' pages (included by app.py): the Champions League under /champions-league/, the
Europa League under /europa-league/ and the Conference League under /conference-league/, each with
upcoming games, match pages, results, the league-phase table with every club's odds, team pages and
the track record; plus Europe's club ratings, shared by all three."""

import json
import sqlite3
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

import colors
from web import BASE, ROOT, day_label, disk_cached, edge, odds_text, slugify, templates, typical_probability
from ucl import elo
from ucl.model import EUROPE, LEAGUE_XG_WEIGHT, fit, load_games, load_values, predict, rating, single_game_winner, tie_chance, window
from ucl.predict_upcoming import MODEL_NAME
from ucl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, SUMMARY_PATH
from ucl.cups import CUPS
from ucl.season import RECORD_PATH as ODDS_PATH

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter()
_cache: dict = {}
STAGES = {"league-phase": "League phase", "knockout-round-playoffs": "Knockout play-offs", "round-of-16": "Round of 16",
          "quarterfinals": "Quarter-finals", "semifinals": "Semi-finals", "final": "Final", "group-stage": "Group stage"}
LEAGUE_NAMES = {"eng.1": "England", "esp.1": "Spain", "ger.1": "Germany", "ita.1": "Italy", "fra.1": "France",
                "por.1": "Portugal", "ned.1": "Netherlands", "bel.1": "Belgium", "tur.1": "Turkey", "sco.1": "Scotland",
                "gre.1": "Greece", "aut.1": "Austria", "sui.1": "Switzerland", "den.1": "Denmark", "nor.1": "Norway",
                "swe.1": "Sweden", "isr.1": "Israel", "rou.1": "Romania", "rus.1": "Russia", "other": "Other leagues"}


def connect():
    return closing(sqlite3.connect(DB_PATH))


def has_data() -> bool:
    if not DB_PATH.exists():
        return False
    with connect() as conn:
        return conn.execute("SELECT name FROM sqlite_master WHERE name = 'cup_game_info'").fetchone() is not None


SAVED = "Saved before kickoff"      # the live record's label, as on the Premier League's pages


def season_label(season: int) -> str:
    """2026 -> '2026/27', as on the Premier League's pages."""
    return f"{season}/{(season + 1) % 100:02d}"


def stage_label(stage, leg=None) -> str:
    text = STAGES.get(stage, str(stage or "").replace("-", " ").capitalize())
    if leg is not None and pd.notna(leg) and stage not in ("league-phase", "final"):
        text += f", {'first' if int(leg) == 1 else 'second'} leg"
    return text


def team_url(name: str, cup: str = "UCL") -> str:
    return f"{BASE}{CUPS[cup]['url']}team/{slugify(name)}/"


def match_url(espn_id, cup: str = "UCL") -> str:
    return f"{BASE}{CUPS[cup]['url']}match/{espn_id}/"


def pct(p) -> int:
    return int(round(float(p) * 100))


def percentages(home, draw, away) -> dict:
    values = [home, draw, away]
    rounded = [round(v * 100) for v in values]
    rounded[int(np.argmax(values))] += 100 - sum(rounded)
    return dict(zip(("home", "draw", "away"), rounded))


# ---------------------------------------------------------------- the data

def data() -> dict:
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if _cache.get("stamp") == stamp and "games" in _cache:
        return _cache
    _cache.clear()
    _cache["stamp"] = stamp
    empty = {"games": pd.DataFrame(), "soccer": pd.DataFrame(), "teams": pd.DataFrame(), "colors": {},
             "predictions": pd.DataFrame(), "market": pd.DataFrame()}
    if not has_data():
        _cache.update(empty)
        return _cache
    with connect() as conn:
        games = pd.read_sql("""SELECT g.id, l.code AS cup, g.season, g.date, g.kickoff, g.status, g.home_score, g.away_score,
                                      h.name AS home, a.name AS away, hs.espn_id AS home_id, aw.espn_id AS away_id,
                                      i.espn_id, i.stage, i.leg, i.neutral, i.home_pens, i.away_pens, i.cancelled
                               FROM games g JOIN leagues l ON l.id = g.league_id AND l.code IN ('UCL', 'UEL', 'UECL', 'ESP', 'GER', 'ITA', 'FRA')
                               JOIN cup_game_info i ON i.game_id = g.id
                               JOIN cup_teams hs ON hs.team_id = g.home_team_id AND hs.league_id = g.league_id
                               JOIN cup_teams aw ON aw.team_id = g.away_team_id AND aw.league_id = g.league_id
                               JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                               ORDER BY g.date, g.kickoff""", conn)
        soccer = load_games(conn)
        values = load_values(conn)
        teams = pd.read_sql("SELECT * FROM soccer_teams", conn)
        preds = pd.read_sql("""SELECT p.game_id AS id, p.model_name, p.model_version,
                                      MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
                                      MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS model_draw,
                                      MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS model_away,
                                      MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
                               FROM predictions p LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                    AND pg.model_name = p.model_name AND pg.model_version = p.model_version
                               WHERE p.model_name IN (?, ?) GROUP BY p.game_id, p.model_name, p.model_version""",
                            conn, params=(MODEL_NAME, BACKTEST_NAME))
        market = pd.read_sql("""SELECT game_id AS id, outcome, price, is_closing FROM odds
                                WHERE bookmaker = 'DraftKings' AND market = '1x2'""", conn)
    ext = soccer[["event_id", "extra_time"]].rename(columns={"event_id": "espn_id"})
    games = games.merge(ext, on="espn_id", how="left")
    live = preds[(preds["model_name"] == MODEL_NAME)].sort_values("model_version").drop_duplicates("id", keep="last")
    bt = preds[(preds["model_name"] == BACKTEST_NAME) & (preds["model_version"] == BACKTEST_VERSION)]
    bt = bt[~bt["id"].isin(live["id"])]
    predictions = pd.concat([live.assign(source=SAVED), bt.assign(source="Backtest")], ignore_index=True)
    fair = pd.DataFrame()
    if not market.empty:
        close = market[market["is_closing"] == 1]
        wide = close.pivot_table(index="id", columns="outcome", values="price")
        if {"home", "draw", "away"} <= set(wide.columns):
            implied = 1 / wide[["home", "draw", "away"]]
            fair = implied.div(implied.sum(axis=1), axis=0).add_prefix("mkt_").reset_index()
    _cache.update({"games": games, "soccer": soccer, "values": values, "teams": teams, "predictions": predictions, "market": fair,
                   "colors": {r.name: tuple(f"#{c}" for c in (r.color, r.alternate_color) if isinstance(c, str) and len(c) == 6)
                              for r in teams.itertuples() if isinstance(r.name, str)}})
    return _cache


def team_colors() -> dict:
    return data().get("colors", {})


templates.env.globals["cup_team_url"] = team_url
templates.env.globals["ucl_colors"] = lambda home, away: colors.style(team_colors(), home, away)
templates.env.globals["ucl_team_colors"] = lambda team: colors.team_style(team_colors(), team)
templates.env.globals["ucl_chip"] = lambda team: colors.chip(team_colors(), team)


def current_model(league: bool = False) -> dict | None:
    """Today's ratings (kept on disk until the next update). league: the model for the big leagues'
    games, which learns from xG as well as goals (ucl/tune_xg.py); the cups' learns from goals."""
    d = data()
    if d["soccer"].empty:
        return None
    today = date.today().isoformat()
    key = "league_model" if league else "model"
    if _cache.get(key + "_day") != today:
        def make():
            train = window(d["soccer"], today)
            if train.empty:
                return None
            return elo.attach(fit(train, today, values=d["values"], xg_weight=LEAGUE_XG_WEIGHT if league else None), d["soccer"])
        _cache[key] = disk_cached(DB_PATH.parent / "cache", "ucl_" + key, (d["stamp"], today, "v4"), make)
        _cache[key + "_day"] = today
    return _cache[key]


def cup_games(cup: str) -> pd.DataFrame:
    g = data()["games"]
    return g[g["cup"] == cup] if not g.empty else g


def current_season(cup: str = "UCL") -> int | None:
    g = cup_games(cup)
    return int(g["season"].max()) if not g.empty else None


def name_of(espn_id) -> str:
    t = data()["teams"]
    found = t.loc[t["team_id"] == str(espn_id), "name"]
    return found.iloc[0] if len(found) else str(espn_id)


def short_names() -> dict:
    """Each club's short name (PSG, Dortmund), for tables on phones."""
    t = data()["teams"]
    if t.empty or "short_name" not in t:
        return {}
    return {n: s for n, s in zip(t["name"], t["short_name"]) if isinstance(s, str) and s}


def render(request: Request, template: str, context: dict, cup: str = "UCL", status_code: int = 200):
    return templates.TemplateResponse(request, f"ucl/{template}", {"sport": "europe", "cup": CUPS[cup], "soccer_code": cup,
                                                                   **context}, status_code=status_code)


# ---------------------------------------------------------------- cards

def result_of(row) -> str:
    if row.get("extra_time") == 1 or (pd.notna(row.get("home_pens")) and row["home_score"] == row["away_score"]):
        return "draw"
    return "home" if row["home_score"] > row["away_score"] else "away" if row["away_score"] > row["home_score"] else "draw"


def card(row, model: dict | None = None) -> dict:
    """One game as a card: chances (saved prediction, or today's model for games beyond the
    prediction window), market, and the result once played."""
    finished = row["status"] == "final"
    cup = row["cup"]
    c = {"home": row["home"], "away": row["away"], "url": match_url(row["espn_id"], cup), "date": day_label(row["date"]),
         "time": kickoff_text(row["kickoff"]), "model": None,
         "stage": stage_label(row["stage"], row["leg"]) if CUPS[cup]["kind"] == "cup" else "",
         "market": None, "edge": None, "source": None, "score": None, "result": None, "over": None, "top_scores": [],
         "tie": None, "teams": f"{cup.lower()}:{row['home']}|{cup.lower()}:{row['away']}", "cup": cup}
    if pd.notna(row.get("model_home")):
        c["model"] = percentages(row["model_home"], row["model_draw"], row["model_away"])
        c["source"] = row.get("source")
        exp = (row.get("exp_home"), row.get("exp_away"))
    elif model is not None and not finished:
        p = predict(model, row["home_id"], row["away_id"], bool(row["neutral"]), CUPS[cup]["kind"] == "cup")
        c["model"] = percentages(p["home"], p["draw"], p["away"])
        c["source"] = "Today's ratings"
        exp = (p["exp_home"], p["exp_away"])
    else:
        exp = (None, None)
    if pd.notna(row.get("mkt_home")):
        c["market"] = percentages(row["mkt_home"], row["mkt_draw"], row["mkt_away"])
    if c["model"] and c["market"] and not finished:
        c["edge"] = edge(c["model"], c["market"], {"home": row["home"], "away": row["away"]})
    if exp[0] is not None and pd.notna(exp[0]):
        from epl.model import goal_markets, scoreline_grid
        grid = scoreline_grid(exp[0], exp[1])
        g = goal_markets(grid, top=3)
        c["over"] = pct(g["over_2.5"])
        c["top_scores"] = [{"score": f"{i}–{j}", "pct": f"{p * 100:.0f}"} for i, j, p in g["top_scores"]]
        if finished and max(int(row["home_score"]), int(row["away_score"])) < grid.shape[0]:
            c["exact_pct"] = f"{grid[int(row['home_score']), int(row['away_score'])] * 100:.1f}"
    if finished:
        c["score"] = f"{int(row['home_score'])}–{int(row['away_score'])}"
        c["home_score"], c["away_score"] = int(row["home_score"]), int(row["away_score"])
        c["result"] = result_of(row)
        if pd.notna(row.get("home_pens")):
            c["pens"] = f"{int(row['home_pens'])}–{int(row['away_pens'])} on penalties"
        elif row.get("extra_time") == 1:
            c["pens"] = "after extra time"
        if c["model"]:
            c["model_actual"] = c["model"][c["result"]]
        if c["market"]:
            c["market_actual"] = c["market"][c["result"]]
    return c


def kickoff_text(kickoff) -> str:
    from overview import kickoff_text as text
    return text(kickoff)


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    d = data()
    out = games.merge(d["predictions"], on="id", how="left")
    if not d["market"].empty:
        out = out.merge(d["market"], on="id", how="left")
    return out


def tie_note(row, games: pd.DataFrame, model: dict | None) -> str | None:
    """For a knockout tie: each side's chance of going through (first legs: before the tie; second
    legs: given the first leg's score)."""
    if model is None or row["stage"] in ("league-phase", "group-stage") or pd.isna(row["leg"]):
        return None
    if int(row["leg"]) == 1:
        p = tie_chance(model, row["home_id"], row["away_id"])
        first = row["home"]
        other = row["away"]
    else:
        first_leg = games[(games["stage"] == row["stage"]) & (games["season"] == row["season"]) & (games["home_id"] == row["away_id"])
                          & (games["away_id"] == row["home_id"]) & (games["status"] == "final")]
        score = (int(first_leg.iloc[0]["home_score"]), int(first_leg.iloc[0]["away_score"])) if len(first_leg) else None
        p = tie_chance(model, row["away_id"], row["home_id"], score)
        first, other = row["away"], row["home"]
    return f"To go through: {first} {pct(p)}%, {other} {pct(1 - p)}%"


# ---------------------------------------------------------------- pages

def upcoming_page(request: Request, cup: str):
    games = cup_games(cup)
    model = current_model(CUPS[cup]["kind"] == "league")
    today = date.today().isoformat()
    days = []
    if not games.empty:
        ahead = games[(games["status"] == "scheduled") & (games["cancelled"] == 0) & (games["date"] >= today)]
        ahead = ahead[ahead["date"] <= (pd.Timestamp(today) + pd.Timedelta(days=21)).strftime("%Y-%m-%d")]
        ahead = with_predictions(ahead)
        for day, part in ahead.groupby("date"):
            cards = []
            for _, r in part.sort_values(["kickoff", "home"]).iterrows():
                c = card(r, model)
                c["tie"] = tie_note(r, games, model)
                cards.append(c)
            days.append({"label": day_label(day), "cards": cards})
    return render(request, "upcoming.html", {"active": "upcoming", "days": days, "has_data": not games.empty}, cup)


def results_page(request: Request, cup: str, season: int | None = None):
    games = cup_games(cup)
    if games.empty:
        return render(request, "results.html", {"active": "results", "days": [], "seasons": [], "season": None}, cup)
    season = season or current_season(cup)
    played = games[(games["season"] == season) & (games["status"] == "final")]
    if played.empty and season != current_season(cup):
        raise HTTPException(status_code=404)
    played = with_predictions(played)
    days = [{"label": day_label(day, with_year=True), "cards": [card(r) for _, r in part.sort_values(["kickoff", "home"]).iterrows()]}
            for day, part in sorted(played.groupby("date"), reverse=True)]
    seasons = sorted(games.loc[games["status"] == "final", "season"].unique(), reverse=True)
    return render(request, "results.html", {"active": "results", "days": days, "season": season, "season_name": season_label(season),
                                            "seasons": [{"label": season_label(s), "url": f"{BASE}{CUPS[cup]['url']}results/?season={s}",
                                                         "selected": s == season} for s in seasons]}, cup)


def standings(games: pd.DataFrame, season: int) -> pd.DataFrame:
    lp = games[(games["season"] == season) & (games["stage"] == "league-phase")]
    teams = sorted(set(lp["home"]) | set(lp["away"]))
    rows = {t: {"team": t, "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "away_gf": 0, "away_won": 0}
            for t in teams}
    for g in lp[lp["status"] == "final"].itertuples():
        for side, other, h, a in (("home", "away", g.home_score, g.away_score), ("away", "home", g.away_score, g.home_score)):
            r = rows[getattr(g, side)]
            r["played"] += 1
            r["gf"] += h
            r["ga"] += a
            r["won"] += h > a
            r["drawn"] += h == a
            r["lost"] += h < a
            if side == "away":
                r["away_gf"] += h
                r["away_won"] += h > a
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df = df.astype({c: int for c in ("played", "won", "drawn", "lost", "gf", "ga", "away_gf", "away_won")})
    df["gd"] = df["gf"] - df["ga"]
    df["points"] = 3 * df["won"] + df["drawn"]
    return df.sort_values(["points", "gd", "gf", "away_gf", "won", "away_won", "team"],
                          ascending=[False] * 6 + [True]).reset_index(drop=True)


def season_odds(season: int, cup: str = "UCL") -> pd.DataFrame:
    if not ODDS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ODDS_PATH, dtype={"team_id": str, "date": str})
    df = df[(df["season"] == season) & (df["competition"] == cup)]
    return df[df["date"] == df["date"].max()] if not df.empty else df


def league_table_page(request: Request, cup: str):
    """A league's table so far, and each club's chances from the season simulation."""
    season = current_season(cup)
    if season is None:
        return render(request, "league_table.html", {"active": "table", "rows": [], "season_name": None}, cup)
    part = cup_games(cup)
    part = part[part["season"] == season]
    table = standings(part.assign(stage="league-phase"), season)
    if not table.empty:                              # leagues break ties on goal difference, then goals scored
        table = table.sort_values(["points", "gd", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    odds = season_odds(season, cup)
    by_id = odds.set_index("team_id").to_dict("index") if not odds.empty else {}
    ids = dict(zip(part["home"], part["home_id"].astype(str))) | dict(zip(part["away"], part["away_id"].astype(str)))
    fmt = odds_text
    n = len(table)
    short = short_names()
    rows = []
    for k, r in table.iterrows():
        o = by_id.get(ids.get(r["team"]), {})
        rows.append({"pos": k + 1, **{c: r[c] for c in ("team", "played", "won", "drawn", "lost", "gf", "ga", "points")},
                     "gd": f"{r['gd']:+d}" if r["gd"] else "0", "url": team_url(r["team"], cup), "short": short.get(r["team"], r["team"]),
                     "zone": "top4" if k < 4 else "bottom" if k >= n - 3 else "mid",
                     "odds": {c: fmt(o.get(c)) for c in ("champion", "top4", "bottom3")},
                     "exp_points": f"{o['points']:.0f}" if o else "–"})
    favorites = sorted(((t, v) for t, v in by_id.items() if v.get("champion", 0) >= 0.005), key=lambda kv: -kv[1]["champion"])[:6]
    return render(request, "league_table.html", {"active": "table", "rows": rows, "season_name": season_label(season),
                                                 "odds_date": day_label(odds["date"].iloc[0]) if not odds.empty else None,
                                                 "games_each": 2 * (n - 1),
                                                 "favorites": [{"team": name_of(t), "pct": fmt(v["champion"]), "url": team_url(name_of(t), cup)}
                                                               for t, v in favorites]}, cup)


def table_page(request: Request, cup: str):
    if CUPS[cup]["kind"] == "league":
        return league_table_page(request, cup)
    d = data()
    season = current_season(cup)
    if season is None:
        return render(request, "table.html", {"active": "table", "rows": [], "season_name": None}, cup)
    table = standings(cup_games(cup), season)
    odds = season_odds(season, cup)
    names = dict(zip(d["teams"]["team_id"], d["teams"]["name"]))
    by_name = {}
    if not odds.empty:
        odds = odds.assign(team=odds["team_id"].map(names).fillna(odds["team"]))
        by_name = odds.set_index("team").to_dict("index")
    fmt = odds_text
    rows = []
    short = short_names()
    for k, r in table.iterrows():
        o = by_name.get(r["team"], {})
        rows.append({"pos": k + 1, **{c: r[c] for c in ("team", "played", "won", "drawn", "lost", "gf", "ga", "points")},
                     "gd": f"{r['gd']:+d}" if r["gd"] else "0", "url": team_url(r["team"], cup), "short": short.get(r["team"], r["team"]),
                     "zone": "top8" if k < 8 else "playoff" if k < 24 else "out",
                     "odds": {c: fmt(o.get(c)) for c in ("top8", "top24", "round_of_16", "quarterfinals", "semifinals", "final", "champion")},
                     "exp_points": f"{o['points']:.1f}" if o else "–"})
    favorites = sorted(by_name.items(), key=lambda kv: -kv[1].get("champion", 0))[:8]
    return render(request, "table.html", {"active": "table", "rows": rows, "season_name": season_label(season),
                                          "odds_date": day_label(odds["date"].iloc[0]) if not odds.empty else None,
                                          "games_each": int(round(2 * len(cup_games(cup).query("season == @season and stage == 'league-phase'")) / max(len(table), 1))),
                                          "favorites": [{"team": t, "pct": fmt(v["champion"]), "url": team_url(t, cup)} for t, v in favorites]}, cup)


def match_page(request: Request, cup: str, espn_id: str):
    from app import heatmap
    d = data()
    games = cup_games(cup)
    row = games[games["espn_id"] == espn_id]
    if row.empty:
        raise HTTPException(status_code=404)
    r = with_predictions(row).iloc[0]
    model = current_model(CUPS[cup]["kind"] == "league")
    c = card(r, model)
    finished = r["status"] == "final"
    goals = None
    if pd.notna(r.get("exp_home")) or (model is not None and not finished):
        if pd.notna(r.get("exp_home")):
            eh, ea = float(r["exp_home"]), float(r["exp_away"])
        else:
            p = predict(model, r["home_id"], r["away_id"], bool(r["neutral"]), CUPS[cup]["kind"] == "cup")
            eh, ea = p["exp_home"], p["exp_away"]
        from epl.model import goal_markets, scoreline_grid
        grid = scoreline_grid(eh, ea)
        g = goal_markets(grid)
        actual = (int(r["home_score"]), int(r["away_score"])) if finished else None
        goals = {"exp_home": f"{eh:.2f}", "exp_away": f"{ea:.2f}", "heatmap": heatmap(grid, actual), "btts": pct(g["btts"]),
                 "lines": [{"line": line, "over": pct(g[f"over_{line}"]), "under": pct(g[f"under_{line}"])} for line in (1.5, 2.5, 3.5)]}
    tie = tie_note(r, games, model) if not finished else None
    final_chance = None
    if r["stage"] == "final" and model is not None and not finished:
        final_chance = pct(single_game_winner(model, r["home_id"], r["away_id"], bool(r["neutral"])))
    soccer = d["soccer"]
    form = {side: recent_games(soccer, r[f"{side}_id"], r["date"]) for side in ("home", "away")}
    ratings = None
    if model is not None:
        ratings = [{"team": r[side], "rating": f"{rating(model, r[side + '_id']):+.2f}",
                    "league": LEAGUE_NAMES.get(league_of(model, r[side + "_id"]), "Other leagues")} for side in ("home", "away")]
    meetings = games[(((games["home_id"] == r["home_id"]) & (games["away_id"] == r["away_id"]))
                      | ((games["home_id"] == r["away_id"]) & (games["away_id"] == r["home_id"])))
                     & (games["status"] == "final") & (games["date"] < r["date"])].sort_values("date", ascending=False)
    h2h = [{"date": day_label(g["date"], with_year=True), "home": g["home"], "away": g["away"],
            "score": f"{int(g['home_score'])}–{int(g['away_score'])}", "url": match_url(g["espn_id"], cup),
            "result": result_of(g), "stage": stage_label(g["stage"]) if CUPS[cup]["kind"] == "cup" else season_label(g["season"])}
           for _, g in meetings.head(8).iterrows()]
    return render(request, "match.html", {"m": c, "finished": finished, "goals": goals, "tie": tie, "final_chance": final_chance,
                                          "form_home": form["home"], "form_away": form["away"], "ratings": ratings,
                                          "h2h": h2h, "date": day_label(r["date"], with_year=True)}, cup)


def league_of(model: dict, espn_id) -> str:
    r = model["ratings"]
    return r.at[espn_id, "league"] if espn_id in r.index else "other"


def recent_games(soccer: pd.DataFrame, espn_id, before: str, n: int = 6) -> list[dict]:
    """A club's last games in any competition: domestic and European."""
    part = soccer[((soccer["home_id"] == espn_id) | (soccer["away_id"] == espn_id)) & (soccer["status"] == "final") & (soccer["date"] < before)]
    out = []
    for g in part.sort_values("date").tail(n).itertuples():
        home = g.home_id == espn_id
        us, them = (g.home_score, g.away_score) if home else (g.away_score, g.home_score)
        out.append({"outcome": "W" if us > them else "L" if us < them else "D", "score": f"{int(us)}–{int(them)}",
                    "venue": "H" if home else "A", "opponent": name_of(g.away_id if home else g.home_id),
                    "competition": "Europe" if g.competition in EUROPE else LEAGUE_NAMES.get(g.competition, g.competition),
                    "date": day_label(g.date)})
    return out


def team_page(request: Request, cup: str, slug: str):
    d = data()
    games = cup_games(cup)
    names = pd.concat([games[["home", "home_id"]].set_axis(["name", "id"], axis=1), games[["away", "away_id"]].set_axis(["name", "id"], axis=1)]).drop_duplicates()
    found = names[names["name"].map(slugify) == slug]
    if found.empty:
        raise HTTPException(status_code=404)
    name, espn_id = found.iloc[0]["name"], found.iloc[0]["id"]
    model = current_model(CUPS[cup]["kind"] == "league")
    season = current_season(cup)
    mine = games[((games["home_id"] == espn_id) | (games["away_id"] == espn_id))]
    this = with_predictions(mine[mine["season"] == season])
    cards = [card(r, model) for _, r in this.sort_values("date").iterrows()]
    history = mine[mine["status"] == "final"].groupby("season").size()
    info = None
    if model is not None:
        info = {"rating": f"{rating(model, espn_id):+.2f}", "league": LEAGUE_NAMES.get(league_of(model, espn_id), "Other leagues"),
                "rank": int((pd.Series({t: rating(model, t) for t in model["ratings"].index}) > rating(model, espn_id)).sum()) + 1}
    odds = season_odds(season, cup) if season else pd.DataFrame()
    my_odds = odds[odds["team_id"] == espn_id].to_dict("records") if not odds.empty else []
    return render(request, "team.html", {"team": name, "cards": cards, "info": info, "season_name": season_label(season) if season else "",
                                         "recent": recent_games(d["soccer"], espn_id, "9999-12-31", 10)[::-1],
                                         "odds": {k: {"✓": "100%", "–": "0%"}.get(odds_text(v), odds_text(v)) for k, v in my_odds[0].items()
                                                  if k in ("top8", "top24", "round_of_16", "quarterfinals", "semifinals", "final", "champion", "top4", "bottom3")
                                                  and pd.notna(v)} if my_odds else None,
                                         "seasons_played": len(history),
                                         "other_cups": [{"name": CUPS[c]["name"], "url": team_url(name_c, c)} for c, name_c in other_cups(espn_id, cup)]}, cup)


def ratings_page(request: Request, cup: str):
    model = current_model(CUPS[cup]["kind"] == "league")
    if model is None:
        return render(request, "ratings.html", {"active": "ratings", "clubs": [], "leagues": []}, cup)
    r = model["ratings"]
    recent = r[r["games"] >= 10]
    values = pd.Series({t: rating(model, t) for t in recent.index}).sort_values(ascending=False)
    this = cup_games(cup)
    this = this[this["season"] == current_season(cup)]
    in_cup = {t: n for t, n in zip(this["home_id"], this["home"])} | {t: n for t, n in zip(this["away_id"], this["away"])}
    clubs = [{"rank": k + 1, "team": name_of(t), "rating": f"{v:+.2f}", "league": LEAGUE_NAMES.get(r.at[t, "league"], "Other leagues"),
              "in_cup": t in in_cup, "url": team_url(in_cup[t], cup) if t in in_cup else None}
             for k, (t, v) in enumerate(values.head(60).items())]
    by_league = pd.DataFrame({"league": r.loc[values.index, "league"], "rating": values})
    leagues = by_league.groupby("league")["rating"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
    leagues = [{"league": LEAGUE_NAMES.get(g, g), "average": f"{row['mean']:+.2f}", "best": f"{row['max']:+.2f}", "clubs": int(row["size"])}
               for g, row in leagues.iterrows()]
    return render(request, "ratings.html", {"active": "ratings", "clubs": clubs, "leagues": leagues,
                                            "home_europe": f"{np.exp(model['home_europe']) * 100 - 100:.0f}",
                                            "as_of": day_label(model["as_of"])}, cup)


def track_record_page(request: Request, cup: str):
    from charts import calibration_chart
    games = cup_games(cup)
    rows, summary, calib = [], None, None
    if not games.empty:
        done = with_predictions(games[games["status"] == "final"])
        live = done[done["source"] == SAVED].copy()
        if not live.empty:
            live["result"] = [result_of(r) for _, r in live.iterrows()]
            p = np.array([r[f"model_{r['result']}"] for _, r in live.iterrows()])
            loss = float(-np.mean(np.log(np.clip(p, 1e-9, 1))))
            with_market = live.dropna(subset=["mkt_home"]) if "mkt_home" in live else live.iloc[0:0]
            mloss = (float(-np.mean(np.log([r[f"mkt_{r['result']}"] for _, r in with_market.iterrows()])))
                     if not with_market.empty else None)
            mloss_model = (float(-np.mean(np.log([r[f"model_{r['result']}"] for _, r in with_market.iterrows()])))
                           if not with_market.empty else None)
            summary = {"games": len(live), "loss": f"{loss:.3f}", "typical": typical_probability(loss),
                       "market_games": len(with_market), "market_loss": f"{mloss:.3f}" if mloss else None,
                       "model_loss_vs_market": f"{mloss_model:.3f}" if mloss_model else None}
            rows = [card(r) for _, r in live.sort_values("date", ascending=False).head(40).iterrows()]
        past = done[(done["source"] == "Backtest") & (done["season"] >= HELD_OUT_FROM)].copy()
        three = lambda f: (f[["model_home", "model_draw", "model_away"]].to_numpy(float).ravel(),
                           np.column_stack([[result_of(r) == o for _, r in f.iterrows()] for o in ("home", "draw", "away")]).astype(float).ravel())
        if not live.empty or not past.empty:
            calib = calibration_chart(three(live) if not live.empty else (np.array([]), np.array([])),
                                      three(past) if not past.empty else None, two_way=False,
                                      past_from=season_label(HELD_OUT_FROM))
    backtest = None
    if SUMMARY_PATH.exists():
        everything = json.loads(SUMMARY_PATH.read_text())
        backtest = everything.get("competitions", {}).get(cup)
        if backtest:
            for s in backtest["seasons"]:
                s["label"] = season_label(s["season"])
            h = backtest["held_out"]
            backtest["typical"] = typical_probability(h["model"]) if h.get("model") else None
            backtest["held_label"] = season_label(everything["held_out_from"])
            backtest["first_label"] = season_label(min(s["season"] for s in backtest["seasons"]))
    return render(request, "track_record.html", {"active": "track", "summary": summary, "rows": rows, "calibration": calib,
                                                 "backtest": backtest}, cup)


def other_cups(espn_id, cup: str) -> list[tuple[str, str]]:
    """The other cups a club is in this season: (cup code, its name there)."""
    out = []
    for c in CUPS:
        if c == cup:
            continue
        g = cup_games(c)
        g = g[g["season"] == current_season(c)] if not g.empty else g
        found = g.loc[g["home_id"] == espn_id, "home"]
        found = found if len(found) else g.loc[g["away_id"] == espn_id, "away"]
        if len(found):
            out.append((c, found.iloc[0]))
    return out


def register(code: str) -> None:
    """Each cup's pages under its own address."""
    prefix = CUPS[code]["url"].rstrip("/")

    def upcoming(request: Request):
        return upcoming_page(request, code)

    def results(request: Request, season: int | None = None):
        return results_page(request, code, season)

    def table(request: Request):
        return table_page(request, code)

    def match(request: Request, espn_id: str):
        return match_page(request, code, espn_id)

    def team(request: Request, slug: str):
        return team_page(request, code, slug)

    def ratings(request: Request):
        return ratings_page(request, code)

    def track(request: Request):
        return track_record_page(request, code)
    for path, endpoint in (("/", upcoming), ("/results/", results), ("/table/", table), ("/match/{espn_id}/", match),
                           ("/team/{slug}/", team), ("/ratings/", ratings), ("/track-record/", track)):
        router.add_api_route(prefix + path, endpoint, methods=["GET"], response_class=HTMLResponse)


for _code in CUPS:
    register(_code)
