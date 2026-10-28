"""NFL pages of the website, under /nfl/: upcoming games, results, standings, team and game pages,
track record, backtest and team ratings. app.py includes them.

Games are listed the American way, away team first ("SEA at DEN"); in the probability bars
the away team is gold on the left and the home team blue on the right.
"""

import sqlite3
from contextlib import closing
from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from nfl.elo import load_nfl_games
from nfl.model import (HALF_LIFE_DAYS, INEXPERIENCE_STARTS, MARGIN_SD, TRAINING_YEARS, add_qb_changes, fit, injury_effect,
                       missing, qb_value)
from nfl.starters import add_to_games, missing_starters
from nfl.predict_upcoming import MODEL_NAME
from nfl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
from web import BASE, ROOT, calibration_pairs, day_label, edge, ordinal, signed_number, slugify, templates, typical_probability, why
import betting
import blend

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter(prefix="/nfl")

RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/nfl_live_predictions.csv"
RECORD_HISTORY_URL = "https://github.com/emrozin/Sports-Probabilities/commits/main/records/nfl_live_predictions.csv"
PLAYER_RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/nfl_player_projections.csv"

# Held-out results (2018-2025, 2,219 games, ties excluded) from nfl/tune_model.py, October 2026.
DEVELOPMENT = [
    ("Elo ratings (baseline)", 0.6382),
    ("Points model: offense and defense ratings", 0.6374),
    ("+ usual-starter quarterback flag", 0.6326),
    ("+ individual quarterback ratings, replacing the flag", 0.6298),
    ("+ starters checked against the box scores (current)", 0.6295),
    ("Betting market (moneylines)", 0.6089),
]

# Player projections vs simple benchmarks, held-out seasons 2018-2025 (nfl/players.py), October 2026:
# average miss per player-game.
PLAYER_ACCURACY = [
    ("Passing yards", "61.4", "71.6", "70.0"),
    ("Passing touchdowns", "0.92", "1.00", "0.98"),
    ("Rushing yards", "8.2", "8.9", "8.7"),
    ("Receptions", "1.28", "1.38", "1.34"),
    ("Receiving yards", "17.0", "18.6", "18.0"),
]

from nfl.teams import DIVISIONS_FROM, TEAMS  # noqa: E402  (names, divisions)
PLAYOFF_ROUNDS = {"WC": "Wild Card round", "DIV": "Divisional round", "CON": "Conference championships",
                  "SB": "Super Bowl"}


# ---------------------------------------------------------------- formatting

def team_name(code: str) -> str:
    return TEAMS.get(code, (code,))[0]


def nickname(code: str) -> str:
    """'KC' -> 'Chiefs': the last word of the name ('Commanders', '49ers')."""
    return team_name(code).split()[-1]


def team_url(code: str, season: int | None = None) -> str:
    return f"{BASE}/nfl/team/{slugify(team_name(code))}/" + (f"{season}/" if season is not None else "")


def game_url(source_id: str) -> str:
    """'2026_06_SEA_DEN' -> '/nfl/game/2026-06-sea-den/'"""
    return f"{BASE}/nfl/game/{source_id.lower().replace('_', '-')}/"


templates.env.globals["nfl_team_url"] = team_url
templates.env.globals["nfl_name"] = team_name
templates.env.globals["nfl_nickname"] = lambda code: nickname(code)


def week_label(game_type: str, week: int) -> str:
    return PLAYOFF_ROUNDS.get(game_type, f"Week {week}")


def kickoff_label(kickoff: str | None) -> str | None:
    """'20:15' -> '8:15 PM ET'"""
    if not kickoff or not isinstance(kickoff, str):
        return None
    t = datetime.strptime(kickoff, "%H:%M")
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'} ET"


def two_way(p_home: float) -> dict:
    """Whole-number percentages for away and home that add up to 100."""
    home = int(round(p_home * 100))
    return {"away": 100 - home, "home": home}


def line_text(home_margin: float | None, home: str, away: str, decimals: int = 1) -> str | None:
    """Expected home margin -> 'KC by 3.5' (the favorite) or 'Even'."""
    if home_margin is None or pd.isna(home_margin):
        return None
    if round(abs(home_margin), decimals) == 0:
        return "Even"
    return f"{home if home_margin > 0 else away} by {abs(home_margin):.{decimals}f}"


def no_vig(home_moneyline, away_moneyline) -> float | None:
    """Market home win probability from American moneylines, with the bookmaker's margin removed."""
    if pd.isna(home_moneyline) or pd.isna(away_moneyline):
        return None
    implied = lambda ml: -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)
    h, a = implied(float(home_moneyline)), implied(float(away_moneyline))
    return h / (h + a)


def outcome_letter(team_score: int, opponent_score: int) -> str:
    return "W" if team_score > opponent_score else "L" if team_score < opponent_score else "T"


# ---------------------------------------------------------------- data

def connect():
    return closing(sqlite3.connect(DB_PATH))


def load_live(conn) -> pd.DataFrame:
    """Live predictions from every model version; for each game, the most recently saved one."""
    df = pd.read_sql("""
        SELECT p.game_id AS id, p.model_version, MAX(p.created_at) AS created_at,
               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
               MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
        FROM predictions p
        LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                     AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        WHERE p.model_name = ?
        GROUP BY p.game_id, p.model_version
    """, conn, params=(MODEL_NAME,))
    df = df.sort_values(["created_at", "model_version"]).drop_duplicates("id", keep="last")
    return df[["id", "model_home", "exp_home", "exp_away"]]


def load_predictions(conn, model_name: str, model_version: str) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT p.game_id AS id, MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
               MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
        FROM predictions p
        LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                     AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        WHERE p.model_name = ? AND p.model_version = ?
        GROUP BY p.game_id
    """, conn, params=(model_name, model_version))


def load_closing(conn) -> pd.DataFrame:
    """Closing market home win probability (moneylines, margin removed) for finished games."""
    df = pd.read_sql("""
        SELECT h.game_id AS id, 1 / h.price AS h, 1 / a.price AS a
        FROM odds h JOIN odds a ON a.game_id = h.game_id AND a.market = 'moneyline' AND a.outcome = 'away'
        WHERE h.market = 'moneyline' AND h.outcome = 'home' AND h.bookmaker = 'nflverse'
    """, conn)
    df["mkt_home"] = df["h"] / (df["h"] + df["a"])
    return df[["id", "mkt_home"]]


_cache: dict = {}


GAME_COLUMNS = ["id", "season", "date", "status", "home", "away", "home_score", "away_score", "neutral_site",
                "game_type", "week", "spread_line", "home_qb", "away_qb", "home_qb_change", "away_qb_change",
                "home_qb_expected", "away_qb_expected", "home_qb_starts", "away_qb_starts",
                "kickoff", "source_game_id", "total_line", "home_moneyline", "away_moneyline", "stadium", "overtime"]


def empty_data(stamp) -> dict:
    """No NFL data yet (a database without NFL games): every page shows its empty state."""
    prediction_columns = ["id", "model_home", "exp_home", "exp_away", "source"]
    return {"stamp": stamp, "games": pd.DataFrame(columns=GAME_COLUMNS),
            "predictions": pd.DataFrame(columns=prediction_columns),
            "backtest": pd.DataFrame(columns=prediction_columns), "closing": pd.DataFrame(columns=["id", "mkt_home"])}


def data() -> dict:
    """Everything the NFL pages read, loaded once and reused until the database file changes."""
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if _cache.get("stamp") != stamp or "games" not in _cache:
        has_nfl = False
        if stamp is not None:
            with connect() as conn:
                has_nfl = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_game_info'").fetchone() is not None
        if not has_nfl:
            _cache.clear()
            _cache.update(empty_data(stamp))
            return _cache
        with connect() as conn:
            games = load_nfl_games(conn)
            extra = pd.read_sql("""SELECT i.game_id AS id, g.kickoff, i.source_game_id, i.total_line,
                                          i.home_moneyline, i.away_moneyline, i.stadium, i.overtime
                                   FROM nfl_game_info i JOIN games g ON g.id = i.game_id""", conn)
            games = add_to_games(add_qb_changes(games).merge(extra, on="id"),
                                 missing_starters(conn, since=int(games["season"].max()) if len(games) else 0))
            live = load_live(conn).assign(source="Saved before kickoff")
            back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).assign(source="Backtest")
            predictions = pd.concat([live, back[~back["id"].isin(live["id"])]], ignore_index=True)
            _cache.clear()
            _cache.update(stamp=stamp, games=games, predictions=predictions, backtest=back,
                          closing=load_closing(conn))
    return _cache


def finished_games() -> pd.DataFrame:
    g = data()["games"]
    return g[g["status"] == "final"]


def cached_model() -> dict:
    """Current ratings, fitted on finished games before today; refitted when the data or the date changes."""
    d, today = data(), date.today().isoformat()
    if d.get("model_day") != today:
        played = finished_games()
        start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
        train = played[(played["date"] >= start) & (played["date"] < today)]
        d["model"] = fit(train, today) if not train.empty else None
        if d["model"] is not None:
            d["model"]["games_used"] = len(train)
        d["model_day"] = today
    return d["model"]


def season_options(seasons, selected, url) -> list[dict]:
    return [{"label": str(s), "url": url(s), "selected": s == selected} for s in sorted(seasons, reverse=True)]


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request, f"nfl/{template}", {"sport": "nfl", **context},
                                      status_code=status_code)


# ---------------------------------------------------------------- game cards

def game_card(g, current_team: str | None = None) -> dict:
    """Everything a game row needs: g has game columns plus (optionally) prediction and market columns."""
    finished = g.status == "final"
    home_margin_model = (g.exp_home - g.exp_away) if pd.notna(getattr(g, "exp_home", np.nan)) else None
    card = {
        "home": g.home, "away": g.away, "url": game_url(g.source_game_id),
        "date": day_label(g.date, with_year=True), "kickoff": kickoff_label(g.kickoff),
        "week": week_label(g.game_type, g.week), "playoff": g.game_type != "REG",
        "neutral": bool(g.neutral_site), "finished": finished,
        "model": None, "market": None, "source": getattr(g, "source", None),
        "model_line": line_text(home_margin_model, g.home, g.away),
        "model_total": f"{g.exp_home + g.exp_away:.1f}" if home_margin_model is not None else None,
        "market_line": line_text(g.spread_line, g.home, g.away),
        "market_total": f"{g.total_line:g}" if pd.notna(g.total_line) else None,
    }
    if pd.notna(getattr(g, "model_home", np.nan)):
        card["model"] = two_way(g.model_home)
    market_home = getattr(g, "mkt_home", np.nan)
    if pd.isna(market_home):
        market_home = no_vig(g.home_moneyline, g.away_moneyline)
    if market_home is not None and pd.notna(market_home):
        card["market"] = two_way(market_home)
    card["edge"] = edge(card["model"], card["market"], {"home": g.home, "away": g.away})
    if finished:
        hs, as_ = int(g.home_score), int(g.away_score)
        card["score"] = f"{as_}–{hs}"   # away first, like the team names
        card["home_score"], card["away_score"] = hs, as_
        card["result"] = "home" if hs > as_ else "away" if as_ > hs else "tie"
        card["actual_line"] = line_text(hs - as_, g.home, g.away, decimals=0).replace("Even", "Tie")
        card["actual_total"] = hs + as_
        if card["model"] and card["result"] != "tie":
            card["model_actual"] = card["model"][card["result"]]
            card["market_actual"] = card["market"][card["result"]] if card["market"] else None
        if current_team:
            at_home = g.home == current_team
            card["outcome"] = outcome_letter(hs if at_home else as_, as_ if at_home else hs)
    if current_team:
        card["venue"] = "vs" if g.home == current_team else "at"
        card["opponent"] = g.away if g.home == current_team else g.home
    return card


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    d = data()
    return games.merge(d["predictions"], on="id", how="left").merge(d["closing"], on="id", how="left")


def summary(df: pd.DataFrame) -> dict | None:
    """Typical probability for the actual winner, model vs market, on finished non-tied games that have both."""
    both = df[(df["status"] == "final") & (df["home_score"] != df["away_score"])].dropna(subset=["model_home", "mkt_home"])
    if both.empty:
        return None
    home_won = both["home_score"] > both["away_score"]
    model_p = np.where(home_won, both["model_home"], 1 - both["model_home"])
    market_p = np.where(home_won, both["mkt_home"], 1 - both["mkt_home"])
    return {"games": len(both),
            "model_typical": typical_probability(-np.log(model_p).mean()),
            "market_typical": typical_probability(-np.log(market_p).mean()),
            "model_favorite_won": int((model_p > 0.5).sum()),
            "model_closer": int((model_p > market_p).sum())}


# ---------------------------------------------------------------- pages

@router.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    d = data()
    today = date.today().isoformat()
    live = d["predictions"][d["predictions"]["source"] == "Saved before kickoff"]
    games = (d["games"][(d["games"]["status"] == "scheduled") & (d["games"]["date"] >= today)]
             .merge(live, on="id").sort_values(["date", "kickoff", "home"]))
    picks = likeliest([int(i) for i in games["id"]], "date", PROPS_PER_DAY)
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()],
             "props": prop_rows(picks[picks["date"] == day]) if not picks.empty else []}
            for day, group in games.groupby("date", sort=True)]
    week = week_label(games.iloc[0]["game_type"], int(games.iloc[0]["week"])) if len(games) else None
    return render(request, "upcoming.html", {"active": "upcoming", "days": days, "week": week,
                                            "props_per_day": PROPS_PER_DAY, "props_per_game": PROPS_PER_GAME})


def week_key(season: int, week: int) -> str:
    return f"{BASE}/nfl/results/{season}/{week}/"


@router.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request):
    return results_page(request, None, None)


@router.get("/results/{season}/{week}/", response_class=HTMLResponse)
def results_week(request: Request, season: int, week: int):
    return results_page(request, season, week)


def results_page(request: Request, season: int | None, week: int | None):
    played = with_predictions(finished_games()).dropna(subset=["model_home"])
    if played.empty:
        return render(request, "results.html", {"active": "results", "days": []})
    weeks = sorted(set(zip(played["season"], played["week"])))
    if season is None:
        season, week = weeks[-1]
    elif (season, week) not in weeks:
        raise HTTPException(status_code=404)
    position = weeks.index((season, week))
    wk = played[(played["season"] == season) & (played["week"] == week)].sort_values(["date", "kickoff", "home"])
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in wk.groupby("date", sort=True)]
    return render(request, "results.html", {
        "active": "results", "days": days, "summary": summary(wk),
        "week_label": f"{season} season, {week_label(wk.iloc[0]['game_type'], week)}",
        "previous_url": week_key(*weeks[position - 1]) if position > 0 else None,
        "next_url": week_key(*weeks[position + 1]) if position < len(weeks) - 1 else None,
    })


def standings(season: int) -> pd.DataFrame:
    """Regular-season records for one season: wins, losses, ties, points for and against, last 5."""
    g = finished_games()
    g = g[(g["season"] == season) & (g["game_type"] == "REG")]
    rows = pd.concat([
        pd.DataFrame({"team": g["home"], "date": g["date"], "pf": g["home_score"], "pa": g["away_score"]}),
        pd.DataFrame({"team": g["away"], "date": g["date"], "pf": g["away_score"], "pa": g["home_score"]}),
    ], ignore_index=True)
    rows[["pf", "pa"]] = rows[["pf", "pa"]].astype(int)
    rows["outcome"] = [outcome_letter(pf, pa) for pf, pa in zip(rows["pf"], rows["pa"])]
    table = rows.groupby("team").agg(
        won=("outcome", lambda s: (s == "W").sum()), lost=("outcome", lambda s: (s == "L").sum()),
        tied=("outcome", lambda s: (s == "T").sum()), pf=("pf", "sum"), pa=("pa", "sum"))
    table["played"] = table["won"] + table["lost"] + table["tied"]
    table["pct"] = (table["won"] + 0.5 * table["tied"]) / table["played"]
    table["diff"] = table["pf"] - table["pa"]
    table["form"] = rows.sort_values("date").groupby("team")["outcome"].apply(lambda s: list(s.tail(5)))
    table["division"] = [TEAMS.get(t, ("", ""))[1] for t in table.index]
    return table.reset_index().sort_values(["pct", "diff", "pf"], ascending=False).reset_index(drop=True)


def record_text(r) -> str:
    return f"{r.won}–{r.lost}" + (f"–{r.tied}" if r.tied else "")


# ---------------------------------------------------------------- playoff odds

def odds_text(p: float) -> str:
    """A simulated chance for the page: '<1%' and '>99%' near the ends; '–' if it never happened."""
    if p <= 0:
        return "–"
    if p < 0.005:
        return "<1%"
    if p > 0.995:
        return ">99%"
    return f"{p * 100:.0f}%"


def load_season_odds() -> pd.DataFrame:
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_season_odds'").fetchone() is None:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM nfl_season_odds", conn)


def season_odds() -> dict | None:
    """The latest playoff odds for the latest season that has them, and the ones from about a week
    earlier (for the change column). Cached with the rest of the data."""
    d = data()
    if "season_odds" not in d:
        odds = load_season_odds()
        result = None
        if not odds.empty:
            season = int(odds["season"].max())
            odds = odds[odds["season"] == season]
            dates = sorted(odds["as_of"].unique())
            latest = dates[-1]
            week_ago = [x for x in dates if pd.Timestamp(x) <= pd.Timestamp(latest) - pd.Timedelta(days=6)]
            result = {"season": season, "as_of": latest, "all": odds,
                      "latest": odds[odds["as_of"] == latest].set_index("team"),
                      "before": odds[odds["as_of"] == week_ago[-1]].set_index("team") if week_ago else None,
                      "before_as_of": week_ago[-1] if week_ago else None}
        d["season_odds"] = result
    return d["season_odds"]


@router.get("/playoffs/", response_class=HTMLResponse)
def playoffs_page(request: Request):
    from nfl.season import RATING_SD, SIMULATIONS
    odds = season_odds()
    if odds is None:
        return render(request, "playoffs.html", {"active": "playoffs", "conferences": []})
    latest, before = odds["latest"], odds["before"]
    table = standings(odds["season"]).set_index("team")
    columns = ["division", "playoffs", "top_seed", "conference", "champion"]
    conferences = []
    for conf in ("AFC", "NFC"):
        teams = [t for t in latest.index if TEAMS.get(t, ("", ""))[1].startswith(conf)]
        teams.sort(key=lambda t: (-latest.at[t, "playoffs"], -latest.at[t, "wins"]))
        rows = []
        for t in teams:
            r = latest.loc[t]
            change = None
            if before is not None and t in before.index:
                delta = round((r["playoffs"] - before.at[t, "playoffs"]) * 100)
                change = {"text": signed_number(delta) if delta else "0", "dir": "up" if delta > 0 else "down" if delta < 0 else ""}
            rows.append({"team": t, "name": team_name(t), "url": team_url(t),
                         "record": record_text(table.loc[t]) if t in table.index else "0–0",
                         "wins": f"{r['wins']:.1f}", "change": change,
                         **{c: {"text": odds_text(r[c]), "value": float(r[c])} for c in columns}})
        conferences.append({"label": conf, "rows": rows})
    return render(request, "playoffs.html", {
        "active": "playoffs", "conferences": conferences, "season": odds["season"],
        "as_of": day_label(odds["as_of"]), "before_as_of": day_label(odds["before_as_of"]) if odds["before_as_of"] else None,
        "simulations": f"{SIMULATIONS:,}", "rating_sd": f"{RATING_SD:g}",
    })


def playoff_history(team: str) -> dict | None:
    """A team's playoff chance after each update this season, for the line on its page."""
    from charts import line_chart
    odds = season_odds()
    if odds is None:
        return None
    own = odds["all"][odds["all"]["team"] == team].sort_values("as_of")
    if len(own) < 2:
        return None
    points = [{"label": f"{pd.Timestamp(r.as_of):%-d %b}", "value": r.playoffs * 100 - 50,
               "tip": f"{day_label(r.as_of)}: {odds_text(r.playoffs)} to make the playoffs, {r.wins:.1f} wins expected"}
              for r in own.itertuples()]
    chart = line_chart(points, step=25, minimum=50)
    for tick in chart["ticks"]:
        tick["label"] = f"{tick['value'] + 50:.0f}%"
    return {"chart": chart, "season": odds["season"], "now": odds_text(own.iloc[-1]["playoffs"])}


@router.get("/standings/", response_class=HTMLResponse)
def standings_latest(request: Request):
    return standings_page(request, None)


@router.get("/standings/{season}/", response_class=HTMLResponse)
def standings_season(request: Request, season: int):
    return standings_page(request, season)


def standings_page(request: Request, season: int | None):
    played = finished_games()
    seasons = sorted(played.loc[played["game_type"] == "REG", "season"].unique())
    if not seasons:
        return render(request, "standings.html", {"active": "standings", "groups": []})
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    table = standings(season)

    def row(r):
        return {"team": r.team, "name": team_name(r.team), "url": team_url(r.team, season), "record": record_text(r),
                "won": r.won, "lost": r.lost, "tied": r.tied, "pct": f"{r.pct:.3f}".lstrip("0"),
                "pf": r.pf, "pa": r.pa, "diff": signed_number(r.diff), "form": r.form}

    if season >= DIVISIONS_FROM:
        groups = []
        for conference in ("AFC", "NFC"):
            for division in ("East", "North", "South", "West"):
                name = f"{conference} {division}"
                members = table[table["division"] == name]
                groups.append({"label": name, "rows": [row(r) for r in members.itertuples()]})
    else:
        groups = [{"label": "All teams", "rows": [row(r) for r in table.itertuples()]}]
    return render(request, "standings.html", {
        "active": "standings", "groups": groups, "season": season, "is_current": season == seasons[-1],
        "has_ties": bool(table["tied"].sum()), "by_division": season >= DIVISIONS_FROM,
        "seasons": season_options(seasons, season, lambda s: f"{BASE}/nfl/standings/{s}/"),
    })


def find_team(slug: str) -> str:
    teams = {slugify(team_name(t)): t for t in pd.concat([data()["games"]["home"], data()["games"]["away"]]).unique()}
    if slug not in teams:
        raise HTTPException(status_code=404)
    return teams[slug]


def rating_row(model: dict, team: str) -> dict | None:
    if model is None or team not in model["ratings"].index:
        return None
    r = model["ratings"].loc[team]
    return {"offense": signed_number(r["offense"], 1), "defense": signed_number(r["defense"], 1),
            "overall": signed_number(r["offense"] + r["defense"], 1),
            "offense_class": "good" if r["offense"] > 0 else "bad" if r["offense"] < 0 else "",
            "defense_class": "good" if r["defense"] > 0 else "bad" if r["defense"] < 0 else "",
            "overall_class": "good" if r["offense"] + r["defense"] > 0 else "bad"}


SHORT_WEEKS = {"WC": "WC", "DIV": "DIV", "CON": "CONF", "SB": "SB"}


def season_ratings(season: int) -> pd.DataFrame:
    """Every team's overall rating (offense + defense, points per game vs an average team) at the
    start of a season and after each week played: the model refitted as it stood then.
    Columns: step, label, team, overall, rank. Cached per season (about a third of a second to build)."""
    d = data()
    key = f"season_ratings_{season}"
    if key in d:
        return d[key]
    played = finished_games()
    season_games = played[played["season"] == season]
    rows = []
    if not season_games.empty:
        weeks = (season_games.groupby(["game_type", "week"], sort=False)["date"].agg(["min", "max"])
                 .reset_index().sort_values("min"))
        cutoffs = [("Start", "before the season", weeks.iloc[0]["min"])]
        for w in weeks.itertuples():
            after = (pd.Timestamp(w.max) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            short = SHORT_WEEKS.get(w.game_type, str(w.week))
            cutoffs.append((short, f"after {week_label(w.game_type, w.week).lower()}", after))
        for step, (label, when, cutoff) in enumerate(cutoffs):
            start = (pd.Timestamp(cutoff) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
            train = played[(played["date"] >= start) & (played["date"] < cutoff)]
            if train.empty:
                continue
            r = fit(train, cutoff)["ratings"]
            overall = (r["offense"] + r["defense"]).sort_values(ascending=False)
            for rank, (team, value) in enumerate(overall.items(), 1):
                rows.append({"step": step, "label": label, "when": when, "team": team,
                             "overall": float(value), "rank": rank, "teams": len(overall)})
    d[key] = pd.DataFrame(rows)
    return d[key]


def team_charts(team: str, season: int, played: pd.DataFrame) -> dict:
    """The rating line and the per-game win chance columns for a team page."""
    from charts import chance_bars, line_chart
    ratings = season_ratings(season)
    line = None
    if not ratings.empty:
        own = ratings[ratings["team"] == team].sort_values("step")
        line = line_chart([{"label": r.label, "value": r.overall,
                            "tip": f"{r.when.capitalize()}: {signed_number(r.overall, 1)} points per game, "
                                   f"{ordinal(r.rank)} of {r.teams}"} for r in own.itertuples()])
    bars = []
    for g in played.sort_values("date").itertuples():
        if pd.isna(g.model_home):
            continue
        at_home = g.home == team
        chance = g.model_home if at_home else 1 - g.model_home
        own_score, opp_score = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        outcome = outcome_letter(own_score, opp_score)
        opponent = g.away if at_home else g.home
        word = {"W": "won", "L": "lost", "T": "tied"}[outcome]
        bars.append({"label": SHORT_WEEKS.get(g.game_type, str(g.week)), "chance": chance, "outcome": outcome,
                     "tip": f"{week_label(g.game_type, g.week)}, {'vs' if at_home else 'at'} {opponent}: "
                            f"model {chance * 100:.0f}%, {word} {int(own_score)}–{int(opp_score)}"})
    return {"rating_chart": line, "chance_chart": chance_bars(bars) if bars else None}


@router.get("/team/{slug}/", response_class=HTMLResponse)
def team_latest(request: Request, slug: str):
    return team_page(request, slug, None)


@router.get("/team/{slug}/{season}/", response_class=HTMLResponse)
def team_season(request: Request, slug: str, season: int):
    return team_page(request, slug, season)


def team_page(request: Request, slug: str, season: int | None):
    team = find_team(slug)
    games = data()["games"]
    own = games[(games["home"] == team) | (games["away"] == team)]
    seasons = sorted(own["season"].unique())
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    latest = games["season"].max()
    shown = with_predictions(own[own["season"] == season]).sort_values("date")
    played = shown[shown["status"] == "final"]
    coming = shown[shown["status"] == "scheduled"]

    record = None
    table = standings(season)
    if team in set(table["team"]):
        r = table.set_index("team").loc[team]
        division = r["division"]
        div_rank = None
        if season >= DIVISIONS_FROM:
            div_teams = table[table["division"] == division]["team"].tolist()
            div_rank = ordinal(div_teams.index(team) + 1)
        record = {"text": record_text(r), "pf": int(r["pf"]), "pa": int(r["pa"]), "diff": signed_number(r["diff"]),
                  "division": division if season >= DIVISIONS_FROM else None, "div_rank": div_rank}

    qbs = (pd.concat([played.loc[played["home"] == team, "home_qb"], played.loc[played["away"] == team, "away_qb"]])
           .dropna().value_counts())
    return render(request, "team.html", {
        "active": None, "team": team, "name": team_name(team), "division": TEAMS.get(team, ("", ""))[1],
        "season": season, "is_current": season == latest, "record": record,
        "ratings": rating_row(cached_model(), team) if season == latest else None,
        "quarterbacks": [{"name": q, "starts": int(n)} for q, n in qbs.items()],
        "summary": summary(played),
        "played": [game_card(g, team) for g in played.sort_values("date", ascending=False).itertuples()],
        "coming": [game_card(g, team) for g in coming.dropna(subset=["model_home"]).itertuples()],
        "schedule": [{"date": day_label(g.date, with_year=True), "week": week_label(g.game_type, g.week),
                      "venue": "vs" if g.home == team else "at", "opponent": g.away if g.home == team else g.home,
                      "kickoff": kickoff_label(g.kickoff), "url": game_url(g.source_game_id)}
                     for g in coming[coming["model_home"].isna()].itertuples()],
        "standings_url": f"{BASE}/nfl/standings/{season}/",
        "seasons": season_options(seasons, season, lambda s: team_url(team, s)),
        **team_charts(team, season, played),
        "playoff_chart": playoff_history(team) if season == latest else None,
    })


def form(games: pd.DataFrame, team: str, before: str, n: int = 5) -> list[dict]:
    """A team's last n games before a date, oldest first."""
    own = games[((games["home"] == team) | (games["away"] == team)) & (games["date"] < before)
                & (games["status"] == "final")]
    rows = []
    for g in own.sort_values("date").tail(n).itertuples():
        at_home = g.home == team
        ts, os_ = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        rows.append({"outcome": outcome_letter(ts, os_), "score": f"{int(ts)}–{int(os_)}",
                     "venue": "vs" if at_home else "at", "opponent": g.away if at_home else g.home,
                     "url": game_url(g.source_game_id)})
    return rows


def head_to_head(games: pd.DataFrame, a: str, b: str, before: str) -> dict:
    """Every earlier meeting of two teams (since 1999), most recent first, with the series record."""
    met = games[(((games["home"] == a) & (games["away"] == b)) | ((games["home"] == b) & (games["away"] == a)))
                & (games["date"] < before) & (games["status"] == "final")].sort_values("date", ascending=False)
    wins = {a: 0, b: 0}
    ties = 0
    meetings = []
    for g in met.itertuples():
        winner = g.home if g.home_score > g.away_score else g.away if g.away_score > g.home_score else None
        if winner:
            wins[winner] += 1
        else:
            ties += 1
        meetings.append({"date": day_label(g.date, with_year=True), "season": g.season,
                         "week": week_label(g.game_type, g.week), "home": g.home, "away": g.away,
                         "score": f"{int(g.away_score)}–{int(g.home_score)}", "winner": winner,
                         "url": game_url(g.source_game_id)})
    return {"meetings": meetings, "wins_a": wins[a], "wins_b": wins[b], "ties": ties,
            "a": a, "b": b, "first": int(met["season"].min()) if len(met) else None}


def player_projections(game_id: int, finished: bool) -> list[dict] | None:
    """Each team's projected players for a game (quarterback first, then by yards), with the actual
    numbers next to them once it's been played. None if the game has no projections."""
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone() is None:
            return None
        proj = pd.read_sql("SELECT * FROM nfl_player_projections WHERE game_id = ?", conn, params=(game_id,))
        actual = pd.read_sql("SELECT * FROM nfl_player_stats WHERE game_id = ?", conn, params=(game_id,)) \
            if finished else pd.DataFrame()
    if proj.empty:
        return None
    if not actual.empty:
        actual_columns = ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
                          "receptions", "receiving_yards", "receiving_tds"]
        proj = proj.merge(actual[["player_id", *actual_columns]].rename(columns={c: f"act_{c}" for c in actual_columns}),
                          on="player_id", how="left")
    fmt = lambda v, d=0: "" if v is None or pd.isna(v) else f"{v:.{d}f}"
    from nfl.players import chance_at_least, load_ranges, milestone
    ranges = load_ranges() if not finished else None

    def chance(stat, value):
        """'38% for 60+': the chance of reaching the round number nearest the projection, for upcoming games."""
        if not ranges or value is None or pd.isna(value) or value < {"receptions": 1}.get(stat, 5):
            return None   # too small a role for a milestone to mean much
        m = milestone(stat, value)
        p = chance_at_least(ranges, stat, value, m)
        shown = "<5%" if p <= 0.05 else ">95%" if p >= 0.95 else f"{p * 100:.0f}%"
        return f"{shown} for {m}+"

    teams = []
    for team, rows in proj.groupby("team"):
        rows = rows.assign(order=(rows["position"] != "QB").astype(int),
                           yards=rows["pass_yds"] + rows["rush_yds"] + rows["rec_yds"])
        rows = rows.sort_values(["order", "yards"], ascending=[True, False])
        players = []
        for r in rows.itertuples():
            if r.position != "QB" and r.rush_yds + r.rec_yds < 5:
                continue   # too small a role to be worth listing
            row = {"name": r.name, "url": player_url(r.player_id), "position": r.position, "qb": r.position == "QB",
                   "pass_yds": fmt(r.pass_yds), "pass_tds": fmt(r.pass_tds, 1), "ints": fmt(r.ints, 1),
                   "rush_yds": fmt(r.rush_yds), "receptions": fmt(r.receptions, 1), "rec_yds": fmt(r.rec_yds),
                   "anytime_td": f"{r.anytime_td * 100:.0f}%", "actual": None,
                   "chances": {stat: chance(stat, getattr(r, stat)) for stat in ("pass_yds", "rush_yds", "receptions", "rec_yds")}}
            if "act_receiving_yards" in rows:
                played = not pd.isna(r.act_rushing_yards)
                tds = (r.act_rushing_tds or 0) + (r.act_receiving_tds or 0) if played else None
                row["actual"] = {"pass_yds": fmt(r.act_passing_yards), "pass_tds": fmt(r.act_passing_tds),
                                 "ints": fmt(r.act_interceptions), "rush_yds": fmt(r.act_rushing_yards),
                                 "receptions": fmt(r.act_receptions), "rec_yds": fmt(r.act_receiving_yards),
                                 "scored": None if tds is None else bool(tds > 0), "played": played}
            players.append(row)
        teams.append({"team": team, "players": players})
    return teams


PROPS_PER_GAME = 5
PROPS_PER_DAY = 10


def likeliest(game_ids: list[int], by: str, n: int) -> pd.DataFrame:
    """The likeliest player props for these games (one per player), n per game or per day (by="game_id" or
    "date"), with whether each hit once the game is final. Empty if there are no projections."""
    from nfl.players import likeliest_props, load_ranges, prop_chances, prop_hit
    table = load_ranges()
    if not game_ids or table is None:
        return pd.DataFrame()
    marks = ",".join("?" * len(game_ids))
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone() is None:
            return pd.DataFrame()
        proj = pd.read_sql(f"SELECT * FROM nfl_player_projections WHERE game_id IN ({marks})", conn, params=game_ids)
        actual = pd.read_sql(f"SELECT * FROM nfl_player_stats WHERE game_id IN ({marks})", conn, params=game_ids)
    if proj.empty:
        return pd.DataFrame()
    games = data()["games"].set_index("id")
    chances = prop_chances(table, proj)
    if chances.empty:
        return pd.DataFrame()
    chances["date"] = chances["game_id"].map(games["date"])
    picks = likeliest_props(chances, by, n).merge(proj[["game_id", "player_id", "name", "team", "position"]],
                                                   on=["game_id", "player_id"])
    actual = actual.set_index(["game_id", "player_id"])
    final = set(games.index[games["status"] == "final"])

    def hit(r):
        if r["game_id"] not in final:
            return None
        key = (r["game_id"], r["player_id"])
        return prop_hit(r, actual.loc[key]) if key in actual.index else False   # recorded nothing: missed

    picks["hit"] = [hit(r) for _, r in picks.iterrows()]
    picks["matchup"] = [f"{games.at[g, 'away']} at {games.at[g, 'home']}" for g in picks["game_id"]]
    picks["url"] = [game_url(games.at[g, "source_game_id"]) for g in picks["game_id"]]
    return picks.sort_values(["chance", "threshold"], ascending=False)


def prop_rows(picks: pd.DataFrame) -> list[dict]:
    return [{"name": r.name, "player_url": player_url(r.player_id), "team": r.team, "position": r.position, "prop": r.prop,
             "chance": ">95%" if r.chance >= 0.95 else f"{r.chance * 100:.0f}%", "hit": r.hit,
             "matchup": r.matchup, "url": r.url} for r in picks.itertuples()]


def prop_record(picks: pd.DataFrame) -> dict | None:
    done = picks[picks["hit"].notna()]
    if done.empty:
        return None
    return {"props": len(done), "hits": int(done["hit"].astype(bool).sum()),
            "hit_rate": f"{done['hit'].astype(bool).mean() * 100:.0f}%",
            "expected": f"{done['chance'].mean() * 100:.0f}%"}


def why_pick(g) -> dict | None:
    """The predicted margin from today's ratings, split into offense, defense, quarterbacks and home
    field (points, + toward the home team)."""
    model = cached_model()
    if model is None or pd.isna(getattr(g, "model_home", np.nan)):
        return None
    r = model["ratings"]
    if g.home not in r.index or g.away not in r.index:
        return None
    qb = {side: qb_value(model, getattr(g, f"{side}_qb_expected"), getattr(g, f"{side}_qb_starts")) for side in ("home", "away")}
    names = {side: getattr(g, f"{side}_qb_expected") for side in ("home", "away")}
    rows = [{"label": "Offense", "value": r.at[g.home, "offense"] - r.at[g.away, "offense"],
             "note": "Apart from the quarterback, against the defenses faced"},
            {"label": "Defense", "value": r.at[g.home, "defense"] - r.at[g.away, "defense"],
             "note": "Points allowed, against the offenses faced"},
            {"label": "Quarterbacks", "value": qb["home"] - qb["away"],
             "note": (f"{names['away']} vs {names['home']}, each against an average starter"
                      if all(isinstance(n, str) for n in names.values()) else "")},
            {"label": "Home field", "value": 0.0 if g.neutral_site else model["home_adv"],
             "note": "Neutral site" if g.neutral_site else ""}]
    names = [f"{team_name(getattr(g, side))}: {getattr(g, f'{side}_missing_names')}" for side in ("away", "home")
             if isinstance(getattr(g, f"{side}_missing_names", None), str)]
    rows.append({"label": "Missing starters", "value": injury_effect(missing(g, "home"), missing(g, "away")),
                 "note": "; ".join(names) if names else "None of either team's regular starters ruled out"})
    market_home = getattr(g, "mkt_home", np.nan)
    if pd.isna(market_home):
        market_home = no_vig(g.home_moneyline, g.away_moneyline)
    return why(rows, team_name(g.home), team_name(g.away), float(g.model_home), market_home if market_home is not None else None)


@router.get("/game/{key}/", response_class=HTMLResponse)
def game_page(request: Request, key: str):
    d = data()
    games = d["games"]
    found = games[games["source_game_id"].str.lower().str.replace("_", "-") == key]
    if found.empty:
        raise HTTPException(status_code=404)
    g = next(with_predictions(found).itertuples())
    card = game_card(g)
    finished = g.status == "final"
    qb = None
    if isinstance(g.home_qb_expected, str) or isinstance(g.away_qb_expected, str):
        model = cached_model() if not finished else None   # current ratings only describe upcoming games
        value = lambda side: (signed_number(qb_value(model, getattr(g, f"{side}_qb_expected"),
                                                     getattr(g, f"{side}_qb_starts")), 1) if model else None)
        qb = {side: {"name": getattr(g, f"{side}_qb_expected"), "listed": isinstance(getattr(g, f"{side}_qb"), str),
                     "change": bool(getattr(g, f"{side}_qb_change")), "value": value(side),
                     "replaced": (getattr(g, f"{side}_qb_listed") if isinstance(getattr(g, f"{side}_qb_listed"), str)
                                  and getattr(g, f"{side}_qb_listed") != getattr(g, f"{side}_qb") else None),
                     "starts": None if pd.isna(getattr(g, f"{side}_qb_starts")) else int(getattr(g, f"{side}_qb_starts"))}
              for side in ("away", "home")}
    reasons = why_pick(g) if not finished else None
    with connect() as conn:
        moved = betting.movement_chart(conn, int(g.id), g.home, g.away)
    expected = None
    if pd.notna(getattr(g, "exp_home", np.nan)):
        expected = {"home": f"{g.exp_home:.1f}", "away": f"{g.exp_away:.1f}"}
    return render(request, "game.html", {
        "active": None, "g": card, "finished": finished, "expected": expected, "qb": qb, "why": reasons, "moved": moved,
        "date": day_label(g.date), "season": int(g.season), "stadium": g.stadium,
        "home_score": int(g.home_score) if finished else None, "away_score": int(g.away_score) if finished else None,
        "form_home": form(games, g.home, g.date), "form_away": form(games, g.away, g.date),
        "h2h": head_to_head(games, g.away, g.home, g.date),
        "margin_sd": f"{MARGIN_SD:g}",
        "players": player_projections(int(g.id), finished),
        "props": prop_rows(likeliest([int(g.id)], "game_id", PROPS_PER_GAME)),
    })


def live_scores(df: pd.DataFrame) -> dict:
    """The model's record (and the closing market's, where available) on finished live predictions."""
    decided = df[df["home_score"] != df["away_score"]]
    home_won = decided["home_score"] > decided["away_score"]
    model_p = np.where(home_won, decided["model_home"], 1 - decided["model_home"])
    priced = decided.dropna(subset=["mkt_home"])
    priced_won = priced["home_score"] > priced["away_score"]
    market_p = np.where(priced_won, priced["mkt_home"], 1 - priced["mkt_home"])
    model_on_priced = np.where(priced_won, priced["model_home"], 1 - priced["model_home"])
    return {"games": len(decided), "ties": len(df) - len(decided),
            "model_typical": typical_probability(-np.log(model_p).mean()) if len(decided) else None,
            "favorite_won": int((model_p > 0.5).sum()),
            "compared": len(priced),
            "model_typical_compared": typical_probability(-np.log(model_on_priced).mean()) if len(priced) else None,
            "market_typical": typical_probability(-np.log(market_p).mean()) if len(priced) else None}


STAT_LABELS = {"pass_yds": "Passing yards", "pass_tds": "Passing touchdowns", "rush_yds": "Rushing yards",
               "receptions": "Receptions", "rec_yds": "Receiving yards"}


def player_track() -> dict | None:
    """Live player projections on finished games against what happened and the simple benchmarks."""
    from nfl.players import STATS, accuracy, compare
    d = data()
    if "player_track" in d:
        return d["player_track"]
    result = None
    with connect() as conn:
        has = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone()
        proj = pd.read_sql("""SELECT p.* FROM nfl_player_projections p JOIN games g ON g.id = p.game_id
                              WHERE g.status = 'final'""", conn) if has else pd.DataFrame()
        if not proj.empty:
            games = pd.read_sql("""SELECT g.id, g.season, g.date, i.week, i.game_type FROM games g
                                   JOIN nfl_game_info i ON i.game_id = g.id""", conn)
            ids = sorted(set(proj["player_id"]))
            players = pd.concat([pd.read_sql(f"SELECT * FROM nfl_player_stats WHERE player_id IN "
                                             f"({','.join('?' * len(ids[i:i + 500]))})", conn, params=ids[i:i + 500])
                                 for i in range(0, len(ids), 500)], ignore_index=True)
    if not proj.empty:
        columns = ["game_id", "player_id", *STATS, "anytime_td"]
        df = compare(proj[columns], players, games)
        missing = len(proj) - len(df)
        acc = accuracy(df).set_index("stat")
        overall = [{"label": STAT_LABELS[s], "players": int(acc.at[s, "players"]),
                    "model": f"{acc.at[s, 'model']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "last3": f"{acc.at[s, 'last 3 games']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "season": f"{acc.at[s, 'season average']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "best": acc.at[s, "model"] <= min(acc.at[s, "last 3 games"], acc.at[s, "season average"])}
                   for s in STATS if acc.at[s, "players"] > 0]

        df = df.merge(games[["id", "week", "game_type"]].rename(columns={"id": "game_id"}), on="game_id")
        df["proj_total"] = df["proj_pass_yds"] + df["proj_rush_yds"] + df["proj_rec_yds"]
        df["actual_total"] = df[["passing_yards", "rushing_yards", "receiving_yards"]].fillna(0).sum(axis=1)
        weeks = []
        for (season, week), part in sorted(df.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True):
            miss = lambda col, actual: (part[col] - part[actual].fillna(0)).abs().mean()
            weeks.append({"label": f"{season} {week_label(part.iloc[0]['game_type'], week)}", "players": len(part),
                          "rec_model": f"{miss('proj_rec_yds', 'receiving_yards'):.1f}",
                          "rec_season": f"{miss('season_rec_yds', 'receiving_yards'):.1f}",
                          "rush_model": f"{miss('proj_rush_yds', 'rushing_yards'):.1f}",
                          "rush_season": f"{miss('season_rush_yds', 'rushing_yards'):.1f}"})

        scored = (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)) > 0
        bins = pd.cut(df["proj_anytime_td"], [0, 0.15, 0.3, 0.45, 1])
        calibration = [{"range": f"{int(b.left * 100)}\u2013{int(b.right * 100)}%", "players": len(part),
                        "predicted": f"{part['proj_anytime_td'].mean() * 100:.0f}%",
                        "actual": f"{scored[part.index].mean() * 100:.0f}%"}
                       for b, part in df.groupby(bins, observed=True)]

        latest = df[(df["season"] == df["season"].max())]
        latest = latest[latest["week"] == latest["week"].max()].copy()
        latest["miss"] = latest["actual_total"] - latest["proj_total"]
        surprises = [{"name": r.name, "team": r.team, "projected": f"{r.proj_total:.0f}", "actual": f"{r.actual_total:.0f}",
                      "url": game_url(data()["games"].set_index("id").at[r.game_id, "source_game_id"])}
                     for r in latest.reindex(latest["miss"].abs().sort_values(ascending=False).index).head(6).itertuples()]
        props = None
        picks = likeliest(sorted(int(i) for i in proj["game_id"].unique()), "date", PROPS_PER_DAY)
        if not picks.empty and prop_record(picks):
            info = data()["games"].set_index("id")
            picks["season"], picks["week"] = picks["game_id"].map(info["season"]), picks["game_id"].map(info["week"])
            props = {"overall": prop_record(picks), "weeks": [
                {"label": f"{season} {week_label(info.at[part.iloc[0]['game_id'], 'game_type'], week)}", **prop_record(part)}
                for (season, week), part in sorted(picks.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True)
                if prop_record(part)]}
        result = {"overall": overall, "weeks": weeks, "calibration": calibration, "surprises": surprises,
                  "surprise_week": weeks[0]["label"] if weeks else None, "games": int(df["game_id"].nunique()),
                  "missing": missing, "props": props}
    d["player_track"] = result
    return result


@router.get("/track-record/", response_class=HTMLResponse)
def track_record(request: Request):
    d = data()
    live = d["predictions"][d["predictions"]["source"] == "Saved before kickoff"]
    df = (finished_games().merge(live, on="id").merge(d["closing"], on="id", how="left")
          .sort_values(["date", "kickoff", "home"]))
    from charts import calibration_chart
    past = finished_games().merge(d["backtest"], on="id")
    calib = calibration_chart(calibration_pairs(df), calibration_pairs(past[past["season"] >= HELD_OUT_FROM]),
                              past_from=f"{HELD_OUT_FROM}")
    context = {"active": "track", "record_url": RECORD_URL, "history_url": RECORD_HISTORY_URL, "weeks": [],
               "players": player_track(), "player_record_url": PLAYER_RECORD_URL, "calibration": calib}
    if df.empty:
        return render(request, "track_record.html", context)
    weeks = []
    for (season, week), group in sorted(df.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True):
        weeks.append({"label": f"{season} {week_label(group.iloc[0]['game_type'], week)}",
                      "scores": live_scores(group), "games": [game_card(g) for g in group.itertuples()]})
    held = backtest_frame()
    long_run = typical_probability(held["model_logloss"].mean()) if held is not None else None
    from charts import record_chart
    chart_weeks = []
    for (season, week), group in sorted(df.groupby(["season", "week"]), key=lambda kv: kv[0]):
        g = group[(group["home_score"] != group["away_score"])].dropna(subset=["mkt_home"])
        won = (g["home_score"] > g["away_score"]).to_numpy()
        chart_weeks.append({"label": f"Wk {week}" if group.iloc[0]["game_type"] == "REG" else group.iloc[0]["game_type"],
                            "model": np.where(won, g["model_home"], 1 - g["model_home"]),
                            "benchmark": np.where(won, g["mkt_home"], 1 - g["mkt_home"])})
    return render(request, "track_record.html", {**context, "overall": live_scores(df), "weeks": weeks,
                                                 "first_date": day_label(df["date"].min(), with_year=True),
                                                 "long_run": long_run, "record_chart": record_chart(chart_weeks, "the betting market")})


def backtest_frame() -> pd.DataFrame | None:
    """Held-out backtest games (2018 onward, ties left out) with model and market log loss."""
    d = data()
    df = (finished_games().merge(d["backtest"], on="id").merge(d["closing"], on="id")
          .query("home_score != away_score"))
    if df.empty:
        return None
    won = df["home_score"] > df["away_score"]
    df["model_logloss"] = -np.log(np.where(won, df["model_home"], 1 - df["model_home"]))
    df["mkt_logloss"] = -np.log(np.where(won, df["mkt_home"], 1 - df["mkt_home"]))
    return df


@router.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    df = backtest_frame()
    if df is None:
        return render(request, "backtest.html", {"active": "backtest", "seasons": []})
    by_season = (df.groupby("season").agg(games=("id", "size"), model=("model_logloss", "mean"),
                                           market=("mkt_logloss", "mean")).reset_index())
    by_season["gap"] = by_season["model"] - by_season["market"]
    widest = by_season["gap"].abs().max() or 1
    seasons = [{"label": str(r.season), "games": r.games, "model": f"{r.model:.3f}", "market": f"{r.market:.3f}",
                "gap": f"{abs(r.gap):.3f}", "market_better": r.gap > 0,
                "width": round(abs(r.gap) / widest * 100, 1), "held_out": r.season >= HELD_OUT_FROM}
               for r in by_season.itertuples()]

    held = df[df["season"] >= HELD_OUT_FROM].copy()
    margin = held["home_score"] - held["away_score"]
    held_all = finished_games().merge(data()["backtest"], on="id")
    held_all = held_all[held_all["season"] >= HELD_OUT_FROM]
    actual_margin = held_all["home_score"] - held_all["away_score"]
    model_miss = (actual_margin - (held_all["exp_home"] - held_all["exp_away"])).abs().mean()
    spread_miss = (actual_margin - held_all["spread_line"]).abs().mean()

    # Calibration: when the model gave the home team a given chance, how often did it win?
    bins = pd.cut(held["model_home"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1])
    calibration = [{"range": f"{int(b.left * 100)}–{int(b.right * 100)}%", "games": len(part),
                    "predicted": f"{part['model_home'].mean() * 100:.0f}%",
                    "actual": f"{(part['home_score'] > part['away_score']).mean() * 100:.0f}%"}
                   for b, part in held.groupby(bins, observed=True)]
    return render(request, "backtest.html", {
        "blend": blend.load(ROOT / "nfl" / "blend.json"),
        "active": "backtest", "seasons": seasons,
        "headline": {"games": len(held), "from": HELD_OUT_FROM,
                     "model": f"{held['model_logloss'].mean():.4f}", "market": f"{held['mkt_logloss'].mean():.4f}",
                     "model_typical": typical_probability(held["model_logloss"].mean()),
                     "market_typical": typical_probability(held["mkt_logloss"].mean()),
                     "picked": round(((held["model_home"] > 0.5) == (margin > 0)).mean() * 100),
                     "market_picked": round(((held["mkt_home"] > 0.5) == (margin > 0)).mean() * 100)},
        "margins": {"model": f"{model_miss:.1f}", "spread": f"{spread_miss:.1f}"},
        "calibration": calibration,
        "development": [{"label": label, "value": f"{value:.4f}"} for label, value in DEVELOPMENT],
        "player_accuracy": PLAYER_ACCURACY,
        "start": START_SEASON, "tuned_through": HELD_OUT_FROM - 1, "half_life": HALF_LIFE_DAYS,
        "training_years": TRAINING_YEARS,
    })


@router.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request):
    model = cached_model()
    games = data()["games"]
    if model is None:
        return render(request, "ratings.html", {"active": "ratings", "rows": []})
    current = games[games["season"] == games["season"].max()]
    teams = sorted(set(current["home"]) | set(current["away"]))
    r = model["ratings"].loc[[t for t in teams if t in model["ratings"].index]].copy()
    r["overall"] = r["offense"] + r["defense"]
    rows = [{"team": t, "name": team_name(t), "division": TEAMS.get(t, ("", ""))[1], **rating_row(model, t)}
            for t in r.sort_values("overall", ascending=False).index]
    return render(request, "ratings.html", {
        "active": "ratings", "rows": rows, "as_of": day_label(date.today().isoformat()),
        "games_used": model["games_used"], "home_adv": f"{model['home_adv']:.1f}",
        "inexperience": f"{abs(model['inexperience']):.1f}", "inexperience_starts": INEXPERIENCE_STARTS,
        "average_points": f"{model['base']:.1f}", "quarterbacks": quarterback_rows(model, teams),
    })


def quarterback_rows(model: dict, teams: list[str]) -> list[dict]:
    """Each current team's expected starter for its next game (or its latest starter), best first."""
    games = data()["games"]
    rows = []
    for team in teams:
        own = games[(games["home"] == team) | (games["away"] == team)].sort_values("date")
        upcoming = own[own["status"] == "scheduled"]
        g = (upcoming if len(upcoming) else own).iloc[0 if len(upcoming) else -1]
        side = "home" if g["home"] == team else "away"
        qb, starts = g[f"{side}_qb_expected"], g[f"{side}_qb_starts"]
        if not isinstance(qb, str):
            continue
        value = qb_value(model, qb, starts)
        rows.append({"team": team, "name": qb, "value": signed_number(value, 1), "raw": value,
                     "class": "good" if value > 0 else "bad" if value < 0 else "",
                     "starts": None if pd.isna(starts) else int(starts)})
    return sorted(rows, key=lambda r: -r["raw"])


# ---------------------------------------------------------------- player pages

def player_url(player_id) -> str:
    return f"{BASE}/nfl/player/{player_id}/"


templates.env.globals["nfl_player_url"] = player_url
GAME_LOG = 10
LOG_STATS = ["completions", "pass_attempts", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards",
             "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"]


@router.get("/player/{player_id}/", response_class=HTMLResponse)
def player_page(request: Request, player_id: str):
    if not DB_PATH.exists():
        raise HTTPException(status_code=404)
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_stats'").fetchone() is None:
            raise HTTPException(status_code=404)
        log = pd.read_sql(f"""SELECT s.game_id, s.name, s.position, s.team, {', '.join('s.' + c for c in LOG_STATS)},
                                     g.date, g.season, i.week, i.game_type, i.source_game_id, h.name AS home, a.name AS away,
                                     g.home_score, g.away_score
                              FROM nfl_player_stats s JOIN games g ON g.id = s.game_id JOIN nfl_game_info i ON i.game_id = g.id
                              JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                              WHERE s.player_id = ? ORDER BY g.date""", conn, params=(player_id,))
        has = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone()
        proj = pd.read_sql("SELECT * FROM nfl_player_projections WHERE player_id = ?", conn, params=(player_id,)) if has else pd.DataFrame()
    if log.empty and proj.empty:
        raise HTTPException(status_code=404)
    latest = log.iloc[-1] if not log.empty else proj.iloc[-1]
    qb = latest["position"] == "QB"
    seasons = []
    for season, part in sorted(log.groupby("season"), key=lambda kv: kv[0], reverse=True)[:3]:
        reg = part[part["game_type"] == "REG"]
        if reg.empty:
            continue
        tot = reg[LOG_STATS].fillna(0).sum()
        seasons.append({"season": int(season), "games": len(reg), "pass_yds": int(tot["passing_yards"]),
                        "pass_tds": int(tot["passing_tds"]), "ints": int(tot["interceptions"]),
                        "rush_yds": int(tot["rushing_yards"]), "rec": int(tot["receptions"]), "rec_yds": int(tot["receiving_yards"]),
                        "tds": int(tot["rushing_tds"] + tot["receiving_tds"])})
    by_game = proj.set_index("game_id") if not proj.empty else pd.DataFrame()
    fmt = lambda v, d=0: "–" if v is None or pd.isna(v) else f"{v:.{d}f}"
    games_log = []
    for r in log.iloc[::-1].head(GAME_LOG).itertuples():
        at_home = r.home == r.team
        hs, as_ = int(r.home_score), int(r.away_score)
        own, opp_score = (hs, as_) if at_home else (as_, hs)
        p = by_game.loc[r.game_id] if r.game_id in by_game.index else None
        games_log.append({"label": f"{r.season} {week_label(r.game_type, r.week)}", "opponent": r.away if at_home else r.home,
                          "venue": "vs" if at_home else "at", "url": game_url(r.source_game_id),
                          "result": f"{'W' if own > opp_score else 'L' if own < opp_score else 'T'} {own}–{opp_score}",
                          "pass_yds": fmt(r.passing_yards), "pass_tds": fmt(r.passing_tds), "rush_yds": fmt(r.rushing_yards),
                          "rec": fmt(r.receptions), "rec_yds": fmt(r.receiving_yards),
                          "tds": fmt((r.rushing_tds or 0) + (r.receiving_tds or 0)),
                          "proj": {"pass_yds": fmt(p["pass_yds"]), "pass_tds": fmt(p["pass_tds"], 1), "rush_yds": fmt(p["rush_yds"]),
                                   "rec": fmt(p["receptions"], 1), "rec_yds": fmt(p["rec_yds"]),
                                   "td": f"{p['anytime_td'] * 100:.0f}%"} if p is not None else None})
    upcoming = []
    games = data()["games"].set_index("id")
    for r in proj.itertuples() if not proj.empty else []:
        if r.game_id in games.index and games.at[r.game_id, "status"] == "scheduled":
            g = games.loc[r.game_id]
            upcoming.append({"label": f"{g['date']}", "date": day_label(g["date"]), "matchup": f"{g['away']} at {g['home']}",
                             "url": game_url(g["source_game_id"]), "pass_yds": fmt(r.pass_yds), "pass_tds": fmt(r.pass_tds, 1),
                             "rush_yds": fmt(r.rush_yds), "rec": fmt(r.receptions, 1), "rec_yds": fmt(r.rec_yds),
                             "td": f"{r.anytime_td * 100:.0f}%"})
    return render(request, "player.html", {"active": None, "name": latest["name"], "team": latest["team"],
                                           "team_name": team_name(latest["team"]), "position": latest["position"], "qb": qb,
                                           "seasons": seasons, "log": games_log, "upcoming": upcoming})
