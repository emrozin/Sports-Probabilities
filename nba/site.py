"""The NBA pages of the website, under /nba/ (included by app.py).

Predictions come from player ratings and who's available (nba/model.py); betting lines from
ESPN (espn.py) where saved; injuries and rosters from ESPN (nba/live.py). There are no free
historical betting lines for the NBA, so the backtest compares the model with a simple Elo rating.
"""

import sqlite3
from contextlib import closing
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

import betting
import colors
import espn
from cfb.site import kickoff_label, line_text, two_way
from charts import chance_bars, record_chart
from nba import live
from nba.model import MARGIN_SD, breakdown, elo_benchmark, fit, lineup_strength, load_games, load_players
from nba.players import (PROPS, STAT_NAMES, STATS, add_rates, chance_at_least, likeliest_props, prop_chances)
from nba.predict_upcoming import MODEL_NAME
from nba.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
from web import BASE, ROOT, day_label, disk_cached, edge, ordinal, signed_number, slugify, templates, typical_probability, why

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter(prefix="/nba")
RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/nba_live_predictions.csv"
GAME_COLUMNS = ["id", "season", "date", "kickoff", "status", "home", "away", "home_score", "away_score", "season_type",
                "neutral_site", "notes", "source_game_id", "week"]
PROPS_PER_GAME, PROPS_PER_DAY = 8, 10
SEASON_TYPE_LABEL = {"playin": "Play-in", "playoffs": "Playoffs", "cup_final": "NBA Cup final"}
# How the model's versions compared on the held-out seasons (python -m nba.run_backtest and the
# comparisons in the README): log loss, 2014-15 to 2025-26
DEVELOPMENT: list[tuple[str, str]] = [
    ("A simple Elo rating", "0.6265"),
    ("Team ratings only, no players", "0.6211"),
    ("Players, ignoring who's available (each team's players from its last five games)", "0.6179"),
    ("Players, knowing who's available, with usual minutes", "0.6135"),
    ("...plus back-to-backs (what the site does)", "0.6124"),
    ("Players with the minutes they really played (an upper bound: not known beforehand)", "0.6086"),
]

_cache: dict = {}


def connect():
    return closing(sqlite3.connect(DB_PATH))


def season_label(season: int) -> str:
    return f"{season}–{(season + 1) % 100:02d}"


def team_url(team: str) -> str:
    return f"{BASE}/nba/team/{slugify(team)}/"


def game_url(source_id) -> str:
    return f"{BASE}/nba/game/{source_id}/"


def team_colors() -> dict:
    return data().get("colors", {})


def abbreviation(team: str) -> str:
    info = data()["teams"]
    return info.at[team, "abbreviation"] if team in info.index else team


def nickname(team: str) -> str:
    """'Boston Celtics' -> 'Celtics', 'Portland Trail Blazers' -> 'Trail Blazers'."""
    two_word = ("Trail Blazers",)
    return next((n for n in two_word if team.endswith(n)), team.split()[-1])


templates.env.globals["nba_team_url"] = team_url
templates.env.globals["nba_colors"] = lambda home, away: colors.style(team_colors(), home, away)
templates.env.globals["nba_team_colors"] = lambda team: colors.team_style(team_colors(), team)
templates.env.globals["nba_chip"] = lambda team: colors.chip(team_colors(), team)
templates.env.globals["nba_abbr"] = abbreviation
templates.env.globals["nba_nickname"] = nickname


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request, f"nba/{template}", {"sport": "nba", **context}, status_code=status_code)


# ---------------------------------------------------------------- data

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
    _cache["teams"] = pd.DataFrame(columns=["abbreviation", "conference", "division"])
    if stamp is not None:
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_game_info'").fetchone():
                games = load_games(conn)
                live_preds = load_predictions(conn, MODEL_NAME, None).assign(source="Saved before tip-off")
                back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).assign(source="Backtest")
                _cache["predictions"] = pd.concat([live_preds, back[~back["id"].isin(live_preds["id"])]], ignore_index=True)
                _cache["live"] = live_preds
                _cache["teams"] = pd.read_sql("SELECT name, abbreviation, conference, division FROM nba_teams", conn).set_index("name")
                _cache["colors"] = {n: (c, a or "#7F7F7F") for n, c, a in conn.execute("SELECT name, color, alt_color FROM nba_teams") if c}
                _cache["market"] = espn.market_lines(conn, "NBA")
                _cache["rosters"], _cache["injuries"] = live.load(conn)
                has_proj = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_player_projections'").fetchone()
                _cache["projections"] = pd.read_sql("SELECT * FROM nba_player_projections", conn) if has_proj else pd.DataFrame()
    for key in ("predictions", "live"):
        _cache.setdefault(key, pd.DataFrame(columns=["id", "model_home", "exp_home", "exp_away", "source"]))
        for col in ("model_home", "exp_home", "exp_away"):
            _cache[key][col] = pd.to_numeric(_cache[key][col], errors="coerce")
    _cache.setdefault("market", pd.DataFrame(columns=["id", "market_home", "market_spread", "market_total", "market_final"]))
    _cache.setdefault("rosters", pd.DataFrame(columns=["team", "player_id", "name", "position"]))
    _cache.setdefault("injuries", pd.DataFrame(columns=["team", "player_id", "name", "status", "return_date", "comment"]))
    _cache.setdefault("projections", pd.DataFrame())
    _cache["games"] = games
    return _cache


def players() -> pd.DataFrame:
    """Every player line (loaded on first use: it takes a few seconds)."""
    d = data()
    if "players" not in d:
        has = False
        if DB_PATH.exists():
            with connect() as conn:
                has = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_player_games'").fetchone() is not None
                if has:
                    d["players"] = add_rates(load_players(conn))
        if not has:
            d["players"] = pd.DataFrame(
                columns=["game_id", "team", "player_id", "name", "position", "minutes", "date", "season", "side",
                         "expected_minutes", "minutes_after", *STATS])
    return d["players"]


def current() -> dict | None:
    """Today's model, each player's latest line, and who's out (cached for the day, and on disk until
    the database changes: fitting needs every player line, which takes a while to load)."""
    d = data()
    today = date.today().isoformat()
    if d.get("current_day") != today:
        def make():
            games, lines = d["games"], players()
            model = fit(games, lines, today) if not games.empty and not lines.empty else None
            latest = live.latest_lines(lines) if not lines.empty else pd.DataFrame(columns=["team", "name", "position", "date", "minutes_after"])
            return {"model": model, "latest": latest} if model else None
        found = disk_cached(DB_PATH.parent / "cache", "nba_current", (d["stamp"], today, "v2"), make) if DB_PATH.exists() else None
        d["current"] = {**found, "out": live.unavailable(d["injuries"])} if found else None
        d["current_day"] = today
    return d["current"]


def box_lines(game_ids) -> pd.DataFrame:
    """The box score lines of a few games (for results next to projections), straight from the database."""
    ids = [int(i) for i in game_ids]
    if not ids or not DB_PATH.exists():
        return pd.DataFrame(columns=["game_id", "team", "player_id", "name", "minutes", *STATS])
    with connect() as conn:
        out = []
        for k in range(0, len(ids), 500):
            part = ids[k:k + 500]
            out.append(pd.read_sql(f"""SELECT game_id, team, player_id, name, minutes, {', '.join(STATS)} FROM nba_player_games
                                       WHERE game_id IN ({','.join('?' * len(part))})""", conn, params=part))
    return pd.concat(out, ignore_index=True)


def latest_season() -> int | None:
    games = data()["games"]
    return int(games["season"].max()) if not games.empty else None


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    d = data()
    return games.merge(d["predictions"], on="id", how="left").merge(d["market"], on="id", how="left")


def injury_notes(team: str) -> list[dict]:
    """The team's players on the injury report who are ruled out, for cards and pages."""
    inj = data()["injuries"]
    out = inj[(inj["team"] == team) & inj["status"].isin(live.UNAVAILABLE)]
    return [{"name": r.name, "status": r.status, "comment": r.comment if isinstance(r.comment, str) else "",
             "back": day_label(r.return_date) if isinstance(r.return_date, str) and r.return_date else None}
            for r in out.itertuples()]


def game_card(g, current_team: str | None = None) -> dict:
    finished = g.status == "final"
    has_model = pd.notna(getattr(g, "model_home", np.nan))
    margin = (g.exp_home - g.exp_away) if pd.notna(getattr(g, "exp_home", np.nan)) else None
    market_home = getattr(g, "market_home", np.nan)
    card = {"home": g.home, "away": g.away, "finished": finished, "neutral": bool(g.neutral_site),
            "model": two_way(g.model_home) if has_model else None,
            "market": two_way(market_home) if pd.notna(market_home) else None,
            "source": g.source if isinstance(getattr(g, "source", None), str) else None,
            "model_line": line_text(margin, nickname(g.home), nickname(g.away)),
            "model_total": f"{g.exp_home + g.exp_away:.0f}" if margin is not None else None,
            "market_line": line_text(-g.market_spread, nickname(g.home), nickname(g.away))
            if pd.notna(getattr(g, "market_spread", np.nan)) else None,
            "market_total": f"{g.market_total:g}" if pd.notna(getattr(g, "market_total", np.nan)) else None,
            "url": game_url(g.source_game_id)}
    when = [day_label(g.date, with_year=True)] if current_team else []
    when += [kickoff_label(g.kickoff)] if isinstance(g.kickoff, str) and not finished else []
    when += [SEASON_TYPE_LABEL[g.season_type]] if g.season_type in SEASON_TYPE_LABEL and not isinstance(g.notes, str) else []
    when += [g.notes] if isinstance(g.notes, str) else []
    card["when"] = ", ".join(x for x in when if x)
    if has_model and card["market"] and not finished:
        card["edge"] = edge(card["model"], card["market"], {"home": nickname(g.home), "away": nickname(g.away)})
    if not finished:
        card["out"] = {side: [i["name"] for i in injury_notes(getattr(g, side))] for side in ("home", "away")}
    else:
        hs, as_ = int(g.home_score), int(g.away_score)
        card.update(home_score=hs, away_score=as_, result="home" if hs > as_ else "away")
        if card["model"]:
            card["model_actual"] = card["model"][card["result"]]
        if card["market"]:
            card["market_actual"] = card["market"][card["result"]]
        if current_team:
            own, other = (hs, as_) if g.home == current_team else (as_, hs)
            card["outcome"] = "W" if own > other else "L"
    return card


def summary(df: pd.DataFrame) -> dict | None:
    df = df.dropna(subset=["model_home"])
    if df.empty:
        return None
    won = (df["home_score"] > df["away_score"]).to_numpy()
    p = np.where(won, df["model_home"].astype(float), 1 - df["model_home"].astype(float))
    out = {"games": len(df), "model_typical": typical_probability(-np.log(np.clip(p, 1e-6, 1)).mean()),
           "model_favorite_won": int((p > 0.5).sum())}
    if "market_home" in df and df["market_home"].notna().any():
        m = df.dropna(subset=["market_home"])
        mw = (m["home_score"] > m["away_score"]).to_numpy()
        q = np.where(mw, m["market_home"].astype(float), 1 - m["market_home"].astype(float))
        pm = np.where(mw, m["model_home"].astype(float), 1 - m["model_home"].astype(float))
        out.update(market_games=len(m), market_typical=typical_probability(-np.log(np.clip(q, 1e-6, 1)).mean()),
                   model_typical_same=typical_probability(-np.log(np.clip(pm, 1e-6, 1)).mean()))
    return out


# ---------------------------------------------------------------- props

def prop_rows(proj: pd.DataFrame, games: pd.DataFrame, n_by: str, n: int) -> list[dict]:
    """The likeliest props from saved projections, with the game and (for finished games) the result."""
    if proj.empty:
        return []
    proj = proj.rename(columns={s: f"proj_{s}" for s in STATS})
    chances = prop_chances(proj)
    chances = chances.merge(games[["id", "date", "home", "away", "source_game_id", "status"]], left_on="game_id", right_on="id")
    picks = likeliest_props(chances, n_by, n)
    lines = box_lines(picks.loc[picks["status"] == "final", "game_id"].unique())
    actual = lines.set_index(["game_id", "player_id"]) if not lines.empty else None
    rows = []
    for r in picks.itertuples():
        hit = None
        if r.status == "final" and actual is not None and (r.game_id, str(r.player_id)) in actual.index:
            hit = bool(actual.loc[(r.game_id, str(r.player_id)), r.stat] >= r.line)
        rows.append({"name": r.name, "player_url": player_url(r.player_id), "team": abbreviation(r.team), "position": "", "prop": f"{r.line}+ {STAT_NAMES[r.stat]}",
                     "chance": f"{r.chance * 100:.0f}%", "url": game_url(r.source_game_id), "hit": hit,
                     "matchup": f"{abbreviation(r.away)} at {abbreviation(r.home)}", "date": r.date})
    return rows


# ---------------------------------------------------------------- pages

@router.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    d = data()
    games = d["games"]
    today = date.today().isoformat()
    soon = games[(games["status"] == "scheduled") & (games["date"] >= today)]
    predicted = with_predictions(soon[soon["id"].isin(d["live"]["id"])].drop(columns=[], errors="ignore"))
    predicted = predicted.drop(columns=["model_home", "exp_home", "exp_away", "source"]).merge(d["live"], on="id")
    proj = d["projections"]
    days = []
    for day, group in predicted.sort_values(["date", "kickoff", "home"]).groupby("date", sort=True):
        day_proj = proj[proj["game_id"].isin(group["id"])] if not proj.empty else proj
        days.append({"label": day_label(day), "games": [game_card(g) for g in group.itertuples()],
                     "props": prop_rows(day_proj, games, "date", PROPS_PER_DAY)})
    first = soon["date"].min() if not soon.empty else None
    return render(request, "upcoming.html", {"active": "upcoming", "days": days,
                                             "season_starts": day_label(first) if first else None,
                                             "props_per_day": PROPS_PER_DAY, "props_per_game": PROPS_PER_GAME})


@router.get("/props/", response_class=HTMLResponse)
def props_page(request: Request):
    d = data()
    games, proj = d["games"], d["projections"]
    today = date.today().isoformat()
    days = []
    if not proj.empty:
        dated = proj.merge(games[["id", "date"]], left_on="game_id", right_on="id")
        for day in sorted(dated["date"].unique(), reverse=True)[:14]:
            if day < (pd.Timestamp(today) - pd.Timedelta(days=7)).strftime("%Y-%m-%d"):
                continue
            rows = prop_rows(dated[dated["date"] == day].drop(columns=["id", "date"]), games, "date", PROPS_PER_DAY)
            done = [r for r in rows if r["hit"] is not None]
            days.append({"label": day_label(day), "props": rows, "upcoming": day >= today,
                         "hits": sum(r["hit"] for r in done), "settled": len(done)})
    days.sort(key=lambda x: (not x["upcoming"], x["label"]))
    vs_book, record = sportsbook_props(games, today)
    return render(request, "props.html", {"active": "props", "days": days, "props_per_day": PROPS_PER_DAY,
                                          "vs_book": vs_book, "book_record": record,
                                          "lines": {STAT_NAMES[s]: ", ".join(f"{x}+" for x in v) for s, v in PROPS.items()}})


def sportsbook_props(games: pd.DataFrame, today: str) -> tuple[list[dict], dict | None]:
    """The model against DraftKings' prop lines (nba/prop_lines.py): today's and tomorrow's biggest
    disagreements, and the record on settled lines."""
    from nba.prop_lines import compare, scores
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_prop_lines'").fetchone() is None:
            return [], None
        df = compare(conn)
    if df.empty:
        return [], None
    record = scores(df)
    if record:
        record = {**record, "model": f"{record['model']:.3f}", "market": f"{record['market']:.3f}",
                  "roi": f"{record['roi'] * 100:+.1f}%" if record["roi"] is not None else "–",
                  "range": (f"{record['roi_low'] * 100:+.1f}% to {record['roi_high'] * 100:+.1f}%" if record["roi"] is not None else ""),
                  "expected": f"{record['expected'] * 100:+.1f}%" if record["expected"] is not None else "–",
                  "by_stat": [{"stat": STAT_NAMES[k], "lines": v["lines"], "model": f"{v['model']:.3f}", "market": f"{v['market']:.3f}"}
                              for k, v in record["by_stat"].items()]}
    ahead = df[(df["status"] == "scheduled") & (df["date"] >= today)].copy()
    ahead["gap"] = ahead["model_over"] - ahead["market_over"]
    ahead = ahead.reindex(ahead["gap"].abs().sort_values(ascending=False).index).head(15)
    info = games.set_index("source_game_id")
    rows = []
    for r in ahead.itertuples():
        g = info.loc[r.game] if r.game in info.index else None
        over = r.gap > 0
        price = r.over_price if over else r.under_price
        chance = r.model_over if over else 1 - r.model_over
        rows.append({"name": r.name, "player_url": player_url(r.player_id), "team": abbreviation(r.team),
                     "prop": f"{'Over' if over else 'Under'} {r.line:g} {STAT_NAMES[r.stat]}", "projected": f"{r.projected:.1f}",
                     "model": f"{chance * 100:.0f}%", "market": f"{(r.market_over if over else 1 - r.market_over) * 100:.0f}%",
                     "price": betting_american(price), "edge": f"{(chance * price - 1) * 100:+.0f}%",
                     "matchup": f"{abbreviation(g['away'])} at {abbreviation(g['home'])}" if g is not None else "",
                     "url": game_url(r.game)})
    return rows, record


def betting_american(price: float) -> str:
    from betting import decimal_to_american
    return decimal_to_american(price)


FORM_GAMES = 10       # "going into the game": each team's last 10, the NBA's usual measure
H2H_SHOWN = 10        # the most recent meetings listed (the record counts all of them)


def outcome_letter(team_score, opponent_score) -> str:
    return "W" if team_score > opponent_score else "L"


def form(games: pd.DataFrame, team: str, before: str, game_id) -> dict:
    """A team going into a game: its last FORM_GAMES games (oldest first), its record in them, its
    regular-season record so far, and days of rest."""
    own = games[((games["home"] == team) | (games["away"] == team)) & (games["status"] == "final")
                & (games["date"] < before)].sort_values(["date", "id"])
    if own.empty:
        return {"games": [], "record": None}
    rows = []
    for g in own.tail(FORM_GAMES).itertuples():
        at_home = g.home == team
        ts, os_ = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        rows.append({"outcome": outcome_letter(ts, os_), "score": f"{int(ts)}–{int(os_)}", "venue": "vs" if at_home else "at",
                     "opponent": g.away if at_home else g.home, "url": game_url(g.source_game_id),
                     "date": day_label(g.date)})
    season = int(games.loc[games["id"] == game_id, "season"].iloc[0]) if (games["id"] == game_id).any() else None
    this_season = own[(own["season"] == season) & (own["season_type"] == "regular")] if season is not None else own.iloc[0:0]
    won = lambda df: int(((df["home"] == team) & (df["home_score"] > df["away_score"])).sum()
                         + ((df["away"] == team) & (df["away_score"] > df["home_score"])).sum())
    rest = (pd.Timestamp(before) - pd.Timestamp(own["date"].iloc[-1])).days
    return {"games": rows, "last": f"{sum(r['outcome'] == 'W' for r in rows)}–{sum(r['outcome'] == 'L' for r in rows)}",
            "season": f"{won(this_season)}–{len(this_season) - won(this_season)}" if len(this_season) else None,
            "rest": rest, "b2b": rest == 1, "opener": season is not None and int(own["season"].iloc[-1]) != season}


def head_to_head(games: pd.DataFrame, a: str, b: str, before: str, game_id) -> dict:
    """Every earlier meeting of two teams in the data (since 2002-03): the record, this season's
    meetings, and the most recent H2H_SHOWN."""
    met = games[(((games["home"] == a) & (games["away"] == b)) | ((games["home"] == b) & (games["away"] == a)))
                & (games["status"] == "final") & (games["date"] < before)].sort_values(["date", "id"], ascending=False)
    wins = {a: 0, b: 0}
    meetings = []
    for g in met.itertuples():
        winner = g.home if g.home_score > g.away_score else g.away
        wins[winner] += 1
        if len(meetings) < H2H_SHOWN:
            meetings.append({"date": day_label(g.date, with_year=True), "home": g.home, "away": g.away,
                             "score": f"{int(g.away_score)}–{int(g.home_score)}", "winner": winner,
                             "stage": SEASON_TYPE_LABEL.get(g.season_type, ""), "url": game_url(g.source_game_id)})
    playoff = met[met["season_type"] == "playoffs"]
    return {"meetings": meetings, "count": len(met), "wins_a": wins[a], "wins_b": wins[b], "a": a, "b": b,
            "first": season_label(int(met["season"].min())) if len(met) else None,
            "playoff_games": len(playoff)}


ABSENCE_MINUTES = 10   # absences worth explaining: players who usually play at least this much


def why_pick(g, lineups: dict, cur: dict) -> dict | None:
    """The predicted margin split into its parts, and what each notable absence costs."""
    model = cur["model"]
    playing = {side: lu.loc[~lu["out"], ["player_id", "minutes"]] for side, lu in lineups.items()}
    neutral = bool(g["neutral_site"])
    hb, ab = int(g.get("home_b2b", 0) or 0), int(g.get("away_b2b", 0) or 0)
    parts = breakdown(model, g["home"], g["away"], playing["home"], playing["away"], neutral, hb, ab)
    notes = {"players": [], "rest": ""}
    for side in ("away", "home"):
        lu, team = lineups[side], g[side]
        base = lineup_strength(model, playing[side])
        for r in lu[lu["out"] & (lu["usual_minutes"] >= ABSENCE_MINUTES)].head(4).itertuples():
            back = live.expected_lineup(team, cur["latest"], data()["rosters"], cur["out"] - {r.player_id}, g["date"])
            cost = lineup_strength(model, back.loc[~back["out"], ["player_id", "minutes"]]) - base
            if abs(cost) >= 0.5:
                notes["players"].append(f"{r.name} out ({team} {'−' if cost > 0 else '+'}{abs(cost):.1f})")
    if hb or ab:
        tired = [g[s] for s, b in (("home", hb), ("away", ab)) if b]
        notes["rest"] = f"{' and '.join(tired)} played yesterday"
    rows = [{"label": "Players expected to play", "value": parts["players"],
             "note": "Includes: " + "; ".join(notes["players"]) if notes["players"] else "Ratings of each side's available players, weighted by minutes"},
            {"label": "Team beyond its players", "value": parts["team"], "note": "Coaching, system, fit"},
            {"label": "Home court", "value": parts["home_court"], "note": "Neutral site" if neutral else ""}]
    if hb or ab:
        rows.append({"label": "Back-to-back", "value": parts["rest"], "note": notes["rest"]})
    home_win = g["model_home"] if pd.notna(g.get("model_home")) else None
    if home_win is None:
        return None
    return why(rows, g["home"], g["away"], float(home_win), g.get("market_home"),
               ["The saved prediction may differ slightly from these parts if the injury report changed since it was made."])


@router.get("/game/{source_id}/", response_class=HTMLResponse)
def game_page(request: Request, source_id: str):
    d = data()
    games = d["games"]
    found = games[games["source_game_id"].astype(str) == source_id]
    if found.empty:
        raise HTTPException(status_code=404)
    g = with_predictions(found).iloc[0]
    card = game_card(next(with_predictions(found).itertuples()))
    lines = box_lines([g["id"]]) if g["status"] == "final" else box_lines([])
    proj = d["projections"]
    proj = proj[proj["game_id"] == g["id"]].set_index("player_id") if not proj.empty else pd.DataFrame()
    cur = current()
    sides = []
    lineups = {}
    for side in ("away", "home"):
        team = g[side]
        rows = []
        if g["status"] == "final":
            box = lines[(lines["game_id"] == g["id"]) & (lines["team"] == team) & (lines["minutes"] > 0)].sort_values("minutes", ascending=False)
            for r in box.itertuples():
                p = proj.loc[r.player_id] if r.player_id in proj.index else None
                rows.append({"name": r.name, "url": player_url(r.player_id), "minutes": f"{r.minutes:.0f}", "played": True,
                             "proj_minutes": f"{p['minutes']:.0f}" if p is not None else None,
                             "stats": {s: (f"{getattr(r, s):.0f}" if pd.notna(getattr(r, s)) else "0") for s in STATS},
                             "proj": {s: f"{p[s]:.1f}" for s in STATS} if p is not None else None})
            if not proj.empty:      # projected, but didn't play
                missed = proj[(proj["team"] == team) & ~proj.index.isin(box["player_id"])]
                for pid, p in missed.sort_values("minutes", ascending=False).iterrows():
                    rows.append({"name": p["name"], "url": player_url(pid), "minutes": "0", "played": False, "proj_minutes": f"{p['minutes']:.0f}",
                                 "stats": None, "proj": {s: f"{p[s]:.1f}" for s in STATS}})
            outs = []
        else:
            lineup = (live.expected_lineup(team, cur["latest"], d["rosters"], cur["out"], g["date"])
                      if cur else pd.DataFrame(columns=["player_id", "name", "minutes", "out"]))
            lineups[side] = lineup
            for r in lineup[~lineup["out"]].itertuples():
                p = proj.loc[r.player_id] if r.player_id in proj.index else None
                if p is None and r.minutes < 1:
                    continue
                rating = cur["model"]["ratings"].get(r.player_id) if cur else None
                rows.append({"name": r.name, "url": player_url(r.player_id), "minutes": f"{p['minutes'] if p is not None else r.minutes:.0f}",
                             "rating": signed_number(rating, 1) if rating is not None else "–",
                             "proj": {s: f"{p[s]:.1f}" for s in STATS} if p is not None else None})
            outs = injury_notes(team)
        side_proj = proj[proj["team"] == team] if not proj.empty else proj
        total = {"proj": f"{side_proj['points'].sum():.0f}" if len(side_proj) else None,
                 "actual": int(g[f"{side}_score"]) if g["status"] == "final" else None}
        sides.append({"team": team, "rows": rows, "out": outs, "total": total})
    props = prop_rows(proj.reset_index().assign(game_id=g["id"]), games, "game_id", PROPS_PER_GAME) if not proj.empty else []
    reasons = why_pick(g, lineups, cur) if g["status"] != "final" and cur and len(lineups) == 2 else None
    forms = {side: form(games, g[side], g["date"], g["id"]) for side in ("away", "home")}
    h2h = head_to_head(games, g["away"], g["home"], g["date"], g["id"])
    with connect() as conn:
        moved = betting.movement_chart(conn, int(g["id"]), g["home"], g["away"])
    return render(request, "game.html", {"active": None, "g": card, "date": day_label(g["date"], with_year=True), "why": reasons, "moved": moved, "forms": forms, "h2h": h2h,
                                         "sides": sides, "props": props, "finished": g["status"] == "final",
                                         "props_per_game": PROPS_PER_GAME})


@router.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request):
    return results_page(request, None)


@router.get("/results/{week}/", response_class=HTMLResponse)
def results_week(request: Request, week: str):
    return results_page(request, week)


def results_page(request: Request, week: str | None):
    finished = data()["games"]
    finished = finished[finished["status"] == "final"]
    if finished.empty:
        return render(request, "results.html", {"active": "results", "days": []})
    weeks = sorted(finished["week"].unique())
    if week is None:
        week = weeks[-1]
    elif week not in weeks:
        raise HTTPException(status_code=404)
    i = weeks.index(week)
    shown = with_predictions(finished[finished["week"] == week]).sort_values(["date", "kickoff", "home"])
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]} for day, group in shown.groupby("date")]
    return render(request, "results.html", {
        "active": "results", "days": days, "summary": summary(shown), "label": f"Week of {day_label(week)}",
        "previous_url": f"{BASE}/nba/results/{weeks[i - 1]}/" if i > 0 else None,
        "next_url": f"{BASE}/nba/results/{weeks[i + 1]}/" if i + 1 < len(weeks) else None})


def standings_table(season: int) -> pd.DataFrame:
    """Each team's regular-season record (the NBA Cup final doesn't count), home and away."""
    s = data()["games"]
    s = s[(s["season"] == season) & (s["status"] == "final") & (s["season_type"] == "regular")]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": s[side], "won": (s[f"{side}_score"] > s[f"{other}_score"]).astype(int),
                                  "lost": (s[f"{side}_score"] < s[f"{other}_score"]).astype(int), "side": side,
                                  "pf": s[f"{side}_score"].astype(float), "pa": s[f"{other}_score"].astype(float)}))
    rows = pd.concat(rows, ignore_index=True)
    info = data()["teams"]
    if rows.empty:
        table = pd.DataFrame({"team": info.index, "won": 0, "lost": 0, "home_won": 0, "home_lost": 0, "pf": 0.0, "pa": 0.0})
    else:
        rows["home_won"] = rows["won"] * (rows["side"] == "home")
        rows["home_lost"] = rows["lost"] * (rows["side"] == "home")
        table = rows.groupby("team")[["won", "lost", "home_won", "home_lost", "pf", "pa"]].sum().reset_index()
        table = pd.concat([table, pd.DataFrame({"team": [t for t in info.index if t not in set(table["team"])]})], ignore_index=True).fillna(0)
    table = table.join(info, on="team")
    games_played = (table["won"] + table["lost"]).replace(0, np.nan)
    table["pct"] = (table["won"] / games_played).fillna(0)
    table["diff"] = ((table["pf"] - table["pa"]) / games_played).fillna(0)
    return table.sort_values(["pct", "won", "diff"], ascending=False)


def team_strengths(with_injuries: bool) -> pd.Series:
    """Each team's current strength: points per game better than average, from its players'
    ratings and expected minutes plus its team term (healthy, or with today's injuries)."""
    cur = current()
    if cur is None:
        return pd.Series(dtype=float)
    out = cur["out"] if with_injuries else set()
    today = date.today().isoformat()
    values = {}
    for team in data()["teams"].index:
        lineup = live.expected_lineup(team, cur["latest"], data()["rosters"], out, today)
        values[team] = lineup_strength(cur["model"], lineup) + float(cur["model"]["teams"].get(team, 0.0))
    s = pd.Series(values)
    return s - s.mean()


@router.get("/standings/", response_class=HTMLResponse)
def standings_latest(request: Request, view: str = "conference"):
    return standings_page(request, None, view)


@router.get("/standings/{season}/", response_class=HTMLResponse)
def standings_season(request: Request, season: int, view: str = "conference"):
    return standings_page(request, season, view)


def standings_page(request: Request, season: int | None, view: str):
    games = data()["games"]
    seasons = sorted(s for s in games.loc[games["season_type"] == "regular", "season"].unique() if s >= START_SEASON)
    if not seasons:
        return render(request, "standings.html", {"active": "standings", "groups": []})
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    if view not in ("conference", "division"):
        view = "conference"
    table = standings_table(season)
    rated = team_strengths(False) if season == seasons[-1] else None
    groups = []
    keys = ["East", "West"] if view == "conference" else ["Atlantic", "Central", "Southeast", "Northwest", "Pacific", "Southwest"]
    for key in keys:
        members = table[table[view] == key]
        if members.empty:
            continue
        lead = members.iloc[0]
        rows = []
        for rank, r in enumerate(members.itertuples(), 1):
            gb = ((lead.won - r.won) + (r.lost - lead.lost)) / 2
            rows.append({"rank": rank, "team": r.team, "url": team_url(r.team), "won": int(r.won), "lost": int(r.lost),
                         "pct": f"{r.pct:.3f}".lstrip("0") if r.won + r.lost else "–", "gb": "–" if gb == 0 else f"{gb:g}",
                         "home": f"{int(r.home_won)}–{int(r.home_lost)}",
                         "away": f"{int(r.won - r.home_won)}–{int(r.lost - r.home_lost)}",
                         "diff": signed_number(r.diff, 1) if r.won + r.lost else "–",
                         "rating": signed_number(rated[r.team], 1) if rated is not None and r.team in rated.index else None,
                         "line": view == "conference" and rank in (6, 10)})
        groups.append({"label": key if view == "division" else f"{key}ern Conference", "rows": rows})
    return render(request, "standings.html", {
        "active": "standings", "groups": groups, "season": season_label(season), "is_current": season == seasons[-1],
        "view": view, "base_url": f"{BASE}/nba/standings/{season}/",
        "seasons": [{"label": season_label(s), "url": f"{BASE}/nba/standings/{s}/?view={view}", "selected": s == season}
                    for s in sorted(seasons, reverse=True)]})


@router.get("/playoffs/", response_class=HTMLResponse)
def playoffs_page(request: Request):
    from nba.season import RATING_SD, SIMULATIONS
    from nfl.site import odds_text
    odds = pd.DataFrame()
    if DB_PATH.exists():
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_season_odds'").fetchone():
                odds = pd.read_sql("SELECT * FROM nba_season_odds", conn)
    if odds.empty:
        return render(request, "playoffs.html", {"active": "playoffs", "conferences": []})
    season = int(odds["season"].max())
    odds = odds[odds["season"] == season]
    dates = sorted(odds["as_of"].unique())
    latest = odds[odds["as_of"] == dates[-1]]
    week_ago = [x for x in dates if pd.Timestamp(x) <= pd.Timestamp(dates[-1]) - pd.Timedelta(days=6)]
    before = odds[odds["as_of"] == week_ago[-1]].set_index("team") if week_ago else None
    table = standings_table(season).set_index("team")
    cell = lambda p: {"text": odds_text(p), "value": float(p)}
    conferences = []
    for conf in ("East", "West"):
        rows = []
        for r in latest[latest["conference"] == conf].sort_values(["champion", "playoffs", "wins"], ascending=False).itertuples():
            change = None
            if before is not None and r.team in before.index:
                delta = (r.playoffs - before.at[r.team, "playoffs"]) * 100
                if round(delta) != 0:
                    change = {"text": f"{delta:+.0f}".replace("-", "−"), "dir": "up" if delta > 0 else "down"}
            rec = table.loc[r.team] if r.team in table.index else None
            rows.append({"team": r.team, "url": team_url(r.team), "record": f"{int(rec.won)}–{int(rec.lost)}" if rec is not None else "0–0",
                         "wins": f"{r.wins:.1f}", "change": change,
                         **{k: cell(getattr(r, k)) for k in ("top_six", "playoffs", "top_seed", "conference_finals", "finals", "champion")}})
        conferences.append({"label": f"{conf}ern Conference", "rows": rows})
    return render(request, "playoffs.html", {
        "active": "playoffs", "conferences": conferences, "season": season_label(season), "as_of": day_label(dates[-1]),
        "before_as_of": day_label(week_ago[-1]) if week_ago else None, "simulations": f"{SIMULATIONS:,}", "rating_sd": f"{RATING_SD:g}"})


@router.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request):
    cur = current()
    if cur is None:
        return render(request, "ratings.html", {"active": "ratings", "teams": [], "players": []})
    healthy, now = team_strengths(False), team_strengths(True)
    fmt = lambda v: {"text": signed_number(v, 1), "cls": "good" if round(v, 1) > 0 else "bad" if round(v, 1) < 0 else ""}
    teams = [{"rank": i, "team": t, "url": team_url(t), "conference": data()["teams"].at[t, "conference"],
              "healthy": fmt(healthy[t]), "now": fmt(now[t]), "out": ", ".join(x["name"] for x in injury_notes(t))}
             for i, t in enumerate(healthy.sort_values(ascending=False).index, 1)]
    latest, model = cur["latest"], cur["model"]
    recent = latest[(pd.to_datetime(latest["date"]) >= pd.Timestamp(date.today()) - pd.Timedelta(days=400))
                    & (latest["minutes_after"] >= 15)]
    r = model["ratings"].reindex(recent.index).dropna()
    rows = recent.loc[r.index].assign(rating=r, impact=r * recent.loc[r.index, "minutes_after"] / 48)
    top = rows.sort_values("impact", ascending=False).head(50)
    players_rows = [{"rank": i, "name": p.name, "team": p.team, "url": team_url(p.team), "rating": fmt(p.rating),
                     "minutes": f"{p.minutes_after:.0f}", "impact": fmt(p.impact)}
                    for i, p in enumerate(top.itertuples(), 1)]
    return render(request, "ratings.html", {"active": "ratings", "teams": teams, "players": players_rows,
                                            "as_of": day_label(date.today().isoformat()), "home_adv": f"{model['home_adv']:.1f}"})


def find_team(slug: str) -> str:
    for team in data()["teams"].index:
        if slugify(team) == slug:
            return team
    raise HTTPException(status_code=404)


@router.get("/team/{slug}/", response_class=HTMLResponse)
def team_latest(request: Request, slug: str):
    return team_page(request, slug, None)


@router.get("/team/{slug}/{season}/", response_class=HTMLResponse)
def team_season(request: Request, slug: str, season: int):
    return team_page(request, slug, season)


def team_page(request: Request, slug: str, season: int | None):
    team = find_team(slug)
    d = data()
    games = d["games"]
    own = games[(games["home"] == team) | (games["away"] == team)]
    seasons = sorted(s for s in own["season"].unique() if s >= START_SEASON)
    if not seasons:
        raise HTTPException(status_code=404)
    current_season = seasons[-1]
    if season is None:
        season = current_season
    elif season not in seasons:
        raise HTTPException(status_code=404)
    shown = with_predictions(own[own["season"] == season]).sort_values("date")
    played, coming = shown[shown["status"] == "final"], shown[shown["status"] == "scheduled"]
    reg = played[played["season_type"] == "regular"]
    won = sum((g.home == team) == (g.home_score > g.away_score) for g in reg.itertuples())
    bars = []
    for g in played.dropna(subset=["model_home"]).itertuples():
        at_home = g.home == team
        chance = g.model_home if at_home else 1 - g.model_home
        own_s, opp_s = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        outcome = "W" if own_s > opp_s else "L"
        bars.append({"label": "", "chance": chance, "outcome": outcome,
                     "tip": f"{day_label(g.date, with_year=True)}, {'vs' if at_home else 'at'} {g.away if at_home else g.home}: "
                            f"model {chance * 100:.0f}%, {'won' if outcome == 'W' else 'lost'} {int(own_s)}–{int(opp_s)}"})
    roster = []
    cur = current() if season == current_season else None
    if cur:
        lineup = live.expected_lineup(team, cur["latest"], d["rosters"], cur["out"], date.today().isoformat())
        total = lineup["minutes"].sum()
        notes = {i["name"]: i for i in injury_notes(team)}
        for r in lineup.itertuples():
            rating = cur["model"]["ratings"].get(r.player_id)
            roster.append({"name": r.name, "usual": f"{r.usual_minutes:.0f}",
                           "minutes": f"{r.minutes * 240 / total:.0f}" if total > 0 and not r.out else "0",
                           "rating": signed_number(rating, 1) if rating is not None else "–",
                           "out": notes.get(r.name)})
    strength = team_strengths(True) if cur else None
    info = d["teams"].loc[team] if team in d["teams"].index else None
    return render(request, "team.html", {
        "active": None, "team": team, "season": season_label(season), "is_current": season == current_season,
        "conference": info["conference"] if info is not None else None, "division": info["division"] if info is not None else None,
        "record": f"{won}–{len(reg) - won}", "summary": summary(played),
        "rating": signed_number(strength[team], 1) if strength is not None and team in strength.index else None,
        "chance_chart": chance_bars(bars) if bars else None, "roster": roster,
        "played": [game_card(g, team) for g in played.sort_values("date", ascending=False).itertuples()],
        "coming": [game_card(g, team) for g in coming.dropna(subset=["model_home"]).itertuples()],
        "schedule": [{"date": day_label(g.date, with_year=True), "venue": "vs" if g.home == team else "at",
                      "opponent": g.away if g.home == team else g.home} for g in coming[coming["model_home"].isna()].head(10).itertuples()],
        "seasons": [{"label": season_label(s), "url": f"{team_url(team)}{s}/", "selected": s == season} for s in sorted(seasons, reverse=True)]})


@router.get("/track-record/", response_class=HTMLResponse)
def track_record(request: Request):
    d = data()
    df = d["games"]
    df = df[df["status"] == "final"].merge(d["live"], on="id").merge(d["market"], on="id", how="left").sort_values(["date", "home"])
    weeks = [{"label": f"Week of {day_label(w)}", "summary": summary(group), "games": [game_card(g) for g in group.itertuples()]}
             for w, group in sorted(df.groupby("week"), key=lambda kv: kv[0], reverse=True)]
    chart = None
    if not df.empty:
        use_market = df["market_home"].notna().mean() >= 0.8
        if not use_market:
            if "elo" not in d:
                d["elo"] = elo_benchmark(d["games"])
            df = df.merge(d["elo"].rename("elo_home"), left_on="id", right_index=True, how="left")
        bench = "market_home" if use_market else "elo_home"
        chart_weeks = []
        for w, group in sorted(df.groupby("week"), key=lambda kv: kv[0]):
            g = group.dropna(subset=["model_home", bench])
            won = (g["home_score"] > g["away_score"]).to_numpy()
            chart_weeks.append({"label": f"{pd.Timestamp(w):%-d %b}", "model": np.where(won, g["model_home"], 1 - g["model_home"]),
                                "benchmark": np.where(won, g[bench], 1 - g[bench])})
        chart = record_chart(chart_weeks, "the betting market" if use_market else "a simple Elo rating")
    from charts import calibration_chart
    from web import calibration_pairs
    past = d["games"][(d["games"]["status"] == "final") & (d["games"]["season"] >= HELD_OUT_FROM)].merge(
        d["predictions"][d["predictions"]["source"] == "Backtest"], on="id")
    calib = calibration_chart(calibration_pairs(df) if not df.empty else (np.array([]), np.array([])),
                              calibration_pairs(past), past_from=f"{season_label(HELD_OUT_FROM)}")
    return render(request, "track_record.html", {"active": "track", "weeks": weeks, "overall": summary(df) if not df.empty else None,
                                                 "record_url": RECORD_URL, "record_chart": chart, "calibration": calib})


@router.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    d = data()
    games = d["games"]
    if "elo" not in d:
        d["elo"] = elo_benchmark(games) if not games.empty else pd.Series(dtype=float, name="elo_home")
    df = games[(games["status"] == "final") & (games["season"] >= START_SEASON)].merge(d["predictions"], on="id")
    df = df[df["source"] == "Backtest"].dropna(subset=["model_home"])
    df = df[df["season"] < latest_season()] if latest_season() is not None else df
    if df.empty:
        return render(request, "backtest.html", {"active": "backtest", "seasons": []})
    df = df.merge(d["elo"].rename("elo_home"), left_on="id", right_index=True, how="left")
    won = (df["home_score"] > df["away_score"]).to_numpy()
    loss = lambda p: -np.log(np.clip(np.where(won, p.astype(float), 1 - p.astype(float)), 1e-6, 1))
    df = df.assign(model_ll=loss(df["model_home"]), elo_ll=loss(df["elo_home"].fillna(0.5)))
    held = df[df["season"] >= HELD_OUT_FROM]
    if held.empty:
        held = df
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
        "to": season_label(int(held["season"].max())), "margin_sd": f"{MARGIN_SD:g}", "development": DEVELOPMENT,
        "headline": {"games": len(held), "model": f"{held['model_ll'].mean():.4f}", "elo": f"{held['elo_ll'].mean():.4f}",
                     "model_typical": typical_probability(held["model_ll"].mean()),
                     "elo_typical": typical_probability(held["elo_ll"].mean()),
                     "favorite": f"{((held['model_home'] > 0.5) == won_held).mean() * 100:.1f}%"}})


# ---------------------------------------------------------------- player pages

def player_url(player_id) -> str:
    return f"{BASE}/nba/player/{player_id}/"


templates.env.globals["nba_player_url"] = player_url
GAME_LOG = 15


@router.get("/player/{player_id}/", response_class=HTMLResponse)
def player_page(request: Request, player_id: str):
    d = data()
    if not DB_PATH.exists():
        raise HTTPException(status_code=404)
    with connect() as conn:
        log = pd.read_sql(f"""SELECT p.game_id, p.team, p.name, p.position, p.minutes, p.did_not_play, p.reason,
                                     {', '.join('p.' + s for s in STATS)}, g.date, g.season, i.season_type, i.source_game_id,
                                     h.name AS home, a.name AS away, g.home_score, g.away_score
                              FROM nba_player_games p JOIN games g ON g.id = p.game_id
                              JOIN nba_game_info i ON i.game_id = g.id
                              JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                              WHERE p.player_id = ? ORDER BY g.date""", conn, params=(player_id,))
        has_proj = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nba_player_projections'").fetchone()
        proj = pd.read_sql("SELECT * FROM nba_player_projections WHERE player_id = ?", conn, params=(player_id,)) if has_proj else pd.DataFrame()
    if log.empty and proj.empty:
        raise HTTPException(status_code=404)
    latest = log.iloc[-1] if not log.empty else proj.iloc[-1]
    team = latest["team"]
    played = log[log["minutes"].fillna(0) > 0]
    seasons = []
    for season, part in sorted(played.groupby("season"), key=lambda kv: kv[0], reverse=True)[:3]:
        reg = part[part["season_type"] == "regular"]
        if reg.empty:
            continue
        seasons.append({"label": season_label(int(season)), "games": len(reg), "minutes": f"{reg['minutes'].mean():.1f}",
                        **{s: f"{reg[s].mean():.1f}" for s in STATS}})
    by_game = proj.set_index("game_id") if not proj.empty else pd.DataFrame()
    games_log = []
    for r in log.iloc[::-1].head(GAME_LOG).itertuples():
        at_home = r.home == r.team
        opp = r.away if at_home else r.home
        won = (r.home_score > r.away_score) == at_home
        p = by_game.loc[r.game_id] if r.game_id in by_game.index else None
        games_log.append({"date": day_label(r.date), "opponent": opp, "venue": "vs" if at_home else "at", "url": game_url(r.source_game_id),
                          "result": f"{'W' if won else 'L'} {int(max(r.home_score, r.away_score))}–{int(min(r.home_score, r.away_score))}",
                          "played": bool(r.minutes and r.minutes > 0),
                          "dnp": r.reason if isinstance(r.reason, str) and r.reason else "Didn't play",
                          "minutes": f"{r.minutes:.0f}" if r.minutes else "0",
                          "stats": {s: f"{getattr(r, s):.0f}" if pd.notna(getattr(r, s)) else "0" for s in STATS},
                          "proj": {s: f"{p[s]:.1f}" for s in ["minutes", *STATS]} if p is not None else None})
    upcoming = []
    if not proj.empty:
        games = d["games"].set_index("id")
        for r in proj.itertuples():
            if r.game_id in games.index and games.at[r.game_id, "status"] == "scheduled":
                g = games.loc[r.game_id]
                upcoming.append({"date": day_label(g["date"]), "matchup": f"{g['away']} at {g['home']}", "url": game_url(g["source_game_id"]),
                                 "minutes": f"{r.minutes:.0f}", **{s: f"{getattr(r, s):.1f}" for s in STATS}})
    cur = current()
    rating = cur["model"]["ratings"].get(str(player_id)) if cur else None
    injury = d["injuries"][d["injuries"]["player_id"].astype(str) == str(player_id)]
    return render(request, "player.html", {
        "active": None, "name": latest["name"], "team": team, "position": latest["position"] if isinstance(latest.get("position"), str) else "",
        "rating": signed_number(rating, 1) if rating is not None else None, "seasons": seasons, "log": games_log,
        "upcoming": upcoming, "injury": injury.iloc[0].to_dict() if not injury.empty else None})
