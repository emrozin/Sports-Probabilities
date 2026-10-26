"""Pages across all sports (included by app.py): /today/ (every game today, the biggest
disagreements with the market, suggested bets and props) and /betting/ (the betting rule's
suggestions, stakes and record: betting.py)."""

import sqlite3
from contextlib import closing
from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

import betting
from web import BASE, ROOT, day_label, slugify, templates

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter()
_cache: dict = {}


def connect():
    return closing(sqlite3.connect(DB_PATH))


def now_eastern() -> datetime:
    from nfl.predict_upcoming import EASTERN
    return datetime.now(EASTERN)


def updated_at() -> str | None:
    """When the database last changed (the last update), for the page footers."""
    if not DB_PATH.exists():
        return None
    stamp = datetime.fromtimestamp(DB_PATH.stat().st_mtime)
    return f"{day_label(stamp.date().isoformat())}, {stamp:%-I:%M %p}"


templates.env.globals["data_updated"] = updated_at


def cached(key: str, make):
    """A value computed once per version of the database."""
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if _cache.get("stamp") != stamp:
        _cache.clear()
        _cache["stamp"] = stamp
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def american(price: float) -> str:
    return betting.decimal_to_american(price)


def pick_label(row) -> str:
    return "Draw" if row["outcome"] == "draw" else row[row["outcome"]]


def game_link(league: str, row) -> str:
    """The best page for a game: its own page where the sport has one."""
    if league == "EPL":
        return f"{BASE}/match/{row['date']}/{slugify(row['home'])}-v-{slugify(row['away'])}/"
    with connect() as conn:
        if league == "NFL":
            found = conn.execute("SELECT source_game_id FROM nfl_game_info WHERE game_id = ?", (int(row["game_id"]),)).fetchone()
            if found:
                return f"{BASE}/nfl/game/{found[0].lower().replace('_', '-')}/"
        if league == "NBA":
            found = conn.execute("SELECT source_game_id FROM nba_game_info WHERE game_id = ?", (int(row["game_id"]),)).fetchone()
            if found:
                return f"{BASE}/nba/game/{found[0]}/"
    return f"{BASE}{betting.sports()[league]['url']}"


def kickoff_text(kickoff) -> str:
    if not isinstance(kickoff, str) or not kickoff:
        return "Time TBD"
    t = datetime.strptime(kickoff, "%H:%M")
    return f"{t:%-I:%M %p}".replace(":00 ", " ") + " ET"


# ---------------------------------------------------------------- the betting verdicts

def histories() -> dict:
    def make():
        if not DB_PATH.exists():
            return {}
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'games'").fetchone() is None:
                return {}
            return {league: betting.history(conn, league) for league in betting.sports()}
    return cached("histories", make)


def verdict(league: str) -> dict:
    """What the past-season test says about betting this sport's suggestions."""
    h = histories().get(league)
    if h is None:
        return {"status": "untested", "text": "Not tested yet: no past prices. The live record will tell."}
    if h["roi_high"] < 0:
        return {"status": "losing", "text": f"Lost {abs(h['roi']) * 100:.1f}% per bet on past seasons: not recommended."}
    if h["roi_low"] > 0:
        return {"status": "winning", "text": f"Made {h['roi'] * 100:.1f}% per bet on past seasons."}
    return {"status": "unclear", "text": f"{h['roi'] * 100:+.1f}% per bet on past seasons, within what luck could do: not recommended."}


def suggestion_rows(picks: pd.DataFrame, bankroll: float, fraction: float) -> list[dict]:
    rows = []
    for _, r in picks.iterrows():
        v = verdict(r["league"])
        rule_stake = float(betting.stake_share(r["kelly"], fraction)) * bankroll
        rows.append({"sport": betting.sports()[r["league"]]["name"], "league": r["league"], "date": day_label(r["date"]),
                     "time": kickoff_text(r["kickoff"]), "matchup": f"{r['away']} at {r['home']}", "url": game_link(r["league"], r),
                     "pick": pick_label(r), "price": american(r["price"]), "decimal": f"{r['price']:.2f}",
                     "model": f"{r['p'] * 100:.0f}%", "market": f"{r['market'] * 100:.0f}%", "edge": f"{r['edge'] * 100:+.1f}%",
                     "rule_stake": f"${rule_stake:,.0f}", "recommended": f"${rule_stake:,.0f}" if v["status"] == "winning" else "$0",
                     "status": v["status"]})
    return rows


# ---------------------------------------------------------------- /betting/

def experiment_rows() -> list[dict]:
    """What was tried to improve the rule (betting_experiments.py), for the Betting page."""
    from betting_experiments import RECORD_PATH as path
    if not path.exists():
        return []
    df = pd.read_csv(path)
    signed = lambda x: f"{x * 100:+.1f}%".replace("-", "−")
    return [{"sport": r["sport"], "rule": r["rule"], "seasons": r["seasons"], "bets": f"{int(r['bets']):,}",
             "roi": signed(r["roi"]) if r["bets"] else "–",
             "range": f"{signed(r['roi_low'])} to {signed(r['roi_high'])}" if r["bets"] else "",
             "kelly": f"${r['kelly_final']:,.0f}", "kept": str(r["kept"]) == "True",
             "in_sample": str(r.get("in_sample")) == "True", "losing": r["bets"] > 0 and r["roi_high"] < 0,
             "note": r["note"] if isinstance(r["note"], str) else ""} for _, r in df.iterrows()]



@router.get("/betting/", response_class=HTMLResponse)
def betting_page(request: Request, bankroll: float = 1000.0, fraction: float = betting.KELLY_FRACTION):
    bankroll = float(np.clip(bankroll, 10, 10_000_000))
    fraction = float(np.clip(fraction, 0.05, 1.0))
    picks = pd.DataFrame()
    log = pd.DataFrame()
    if DB_PATH.exists():
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'games'").fetchone():
                conn.executescript(open(ROOT / "schema.sql").read())
                picks = betting.suggestions(conn)
                log = betting.logged(conn)
    tests = []
    for league, h in histories().items():
        if h is None:
            continue
        tests.append({"sport": h["name"], "seasons": f"{h['seasons'][0]}–{h['seasons'][1]}", "bets": f"{h['bets']:,}",
                      "games": f"{h['games']:,}", "won": f"{h['won'] / h['bets'] * 100:.0f}%",
                      "average_price": american(h["average_price"]), "expected": f"{h['expected'] * 100:+.1f}%",
                      "roi": f"{h['roi'] * 100:+.1f}%", "range": f"{h['roi_low'] * 100:+.1f}% to {h['roi_high'] * 100:+.1f}%",
                      "kelly": f"${h['kelly_final']:,.0f}", "drawdown": f"{h['drawdown'] * 100:.0f}%",
                      "clv": f"{h['clv'] * 100:+.1f}%" if h["clv"] is not None else "–",
                      "buckets": [{**b, "expected": f"{b['expected'] * 100:+.1f}%", "roi": f"{b['roi'] * 100:+.1f}%"} for b in h["buckets"]],
                      "assumed_margin": h["assumed_margin"], "losing": h["roi_high"] < 0})
    record = None
    if not log.empty:
        done = log.dropna(subset=["result"])
        recent = log.sort_values(["date", "kickoff"], ascending=False).head(40)
        record = {"logged": len(log), "settled": len(done), "won": int((done["result"] == "won").sum()),
                  "lost": int((done["result"] == "lost").sum()), "push": int((done["result"] == "push").sum()),
                  "units": f"{done['profit'].sum():+.1f}", "roi": f"{done['profit'].mean() * 100:+.1f}%" if len(done) else "–",
                  "clv": f"{done['clv'].mean() * 100:+.1f}%" if done["clv"].notna().any() else "–",
                  "rows": [{"date": day_label(r["date"]), "sport": betting.sports()[r["league"]]["name"],
                            "matchup": f"{r['away']} at {r['home']}", "pick": pick_label(r), "price": american(r["price"]),
                            "edge": f"{r['edge'] * 100:+.1f}%", "result": r["result"] if isinstance(r["result"], str) else "pending",
                            "profit": f"{r['profit']:+.2f}" if pd.notna(r["profit"]) else "",
                            "clv": f"{r['clv'] * 100:+.1f}%" if pd.notna(r["clv"]) else "–"} for _, r in recent.iterrows()]}
    any_winning = any(verdict(l)["status"] == "winning" for l in betting.sports())
    return templates.TemplateResponse(request, "betting.html", {
        "sport": "betting", "active": "betting", "bankroll": f"{bankroll:g}", "fraction": fraction,
        "fractions": [(0.1, "a tenth"), (0.25, "a quarter"), (0.5, "half"), (1.0, "full")],
        "suggestions": suggestion_rows(picks, bankroll, fraction), "tests": tests, "record": record,
        "experiments": experiment_rows(),
        "any_winning": any_winning, "min_edge": f"{betting.MIN_EDGE * 100:.0f}%", "cap": f"{betting.MAX_STAKE * 100:.0f}%",
        "verdicts": {betting.sports()[l]["name"]: verdict(l) for l in betting.sports()}})


# ---------------------------------------------------------------- /today/

def today_games() -> pd.DataFrame:
    """Every game today in every sport, with the model's favorite and the market's chance for it."""
    today = now_eastern().date().isoformat()
    out = []
    if not DB_PATH.exists():
        return pd.DataFrame()
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'games'").fetchone() is None:
            return pd.DataFrame()
        for league, info in betting.sports().items():
            rows = betting.games(conn, league)
            rows = rows[rows["date"] == today]
            if rows.empty:
                continue
            chances = betting.model_chances(conn, info["live"])
            chances = chances[chances["game_id"].isin(rows["game_id"])]
            fav = chances.sort_values("p", ascending=False).drop_duplicates("game_id")
            offered = betting.prices(conn, league)
            offered = offered[offered["game_id"].isin(rows["game_id"])]
            if not offered.empty:
                implied = 1 / offered["price"]
                offered = offered.assign(market=implied / implied.groupby(offered["game_id"]).transform("sum"))
            merged = rows.merge(fav, on="game_id", how="left")
            if not offered.empty:
                merged = merged.merge(offered[["game_id", "outcome", "market"]], on=["game_id", "outcome"], how="left")
            else:
                merged["market"] = np.nan
            out.append(merged.assign(league=league))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


@router.get("/today/", response_class=HTMLResponse)
def today_page(request: Request):
    from nfl.predict_upcoming import has_started
    now = now_eastern()
    df = today_games()
    groups, gaps = [], []
    if not df.empty:
        for league, info in betting.sports().items():
            part = df[df["league"] == league].sort_values(["kickoff", "home"], na_position="last")
            if part.empty:
                continue
            rows = []
            for _, r in part.iterrows():
                finished = r["status"] == "final"
                started = finished or has_started(r["date"], r["kickoff"] if isinstance(r["kickoff"], str) else None, now)
                row = {"time": kickoff_text(r["kickoff"]), "matchup": f"{r['away']} at {r['home']}", "url": game_link(league, r),
                       "finished": finished, "started": started,
                       "score": f"{int(r['away_score'])}–{int(r['home_score'])}" if finished else None,
                       "pick": pick_label(r) if pd.notna(r.get("p")) else None,
                       "model": f"{r['p'] * 100:.0f}%" if pd.notna(r.get("p")) else None,
                       "market": f"{r['market'] * 100:.0f}%" if pd.notna(r.get("market")) else None}
                if finished and row["pick"]:
                    w = betting.winner(r)
                    row["pick_won"] = w == r["outcome"]
                rows.append(row)
                if not started and pd.notna(r.get("p")) and pd.notna(r.get("market")):
                    gaps.append({**row, "sport": info["name"], "gap": abs(r["p"] - r["market"]),
                                 "gap_text": f"{(r['p'] - r['market']) * 100:+.0f} pts"})
            groups.append({"sport": info["name"], "url": f"{BASE}{info['url']}", "rows": rows})
    gaps = sorted(gaps, key=lambda g: g["gap"], reverse=True)[:5]
    picks = pd.DataFrame()
    if DB_PATH.exists():
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'games'").fetchone():
                picks = betting.suggestions(conn, now)
    if not picks.empty:
        picks = picks[picks["date"] == now.date().isoformat()]
    return templates.TemplateResponse(request, "today.html", {
        "sport": "today", "active": "today", "date": day_label(now.date().isoformat()), "groups": groups, "gaps": gaps,
        "bets": suggestion_rows(picks, 1000.0, betting.KELLY_FRACTION), "props": today_props(now.date().isoformat())})


def today_props(today: str) -> list[dict]:
    """The likeliest player props today: NBA and NFL."""
    rows = []
    try:
        import nba.site as nba_site
        d = nba_site.data()
        proj = d["projections"]
        if not proj.empty:
            games = d["games"]
            ids = set(games.loc[(games["date"] == today) & (games["status"] == "scheduled"), "id"])
            part = proj[proj["game_id"].isin(ids)]
            rows += [{**p, "sport": "NBA"} for p in nba_site.prop_rows(part, games, "date", 6)]
    except Exception:
        pass
    try:
        import nfl.site as nfl_site
        games = nfl_site.data()["games"]
        ids = [int(i) for i in games.loc[(games["date"] == today) & (games["status"] == "scheduled"), "id"]]
        if ids:
            rows += [{**p, "sport": "NFL"} for p in nfl_site.prop_rows(nfl_site.likeliest(ids, "date", 6))]
    except Exception:
        pass
    return rows
