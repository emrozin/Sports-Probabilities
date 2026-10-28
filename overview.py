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
from web import BASE, ROOT, day_label, disk_cached, slugify, templates

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


def suggestions_now(now: datetime | None = None) -> pd.DataFrame:
    """The bets the rule suggests now: worked out at most every five minutes (until the data changes)."""
    now = now or now_eastern()
    bucket = f"{now:%Y%m%d%H}{now.minute // 5}"

    def make():
        if not DB_PATH.exists():
            return pd.DataFrame()
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'games'").fetchone() is None:
                return pd.DataFrame()
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'bets'").fetchone() is None:
                conn.executescript(open(ROOT / "schema.sql").read())
            return betting.suggestions(conn, now)
    return cached(f"suggestions {bucket}", make)


def bet_log() -> pd.DataFrame:
    def make():
        if not DB_PATH.exists():
            return pd.DataFrame()
        with connect() as conn:
            if conn.execute("SELECT name FROM sqlite_master WHERE name = 'bets'").fetchone() is None:
                return pd.DataFrame()
            return betting.logged(conn)
    return cached("bet log", make)


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
        if league in ("UCL", "UEL", "UECL", "ESP", "GER", "ITA", "FRA"):
            from ucl.cups import CUPS
            found = conn.execute("SELECT espn_id FROM cup_game_info WHERE game_id = ?", (int(row["game_id"]),)).fetchone()
            if found:
                return f"{BASE}{CUPS[league]['url']}match/{found[0]}/"
        if league in ("CFB", "CBB"):
            table = "cfb_game_info" if league == "CFB" else "cbb_game_info"
            found = conn.execute(f"SELECT source_game_id FROM {table} WHERE game_id = ?", (int(row["game_id"]),)).fetchone()
            if found:
                return f"{BASE}/{league.lower()}/game/{found[0]}/"
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
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    return cached("histories", lambda: disk_cached(DB_PATH.parent / "cache", "betting_histories", (stamp, "v1"), make))


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
                     "teams": f"{r['league'].lower()}:{r['home']}|{r['league'].lower()}:{r['away']}",
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


def spread_total_rows() -> list[dict]:
    """Spreads and totals on the held-out seasons (betting_lines.py), for the Betting page."""
    from betting_lines import RECORD_PATH as path
    if not path.exists():
        return []
    df = pd.read_csv(path)
    signed = lambda x: f"{x * 100:+.1f}%".replace("-", "−")
    return [{"name": r["name"], "seasons": str(r["seasons"]).replace("-", "–"), "bets": f"{int(r['bets']):,}",
             "expected": signed(r["expected"]), "roi": signed(r["roi"]),
             "range": f"{signed(r['roi_low'])} to {signed(r['roi_high'])}", "kelly": f"${r['bankroll']:,.0f}",
             "losing": r["roi_high"] < 0, "model_loss": f"{r['model_loss']:.3f}", "market_loss": f"{r['market_loss']:.3f}",
             "beat_market": r["model_loss"] < r["market_loss"]} for _, r in df.iterrows()]


@router.get("/betting/", response_class=HTMLResponse)
def betting_page(request: Request, bankroll: float = 1000.0, fraction: float = betting.KELLY_FRACTION):
    bankroll = float(np.clip(bankroll, 10, 10_000_000))
    fraction = float(np.clip(fraction, 0.05, 1.0))
    picks = suggestions_now()
    log = bet_log()
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
        "experiments": experiment_rows(), "spreads_totals": spread_total_rows(),
        "any_winning": any_winning, "min_edge": f"{betting.MIN_EDGE * 100:.0f}%", "cap": f"{betting.MAX_STAKE * 100:.0f}%",
        "verdicts": {betting.sports()[l]["name"]: verdict(l) for l in betting.sports()}})


# ---------------------------------------------------------------- /methodology/

@router.get("/methodology/", response_class=HTMLResponse)
def methodology_page(request: Request):
    import json
    from ucl.run_backtest import SUMMARY_PATH
    cups = []
    if SUMMARY_PATH.exists():
        from ucl.cups import CUPS
        s = json.loads(SUMMARY_PATH.read_text())
        for code, c in s.get("competitions", {}).items():
            h = c["held_out"]
            last = max(x["season"] for x in c["seasons"])
            if h.get("model"):
                cups.append({"name": CUPS[code]["name"], "model": f"{h['model']:.3f}", "elo": f"{h['elo']:.3f}",
                             "seasons": f"{s['held_out_from']}-{(s['held_out_from'] + 1) % 100:02d} to {last}-{(last + 1) % 100:02d}"})
    return templates.TemplateResponse(request, "methodology.html", {"sport": "today", "active": "methodology", "cups": cups})


# ---------------------------------------------------------------- /today/

def today_games() -> pd.DataFrame:
    """Every game today in every sport, with the model's favorite and the market's chance for it."""
    today = now_eastern().date().isoformat()
    return cached(f"today games {today}", lambda: _today_games(today))


def _today_games(today: str) -> pd.DataFrame:
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
                       "teams": f"{league.lower()}:{r['home']}|{league.lower()}:{r['away']}",
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
    picks = suggestions_now(now)
    if not picks.empty:
        picks = picks[picks["date"] == now.date().isoformat()]
    return templates.TemplateResponse(request, "today.html", {
        "sport": "today", "active": "today", "date": day_label(now.date().isoformat()), "groups": groups, "gaps": gaps,
        "bets": suggestion_rows(picks, 1000.0, betting.KELLY_FRACTION), "props": today_props(now.date().isoformat())})


def today_props(today: str) -> list[dict]:
    """The likeliest player props today: NBA and NFL."""
    return cached(f"today props {today}", lambda: _today_props(today))


def _today_props(today: str) -> list[dict]:
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


# ---------------------------------------------------------------- /search/

def search_index() -> list[dict]:
    """Every team and recent player the site has a page for: name, what it is, sport, link."""
    def make():
        items = []
        if not DB_PATH.exists():
            return items
        has = lambda conn, t: conn.execute("SELECT name FROM sqlite_master WHERE name = ?", (t,)).fetchone() is not None
        with connect() as conn:
            if not has(conn, "games"):
                return items
            for league, label, url in (("EPL", "Premier League", lambda t: f"{BASE}/team/{slugify(t)}/"),
                                       ("NBA", "NBA", lambda t: f"{BASE}/nba/team/{slugify(t)}/")):
                for (team,) in conn.execute("""SELECT DISTINCT t.name FROM teams t JOIN leagues l ON l.id = t.league_id AND l.code = ?
                                               JOIN games g ON g.home_team_id = t.id
                                               WHERE g.season >= (SELECT MAX(season) FROM games WHERE league_id = l.id) - 3""", (league,)):
                    items.append({"name": team, "kind": "Team", "sport": label, "url": url(team)})
            if has(conn, "cup_game_info"):
                from ucl.cups import CUPS
                from ucl.site import team_url as cup_team_url
                for code, cup in CUPS.items():
                    season = conn.execute("""SELECT MAX(g.season) FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = ?""",
                                          (code,)).fetchone()[0]
                    for (team,) in conn.execute("""SELECT DISTINCT t.name FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = ?
                                                   JOIN teams t ON t.id IN (g.home_team_id, g.away_team_id) WHERE g.season = ?""",
                                                (code, season)):
                        items.append({"name": team, "kind": "Team", "sport": cup["name"], "url": cup_team_url(team, code)})
            from nfl.site import team_url as nfl_team_url
            from nfl.teams import TEAMS
            items += [{"name": name, "kind": "Team", "sport": "NFL", "url": nfl_team_url(code)} for code, (name, _) in TEAMS.items()]
            if has(conn, "cfb_game_info"):
                for (team,) in conn.execute("""SELECT DISTINCT t.name FROM games g JOIN cfb_game_info i ON i.game_id = g.id
                                               JOIN teams t ON t.id = g.home_team_id
                                               WHERE i.home_division = 'fbs' AND g.season >= (SELECT MAX(season) FROM games
                                                     WHERE id IN (SELECT game_id FROM cfb_game_info)) - 1"""):
                    items.append({"name": team, "kind": "Team", "sport": "College football", "url": f"{BASE}/cfb/team/{slugify(team)}/"})
            if has(conn, "cbb_game_info"):
                for (team,) in conn.execute("""SELECT DISTINCT t.name FROM games g JOIN cbb_game_info i ON i.game_id = g.id
                                               JOIN teams t ON t.id = g.home_team_id
                                               WHERE i.home_conference IS NOT NULL AND g.season >= (SELECT MAX(season) FROM games
                                                     WHERE id IN (SELECT game_id FROM cbb_game_info)) - 1"""):
                    items.append({"name": team, "kind": "Team", "sport": "College basketball", "url": f"{BASE}/cbb/team/{slugify(team)}/"})
            if has(conn, "nba_player_games"):
                rows = conn.execute("""SELECT p.player_id, p.name, p.team, MAX(g.date) FROM nba_player_games p JOIN games g ON g.id = p.game_id
                                       WHERE g.season >= (SELECT MAX(season) FROM games WHERE id IN (SELECT game_id FROM nba_game_info)) - 1
                                       AND p.minutes > 0 GROUP BY p.player_id""").fetchall()
                items += [{"name": name, "kind": "Player", "sport": "NBA", "detail": team, "url": f"{BASE}/nba/player/{pid}/"}
                          for pid, name, team, _ in rows]
            if has(conn, "nfl_player_stats"):
                rows = conn.execute("""SELECT s.player_id, s.name, s.position, s.team, MAX(g.date) FROM nfl_player_stats s
                                       JOIN games g ON g.id = s.game_id
                                       WHERE g.season >= (SELECT MAX(season) FROM games WHERE id IN (SELECT game_id FROM nfl_game_info)) - 1
                                       GROUP BY s.player_id""").fetchall()
                from nfl.site import team_name as nfl_team_name
                items += [{"name": name, "kind": "Player", "sport": "NFL", "detail": f"{pos or ''} {nfl_team_name(team)}".strip(),
                           "url": f"{BASE}/nfl/player/{pid}/"} for pid, name, pos, team, _ in rows]
        return items
    return cached("search index", make)


def fold(text: str) -> str:
    """'Jokić' -> 'jokic': lowercase, accents and punctuation removed, for matching."""
    import unicodedata
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return "".join(ch if ch.isalnum() else " " for ch in text).strip()


def search(query: str, limit: int = 40) -> list[dict]:
    q = fold(query)
    if len(q) < 2:
        return []
    words = q.split()
    scored = []
    for item in search_index():
        name = fold(item["name"])
        if all(w in name for w in words):
            starts = name.startswith(q) or any(part.startswith(q) for part in name.split())
            scored.append((0 if name == q else 1 if starts else 2, item["kind"] != "Team", len(name), item))
    scored.sort(key=lambda s: s[:3])
    return [s[3] for s in scored[:limit]]


@router.get("/search/", response_class=HTMLResponse)
def search_page(request: Request, q: str = ""):
    results = search(q)
    return templates.TemplateResponse(request, "search.html", {"sport": "today", "active": None, "query": q,
                                                               "results": results})
