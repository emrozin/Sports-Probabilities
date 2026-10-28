"""What every sport's web pages share: the page templates, the site's base path and formatting helpers.

app.py (Premier League) and nfl/site.py (NFL) both build their pages with these.
"""

import math
import os
import re
from pathlib import Path

import pandas as pd
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("SITE_BASE", "").rstrip("/")  # "" locally, the repository name on GitHub Pages

templates = Jinja2Templates(directory=ROOT / "templates")
templates.env.globals["base"] = BASE
# The sport menu in the header: (label, sport code, home page)
templates.env.globals["sports"] = [("Today", "today", f"{BASE}/"), ("Soccer", "soccer", f"{BASE}/premier-league/"),
                                   ("NFL", "nfl", f"{BASE}/nfl/"),
                                   ("NBA", "nba", f"{BASE}/nba/"), ("College football", "cfb", f"{BASE}/cfb/"),
                                   ("College basketball", "cbb", f"{BASE}/cbb/"),
                                   ("Betting", "betting", f"{BASE}/betting/")]

# Team colors (colors.py): style="..." strings that set --home/--away (a game) or --team (a team page)
import colors  # noqa: E402
from ucl.cups import SWITCH  # noqa: E402

# The Soccer tab's competition switcher (Premier League, Europe's other big leagues, the European cups)
templates.env.globals["soccer_switch"] = SWITCH

templates.env.globals["nfl_colors"] = lambda home, away: colors.style(colors.NFL, home, away)
templates.env.globals["epl_colors"] = lambda home, away: colors.style(colors.EPL, home, away)
templates.env.globals["nfl_team_colors"] = lambda team: colors.team_style(colors.NFL, team)
templates.env.globals["epl_team_colors"] = lambda team: colors.team_style(colors.EPL, team)
# a small square in a team's main color, before its name in tables
templates.env.globals["nfl_chip"] = lambda team: colors.chip(colors.NFL, team)
templates.env.globals["epl_chip"] = lambda team: colors.chip(colors.EPL, team)
templates.env.globals["cfb_colors"] = lambda home, away: colors.style(colors.CFB, home, away)
templates.env.globals["cfb_team_colors"] = lambda team: colors.team_style(colors.CFB, team)
templates.env.globals["cfb_chip"] = lambda team: colors.chip(colors.CFB, team)


def edge(model: dict | None, market: dict | None, sides: dict, threshold: int = 5) -> dict | None:
    """Where the model and the betting market disagree by at least `threshold` percentage points on
    a team's chance of winning: {"team", "points"} for the team the model rates higher, else None.
    sides maps 'home'/'away' to team names."""
    if not model or not market:
        return None
    best = max(("home", "away"), key=lambda s: model[s] - market[s])
    gap = model[best] - market[best]
    return {"team": sides[best], "points": int(gap)} if gap >= threshold else None



def slugify(name: str) -> str:
    """'Nott'm Forest' -> 'nottm-forest'"""
    name = name.lower().replace("'", "").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def day_label(iso_date: str, with_year: bool = False) -> str:
    """'2026-10-03' -> 'Saturday 3 October' (or 'Sat 3 Oct 2026' with_year)"""
    d = pd.Timestamp(iso_date)
    if with_year:
        return f"{d:%a} {d.day} {d:%b %Y}"
    return f"{d:%A} {d.day} {d:%B}"


def odds_text(p) -> str:
    """A season chance in a table: '37%', '<1%', '>99%', '✓' when certain, '–' when impossible or unknown
    (the same everywhere: the Premier League's and the other soccer tables)."""
    if p is None or pd.isna(p) or p == 0:
        return "–"
    if p > 0.9995:
        return "✓"
    if p > 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p * 100:.0f}%"


def ordinal(n: int) -> str:
    """1 -> '1st', 12 -> '12th', 22 -> '22nd'"""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def signed_number(value: float, decimals: int = 0) -> str:
    """Goal or point difference style: +7, \u22123, 0"""
    text = f"{abs(value):.{decimals}f}"
    if round(value, decimals) > 0:
        return "+" + text
    if round(value, decimals) < 0:
        return "\u2212" + text
    return text


def typical_probability(log_loss: float) -> str:
    """Log loss as a percentage: the typical probability given to what actually happened.

    One decimal place, because the model and the market are often less than a point apart.
    """
    return f"{math.exp(-log_loss) * 100:.1f}"


templates.env.globals["signed_number"] = signed_number


def why(parts: list[dict], home: str, away: str, home_win: float, market_home: float | None = None,
        notes: list[str] | None = None) -> dict | None:
    """'Why this pick': the predicted home margin split into its parts, for a diverging bar chart.
    parts: [{"label", "value" (points, + toward the home team), "note" (optional)}]. home_win: the
    model's home win chance as shown on the page; market_home: the market's, if known."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    margin = sum(p["value"] for p in parts)
    biggest = max([abs(p["value"]) for p in parts] + [abs(margin), 0.5])
    fav_home = home_win >= 0.5
    fav = home if fav_home else away
    rows = [{"label": p["label"], "note": p.get("note", ""), "team": home if p["value"] > 0 else away,
             "points": f"{abs(p['value']):.1f}", "home": p["value"] > 0, "zero": abs(p["value"]) < 0.05,
             "width": f"{abs(p['value']) / biggest * 100:.0f}%"} for p in parts]
    total = {"label": "Expected margin", "team": home if margin > 0 else away, "points": f"{abs(margin):.1f}",
             "home": margin > 0, "zero": abs(margin) < 0.05, "width": f"{abs(margin) / biggest * 100:.0f}%"}
    out = {"rows": rows, "total": total, "home": home, "away": away, "favorite": fav,
           "chance": f"{max(home_win, 1 - home_win) * 100:.0f}%", "margin": f"{abs(margin):.1f}",
           "notes": notes or [], "market": None, "gap": None}
    if market_home is not None and not math.isnan(market_home):
        q = market_home if fav_home else 1 - market_home
        out["market"] = f"{q * 100:.0f}%"
        gap = (max(home_win, 1 - home_win) - q) * 100
        if abs(gap) >= 10:
            out["gap"] = f"{abs(gap):.0f}"
    return out


def calibration_pairs(df: pd.DataFrame) -> tuple:
    """(home win chances, home won) for finished, decided games with a prediction."""
    df = df.dropna(subset=["model_home"])
    df = df[df["home_score"] != df["away_score"]]
    return df["model_home"].to_numpy(float), (df["home_score"] > df["away_score"]).to_numpy(float)


def disk_cached(folder: Path, name: str, key, make):
    """make() computed once per key and kept as a file in `folder` (data/cache next to the database),
    so a page that needs a slow model fit only pays for it once per update: update.py's last step
    (warm_cache.py) fills these, and every visit after that reads the file. Any problem reading or
    writing the file just means computing it again."""
    import pickle
    path = Path(folder) / f"{name}.pkl"
    try:
        if path.exists():
            stored = pickle.loads(path.read_bytes())
            if stored.get("key") == key:
                return stored["value"]
    except Exception:
        pass
    value = make()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"key": key, "value": value}))
        tmp.replace(path)
    except Exception:
        pass
    return value
