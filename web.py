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
templates.env.globals["sports"] = [("Today", "today", f"{BASE}/today/"), ("Premier League", "epl", f"{BASE}/"), ("NFL", "nfl", f"{BASE}/nfl/"),
                                   ("NBA", "nba", f"{BASE}/nba/"), ("College football", "cfb", f"{BASE}/cfb/"),
                                   ("College basketball", "cbb", f"{BASE}/cbb/"),
                                   ("Betting", "betting", f"{BASE}/betting/")]

# Team colors (colors.py): style="..." strings that set --home/--away (a game) or --team (a team page)
import colors  # noqa: E402

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
