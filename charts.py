"""Small charts for the team pages, drawn as HTML and SVG on the server (no JavaScript library).

Each function turns data into positions (percentages of the chart's width and height); the
templates in templates/charts.html draw them, so the charts stay sharp at any width.
"""

import math

import numpy as np


def nice_limit(values, step: float = 2.0, minimum: float = 4.0) -> float:
    """A symmetric axis limit around 0 that fits every value, rounded up to a multiple of step."""
    biggest = max([abs(v) for v in values] + [minimum])
    return math.ceil(biggest / step) * step


def line_chart(points: list[dict], step: float = 2.0, minimum: float = 4.0) -> dict | None:
    """points: [{"label", "value", "tip"}] in order. Values are centered on 0 (an average team).
    Returns the SVG path and each point's position for the hover markers."""
    if len(points) < 2:
        return None
    limit = nice_limit([p["value"] for p in points], step, minimum)
    n = len(points)
    placed = []
    for i, p in enumerate(points):
        x = i / (n - 1) * 100
        y = (limit - p["value"]) / (2 * limit) * 100
        placed.append({**p, "x": round(x, 2), "y": round(y, 2)})
    path = "M " + " L ".join(f"{p['x']} {p['y']}" for p in placed)
    ticks = [{"value": v, "y": round((limit - v) / (2 * limit) * 100, 2)}
             for v in (limit, limit / 2, 0, -limit / 2, -limit)]
    # x labels: no more than about 9, always the first and last
    every = max(1, math.ceil(n / 9))
    labels = [p for i, p in enumerate(placed) if i % every == 0 or i == n - 1]
    if len(labels) > 1 and labels[-1] is not labels[-2] and placed.index(labels[-1]) - placed.index(labels[-2]) < every / 2:
        labels.pop(-2)
    return {"points": placed, "path": path, "ticks": ticks, "labels": labels, "limit": limit}


def chance_bars(games: list[dict]) -> list[dict]:
    """games: [{"label", "chance" (0-1, the model's chance of this team winning), "outcome" (W/D/L/T),
    "tip"}]. Adds the bar height in percent."""
    return [{**g, "height": round(g["chance"] * 100, 1)} for g in games]


def record_chart(weeks: list[dict], benchmark: str) -> dict | None:
    """The running track record: after each week, the typical probability the model gave the actual
    result so far, next to a benchmark's on the same games.

    weeks: [{"label", "model": [probabilities given to what happened], "benchmark": [same games']}],
    oldest first. Games without a benchmark probability should be left out of both lists."""
    total_m = total_b = count = 0.0
    points = []
    for w in weeks:
        m = np.clip(np.asarray(w["model"], float), 1e-6, 1)
        b = np.clip(np.asarray(w["benchmark"], float), 1e-6, 1)
        if len(m) == 0:
            continue
        total_m += float(-np.log(m).sum())
        total_b += float(-np.log(b).sum())
        count += len(m)
        points.append({"label": w["label"], "model": math.exp(-total_m / count) * 100,
                       "benchmark": math.exp(-total_b / count) * 100, "games": int(count)})
    if len(points) < 2:
        return None
    values = [p[k] for p in points for k in ("model", "benchmark")]
    low, high = math.floor(min(values) / 2.5) * 2.5 - 2.5, math.ceil(max(values) / 2.5) * 2.5 + 2.5
    if high - low < 10:
        mid = (high + low) / 2
        low, high = mid - 5, mid + 5
    n = len(points)
    y = lambda v: round((high - v) / (high - low) * 100, 2)
    for i, p in enumerate(points):
        p["x"] = round(i / (n - 1) * 100, 2)
        p["ym"], p["yb"] = y(p["model"]), y(p["benchmark"])
        p["tip"] = (f"Through {p['label']} ({p['games']} games): model {p['model']:.1f}%, "
                    f"{benchmark} {p['benchmark']:.1f}%")
    step = 2.5 if high - low <= 15 else 5
    ticks = [{"label": f"{v:g}%", "y": y(v)} for v in np.arange(math.ceil(low / step) * step, high + 0.01, step)]
    every = max(1, math.ceil(n / 8))
    return {"points": points, "benchmark": benchmark, "ticks": ticks,
            "labels": [p for i, p in enumerate(points) if i % every == 0 or i == n - 1],
            "model_path": "M " + " L ".join(f"{p['x']} {p['ym']}" for p in points),
            "bench_path": "M " + " L ".join(f"{p['x']} {p['yb']}" for p in points),
            "last": points[-1]}


def movement_chart(snaps, home: str, away: str) -> dict | None:
    """How the market's (and the model's) chance for one team moved before a game. snaps: rows with
    taken_at ('open' or 'YYYY-MM-DD HH:MM'), market_home, model_home, oldest first. The team shown is
    the one the latest model (or market) favours. None with fewer than two market snapshots."""
    snaps = [s for s in snaps if s.get("market_home") is not None and not math.isnan(s["market_home"])]
    if len(snaps) < 2:
        return None
    last = snaps[-1]
    ref = last["model_home"] if last.get("model_home") is not None and not math.isnan(last["model_home"]) else last["market_home"]
    home_side = ref >= 0.5
    team = home if home_side else away
    side = (lambda v: v * 100) if home_side else (lambda v: (1 - v) * 100)
    points = []
    for s in snaps:
        model = s.get("model_home")
        model = None if model is None or math.isnan(model) else side(model)
        if s["taken_at"] == "open":
            label = "Open"
        else:
            from datetime import datetime
            t = datetime.strptime(s["taken_at"], "%Y-%m-%d %H:%M")
            label = f"{t:%b} {t.day}, {t:%-I:%M %p}".replace(":00 ", " ")
        points.append({"label": label, "market": side(s["market_home"]), "model": model})
    values = [p["market"] for p in points] + [p["model"] for p in points if p["model"] is not None]
    low, high = math.floor(min(values) / 5) * 5 - 5, math.ceil(max(values) / 5) * 5 + 5
    low, high = max(low, 0), min(high, 100)
    n = len(points)
    y = lambda v: round((high - v) / (high - low) * 100, 2)
    for i, p in enumerate(points):
        p["x"] = round(i / (n - 1) * 100, 2)
        p["ym"] = y(p["market"])
        p["yb"] = y(p["model"]) if p["model"] is not None else None
        p["tip"] = (f"{p['label']}: market {p['market']:.0f}%" + (f", model {p['model']:.0f}%" if p["model"] is not None else ""))
    step = 5 if high - low <= 30 else 10
    ticks = [{"label": f"{v:g}%", "y": y(v)} for v in np.arange(math.ceil(low / step) * step, high + 0.01, step)]
    modelled = [p for p in points if p["yb"] is not None]
    every = max(1, math.ceil(n / 5))
    first, end = points[0]["market"], points[-1]["market"]
    possessive = f"{team}'" if team.endswith("s") else f"{team}'s"
    return {"team": team, "possessive": possessive, "points": points, "ticks": ticks,
            "labels": [p for i, p in enumerate(points) if i % every == 0 or i == n - 1],
            "market_path": "M " + " L ".join(f"{p['x']} {p['ym']}" for p in points),
            "model_path": ("M " + " L ".join(f"{p['x']} {p['yb']}" for p in modelled)) if len(modelled) > 1 else None,
            "model_last": modelled[-1] if modelled else None,
            "first": f"{first:.0f}%", "now": f"{end:.0f}%", "change": end - first,
            "toward_model": (modelled[-1]["model"] - first) * (end - first) > 0 if modelled and abs(end - first) >= 1 else None}
