"""Does ESPN's feed have sportsbook prices for NBA player props? python -m nba.prop_lines_check

A one-off check, run on your Mac (ESPN can't be reached from where the site was built). It looks
up today's NBA games in ESPN's public "core" API, lists each game's odds providers, and follows any
player-prop link it finds. It prints what it found and saves the raw answers to
logs/espn_props_sample.json, so the prop prices can be read properly if they're there. It changes
nothing in the database.
"""

import json
from datetime import date
from pathlib import Path

import requests

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT = Path("logs/espn_props_sample.json")


def get(url: str, **params) -> dict | None:
    try:
        r = requests.get(url, params={"lang": "en", "region": "us", **params}, headers=HEADERS, timeout=30)
    except requests.RequestException as error:
        print(f"  {url}: couldn't connect ({error})")
        return None
    if r.status_code != 200:
        print(f"  {url}: HTTP {r.status_code}")
        return None
    return r.json()


def main(day: date | None = None) -> None:
    day = day or date.today()
    saved = {}
    events = get(f"{CORE}/events", dates=f"{day:%Y%m%d}", limit=50)
    refs = [item["$ref"] for item in (events or {}).get("items", []) if "$ref" in item]
    print(f"{len(refs)} NBA games on {day}.")
    for ref in refs[:2]:                    # two games are enough to see what's there
        event_id = ref.rstrip("/").split("/")[-1].split("?")[0]
        odds_url = f"{CORE}/events/{event_id}/competitions/{event_id}/odds"
        odds = get(odds_url)
        saved[odds_url] = odds
        providers = (odds or {}).get("items", [])
        print(f"Game {event_id}: {len(providers)} odds providers")
        for item in providers:
            provider = item.get("provider", {})
            pid = provider.get("id")
            print(f"  provider {pid}: {provider.get('name')}")
            props_url = f"{CORE}/events/{event_id}/competitions/{event_id}/odds/{pid}/propBets"
            props = get(props_url, limit=1000)
            if props:
                saved[props_url] = props
                items = props.get("items", [])
                print(f"    player props: {props.get('count', len(items))} lines")
                for line in items[:5]:
                    print(f"      {json.dumps(line)[:200]}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(saved, indent=1))
    print(f"Saved what ESPN answered to {OUT}.")


if __name__ == "__main__":
    main()
