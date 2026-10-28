"""Run the whole update for every sport, in the right order.

Usage:
    python update.py            # update data and predictions
    python update.py --build    # ...and rebuild the static site in docs/
    python update.py --record   # ...and update the official prediction records (the automated run uses this)
    python update.py --late     # only the steps that use late news: NBA injury report and predictions,
                                # ESPN's lines and the bet log (the 6:37 pm run, before most tip-offs)

Premier League (epl/):
    1. import_epl          new results, stats and odds from football-data.co.uk
    2. import_understat    expected goals (xG) for the new results
    3. run_backtest        backtest predictions, now including the latest matches
    4. import_fixtures     the next round's fixtures and pre-match odds
    5. predict_upcoming    predictions for those fixtures, saved before kickoff
    6. fetch_availability  a snapshot of player injuries and availability (collected for later use)

NFL (nfl/):
    1. import_nfl          results, schedule and betting lines from nflverse
    2. import_nfl_stats    team and player stats from nflverse
    3. run_backtest        backtest predictions, now including the latest games
    4. predict_upcoming    predictions for the next 8 days' games, saved before kickoff
    5. predict_players     player projections for those games, saved before kickoff
    6. season              playoff odds: the rest of the season simulated 10,000 times

NBA (nba/):
    1. import_nba          games, results and player box scores from sportsdataverse-data
    2. live                current rosters and the injury report from ESPN
    3. run_backtest        backtest predictions for the current season
    4. predict_upcoming    predictions for the next 8 days' games, saved before tip-off
    5. predict_players     player projections (points, rebounds, assists, threes) for those games
    6. season              playoff odds: the rest of the season simulated 10,000 times

College football (cfb/):
    1. import_cfb          games, results and betting lines from cfbfastR-data
    2. run_backtest        backtest predictions, now including the latest games
    3. predict_upcoming    predictions for the next 8 days' games, saved before kickoff
    4. conference          conference title odds: the rest of the season simulated 10,000 times

College basketball (cbb/):
    1. import_cbb          games and results from sportsdataverse-data
    2. run_backtest        backtest predictions, now including the latest games
    3. predict_upcoming    predictions for the next 8 days' games, saved before tip-off

Betting lines (espn.py): ESPN's lines for recent and upcoming NBA, college basketball and college
football games, saved as the season goes (run last, after every sport's games are imported); then
betting.py logs the bets its rule suggests, for the betting page's live record. Last, warm_cache.py
prepares the slow parts of the website (the NBA model, the betting page's past-season tests).

Each step is a module run with "python -m", e.g. python -m epl.import_epl. Within a sport, a
failed step stops that sport's remaining steps (they depend on it), but the other sport still
runs, so a problem with one data source never holds up the other. The exit code is 1 if
anything failed, so the automated run shows it.

The automated run (scheduled/install.sh) runs the whole update at 10:07 am and 5:37 pm, and the
late steps at 6:37 pm: the NBA's injury report changes through the afternoon, and lines are slow to
catch up with late scratches, so the predictions and the bet log should see the news as soon as
possible.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SPORTS = {
    "Premier League": [
        ("epl.import_epl", "Results, stats and odds"),
        ("epl.import_epl", "Championship results (to rate newly promoted teams)", ["--league", "CHAMP"]),
        ("epl.import_understat", "Expected goals (xG)"),
        ("epl.run_backtest", "Backtest predictions"),
        ("epl.import_fixtures", "Upcoming fixtures and odds"),
        ("epl.predict_upcoming", "Predictions for upcoming matches"),
        ("epl.fetch_availability", "Player availability snapshot"),
    ],
    "NFL": [
        ("nfl.import_nfl", "Games, results and lines"),
        ("nfl.import_nfl_stats", "Team and player stats"),
        ("nfl.run_backtest", "Backtest predictions"),
        ("nfl.predict_upcoming", "Predictions for upcoming games"),
        ("nfl.predict_players", "Player projections for upcoming games"),
        ("nfl.season", "Playoff odds (season simulation)"),
    ],
    "NBA": [
        ("nba.import_nba", "Games, results and box scores"),
        ("nba.live", "Rosters and injury report (ESPN)"),
        ("nba.run_backtest", "Backtest predictions"),
        ("nba.predict_upcoming", "Predictions for upcoming games"),
        ("nba.predict_players", "Player projections for upcoming games"),
        ("nba.prop_lines", "Player prop lines to test against (DraftKings via ESPN)"),
        ("nba.season", "Playoff odds (season simulation)"),
    ],
    "College football": [
        ("cfb.import_cfb", "Games, results and lines"),
        ("cfb.run_backtest", "Backtest predictions"),
        ("cfb.predict_upcoming", "Predictions for upcoming games"),
        ("cfb.conference", "Conference title odds (season simulation)"),
    ],
    "College basketball": [
        ("cbb.import_cbb", "Games and results"),
        ("cbb.run_backtest", "Backtest predictions"),
        ("cbb.predict_upcoming", "Predictions for upcoming games"),
    ],
    "Champions League": [
        ("ucl.import_espn", "Games, results, fixtures and odds (ESPN)"),
        ("ucl.import_understat", "Expected goals (xG) for the big five leagues (Understat)"),
        ("ucl.squad_values", "Squad values (Transfermarkt)"),
        ("ucl.run_backtest", "Backtest predictions (new match days)"),
        ("ucl.predict_upcoming", "Predictions for upcoming games"),
        ("ucl.season", "Knockout odds (season simulation)"),
    ],
    "Betting lines": [
        ("espn", "Lines for NBA and college games (ESPN)"),
        ("betting", "Suggested bets and line movement, logged (betting.py)"),
    ],
    "Website": [
        ("warm_cache", "Prepare the slow parts of the pages (warm_cache.py)"),
    ],
}
LATE_STEPS = {"Late news": [
    ("nba.live", "NBA rosters and injury report (ESPN)"),
    ("nba.predict_upcoming", "NBA predictions for upcoming games"),
    ("nba.predict_players", "NBA player projections"),
    ("nba.prop_lines", "NBA player prop lines (DraftKings via ESPN)"),
    ("espn", "Lines for NBA and college games (ESPN)"),
    ("betting", "Suggested bets and line movement, logged (betting.py)"),
    ("warm_cache", "Prepare the slow parts of the pages (warm_cache.py)"),
]}
RECORDING_STEPS = {"epl.predict_upcoming", "nfl.predict_upcoming", "nfl.predict_players", "nfl.season",
                   "nba.live", "nba.predict_upcoming", "nba.predict_players", "nba.prop_lines", "nba.season",
                   "cfb.predict_upcoming", "cfb.conference", "cbb.predict_upcoming", "ucl.predict_upcoming", "ucl.season",
                   "espn", "betting"}
BUILD_STEP = ("build_site", "Static website (docs/)")


def run(module: str, label: str, args: list[str]) -> tuple[bool, float]:
    """Run one step with this project's Python, showing its output. Returns (succeeded, seconds)."""
    print(f"\n  {label}  (python -m {module})")
    print("  " + "-" * 58, flush=True)
    started = time.time()
    # sys.executable is the Python running this script, i.e. the one in .venv
    result = subprocess.run([sys.executable, "-m", module, *args])
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"\n  Failed: {module} (exit code {result.returncode}) after {elapsed:.0f}s.")
    return result.returncode == 0, elapsed


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)  # works no matter which folder it's run from
    args = sys.argv[1:]
    record = ["--record"] if "--record" in args else []
    started = time.time()
    timings, failed = [], []

    for sport, steps in (LATE_STEPS if "--late" in args else SPORTS).items():
        print(f"\n{'=' * 60}\n{sport}")
        for module, label, *fixed in steps:          # a step can carry its own arguments
            ok, seconds = run(module, label, (fixed[0] if fixed else []) + (record if module in RECORDING_STEPS else []))
            timings.append((f"{sport}: {label}", seconds))
            if not ok:
                failed.append(f"{sport}: {label} ({module})")
                print(f"  Skipping the rest of the {sport} update; it depends on this step.")
                break

    if "--build" in args and not failed:
        ok, seconds = run(*BUILD_STEP, [])
        timings.append((BUILD_STEP[1], seconds))
        if not ok:
            failed.append(BUILD_STEP[1])

    print("\n" + "=" * 60)
    for label, seconds in timings:
        print(f"  {label:<52} {seconds:>5.0f}s")
    print(f"  {'Total':<52} {time.time() - started:>5.0f}s")
    if failed:
        print("\nThese steps failed (see the errors above):")
        for step in failed:
            print(f"  {step}")
        print("Every step is safe to re-run, so running update.py again is fine.")
        sys.exit(1)
    print("\nUpdate complete.")
    if "--build" not in args:
        print("Preview the site with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
