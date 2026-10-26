"""Run the whole update for every sport, in the right order.

Usage:
    python update.py            # update data and predictions
    python update.py --build    # ...and rebuild the static site in docs/
    python update.py --record   # ...and update the official prediction records (the automated run uses this)

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
betting.py logs the bets its rule suggests, for the betting page's live record.

Each step is a module run with "python -m", e.g. python -m epl.import_epl. Within a sport, a
failed step stops that sport's remaining steps (they depend on it), but the other sport still
runs, so a problem with one data source never holds up the other. The exit code is 1 if
anything failed, so the automated run shows it.

The automated run happens on Tuesdays and Fridays: Tuesday covers Thursday night NFL games
and midweek football; Friday covers the weekend.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

SPORTS = {
    "Premier League": [
        ("epl.import_epl", "Results, stats and odds"),
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
    "Betting lines": [
        ("espn", "Lines for NBA and college games (ESPN)"),
        ("betting", "Suggested bets, logged (betting.py)"),
    ],
}
RECORDING_STEPS = {"epl.predict_upcoming", "nfl.predict_upcoming", "nfl.predict_players", "nfl.season",
                   "nba.live", "nba.predict_upcoming", "nba.predict_players", "nba.season",
                   "cfb.predict_upcoming", "cfb.conference", "cbb.predict_upcoming", "espn", "betting"}
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

    for sport, steps in SPORTS.items():
        print(f"\n{'=' * 60}\n{sport}")
        for module, label in steps:
            ok, seconds = run(module, label, record if module in RECORDING_STEPS else [])
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
