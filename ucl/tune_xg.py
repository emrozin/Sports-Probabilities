"""Should the Europe-wide model learn from expected goals? python -m ucl.tune_xg [held_out [weight]]

Needs Understat's xG: python -m ucl.import_understat

Walk-forward backtests of the big four leagues' games (La Liga, Bundesliga, Serie A, Ligue 1) and the
Champions League's, with the ratings learned from a blend of goals and xG (xg_weight 0 is goals
only, today's model), on every third match day to keep it quick. The weight is chosen on 2017-18 to
2020-21 (xG starts in 2014-15, so every training window has it), then checked once against goals
only on 2021-22 to 2025-26. Scored by log loss of the chances as the site gives them (league games blend in
Elo, ucl/elo.py).

Result (log loss; tuning on every third match day):

    xG weight                         0.00    0.25    0.50    0.75
    leagues 2017-18 to 2020-21        0.9734  0.9730  0.9727  0.9725
    Champions League, same seasons    0.8904  0.8908  0.8913  0.8920

    held out, 2021-22 to 2025-26      0.00    0.75
    leagues                           0.9910  0.9903
    Champions League                  0.8904  0.8935

So the leagues' model learns 75% from xG and 25% from goals (a small gain, the same on the held-out
seasons), and the cups' model keeps learning from goals only: xG made it slightly worse both times,
perhaps because xG covers only five leagues, so it tilts the ratings toward their clubs. With xG, Elo's
60% share of the league chances (ucl/elo.py) was still the best on the tuning seasons.
"""

import sqlite3
import sys

from db import DB_PATH
from ucl import model as m

LEAGUES = ["esp.1", "ger.1", "ita.1", "fra.1"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75]
TUNING, HELD_OUT = (2017, 2020), (2021, 2025)


def score(games, values, span, competitions, weight) -> tuple[int, float]:
    test = m.scored_games(games, competitions)
    test = test[test["season"].between(*span)]
    dates = set(sorted(test["date"].unique())[::3])
    results = m.backtest(games, span[0], span[1], dates=dates, competitions=competitions, values=values, xg_weight=weight)
    return len(results), m.log_loss(results, games)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games, values = m.load_games(conn), m.load_values(conn)
    conn.close()
    if games["home_xg"].notna().sum() == 0:
        raise SystemExit("No xG found. Run python -m ucl.import_understat first.")
    if "held_out" in argv:
        i = argv.index("held_out")
        chosen = float(argv[i + 1]) if len(argv) > i + 1 else 0.5
        span, weights = HELD_OUT, [0.0, chosen]
    else:
        span, weights = TUNING, WEIGHTS
    print(f"Seasons {span[0]}-{span[1]}:")
    for w in weights:
        n, leagues = score(games, values, span, LEAGUES, w)
        k, cup = score(games, values, span, [m.UCL], w)
        print(f"  xG weight {w:.2f}: big four leagues {leagues:.4f} ({n} games), Champions League {cup:.4f} ({k} games)", flush=True)


if __name__ == "__main__":
    main()
