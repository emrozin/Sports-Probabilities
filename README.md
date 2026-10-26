# Sports Probabilities: Premier League, NFL, NBA, college football and college basketball

Statistical models that estimate win probabilities for English Premier League matches, NFL and NBA games and college football and basketball games, evaluated honestly against the sharpest benchmark available: betting-market odds. Predictions for upcoming games are published to this repository before kickoff, so each model's live track record can be checked by anyone. All five sports share one database, one automated update and one website, with a sport menu.

The Premier League model came first and is described in most detail below; the NFL, NBA and college sections follow.

The project covers the full pipeline: collecting and validating 30+ seasons of data from four sources, building and tuning a Poisson goals model with expected goals (xG), testing it with walk-forward backtesting on seasons it never saw during development, and serving it through a web app with an automated, tested weekly update.

**Tech:** Python, SQL (SQLite), pandas, NumPy, SciPy, FastAPI, Jinja, pytest, GitHub Actions

## Premier League results

All numbers below are from **held-out seasons (2021/22 to 2025/26, 1,921 matches)** that played no part in choosing the model's settings. Lower log loss is better; guessing one-third for every outcome scores 1.0986.

| Model | Log loss |
|---|---|
| Goals only, initial settings | 0.9824 |
| Goals only, tuned | 0.9813 |
| + expected goals (xG) blend | 0.9778 |
| + xG with re-tuned recency weighting (**current model**) | **0.9754** |
| Closing market (Pinnacle / Betfair Exchange, margin removed) | 0.9592 |

Adding xG and re-tuning closed **27% of the gap** between the goals-only model and the closing market. Put another way, the current model gives the actual result a typical probability of **37.7%**, against the market's **38.3%**.

The model does not beat the market, and isn't expected to: closing odds reflect injuries, confirmed lineups and professional bettors' information, while this model only sees past results. The goal is to measure how close a transparent, results-based model can get, and which changes actually help.

### Live track record

Since October 2026, every prediction for an upcoming match is committed to [`records/live_predictions.csv`](records/live_predictions.csv) before kickoff by an automated job, and never changed afterwards; the Git history shows when each one was made. The website's Track record page scores these predictions, and only these, once the matches are played. A few rounds is far too small a sample to judge the model by, so the backtest above is the main evidence until the record grows.

## Data

| Source | Coverage | Contents |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | 1993/94 onward | Results, half-time scores, match stats (from 2000/01), 1X2 / over-under / Asian handicap odds from 20+ bookmakers, pre-match and closing; upcoming fixtures |
| [Understat](https://understat.com/) | 2014/15 onward | Expected goals (xG) per team per match; results before football-data.co.uk posts them |
| [Fantasy Premier League](https://fantasy.premierleague.com/) | 2016/17 onward | Player prices (via the [FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League)), current injuries and availability |
| [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) | 2013/14 to June 2026 | Player market valuations, used in the squad-value experiments |

The database holds 13,000+ Premier League matches. Some of the data-quality work involved:

- **Inconsistent file formats across 30+ seasons.** Column names change over time (e.g. `BbAv` became `Avg`), and some seasons have rows with more fields than the header. The importer handles both, and every original row is also stored as JSON so no source column is ever lost.
- **Unmapped column report.** Each import lists any source columns not mapped to a table, which is how new bookmakers and a new xG column were discovered.
- **Cross-source validation.** Each source spells team names differently (e.g. "Manchester United" vs "Man United"), resolved through an alias table. All 4,610 matches linked between Understat and football-data.co.uk were checked against both sources' final scores, with 100% agreement.
- **Source changes and delays.** Midway through development, Understat stopped embedding data in its pages; the importer failed loudly rather than importing nothing, and the fix was isolated to one function. football-data.co.uk can take days to post results, so results are filled in from Understat in the meantime and confirmed when football-data.co.uk catches up, without creating duplicate matches.
- **No look-ahead.** Squad values for any week are built only from information available by that date, so a backtest never benefits from knowing, for example, a player's later transfer.

## Model

Each team has an **attack** and a **defense** rating, plus a league-wide **home advantage**. A team's expected goals in a match are:

```
expected goals = base rate × home advantage (if at home) × attack ÷ opponent's defense
```

Each team's goals are modeled as a Poisson distribution around its expected goals, which gives the probability of every exact scoreline. From those come home/draw/away probabilities, over/under totals, both teams to score, and the likeliest scores.

Ratings are fitted by **maximum likelihood** (SciPy's L-BFGS-B with an analytic gradient, about 30× faster than numerical differentiation) using the last three years of matches, with:

- **Recency weighting:** a match's weight halves every 180 days.
- **Shrinkage (L2 regularization):** ratings are pulled toward average, so teams with few matches, such as newly promoted sides, don't get extreme ratings from a handful of results.
- **xG blend:** ratings are learned from a 50/50 blend of actual goals and xG.

A **season simulator** plays out the rest of the season 10,000 times with the model to estimate each team's chances of winning the title, finishing in the top four, or being relegated.

## Evaluation

- **Walk-forward backtest:** for every week since 2016/17, the model is trained only on matches played before that week, then predicts that week's games. It never sees a result it is predicting.
- **Validation / test split:** all settings were tuned on 2016/17–2020/21, and final results are reported on 2021/22–2025/26.
- **Metrics:** log loss and Brier score, compared against closing odds from Pinnacle (Betfair Exchange where Pinnacle is unavailable), with the bookmaker margin removed.

## Findings

- **The gap to the market is spread evenly.** An error analysis split the backtest by team, stage of the season, favorite strength and outcome. The model is well calibrated overall (slightly too cautious on strong favorites), and no single group accounts for the shortfall. That pattern points to missing information, mainly team news, rather than a flaw in the math.
- **Squad value helps early in the season, and only then.** See below.
- **Promoted teams aren't a weak spot.** A dedicated check found the model's error on newly promoted teams' matches only slightly worse than usual, so importing second-division data to rate them wasn't worth the added complexity.
- **Home advantage is shrinking.** Home teams won around 45–50% of matches in the 1990s and 2000s, and closer to 41–48% recently.
- **Crowds matter.** In 2020/21, played without fans, away teams won more often than home teams (40.3% vs 37.9%), the only season where that happened.
- **Scoring jumped in 2023/24** to 3.28 goals per game, the highest on record, coinciding with a crackdown on time-wasting that added stoppage time. The goals-only model was slow to adapt that season.
- **Goals and xG work best together.** On validation seasons, goals only scored 0.9571, a 50/50 blend 0.9532, and pure xG 0.9552. Actual goals carry information xG misses, such as finishing and goalkeeping quality.
- **xG allows a shorter memory.** With xG, the best recency half-life dropped from 365 to 180 days, and less shrinkage was needed.
- **Dixon–Coles didn't help.** This well-known correction for low-scoring draws made predictions slightly worse: the model already predicted slightly more draws than occurred (23.7% vs 22.5%). The apparent excess of low scores in the raw data came from pooling strong and weak teams, which team-specific ratings already account for.

### Squad value

Pulling ratings toward each squad's value (FPL prices or Transfermarkt valuations) improved early-season predictions but made later ones slightly worse, since by then results already show how good a team is. Making its influence fade as the season goes on (halving every 12 rounds) kept nearly all of the early gain. Log loss on the 1,873 held-out matches with squad values available:

| | Current model | Squad value, full strength | Squad value, fading | Market |
|---|---|---|---|---|
| All matches | 0.9735 | 0.9737 | **0.9718** | 0.9570 |
| Rounds 1–10 | 0.9558 | **0.9492** | 0.9500 | 0.9473 |
| Rounds 11–38 | 0.9800 | 0.9826 | **0.9797** | 0.9605 |

The fading version closes 11% of the overall gap to the market, and about two-thirds of the gap in rounds 1–10. One caveat: on the tuning seasons the full-strength version scored slightly better, so the held-out seasons decided between two finalists. The fading version is preferred because, by design, it can't cause the late-season loss. It isn't used for live predictions yet.

## NFL

Data: every game since 1999 from [nflverse](https://github.com/nflverse/nfldata) (scores, spreads, totals, moneylines from 2006, starting quarterbacks, rest days, weather), plus weekly team and player stats with expected points added (EPA) from [nflverse-data](https://github.com/nflverse/nflverse-data): about 126,000 player-games, in the same database.

**Model.** Each team has an offense rating (points scored above average) and a defense rating (points prevented), plus a home advantage fitted from recent games, so it follows the league's shrinking home edge (about 2.7 points in the 2000s, under 2 since 2020). Ratings are fitted by weighted least squares on four years of games, with a 150-day half-life and shrinkage. The expected margin becomes a win probability through a normal distribution (actual margins land about 12 points either side). **Every starting quarterback has his own rating**, fitted alongside the team ratings and pulled toward average, plus an inexperience term for quarterbacks with few career starts; live predictions use nflverse's expected starters for the current week, checked against the box scores (`nfl/quarterbacks.py`). nflverse's listed starter can go stale: in 2026 it listed Drew Lock for Seattle for four weeks after Sam Darnold returned. So a listed quarterback who didn't throw a pass loses the start to the one who threw the most (56 games corrected since 1999), and for upcoming games a listing that names neither the team's last starter nor its usual one is overruled when the last starter finished his game and isn't ruled out on the injury report. `nfl/qb_overrides.csv` names the starter by hand when the data can't know (a benching); overrides apply only to games not yet played, so a wrong guess never reaches the ratings. **Injured reserve:** players on injured reserve aren't on the weekly injury report, so nflverse's weekly rosters are imported too. When the expected starter is on a reserve list (or released), the team's healthy quarterback who most recently started for it takes over (in October 2026 the schedule kept listing Jalen Hurts after he went on injured reserve; Tanner McKee started), and reserve-list players aren't given player projections.

**Results** on held-out seasons (2018-2025, 2,219 games, ties left out; tuned on 2010-2017), log loss:

| Model | Log loss | Share of the market's edge over a coin flip |
|---|---|---|
| Coin flip | 0.6931 | 0% |
| Elo ratings (baseline) | 0.6382 | 65% |
| Points model (offense/defense ratings) | 0.6374 | 66% |
| Points model + usual-starter quarterback flag | 0.6326 | 72% |
| Points model + individual quarterback ratings | 0.6298 | 75% |
| **+ starters checked against the box scores** | **0.6295** | **75%** |
| Market moneylines | 0.6089 | 100% |

Offense and defense ratings alone did no better than Elo on the held-out seasons, despite looking better on the tuning seasons. Quarterbacks made the real difference. A simple flag for teams not starting their usual quarterback came first (the model had been overrating those teams by about two points); individual quarterback ratings then replaced it, helping most in the first four weeks of a season, when teams have new starters (held-out log loss 0.6493 to 0.6458 in weeks 1-4).

**Playoff odds** (`nfl/season.py`). The rest of the season is played out 10,000 times: every remaining game gets a random result from the model (expected margin plus the usual game-to-game spread), each simulated season shifts every team's rating by a random amount (3.5 points; ratings are estimates, and without this the odds are overconfident), and division winners and seeds follow the main NFL tiebreakers (head-to-head, division or conference record, point differential). Then the playoffs are simulated with the 7-team format (6 before 2020). Tested after weeks 4, 8 and 12 of each season: on the held-out 2018-2025 seasons (768 team-seasons) the playoff chances scored a log loss of 0.403 (0.416 without the rating shift, 0.681 for a flat league-wide rate) and are calibrated: teams given 20-40% made the playoffs 35% of the time, 40-60% 52%, 60-80% 67%. Odds are saved to `records/nfl_season_odds.csv` on every update, so each team page charts its playoff chance through the season.

**Blending with the market** (`blend.py`, `nfl/blend.py`). The model and the closing moneyline were combined on the log-odds scale, with the weight chosen on 2010-2017 (20% model). On the held-out seasons the blend scored 0.6104, worse than the market alone (0.6089), and lost in 6 of 8 seasons: the closing line already contains what the model knows. So the site doesn't show an NFL blend, and the backtest page reports the result. (Closing odds are the sharpest; a live blend would use earlier odds, which can't be tested with nflverse's single line per game.) The same test for the Premier League is `python -m epl.blend`; the site shows a blend only if it beats the market on the held-out seasons.

**What didn't help** (`nfl/experiments.py`): rest days (byes, short weeks), travel across time zones, domes, weather, and counting last season's games less all scored the same or worse on the tuning seasons, so none are in the model. Weather also can't be used live, since nflverse records it only after a game. Learning from **EPA** as well as the score, the NFL's equivalent of the xG blend that helped the Premier League model most, made win predictions worse at every weight tried (it slightly improved total points): once quarterbacks are rated, the score already carries most of the signal.

**Player projections.** Passing, rushing and receiving yards, touchdowns and receptions for every player in upcoming games, from (1) the team's recent passing and running volume, (2) each player's recent share of targets and carries, nudged by his latest snap count (the listed starting quarterback throws the passes), (3) the weekly injury report (Out and Doubtful players aren't projected, Questionable players get a slightly smaller role, players back from injury return), (4) his efficiency pulled toward his position's average, adjusted for the opposing defense, and (5) touchdowns scaled by the team model's expected points. Game pages also show the chance of reaching a round number near each projection (for example "38% for 60+" receiving yards). Tuned on 2012-2017; average miss per player-game on the held-out 2018-2025 seasons (about 35,000 player-games):

| | Projection | Last 3 games | Season average |
|---|---|---|---|
| Passing yards | **61.4** | 71.6 | 70.0 |
| Passing touchdowns | **0.92** | 1.00 | 0.98 |
| Rushing yards | **8.2** | 8.9 | 8.7 |
| Receptions | **1.28** | 1.38 | 1.34 |
| Receiving yards | **17.0** | 18.6 | 18.0 |

Each addition was tested with a measure that counts every player who was projected or played (`nfl/player_experiments.py`), so a version can't look better by projecting fewer players. Injury reports cut misses by 4-5% for rushing yards, receptions and receiving yards on the held-out seasons, and snap counts by about 1% more. Not kept: redistributing an injured player's share to teammates, defense by position, air yards (depth of target), and adjusting passing volume for the expected score. The first version of the injury rules looked better than it was because it was scored on a different set of players, and the fair comparison caught it: players who drop off the injury report are often on injured reserve, not returning. The chances of reaching those round numbers are well calibrated on the held-out seasons (predicted 44% happened 44% of the time, predicted 56% happened 55%). There are no free historical betting lines for player stats, so these are compared with simple benchmarks rather than the market. Projections are saved before kickoff to `records/nfl_player_projections.csv`, and the website's track record compares them with what happened and with the benchmarks.

**Likeliest player props.** For the common sportsbook props (passing yards from 225, rushing and receiving yards from 50, receptions from 4, and scoring a touchdown), the site lists the likeliest to hit in each game and the ten likeliest each day, one per player, and the track record counts how many hit. On the held-out seasons the ten likeliest each day were given 64% on average and 66% hit (4,838 props); the five likeliest per game were given 65% and 65% hit. Likely to hit isn't the same as a good bet, since these aren't compared with the sportsbook's price.

## NBA

Every game and every player's box score line since 2002-03 from [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) (ESPN's data): about 31,000 games and 818,000 player lines, including who didn't play and why (a coach's decision, or injured). Teams are identified by ESPN's franchise id, so a move or rename (Seattle to Oklahoma City) keeps one history; All-Star and preseason games are left out, postponed games appear once (on the date they were played), and the NBA Cup final doesn't count in the standings.

**Model** (`nba/model.py`): **players, and who's available.** In basketball one player can swing a game by several points, and stars sit out often (injuries, rest), so the model rates players, not just teams. A game's margin is home court plus, for each team, the sum over its players of (share of the game on the floor x rating), plus a smaller team term (coaching, system). A player's rating is how many points per game his team gains with him on the floor for a whole game, against an average player, after allowing for everyone else on both teams. All the ratings are fitted together by ridge regression on three seasons of games (recent ones counting more), so a player is separated from his teammates as lineups change through injuries, rest and trades. For a game still to play, the lineup is the team's current roster from ESPN minus the players ESPN's injury report rules out (`nba/live.py`), and the game's 240 minutes go to the available players in order of their usual minutes (games the coach didn't play them count as zero, so the end of the bench gets what's left). The expected score splits a team offense/defense model's total by the predicted margin. Tuned on 2010-11 to 2013-14 (`nba/tune_model.py`).

**Results** on held-out seasons (2014-15 to 2025-26, 15,476 games), log loss; each backtest week refits the model on earlier games only and builds each lineup the way a live prediction does (every player available for the game, from its box score: what the injury report tells a forecast before tip-off):

| Version | Log loss |
|---|---|
| A simple Elo rating (FiveThirtyEight's settings) | 0.6265 |
| Team ratings only, no players | 0.6211 |
| Players, ignoring who's available | 0.6179 |
| **Players and who's available (the site)** | **0.6135** |
| Players with the minutes they really played (an upper bound) | 0.6086 |

(Building the backtest lineup from just the players who got into each game scored 0.6115, but that knows a little more than a forecast can: late scratches and who the coach left out. The site's number is the honest one.)

The model beats Elo in every held-out season and picks the winner 65.7% of the time (Elo 65.2%). Knowing who's available is worth about as much as the player ratings themselves: ignoring it gives back most of the gain over team ratings. Sharing minutes in order of each player's usual minutes beat scaling everyone down evenly. There are no free historical betting lines for the NBA; ESPN's lines are saved from this season on (below), so the track record will compare the model with the market.

**Player props** (`nba/players.py`). Points, rebounds, assists and threes for every player in upcoming games, projected the way the game model builds lineups: his recent rate per minute (pulled toward his position's average when he has few minutes) times his expected minutes, as a share of what the model expects his team to get, so when a teammate sits his share goes up. The chance of reaching each line (15+ points, 10+ rebounds...) comes from a negative binomial spread whose variance grows with the projection, fitted on 2014-15 to 2016-17 (a fixed-width spread made stars' chances too wide). On held-out 2017-18 onward (239,683 player-games), the average miss was smaller than each player's season average by 2.3% for points, 0.7% rebounds, 3.1% assists and 2.4% threes, and the chances are calibrated (props given 40-60% hit 49% of the time, 60-75% 67%, 75-90% 81%). The five likeliest props each day were given 93.7% on average and hit 91.8% of the time. Likely to hit isn't the same as a good bet: these aren't compared with the sportsbook's price.

**Playoff odds** (`nba/season.py`). The rest of the season is played out 10,000 times with the players-and-availability model, leaving injured players out until ESPN's expected return date, with a random strength shift for each team in each simulated season (3 points per game; chosen on 2020-21 and 2021-22). Standings use head-to-head and conference record for ties, then the play-in (7th v 8th, the loser v the winner of 9th v 10th) and four best-of-seven rounds with home court. On the held-out 2022-23 to 2025-26 seasons (odds made on 1 December, 15 January and 1 March): making the playoffs scored a log loss of 0.318 (0.343 without the shift; 0.691 for a flat league-wide rate), and teams given 10-30% made it 18% of the time, 30-50% 38%, 70-90% 76%.

## College football

Games involving a top-division (FBS) school since 2001 from [cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data), which republishes CollegeFootballData's schedules, scores, conferences and pre-game Elo ratings, plus several sportsbooks' betting lines (2006-2025). Second-division (FCS) games are included so the FCS schools that FBS teams play are rated properly. Betting lines are turned into one consensus per game: each book's moneyline with its margin removed, averaged; team names in old lines are abbreviations, matched to teams by the games they appear in.

**Model** (`cfb/model.py`). The NFL's points model: offense and defense ratings for every team, home advantage (about 3 points), fitted by weighted least squares on four seasons, with recent games counting more and each earlier season counting less again, since college rosters turn over quickly. Tuned on 2010-2017 (`cfb/tune_model.py`): log loss went from 0.536 with first guesses to 0.479. Almost all of that came from much weaker shrinkage than the NFL's: college teams really are far apart (the best are 40+ points better than the worst), and with old games down-weighted, strong shrinkage pulled every rating toward average.

**Quarterbacks.** Each team's starter is found from the play-by-play (the quarterback with the most dropbacks, from 2014; `import_cfb.py`). The model rates quarterbacks the way the NFL model does, plus a penalty for a new starter that fades over his first 12 starts. Predictions use the expected starter (each team's previous starter), since that's all a forecast made before kickoff knows. The gain is small (held-out log loss 0.4887 to 0.4877, better in 6 of 8 seasons): college teams change quarterbacks less often mid-season than NFL teams, and a starter's quality is mostly already in his team's rating.

**Results** on held-out seasons (2018-2025), log loss (ties left out):

| | Games with an Elo rating (5,849) | Games with odds (5,430) |
|---|---|---|
| This model | **0.5340** | 0.5227 |
| CollegeFootballData's Elo | 0.5538 | |
| Betting market (closing consensus) | | **0.5043** |

The model beats CollegeFootballData's Elo but not the market, which knows about injuries, transfers and suspensions the free data doesn't capture; the gap (0.018) is about the same as the NFL model's. The free lines don't cover the current season yet, so upcoming games are compared with Elo instead.

**Conference title odds** (`cfb/conference.py`). The rest of the season is played out 10,000 times with the model, like the NFL playoff odds: each game gets a random result, and each team a random "how good are they really" shift for the whole simulated season (8 points, chosen on 2022-2023; ratings from a dozen games across 130+ teams are much less certain than NFL ones). The top two teams in conference play meet in the title game (the Sun Belt's division winners), with ties settled by head-to-head record, then overall record, then rating. Tested on 2024-2025, from before week 1, 5, 9 and 12 (72 conference races): the eventual champion got about a 13% chance on average, against 7% picking a team at random (log loss 2.01 vs 2.66; 2.22 without the rating shift), and the chances are calibrated (teams given 15-30% won 20% of the time, 30-50% 34%). The site also shows standings for every season, filtered by conference.

## College basketball

Men's Division I games since 2002-03 from [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data), which republishes ESPN's schedules, scores and conferences (and team colors). Games against schools outside Division I are kept, with those schools sharing one rating, since they only play a game or two a year. Seasons are named by their starting year (2025 is 2025-26).

**Model** (`cbb/model.py`). The same points model as college football, without quarterbacks: offense and defense ratings for every team and home court (about 3 points), fitted on three seasons with a 120-day half-life, refitted every week. Tuned on 2010-11 to 2017-18 (`cbb/tune_model.py`). There are no free historical betting lines for college basketball, so the benchmark is a simple margin-aware Elo rating.

**Results** on held-out seasons (2018-19 to 2025-26, 43,330 games):

| | Log loss | Typical probability | Picks the winner |
|---|---|---|---|
| This model | **0.5382** | 58.4% | 72.0% |
| Simple Elo | 0.5491 | 57.7% | 71.2% |

The model is better than Elo in all eight seasons. Tuning helped (0.5426 with first guesses), mostly through weaker shrinkage, as in football.

## Betting: does following the model make money?

`betting.py` turns the model into a betting rule and tests it. A bet is suggested when the model's chance times the bookmaker's decimal price gives an expected return of at least 3%, at most one bet per game. The stake is a quarter of the Kelly criterion (the stake that maximises a bankroll's long-run growth if the chances were exact), capped at 3% of the bankroll, since the chances are estimates. The same rule was run on each sport's held-out backtest seasons, at the prices of the time:

| | Seasons | Bets | Return per bet | 95% range (luck) | $1,000 at quarter Kelly | Closing-line value |
|---|---|---|---|---|---|---|
| Premier League (best bookmaker price) | 2021-22 to 2025-26 | 1,747 | -3.4% | -12.5% to +5.5% | $102 | +0.8% |
| Premier League (average price: the first version) | 2021-22 to 2025-26 | 1,438 | -9.7% | -18.6% to -0.1% | $56 | -5.1% |
| NFL (nflverse moneylines) | 2018-2025 | 1,545 | -7.0% | -13.5% to -0.2% | $24 | |
| College football (consensus, with a 4.5% margin added back) | 2018-2025 | 3,160 | -12.1% | -17.2% to -6.9% | $0 | |

**It lost money in every sport.** At the average price it lost beyond what luck explains, and more than betting at random would (a bookmaker's margin alone costs 3-5%). The bigger the model's claimed edge, the worse it did: its "edges" of 20% or more lost 10-17% per bet. At the average price, the Premier League price moved away from the model's picks before kickoff (closing-line value -5%), the clearest sign the market knew better.

**What was tried to improve it** (`betting_experiments.py`, results in `records/betting_experiments.csv` and on the Betting page). Each change was tried on the earlier seasons, then checked once on the held-out ones:

| Change (held-out seasons) | Premier League | NFL | College football |
|---|---|---|---|
| The first rule | -9.7% | -7.0% | -12.1% |
| Bet at the best price across bookmakers | **-3.4%** (kept) | one price only | one price only |
| Skip games more than 10 points off the market | -2.2% | -6.7% | -15.4% |
| Blend the model with the market (weight chosen by log loss) | -12.6% | -11.4% | -80.1% |
| Blend, minimum edge and cap tuned on the earlier seasons | -7.0% (+12.3% on the tuning seasons) | -30.1% (+4.4%) | -56.4% |

Shopping for the best price is the only change kept: about 6 points per bet, the range now includes zero, and the closing-line value turns slightly positive. Blending backfired: with the chances pulled toward the market, the only edges left were long shots, where the bookmaker's margin is biggest. The tuned rules are textbook overfitting. And the blend predicts the held-out games no better than the market alone (log loss 0.9611 vs 0.9599 for the Premier League, 0.6104 vs 0.6089 for the NFL), so the models don't know anything the prices don't already include: better bets need better information, not a cleverer rule. This matches the backtests: every model here is less accurate than the betting market, so the games where they disagree most are mostly games where the market is right.

So the site's **Betting** page leads with that verdict and recommends a stake of $0 for every sport whose past-season test lost (and for the NBA and college basketball, which have no past prices) while showing what the rule would stake for any bankroll and Kelly fraction, so the sizing logic is visible. Every suggested bet is logged the first time it's suggested, at that moment's price, and settled after the game (`records/bets.csv`); the live record, including closing-line value from ESPN's closing lines, shows whether anything changes. This is a statistics project, not betting advice.

## Website

A FastAPI app with Jinja templates, currently run locally. Tabs in the header switch between the sports. Every game is a card in the two teams' own colors (`colors.py`; when two teams' colors are too alike, the away team wears its second one), with big win chances, the model's bar next to the betting market's, and a tag where the model and the market disagree by 5 points or more. Team pages chart the model's win chance for each game (and, for NFL teams, the model's rating week by week, refitted as it stood at the time). The site follows the device's light or dark mode, works on phones, and keeps longer explanations behind "How to read this". Premier League pages:

- **Upcoming:** next fixtures with model probabilities next to the bookmakers' odds
- **Match pages:** exact-score grid, over/under (with the market's view), both teams to score
- **Results** by week and **league tables** for every season since 1993/94, with team pages
- **Forecast:** season simulation with title, top-four and relegation chances
- **Track record:** live predictions only, scored against results and the closing market
- **Backtest** and **Ratings:** held-out performance and current team ratings

NBA pages (under `/nba/`): upcoming games with the market's line and who's out, game pages with each team's expected lineup and player projections, the day's likeliest player props, results, standings by conference or division, playoff odds, team and player ratings, team pages with the roster, track record and backtest.

**Today** (`/today/`): every game today in every sport, with the model's pick and chance next to the market's, the biggest disagreements, the bets the rule would make, and the day's likeliest player props. **Betting** (`/betting/`): above. Every page shows when the data was last updated.

Every sport's **track record** page charts the live record week by week: the typical probability the model gave the actual result so far, next to its benchmark (the betting market, or Elo for college games).

NFL pages (under `/nfl/`): upcoming games with win chances, expected margin and total next to the market's moneyline, spread and total; results by week; standings by division; team pages with schedule and quarterbacks; game pages with expected starters, form and every earlier meeting; track record; backtest with calibration; team ratings; playoff odds.

College football (`/cfb/`) and college basketball (`/cbb/`) pages: upcoming games, results, standings for every season filtered by conference, team ratings, team pages, track record and backtest; college football also has conference title odds and each team's quarterbacks.

`build_site.py` can also render the whole site as static HTML for GitHub Pages.

**Betting lines from ESPN** (`espn.py`). There are no free historical lines for the NBA or college basketball, and the college football source has none for the current season, so each update saves the lines ESPN shows (DraftKings: moneyline, spread and total, opening and latest) for recent and upcoming NBA, college basketball and college football games, to `records/espn_lines.csv`. ESPN drops a game's lines once it's over, so the last line saved before a game becomes its closing line (the updates run at 10:07 and 17:37). The site shows the market next to the model on those games, and the track records will compare them as the season goes.

## Automation and testing

- **One command updates everything:** `update.py` runs the results, xG, backtest, fixtures, prediction and availability steps in order, stopping at the first failure.
- **Automatic updates on the Mac:** `scheduled/install.sh` sets up a macOS launchd job that runs `update.py --record` every day at 10:07 and 17:37 (and once on waking, if the Mac was asleep), logging to `logs/update.log`; `scheduled/status.sh` shows how the last runs went. The update now takes too long for GitHub's free runners with the NBA backtest, so the scheduled GitHub run is switched off (its manual button still works) and the records are committed from the Mac.
- **GitHub Actions** runs the test suite on every push, and used to run the update twice a week (Tuesdays and Fridays) runs the update for every sport and commits new predictions to `records/` before kickoff. If one sport's data source fails, the others still update.
- **245 pytest tests** cover the importers (including format quirks and source changes), the model (valid probabilities, recovering known team strengths from simulated leagues, no look-ahead in the backtest), the prediction record (nothing changes after kickoff), the simulator, the NFL importers, models and player projections (including that nothing uses the game being predicted), the college and NBA importers, models, player lineups and props, ESPN's feeds (read from saved samples), playoff and conference title simulations (including tiebreakers), the betting rule (stakes, settling, the bankroll simulation and the live log), and every sport's website. Tests use simulated leagues and temporary databases, never real data or the network.

## Project structure

```
update.py               Runs the weekly update, step by step
db.py                   Shared database helpers; schema.sql defines one database for every sport
app.py, web.py          The website: Premier League pages and shared helpers (NFL pages in nfl/site.py)
colors.py               Team colors for the website, with clash and dark-mode handling
charts.py               The charts: team pages and track records (drawn as HTML and SVG, no chart library)
blend.py                Blending a model with the betting market (weighted on the log-odds scale)
espn.py                 Betting lines from ESPN for NBA and college games, saved as the season goes
betting.py              The betting rule: edges, Kelly stakes, past-season test and the live bet log
betting_experiments.py  What was tried to improve the rule, tested on held-out seasons
overview.py             The Today and Betting pages
scheduled/              The automatic update on the Mac (install, uninstall, status, and the run script)
templates/, static/     Page templates and styles; build_site.py renders the site as static HTML
records/                Official live predictions, saved before kickoff
tests/                  pytest suite

epl/                    Premier League
  import_epl.py           football-data.co.uk results, stats and odds (cached, safe to re-run)
  import_understat.py     Understat xG; fills in results before football-data.co.uk posts them
  import_fixtures.py      Upcoming fixtures and pre-match odds
  model.py                The model, backtest and scoring
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts upcoming matches and records them before kickoff
  records.py              The official prediction record in records/
  simulate.py             Monte Carlo season simulator
  fetch_availability.py   Daily snapshots of player injuries and availability
  error_analysis.py       Where the model loses ground to the market
  promoted_teams.py       How the model handles newly promoted teams
  import_fpl_values.py, import_transfermarkt.py, tune_squad_value.py, tune_value_fade.py
                          Squad-value data and experiments
  blend.py                The model-market blend test (python -m epl.blend; result in epl/blend.json)
  team_aliases.sql        Team-name mappings between data sources
  checks.sql              SQL sanity checks and data exploration queries
  poisson_model.ipynb     Model development notebook

nfl/                    NFL
  import_nfl.py           nflverse games, results, spreads, totals, moneylines and quarterbacks since 1999
  elo.py, tune_elo.py     Elo baseline and its tuning
  model.py, tune_model.py Points model with quarterback ratings, and its tuning and comparison
  experiments.py          Tests of rest, travel, domes, weather, EPA and other ideas (none helped)
  import_nfl_stats.py     Weekly team and player stats with EPA, injury reports and snap counts from nflverse-data
  quarterbacks.py         Who actually starts: nflverse's listed starters checked against box scores and injury reports
  qb_overrides.csv        Starting quarterbacks named by hand (e.g. a benching the data can't see)
  players.py              Player projections, how results spread around them (ranges.json, for the chances on game pages), and their test against benchmarks
  player_experiments.py   Tests of injury reports, snap counts, air yards and defense by position
  predict_players.py      Projects players for upcoming games and records them before kickoff
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts the next 8 days' games and records them before kickoff
  records.py              The official NFL prediction record in records/
  season.py               Playoff odds: the rest of the season simulated 10,000 times, tested on past seasons
  teams.py                Team names, divisions and conferences
  blend.py, blend.json    The model-market blend test and its result
  site.py                 The NFL web pages

nba/                    NBA
  import_nba.py           Games, scores and every player's box score line from sportsdataverse-data (ESPN)
  model.py, tune_model.py Player ratings and lineups (who's available, expected minutes), the Elo benchmark, tuning
  live.py                 Current rosters and the injury report from ESPN, and each team's expected lineup
  run_backtest.py         Saves walk-forward backtest predictions (only new seasons, unless --full)
  predict_upcoming.py     Predicts the next 8 days' games and records them before tip-off
  players.py              Player projections and props (points, rebounds, assists, threes) and their test
  predict_players.py      Projects players for upcoming games and records them before tip-off
  season.py               Playoff odds: the rest of the season with the play-in and playoffs, tested on past seasons
  records.py              The official NBA prediction record in records/
  site.py                 The NBA web pages

cfb/                    College football
  import_cfb.py           Games, scores, conferences, Elo and consensus betting lines from cfbfastR-data
  model.py, tune_model.py Points model (offense and defense ratings) and its tuning
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts the next 8 days' games and records them before kickoff
  records.py              The official college football prediction record in records/
  conference.py           Conference title odds: the rest of the season simulated 10,000 times, tested on past seasons
  site.py                 The college football web pages
  checks.sql              SQL data checks and exploration queries

cbb/                    Men's college basketball
  import_cbb.py           Games, scores, conferences and team colors from sportsdataverse-data (ESPN)
  model.py, tune_model.py Points model (offense and defense ratings), a simple Elo benchmark, and the tuning
  run_backtest.py         Saves walk-forward backtest predictions
  predict_upcoming.py     Predicts the next 8 days' games and records them before tip-off
  records.py              The official basketball prediction record in records/
  site.py                 The college basketball web pages
```

## Running it

Requires Python 3.11+. Run everything from the project folder; each script is a module run with `python -m`.

```bash
git clone https://github.com/emrozin/sports-probabilities.git
cd sports-probabilities
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python update.py              # builds data/sports.db for every sport (the first run downloads every season)
uvicorn app:app --reload      # then open http://127.0.0.1:8000
pytest                        # run the tests
```

## Roadmap

- Betting lines for player stats going forward, to compare player projections with the market
- Compare the NBA and college models with the market as ESPN's lines build up this season
- Conference tournament and NCAA tournament odds for college basketball
- Betting lines for the current college football season, to compare upcoming games with the market
- Player availability: adjust for injured and suspended players, the largest remaining source of the gap to the market
- Test the model-market blend with pre-match (not closing) odds, once they're collected
- Use the fading squad-value adjustment for live predictions

## Acknowledgements

Data comes from football-data.co.uk, Understat, Fantasy Premier League, nflverse, CollegeFootballData (via sportsdataverse's cfbfastR-data) and ESPN (via sportsdataverse-data, and its public scoreboard, roster and injury feeds); please respect their terms of use. Historical FPL prices come from the [FPL Historical Dataset](https://github.com/vaastav/Fantasy-Premier-League) by Vaastav Anand (MIT License), and player valuations from [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0). Understat has no official API, so its importer may need updating if the site changes. This is a statistics project, not betting advice.
