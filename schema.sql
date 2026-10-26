-- Sports probabilities database (SQLite).
-- Designed so other leagues (La Liga, NFL, ...) fit without schema changes.
-- Every statement uses IF NOT EXISTS, so running this file twice is harmless.

CREATE TABLE IF NOT EXISTS leagues (
    id     INTEGER PRIMARY KEY,
    code   TEXT NOT NULL UNIQUE,          -- 'EPL' now; later 'LALIGA', 'NFL', ...
    name   TEXT NOT NULL,
    sport  TEXT NOT NULL                  -- 'soccer', 'american_football', ...
);

CREATE TABLE IF NOT EXISTS teams (
    id         INTEGER PRIMARY KEY,
    league_id  INTEGER NOT NULL REFERENCES leagues(id),
    name       TEXT NOT NULL,
    UNIQUE (league_id, name)
);

CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY,
    league_id     INTEGER NOT NULL REFERENCES leagues(id),
    season        INTEGER NOT NULL,       -- start year: 2024 means 2024/25
    date          TEXT NOT NULL,          -- ISO format 'YYYY-MM-DD'
    home_team_id  INTEGER NOT NULL REFERENCES teams(id),
    away_team_id  INTEGER NOT NULL REFERENCES teams(id),
    home_score    INTEGER,                -- NULL until the game is played
    away_score    INTEGER,
    status        TEXT NOT NULL DEFAULT 'scheduled'
                  CHECK (status IN ('scheduled', 'final')),
    kickoff       TEXT,                   -- UK local time 'HH:MM', when known (from fixtures.csv)
    UNIQUE (league_id, season, date, home_team_id, away_team_id)
);

-- Soccer-specific match stats. Other sports get their own stats table later.
-- Any column is NULL when the source doesn't have it for that season.
CREATE TABLE IF NOT EXISTS soccer_match_stats (
    game_id               INTEGER PRIMARY KEY REFERENCES games(id),
    ht_home_score         INTEGER,
    ht_away_score         INTEGER,
    home_shots            INTEGER,
    away_shots            INTEGER,
    home_shots_on_target  INTEGER,
    away_shots_on_target  INTEGER,
    home_fouls            INTEGER,
    away_fouls            INTEGER,
    home_corners          INTEGER,
    away_corners          INTEGER,
    home_yellows          INTEGER,
    away_yellows          INTEGER,
    home_reds             INTEGER,
    away_reds             INTEGER,
    referee               TEXT,
    home_xg               REAL,           -- football-data.co.uk's xG (2026/27 on);
                                          -- Understat's xG is in game_xg
    away_xg               REAL
);

CREATE TABLE IF NOT EXISTS odds (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER NOT NULL REFERENCES games(id),
    bookmaker   TEXT NOT NULL,            -- 'Pinnacle', 'Bet365', 'Market average', ...
    market      TEXT NOT NULL,            -- '1x2', 'total', 'asian_handicap' (NFL: 'spread')
    outcome     TEXT NOT NULL,            -- 'home'/'draw'/'away', 'over'/'under'
    price       REAL NOT NULL,            -- decimal odds, e.g. 2.50
    line        REAL,                     -- NULL for 1x2; 2.5 for totals;
                                          -- asian_handicap: the HOME team's handicap on both rows
    is_closing  INTEGER NOT NULL CHECK (is_closing IN (0, 1))
);

-- NULLs never count as equal in a UNIQUE constraint, so a plain
-- UNIQUE(..., line, ...) would let duplicate 1x2 rows in.
-- Indexing COALESCE(line, 0) instead makes re-imports skip existing rows.
CREATE UNIQUE INDEX IF NOT EXISTS odds_unique
    ON odds (game_id, bookmaker, market, outcome, COALESCE(line, 0), is_closing);

-- Every original CSV row, stored as JSON, so no column from the source is ever lost.
-- Query any field with json_extract, e.g. json_extract(data, '$.HS').
CREATE TABLE IF NOT EXISTS raw_source_rows (
    game_id  INTEGER PRIMARY KEY REFERENCES games(id),
    source   TEXT NOT NULL,               -- e.g. 'football-data.co.uk E0 1993'
    data     TEXT NOT NULL                -- JSON object of the original row
);

-- Maps the team names other data sources use to our teams,
-- e.g. Understat's 'Manchester United' -> our 'Man United'. Filled from team_aliases.sql.
CREATE TABLE IF NOT EXISTS team_aliases (
    source       TEXT NOT NULL,           -- e.g. 'understat'
    league_id    INTEGER NOT NULL REFERENCES leagues(id),
    source_name  TEXT NOT NULL,           -- the name as that source writes it
    team_id      INTEGER NOT NULL REFERENCES teams(id),
    PRIMARY KEY (source, league_id, source_name)
);

-- Expected goals per game, kept per source because providers use different xG models.
CREATE TABLE IF NOT EXISTS game_xg (
    game_id          INTEGER NOT NULL REFERENCES games(id),
    source           TEXT NOT NULL,       -- e.g. 'understat'
    home_xg          REAL NOT NULL,
    away_xg          REAL NOT NULL,
    source_match_id  TEXT,                -- the source's own id for the match
    PRIMARY KEY (game_id, source)
);

-- Each prediction's expected goals. Every goals market (over/under, exact scores,
-- both teams to score) can be recalculated exactly from these two numbers.
CREATE TABLE IF NOT EXISTS prediction_goals (
    game_id        INTEGER NOT NULL REFERENCES games(id),
    model_name     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    exp_home       REAL NOT NULL,
    exp_away       REAL NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY,
    game_id        INTEGER NOT NULL REFERENCES games(id),
    model_name     TEXT NOT NULL,         -- e.g. 'poisson'
    model_version  TEXT NOT NULL,         -- e.g. 'v1', so model upgrades can be compared
    outcome        TEXT NOT NULL,
    probability    REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (game_id, model_name, model_version, outcome)
);

-- NFL-specific game details from nflverse. One row per NFL game.
CREATE TABLE IF NOT EXISTS nfl_game_info (
    game_id         INTEGER PRIMARY KEY REFERENCES games(id),
    source_game_id  TEXT NOT NULL UNIQUE,  -- nflverse's id, e.g. '2026_05_SF_SEA'
    game_type       TEXT NOT NULL,         -- 'REG', or playoffs: 'WC', 'DIV', 'CON', 'SB'
    week            INTEGER NOT NULL,
    neutral_site    INTEGER NOT NULL CHECK (neutral_site IN (0, 1)),  -- international games, Super Bowls
    overtime        INTEGER,
    home_rest       INTEGER,               -- days since each team's previous game
    away_rest       INTEGER,
    div_game        INTEGER,               -- 1 if the teams are in the same division
    roof            TEXT,                  -- 'outdoors', 'dome', 'closed', 'open'
    surface         TEXT,
    temp            INTEGER,               -- degrees Fahrenheit (outdoor games)
    wind            INTEGER,               -- mph
    home_qb         TEXT,                  -- starting quarterbacks
    away_qb         TEXT,
    home_coach      TEXT,
    away_coach      TEXT,
    referee         TEXT,
    stadium         TEXT,
    spread_line     REAL,                  -- the market's expected HOME margin: +3.5 = home favored by 3.5
    total_line      REAL,                  -- the market's expected total points
    home_moneyline  REAL,                  -- latest American odds (upcoming games: the current market)
    away_moneyline  REAL
);

-- NFL weekly team and player stats from nflverse (nfl/import_nfl_stats.py), one row per team or
-- player per game. Players appear only in games where they recorded a stat.
CREATE TABLE IF NOT EXISTS nfl_team_stats (
    game_id         INTEGER NOT NULL REFERENCES games(id),
    team            TEXT NOT NULL,
    pass_attempts   INTEGER,
    sacks           INTEGER,
    carries         INTEGER,
    passing_yards   INTEGER,
    rushing_yards   INTEGER,
    passing_epa     REAL,                  -- expected points added on passing plays
    rushing_epa     REAL,
    PRIMARY KEY (game_id, team)
);

CREATE TABLE IF NOT EXISTS nfl_player_stats (
    game_id          INTEGER NOT NULL REFERENCES games(id),
    player_id        TEXT NOT NULL,        -- nflverse's (GSIS) player id
    name             TEXT NOT NULL,
    position         TEXT,
    team             TEXT NOT NULL,
    completions      INTEGER, pass_attempts INTEGER, passing_yards INTEGER,
    passing_tds      INTEGER, interceptions INTEGER,
    carries          INTEGER, rushing_yards INTEGER, rushing_tds INTEGER,
    targets          INTEGER, receptions INTEGER, receiving_yards INTEGER, receiving_tds INTEGER,
    receiving_air_yards INTEGER,           -- how far downfield his targets were thrown, in total
    PRIMARY KEY (game_id, player_id)
);

-- Live NFL player projections (nfl/predict_players.py), saved before kickoff like predictions.
CREATE TABLE IF NOT EXISTS nfl_player_projections (
    game_id      INTEGER NOT NULL REFERENCES games(id),
    player_id    TEXT NOT NULL,
    name         TEXT NOT NULL,
    position     TEXT,
    team         TEXT NOT NULL,
    pass_att REAL, pass_yds REAL, pass_tds REAL, ints REAL,
    targets REAL, receptions REAL, rec_yds REAL, rec_tds REAL,
    carries REAL, rush_yds REAL, rush_tds REAL, anytime_td REAL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (game_id, player_id)
);

-- NFL weekly injury reports from nflverse (nfl/import_nfl_stats.py): each listed player's game status.
-- Weekly NFL rosters from nflverse-data (nfl/import_nfl_stats.py): every player's status each week.
CREATE TABLE IF NOT EXISTS nfl_rosters (
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    team       TEXT NOT NULL,
    player_id  TEXT NOT NULL,              -- nflverse's (GSIS) player id
    name       TEXT,
    position   TEXT,
    status     TEXT,                       -- 'ACT' active, 'RES' injured reserve and other reserve lists, 'INA' inactive,
                                           -- 'DEV' practice squad, 'CUT' released, ...
    PRIMARY KEY (season, week, team, player_id)
);

CREATE TABLE IF NOT EXISTS nfl_injuries (
    season     INTEGER NOT NULL,
    week       INTEGER NOT NULL,
    team       TEXT NOT NULL,
    player_id  TEXT NOT NULL,              -- nflverse's (GSIS) player id, as in nfl_player_stats
    name       TEXT,
    position   TEXT,
    status     TEXT,                       -- 'Out', 'Doubtful', 'Questionable', 'Probable' (older seasons), or empty
    PRIMARY KEY (season, week, team, player_id)
);

-- Share of each team's offensive snaps per player (from 2013), from nflverse (nfl/import_nfl_stats.py).
-- nflverse identifies these players by a different id, so they're matched to nfl_player_stats by
-- game, team and name; player_id is empty when there's no match (e.g. linemen, who record no stats).
CREATE TABLE IF NOT EXISTS nfl_snap_counts (
    game_id       INTEGER NOT NULL REFERENCES games(id),
    team          TEXT NOT NULL,
    name          TEXT NOT NULL,
    position      TEXT,
    player_id     TEXT,
    offense_snaps INTEGER,
    offense_pct   REAL,
    PRIMARY KEY (game_id, team, name)
);

-- NFL playoff odds (nfl/season.py): each run's simulated chances, one row per team, kept as a history.
CREATE TABLE IF NOT EXISTS nfl_season_odds (
    season     INTEGER NOT NULL,
    as_of      TEXT NOT NULL,              -- the date the odds were made (games before it are known)
    team       TEXT NOT NULL,
    wins       REAL NOT NULL,              -- average regular-season wins
    division   REAL NOT NULL,              -- chance of winning the division
    playoffs   REAL NOT NULL,
    top_seed   REAL NOT NULL,
    conference REAL NOT NULL,              -- chance of reaching the Super Bowl
    champion   REAL NOT NULL,              -- chance of winning it
    PRIMARY KEY (season, as_of, team)
);

-- College football game details from cfbfastR-data (cfb/import_cfb.py), one row per game.
CREATE TABLE IF NOT EXISTS cfb_game_info (
    game_id          INTEGER PRIMARY KEY REFERENCES games(id),
    source_game_id   TEXT NOT NULL UNIQUE,     -- ESPN's game id, as used by CollegeFootballData
    week             INTEGER NOT NULL,
    season_type      TEXT NOT NULL,            -- 'regular' or 'postseason'
    neutral_site     INTEGER NOT NULL CHECK (neutral_site IN (0, 1)),
    conference_game  INTEGER,
    home_division    TEXT, away_division TEXT, -- 'fbs' or 'fcs'
    home_conference  TEXT, away_conference TEXT,
    home_elo         REAL, away_elo REAL,      -- CollegeFootballData's pre-game Elo (a benchmark)
    venue            TEXT,
    notes            TEXT,                     -- e.g. bowl names
    spread_line      REAL,                     -- consensus closing spread, the HOME team's handicap
    total_line       REAL                      -- consensus closing total
);

-- College football starting quarterbacks (cfb/import_cfb.py), from cfbfastR-data's play-by-play player
-- stats (2014 on): the player with the most dropbacks for each team in each game.
CREATE TABLE IF NOT EXISTS cfb_starters (
    game_id   INTEGER NOT NULL REFERENCES games(id),
    side      TEXT NOT NULL CHECK (side IN ('home', 'away')),
    qb_id     TEXT NOT NULL,        -- ESPN's player id (follows a player who transfers)
    qb_name   TEXT NOT NULL,
    attempts  INTEGER,              -- his passes in the game (including sacks)
    PRIMARY KEY (game_id, side)
);

-- Men's college basketball game details from sportsdataverse-data (cbb/import_cbb.py).
-- season is the starting year: 2025 means the 2025-26 season.
CREATE TABLE IF NOT EXISTS cbb_game_info (
    game_id          INTEGER PRIMARY KEY REFERENCES games(id),
    source_game_id   TEXT NOT NULL UNIQUE,     -- ESPN's game id
    season_type      TEXT NOT NULL,            -- 'regular' or 'postseason'
    neutral_site     INTEGER NOT NULL CHECK (neutral_site IN (0, 1)),
    conference_game  INTEGER,
    home_conference  TEXT, away_conference TEXT, -- NULL for schools outside Division I
    notes            TEXT                      -- e.g. "NCAA Men's Basketball Championship - Final Four"
);

-- Team colors for college basketball (from the same source).
CREATE TABLE IF NOT EXISTS cbb_teams (
    name       TEXT PRIMARY KEY,
    color      TEXT,
    alt_color  TEXT
);

-- College football conference title odds (cfb/conference.py), one row per team per date.
CREATE TABLE IF NOT EXISTS cfb_conference_odds (
    season     INTEGER NOT NULL,
    as_of      TEXT NOT NULL,              -- the date the odds were made (games before it are known)
    conference TEXT NOT NULL,
    team       TEXT NOT NULL,
    conf_wins  REAL NOT NULL,              -- average conference wins
    conf_games INTEGER NOT NULL,
    wins       REAL NOT NULL,              -- average regular-season wins
    games      INTEGER NOT NULL,
    title_game REAL NOT NULL,              -- chance of reaching the conference title game
    champion   REAL NOT NULL,              -- chance of winning it
    PRIMARY KEY (season, as_of, team)
);

-- NBA game details from sportsdataverse-data's ESPN files (nba/import_nba.py), one row per game.
CREATE TABLE IF NOT EXISTS nba_game_info (
    game_id        INTEGER PRIMARY KEY REFERENCES games(id),
    source_game_id TEXT UNIQUE NOT NULL,       -- ESPN's game id
    season_type    TEXT NOT NULL,              -- 'regular', 'cup_final' (doesn't count in the standings), 'playin', 'playoffs'
    neutral_site   INTEGER NOT NULL DEFAULT 0,
    notes          TEXT                        -- e.g. 'NBA Cup - Group Play', 'East 1st Round - Game 3'
);

-- The 30 NBA franchises under their current names (ESPN ids), with conference, division and colors.
CREATE TABLE IF NOT EXISTS nba_teams (
    espn_id      INTEGER PRIMARY KEY,
    name         TEXT UNIQUE NOT NULL,
    abbreviation TEXT,
    conference   TEXT,
    division     TEXT,
    color        TEXT,
    alt_color    TEXT
);

-- Every player's line in every finished NBA game, including players who didn't play (with the reason).
CREATE TABLE IF NOT EXISTS nba_player_games (
    game_id      INTEGER NOT NULL REFERENCES games(id),
    team         TEXT NOT NULL,
    player_id    TEXT NOT NULL,                -- ESPN athlete id
    name         TEXT NOT NULL,
    position     TEXT,
    starter      INTEGER NOT NULL DEFAULT 0,
    minutes      REAL NOT NULL DEFAULT 0,
    did_not_play INTEGER NOT NULL DEFAULT 0,
    reason       TEXT,
    points REAL, rebounds REAL, assists REAL, threes REAL, steals REAL, blocks REAL, turnovers REAL, plus_minus REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS nba_player_games_player ON nba_player_games (player_id);

-- Betting lines from ESPN's scoreboard feed (espn.py): one row per game, the latest seen (the
-- closing line once final = 1). Moneylines as decimal odds; spread is the home team's line.
CREATE TABLE IF NOT EXISTS espn_lines (
    league         TEXT NOT NULL,              -- 'NBA', 'CBB', 'CFB'
    source_game_id TEXT NOT NULL,              -- ESPN's game id
    game_id        INTEGER NOT NULL REFERENCES games(id),
    date           TEXT,
    provider       TEXT,                       -- the sportsbook ESPN shows, e.g. 'DraftKings'
    home_ml_open   REAL,
    away_ml_open   REAL,
    home_ml        REAL,
    away_ml        REAL,
    spread         REAL,
    total          REAL,
    final          INTEGER NOT NULL DEFAULT 0, -- 1 once the game has started: these are the closing lines
    updated_at     TEXT,
    PRIMARY KEY (league, source_game_id)
);

-- The NBA's current rosters and injury report from ESPN (nba/live.py), replaced on each update.
CREATE TABLE IF NOT EXISTS nba_rosters_now (
    team       TEXT NOT NULL,
    player_id  TEXT NOT NULL,                  -- ESPN athlete id, as in nba_player_games
    name       TEXT,
    position   TEXT,
    fetched_at TEXT,
    PRIMARY KEY (team, player_id)
);
CREATE TABLE IF NOT EXISTS nba_injuries_now (
    team        TEXT NOT NULL,
    player_id   TEXT NOT NULL,
    name        TEXT,
    status      TEXT,                          -- 'Out', 'Doubtful', 'Day-To-Day', ...
    return_date TEXT,                          -- ESPN's expected return date, if given
    comment     TEXT,
    updated     TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (team, player_id)
);

-- NBA playoff odds (nba/season.py), one row per team per date.
CREATE TABLE IF NOT EXISTS nba_season_odds (
    season            INTEGER NOT NULL,
    as_of             TEXT NOT NULL,
    team              TEXT NOT NULL,
    conference        TEXT,
    wins              REAL NOT NULL,          -- average regular-season wins
    top_six           REAL NOT NULL,          -- straight into the playoffs
    playoffs          REAL NOT NULL,          -- directly or through the play-in
    top_seed          REAL NOT NULL,
    conference_finals REAL NOT NULL,
    finals            REAL NOT NULL,
    champion          REAL NOT NULL,
    PRIMARY KEY (season, as_of, team)
);

-- NBA player projections for upcoming games (nba/predict_players.py), saved before tip-off.
CREATE TABLE IF NOT EXISTS nba_player_projections (
    game_id       INTEGER NOT NULL REFERENCES games(id),
    team          TEXT NOT NULL,
    player_id     TEXT NOT NULL,
    name          TEXT,
    minutes       REAL,
    points        REAL,
    rebounds      REAL,
    assists       REAL,
    threes        REAL,
    model_version TEXT,
    created_at    TEXT,
    PRIMARY KEY (game_id, player_id)
);

-- Bets the betting rule suggested (betting.py), logged the first time each was suggested, at that
-- moment's price, and never changed: one per game.
CREATE TABLE IF NOT EXISTS bets (
    league       TEXT NOT NULL,
    game_id      INTEGER NOT NULL REFERENCES games(id),
    outcome      TEXT NOT NULL,              -- 'home', 'away' or 'draw'
    p            REAL NOT NULL,              -- the model's chance
    market       REAL NOT NULL,              -- the price's chance, bookmaker's margin removed
    price        REAL NOT NULL,              -- decimal odds
    edge         REAL NOT NULL,              -- expected return per $1: p x price - 1
    kelly        REAL NOT NULL,              -- full Kelly stake as a share of the bankroll
    suggested_at TEXT NOT NULL,
    PRIMARY KEY (league, game_id)
);

-- How the market (and the model) moved before each game (betting.snapshot_lines): one row per update
-- when either changed. taken_at 'open' is the sportsbook's opening line, where ESPN gives it.
CREATE TABLE IF NOT EXISTS line_history (
    league       TEXT NOT NULL,
    game_id      INTEGER NOT NULL REFERENCES games(id),
    taken_at     TEXT NOT NULL,              -- 'YYYY-MM-DD HH:MM' Eastern, or 'open'
    market_home  REAL,                       -- the market's chances, bookmaker's margin removed
    market_draw  REAL,
    model_home   REAL,                       -- the model's chances at that moment
    model_draw   REAL,
    PRIMARY KEY (game_id, taken_at)
);
