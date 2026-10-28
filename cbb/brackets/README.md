# NCAA tournament brackets

On Selection Sunday, save the field as `bracket_<season>.csv` here (season = the year the season
starts: the 2026-27 tournament is `bracket_2026.csv`), with one row per team:

```
region,seed,team
East,1,Duke
East,16,American University
East,16,Mount St. Mary's
...
```

- `team` must be the name the site uses (as on the team's page).
- Two teams on the same region and seed are a First Four game.
- List the regions in Final Four order: the first two regions' winners meet in one semifinal, the
  last two in the other.

The tournament page (/cbb/tournament/) then shows every team's chance of reaching each round, and
updates as results come in (python -m cbb.tournament checks the method on past tournaments).
