"""Prepare the website's slow parts after an update, so no visitor waits for them: the NBA model
(it needs every player line), the Champions League ratings (every club in Europe) and the betting page's past-season tests. They're saved in
data/cache/ and reused until the database changes. Run by update.py as its last step; safe to run
any time: python -m warm_cache
"""

import time


def main() -> None:
    started = time.time()
    import nba.site as nba_site
    import overview
    nba_site._cache.clear()
    ready = []
    if nba_site.current() is not None:
        ready.append("the NBA model")
    import ucl.site as ucl_site
    ucl_site._cache.clear()
    if ucl_site.current_model() is not None and ucl_site.current_model(league=True) is not None:
        ready.append("the European cups' and leagues' ratings")
    overview._cache.clear()
    if overview.histories():
        ready.append("the betting page's past-season tests")
    print(f"Website cache: {', '.join(ready) if ready else 'nothing to prepare'} ready ({time.time() - started:.0f}s).")


if __name__ == "__main__":
    main()
