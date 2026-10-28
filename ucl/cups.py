"""The soccer competitions with pages from the Europe-wide model (ucl/model.py): the three European cups
and four of the big five leagues. The Premier League has its own model and pages (app.py), and sits
first in the same Soccer menu."""

CUPS = {   # our league code: ESPN's slug, name, the site's address, kind
    "UCL": {"code": "UCL", "slug": "uefa.champions", "name": "Champions League", "url": "/champions-league/", "kind": "cup"},
    "UEL": {"code": "UEL", "slug": "uefa.europa", "name": "Europa League", "url": "/europa-league/", "kind": "cup"},
    "UECL": {"code": "UECL", "slug": "uefa.europa.conf", "name": "Conference League", "url": "/conference-league/", "kind": "cup"},
    "ESP": {"code": "ESP", "slug": "esp.1", "name": "La Liga", "url": "/la-liga/", "kind": "league"},
    "GER": {"code": "GER", "slug": "ger.1", "name": "Bundesliga", "url": "/bundesliga/", "kind": "league"},
    "ITA": {"code": "ITA", "slug": "ita.1", "name": "Serie A", "url": "/serie-a/", "kind": "league"},
    "FRA": {"code": "FRA", "slug": "fra.1", "name": "Ligue 1", "url": "/ligue-1/", "kind": "league"},
}
BY_SLUG = {c["slug"]: c for c in CUPS.values()}
LEAGUES = tuple(c for c, v in CUPS.items() if v["kind"] == "league")
EUROPEAN_CUPS = tuple(c for c, v in CUPS.items() if v["kind"] == "cup")
# The Soccer menu: the Premier League first (its own pages), then the rest.
SWITCH = [{"code": "EPL", "name": "Premier League", "pages": {"upcoming": "/premier-league/", "results": "/results/", "table": "/table/",
                                                                "track": "/track-record/", "ratings": "/ratings/"}}] + [
    {"code": c["code"], "name": c["name"], "pages": {"upcoming": c["url"], "results": c["url"] + "results/", "table": c["url"] + "table/",
                                                      "track": c["url"] + "track-record/", "ratings": c["url"] + "ratings/"}}
    for c in (CUPS["ESP"], CUPS["GER"], CUPS["ITA"], CUPS["FRA"], CUPS["UCL"], CUPS["UEL"], CUPS["UECL"])]
