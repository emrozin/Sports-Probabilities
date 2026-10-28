// Small helpers for every page: favorite teams (saved in this browser only), the "Only my teams"
// filter, and the page menu on phones. The site works without them.
(function () {
  "use strict";
  var KEY = "favoriteTeams";

  function read() {
    try { return new Set(JSON.parse(window.localStorage.getItem(KEY) || "[]")); } catch (e) { return new Set(); }
  }
  function write(set) {
    try { window.localStorage.setItem(KEY, JSON.stringify(Array.from(set))); } catch (e) { /* private mode: not saved */ }
  }

  function apply() {
    var favs = read();
    document.querySelectorAll("[data-follow]").forEach(function (b) {
      var on = favs.has(b.dataset.follow);
      b.setAttribute("aria-pressed", on ? "true" : "false");
      var label = b.querySelector(".follow-label");
      if (label) label.textContent = on ? "Following" : "Follow";
    });
    var mine = [];
    document.querySelectorAll("[data-teams]").forEach(function (el) {
      var is = el.dataset.teams.split("|").some(function (t) { return favs.has(t); });
      el.classList.toggle("is-mine", is);
      if (is) mine.push(el);
    });
    // your teams' games first in each list
    mine.slice().reverse().forEach(function (el) {
      if (el.parentElement && !el.parentElement.hasAttribute("data-keep-order")) el.parentElement.prepend(el);
    });
    var filter = document.querySelector(".mine-filter");
    if (filter) filter.hidden = mine.length === 0;
    if (mine.length === 0) document.body.classList.remove("only-mine");
  }

  document.addEventListener("click", function (e) {
    var follow = e.target.closest("[data-follow]");
    if (follow) {
      var favs = read(), key = follow.dataset.follow;
      if (favs.has(key)) favs.delete(key); else favs.add(key);
      write(favs);
      apply();
      return;
    }
    var only = e.target.closest(".mine-filter button");
    if (only) {
      var on = document.body.classList.toggle("only-mine");
      only.setAttribute("aria-pressed", on ? "true" : "false");
      only.textContent = on ? "Show every game" : "★ Only my teams";
      return;
    }
    var searchButton = e.target.closest(".search-toggle");
    if (searchButton) {
      document.body.classList.toggle("search-open");
      var input = document.querySelector(".site-search input");
      if (input && document.body.classList.contains("search-open")) input.focus();
      return;
    }
    var toggle = e.target.closest(".nav-toggle");
    if (toggle) {
      var open = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      document.getElementById(toggle.getAttribute("aria-controls")).classList.toggle("open", open);
    }
  });

  // the phone menu button shows the current page's name
  var current = document.querySelector("#page-nav a[aria-current='page']");
  var label = document.querySelector(".nav-toggle-label");
  if (current && label) label.textContent = current.textContent;
  // the selected sport tab is scrolled into view on narrow screens
  var tab = document.querySelector(".sport-tabs a[aria-current='page']");
  if (tab && tab.scrollIntoView && window.innerWidth < 640) tab.scrollIntoView({ block: "nearest", inline: "center" });

  apply();
})();
