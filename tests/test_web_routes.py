"""The HTTP route table the snapshot loop serves from."""

from __future__ import annotations

from fifa_fantasy.web.render import PAGES, resolve_page


def test_pages_registry_is_exactly_four():
    assert set(PAGES) == {"index.html", "algorithms.html",
                          "intelligence.html", "research.html"}
    for builder in PAGES.values():
        assert callable(builder)


def test_resolve_page_mappings():
    assert resolve_page("/") == "index.html"
    assert resolve_page("") == "index.html"
    assert resolve_page("/index.html") == "index.html"
    assert resolve_page("/algorithms") == "algorithms.html"
    assert resolve_page("/algorithms.html") == "algorithms.html"
    assert resolve_page("/algorithms?x=1") == "algorithms.html"
    assert resolve_page("/algorithms.html#gbm") == "algorithms.html"
    assert resolve_page("/intelligence") == "intelligence.html"
    assert resolve_page("/research") == "research.html"


def test_resolve_page_static_paths_fall_through():
    assert resolve_page("/foo.json") is None
    assert resolve_page("/figures/fig_gk_sweep.svg") is None
    assert resolve_page("/report/report_data.json") is None
    assert resolve_page("/a_recommendation_ensemble_QF_1.md") is None
