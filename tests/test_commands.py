"""Unit tests for the cockpit's prompt parser (celestrium.tui.commands).

Pure text-in/Intent-out — no Textual app, no network. These pin the dispatch
heuristics that previously lived inline in app._handle_command.
"""
import pytest

pytest.importorskip("textual")
from celestrium.tui import commands  # noqa: E402


def parse(text, mode="resolve"):
    return commands.parse(text, mode=mode)


# ----- slash commands ------------------------------------------------------ #
def test_empty_and_clear_and_help():
    assert parse("").kind == "empty"
    assert parse("   ").kind == "empty"
    assert parse("clear").kind == "clear"
    assert parse("/clear").kind == "clear"
    assert parse("/help").kind == "help"
    assert parse("?").kind == "help"


def test_browser_modes():
    assert parse("/history").args["mode"] == "history"
    assert parse("/runbooks").args["mode"] == "runbooks"
    assert parse("/candidates").args["mode"] == "candidates"
    assert parse("/global").args == {"mode": "global"}


def test_global_with_feed_argument_maps_aliases():
    intent = parse("/global satellites")
    assert intent.kind == "mode"
    assert intent.args == {"mode": "global", "feed": "satellite"}
    assert parse("/feeds cneos").args["feed"] == "neo"


def test_slash_resolve_with_and_without_target():
    intent = parse("/resolve TRAPPIST-1")
    assert intent.kind == "resolve"
    assert intent.args == {"target": "TRAPPIST-1"}
    bare = parse("/resolve")
    assert bare.kind == "mode"
    assert bare.args["mode"] == "resolve"
    assert "usage" in bare.args


def test_slash_query_with_archive_prefix():
    intent = parse("/query irsa: SELECT TOP 1 ra, dec FROM demo")
    assert intent.kind == "query"
    assert intent.args == {"archive": "irsa",
                           "adql": "SELECT TOP 1 ra, dec FROM demo"}


def test_slash_query_defaults_to_gaia():
    intent = parse("/query SELECT TOP 5 source_id FROM gaiadr3.gaia_source")
    assert intent.args["archive"] == "gaia"


def test_unknown_slash_command():
    intent = parse("/frobnicate now")
    assert intent.kind == "unknown"
    assert intent.args["command"] == "/frobnicate"


# ----- prompt idioms ------------------------------------------------------- #
def test_crossmatch_prefixes():
    assert parse("match vizier:VIII/65/nvss").args["catalog"] == "vizier:VIII/65/nvss"
    assert parse("×vizier:VIII/65/nvss").kind == "crossmatch"


def test_runbook_and_feed_prefixes():
    assert parse("run euclid-q1").args["name"] == "euclid-q1"
    intent = parse("feed tles")
    assert intent.kind == "feed"
    assert intent.args["key"] == "satellite"


def test_archive_prefixed_adql():
    intent = parse("heasarc: SELECT TOP 5 name FROM chanmaster")
    assert intent.kind == "query"
    assert intent.args == {"archive": "heasarc",
                           "adql": "SELECT TOP 5 name FROM chanmaster"}


def test_bare_select_uses_active_archive():
    intent = parse("SELECT TOP 5 source_id FROM gaiadr3.gaia_source")
    assert intent.kind == "query"
    assert intent.args["archive"] is None  # app substitutes the active archive


def test_unknown_prefix_is_not_a_query():
    # 'NGC: 1275' is not a registered archive — treat as a resolve target
    intent = parse("NGC: 1275")
    assert intent.kind == "resolve"


# ----- literature ---------------------------------------------------------- #
def test_ads_field_tokens_route_to_literature():
    intent = parse('abs:"cosmic dipole" year:2024-2026')
    assert intent.kind == "literature"
    assert "cosmic dipole" in intent.args["query"]


def test_papers_prefix_and_bare_papers():
    intent = parse("papers Euclid Q1")
    assert intent.kind == "literature"
    assert intent.args["query"] == "Euclid Q1"
    assert parse("papers").kind == "papers_context"


# ----- eggs (resolve-mode folklore only) ----------------------------------- #
def test_eggs_fire_only_in_resolve_mode():
    assert parse("42", mode="resolve").kind == "egg"
    assert parse("xyzzy", mode="resolve").args["word"] == "xyzzy"
    # outside resolve mode the same word is just a resolve target
    assert parse("42", mode="query").kind == "resolve"


# ----- products + default -------------------------------------------------- #
def test_product_words():
    for word in ("image", "cutout", "panel", "spectrum"):
        intent = parse(word)
        assert intent.kind == "product"
        assert intent.args["word"] == word


def test_default_is_resolve():
    assert parse("M87").args == {"target": "M87"}
    assert parse("187.7059 12.3911").kind == "resolve"


# ----- promoted slash forms (discoverability parity with bare idioms) ------ #
def test_slash_match_run_feed_parse_like_bare_forms():
    assert parse("/match vizier:VIII/65/nvss").kind == "crossmatch"
    assert (parse("/match vizier:VIII/65/nvss").args
            == parse("match vizier:VIII/65/nvss").args)
    assert parse("/run euclid-q1").args == parse("run euclid-q1").args
    assert parse("/feed tles").args == parse("feed tles").args


def test_bare_slash_idioms_give_usage_not_unknown():
    for text, mode in (("/match", "crossmatch"), ("/run", "runbooks"),
                       ("/feed", "global")):
        intent = parse(text)
        assert intent.kind == "mode", text
        assert intent.args["mode"] == mode
        assert "usage" in intent.args


def test_slash_papers_forms():
    intent = parse("/papers Euclid Q1 AGN")
    assert intent.kind == "literature"
    assert intent.args["query"] == "Euclid Q1 AGN"
    assert parse("/papers").kind == "papers_context"


# ----- imaging-wing idioms (dossier / field / poster) ----------------------- #
def test_dossier_idiom_slash_and_bare():
    assert parse("/dossier M87").args == {"target": "M87"}
    assert parse("dossier M87").args == {"target": "M87"}
    assert parse("/dossier").args == {"target": None}   # app uses active target
    assert parse("dossier").kind == "dossier"


def test_field_idiom_passes_position_text():
    assert parse("/field 187.70 12.39").args == {"position": "187.70 12.39"}
    assert parse("field").args == {"position": None}


def test_poster_idiom_slash_and_bare():
    assert parse("/poster 3C 273").args == {"target": "3C 273"}
    assert parse("poster").args == {"target": None}     # highlighted row / target


# ----- table-driven surfaces ----------------------------------------------- #
def test_suggestions_are_slash_commands():
    sugg = commands.suggestions()
    assert "/resolve" in sugg and "/query" in sugg
    # the once-hidden idioms are now discoverable from the prompt
    for name in ("/match", "/run", "/feed", "/papers",
                 "/dossier", "/field", "/poster"):
        assert name in sugg
    assert all(s.startswith("/") for s in sugg)


def test_help_rows_cover_every_command():
    rows = commands.help_rows()
    assert len(rows) == len(commands.COMMANDS)
    assert all(usage and desc for usage, desc in rows)
