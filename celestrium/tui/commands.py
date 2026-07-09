"""Pure command parsing for the cockpit's unified prompt.

The prompt accepts slash commands, archive-prefixed ADQL, literature queries,
crossmatch/runbook/feed idioms, and bare object names. `parse()` turns raw text
into an `Intent` with no widgets and no network, so the dispatch heuristics are
unit-testable and the single `COMMANDS` table can power parsing, autocomplete,
and the help card without drifting apart.

The app maps each Intent.kind to a worker; this module never executes anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import registry

# --------------------------------------------------------------------------- #
# The command table — one row per prompt idiom (drives parse + suggest + help)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CommandSpec:
    name: str           # what the user types (or the leading idiom)
    usage: str
    description: str
    slash: bool = True  # slash commands feed autocomplete; idioms only feed help


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/resolve", "/resolve <object | 'RA Dec'>",
                "Resolve a target via SIMBAD/Horizons and rank its products."),
    CommandSpec("/query", "/query [archive:] <ADQL>",
                "Run ADQL against gaia/irsa/euclid/heasarc/… (cached)."),
    CommandSpec("/match", "/match <catalogue>  (or: match <cat> · ×<cat>)",
                "Crossmatch the current table against a VizieR catalogue."),
    CommandSpec("/run", "/run <runbook>  (or: run <runbook>)",
                "Execute a runbook with live progress."),
    CommandSpec("/feed", "/feed <neo|satellite|transient>  (or: feed <x>)",
                "Run a global feed into the Query view."),
    CommandSpec("/papers", "/papers [ADS query]  (bare 'papers' = active target)",
                "Literature search via ADS/SciX."),
    CommandSpec("/dossier", "/dossier [target]",
                "Object packet: identity + images + literature → Markdown report."),
    CommandSpec("/field", "/field [RA Dec]",
                "Field packet for a sky position (defaults to the active target)."),
    CommandSpec("/poster", "/poster [target]",
                "Wallpaper poster of a target, the highlighted row, or the active target."),
    CommandSpec("/plot", "/plot <x> [y] [scatter|hist|sky|cmd]",
                "Quick-look plot of the retained table; Ctrl+O opens the PNG."),
    CommandSpec("/sweep", "/sweep [target]",
                "Ask archive products what they know about the active target."),
    CommandSpec("/global", "/global [feed]",
                "Browse live sky/event feeds (NEOs, satellites, transients)."),
    CommandSpec("/history", "/history", "Browse the provenance manifest."),
    CommandSpec("/candidates", "/candidates", "Browse saved candidate lists."),
    CommandSpec("/runbooks", "/runbooks", "Browse curated runbook workflows."),
    CommandSpec("/help", "/help  (or ?)", "Open the command reference."),
    CommandSpec("/clear", "/clear", "Clear the canvas and session state."),
)

SLASH_NAMES: tuple[str, ...] = tuple(c.name for c in COMMANDS if c.slash)

EGGS = {
    "42": "[b gold1]42[/] — the Answer to the Ultimate Question of Life, "
          "the Universe, and Everything. (Now find the Question.)",
    "elite": "[b green]RIGHT ON, COMMANDER![/] Wireframe drive engaged.",
    "thargoid": "[b green]⚠ THARGOID DETECTED[/] — raise shields, deploy E.C.M.",
    "xyzzy": "[dim]Nothing happens.[/]",
    "tea": "[b]Share and Enjoy.[/] ☕  (the Sirius Cybernetics Corp. thanks you)",
    "cake": "[yellow]The cake is a lie.[/]",
}

# Canonical feed-alias table lives in registry (shared with the CLI `feed`).
FEED_ALIASES = registry.FEED_ALIASES
feed_key = registry.feed_key

_LITERATURE_FIELDS = ("abs:", "author:", "year:", "title:", "bibcode:")
_PRODUCT_WORDS = ("image", "cutout", "panel", "spectrum")
_PLOT_KINDS = ("scatter", "hist", "sky", "cmd")


@dataclass(frozen=True)
class Intent:
    """A parsed prompt entry. `kind` selects the worker; `args` parameterise it.

    kinds: empty · clear · help · mode · resolve · query · crossmatch ·
    runbook · feed · literature · papers_context · product · dossier ·
    field · poster · plot · sweep · egg · unknown
    """
    kind: str
    args: dict[str, Any] = field(default_factory=dict)


def _mode(mode: str, **extra) -> Intent:
    return Intent("mode", {"mode": mode, **extra})


def _split_archive_prefix(text: str) -> tuple[str | None, str]:
    """'irsa: SELECT …' -> ('irsa', 'SELECT …'); no known prefix -> (None, text)."""
    for arch in registry.ARCHIVES:
        prefix = f"{arch}:"
        if text.startswith(prefix):
            return arch, text[len(prefix):].strip()
    return None, text


def parse(text: str, mode: str = "resolve") -> Intent:
    """Map raw prompt text to an Intent. `mode` only gates the easter eggs
    (magic words are Resolve-mode folklore, not global commands)."""
    text = text.strip()
    if not text:
        return Intent("empty")
    lower = text.lower()

    if lower in ("clear", "/clear"):
        return Intent("clear")
    if lower in ("/help", "?"):
        return Intent("help")
    if lower.startswith("/history"):
        return _mode("history")
    if lower.startswith("/global") or lower.startswith("/feeds"):
        parts = text.split(maxsplit=1)
        if len(parts) > 1:
            return _mode("global", feed=feed_key(parts[1]))
        return _mode("global")
    if lower.startswith("/runbooks"):
        return _mode("runbooks")
    if lower.startswith("/candidates"):
        return _mode("candidates")

    if lower.startswith("/resolve"):
        target = text.removeprefix("/resolve").strip()
        if target:
            return Intent("resolve", {"target": target})
        return _mode("resolve", usage="/resolve <object name or coordinates>")

    if lower.startswith("/query"):
        q = text.removeprefix("/query").strip()
        if not q:
            return _mode("query", usage="/query <ADQL statement>")
        archive, adql = _split_archive_prefix(q)
        return Intent("query", {"archive": archive or "gaia", "adql": adql})

    if lower.startswith("/dossier") or lower == "dossier" or lower.startswith("dossier "):
        target = text.removeprefix("/dossier").removeprefix("dossier").strip()
        return Intent("dossier", {"target": target or None})

    if lower.startswith("/field") or lower == "field" or lower.startswith("field "):
        position = text.removeprefix("/field").removeprefix("field").strip()
        return Intent("field", {"position": position or None})

    if lower.startswith("/poster") or lower == "poster" or lower.startswith("poster "):
        target = text.removeprefix("/poster").removeprefix("poster").strip()
        return Intent("poster", {"target": target or None})

    if lower.startswith("/plot") or lower == "plot" or lower.startswith("plot "):
        rest = text.removeprefix("/plot").removeprefix("plot").strip()
        parts = rest.split()
        kind = "scatter"
        if len(parts) == 1 and parts[0].lower() in _PLOT_KINDS:
            kind = parts.pop().lower()
        elif parts and parts[-1].lower() in _PLOT_KINDS:
            kind = parts.pop().lower()
        x = parts[0] if parts else None
        y = parts[1] if len(parts) > 1 else None
        return Intent("plot", {"x": x, "y": y, "kind": kind})

    if lower.startswith("/sweep") or lower == "sweep" or lower.startswith("sweep "):
        target = text.removeprefix("/sweep").removeprefix("sweep").strip()
        return Intent("sweep", {"target": target or None})

    if lower.startswith("/papers"):
        q = text.removeprefix("/papers").strip()
        if q:
            return Intent("literature", {"query": q})
        return Intent("papers_context")

    if mode == "resolve" and lower in EGGS:
        return Intent("egg", {"word": lower, "markup": EGGS[lower]})

    if text.startswith("×") or text.startswith("match ") or lower.startswith("/match"):
        catalog = (text.removeprefix("×").removeprefix("/match")
                   .removeprefix("match ").strip())
        if not catalog:
            return _mode("crossmatch", usage="/match <catalogue>, e.g. /match vizier:VIII/65/nvss")
        return Intent("crossmatch", {"catalog": catalog})

    if text.startswith("run ") or lower.startswith("/run"):
        name = text.removeprefix("/run").removeprefix("run ").strip()
        if not name:
            return _mode("runbooks", usage="/run <runbook> — Enter on a row also runs it")
        return Intent("runbook", {"name": name})

    if text.startswith("feed ") or lower.startswith("/feed"):
        key = text.removeprefix("/feed").removeprefix("feed ").strip()
        if not key:
            return _mode("global", usage="/feed <neo|satellite|transient>")
        return Intent("feed", {"key": feed_key(key)})

    archive, adql = _split_archive_prefix(text)
    if archive is not None:
        return Intent("query", {"archive": archive, "adql": adql})
    if lower.startswith("select ") or lower.startswith("select\n"):
        # bare ADQL: archive=None means "whatever archive is active"
        return Intent("query", {"archive": None, "adql": text})

    if any(p in lower for p in _LITERATURE_FIELDS):
        return Intent("literature", {"query": text})
    if text.startswith("papers "):
        return Intent("literature", {"query": text.removeprefix("papers ").strip()})
    if lower == "papers":
        return Intent("papers_context")

    if lower in _PRODUCT_WORDS:
        return Intent("product", {"word": lower})

    if text.startswith("/"):
        return Intent("unknown", {"command": text.split()[0]})

    return Intent("resolve", {"target": text})


def suggestions() -> list[str]:
    """Slash-command completions for the prompt's suggester."""
    return list(SLASH_NAMES)


def help_rows() -> list[tuple[str, str]]:
    """(usage, description) rows for the help card, straight from COMMANDS."""
    return [(c.usage, c.description) for c in COMMANDS]
