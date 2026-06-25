"""access/ — the repo's data-access layer.

Query snippets (gaia, euclid, irsa, mast, heasarc, vizier, datalab_desi,
vo_generic), plus cache (provenance), xmatch (audits), resolvers (object ID),
ads (literature), cutouts (imaging). On top: a shared service layer — registry
(archives/recipes/targets/runbooks as data) + packets (dataclass builders) — that
the hub CLI and future TUI both present over. See README.md and ../roadmap.md.
The `hub` CLI ties them together: `python -m access.hub --help`.
"""
