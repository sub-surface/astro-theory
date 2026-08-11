"""Canonical on-disk locations, shared by every surface.

`hub.py` keeps its own module-level dir attrs (the test suite monkeypatches
them); everything else reads these. All of `data/` is git-ignored (see
CLAUDE.md) — the *durable* record lives in the ledger and, for anything worth
keeping, in the vault (see `celestrium/vault/`).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
REPORTS_DIR = DATA / "reports"
ATLAS_DIR = DATA / "atlas"
POSTERS_DIR = DATA / "posters"
EXPORTS_DIR = DATA / "exports"

# The kernel's world: one SQLite ledger, one payload tree beneath it.
LEDGER_DB = DATA / "celestrium.db"
ARTIFACTS = DATA / "artifacts"

# Where vault writes are staged when direct-write is off (see vault.sync).
VAULT_OUT = DATA / "vault-out"
