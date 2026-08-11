"""The Obsidian bridge — the vault is the notebook, the ledger is the instrument.

`Psychograph-Vault/Astronomy/` already declares this division in its own index
note: *"The notebook, not the instrument… This folder holds the claims."* So
Celestrium does not invent a second schema. It reads the claims, writes back
what a machine can honestly know (status, rows, verdict-supporting numbers, run
logs), and never touches prose.

Two guardrails, both structural:
  * frontmatter writes are surgical and confined to known keys (plus
    `celestrium_*`); body writes go inside a `<!-- celestrium:begin/end -->`
    fence and nowhere else.
  * generated notes land in `Log/runs/`, `Objects/generated/`,
    `Fields/generated/` and `_attachments/` — all gitignored in the vault, so
    machine output never lands in a public commit.
"""
from .sync import (astronomy_dir, available, log_run, publish_artifact,
                   read_studies, read_studies_safe, update_observable,
                   vault_root)

__all__ = ["vault_root", "astronomy_dir", "available", "read_studies",
           "read_studies_safe", "log_run", "publish_artifact",
           "update_observable"]
