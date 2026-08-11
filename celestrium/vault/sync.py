"""Read claims from the vault; write results back without disturbing prose."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .. import config, paths

DEFAULT_VAULT = paths.REPO.parent / "Psychograph-Vault"
BLOCK_START = "<!-- celestrium:begin -->"
BLOCK_END = "<!-- celestrium:end -->"

# Generated subfolders — one gitignore rule each in the vault (see `celestrium
# vault init`). Hand-written notes live in the parent folders and stay tracked.
RUNS_DIR = ("Log", "runs")
OBJECTS_DIR = ("Objects", "generated")
FIELDS_DIR = ("Fields", "generated")
ATTACH_DIR = ("_attachments",)
OBSERVABLES_DIR = ("Observables",)

# Frontmatter keys Celestrium may write. `verdict` and `status` are the vault's
# own schema; everything machine-specific is namespaced so it is greppable and
# obviously not hand-authored.
WRITABLE_KEYS = ("status", "verdict", "rows")
MACHINE_PREFIX = "celestrium_"


# --------------------------------------------------------------------------- #
# locating the vault
# --------------------------------------------------------------------------- #
def vault_root() -> Optional[Path]:
    configured = config.get("vault_path")
    candidate = Path(configured) if configured else DEFAULT_VAULT
    return candidate if candidate.is_dir() else None


def astronomy_dir() -> Optional[Path]:
    root = vault_root()
    if root is None:
        return None
    target = root / "Astronomy"
    return target if target.is_dir() else None


def available() -> bool:
    return astronomy_dir() is not None


def _require() -> Path:
    target = astronomy_dir()
    if target is None:
        raise FileNotFoundError(
            "no vault Astronomy folder; set one with "
            "`celestrium vault path <dir>` (looked for "
            f"{DEFAULT_VAULT / 'Astronomy'})")
    return target


def _dir(*parts: str) -> Path:
    target = _require().joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------- #
# notes: parse + surgical write
# --------------------------------------------------------------------------- #
def parse_note(path: Path) -> tuple:
    """(frontmatter dict, body text). Tolerates a BOM and a missing header."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        import yaml
        front = yaml.safe_load(parts[1]) or {}
    except Exception:
        front = {}
    return (front if isinstance(front, dict) else {}), parts[2].lstrip("\n")


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ':#"\'\n') or text.strip() != text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def set_frontmatter(text: str, updates: dict) -> str:
    """Replace/insert *scalar* frontmatter keys, leaving every other line — and
    every list, comment and blank — exactly as it was."""
    if not updates:
        return text
    had_bom = text.startswith("﻿")
    body = text.lstrip("﻿")
    if not body.startswith("---"):
        header = "\n".join(f"{k}: {_format_value(v)}" for k, v in updates.items())
        return ("﻿" if had_bom else "") + f"---\n{header}\n---\n\n{body}"

    lines = body.split("\n")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return text

    remaining = dict(updates)
    for index in range(1, end):
        match = re.match(r"^([A-Za-z0-9_\-]+):(.*)$", lines[index])
        if not match:
            continue
        key = match.group(1)
        if key not in remaining:
            continue
        # A key whose value continues onto indented lines (a YAML list) is left
        # alone — we only ever rewrite scalars.
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if match.group(2).strip() == "" and following.startswith((" ", "\t", "-")):
            remaining.pop(key)
            continue
        lines[index] = f"{key}: {_format_value(remaining.pop(key))}"

    if remaining:
        insert = [f"{k}: {_format_value(v)}" for k, v in remaining.items()]
        lines[end:end] = insert
    return ("﻿" if had_bom else "") + "\n".join(lines)


def upsert_block(text: str, content: str) -> str:
    """Write machine content inside the fence, creating it at the end if absent."""
    block = f"{BLOCK_START}\n{content.rstrip()}\n{BLOCK_END}"
    if BLOCK_START in text and BLOCK_END in text:
        head, _, rest = text.partition(BLOCK_START)
        _, _, tail = rest.partition(BLOCK_END)
        return head + block + tail
    return text.rstrip() + "\n\n" + block + "\n"


# --------------------------------------------------------------------------- #
# reading claims
# --------------------------------------------------------------------------- #
def read_studies() -> dict:
    """`Observables/*.md` frontmatter → study fields, keyed by the note's `id`."""
    folder = _require().joinpath(*OBSERVABLES_DIR)
    found: dict = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.md")):
        front, _ = parse_note(path)
        study_id = str(front.get("id") or path.stem).strip()
        if not study_id or study_id.lower() in ("observables", "index"):
            continue
        archives = front.get("archive") or front.get("archives") or ()
        if isinstance(archives, str):
            archives = (archives,)
        found[study_id] = {
            "title": str(front.get("title") or path.stem).strip('"'),
            "signature": str(front.get("signature") or ""),
            "refutation": str(front.get("refutation") or ""),
            "status": str(front.get("status") or "idea"),
            "leverage": str(front.get("leverage") or ""),
            "rows": str(front.get("rows") or ""),
            "verdict": "" if front.get("verdict") in (None, "null") else str(front["verdict"]),
            "archives": tuple(str(a) for a in archives),
            "note": path.name,
        }
    return found


def read_studies_safe() -> dict:
    try:
        return read_studies()
    except Exception:
        return {}


def observable_path(study_id: str) -> Optional[Path]:
    folder = _require().joinpath(*OBSERVABLES_DIR)
    if not folder.is_dir():
        return None
    for path in sorted(folder.glob("*.md")):
        front, _ = parse_note(path)
        if str(front.get("id") or path.stem).strip() == study_id:
            return path
    return None


# --------------------------------------------------------------------------- #
# writing results
# --------------------------------------------------------------------------- #
def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\- ]+", "", str(text)).strip().replace(" ", "-")
    return re.sub(r"-+", "-", cleaned).lower()[:60] or "untitled"


def log_run(*, title: str, study: Optional[str], artifact_id: str, cap: str,
            summary: str, body: str = "", dry_run: bool = False) -> Path:
    """Append a run log. The machine half is filled in; the reasoning half is
    left blank on purpose — `Log.md` asks for *why these cuts, what surprised
    you*, which is the one thing the instrument must not invent."""
    folder = _dir(*RUNS_DIR)
    path = folder / f"{_stamp()}-{_slug(study or cap)}-{artifact_id[:6]}.md"
    content = f"""---
id: run-{artifact_id}
title: "{title}"
tags: [astronomy, run, celestrium]
study: {study or "null"}
artifact: {artifact_id}
capability: {cap}
created: {_stamp()}
---

# {title}

{summary}

{body}

## What I expected

## What surprised me

## What to vary next
"""
    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return path


def publish_artifact(kernel, ref: str, *, kind: str = "auto",
                     dry_run: bool = False) -> Path:
    """Put a finished artifact into the notebook, in the folder its shape implies."""
    artifact = kernel.ledger.find(ref)
    if artifact is None:
        raise KeyError(f"no artifact {ref!r}")
    methods = kernel.methods(artifact.id)

    if kind == "auto":
        kind = ("object" if artifact.cap.startswith("object.dossier")
                else "field" if artifact.cap.startswith("lit.census")
                else "run")

    if kind == "object":
        folder = _dir(*OBJECTS_DIR)
        name = artifact.params.get("target") or artifact.label or artifact.id
        path = folder / f"{_slug(name)}.md"
        header = (f"---\nid: object-{_slug(name)}\ntitle: \"{name}\"\n"
                  f"tags: [astronomy, object, celestrium]\n"
                  f"artifact: {artifact.id}\ncreated: {_stamp()}\n---\n\n")
        body = kernel.load(artifact.id) if artifact.kind == "document" else artifact.label
        content = header + str(body) + "\n\n## Provenance\n\n" + methods + "\n"
    elif kind == "field":
        folder = _dir(*FIELDS_DIR)
        path = folder / f"{_slug(artifact.label or artifact.cap)}.md"
        content = (f"---\nid: field-{_slug(artifact.label or artifact.cap)}\n"
                   f"title: \"{artifact.label or artifact.cap}\"\n"
                   f"tags: [astronomy, field, celestrium]\n"
                   f"artifact: {artifact.id}\ncreated: {_stamp()}\n---\n\n"
                   f"# {artifact.label or artifact.cap}\n\n"
                   f"## Provenance\n\n{methods}\n")
    else:
        return log_run(title=artifact.label or artifact.cap, study=artifact.study,
                       artifact_id=artifact.id, cap=artifact.cap,
                       summary=artifact.label, body=methods, dry_run=dry_run)

    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return path


def copy_figure(artifact, *, dry_run: bool = False) -> Optional[str]:
    """Copy an image/figure into the vault attachments; return its embed name."""
    source = artifact.full_path
    if source is None or not source.exists():
        return None
    folder = _dir(*ATTACH_DIR)
    target = folder / f"{artifact.id}{source.suffix}"
    if not dry_run:
        target.write_bytes(source.read_bytes())
    return target.name


def update_observable(study_id: str, updates: dict, *, block: str = "",
                      dry_run: bool = False) -> Optional[Path]:
    """Write machine facts back to the claim note — frontmatter keys and the
    fenced block only. Prose is never touched."""
    path = observable_path(study_id)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8-sig")
    allowed = {k: v for k, v in updates.items()
               if k in WRITABLE_KEYS or k.startswith(MACHINE_PREFIX)}
    text = set_frontmatter(text, allowed)
    if block:
        text = upsert_block(text, block)
    if not dry_run:
        path.write_text(text, encoding="utf-8")
    return path


def gitignore_lines() -> list:
    return [
        "",
        "# Celestrium — generated research output (the ledger is authoritative)",
        "Astronomy/Log/runs/",
        "Astronomy/Objects/generated/",
        "Astronomy/Fields/generated/",
        "Astronomy/_attachments/",
    ]


def ensure_gitignore(dry_run: bool = False) -> tuple:
    """Add the generated-output ignores to the vault's .gitignore (idempotent)."""
    root = vault_root()
    if root is None:
        raise FileNotFoundError("no vault configured")
    path = root / ".gitignore"
    existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    missing = [line for line in gitignore_lines()
               if line.strip() and line not in existing]
    if missing and not dry_run:
        block = "\n".join(gitignore_lines()) if "Celestrium" not in existing \
            else "\n" + "\n".join(missing)
        path.write_text(existing.rstrip("\n") + block + "\n", encoding="utf-8")
    return path, missing
