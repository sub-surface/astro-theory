"""Artifact — the one typed thing every capability produces.

Content-addressed: `id = blake2b(capability, canonical(params), inputs)`. Four
things fall out of that single choice, none of which needs its own machinery:

  * **the cache is the ledger** — "is this cached?" is "does this id exist?"
  * **lineage** — `inputs` are parent ids, so the ledger is a DAG not a log
  * **reproducibility** — the id *is* the recipe; `repro <id>` rebuilds it
  * **dedupe** across surfaces, sessions and studies, for free

`canonical_text` (deterministic, for hashing) is deliberately separate from
`jsonable` (faithful, for storage): floats hash via `repr` so 0.1 can never
drift to 0.09999999999999999 between runs, but they're stored as real numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import paths

# Payload kinds. `surface` is a study result grid (params → metric); it is a
# table, but tagged apart so views can render it as the answer, not as rows.
KINDS = ("table", "image", "figure", "spectrum", "document", "data", "surface")

SUFFIX = {
    "table": ".ecsv", "surface": ".ecsv", "document": ".md", "data": ".json",
    "image": ".png", "figure": ".png", "spectrum": ".png",
}


def _unwrap(value: Any) -> Any:
    """numpy scalars / astropy quantities → plain Python."""
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        try:
            return item()
        except Exception:
            pass
    return value


def jsonable(value: Any) -> Any:
    """Faithful JSON-safe conversion — numbers stay numbers (for storage)."""
    value = _unwrap(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no NaN/Infinity. NaN means "no value" -> null (a "nan"
        # string would read back as text in a numeric field); +-inf stay
        # representable as strings.
        if math.isnan(value):
            return None
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):          # numpy arrays
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def canonical_text(value: Any) -> str:
    """Deterministic string form used for hashing (floats via repr)."""
    value = _unwrap(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Path):
        return json.dumps(value.as_posix())
    if isinstance(value, dict):
        inner = ",".join(f"{json.dumps(str(k))}:{canonical_text(v)}"
                         for k, v in sorted(value.items(), key=lambda kv: str(kv[0])))
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_text(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "[" + ",".join(sorted(canonical_text(v) for v in value)) + "]"
    if hasattr(value, "tolist"):
        try:
            return canonical_text(value.tolist())
        except Exception:
            pass
    return json.dumps(str(value))


def artifact_id(cap: str, params: dict, inputs: tuple = ()) -> str:
    """The content address of a (capability, params, inputs) triple."""
    blob = "|".join((cap, canonical_text(params or {}), canonical_text(list(inputs))))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Artifact:
    """One produced thing. `path` is repo-relative so the ledger stays portable."""
    id: str
    kind: str
    cap: str
    params: dict = field(default_factory=dict)
    inputs: tuple = ()
    path: str | None = None
    meta: dict = field(default_factory=dict)
    label: str = ""
    study: str | None = None
    created: str = ""

    @property
    def full_path(self) -> Path | None:
        if not self.path:
            return None
        candidate = Path(self.path)
        return candidate if candidate.is_absolute() else paths.REPO / candidate

    @property
    def exists(self) -> bool:
        target = self.full_path
        return bool(target and target.exists())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "cap": self.cap,
            "params": jsonable(self.params), "inputs": list(self.inputs),
            "path": self.path, "meta": jsonable(self.meta), "label": self.label,
            "study": self.study, "created": self.created,
        }

    def one_line(self) -> str:
        bits = [f"{self.id}  {self.cap}"]
        if self.label:
            bits.append(self.label)
        rows = self.meta.get("nrows")
        if rows is not None:
            bits.append(f"{rows}r")
        return "  ".join(str(b) for b in bits)


def relative(path: Path | str) -> str:
    """Store paths repo-relative when we can, absolute when we can't."""
    target = Path(path)
    try:
        return target.resolve().relative_to(paths.REPO).as_posix()
    except (ValueError, OSError):
        return target.as_posix()
