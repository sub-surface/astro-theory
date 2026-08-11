"""One event model, rendered three ways.

The kernel emits these; a TUI draws cards from them, the CLI draws progress
lines, an agent gets JSON. This is what removes the old per-result
`_accept_*` handlers: a view binds to a `run_id`, not to a shared slot, so a
slow result can never overwrite a fresh one — it just updates its own card.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

QUEUED = "queued"
STARTED = "started"
PROGRESS = "progress"
ARTIFACT = "artifact"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Event:
    type: str
    run_id: str
    cap: str
    message: str = ""
    artifact: Any = None          # core.Artifact on ARTIFACT/DONE
    cached: bool = False
    error: str = ""
    ms: int = 0
    study: Optional[str] = None
    attempt: int = 1

    def to_dict(self) -> dict:
        return {
            "type": self.type, "run_id": self.run_id, "cap": self.cap,
            "message": self.message, "cached": self.cached, "error": self.error,
            "ms": self.ms, "study": self.study, "attempt": self.attempt,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
        }


Listener = Callable[[Event], None]


def fan_out(*listeners: Optional[Listener]) -> Listener:
    """Combine listeners; a misbehaving one must never kill a run."""
    active = [fn for fn in listeners if fn is not None]

    def _emit(event: Event) -> None:
        for fn in active:
            try:
                fn(event)
            except Exception:
                pass

    return _emit
