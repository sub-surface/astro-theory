"""`@capability` — one declaration per thing the instrument can do.

This is the consolidation that matters. Before, a single product was declared
in `resolvers.SOURCE_CAPABILITIES`, implemented in `products.PRODUCT_EXECUTORS`,
routed in `tui/commands.COMMANDS`, dispatched in `app._handle_command`, and
exposed again as a hand-written `hub.py` command. Now it is one object, and
every surface *enumerates the registry* instead of hardcoding a list:

    CLI commands · TUI palette + autocomplete · planner ranking · JSON/agent API

so a capability cannot exist on one surface and be missing from another.

A `Param` of kind ``artifact`` is special: its value is an artifact id, and the
kernel folds it into that run's `inputs` automatically — which is how lineage
gets recorded without any capability having to think about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

MISSING = object()

# Cost bands — the honest "what will this spend" axis. `network` runs get
# retries; `heavy` ones are the reason study grids print an estimate first.
COSTS = ("free", "network", "heavy")
WINGS = ("theory", "validation", "imaging", "core")


class ParamError(ValueError):
    """Raised when a caller's parameters don't satisfy a capability."""


@dataclass(frozen=True)
class Param:
    kind: str = "str"            # str|int|float|bool|enum|json|artifact
    default: Any = MISSING
    choices: tuple = ()
    help: str = ""

    @property
    def required(self) -> bool:
        return self.default is MISSING

    def coerce(self, value: Any, name: str = "") -> Any:
        if value is None:
            if self.required:
                raise ParamError(f"{name}: required")
            return self.default
        try:
            if self.kind == "int":
                return int(value)
            if self.kind == "float":
                return float(value)
            if self.kind == "bool":
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
            if self.kind == "enum":
                text = str(value)
                if self.choices and text not in self.choices:
                    raise ParamError(
                        f"{name}: {text!r} not in {', '.join(map(str, self.choices))}")
                return text
            if self.kind == "json":
                if isinstance(value, str):
                    import json
                    return json.loads(value)
                return value
            return str(value)          # str and artifact both store as text
        except ParamError:
            raise
        except (TypeError, ValueError) as exc:
            raise ParamError(f"{name}: expected {self.kind} ({exc})") from exc

    def describe(self) -> str:
        bits = [self.kind]
        if self.choices:
            bits.append("{" + "|".join(map(str, self.choices)) + "}")
        if not self.required:
            bits.append(f"={self.default!r}")
        else:
            bits.append("required")
        return " ".join(bits)


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    fn: Callable
    params: dict = field(default_factory=dict)
    summary: str = ""
    wing: str = "core"
    cost: str = "free"
    applies_to: Any = None       # None/"*" = any target; tuple of classes; or callable
    tags: tuple = ()

    def bind(self, raw: Optional[dict] = None) -> dict:
        """Validate + coerce + fill defaults. Unknown keys are an error, not a
        silent no-op — a typo'd parameter that changes nothing is the worst
        possible outcome for a content-addressed cache."""
        raw = dict(raw or {})
        unknown = set(raw) - set(self.params)
        if unknown:
            raise ParamError(
                f"{self.name}: unknown parameter(s) {', '.join(sorted(unknown))}; "
                f"accepts {', '.join(self.params) or '(none)'}")
        bound = {}
        for key, spec in self.params.items():
            value = raw.get(key)
            if value is None and not spec.required:
                bound[key] = spec.default
                continue
            bound[key] = spec.coerce(value, f"{self.name}.{key}")
        return bound

    @property
    def artifact_params(self) -> tuple:
        return tuple(k for k, p in self.params.items() if p.kind == "artifact")

    def matches(self, target: Any) -> bool:
        """Does this capability apply to a resolved target? (planner ranking)"""
        if self.applies_to in (None, "*"):
            return True
        if callable(self.applies_to):
            try:
                return bool(self.applies_to(target))
            except Exception:
                return False
        object_class = getattr(target, "object_class", None) or "unknown"
        return object_class in tuple(self.applies_to)

    def usage(self) -> str:
        bits = [self.name]
        for key, spec in self.params.items():
            bits.append(f"<{key}>" if spec.required else f"[{key}]")
        return " ".join(bits)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "wing": self.wing,
            "cost": self.cost, "summary": self.summary, "tags": list(self.tags),
            "params": {k: {"kind": p.kind, "required": p.required,
                           "default": None if p.required else p.default,
                           "choices": list(p.choices), "help": p.help}
                       for k, p in self.params.items()},
        }


CAPS: dict = {}


def capability(*, name: str, kind: str, params: Optional[dict] = None,
               summary: str = "", wing: str = "core", cost: str = "free",
               applies_to: Any = None, tags: tuple = ()) -> Callable:
    """Register a capability. The body is `fn(ctx, **params) -> Payload`."""
    def decorate(fn: Callable) -> Callable:
        cap = Capability(name=name, kind=kind, fn=fn, params=params or {},
                         summary=summary or (fn.__doc__ or "").strip().split("\n")[0],
                         wing=wing, cost=cost, applies_to=applies_to, tags=tags)
        CAPS[name] = cap
        fn.capability = cap
        return fn
    return decorate


def get(name: str) -> Capability:
    cap = CAPS.get(name)
    if cap is None:
        near = [k for k in sorted(CAPS) if name.split(".")[0] in k][:6]
        hint = f"; did you mean {', '.join(near)}?" if near else ""
        raise KeyError(f"unknown capability {name!r}{hint}")
    return cap


def all_caps(wing: Optional[str] = None, kind: Optional[str] = None) -> list:
    out = sorted(CAPS.values(), key=lambda c: (c.wing, c.name))
    if wing:
        out = [c for c in out if c.wing == wing]
    if kind:
        out = [c for c in out if c.kind == kind]
    return out


def for_target(target: Any, *, cost: Optional[str] = None) -> list:
    """Capabilities that apply to a resolved target — replaces the hand-kept
    `resolvers.SOURCE_CAPABILITIES` ranking table."""
    out = [c for c in all_caps() if "target" in c.tags and c.matches(target)]
    if cost:
        out = [c for c in out if c.cost == cost]
    rank = {"free": 0, "network": 1, "heavy": 2}
    return sorted(out, key=lambda c: (rank.get(c.cost, 3), c.name))


def wings() -> dict:
    grouped: dict = {}
    for cap in all_caps():
        grouped.setdefault(cap.wing, []).append(cap)
    return grouped


def load_all() -> int:
    """Import every capability module so the registry is populated. Idempotent."""
    from .. import caps  # noqa: F401  (its __init__ imports each module)
    return len(CAPS)
