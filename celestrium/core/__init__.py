"""Celestrium's execution kernel — the one spine every surface presents.

Five objects, and everything else in the instrument is built from them:

  Artifact    one typed, content-addressed record for everything produced
  Ledger      SQLite store of artifacts + lineage edges + runs + studies
  Capability  one declaration (params, kind, cost) per thing the tool can do
  Kernel      executor: dedupe → run → persist → record → emit events
  Context     what a capability body is handed (progress, children, writers)

The design rule that replaces the old "one brain, two surfaces" convention:
**a capability is a single object**, so no surface can hardcode a capability
list and drift from another. Parity is structural, not maintained.
"""
from . import capability                     # the module, not the decorator
from .artifact import Artifact, KINDS, artifact_id, canonical_text, jsonable
from .capability import CAPS, Capability, Param, all_caps, for_target, wings
from .events import Event
from .kernel import Context, Kernel, Payload
from .ledger import Ledger

# NOTE: `capability` above is deliberately the *submodule*. Re-exporting the
# decorator of the same name here would shadow it, so
# `from celestrium.core import capability as capmod` would silently hand you a
# function. Capability bodies import the decorator directly:
#     from ..core.capability import Param, capability
__all__ = [
    "capability", "Artifact", "KINDS", "artifact_id", "canonical_text",
    "jsonable", "CAPS", "Capability", "Param", "all_caps", "for_target",
    "wings", "Event", "Context", "Kernel", "Payload", "Ledger",
]
