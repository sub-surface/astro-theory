"""Archive pulls — ADQL against the registered TAP services."""
from __future__ import annotations

from .. import config
from ..core.capability import Param, capability

ARCHIVE_KEYS = tuple(config.ARCHIVES)


def _first_line(text: str, limit: int = 60) -> str:
    flat = " ".join(str(text).split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


@capability(
    name="archive.query", kind="table", wing="validation", cost="network",
    params={
        "archive": Param("enum", choices=ARCHIVE_KEYS, help="registered archive key"),
        "adql": Param("str", help="ADQL/SQL statement"),
    },
    summary="Run ADQL against a registered archive (cached by content).",
    tags=("archive",),
)
def archive_query(ctx, archive, adql):
    source = config.resolve_query_source(archive)
    if source is None:
        raise KeyError(f"unknown archive {archive!r}; "
                       f"have {', '.join(ARCHIVE_KEYS)}")
    ctx.progress(f"{archive}: submitting query")
    table = source.query(adql)
    ctx.check_cancel()
    ctx.progress(f"{archive}: {len(table)} rows")
    return ctx.table(table, label=f"{archive}: {_first_line(adql)}",
                     archive=archive, adql=" ".join(adql.split()))


@capability(
    name="archive.sample", kind="table", wing="validation", cost="network",
    params={"recipe": Param("enum", choices=tuple(config.SAMPLE_RECIPES))},
    summary="Run a named low-compute sample recipe.",
    tags=("archive",),
)
def archive_sample(ctx, recipe):
    spec = config.SAMPLE_RECIPES[recipe]
    ctx.progress(f"{recipe}: {spec.description}")
    child = ctx.child("archive.query", archive=spec.archive, adql=spec.adql)
    return ctx.table(ctx.load(child.id), label=f"sample {recipe}",
                     recipe=recipe, description=spec.description,
                     archive=spec.archive)


@capability(
    name="archive.health", kind="data", wing="validation", cost="network",
    params={},
    summary="Ping every archive's availability endpoint.",
)
def archive_health(ctx):
    from .. import doctor
    ctx.progress("pinging archive availability endpoints")
    rows = doctor.check_archives() if hasattr(doctor, "check_archives") else []
    if not rows:                       # doctor's shape varies; fall back to raw
        rows = [{"archive": key, "url": arch.health_url}
                for key, arch in config.ARCHIVES.items()]
    return ctx.data({"archives": rows}, label="archive health")
