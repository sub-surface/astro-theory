"""Live sky/event feeds — global, not target-scoped.

Generated from `registry.GLOBAL_FEED_EXECUTORS` so adding a broker is still a
one-line registry edit, but each feed lands in the registry as a real
capability with its own content address and cache entry.
"""
from __future__ import annotations

import importlib

from .. import registry
from ..core.capability import Param, capability

LABELS = {
    "neo": "JPL CNEOS close approaches",
    "satellite": "Celestrak visible satellites (TLEs)",
    "transient": "ALeRCE/ZTF transient alerts",
}


def _make(key: str, module_path: str, func_name: str):
    @capability(
        name=f"feed.{key}", kind="table", wing="validation", cost="network",
        params={"refresh_tag": Param("str", "", help="bump to force a fresh pull")},
        summary=LABELS.get(key, f"{key} feed"), tags=("feed",),
    )
    def _feed(ctx, refresh_tag, _module=module_path, _func=func_name, _key=key):
        module = importlib.import_module(_module)
        ctx.progress(f"{_key}: fetching")
        table = getattr(module, _func)()
        if table is None or len(table) == 0:
            raise LookupError(f"{_key} feed returned nothing")
        return ctx.table(table, label=LABELS.get(_key, _key), feed=_key)

    _feed.__name__ = f"feed_{key}"
    return _feed


for _key, (_module, _func) in registry.GLOBAL_FEED_EXECUTORS.items():
    _make(_key, _module, _func)


@capability(
    name="feed.near", kind="table", wing="validation", cost="network",
    params={"target": Param("str"), "radius_arcmin": Param("float", 2.0),
            "days": Param("float", 7.0)},
    summary="Transient alerts in a cone around a target.",
    tags=("target", "feed"),
)
def feed_near(ctx, target, radius_arcmin, days):
    from .. import transients
    from .objects import resolve
    found = resolve(ctx, target)
    table = transients.fetch_transients_near(found.ra, found.dec,
                                             radius_arcmin=radius_arcmin,
                                             days=days)
    if table is None or len(table) == 0:
        raise LookupError(f"no alerts within {radius_arcmin}' of "
                          f"{found.display_name} in {days} days")
    return ctx.table(table, label=f"alerts near {found.display_name}")
