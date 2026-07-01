#!/usr/bin/env python
"""Object resolution & per-object bibliography — SIMBAD + NED.

Automates the EsaSky -> ADS "what is this object / what's been written about it"
lookup. SIMBAD for identity/type/bibliography, NED for extragalactic redshifts
and cross-IDs. See ../docs/toolbox.md S2.
Docs: https://astroquery.readthedocs.io/en/latest/simbad/simbad.html
      https://astroquery.readthedocs.io/en/latest/ipac/ned/ned.html
"""
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned


def identify(name: str):
    """SIMBAD: name -> coords, object type, basic data (astropy Table)."""
    s = Simbad()
    s.add_votable_fields("otype")  # request object type alongside defaults
    return s.query_object(name)


def search(name: str, limit: int = 8):
    """SIMBAD relaxed/alias lookup (wildcard) -> Table of candidates or None.

    The fuzzy companion to `identify`: when an exact name misses, a wildcard
    query surfaces near-spellings and catalogue aliases. Returns multiple rows
    for the planner to present as an ambiguity list."""
    s = Simbad()
    s.add_votable_fields("otype")
    s.ROW_LIMIT = limit
    try:
        return s.query_object(name, wildcard=True)
    except Exception:
        return None


def nearby(ra: float, dec: float, radius_arcmin: float = 3.0, limit: int = 8):
    """SIMBAD cone search around a position -> Table of nearby objects or None.

    Lets the planner attach an identity to a bare coordinate (and fall back to
    blank-field planning when nothing is catalogued here)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    s = Simbad()
    s.add_votable_fields("otype")
    s.ROW_LIMIT = limit
    try:
        return s.query_region(SkyCoord(ra, dec, unit=u.deg),
                              radius=radius_arcmin * u.arcmin)
    except Exception:
        return None


def bibliography(name: str, limit: int = 10):
    """SIMBAD: papers (bibcodes + titles) referencing an object, via TAP.

    Per-object bibliography moved to SIMBAD's TAP tables (ref/has_ref/ident);
    the old query_bibobj goes the *other* way (objects in a given paper).
    """
    q = (f"SELECT TOP {limit} bibcode, title "
         "FROM ref JOIN has_ref ON ref.oidbib = has_ref.oidbibref "
         "JOIN ident ON has_ref.oidref = ident.oidref "
         f"WHERE id = '{name}'")
    return Simbad.query_tap(q)


def extragalactic(name: str):
    """NED: redshift + cross-identifications for an extragalactic object."""
    return Ned.query_object(name)


_SOLAR_SYSTEM_IDS = {
    "sun": "10", "sol": "10",
    "mercury": "199", "venus": "299",
    "earth": "399", "moon": "301",
    "mars": "499", "jupiter": "599",
    "saturn": "699", "uranus": "799",
    "neptune": "899", "pluto": "999",
}

def dynamic(name: str):
    """JPL Horizons: resolve a solar system body to its current Ephemeris (RA/Dec).

    Returns a list of dicts (for ambiguity handling or exact match) or None.
    """
    try:
        from astroquery.jplhorizons import Horizons

        target_id = _SOLAR_SYSTEM_IDS.get(name.lower().strip(), name.strip())

        try:
            obj = Horizons(id=target_id)
            eph = obj.ephemerides()
        except ValueError as e:
            err = str(e)
            if "Ambiguous target name" in err:
                import re
                lines = err.split('\n')
                matches = []
                for line in lines:
                    m = re.match(r'^\s*(-?\d+)\s+([A-Za-z0-9 ]+?)\s', line)
                    if m:
                        # Return ambiguity list without coordinates (require selection)
                        matches.append({
                            "main_id": m.group(2).strip(),
                            "otype": "Solar System",
                            "ra": float("nan"),
                            "dec": float("nan"),
                            "jpl_id": m.group(1).strip()
                        })
                return matches if matches else None
            return None

        if eph and len(eph) > 0:
            row = eph[0]
            return [{
                "main_id": str(row["targetname"]).strip(),
                "otype": "Solar System",
                "ra": float(row["RA"]),
                "dec": float(row["DEC"]),
            }]
    except Exception:
        pass
    return None



if __name__ == "__main__":
    print("--- SIMBAD identify(M87) ---")
    print(identify("M87"))
    print("--- SIMBAD bibliography(M  87) ---")  # note SIMBAD's canonical spacing
    print(bibliography("M  87", 3)["bibcode"])
    print("--- NED extragalactic(3C 273) ---")  # the original quasar
    try:  # NED's server can be slow/unreachable; don't let it break the demo
        print(extragalactic("3C 273")["Object Name", "RA", "DEC", "Redshift"])
    except Exception as e:
        print(f"NED unavailable ({type(e).__name__}); retry later.")
