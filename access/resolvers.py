#!/usr/bin/env python
"""Object resolution & per-object bibliography — SIMBAD + NED.

Automates the EsaSky -> ADS "what is this object / what's been written about it"
lookup. SIMBAD for identity/type/bibliography, NED for extragalactic redshifts
and cross-IDs. See ../toolbox.md S2.
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
