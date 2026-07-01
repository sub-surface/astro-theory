"""Transient alerts (Supernovae, TDEs) via open streams."""

from astropy.table import Table

def fetch_latest_transients() -> Table | None:
    """Fetch the latest transient alerts (Supernovae, etc)."""
    # Deliberately returns no rows until a real TNS/ZTF-backed implementation is
    # wired in. The cockpit must not present fabricated alert data as live sky.
    return None
