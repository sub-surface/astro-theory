"""Network hygiene — one place for default timeouts.

Most astroquery clients ship with long (or no) effective timeouts; a stalled
archive then hangs a CLI command or silently eats a TUI worker thread. Every
module that imports an astroquery client calls `set_timeout(Client)` right
after the import, so the policy lives here and the modules stay one-liners.

`urllib`/`requests` call sites pass timeouts explicitly (grep for TIMEOUT).
"""
DEFAULT_TIMEOUT = 30   # seconds — archives that answer at all answer well within this
SLOW_TIMEOUT = 60      # for services the docs already flag as slow (NED)


def set_timeout(client, seconds: int = DEFAULT_TIMEOUT) -> None:
    """Set an astroquery-style class/instance TIMEOUT if the client has one."""
    try:
        if hasattr(client, "TIMEOUT"):
            client.TIMEOUT = seconds
        elif hasattr(client, "timeout"):
            client.timeout = seconds
    except Exception:
        # Some astroquery property setters inspect live TAP capabilities. Timeout
        # policy must never make imports network-dependent; call sites still pass
        # explicit timeouts where the underlying API supports them.
        pass
