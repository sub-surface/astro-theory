#!/usr/bin/env python
"""ADS / SciX — programmatic literature search + BibTeX export.

Talks to the ADS REST API directly (no extra package): search papers, export
canonical BibTeX, and append new entries to the repo-level refs.bib (deduped by
bibcode). This is our literature-census edge made scriptable. See ../toolbox.md S1.

Token (free): https://ui.adsabs.harvard.edu/user/settings/token
Provide it by EITHER:
  - env var:   export ADS_DEV_KEY=...        (PowerShell: $env:ADS_DEV_KEY=...)
  - or a file: ~/.ads/dev_key  (one line, just the token)
"""
import os
import re
from pathlib import Path

import requests

_API = "https://api.adsabs.harvard.edu/v1"
_REPO = Path(__file__).resolve().parent.parent
REFS_BIB = _REPO / "refs.bib"


def _token() -> str:
    tok = os.environ.get("ADS_DEV_KEY")
    if tok:
        return tok.strip()
    keyfile = Path.home() / ".ads" / "dev_key"
    if keyfile.exists():
        return keyfile.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "No ADS token. Get one (free) at "
        "https://ui.adsabs.harvard.edu/user/settings/token then either set "
        "$env:ADS_DEV_KEY or save it to ~/.ads/dev_key (one line)."
    )


def _headers():
    return {"Authorization": f"Bearer {_token()}"}


def search(query: str, rows: int = 10,
           fl=("bibcode", "title", "author", "year", "citation_count")):
    """ADS search -> list of dicts. query is ADS syntax, e.g.
    'abs:\"cosmic dipole\" year:2023-2026' or 'arxiv:2009.14826'."""
    r = requests.get(f"{_API}/search/query", headers=_headers(),
                     params={"q": query, "rows": rows, "fl": ",".join(fl),
                             "sort": "date desc"}, timeout=30)
    r.raise_for_status()
    return r.json()["response"]["docs"]


def bibtex(bibcodes) -> str:
    """Export canonical BibTeX for one or more bibcodes."""
    if isinstance(bibcodes, str):
        bibcodes = [bibcodes]
    r = requests.post(f"{_API}/export/bibtex", headers=_headers(),
                      json={"bibcode": list(bibcodes)}, timeout=30)
    r.raise_for_status()
    return r.json()["export"]


def _existing_keys(text: str):
    return set(re.findall(r"@\w+\{([^,]+),", text))


def add_to_refs(bibcodes, path: Path = REFS_BIB) -> int:
    """Append canonical BibTeX for bibcodes to refs.bib, skipping duplicates.
    Returns the number of new entries written."""
    new = bibtex(bibcodes)
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    have = _existing_keys(current)
    blocks = re.findall(r"@\w+\{[^@]*?\n\}", new, flags=re.S)
    added = [b for b in blocks if re.match(r"@\w+\{([^,]+),", b).group(1) not in have]
    if added:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n\n".join(added) + "\n")
    return len(added)


if __name__ == "__main__":
    try:
        docs = search('abs:"cosmic dipole" quasar year:2024-2026', rows=5)
        for d in docs:
            print(f"{d['bibcode']}  {d.get('title', ['?'])[0][:70]}")
    except Exception as e:
        print(e)
