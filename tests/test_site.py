"""Static checks on the public site (site/): deploy config and the one-page listing."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
INDEX = SITE_DIR / "src" / "pages" / "index.astro"


def test_wrangler_route_configuration():
    content = (REPO_ROOT / "wrangler.toml").read_text(encoding="utf-8")
    assert 'name = "astro"' in content
    assert 'pattern = "astro.subsurfaces.net"' in content
    assert 'directory = "./site/dist"' in content
    assert 'not_found_handling = "404-page"' in content


def test_every_listed_figure_exists():
    src = INDEX.read_text(encoding="utf-8")
    figs = re.findall(r"fig: '([^']+)'", src) + [f"tcrb/{f}" for f in re.findall(r"\['(tcrb_[^']+|srn_[^']+)'", src)]
    assert len(figs) > 20
    for f in figs:
        assert (SITE_DIR / "public" / "fig" / f).exists(), f


def test_ledger_statuses_match_docs_ledger():
    """The site must not show a status that differs from docs/ledger.md."""
    src = INDEX.read_text(encoding="utf-8")
    ledger = (REPO_ROOT / "docs" / "ledger.md").read_text(encoding="utf-8")
    for exp_id, status in re.findall(r"id: '(\w+)', title: '[^']*', data: '[^']*', status: '([^']+)'", src):
        row = re.search(rf"^\| \*\*EXP-2026-{exp_id}\*\* \|.*$", ledger, flags=re.M)
        assert row, exp_id
        assert status.split(" (")[0] in row.group(0), (exp_id, status)


def test_built_pages_if_present():
    dist = SITE_DIR / "dist"
    if not dist.exists():
        pytest.skip("site/dist not built")
    html = (dist / "index.html").read_text(encoding="utf-8")
    assert "Experiment ledger" in html and "T Coronae Borealis" in html
    assert (dist / "404.html").exists()
