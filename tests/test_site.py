"""Static checks on the public site (site/): deploy config, the build-time sync, and the pages."""

from __future__ import annotations

import json
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


FIGS = REPO_ROOT / "docs" / "figures"
PAGES = sorted((SITE_DIR / "src").rglob("*.astro"))


def _fig_refs(src: str) -> list[str]:
    return re.findall(r'<Fig src="([^"]+)"', src)


def test_build_syncs_from_sources_of_truth():
    """Figures and listings are generated at build time (site/scripts/sync.mjs), never hand-copied."""
    site_pkg = json.loads((SITE_DIR / "package.json").read_text(encoding="utf-8"))
    assert site_pkg["scripts"]["prebuild"] == "node scripts/sync.mjs"
    assert (SITE_DIR / "scripts" / "sync.mjs").exists()
    # Cloudflare's git-triggered build runs the root `npm run build` on a fresh clone: it must install site deps.
    root_pkg = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert "npm --prefix site ci" in root_pkg["scripts"]["build"]
    assert "workspaces" not in root_pkg  # CLAUDE.md N2
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "site/public/fig/" in ignore and "site/src/generated/" in ignore


def test_every_page_figure_exists_in_docs_figures():
    refs = [f for p in PAGES for f in _fig_refs(p.read_text(encoding="utf-8"))]
    assert len(refs) > 20
    for f in refs:
        assert (FIGS / f).exists(), f
        if f.endswith(".gif"):  # animations are served as MP4 (scripts/infra/gif_to_mp4.py)
            assert (FIGS / f).with_suffix(".mp4").exists(), f


def test_ledger_figure_links_resolve():
    ledger = (REPO_ROOT / "docs" / "ledger.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(\./figures/([^)]+)\)", ledger)
    assert len(links) > 15
    for f in links:
        assert (FIGS / f).exists(), f


def test_sky_map_positions_are_galactic():
    """Hero map positions must come from a proper ICRS -> Galactic transform (CLAUDE.md I.1)."""
    from astropy.coordinates import SkyCoord

    sky = json.loads((SITE_DIR / "src" / "data" / "sky.json").read_text(encoding="utf-8"))
    assert len(sky["search"]) > 1000
    for o in sky["objects"]:
        g = SkyCoord(o["ra"], o["dec"], unit="deg", frame="icrs").galactic
        assert abs(g.l.deg - o["l"]) < 0.01 and abs(g.b.deg - o["b"]) < 0.01, o["name"]
    tcrb = next(o for o in sky["objects"] if o["name"] == "T CrB")
    assert abs(tcrb["l"] - 42.37) < 0.05 and abs(tcrb["b"] - 48.17) < 0.05


def test_inline_scripts_have_no_html_entities():
    for p in PAGES:
        for block in re.findall(r"<script is:inline>(.*?)</script>", p.read_text(encoding="utf-8"), flags=re.S):
            assert not re.search(r"&(gt|lt|amp);", block), f"{p.name}: HTML entities break inline scripts (N1)"


def test_citation_file():
    cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in cff and "astro-theory" in cff


def test_built_pages_if_present():
    dist = SITE_DIR / "dist"
    if not dist.exists():
        pytest.skip("site/dist not built")
    html = (dist / "index.html").read_text(encoding="utf-8")
    assert "Experiment ledger" in html and "T Coronae Borealis" in html
    assert (dist / "404.html").exists() and (dist / "feed.xml").exists() and (dist / "calendar.ics").exists()


def test_tcrb_page_figures_and_data_exist():
    src = (SITE_DIR / "src" / "pages" / "tcrb.astro").read_text(encoding="utf-8")
    assert len([f for f in _fig_refs(src) if f.startswith("tcrb/")]) >= 12
    data = json.loads((SITE_DIR / "src" / "data" / "tcrb.json").read_text(encoding="utf-8"))
    assert len(data["uvm2"]["t"]) == len(data["uvm2"]["m"]) > 100
    assert data["n"]["uv_latest"] - data["n"]["uv_last_bright"] > 1.5


def test_v347nor_page_figures_and_data_exist():
    src = (SITE_DIR / "src" / "pages" / "v347nor.astro").read_text(encoding="utf-8")
    assert len([f for f in _fig_refs(src) if f.startswith("rn/")]) >= 6
    data = json.loads((SITE_DIR / "src" / "data" / "v347nor.json").read_text(encoding="utf-8"))
    assert data["det"] and data["lim"] and data["cut"]
    assert "G3_onset_from_cutouts" in data["audit"] and data["stamps"]["n_control"] > 100
    assert "CANDIDATE" in src  # must stay labelled a candidate until the original plates are checked


def test_layout_theme_toggle_and_provenance_footer():
    src = (SITE_DIR / "src" / "layouts" / "Base.astro").read_text(encoding="utf-8")
    assert 'id="theme-btn"' in src and ':root[data-theme="dark"]' in src
    assert ':root:not([data-theme="light"])' in src  # OS dark mode must yield to an explicit light choice
    assert "https://subsurfaces.net" in src and "commitUrl" in src
    for block in re.findall(r"<script is:inline>(.*?)</script>", src, flags=re.S):
        assert not re.search(r"&(gt|lt|amp);", block), "HTML entities break inline scripts (CLAUDE.md N1)"
    info = (SITE_DIR / "src" / "build-info.ts").read_text(encoding="utf-8")
    assert "execFileSync" in info and "GIT_SHA" in info
