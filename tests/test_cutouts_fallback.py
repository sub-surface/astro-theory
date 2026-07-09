"""Hermetic tests for the colour-cutout survey fallback (the Andromeda fix).

A HiPS can cover a declination yet have no data at a position (Legacy Surveys at
M31) and return a blank tile -> blank white image. color_auto walks the candidate
surveys until one has signal. We stub hips2fits so this stays network-free.
"""
import numpy as np

from celestrium import cutouts


def test_candidates_ordered_deepest_first_dss2_last():
    cands = cutouts.color_hips_candidates(41.27)        # M31 declination
    assert "DR10/color" in cands[0][0]                   # Legacy deepest, first
    assert any("DSS2" in label for _, label in cands)    # all-sky last resort present


def test_is_blank_detects_uniform_vs_signal():
    assert cutouts._is_blank(np.zeros((16, 16, 3)))      # blank tile
    assert cutouts._is_blank(np.full((16, 16, 3), 255))  # blown-out white
    gradient = np.tile(np.arange(256, dtype="float32"), (16, 1))
    assert not cutouts._is_blank(gradient)               # real structure


def test_color_auto_skips_blank_survey(tmp_path, monkeypatch):
    seen = []

    def fake_query(hips, **kw):
        seen.append(hips)
        if len(seen) == 1:
            return np.full((16, 16, 3), 255, dtype="uint8")  # Legacy: blank white
        return (np.arange(16 * 16 * 3).reshape(16, 16, 3) % 256).astype("uint8")

    monkeypatch.setattr(cutouts, "fetch_direct_cutout", lambda *a, **k: None)
    monkeypatch.setattr(cutouts.hips2fits, "query", fake_query)
    monkeypatch.setattr(cutouts.plt, "imsave", lambda *a, **k: None)  # no disk

    path, label = cutouts.color_auto(10.68, 41.27, fov_arcmin=30, pix=16,
                                     out=tmp_path / "m31.jpg")
    assert len(seen) >= 2          # the blank first survey was rejected
    assert "blank" not in label    # landed on a survey with signal


def test_color_auto_survey_override_tried_first(tmp_path, monkeypatch):
    seen = []

    def fake_query(hips, **kw):
        seen.append(hips)
        return (np.arange(16 * 16 * 3).reshape(16, 16, 3) % 256).astype("uint8")

    monkeypatch.setattr(cutouts, "fetch_direct_cutout", lambda *a, **k: None)
    monkeypatch.setattr(cutouts.hips2fits, "query", fake_query)
    monkeypatch.setattr(cutouts.plt, "imsave", lambda *a, **k: None)

    cutouts.color_auto(10.68, 41.27, survey="panstarrs", out=tmp_path / "x.jpg")
    assert "PanSTARRS" in seen[0]  # the forced survey is queried first


def test_color_auto_raises_when_all_sources_fail(tmp_path, monkeypatch):
    def fail_query(*args, **kwargs):
        raise TimeoutError("all sources slow")

    monkeypatch.setattr(cutouts, "fetch_direct_cutout", lambda *a, **k: None)
    monkeypatch.setattr(cutouts.hips2fits, "query", fail_query)

    out = tmp_path / "missing.jpg"
    try:
        cutouts.color_auto(10.68, 41.27, out=out)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected colour source failure")
    assert not out.exists()


def test_color_survey_metadata_matches_candidates():
    meta = cutouts.color_survey_metadata(41.27)
    keys = [m["key"] for m in meta]
    assert keys[0] == "legacy"
    assert "dss2" in keys
    assert all("coverage" in m for m in meta)


def test_color_survey_metadata_preserves_candidate_labels(monkeypatch):
    dec = 41.27
    monkeypatch.setattr(
        cutouts,
        "COLOR_FOOTPRINTS",
        [
            (
                -90,
                90,
                "CDS/P/DSS2/color",
                "DSS2 colour from candidate table",
            )
        ],
    )
    assert [m["label"] for m in cutouts.color_survey_metadata(dec)] == [
        label for _, label in cutouts.color_hips_candidates(dec)
    ]


def test_color_survey_metadata_defaults_for_unknown_candidate(monkeypatch):
    monkeypatch.setattr(
        cutouts,
        "COLOR_FOOTPRINTS",
        [(-90, 90, "CDS/P/Future/Color", "Future colour survey")],
    )

    meta = cutouts.color_survey_metadata(0.0)

    assert meta[0]["key"] == "CDS/P/Future/Color"
    assert meta[0]["label"] == "Future colour survey"
    assert meta[0]["wavelength"]
    assert meta[0]["coverage"]


def test_poster_falls_back_after_first_survey_timeout(tmp_path, monkeypatch):
    seen = []

    def fake_query(hips, **kw):
        seen.append(hips)
        if len(seen) == 1:
            raise TimeoutError("slow survey")
        return (np.arange(90 * 160 * 3).reshape(90, 160, 3) % 256).astype("uint8")

    monkeypatch.setattr(cutouts, "fetch_direct_cutout", lambda *a, **k: None)
    monkeypatch.setattr(cutouts.hips2fits, "query", fake_query)

    out = cutouts.poster(
        187.7059, 12.3911, fov_arcmin=8.0, width=160, height=90,
        out=tmp_path / "poster.jpg", label="M87", style="science",
    )

    assert out == tmp_path / "poster.jpg"
    assert len(seen) >= 2


def test_poster_uses_color_auto_when_high_res_sources_timeout(tmp_path, monkeypatch):
    def fail_query(*args, **kwargs):
        raise TimeoutError("all high-res sources slow")

    fallback_source = tmp_path / "fallback.jpg"

    def fake_color_auto(*args, **kwargs):
        cutouts.plt.imsave(fallback_source, np.zeros((32, 32, 3), dtype="uint8"))
        return fallback_source, "fallback"

    monkeypatch.setattr(cutouts, "fetch_direct_cutout", lambda *a, **k: None)
    monkeypatch.setattr(cutouts.hips2fits, "query", fail_query)
    monkeypatch.setattr(cutouts, "color_auto", fake_color_auto)

    out = cutouts.poster(
        187.7059, 12.3911, fov_arcmin=8.0, width=160, height=90,
        out=tmp_path / "poster.jpg", label="M87", style="label",
    )

    assert out == tmp_path / "poster.jpg"
