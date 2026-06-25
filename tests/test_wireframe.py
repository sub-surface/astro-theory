"""Hermetic tests for the wireframe engine (pure geometry — no Textual, no I/O)."""
from celestrium.tui import wireframe as wf


def test_five_platonic_solids_well_formed():
    assert set(wf.SOLIDS) == {"tetrahedron", "cube", "octahedron",
                              "icosahedron", "dodecahedron"}
    for name, solid in wf.SOLIDS.items():
        assert len(solid.verts) >= 4, name
        assert len(solid.edges) >= 6, name
        n = len(solid.verts)
        for i, j in solid.edges:  # edges index valid vertices
            assert 0 <= i < n and 0 <= j < n, name


def test_braille_canvas_set_and_line():
    c = wf.BrailleCanvas(10, 6)
    assert c.to_text().strip("⠀\n") == ""   # starts empty (all blank braille)
    c.line(0, 0, 19, 23)                          # a diagonal lights dots
    txt = c.to_text()
    assert any(0x2801 <= ord(ch) <= 0x28ff for ch in txt)


def test_render_solid_is_deterministic_per_frame():
    c1 = wf.BrailleCanvas(24, 12)
    c2 = wf.BrailleCanvas(24, 12)
    wf.render_solid(c1, wf.SOLIDS["icosahedron"], angle=0.84)
    wf.render_solid(c2, wf.SOLIDS["icosahedron"], angle=0.84)
    assert c1.to_text() == c2.to_text()          # frame-counter driven, repeatable
    # a different angle draws a different frame
    c3 = wf.BrailleCanvas(24, 12)
    wf.render_solid(c3, wf.SOLIDS["icosahedron"], angle=1.7)
    assert c3.to_text() != c1.to_text()


def test_solid_order_covers_all():
    assert set(wf.SOLID_ORDER) == set(wf.SOLIDS)
