import json

from astropy.table import Table
from typer.testing import CliRunner

from celestrium import census, hub, packets, planner, plots, products, tables

runner = CliRunner()


def _target():
    return planner.ResolvedTarget(
        display_name="Demo", aliases=(), ra=10.0, dec=-5.0, otype="G",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")


def test_plot_table_writes_png(tmp_path):
    tab = Table({"ra": [10.0, 20.0], "dec": [-1.0, 1.0], "mag": [15.0, 16.0]})
    out = plots.plot_table(tab, "ra", "dec", out=tmp_path / "plot.png")
    assert out.exists()
    assert out.read_bytes().startswith(b"\x89PNG")


def test_census_tally_uses_injected_fetcher(monkeypatch):
    monkeypatch.setattr(census, "_sleep", lambda seconds: None)
    counts = iter([10, 15, 4, 2])

    def fetcher(url):
        return (f"<feed><opensearch:totalResults>{next(counts)}</opensearch:totalResults>"
                "</feed>")

    tab = census.tally({"topic-a": "all:a", "topic-b": "all:b"},
                       years=[2024, 2025], fetcher=fetcher)
    assert list(tab["topic"]) == ["topic-a", "topic-b"]
    assert list(tab["2024"]) == [10, 4]
    assert list(tab["2025"]) == [15, 2]
    assert list(tab["trend"]) == ["+50%", "-50%"]


def test_sweep_packet_records_hits_empties_and_errors():
    target = _target()
    seen_settings = {}

    def execute(tgt, plan, settings, emit):
        seen_settings[plan.product.key] = settings
        if plan.product.key == "vizier":
            return products.ProductResult(
                "table", plan.product.label, tgt.display_name,
                table=Table({"ra": [1.0, 2.0]}), summary="demo rows")
        if plan.product.key == "heasarc":
            return products.ProductResult(
                "table", plan.product.label, tgt.display_name, table=Table({"ra": []}))
        raise RuntimeError("offline")

    pkt = packets.build_sweep_packet(
        target, execute=execute, keys=("vizier", "heasarc", "mast"))
    statuses = {entry["key"]: entry["status"] for entry in pkt.entries}
    assert statuses == {"vizier": "hit", "heasarc": "empty", "mast": "error"}
    assert all(settings["table_only"] is True for settings in seen_settings.values())
    assert pkt.hits == 1
    assert "Sweep: Demo" in pkt.to_markdown()


def test_cli_plot_uses_table_ref_and_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tables, "load_table_ref",
        lambda ref, archives=None: (Table({"ra": [1.0], "dec": [2.0]}), "fake ref"))
    monkeypatch.setattr(
        plots, "plot_table",
        lambda tab, x=None, y=None, kind="scatter", title="", out=None: tmp_path / "p.png")
    result = runner.invoke(hub.app, ["--json", "plot", "demo", "ra", "dec"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["path"].endswith("p.png")
    assert payload["source"] == "fake ref"


def test_cli_sweep_renders_packet(monkeypatch):
    pkt = packets.SweepPacket(
        target={"display_name": "Demo", "ra": 1.0, "dec": 2.0, "object_class": "unknown"},
        entries=[{"key": "vizier", "label": "VizieR", "status": "hit",
                  "rows": 2, "kind": "table", "note": "ok"}],
    )
    monkeypatch.setattr(packets, "build_sweep_packet", lambda target, on_note=None: pkt)
    result = runner.invoke(hub.app, ["sweep", "Demo"])
    assert result.exit_code == 0
    assert "VizieR" in result.output
    assert "1/1" in result.output


def test_cli_census_writes_csv_and_optional_report(tmp_path, monkeypatch):
    tab = Table({"topic": ["demo"], "2025": [3], "trend": [""]})
    monkeypatch.setattr(census, "tally", lambda on_note=None: tab)
    monkeypatch.setattr(census, "write_csv", lambda tab, out=None: out or tmp_path / "census.csv")
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    result = runner.invoke(hub.app, ["census", "--report"])
    assert result.exit_code == 0
    assert "census-topic-trajectories.md" in result.output
