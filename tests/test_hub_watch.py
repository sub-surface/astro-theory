"""CLI `watch` command — hermetic (fake packets.build_watch_packet)."""
import json

from typer.testing import CliRunner

from celestrium import hub, packets

runner = CliRunner()


def _pkt(kind="target", label="M87", total=1):
    return packets.WatchPacket(
        kind=kind, label=label, radius_arcmin=2.0, days=7.0,
        entries=[packets.WatchEntry(label, 187.7059, 12.3911, "hit", total,
                                    [{"main_id": "ZTF1"}] * total)])


def test_watch_target_mode(monkeypatch, tmp_path):
    seen = {}

    def fake_build(**kw):
        seen.update(kw)
        return _pkt()

    monkeypatch.setattr(packets, "build_watch_packet", fake_build)
    result = runner.invoke(hub.app, ["watch", "M87", "--radius", "3", "--days", "5"])
    assert result.exit_code == 0
    assert seen["target"] == "M87"
    assert seen["candidate_list"] is None
    assert seen["radius_arcmin"] == 3.0 and seen["days"] == 5.0
    assert "1 alert" in result.output


def test_watch_list_mode(monkeypatch):
    seen = {}

    def fake_build(**kw):
        seen.update(kw)
        return _pkt(kind="list", label="agn", total=3)

    monkeypatch.setattr(packets, "build_watch_packet", fake_build)
    result = runner.invoke(hub.app, ["watch", "--list", "agn"])
    assert result.exit_code == 0
    assert seen["candidate_list"] == "agn"
    assert seen["target"] is None


def test_watch_requires_exactly_one_source():
    result = runner.invoke(hub.app, ["watch"])
    assert result.exit_code == 1
    result2 = runner.invoke(hub.app, ["watch", "M87", "--list", "agn"])
    assert result2.exit_code == 1


def test_watch_json_payload(monkeypatch):
    monkeypatch.setattr(packets, "build_watch_packet", lambda **kw: _pkt())
    result = runner.invoke(hub.app, ["--json", "watch", "M87"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "target"
    assert payload["total_alerts"] == 1


def test_watch_report_flag_writes_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(packets, "build_watch_packet", lambda **kw: _pkt())
    monkeypatch.setattr(hub, "REPORTS_DIR", tmp_path)
    result = runner.invoke(hub.app, ["--json", "watch", "M87", "--report"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "report" in payload
    from pathlib import Path
    assert Path(payload["report"]).exists()


def test_watch_propagates_build_errors(monkeypatch):
    def boom(**kw):
        raise RuntimeError("could not resolve 'Nothing'")

    monkeypatch.setattr(packets, "build_watch_packet", boom)
    result = runner.invoke(hub.app, ["--json", "watch", "Nothing"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "could not resolve" in payload["message"]
