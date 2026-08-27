import json
from astropy.table import Table
from typer.testing import CliRunner
import pytest

from celestrium import hub, resolvers, config
from celestrium.core.kernel import Kernel
from celestrium.core.artifact import Artifact

runner = CliRunner()

def _fake_target(**kw):
    base = dict(display_name="Demo", aliases=(), ra=10.0, dec=-5.0, otype="G",
                object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    base.update(kw)
    return resolvers.ResolvedTarget(**base)

def test_fetch_runs_named_product(monkeypatch):
    monkeypatch.setattr(resolvers, "resolve_target", lambda text: _fake_target())
    
    seen = {}
    def fake_run(self, key, *args, **kwargs):
        seen["key"] = key
        return Artifact(id="123", kind="table", cap=key, label="vizier")
    
    def fake_load(self, ref):
        return Table({"a": [1, 2]})

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)
    
    result = runner.invoke(hub.app, ["fetch", "Demo", "--product", "vizier"])
    assert result.exit_code == 0
    assert seen["key"] == "vizier"

def test_fetch_json_payload_serialises_table(monkeypatch):
    monkeypatch.setattr(resolvers, "resolve_target", lambda text: _fake_target())
    
    def fake_run(self, key, *args, **kwargs):
        return Artifact(id="123", kind="table", cap=key, label="vizier")
    
    def fake_load(self, ref):
        return Table({"a": [1]})

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)
    
    result = runner.invoke(hub.app, ["--json", "fetch", "Demo", "--product", "vizier"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "table"
    assert payload["id"] == "123"

def test_feed_runs_executor(monkeypatch):
    def fake_run(self, key, *args, **kwargs):
        return Artifact(id="123", kind="table", cap=key, label="neo")
    
    def fake_load(self, ref):
        return Table({"des": ["2026 AB"]})

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)
    
    result = runner.invoke(hub.app, ["feed", "neo", "--refresh"])
    assert result.exit_code == 0
    assert "1 rows" in result.output
