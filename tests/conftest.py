"""Pytest configuration and Windows path fix for Celestrium test suite."""
import os
import tempfile
from pathlib import Path
import pytest

# Fix Windows temp directory permission issues for pytest
@pytest.fixture(autouse=True, scope="session")
def configure_windows_temp(tmp_path_factory):
    """Ensure all temp dirs created by tests use a safe local path."""
    safe_temp = Path(__file__).parent.parent / "scratch" / "test_tmp"
    safe_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(safe_temp)
    os.environ["TEMP"] = str(safe_temp)
    os.environ["TMP"] = str(safe_temp)
    tempfile.tempdir = str(safe_temp)
