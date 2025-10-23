"""Fixtures for use in the test suite."""

import pytest
from pathlib import Path

@pytest.fixture
def data_path() -> Path:
    """Return the directory where data for use in tests are located."""
    return Path(__file__).parent / "data"
