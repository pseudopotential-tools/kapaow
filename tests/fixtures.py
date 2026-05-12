"""Fixtures for use in the test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def data_path() -> Path:
    """Return the directory where data for use in tests are located."""
    return Path(__file__).parent / "data"
