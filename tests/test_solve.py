"""Test the solve module."""

from kapaow.solve import _find_matches


def test_find_matches() -> None:
    """Test the _find_matches function."""
    l_values = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    desired_l_values = [0, 0, 1, 2]
    assert _find_matches(l_values, desired_l_values) == [0, 1, 3, 6]
