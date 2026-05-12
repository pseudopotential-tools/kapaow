"""Tests for the pareto module (pure functions only)."""

import json
from pathlib import Path

import numpy as np

from kapaow.pareto import (
    dump_pareto_json,
    extract_pareto_front,
    find_kink_triplets,
)

# ---------------------------------------------------------------------------
# extract_pareto_front
# ---------------------------------------------------------------------------


def test_pareto_single_point() -> None:
    """A single point is always on the Pareto front."""
    assert extract_pareto_front([1.0], [1.0]) == [0]


def test_pareto_dominated_point() -> None:
    """A point dominated by another should not be on the front."""
    spreads = [1.0, 2.0]
    shifts = [1.0, 2.0]
    assert extract_pareto_front(spreads, shifts) == [0]


def test_pareto_tradeoff() -> None:
    """Two points with a trade-off should both be on the front."""
    spreads = [1.0, 3.0]
    shifts = [3.0, 1.0]
    result = extract_pareto_front(spreads, shifts)
    assert sorted(result) == [0, 1]


def test_pareto_three_points_one_dominated() -> None:
    """Three non-dominated points should all be on the front."""
    spreads = [1.0, 2.0, 3.0]
    shifts = [3.0, 2.0, 1.0]
    result = extract_pareto_front(spreads, shifts)
    assert sorted(result) == [0, 1, 2]


def test_pareto_interior_point() -> None:
    """Interior point should be excluded."""
    spreads = [1.0, 2.0, 3.0]
    shifts = [3.0, 3.5, 1.0]
    result = extract_pareto_front(spreads, shifts)
    assert 1 not in result
    assert sorted(result) == [0, 2]


def test_pareto_identical_points() -> None:
    """Identical points should all be on the front."""
    spreads = [1.0, 1.0, 1.0]
    shifts = [1.0, 1.0, 1.0]
    result = extract_pareto_front(spreads, shifts)
    assert sorted(result) == [0, 1, 2]


def test_pareto_empty() -> None:
    """Empty input should return empty."""
    assert extract_pareto_front([], []) == []


# ---------------------------------------------------------------------------
# find_kink_triplets
# ---------------------------------------------------------------------------


def test_kink_triplets_too_few_pareto_points() -> None:
    """Fewer than 3 Pareto points should return no triplets."""
    spreads = [1.0, 2.0]
    shifts = [2.0, 1.0]
    assert find_kink_triplets(spreads, shifts) == []


def test_kink_triplets_straight_line() -> None:
    """Points forming a straight line should have no kinks."""
    spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
    shifts = [5.0, 4.0, 3.0, 2.0, 1.0]
    triplets = find_kink_triplets(spreads, shifts, threshold=np.radians(1))
    assert triplets == []


def test_kink_triplets_sharp_corner() -> None:
    """A sharp L-shaped Pareto front should have a kink at the corner.

    The front bows outward (away from origin) at the corner.
    For find_kink_triplets, this means a right turn detected via negative cross product.
    The Pareto front goes from high-shift/low-spread to low-shift/high-spread.
    We need the corner to bow outward (below and to the right of the straight line).
    """
    # Pareto front with an outward-bowing kink (right turn):
    # sorted by spread: (1, 10), (9, 5), (10, 1)
    # The front traverses most of the spread range quickly, then drops sharply.
    # ab=(8,-5), bc=(1,-4), cross=8*(-4)-(-5)*1=-27 < 0 (right turn)
    spreads = [1.0, 9.0, 10.0]
    shifts = [10.0, 5.0, 1.0]
    triplets = find_kink_triplets(
        spreads,
        shifts,
        threshold=np.radians(10),
        gap_factor=0.0,
    )
    assert len(triplets) == 1
    assert triplets[0][1] == 1  # middle point is the kink


# ---------------------------------------------------------------------------
# dump_pareto_json
# ---------------------------------------------------------------------------


def test_dump_pareto_json(tmp_path: Path) -> None:
    """dump_pareto_json should write valid JSON with pareto flags."""
    spreads = [1.0, 2.0, 3.0]
    shifts = [3.0, 2.5, 1.0]
    metadata = [
        {"rc": 5.0, "ri_factor": 0.8, "modified_by_confinement": False},
        {"rc": 7.0, "ri_factor": 0.7, "modified_by_confinement": True},
        {"rc": 10.0, "ri_factor": 0.9, "modified_by_confinement": False},
    ]
    path = tmp_path / "pareto.json"
    dump_pareto_json(spreads, shifts, metadata, path)

    data = json.loads(path.read_text())
    assert "points" in data
    assert len(data["points"]) == 3
    assert all("pareto" in p for p in data["points"])
    assert any(p["pareto"] for p in data["points"])


def test_dump_pareto_json_with_upf_path(tmp_path: Path) -> None:
    """dump_pareto_json should include upf_path when provided."""
    spreads = [1.0]
    shifts = [1.0]
    metadata = [{"rc": 5.0, "ri_factor": 0.8, "modified_by_confinement": False}]
    path = tmp_path / "pareto.json"
    dump_pareto_json(spreads, shifts, metadata, path, upf_path=Path("/some/file.upf"))

    data = json.loads(path.read_text())
    assert data["upf_path"] == "/some/file.upf"


def test_dump_pareto_json_fields(tmp_path: Path) -> None:
    """Each point should have the expected fields."""
    spreads = [2.0]
    shifts = [0.5]
    metadata = [{"rc": 8.0, "ri_factor": 0.85, "modified_by_confinement": True}]
    path = tmp_path / "test.json"
    dump_pareto_json(spreads, shifts, metadata, path)

    data = json.loads(path.read_text())
    point = data["points"][0]
    assert point["rc"] == 8.0
    assert point["ri_factor"] == 0.85
    assert point["spread"] == 2.0
    assert point["max_energy_shift"] == 0.5
    assert point["modified_by_confinement"] is True
    assert "pareto" in point
