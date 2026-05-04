"""Tests for :mod:`kapaow.bands`."""

from __future__ import annotations

from pathlib import Path

import pytest

from kapaow.bands import compute_min_nbnd, compute_num_target_bands, orbitals_per_atom
from kapaow.basis import AngularMomentum
from kapaow.extend import (
    BasisExtensionViaAddition,
    BasisExtensionViaChannel,
    BasisExtensionViaPolarization,
)


@pytest.fixture
def upf_dir() -> Path:
    """Tests-only PseudoDojo UPFs (H, Li, F, Si) shipped under tests/data/."""
    return Path(__file__).parent / "data" / "upfs"


@pytest.fixture
def cif_dir() -> Path:
    """Return the path to the example CIF structures shipped with the repo."""
    return Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# orbitals_per_atom
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("upf_name", "expected"),
    [("H.upf", 1), ("Li.upf", 2), ("F.upf", 4), ("Si.upf", 4)],
)
def test_orbitals_per_atom_bare(upf_dir: Path, upf_name: str, expected: int) -> None:
    """Without any extension, orbital count matches the UPF's PSWFC table."""
    assert orbitals_per_atom(upf_dir / upf_name) == expected


def test_orbitals_per_atom_subshell_addition_is_strictly_larger(upf_dir: Path) -> None:
    """Adding the next subshell strictly increases the orbital count."""
    bare = orbitals_per_atom(upf_dir / "Li.upf")
    extended = orbitals_per_atom(upf_dir / "Li.upf", BasisExtensionViaAddition())
    assert extended > bare


def test_orbitals_per_atom_subshell_doubling(upf_dir: Path) -> None:
    """Two-subshell extension adds strictly more than one-subshell extension."""
    one = orbitals_per_atom(upf_dir / "Si.upf", BasisExtensionViaAddition(increment=1))
    two = orbitals_per_atom(upf_dir / "Si.upf", BasisExtensionViaAddition(increment=2))
    assert two > one


def test_orbitals_per_atom_polarization_increases_count(upf_dir: Path) -> None:
    """Polarization mixes additions across all l channels, so adds at least 2l_max+1."""
    bare = orbitals_per_atom(upf_dir / "Si.upf")  # 4 (3s, 3p)
    polarized = orbitals_per_atom(upf_dir / "Si.upf", BasisExtensionViaPolarization())
    assert polarized > bare


@pytest.mark.parametrize(
    ("channel", "expected_increase"),
    [(AngularMomentum.P, 3), (AngularMomentum.D, 5)],
)
def test_orbitals_per_atom_channel_extension(
    upf_dir: Path, channel: AngularMomentum, expected_increase: int
) -> None:
    """Channel extension adds 2l+1 orbitals for the chosen channel."""
    bare = orbitals_per_atom(upf_dir / "H.upf")
    extended = orbitals_per_atom(upf_dir / "H.upf", BasisExtensionViaChannel(channel=channel))
    assert extended - bare == expected_increase


# ---------------------------------------------------------------------------
# compute_num_target_bands
# ---------------------------------------------------------------------------


def test_compute_num_target_bands_monatomic(cif_dir: Path) -> None:
    """A pure-Si structure: total = num_atoms * orbitals_per_Si."""
    n = compute_num_target_bands(cif_dir / "Si.cif", {"Si": 4})
    assert n == 8 * 4


def test_compute_num_target_bands_scales_with_orbitals(cif_dir: Path) -> None:
    """Doubling per-atom orbitals doubles the total."""
    n4 = compute_num_target_bands(cif_dir / "Si.cif", {"Si": 4})
    n8 = compute_num_target_bands(cif_dir / "Si.cif", {"Si": 8})
    assert n8 == 2 * n4


def test_compute_num_target_bands_diatomic(cif_dir: Path) -> None:
    """Multi-species structure sums per-element contributions."""
    n = compute_num_target_bands(cif_dir / "LiF_mp-1138_primitive.cif", {"Li": 2, "F": 4})
    assert n == 2 + 4


def test_compute_num_target_bands_missing_element_raises(cif_dir: Path) -> None:
    """A structure with an element absent from the dict raises KeyError."""
    with pytest.raises(KeyError):
        compute_num_target_bands(cif_dir / "LiF_mp-1138_primitive.cif", {"Li": 2})


def test_compute_num_target_bands_extra_keys_ignored(cif_dir: Path) -> None:
    """Unused entries in the dict are silently ignored."""
    n = compute_num_target_bands(cif_dir / "Si.cif", {"Si": 4, "C": 999, "Au": 999})
    assert n == 8 * 4


# ---------------------------------------------------------------------------
# compute_min_nbnd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_target", "expected"),
    [
        # The +4 floor dominates for small num_target...
        (0, 4),
        (1, 5),
        (4, 8),
        (7, 11),
        # ...and the 1.5x multiplier dominates from num_target=8 upwards.
        (8, 12),
        (10, 15),
        (100, 150),
        (1000, 1500),
    ],
)
def test_compute_min_nbnd_specific_values(num_target: int, expected: int) -> None:
    """Padding rule: max(int(1.5 * n), n + 4)."""
    assert compute_min_nbnd(num_target) == expected


def test_compute_min_nbnd_is_at_least_input() -> None:
    """The padded count never drops below the request."""
    for n in range(0, 200):
        assert compute_min_nbnd(n) >= n


def test_compute_min_nbnd_is_monotonic() -> None:
    """A larger target requires at least as many bands."""
    for n in range(0, 200):
        assert compute_min_nbnd(n) <= compute_min_nbnd(n + 1)


def test_compute_min_nbnd_padding_floor() -> None:
    """For small targets, the +4 floor wins; for large ones, the 1.5x multiplier wins."""
    # +4 dominates when n + 4 > 1.5 * n, i.e. n < 8.
    assert compute_min_nbnd(7) == 11  # 7 + 4
    # 1.5x dominates from n = 8 onwards.
    assert compute_min_nbnd(8) == 12  # int(1.5 * 8)
