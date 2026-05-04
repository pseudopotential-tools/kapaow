"""Test the solve module."""

import numpy as np
import pytest

from kapaow.basis import AngularMomentum, PseudoatomicBasis
from kapaow.solve import OrbitalEnergy, _find_matches, _write_dat_for_basis, read_femdvr_eigenvalues


def test_find_matches() -> None:
    """Test the _find_matches function."""
    l_values = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    desired_l_values = [0, 0, 1, 2]
    assert _find_matches(l_values, desired_l_values) == [0, 1, 3, 6]


# ---------------------------------------------------------------------------
# OrbitalEnergy
# ---------------------------------------------------------------------------


def test_orbital_energy_construction() -> None:
    """OrbitalEnergy should hold l, n_radial, energy correctly."""
    orb = OrbitalEnergy(l=AngularMomentum.P, n_radial=1, energy=-0.5)
    assert orb.l is AngularMomentum.P
    assert orb.n_radial == 1
    assert orb.energy == pytest.approx(-0.5)


def test_orbital_energy_invalid_l() -> None:
    """OrbitalEnergy should reject an invalid angular momentum value."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OrbitalEnergy(l=99, n_radial=0, energy=-1.0)


# ---------------------------------------------------------------------------
# read_femdvr_eigenvalues
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal stand-in for PseudoAtomicResult."""

    def __init__(self, eigenvalues: dict) -> None:
        self.eigenvalues = eigenvalues


def test_read_femdvr_eigenvalues_nscf_preferred() -> None:
    """Nscf task is used when present."""
    fake = _FakeResult(
        {
            "scf": {"0": [-2.0, -0.5], "1": [-1.0]},
            "nscf": {"0": [-1.9, -0.4], "1": [-0.9]},
        }
    )
    orbs = read_femdvr_eigenvalues(fake)
    # nscf values should appear, not scf
    s_orbs = [o for o in orbs if o.l == AngularMomentum.S]
    assert len(s_orbs) == 2
    np.testing.assert_allclose([o.energy for o in s_orbs], [-1.9, -0.4], rtol=1e-12)


def test_read_femdvr_eigenvalues_fallback_to_scf() -> None:
    """Scf task is used when nscf is absent."""
    fake = _FakeResult({"scf": {"0": [-2.0, -0.5], "1": [-1.0]}})
    orbs = read_femdvr_eigenvalues(fake)
    s_orbs = [o for o in orbs if o.l == AngularMomentum.S]
    p_orbs = [o for o in orbs if o.l == AngularMomentum.P]
    assert len(s_orbs) == 2
    assert len(p_orbs) == 1
    # n_radial is 0-based within channel
    assert s_orbs[0].n_radial == 0
    assert s_orbs[1].n_radial == 1
    assert p_orbs[0].n_radial == 0


def test_read_femdvr_eigenvalues_numerical_values() -> None:
    """Regression: energies should be reproduced exactly from the dict."""
    energies_s = [-2.376777, -0.160385]
    energies_p = [-1.434869]
    fake = _FakeResult({"scf": {"0": energies_s, "1": energies_p}})
    orbs = read_femdvr_eigenvalues(fake)
    s_orbs = [o for o in orbs if o.l == AngularMomentum.S]
    p_orbs = [o for o in orbs if o.l == AngularMomentum.P]
    # rtol 1e-10: values round-trip through float, no arithmetic, exact match expected
    np.testing.assert_allclose([o.energy for o in s_orbs], energies_s, rtol=1e-10)
    np.testing.assert_allclose([o.energy for o in p_orbs], energies_p, rtol=1e-10)


# ---------------------------------------------------------------------------
# _write_dat_for_basis
# ---------------------------------------------------------------------------


def test_write_dat_for_basis(tmp_path: pytest.TempPathFactory, data_path) -> None:
    """_write_dat_for_basis should produce a dat file with the correct l_values."""
    from kapaow.io import read_wannier90_dat_file

    # Copy Mo.dat and rename it so it matches the *_qe.dat glob
    src = data_path / "dat_files" / "Mo.dat"
    qe_dat = tmp_path / "Mo_qe.dat"
    qe_dat.write_bytes(src.read_bytes())

    # Mo basis has s, p, d — request just s+d
    basis = PseudoatomicBasis(
        number_of_orbitals={
            AngularMomentum.S: 1,
            AngularMomentum.P: 0,
            AngularMomentum.D: 1,
        }
    )
    out = tmp_path / "filtered.dat"
    _write_dat_for_basis(tmp_path, out, basis)

    _, _, l_values, orbs = read_wannier90_dat_file(out)
    assert l_values == [0, 2]
    assert orbs.shape[0] == 2
