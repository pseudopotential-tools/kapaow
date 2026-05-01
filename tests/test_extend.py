"""Test the extend module."""

import pytest

from kapaow.basis import AngularMomentum, AtomicBasis, PseudoatomicBasis
from kapaow.extend import BasisExtensionViaAddition, BasisExtensionViaPolarization


@pytest.mark.parametrize("increment", [1, 2, 3])
def test_basis_extension_via_addition(increment: int) -> None:
    """Test the BasisExtensionViaAddition class.

    This test should add 4s, then 3d, then 4p orbitals to the basis.
    """
    basis = AtomicBasis(subshells=[{"n": 3, "l": 0}, {"n": 3, "l": 1}])

    extension = BasisExtensionViaAddition(increment=increment)

    new_basis = extension.extend(basis)

    assert isinstance(new_basis, PseudoatomicBasis)
    assert new_basis.number_of_orbitals[AngularMomentum.S] == 2  # two s
    assert new_basis.number_of_orbitals[AngularMomentum.P] == 1 if increment < 3 else 2
    assert new_basis.number_of_orbitals[AngularMomentum.D] == 0 if increment < 2 else 1

    assert len(new_basis) == len(basis) + extension.increment


@pytest.mark.parametrize("increment", [1, 2, 3])
def test_basis_extension_via_polarization(increment: int) -> None:
    """Test the BasisExtensionViaPolarization class."""
    num_s = 1
    num_p = 1
    basis = PseudoatomicBasis(number_of_orbitals={0: num_s, 1: num_p})

    extension = BasisExtensionViaPolarization(increment=increment)

    new_basis = extension.extend(basis)

    assert isinstance(new_basis, PseudoatomicBasis)
    assert new_basis.number_of_orbitals[AngularMomentum.S] == num_s + increment
    assert new_basis.number_of_orbitals[AngularMomentum.P] == num_p + increment
    assert new_basis.number_of_orbitals[AngularMomentum.D] == increment
    assert new_basis.number_of_orbitals[AngularMomentum.F] == increment - 1
    assert new_basis.number_of_orbitals[AngularMomentum.G] == max(increment - 2, 0)
    assert len(new_basis) == (increment + 1) * (2 * len(basis) + increment) / 2
