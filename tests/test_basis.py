"""Test the basis module."""

import pytest

from pao_plusplus.basis import PseudoatomicBasis, AtomicBasis, AngularMomentum, Subshell


def test_ang_mtm():
    """Test the AngMtm enum."""
    assert AngularMomentum.S.value == 0
    assert AngularMomentum.P.value == 1
    assert AngularMomentum.D.value == 2
    assert AngularMomentum.F.value == 3
    assert AngularMomentum.G.value == 4
    assert AngularMomentum.P > AngularMomentum.S
    assert AngularMomentum.D >= AngularMomentum.P
    assert AngularMomentum.F < AngularMomentum.G
    assert AngularMomentum.S + 2 == AngularMomentum.D


def test_subshell():
    """Test the Subshell class."""
    subshell = Subshell(n=2, l=AngularMomentum.P)
    assert subshell.n == 2
    assert subshell.l == AngularMomentum.P

def test_subshell_coercion():
    """Test that Subshell l can be coerced from int."""
    subshell = Subshell(n=2, l=1)
    assert subshell.l == AngularMomentum.P

def test_subshell_n_greater_than_l():
    """Test that Subshell raises an error if n <= l."""
    with pytest.raises(ValueError):
        Subshell(n=1, l=AngularMomentum.P)

def test_pseudoatomic_basis():
    """Test the PseudoatomicBasis class."""
    basis = PseudoatomicBasis(number_of_orbitals={0: 2, 1: 1})
    assert basis.l_max == AngularMomentum.P
    assert basis.number_of_orbitals[AngularMomentum.S] == 2
    assert basis.l_values == [0, 0, 1]

@pytest.fixture
def atomic_basis() -> AtomicBasis:
    """Return an AtomicBasis for testing."""
    return AtomicBasis(subshells=[{'n': 2, 'l': 0}, {'n': 2, 'l': 1}])

def test_atomic_basis_creation(atomic_basis: AtomicBasis):
    """Test the AtomicBasis class."""
    assert atomic_basis.l_max == AngularMomentum.P
    assert atomic_basis.n_max == 2

def test_atomic_basis_extend(atomic_basis: AtomicBasis):
    """Test extending an AtomicBasis."""
    new_basis = atomic_basis.extend([Subshell(n=3, l=AngularMomentum.S)])
    assert len(new_basis) == len(atomic_basis) + 1
    assert new_basis.l_max == AngularMomentum.P
    assert new_basis.n_max == 3

def test_atomic_basis_to_pseudoatomic_basis(atomic_basis: AtomicBasis):
    """Test converting AtomicBasis to PseudoatomicBasis."""
    pseudo_basis = atomic_basis.to_pseudoatomic_basis()
    assert isinstance(pseudo_basis, PseudoatomicBasis)
    assert pseudo_basis.number_of_orbitals[AngularMomentum.S] == 1
    assert pseudo_basis.number_of_orbitals[AngularMomentum.P] == 1
    assert pseudo_basis.number_of_orbitals[AngularMomentum.D] == 0
    assert pseudo_basis.number_of_orbitals[AngularMomentum.F] == 0
    assert pseudo_basis.l_max == AngularMomentum.P
    assert pseudo_basis.n_max == 1  # note: 1, not 2, because we don't have n = 1 orbitals in the pseudo basis
