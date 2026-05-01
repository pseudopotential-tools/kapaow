"""Unit tests for :mod:`kapaow._experimental.symmetrize`.

These tests exercise the numerical pieces (Wigner-D, Serre decomposition,
bond-oriented hybridization). The end-to-end hBN test monkey-patches
the per-species lmax reader so no .dat file is needed.

The module under test lives in :mod:`kapaow._experimental`, whose
import-time guard requires the ``[experimental]`` extras. Without
those extras the entire test file is skipped.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "kapaow._experimental",
    reason="kapaow._experimental requires the [experimental] extras",
)

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from kapaow._experimental import symmetrize as sym

# ---------------------------------------------------------------------------
# Wigner-D
# ---------------------------------------------------------------------------


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) * c + (1 - c) * np.outer(axis, axis) + s * K


def test_wigner_d_l0_is_identity():
    """D^0(R) is the 1x1 identity for any rotation R."""
    R = _rot_axis_angle(np.array([1.0, 2.0, 3.0]), 1.2)
    D = sym._wigner_d_complex(0, R)
    assert D.shape == (1, 1)
    assert np.isclose(D[0, 0], 1.0)


def test_wigner_d_unitary():
    """D^l(R) is unitary for random proper rotations."""
    rng = np.random.default_rng(1)
    for l in (1, 2, 3):
        for _ in range(5):
            axis = rng.standard_normal(3)
            angle = rng.uniform(-np.pi, np.pi)
            D = sym._wigner_d_complex(l, _rot_axis_angle(axis, angle))
            prod = D @ D.conj().T
            assert np.allclose(prod, np.eye(2 * l + 1), atol=1e-10)


def test_wigner_d_composition():
    """D^l(R1 R2) = D^l(R1) @ D^l(R2) (group homomorphism)."""
    rng = np.random.default_rng(2)
    R1 = _rot_axis_angle(rng.standard_normal(3), 0.7)
    R2 = _rot_axis_angle(rng.standard_normal(3), -1.3)
    for l in (1, 2):
        D12 = sym._wigner_d_complex(l, R1 @ R2)
        Dprod = sym._wigner_d_complex(l, R1) @ sym._wigner_d_complex(l, R2)
        assert np.allclose(D12, Dprod, atol=1e-10)


def test_wigner_d_improper_inversion():
    """Inversion -I: D^l = (-1)^l * I."""
    inv = -np.eye(3)
    for l in (0, 1, 2):
        D = sym._wigner_d_complex(l, inv)
        assert np.allclose(D, ((-1) ** l) * np.eye(2 * l + 1), atol=1e-10)


def test_wigner_d_p_z_rotation():
    """pi/2 rotation about z fixes pz (m=0 diagonal entry equals 1)."""
    # A rotation of pi/2 about z mixes (px, py) and leaves pz untouched.
    # In the m = -1, 0, +1 basis, pz corresponds to m=0 only.
    R = _rot_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    D1 = sym._wigner_d_complex(1, R)
    # The m=0 diagonal entry should be exactly 1 (pz invariant under z-axis rotations).
    assert np.isclose(D1[1, 1], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Serre decomposition
# ---------------------------------------------------------------------------


def test_serre_trivial_group():
    """Identity-only group: every column is its own isotypic block."""
    D = [np.eye(4, dtype=np.complex128)]
    U, tags = sym._serre_isotypic_basis(D)
    assert U.shape == (4, 4)
    assert len(tags) == 4


def test_serre_c3v_block_structure():
    """C3v on p-rep gives two isotypic blocks: E (px, py) and A_1 (pz)."""
    # C_3v on the p rep: E (px, py) + A_1 (pz) -> blocks of size 2 + 1.
    # Requires a non-abelian group so that (px, py) genuinely pair.
    rotations = []
    for angle in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        rotations.append(_rot_axis_angle(np.array([0.0, 0.0, 1.0]), angle))
    # Three vertical mirror planes (det = -1, built as improper rotations).
    for phi in (0.0, np.pi / 3, 2 * np.pi / 3):
        # Reflection in the plane containing z-axis and direction (cos phi, sin phi).
        nhat = np.array([-np.sin(phi), np.cos(phi), 0.0])  # plane normal
        M = np.eye(3) - 2 * np.outer(nhat, nhat)
        rotations.append(M)
    D = [sym._wigner_d_complex(1, R) for R in rotations]
    _, tags = sym._serre_isotypic_basis(D)
    counts = sorted([tags.count(t) for t in set(tags)])
    assert counts == [1, 2], f"expected [1, 2], got {counts}"


# ---------------------------------------------------------------------------
# Neighbour finding
# ---------------------------------------------------------------------------


def _hbn_atoms() -> Atoms:
    """Generate a planar hBN unit cell."""
    a = 2.504  # Angstrom, typical hBN
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2, a * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, 15.0],  # vacuum
        ]
    )
    # B at (0,0,0), N at (1/3, 2/3, 0) in fractional coords.
    scaled = np.array([[0.0, 0.0, 0.5], [1.0 / 3, 2.0 / 3, 0.5]])
    atoms = Atoms(
        symbols=["B", "N"],
        scaled_positions=scaled,
        cell=cell,
        pbc=(True, True, False),
    )
    return atoms


def test_hbn_three_neighbours():
    """Each hBN site has exactly 3 nearest neighbours lying in the xy-plane."""
    atoms = _hbn_atoms()
    # Both atoms in kapaow numbering: B=0, N=1 (species order B,N).
    atom_map = {0: 0, 1: 1}
    bonds = sym._find_bond_neighbours(atoms, atom_map, cutoff=None)
    assert len(bonds[0]) == 3
    assert len(bonds[1]) == 3
    # All of B's neighbours should be N atoms.
    for neighbour_idx, _ in bonds[0]:
        assert neighbour_idx == 1
    # Bond vectors should be unit vectors, lying in xy plane.
    for _, v in bonds[0]:
        assert np.isclose(np.linalg.norm(v), 1.0)
        assert abs(v[2]) < 1e-8


# ---------------------------------------------------------------------------
# End-to-end: hBN via monkey-patched lmax reader (no .dat file needed)
# ---------------------------------------------------------------------------


def _hbn_atoms_dict() -> dict:
    return {
        "B": {"num_atoms": 1, "positions": [[0.0, 0.0, 0.5]]},
        "N": {"num_atoms": 1, "positions": [[1.0 / 3, 2.0 / 3, 0.5]]},
    }


def test_hbn_symmetrize_end_to_end(tmp_path, monkeypatch):
    """For hBN with s+p basis: 3 sp2 hybrids + 1 pz irrep per atom, unitary B."""
    atoms = _hbn_atoms()
    structure_file = tmp_path / "hBN.xsf"
    ase_write(str(structure_file), atoms)

    lattice_vectors = np.array(atoms.cell)
    atoms_dict = _hbn_atoms_dict()

    # Short-circuit the .dat reader: s+p on both species.
    monkeypatch.setattr(
        sym,
        "_read_species_l_lists",
        lambda proj_dir, atoms_dict: {"B": [0, 1], "N": [0, 1]},
    )

    B, labels = sym.symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=tmp_path,  # not used (reader is patched)
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=True,
    )

    # Shape and unitarity.
    assert B.shape == (8, 8)
    assert np.allclose(B @ B.conj().T, np.eye(8), atol=1e-10)

    # Block diagonal across atoms (0..3 and 4..7).
    assert np.allclose(B[:4, 4:], 0, atol=1e-12)
    assert np.allclose(B[4:, :4], 0, atol=1e-12)

    # Each atom should have exactly 3 hybrids + 1 irrep complement.
    kinds_per_atom = {0: [], 1: []}
    for lab in labels:
        kinds_per_atom[lab.atom_index].append(lab.kind)
    for atom_idx in (0, 1):
        assert kinds_per_atom[atom_idx].count("hybrid") == 3
        assert kinds_per_atom[atom_idx].count("irrep") == 1

    # The non-bonding (irrep) orbital on atom 0 should be (essentially) pure pz.
    # The flat layout is in W90's real-Ylm basis with l=1 ordered (pz, px, py),
    # so within atom 0's (s, pz, px, py) slots pz lives at local index 1.
    irrep_row = next(B[i] for i in range(4) if labels[i].kind == "irrep")
    weights = np.abs(irrep_row[:4]) ** 2
    assert weights[1] > 0.99, f"pz not isolated in complement: {weights}"

    # Each hybrid lobe on atom 0 should point at a distinct bond_target and
    # together the three bond_targets should be 3 copies of atom index 1 (the
    # three N images under PBC all map back to atom 1).
    hybrid_targets = [
        lab.bond_target for lab in labels if lab.atom_index == 0 and lab.kind == "hybrid"
    ]
    assert all(t == 1 for t in hybrid_targets)
    assert len(hybrid_targets) == 3


def test_hbn_extended_basis_flat_vs_padded(tmp_path, monkeypatch):
    """N with l_list=[0, 0, 1] (2 s + 1 p) stresses the padding case.

    The flat layout has 4 (B) + 5 (N) = 9 slots; the padded layout has
    4 (B) + 8 (N) = 12 slots with 3 zero-padded p-with-n=1 slots on N.
    The two should have the same *real* content (same per-atom hybrids
    and non-bonding orbitals) but different global dimensions.
    """
    atoms = _hbn_atoms()
    structure_file = tmp_path / "hBN.xsf"
    ase_write(str(structure_file), atoms)
    lattice_vectors = np.array(atoms.cell)
    atoms_dict = _hbn_atoms_dict()

    monkeypatch.setattr(
        sym,
        "_read_species_l_lists",
        lambda proj_dir, atoms_dict: {"B": [0, 1], "N": [0, 0, 1]},
    )

    B_flat, labels_flat = sym.symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=tmp_path,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=True,
        with_l_padding=False,
    )
    B_padded, labels_padded = sym.symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=tmp_path,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=True,
        with_l_padding=True,
    )

    # Flat: 4 (B: s+p) + 5 (N: 2s + p) = 9 real orbitals.
    assert B_flat.shape == (9, 9)
    assert np.allclose(B_flat @ B_flat.conj().T, np.eye(9), atol=1e-10)
    assert len(labels_flat) == 9
    assert all(lab.kind != "padding" for lab in labels_flat)

    # Padded: 4 (B rectangular) + 8 (N rectangular, 3 zero-padded) = 12.
    assert B_padded.shape == (12, 12)
    assert np.allclose(B_padded @ B_padded.conj().T, np.eye(12), atol=1e-10)
    assert len(labels_padded) == 12
    # 3 padded slots on N (the absent l=1, n=1 copies).
    padded = [i for i, lab in enumerate(labels_padded) if lab.kind == "padding"]
    assert padded == [7, 9, 11]
    # Padded rows/columns should be identity.
    for i in padded:
        row = B_padded[i].copy()
        row[i] = 0
        assert np.allclose(row, 0, atol=1e-14)
        assert np.isclose(B_padded[i, i], 1.0, atol=1e-14)

    # Both layouts give the same per-atom breakdown (3 hybrids + 2 irrep
    # non-bonding on N, 3 hybrids + 1 irrep non-bonding on B).
    def _counts(labels):
        out: dict[tuple[int, str], int] = {}
        for lab in labels:
            if lab.kind == "padding":
                continue
            out[(lab.atom_index, lab.kind)] = out.get((lab.atom_index, lab.kind), 0) + 1
        return out

    assert _counts(labels_flat) == _counts(labels_padded)
    assert _counts(labels_flat) == {
        (0, "hybrid"): 3,
        (0, "irrep"): 1,
        (1, "hybrid"): 3,
        (1, "irrep"): 2,
    }


def test_hbn_no_hybridize(tmp_path, monkeypatch):
    """With hybridize=False all labels are irreps; B is still unitary."""
    atoms = _hbn_atoms()
    structure_file = tmp_path / "hBN.xsf"
    ase_write(str(structure_file), atoms)
    lattice_vectors = np.array(atoms.cell)
    atoms_dict = _hbn_atoms_dict()

    monkeypatch.setattr(
        sym,
        "_read_species_l_lists",
        lambda proj_dir, atoms_dict: {"B": [0, 1], "N": [0, 1]},
    )

    B, labels = sym.symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=tmp_path,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=False,
    )
    # With hybridize=False all labels should be irrep, no hybrids.
    assert all(lab.kind == "irrep" for lab in labels)
    assert B.shape == (8, 8)
    assert np.allclose(B @ B.conj().T, np.eye(8), atol=1e-10)


# ---------------------------------------------------------------------------
# Pyramidal geometry (phosphorene-like): sp3 with lone pair
# ---------------------------------------------------------------------------


def _pyramidal_atoms() -> Atoms:
    """4-atom cell: one central atom with 3 pyramidal neighbours.

    Mimics phosphorene-like coordination: bond angle ~100 degrees,
    3 bonds pointing downward from the central atom, leaving a lone-pair
    direction pointing upward.
    """
    a = 15.0  # large cell to isolate one coordination shell
    cell = np.diag([a, a, a])
    d = 2.2  # bond length in Angstrom
    # Place bonds in a C3v arrangement around -z (pyramidal).
    # The tetrahedral half-angle from the C3 axis is arccos(1/3) ~ 70.5 deg,
    # giving inter-bond angles of 109.5 deg.
    alpha = np.arccos(1.0 / 3)
    r = d * np.sin(alpha)
    h = d * np.cos(alpha)
    center = np.array([a / 2, a / 2, a / 2])
    neighbours = []
    for phi in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        neighbours.append(center + np.array([r * np.cos(phi), r * np.sin(phi), -h]))
    positions = [center, *neighbours]
    atoms = Atoms(
        symbols=["P"] * 4,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    return atoms


def _pyramidal_atoms_dict() -> dict:
    atoms = _pyramidal_atoms()
    scaled = atoms.get_scaled_positions()
    return {
        "P": {
            "num_atoms": 4,
            "positions": scaled.tolist(),
        },
    }


def test_pyramidal_sp3(tmp_path, monkeypatch):
    """Pyramidal coordination (3 bonds + 1 lone pair) should give sp3."""
    atoms = _pyramidal_atoms()
    structure_file = tmp_path / "pyramidal.xsf"
    ase_write(str(structure_file), atoms)
    lattice_vectors = np.array(atoms.cell)
    atoms_dict = _pyramidal_atoms_dict()

    monkeypatch.setattr(
        sym,
        "_read_species_l_lists",
        lambda proj_dir, atoms_dict: {"P": [0, 1]},
    )

    B, labels = sym.symmetry_adapted_rotation(
        structure_file=structure_file,
        proj_dir=tmp_path,
        atoms_dict=atoms_dict,
        lattice_vectors=lattice_vectors,
        hybridize=True,
    )

    # B should be unitary.
    n = B.shape[0]
    assert np.allclose(B @ B.conj().T, np.eye(n), atol=1e-10)

    # The central atom (index 0) has 3 bonds in a pyramidal arrangement:
    # 3 hybrids + 1 complement (the lone pair direction).
    central_labels = [lab for lab in labels if lab.atom_index == 0]
    assert len(central_labels) == 4
    hybrids = [lab for lab in central_labels if lab.kind == "hybrid"]
    irreps = [lab for lab in central_labels if lab.kind == "irrep"]
    assert len(hybrids) == 3, f"expected 3 hybrids, got {len(hybrids)}"
    assert len(irreps) == 1, f"expected 1 irrep, got {len(irreps)}"

    # Hybrid name should be sp3 (not sp2), reflecting the full hybrid
    # subspace dimension despite only 3 bond-directed hybrids.
    for lab in hybrids:
        assert lab.irrep == "sp3", f"expected sp3, got {lab.irrep}"


# ---------------------------------------------------------------------------
# Complex -> W90 real Ylm basis change
# ---------------------------------------------------------------------------


def test_complex_to_real_ylm_unitary_is_unitary():
    """_complex_to_real_ylm_unitary returns a unitary matrix for l=0..3."""
    for l in range(4):
        U = sym._complex_to_real_ylm_unitary(l)
        assert U.shape == (2 * l + 1, 2 * l + 1)
        assert np.allclose(U @ U.conj().T, np.eye(2 * l + 1), atol=1e-12)


def test_complex_to_real_ylm_unitary_l1_explicit():
    """l=1 unitary matches the explicit W90 pz/px/py ordering."""
    # W90 ordering: row 0 = pz, row 1 = px, row 2 = py.
    # Columns indexed by m = -1, 0, +1.
    U = sym._complex_to_real_ylm_unitary(1)
    s = 1.0 / np.sqrt(2.0)
    expected = np.array(
        [
            [0, 1, 0],
            [s, 0, -s],
            [1j * s, 0, 1j * s],
        ],
        dtype=np.complex128,
    )
    assert np.allclose(U, expected, atol=1e-12)


def test_complex_to_real_ylm_pz_invariant_under_z_mirror():
    """A z-mirror in the complex basis is diag(-1, +1, -1) for l=1.

    After conjugating with U (complex -> real), it should become
    diag(-1, +1, +1) on (pz, px, py): pz flips sign, px and py are
    unchanged. This is the invariant the symmetrize bug was breaking.
    """
    sigma_h = np.diag([1.0, 1.0, -1.0])
    D_complex = sym._wigner_d_complex(1, sigma_h)
    U = sym._complex_to_real_ylm_unitary(1)
    D_real = U @ D_complex @ U.conj().T
    expected = np.diag([-1.0, 1.0, 1.0]).astype(np.complex128)
    assert np.allclose(D_real, expected, atol=1e-12)


def test_complex_to_real_basis_block_block_diagonal():
    """A multi-channel block (s + p) should be block-diagonal."""
    U = sym._complex_to_real_basis_block([0, 1])
    assert U.shape == (4, 4)
    assert np.allclose(U[0, 1:], 0)
    assert np.allclose(U[1:, 0], 0)
    assert np.isclose(U[0, 0], 1.0)
    # The p block matches the standalone l=1 unitary.
    assert np.allclose(U[1:, 1:], sym._complex_to_real_ylm_unitary(1), atol=1e-12)
