"""Symmetry-adapted, bond-oriented projector rotations.

Builds a block-diagonal unitary ``B`` (one block per atom) that rotates
the raw atomic projector basis ``{Y_l^m}`` into a symmetry-adapted,
bond-oriented basis (e.g. three sp2 hybrids + p_z on a D_3h site such
as B in hBN, or sp3d2 + t2g on an octahedrally coordinated transition
metal).

The rotation is applied downstream as ``A' = B A`` to the ``.amn`` file
so Wannier90 consumes the rotated projectors directly.

Projector basis conventions
    * **Internal** (Wigner-D, hybrid construction, irrep adaptation):
      complex spherical harmonics ``Y_l^m`` with Condon-Shortley phase
      and m ordering ``-l, -l+1, ..., +l``. This matches the local
      projector path in :mod:`kapaow.fat_bands` (``AtomicWFC`` /
      ``compute_amn_from_wfc``), used by the rectangular *padded* layout.
    * **Flat / .dat output** (consumed by ``pw2wannier90``'s ``.amn``):
      W90's real spherical harmonic ordering -- l=1: (pz, px, py); l=2:
      (dz², dxz, dyz, dx²-y², dxy); l=3 in the standard W90 m_r order.
      The flat ``B`` returned by :func:`symmetry_adapted_rotation` is
      converted from the internal complex basis to this real basis via
      a unitary right-multiplication so it can be applied directly to
      ``pw2wannier90``'s ``.amn``.
    * Per atom: one radial channel per entry in the species' ``.dat``
      file, iterated in .dat order. Each channel with angular momentum
      ``l`` contributes ``(2l+1)`` contiguous orbitals.

The per-l phase factor (-i)^l used internally by ``AtomicWFC`` when
evaluating projectors on plane waves is not represented here: Wigner-D
matrices commute with it within each l block, and the bond angular
matrix below applies the same phase consistently when mixing across l.

Algorithm (per atom)
--------------------
1. **Hybrid step** (if neighbours present). Build the bond angular matrix
   M of shape ``(n_bonds, n_orb)`` by evaluating every stored orbital at
   every bond unit vector. Lowdin-orthonormalize its rows to get
   ``n_bonds`` orthonormal bond-oriented hybrid coefficient vectors. The
   hybridization spans the *full* atom orbital space, so s+p+d mixing is
   allowed wherever the bond geometry demands it.
2. **Complement step**. Orthonormal complement of the hybrid subspace,
   obtained from the SVD of the hybrid block.
3. **Irrep step**. The site-group representation (complex Wigner-D,
   block-diagonal in l) is projected onto the complement subspace and
   Serre-averaged to produce a symmetry-adapted non-bonding basis (e.g.
   pz on hBN, t2g on octahedral TM sites).

First-pass scope
----------------
* Only ``nmax == 0`` (single radial channel per l) is supported.
* Only l <= 3 is supported for Wigner-D construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from ase.io import read as ase_read
from scipy.special import sph_harm_y

logger = logging.getLogger(__name__)

_SPGLIB_SITE_TOL = 1e-4
_POS_TOL = 1e-3
_DEG_TOL = 1e-6



# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitalLabel:
    """Label describing one rotated projector after symmetrization.

    ``kind == "padding"`` marks zero-padded slots in the rectangular
    ``AtomicWFC`` layout that do not correspond to a real projector and
    whose B rows are left as identity.
    """

    species: str
    atom_index: int
    kind: Literal["hybrid", "irrep", "padding"]
    irrep: str | None = None
    bond_target: int | None = None


@dataclass
class AtomBlock:
    """Orbital layout info for a single atom.

    The module works internally in a "real-slot" basis: one index per
    real projector from the atom's ``.dat`` file (including the 2l+1 m
    components of each radial entry). When the host layout is the
    rectangular ``AtomicWFC`` one (with zero-padding for (l, n) pairs
    that are absent in the ``.dat`` file), ``real_local_slots`` maps
    each real-slot index to its position inside the atom's rectangular
    block.

    Attributes
    ----------
    lmax, nmax
        Dimensions of the rectangular atom block used by
        ``AtomicWFC`` / ``get_channel_indices``.
    l_list
        l value of each radial channel in ``.dat`` file order.
    flat_l_list
        l value of each real-slot index (length ``num_orbitals``).
    real_local_slots
        Position of each real-slot index inside the rectangular block
        of size ``(lmax+1)**2 * (nmax+1)``.
    global_base
        Start of this atom's rectangular block in the global flat
        orbital index (i.e. ``AtomicWFC.start_indices[atom_index]``).
    """

    species: str
    atom_index: int
    position_frac: np.ndarray
    position_cart: np.ndarray
    lmax: int
    nmax: int
    l_list: list[int]
    flat_l_list: list[int]
    real_local_slots: list[int]
    global_base: int

    @property
    def num_orbitals(self) -> int:
        return len(self.real_local_slots)

    @property
    def num_rect_orbitals(self) -> int:
        return (self.lmax + 1) ** 2 * (self.nmax + 1)

    def local_l_of(self) -> list[int]:
        return list(self.flat_l_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def symmetry_adapted_rotation(
    structure_file: Path,
    proj_dir: Path,
    atoms_dict: dict,
    lattice_vectors: npt.NDArray[np.float64],
    *,
    hybridize: bool = True,
    bond_cutoff: float | None = None,
    with_l_padding: bool = False,
) -> tuple[npt.NDArray[np.complex128], list[OrbitalLabel]]:
    """Build the rotation matrix B and the list of rotated-orbital labels.

    Parameters
    ----------
    structure_file
        Path to the structure file used to derive site symmetry and bonds.
        Any ASE-readable format is accepted.
    proj_dir
        Directory containing ``{Element}.dat`` projector files. The per-
        species l inventory is read from these to determine the orbital
        layout; the radial contents are ignored.
    atoms_dict
        Species-grouped atom dictionary (fractional positions).
    lattice_vectors
        3x3 lattice vectors in the same units used for ``atoms_dict``.
    hybridize
        If True (default), apply the bond-oriented hybridization step.
        If False, return only the symmetry-adapted irrep basis.
    bond_cutoff
        Distance cutoff for nearest-neighbour search, in the same length
        units as ``lattice_vectors``. If None, picked automatically.
    with_l_padding
        Selects the output layout.

        * ``False`` (default): flat layout matching the ``.dat`` file
          order. B has shape ``(N_real, N_real)`` where ``N_real`` is the
          total number of real projectors across all atoms. This matches
          the amn written by ``pw2wannier90``.
        * ``True``: rectangular ``AtomicWFC`` layout with
          zero-padding for (l, n) pairs absent from the ``.dat`` file.
          B has shape ``(N_rect, N_rect)`` with identity rows/columns on
          padded slots and the real rotation scattered onto the real
          slots. This matches the amn produced by
          :func:`kapaow.fat_bands.compute_amn_from_wfc`.

    Returns
    -------
    B : ndarray, complex128
        Shape depends on ``with_l_padding``.
    labels : list[OrbitalLabel]
        One entry per row of ``B``. In the padded layout, slots that do
        not correspond to a real projector are tagged ``kind="padding"``.
    """
    species_l_lists = _read_species_l_lists(proj_dir, atoms_dict)
    blocks = _build_atom_blocks(species_l_lists, atoms_dict, lattice_vectors)

    atoms = ase_read(str(structure_file))
    ase_index_map = _match_blocks_to_ase(atoms, blocks)
    site_groups = _site_symmetry_groups(atoms, ase_index_map)
    neighbours = (
        _find_bond_neighbours(atoms, ase_index_map, bond_cutoff)
        if hybridize
        else {b.atom_index: [] for b in blocks}
    )

    # Compute the per-atom real-slot rotation once; the two output
    # layouts differ only in how we assemble these pieces.
    atom_U: list[np.ndarray] = []
    atom_labels: list[list[OrbitalLabel]] = []
    for block in blocks:
        U, labs = _rotate_atom_block(
            block,
            site_groups[block.atom_index],
            neighbours.get(block.atom_index, []),
        )
        atom_U.append(U)
        atom_labels.append(labs)

    if with_l_padding:
        return _assemble_padded(blocks, atom_U, atom_labels)
    return _assemble_flat(blocks, atom_U, atom_labels)


def _assemble_flat(
    blocks: list[AtomBlock],
    atom_U: list[np.ndarray],
    atom_labels: list[list[OrbitalLabel]],
) -> tuple[np.ndarray, list[OrbitalLabel]]:
    """Block-diagonal over atoms in real-slot (``.dat``) order.

    The per-atom rotations ``atom_U`` are built in the internal complex-
    Ylm basis (m = -l..+l). pw2wannier90's ``.amn`` is in W90's real-Ylm
    basis (e.g. l=1 -> pz, px, py), so for the flat layout we right-
    multiply each block by ``U_basis^dagger``, where ``U_basis`` maps
    complex Ylm to W90 real Ylm. Then ``B_real @ A_real`` produces the
    same rotated coefficients as ``B_complex @ A_complex``.
    """
    n_real = sum(b.num_orbitals for b in blocks)
    B = np.eye(n_real, dtype=np.complex128)
    labels: list[OrbitalLabel] = []
    cursor = 0
    for block, U, labs in zip(blocks, atom_U, atom_labels):
        n = block.num_orbitals
        U_basis = _complex_to_real_basis_block(block.l_list)
        B[cursor : cursor + n, cursor : cursor + n] = U @ U_basis.conj().T
        labels.extend(labs)
        cursor += n
    return B, labels


def _assemble_padded(
    blocks: list[AtomBlock],
    atom_U: list[np.ndarray],
    atom_labels: list[list[OrbitalLabel]],
) -> tuple[np.ndarray, list[OrbitalLabel]]:
    """Rectangular ``AtomicWFC`` layout with identity on zero-padded slots."""
    n_rect = sum(b.num_rect_orbitals for b in blocks)
    B = np.eye(n_rect, dtype=np.complex128)
    labels: list[OrbitalLabel] = [
        OrbitalLabel(species="", atom_index=-1, kind="padding")
        for _ in range(n_rect)
    ]
    for block, U, labs in zip(blocks, atom_U, atom_labels):
        global_real = [block.global_base + s for s in block.real_local_slots]
        B[np.ix_(global_real, global_real)] = U
        for local_i, g in enumerate(global_real):
            labels[g] = labs[local_i]
        real_set = set(block.real_local_slots)
        for local_slot in range(block.num_rect_orbitals):
            if local_slot in real_set:
                continue
            labels[block.global_base + local_slot] = OrbitalLabel(
                species=block.species,
                atom_index=block.atom_index,
                kind="padding",
            )
    return B, labels


def apply_rotation_to_amn(
    amn: npt.NDArray[np.complex128],
    B: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply B to an Amn array of shape (num_k, num_orb, num_bands)."""
    return np.einsum("ij,kjn->kin", B, amn)


def group_indices_by_label(
    labels: list[OrbitalLabel],
) -> dict[tuple[str, str], list[int]]:
    """Group global orbital indices by ``(species, irrep)``.

    Hybrids are grouped by their textbook name (e.g. ``"sp2"``, ``"sp3"``,
    ``"sp3d2"``); non-bonding orbitals by their angular character
    (e.g. ``"s"``, ``"pz"``). Padding slots are skipped.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, lab in enumerate(labels):
        if lab.kind == "padding":
            continue
        key = (lab.species, lab.irrep or "irrep")
        groups.setdefault(key, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Atom layout
# ---------------------------------------------------------------------------


def _read_species_l_lists(
    proj_dir: Path, atoms_dict: dict
) -> dict[str, list[int]]:
    """Return ``{species: [l_0, l_1, ...]}`` by reading each ``.dat`` file.

    The list order matches the ``.dat`` file, i.e. the order in which
    pw2wannier90 emits radial channels for that species.
    """
    from kapaow.io import read_wannier90_dat_file

    species_l_lists: dict[str, list[int]] = {}
    for species in atoms_dict:
        dat_file = proj_dir / f"{species}.dat"
        if not dat_file.exists():
            raise FileNotFoundError(
                f"symmetrize: missing projector file {dat_file}"
            )
        _, _, l_values, _ = read_wannier90_dat_file(dat_file)
        l_list = list(l_values)
        if any(l > 3 for l in l_list):
            raise NotImplementedError(
                f"symmetrize: only l <= 3 supported (species {species} has l_list={l_list})"
            )
        species_l_lists[species] = l_list
    return species_l_lists


def _build_atom_blocks(
    species_l_lists: dict[str, list[int]],
    atoms_dict: dict,
    lattice_vectors: np.ndarray,
) -> list[AtomBlock]:
    """Per-atom orbital blocks in the rectangular AtomicWFC layout.

    For each species, uses ``l_list`` from the ``.dat`` file together with
    ``lmax = max(l_list)`` and ``nmax = max(count(l in l_list)) - 1`` to
    build a rectangular atom block of size ``(lmax+1)**2 * (nmax+1)``
    matching ``AtomicWFC``. The real projectors occupy a subset of the
    slots; the remainder are zero-padded and left unrotated.
    """
    blocks: list[AtomBlock] = []
    cursor = 0
    atom_idx = 0
    for species, l_list in species_l_lists.items():
        lmax = max(l_list) if l_list else 0
        counts: dict[int, int] = {}
        for l in l_list:
            counts[l] = counts.get(l, 0) + 1
        nmax = max(counts.values()) - 1 if counts else 0
        num_rect = (lmax + 1) ** 2 * (nmax + 1)

        # Assign each .dat entry an n index by order of appearance per l,
        # and emit one real slot per m component (2l+1 per entry).
        n_used: dict[int, int] = {}
        real_local_slots: list[int] = []
        flat_l_list: list[int] = []
        for l in l_list:
            n = n_used.get(l, 0)
            n_used[l] = n + 1
            for m in range(-l, l + 1):
                angular_idx = l * l + l + m
                slot = angular_idx * (nmax + 1) + n
                real_local_slots.append(slot)
                flat_l_list.append(l)

        positions_frac = atoms_dict[species]["positions"]
        for pos in positions_frac:
            pos_frac = np.asarray(pos, dtype=float)
            pos_cart = pos_frac @ lattice_vectors
            blocks.append(
                AtomBlock(
                    species=species,
                    atom_index=atom_idx,
                    position_frac=pos_frac,
                    position_cart=pos_cart,
                    lmax=lmax,
                    nmax=nmax,
                    l_list=list(l_list),
                    flat_l_list=list(flat_l_list),
                    real_local_slots=list(real_local_slots),
                    global_base=cursor,
                )
            )
            cursor += num_rect
            atom_idx += 1
    return blocks


def _match_blocks_to_ase(atoms, blocks: list[AtomBlock]) -> dict[int, int]:
    """Match each ``AtomBlock.atom_index`` to its ASE index via fractional coords."""
    ase_frac = atoms.get_scaled_positions(wrap=True)
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for block in blocks:
        target = block.position_frac % 1.0
        best: int | None = None
        for i, pos in enumerate(ase_frac):
            if i in used:
                continue
            delta = (pos - target + 0.5) % 1.0 - 0.5
            if np.linalg.norm(delta) < _POS_TOL:
                best = i
                break
        if best is None:
            raise RuntimeError(
                f"Could not match atom block {block.atom_index} "
                f"(species={block.species}, pos={target}) to any ASE atom"
            )
        mapping[block.atom_index] = best
        used.add(best)
    return mapping


# ---------------------------------------------------------------------------
# Site symmetry via spglib
# ---------------------------------------------------------------------------


def _site_symmetry_groups(
    atoms, atom_map: dict[int, int]
) -> dict[int, list[np.ndarray]]:
    """Return per-atom list of Cartesian 3x3 point operations fixing the site."""
    import spglib

    cell = (np.array(atoms.cell), atoms.get_scaled_positions(), atoms.numbers)
    dataset = spglib.get_symmetry_dataset(cell, symprec=_SPGLIB_SITE_TOL)
    rotations_frac = getattr(dataset, "rotations", None)
    translations = getattr(dataset, "translations", None)
    if rotations_frac is None:
        rotations_frac = dataset["rotations"]
        translations = dataset["translations"]

    cell_mat = np.array(atoms.cell)
    inv_cell = np.linalg.inv(cell_mat)
    scaled = atoms.get_scaled_positions()

    out: dict[int, list[np.ndarray]] = {}
    for atom_idx, ase_idx in atom_map.items():
        site = scaled[ase_idx]
        stabilizer: list[np.ndarray] = []
        for R_frac, t in zip(rotations_frac, translations):
            new_pos = R_frac @ site + t
            delta = (new_pos - site + 0.5) % 1.0 - 0.5
            if np.linalg.norm(delta) < _POS_TOL:
                # Fractional -> Cartesian rotation: R_cart = A R_frac A^{-1},
                # where A has lattice vectors as *columns*.
                A = cell_mat.T
                R_cart = A @ R_frac @ np.linalg.inv(A)
                stabilizer.append(R_cart)
        out[atom_idx] = stabilizer
        logger.debug("Atom %d: site-symmetry order = %d", atom_idx, len(stabilizer))
    return out


# ---------------------------------------------------------------------------
# Wigner-D (complex)
# ---------------------------------------------------------------------------


def _euler_zyz(R: np.ndarray) -> tuple[float, float, float]:
    """ZYZ Euler angles for a proper rotation. R = Rz(a) Ry(b) Rz(g)."""
    cb = float(np.clip(R[2, 2], -1.0, 1.0))
    beta = float(np.arccos(cb))
    if abs(np.sin(beta)) > 1e-8:
        alpha = float(np.arctan2(R[1, 2], R[0, 2]))
        gamma = float(np.arctan2(R[2, 1], -R[2, 0]))
    else:
        alpha = float(np.arctan2(R[1, 0], R[0, 0]))
        gamma = 0.0
    return alpha, beta, gamma


def _wigner_small_d(l: int, beta: float) -> np.ndarray:
    from math import cos, factorial, sin, sqrt

    cb2 = cos(beta / 2.0)
    sb2 = sin(beta / 2.0)
    dim = 2 * l + 1
    d = np.zeros((dim, dim), dtype=np.float64)
    for i, mp in enumerate(range(-l, l + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            kmin = max(0, m - mp)
            kmax = min(l - mp, l + m)
            s = 0.0
            for k in range(kmin, kmax + 1):
                num = (-1) ** k * sqrt(
                    factorial(l + mp) * factorial(l - mp)
                    * factorial(l + m) * factorial(l - m)
                )
                den = (
                    factorial(l - mp - k)
                    * factorial(l + m - k)
                    * factorial(k)
                    * factorial(k + mp - m)
                )
                s += num / den * cb2 ** (2 * l + m - mp - 2 * k) * sb2 ** (
                    2 * k + mp - m
                )
            d[i, j] = s
    return d


def _wigner_d_complex(l: int, R: np.ndarray) -> np.ndarray:
    """Complex Wigner-D matrix D^l(R) in the m = -l..+l basis.

    Handles improper rotations via the parity factor (-1)**l, since
    Y_l^m(-r) = (-1)**l Y_l^m(r).
    """
    det = float(np.linalg.det(R))
    if det < 0:
        R_proper = -R
        parity = (-1) ** l
    else:
        R_proper = R
        parity = 1
    alpha, beta, gamma = _euler_zyz(R_proper)
    d = _wigner_small_d(l, beta)
    dim = 2 * l + 1
    D = np.zeros((dim, dim), dtype=np.complex128)
    for i, mp in enumerate(range(-l, l + 1)):
        for j, m in enumerate(range(-l, l + 1)):
            D[i, j] = (
                np.exp(-1j * mp * alpha) * d[i, j] * np.exp(-1j * m * gamma)
            )
    return parity * D


def _complex_to_real_ylm_unitary(l: int) -> np.ndarray:
    """Unitary mapping complex ``Y_l^m`` to W90 real spherical harmonics.

    Rows are indexed by W90's real-Ylm ordering (m_r = 1..2l+1):

        l=0: s
        l=1: pz, px, py
        l=2: dz², dxz, dyz, dx²-y², dxy
        l=3: fz³, fxz², fyz², fz(x²-y²), fxyz, fx(x²-3y²), fy(3x²-y²)

    Columns are indexed by m = -l..+l (column ``j`` ↔ m = j - l).

    Defined so that ``Y_real[k] = sum_m U[k, m+l] Y_complex^m`` in the
    Condon-Shortley phase convention. The standard relations for m > 0
    are::

        Y_lm_cos = (1/√2) (Y_l^{-m} + (-1)^m Y_l^{+m})   (e.g. px, dxz)
        Y_lm_sin = (i/√2) (Y_l^{-m} - (-1)^m Y_l^{+m})   (e.g. py, dyz)

    and ``Y_l0_real = Y_l^0``. Within each l, W90 orders the rows as
    m_r = 0 (the m=0 real harmonic), then for m = 1..l the cos row
    followed by the sin row.
    """
    if l > 3:
        raise NotImplementedError(
            f"_complex_to_real_ylm_unitary: only l <= 3 supported, got l={l}"
        )
    dim = 2 * l + 1
    U = np.zeros((dim, dim), dtype=np.complex128)
    U[0, l] = 1.0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for m in range(1, l + 1):
        sign = (-1) ** m
        cos_row = 2 * m - 1
        sin_row = 2 * m
        U[cos_row, l - m] = inv_sqrt2
        U[cos_row, l + m] = sign * inv_sqrt2
        U[sin_row, l - m] = 1j * inv_sqrt2
        U[sin_row, l + m] = -1j * sign * inv_sqrt2
    return U


def _complex_to_real_basis_block(l_list: list[int]) -> np.ndarray:
    """Block-diagonal complex→real Ylm unitary for one atom's orbitals.

    Channels are iterated in ``l_list`` order. Each channel of angular
    momentum ``l`` contributes a ``(2l+1) x (2l+1)`` block on the
    diagonal, given by :func:`_complex_to_real_ylm_unitary`.
    """
    dim = sum(2 * l + 1 for l in l_list)
    out = np.zeros((dim, dim), dtype=np.complex128)
    offset = 0
    for l in l_list:
        d = 2 * l + 1
        out[offset : offset + d, offset : offset + d] = _complex_to_real_ylm_unitary(l)
        offset += d
    return out


def _site_rep(block: AtomBlock, rotations: list[np.ndarray]) -> list[np.ndarray]:
    """Site-group rep on the atom's orbital block.

    Block-diagonal over the radial channels in ``block.l_list``: each
    channel contributes a ``(2l+1)``-dim Wigner-D block. The per-l phase
    factor (-i)^l commutes within-l so does not affect D^l; it only
    matters when orbitals of different l are mixed (see
    :func:`_bond_angular_matrix`).
    """
    dim = block.num_orbitals
    reps = []
    for R in rotations:
        M = np.zeros((dim, dim), dtype=np.complex128)
        offset = 0
        for l in block.l_list:
            d = 2 * l + 1
            M[offset : offset + d, offset : offset + d] = _wigner_d_complex(l, R)
            offset += d
        reps.append(M)
    return reps


# ---------------------------------------------------------------------------
# Serre irrep decomposition
# ---------------------------------------------------------------------------


def _serre_isotypic_basis(
    D: list[np.ndarray], rng: np.random.Generator | None = None
) -> tuple[np.ndarray, list[int]]:
    """Isotypic decomposition of a rep via Serre averaging.

    Build a random Hermitian H0, average over the group, diagonalize.
    Eigenvectors with near-degenerate eigenvalues sit in the same isotypic
    block.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    dim = D[0].shape[0] if D else 0
    if not D or dim == 0:
        return np.eye(dim, dtype=np.complex128), list(range(dim))

    X = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    H0 = 0.5 * (X + X.conj().T)
    H = np.zeros_like(H0)
    for R in D:
        H += R @ H0 @ R.conj().T
    H /= len(D)
    H = 0.5 * (H + H.conj().T)

    evals, U = np.linalg.eigh(H)
    tags: list[int] = []
    current_tag = 0
    prev: float | None = None
    tol = _DEG_TOL * max(1.0, float(np.max(np.abs(evals)) if evals.size else 1.0))
    for ev in evals:
        if prev is None or abs(ev - prev) > tol:
            if prev is not None:
                current_tag += 1
            prev = ev
        tags.append(current_tag)
    return U, tags


# ---------------------------------------------------------------------------
# Bond neighbours
# ---------------------------------------------------------------------------


def _find_bond_neighbours(
    atoms, atom_map: dict[int, int], cutoff: float | None
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """``{atom_idx: [(neighbour_atom_idx, unit_vector_cartesian), ...]}``.

    ``atom_idx`` / ``neighbour_atom_idx`` are in the AtomicWFC flat ordering.
    Bond vectors are unit Cartesian in ``atoms.cell`` units.
    """
    from ase.neighborlist import NeighborList

    inv_atom_map = {ase_i: atom_i for atom_i, ase_i in atom_map.items()}
    if cutoff is None:
        cutoff = _auto_bond_cutoff(atoms)

    nl = NeighborList(
        cutoffs=[cutoff / 2] * len(atoms),
        self_interaction=False,
        bothways=True,
        skin=0.0,
    )
    nl.update(atoms)

    cell = np.array(atoms.cell)
    positions = atoms.get_positions()

    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for atom_idx, ase_i in atom_map.items():
        indices, offsets = nl.get_neighbors(ase_i)
        bonds: list[tuple[int, np.ndarray]] = []
        for j, offset in zip(indices, offsets):
            neighbour_pos = positions[j] + offset @ cell
            vec = neighbour_pos - positions[ase_i]
            d = float(np.linalg.norm(vec))
            if d < 1e-6:
                continue
            neighbour_atom_idx = inv_atom_map.get(int(j))
            if neighbour_atom_idx is None:
                continue
            bonds.append((neighbour_atom_idx, vec / d))
        out[atom_idx] = bonds
    return out


def _auto_bond_cutoff(atoms) -> float:
    """1.3 x shortest inter-atomic distance (including nearest periodic image)."""
    positions = atoms.get_positions()
    cell = np.array(atoms.cell)
    n = len(atoms)
    best = np.inf
    for i in range(n):
        for j in range(n):
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    for c in (-1, 0, 1):
                        if i == j and a == 0 and b == 0 and c == 0:
                            continue
                        v = (
                            positions[j]
                            - positions[i]
                            + np.array([a, b, c]) @ cell
                        )
                        d = float(np.linalg.norm(v))
                        if 1e-6 < d < best:
                            best = d
    return float(best * 1.3)


# ---------------------------------------------------------------------------
# Per-atom rotation
# ---------------------------------------------------------------------------


def _rotate_atom_block(
    block: AtomBlock,
    rotations: list[np.ndarray],
    bonds: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, list[OrbitalLabel]]:
    """Construct the per-atom unitary and orbital labels."""
    dim = block.num_orbitals
    D = _site_rep(block, rotations)

    # Build hybrid coefficient vectors (rows) that span the bonded subspace.
    if bonds:
        n_bonds = len(bonds)
        if n_bonds > dim:
            raise RuntimeError(
                f"Atom {block.atom_index}: {n_bonds} bonds exceed orbital dim {dim}"
            )
        # Determine the smallest l cut-off such that the s..l subspace has
        # enough room for `n_bonds` hybrids. This forces sp^k hybridization
        # to use only s+p when geometrically possible (preferring valence
        # channels over diffuse d/f), and only pulls in d (or higher) when
        # the bond count exceeds the s+p dimension, as in octahedral sp3d2.
        hybrid_slots = _select_hybrid_slots(block, n_bonds)
        M_full = _bond_angular_matrix(block, bonds)
        M = M_full[:, hybrid_slots]

        S = M @ M.conj().T
        S = 0.5 * (S + S.conj().T)
        evals, evecs = np.linalg.eigh(S)
        if np.any(evals <= 1e-10):
            raise RuntimeError(
                f"Atom {block.atom_index}: bond directions yield a singular "
                f"Lowdin overlap (linearly dependent in the orbital basis)"
            )
        S_inv_half = (evecs * (evals ** -0.5)) @ evecs.conj().T
        hybrid_rows_cut = S_inv_half @ M  # (n_bonds, len(hybrid_slots))

        # Scatter back into the full atom block: non-hybrid slots get 0.
        hybrid_rows = np.zeros((n_bonds, dim), dtype=np.complex128)
        hybrid_rows[:, hybrid_slots] = hybrid_rows_cut
    else:
        hybrid_rows = np.zeros((0, dim), dtype=np.complex128)
        n_bonds = 0

    # Complement: orthonormal basis of the subspace orthogonal to all hybrid rows.
    if n_bonds < dim:
        if n_bonds == 0:
            complement = np.eye(dim, dtype=np.complex128)
        else:
            _, _, Vh = np.linalg.svd(hybrid_rows, full_matrices=True)
            complement = Vh[n_bonds:]  # (dim - n_bonds, dim)
    else:
        complement = np.zeros((0, dim), dtype=np.complex128)

    # Irrep-adapt the complement (if non-trivial and we have a site group).
    if complement.shape[0] > 0 and D:
        # Project each D(R) onto the complement: D_c = C D C^dagger, where C's
        # rows are the complement basis vectors (C has shape (n_c, dim)).
        C = complement
        D_c = [C @ R @ C.conj().T for R in D]
        U_c, _tags = _serre_isotypic_basis(D_c)
        # Orbital = sum_k (U_c^dagger)_{jk} (complement_k), i.e. new rows
        # are U_c^T @ complement.
        complement = U_c.conj().T @ complement

    comp_labels = _label_complement_rows(complement, block)

    # Assemble U: hybrid rows first, then irrep-adapted complement rows.
    U = np.concatenate([hybrid_rows, complement], axis=0)
    assert U.shape == (dim, dim)

    hybrid_name = _hybrid_name_from_rows(hybrid_rows, block)
    hybrid_labels = [
        OrbitalLabel(
            species=block.species,
            atom_index=block.atom_index,
            kind="hybrid",
            irrep=hybrid_name,
            bond_target=neighbour_atom_idx,
        )
        for neighbour_atom_idx, _ in bonds
    ]
    labels = hybrid_labels + comp_labels
    return U, labels


def _select_hybrid_slots(block: AtomBlock, n_bonds: int) -> list[int]:
    """Return the local slot indices that should form the hybrid subspace.

    Picks the smallest l cut-off such that ``sum(2l+1) >= n_bonds`` over
    the radial channels in ``block.l_list`` with l at or below the cut-off.
    This forces sp-only hybridization when geometrically sufficient (Si,
    hBN, graphene) and only pulls d (or higher) into the hybrid subspace
    when the bond count cannot be accommodated otherwise (e.g. octahedral
    sp3d2).
    """
    slots_by_l: dict[int, list[int]] = {}
    offset = 0
    for l in block.l_list:
        slots_by_l.setdefault(l, []).extend(range(offset, offset + 2 * l + 1))
        offset += 2 * l + 1

    selected: list[int] = []
    for l in sorted(slots_by_l):
        selected.extend(slots_by_l[l])
        if len(selected) >= n_bonds:
            return sorted(selected)
    if len(selected) < n_bonds:
        raise RuntimeError(
            f"Atom {block.atom_index}: {n_bonds} bonds exceed total orbital "
            f"dim {len(selected)}"
        )
    return sorted(selected)


def _hybrid_l_weights(
    hybrid_rows: np.ndarray, block: AtomBlock
) -> dict[int, float]:
    """Sum ``|c|^2`` over the orthonormal hybrid rows for each l channel."""
    weights: dict[int, float] = {}
    for row in hybrid_rows:
        w = np.abs(row) ** 2
        for slot, amp in zip(block.flat_l_list, w):
            weights[slot] = weights.get(slot, 0.0) + float(amp)
    return weights


def _hybrid_name_from_rows(
    hybrid_rows: np.ndarray, block: AtomBlock
) -> str:
    """Derive a textbook hybrid name (``"sp2"``, ``"sp3d2"``, ...) from weights.

    The per-l weights are in a rational ratio that identifies the
    hybridization: ``s:p = 1:2`` -> ``sp2``, ``s:p = 1:3`` -> ``sp3``,
    etc. We normalise by the smallest nonzero weight to recover the
    integer ratio.
    """
    if hybrid_rows.size == 0:
        return "hybrid"
    raw = _hybrid_l_weights(hybrid_rows, block)
    nonzero = [w for w in raw.values() if w > 1e-10]
    if not nonzero:
        return "hybrid"
    min_w = min(nonzero)
    parts: list[str] = []
    for l, w in sorted(raw.items()):
        count = round(w / min_w)
        if count == 0:
            continue
        char = _L_CHAR.get(l, f"l{l}")
        parts.append(char if count == 1 else f"{char}{count}")
    return "".join(parts) if parts else "hybrid"


def _label_complement_rows(
    complement: np.ndarray, block: AtomBlock
) -> list[OrbitalLabel]:
    """Tag each complement row with its dominant angular character.

    Each row of ``complement`` is a unit vector in the atom's orbital
    coefficient space. Its weight on each l channel is the sum of
    ``|c|^2`` over the slots belonging to that l. The irrep label is set
    to the dominant l as a short character string (e.g. ``"s"``, ``"p"``,
    ``"d"``). For pure p orbitals we further distinguish ``"pz"`` vs
    ``"p-in-plane"`` by looking at the m=0 vs m=+/-1 weights.
    """
    labels: list[OrbitalLabel] = []
    slot_l = block.local_l_of()
    for row in complement:
        w = np.abs(row) ** 2
        l_weights: dict[int, float] = {}
        for slot, amp in zip(slot_l, w):
            l_weights[slot] = l_weights.get(slot, 0.0) + float(amp)
        dominant_l, _ = max(l_weights.items(), key=lambda kv: kv[1])
        if dominant_l == 1:
            # Distinguish pz (m=0) vs in-plane p (m=+/-1).
            m0_weight = 0.0
            other_weight = 0.0
            offset = 0
            for l in block.l_list:
                if l == 1:
                    m0_weight += float(w[offset + 1])  # m=0 is middle slot
                    other_weight += float(w[offset + 0] + w[offset + 2])
                offset += 2 * l + 1
            irrep = "pz" if m0_weight > other_weight else "p-inplane"
        else:
            irrep = _L_CHAR.get(dominant_l, f"l={dominant_l}")
        labels.append(
            OrbitalLabel(
                species=block.species,
                atom_index=block.atom_index,
                kind="irrep",
                irrep=irrep,
            )
        )
    return labels


_L_CHAR = {0: "s", 1: "p", 2: "d", 3: "f"}


def _bond_angular_matrix(
    block: AtomBlock,
    bonds: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Matrix M_{jk} = (angular part of stored orbital k) evaluated at bond j.

    Covers the full atom orbital block. Each radial channel (l) in
    ``block.l_list`` contributes (2l+1) columns with the angular factor
    ``(-i)^l * Y_l^m(theta, phi)``.
    """
    dim = block.num_orbitals
    M = np.zeros((len(bonds), dim), dtype=np.complex128)
    for j, (_, unit_vec) in enumerate(bonds):
        x, y, z = unit_vec
        theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
        phi = float(np.arctan2(y, x))
        offset = 0
        for l in block.l_list:
            for m in range(-l, l + 1):
                Y = complex(sph_harm_y(l, m, theta, phi))
                M[j, offset + (m + l)] = (-1j) ** l * Y
            offset += 2 * l + 1
    return M
