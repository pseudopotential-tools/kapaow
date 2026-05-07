"""Emit per-rank PAO datasets by post-processing a single femdvr solve."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from upf_tools import UPFDict

from kapaow.basis import AngularMomentum, AtomicBasis, PseudoatomicBasis
from kapaow.extend import BasisExtensionViaAddition
from kapaow.pydantic import BaseModel
from kapaow.solve import (
    ATOMIC_FEMDVR_PATCHES,
    DEFAULT_RI_FACTOR_MAX,
    OrbitalEnergy,
    _write_dat_for_basis,
    _write_upf_for_basis,
    read_femdvr_eigenvalues,
    solve_pseudoatomic_problem,
)

__all__: list[str] = [
    "RankRecord",
    "emit_ranks",
]

# Number of extra subshells (beyond baseline) to include in the trial solve.
# This bounds how many non-baseline orbitals femdvr will compute, which caps
# the maximum rank that can be emitted in a single pass.
_TRIAL_EXTRA_SUBSHELLS = 5

logger = logging.getLogger(__name__)


class RankRecord(BaseModel):
    """Record describing a single emitted augmentation rank.

    Parameters
    ----------
    rank
        Augmentation rank K (0 = baseline only).
    dat_path
        Path to the emitted ``.dat`` file.
    upf_path
        Path to the emitted augmented ``.upf`` file.
    added_orbital
        The orbital added at this rank, or ``None`` for rank 0.
    """

    rank: int
    dat_path: Path
    upf_path: Path
    added_orbital: OrbitalEnergy | None


def _baseline_pseudo_basis(upf_path: Path) -> tuple[AtomicBasis, PseudoatomicBasis]:
    """Return the atomic and pseudoatomic baseline bases from a UPF file.

    Parameters
    ----------
    upf_path
        Path to the UPF pseudopotential file.

    Returns
    -------
    atomic_basis, pseudo_basis
        The :class:`~kapaow.basis.AtomicBasis` from the UPF projectors, and
        its :class:`~kapaow.basis.PseudoatomicBasis` equivalent.
    """
    atomic_basis = AtomicBasis.from_upf(upf_path)
    return atomic_basis, atomic_basis.to_pseudoatomic_basis()


def _build_rank_k_basis(
    baseline_pseudo: PseudoatomicBasis,
    extras: list[OrbitalEnergy],
    k: int,
) -> PseudoatomicBasis:
    """Construct the rank-K pseudoatomic basis = baseline + lowest K extras.

    An extra orbital at ``(l, n_radial)`` is included by ensuring the
    per-channel count in the new basis is at least ``n_radial + 1``.  This
    works because femdvr returns orbitals in ascending energy order within
    each l channel, so n_radial 0 is the lowest, 1 the next, and so on.

    Parameters
    ----------
    baseline_pseudo
        The baseline :class:`~kapaow.basis.PseudoatomicBasis`.
    extras
        Extra orbitals sorted by raw femdvr eigenvalue ascending (most
        bound first), produced by :func:`emit_ranks`.
    k
        Number of extra orbitals to include (0 = baseline only).

    Returns
    -------
    PseudoatomicBasis
        Basis with the baseline orbitals plus the K lowest-energy extras.
    """
    counts: dict[AngularMomentum, int] = dict(baseline_pseudo.number_of_orbitals)
    for orbital in extras[:k]:
        current = counts.get(orbital.l, 0)
        needed = orbital.n_radial + 1
        if needed > current:
            counts[orbital.l] = needed
    return PseudoatomicBasis(number_of_orbitals=counts)


def _resolve_rc_and_ri_factor(
    rc: float | None,
    rc_search_json: Path | None,
    ri_factor: float | None,
) -> tuple[float, float]:
    """Return ``(rc, ri_factor)`` for :func:`emit_ranks`, validating inputs.

    Exactly one of ``rc`` and ``rc_search_json`` must be given. When
    ``rc_search_json`` is used and the caller did not override
    ``ri_factor``, the JSON's value is inherited so the rc-search JSON
    is the single source of truth for confinement geometry.
    """
    if (rc is None) == (rc_search_json is None):
        raise ValueError("Provide exactly one of rc or rc_search_json.")
    if rc_search_json is not None:
        data = json.loads(rc_search_json.read_text(encoding="utf-8"))
        rc = float(data["rc"])
        if ri_factor is None:
            ri_factor = float(data["ri_factor"])
    if ri_factor is None:
        ri_factor = DEFAULT_RI_FACTOR_MAX
    if rc is None:
        raise AssertionError("rc must be set at this point")  # unreachable
    return rc, ri_factor


def emit_ranks(
    upf_path: Path,
    *,
    rc: float | None = None,
    rc_search_json: Path | None = None,
    ri_factor: float | None = None,
    max_rank: int | None = None,
    output_dir: Path = Path("."),
) -> list[RankRecord]:
    """Run a single femdvr solve and emit per-rank ``.dat`` files.

    Exactly one of *rc* / *rc_search_json* must be given.  *rc_search_json*
    is the file written by ``kapaow optimize rc``; its top-level ``"rc"``
    field is used.

    The trial basis is built from the baseline (UPF projectors) extended by
    :data:`_TRIAL_EXTRA_SUBSHELLS` additional subshells in Madelung order
    (lmax capped at :attr:`~kapaow.basis.AngularMomentum.G`). femdvr is
    called once for that trial basis. Rank K = 0 emits only the baseline
    orbitals; rank K = 1 adds the lowest-energy non-baseline orbital; etc.

    Parameters
    ----------
    upf_path
        Path to the UPF pseudopotential file.
    rc
        Confinement radius in Bohr.
    rc_search_json
        Path to a JSON file written by ``kapaow optimize rc``.  The
        ``"rc"`` field is read from it.
    ri_factor
        Inner-radius factor passed to the confinement potential.
    max_rank
        If given, cap the number of extra orbitals at this value (at most
        ``max_rank + 1`` files are written, including rank 0).
    output_dir
        Directory where ``.dat`` and ``.json`` files are written.

    Returns
    -------
    list[RankRecord]
        One record per rank emitted, from rank 0 to the final rank.

    Raises
    ------
    ValueError
        If not exactly one of *rc* / *rc_search_json* is provided.
    """
    rc, ri_factor = _resolve_rc_and_ri_factor(rc, rc_search_json, ri_factor)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = upf_path.stem

    # --- build baseline and trial extension ---
    _, baseline_pseudo = _baseline_pseudo_basis(upf_path)

    element = UPFDict.from_upf(upf_path)["header"]["element"].strip()
    atomic_femdvr_config = ATOMIC_FEMDVR_PATCHES.get(element)

    trial_extension = BasisExtensionViaAddition(increment=_TRIAL_EXTRA_SUBSHELLS)

    logger.info(
        "emit_ranks: upf=%s rc=%.4f ri_factor=%.4f trial_extra_subshells=%d",
        upf_path.name,
        rc,
        ri_factor,
        _TRIAL_EXTRA_SUBSHELLS,
    )

    # Run the solve in a private scratch directory so femdvr's
    # intermediate files (*_qe.dat, *_bessel.h5, *_eigenvalues.dat,
    # *_density_potential.h5, *.log) don't pollute the user's output_dir.
    with tempfile.TemporaryDirectory(prefix=f"emit_ranks_{stem}_") as scratch_str:
        scratch = Path(scratch_str)
        trial_dat = Path(f"{stem}_trial.dat")
        result = solve_pseudoatomic_problem(
            upf_path,
            rc=rc,
            ri_factor=ri_factor,
            extension=trial_extension,
            working_dir=scratch,
            dat_filename=trial_dat,
            output_wfc_bessel=False,
            output_wfc_hdf5=True,
            atomic_femdvr_config=atomic_femdvr_config,
        )

        # --- enumerate all eigenvalues from the solve ---
        all_orbitals = read_femdvr_eigenvalues(result)

        # --- partition into baseline vs extra orbitals ---
        baseline_counts = baseline_pseudo.number_of_orbitals
        extras: list[OrbitalEnergy] = []
        for orb in all_orbitals:
            n_baseline = baseline_counts.get(orb.l, 0)
            if orb.n_radial >= n_baseline:
                extras.append(orb)

        # Sort extras by raw eigenvalue ascending (most bound first).
        extras.sort(key=lambda o: o.energy)

        if max_rank is not None:
            extras = extras[:max_rank]

        logger.info(
            "emit_ranks: %d baseline orbital channels, %d extras available (max_rank=%s)",
            sum(1 for n in baseline_counts.values() if n > 0),
            len(extras),
            max_rank,
        )

        # --- emit one (.dat, .upf) pair per rank, sourced from the scratch *_qe.dat ---
        records: list[RankRecord] = []
        for k in range(len(extras) + 1):
            rank_basis = _build_rank_k_basis(baseline_pseudo, extras, k)
            dat_name = output_dir / f"{stem}_rank{k}.dat"
            upf_name = output_dir / f"{stem}_rank{k}.upf"
            _write_dat_for_basis(scratch, dat_name, rank_basis)
            _write_upf_for_basis(scratch, upf_path, upf_name, rank_basis, all_orbitals)

            added: OrbitalEnergy | None = extras[k - 1] if k > 0 else None
            records.append(
                RankRecord(rank=k, dat_path=dat_name, upf_path=upf_name, added_orbital=added)
            )
            logger.info("  rank %d -> %s, %s", k, dat_name.name, upf_name.name)

    # --- write index JSON ---
    from importlib.metadata import version as _pkg_version

    ranks_json_path = output_dir / f"{stem}_ranks.json"
    index: dict[str, Any] = {
        "upf": str(upf_path),
        "kapaow_version": _pkg_version("kapaow"),
        "rc": rc,
        "ri_factor": ri_factor,
        "ranks": [],
    }
    for rec in records:
        entry: dict[str, Any] = {
            "rank": rec.rank,
            "dat": rec.dat_path.name,
            "upf": rec.upf_path.name,
            "added": None,
        }
        if rec.added_orbital is not None:
            orb = rec.added_orbital
            entry["added"] = {
                "l": orb.l.value,
                "n_radial": orb.n_radial,
                "energy": orb.energy,
            }
        index["ranks"].append(entry)

    ranks_json_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Rank index written to %s", ranks_json_path)

    return records
