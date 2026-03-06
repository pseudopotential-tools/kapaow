"""Solve the pseudoatomic problem."""

from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, solve_pseudo_atomic
from upf_tools import UPFDict

from pao_plusplus.basis import AngularMomentum, AtomicBasis, PseudoatomicBasis, Subshell
from pao_plusplus.extend import BasisExtension
from pao_plusplus.io import read_wannier90_dat_file, write_wannier90_dat_file


def solve_pseudoatomic_problem(
    upf_path: Path,
    rc: float | None = None,
    ri_factor: float | None = None,
    extension: BasisExtension | None = None,
    working_dir: Path = Path("."),
    dat_filename: Path | str | None = None,
    atomic_femdvr_config: PseudoAtomicInput | None = None,
) -> None:
    """Solve the pseudoatomic problem for a given UPF file with a soft confinement potential."""
    # Construct the atomic basis
    upf_dict = UPFDict.from_upf(upf_path)
    atomic_basis = AtomicBasis(
        subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
    )

    # Extend the basis if requested
    if extension is not None:
        pseudo_basis = extension.extend(atomic_basis)
    else:
        pseudo_basis = atomic_basis.to_pseudoatomic_basis()

    # Construct the settings for atomic-femdvr
    if atomic_femdvr_config is None:
        atomic_femdvr_config = PseudoAtomicInput()
    atomic_femdvr_config.sysparams.file_upf = str(upf_path)
    atomic_femdvr_config.sysparams.nmax = pseudo_basis.n_max
    atomic_femdvr_config.sysparams.lmax = pseudo_basis.l_max.value
    atomic_femdvr_config.sysparams.element = upf_dict["header"]["element"]
    atomic_femdvr_config.confinement.type = "softstep"
    atomic_femdvr_config.confinement.polarization_mode = "softcoul"
    atomic_femdvr_config.confinement.Vbarrier = 10.0
    atomic_femdvr_config.output.output_wfc_qe = True
    atomic_femdvr_config.output.output_wfc_bessel = True
    if rc is not None:
        atomic_femdvr_config.confinement.rc = rc
    if ri_factor is not None:
        atomic_femdvr_config.confinement.ri_factor = ri_factor

    # Remove any pre-existing hdf5 files
    hdf5_files = Path().glob("*_density_potential.h5")
    for hdf5_file in hdf5_files:
        hdf5_file.unlink()

    # Solve the pseudoatomic problem
    dat_filename = Path(upf_path.name).with_suffix(".dat") if dat_filename is None else dat_filename
    with redirect_stdout(
        open((working_dir / dat_filename).with_suffix(".log"), "w", encoding="utf-8")
    ):
        solve_pseudo_atomic(
            atomic_femdvr_config, task_list=("scf", "optimize", "nscf"), export_dir=str(working_dir)
        )

    # Regenerate the dat file to only include the desired orbitals
    tmp_dat_file = max(working_dir.glob("*.dat"), key=lambda f: f.stat().st_mtime)
    x, r, l_values, orbitals = read_wannier90_dat_file(tmp_dat_file)
    selected_orbitals = [orbitals[i] for i in _find_matches(l_values, pseudo_basis.l_values)]


    # Write to the requested dat_filename
    write_wannier90_dat_file(
        working_dir / dat_filename, x, r, pseudo_basis.l_values, np.array(selected_orbitals)
    )

    # Filter the Bessel HDF5 file to only include the desired number of orbitals per l
    _filter_bessel_file(working_dir, pseudo_basis)


def _filter_bessel_file(working_dir: Path, pseudo_basis: PseudoatomicBasis) -> None:
    """Rewrite the Bessel HDF5 file to only include the desired orbitals per l channel."""
    bessel_files = sorted(working_dir.glob("*_bessel.h5"), key=lambda f: f.stat().st_mtime)
    if not bessel_files:
        return
    bessel_file = bessel_files[-1]

    with h5py.File(bessel_file, "r") as f:
        qgrid = f["qgrid"][:]
        wf_bessel = f["wf_bessel"][:]  # shape [lmax+1, nmax+1, nq]

    # Build the filtered array: for each l, keep only the desired number of n values
    n_per_l = pseudo_basis.number_of_orbitals
    new_nmax = max(n_per_l.values()) - 1
    new_lmax = pseudo_basis.l_max.value

    filtered = np.zeros([new_lmax + 1, new_nmax + 1, len(qgrid)])
    for l in AngularMomentum:
        if l.value > new_lmax:
            break
        n_orbs = n_per_l.get(l, 0)
        filtered[l.value, :n_orbs, :] = wf_bessel[l.value, :n_orbs, :]

    with h5py.File(bessel_file, "w") as f:
        f.attrs["lmax"] = new_lmax
        f.attrs["nmax"] = new_nmax
        f.create_dataset("qgrid", data=qgrid)
        f.create_dataset("wf_bessel", data=filtered)


def _find_matches(values: list[int], desired_values: list[int]) -> list[int]:
    """Find the indices such that [values[i] for i in indices] == desired_values."""
    matches: list[int] = []
    for desired_value in desired_values:
        for i, value in enumerate(values):
            if value == desired_value and i not in matches:
                matches.append(i)
                break
    if not len(matches) == len(desired_values):
        raise ValueError("Could not find all desired values in the provided list.")
    return matches
