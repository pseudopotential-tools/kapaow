"""Solve the pseudoatomic problem."""

from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, PseudoAtomicResult, solve_pseudo_atomic
from upf_tools import UPFDict

from pao_plusplus.basis import (
    AngularMomentum,
    AtomicBasis,
    PseudoatomicBasis,
    Subshell,
    ordered_subshells,
)
from pao_plusplus.extend import BasisExtension
from pao_plusplus.io import read_wannier90_dat_file, write_wannier90_dat_file

DEFAULT_RC_MIN = 5.0
DEFAULT_RC_MAX = 15.0
DEFAULT_RI_FACTOR_MIN = 0.0
DEFAULT_RI_FACTOR_MAX = 0.95


def solve_pseudoatomic_problem(
    upf_path: Path,
    rc: float = DEFAULT_RC_MAX,
    ri_factor: float = DEFAULT_RI_FACTOR_MAX,
    extension: BasisExtension | None = None,
    working_dir: Path = Path("."),
    dat_filename: Path | str | None = None,
    atomic_femdvr_config: PseudoAtomicInput | None = None,
    output_wfc_bessel: bool = True,
) -> PseudoAtomicResult:
    """Solve the pseudoatomic problem for a given UPF file with a soft confinement potential.

    Set up the atomic-femdvr configuration, run scf + optimize + nscf,
    and export wavefunctions to the working directory.
    """
    upf_dict = UPFDict.from_upf(upf_path)

    # Construct the atomic basis
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
    atomic_femdvr_config.confinement.rc = rc
    atomic_femdvr_config.confinement.ri_factor = ri_factor
    atomic_femdvr_config.output.output_wfc_qe = True
    atomic_femdvr_config.output.output_wfc_bessel = output_wfc_bessel

    # Remove any pre-existing hdf5 files
    hdf5_files = Path().glob("*_density_potential.h5")
    for hdf5_file in hdf5_files:
        hdf5_file.unlink()

    # Solve the pseudoatomic problem
    if dat_filename is None:
        dat_filename = Path(upf_path.name).with_suffix(".dat")
    log_path = (working_dir / dat_filename).with_suffix(".log")
    with redirect_stdout(open(log_path, "w", encoding="utf-8")):
        result = solve_pseudo_atomic(
            atomic_femdvr_config,
            task_list=("scf", "optimize", "nscf"),
            export_dir=str(working_dir),
        )

    return result


def solve_and_export(
    upf_path: Path,
    rc: float = DEFAULT_RC_MAX,
    ri_factor: float = DEFAULT_RI_FACTOR_MAX,
    extension: BasisExtension | None = None,
    working_dir: Path = Path("."),
    dat_filename: Path | str | None = None,
    atomic_femdvr_config: PseudoAtomicInput | None = None,
) -> tuple[PseudoAtomicResult, Path | None]:
    """Solve the pseudoatomic problem, then filter the dat and Bessel files.

    This wraps :func:`solve_pseudoatomic_problem` and additionally:
    - Regenerates the dat file to only include the desired orbitals.
    - Filters the Bessel HDF5 file to match the desired basis.

    Returns the solver result and the path to the filtered Bessel HDF5 file
    (or None if no Bessel file was produced).
    """
    upf_dict = UPFDict.from_upf(upf_path)
    atomic_basis = AtomicBasis(
        subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
    )
    if extension is not None:
        pseudo_basis = extension.extend(atomic_basis)
    else:
        pseudo_basis = atomic_basis.to_pseudoatomic_basis()

    if dat_filename is None:
        dat_filename = Path(upf_path.name).with_suffix(".dat")

    result = solve_pseudoatomic_problem(
        upf_path,
        rc=rc,
        ri_factor=ri_factor,
        extension=extension,
        working_dir=working_dir,
        dat_filename=dat_filename,
        atomic_femdvr_config=atomic_femdvr_config,
    )

    # Regenerate the dat file to only include the desired orbitals
    tmp_dat_file = max(
        working_dir.glob("*_qe.dat"),
        key=lambda f: f.stat().st_mtime,
    )
    x, r, l_values, orbitals = read_wannier90_dat_file(tmp_dat_file)
    selected_orbitals = [orbitals[i] for i in _find_matches(l_values, pseudo_basis.l_values)]

    # Write to the requested dat_filename
    write_wannier90_dat_file(
        working_dir / dat_filename,
        x,
        r,
        pseudo_basis.l_values,
        np.array(selected_orbitals),
    )

    # Filter the Bessel HDF5 file to only include the desired number of orbitals per l
    bessel_path = _filter_bessel_file(working_dir, pseudo_basis)

    # Rename the bessel file to match the dat filename for stable referencing
    if bessel_path is not None:
        stable_bessel_path = (working_dir / dat_filename).with_suffix(".h5")
        if stable_bessel_path != bessel_path:
            import shutil

            shutil.copy2(bessel_path, stable_bessel_path)
            bessel_path = stable_bessel_path

    return result, bessel_path


def _filter_bessel_file(
    working_dir: Path,
    pseudo_basis: PseudoatomicBasis,
) -> Path | None:
    """Rewrite the Bessel HDF5 file to only include the desired orbitals per l channel."""
    bessel_files = sorted(
        working_dir.glob("*_bessel.h5"),
        key=lambda f: f.stat().st_mtime,
    )
    if not bessel_files:
        return None
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

    return bessel_file


def compute_spread(
    dat_file: Path,
    atomic_basis: AtomicBasis,
) -> float:
    r"""Compute the wavefunction spread for the outermost subshell.

    Uses :math:`\Omega = \int dr\, r^4 |R_{nl}(r)|^2`.

    Parameters
    ----------
    dat_file
        Path to the Wannier90 .dat file containing the radial wavefunctions.
    atomic_basis
        The atomic basis, used to identify which orbital(s) belong to the outermost subshell.

    Returns
    -------
    float
        The spread averaged over the outermost subshell orbital(s).
    """
    _, r, l_values, orbitals = read_wannier90_dat_file(dat_file)
    r_arr = np.array(r)

    # Find the outermost subshell via Madelung ordering
    outermost: Subshell | None = None
    for subshell in ordered_subshells[::-1]:
        if subshell in atomic_basis:
            outermost = subshell
            break
    if outermost is None:
        raise ValueError("Basis set is empty.")

    # Find which orbital indices in the dat file correspond to the outermost subshell's l value.
    # The solver may produce more orbitals per l channel than are in the basis (nmax+1 per l),
    # so we use the basis orbital count to pick the correct one.
    target_l = outermost.l.value
    matching_indices = [i for i, l in enumerate(l_values) if l == target_l]
    if not matching_indices:
        raise ValueError(f"No orbital with l={target_l} found in dat file.")

    n_per_l = atomic_basis.to_pseudoatomic_basis().number_of_orbitals
    n_target = n_per_l.get(outermost.l, 0)
    if n_target == 0 or n_target > len(matching_indices):
        raise ValueError(
            f"Expected {n_target} orbital(s) with l={target_l}, found {len(matching_indices)}."
        )
    idx = matching_indices[n_target - 1]
    r_nl = orbitals[idx]

    # Omega = integral dr r^4 |R_nl(r)|^2
    integrand = r_arr**4 * r_nl**2
    return float(np.trapezoid(integrand, r_arr))


def get_outermost_wavefunction(
    dat_file: Path,
    atomic_basis: AtomicBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the outermost radial wavefunction from a dat file.

    Returns
    -------
    r, r_nl
        The radial grid and corresponding wavefunction values.
    """
    _, r, l_values, orbitals = read_wannier90_dat_file(dat_file)
    r_arr = np.array(r)

    outermost: Subshell | None = None
    for subshell in ordered_subshells[::-1]:
        if subshell in atomic_basis:
            outermost = subshell
            break
    if outermost is None:
        raise ValueError("Basis set is empty.")

    target_l = outermost.l.value
    matching_indices = [i for i, l in enumerate(l_values) if l == target_l]
    if not matching_indices:
        raise ValueError(f"No orbital with l={target_l} found in dat file.")

    n_per_l = atomic_basis.to_pseudoatomic_basis().number_of_orbitals
    n_target = n_per_l.get(outermost.l, 0)
    if n_target == 0 or n_target > len(matching_indices):
        raise ValueError(
            f"Expected {n_target} orbital(s) with l={target_l}, found {len(matching_indices)}."
        )
    idx = matching_indices[n_target - 1]

    return r_arr, orbitals[idx]


def _find_matches(values: list[int], desired_values: list[int]) -> list[int]:
    """Find the indices such that [values[i] for i in indices] == desired_values."""
    matches: list[int] = []
    for desired_value in desired_values:
        for i, value in enumerate(values):
            if value == desired_value and i not in matches:
                matches.append(i)
                break
    if len(matches) != len(desired_values):
        raise ValueError("Could not find all desired values in the provided list.")
    return matches
