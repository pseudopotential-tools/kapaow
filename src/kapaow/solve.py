"""Solve the pseudoatomic problem."""

from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
from atomic_femdvr.pseudo_atomic import PseudoAtomicInput, PseudoAtomicResult, solve_pseudo_atomic
from upf_tools import UPFDict

from kapaow.basis import (
    AngularMomentum,
    AtomicBasis,
    PseudoatomicBasis,
    Subshell,
    ordered_subshells,
)
from kapaow.extend import BasisExtension
from kapaow.io import read_wannier90_dat_file, write_wannier90_dat_file
from kapaow.pydantic import BaseModel

__all__: list[str] = [
    "ATOMIC_FEMDVR_PATCHES",
    "DEFAULT_RC_MAX",
    "DEFAULT_RC_MIN",
    "DEFAULT_RI_FACTOR_MAX",
    "DEFAULT_RI_FACTOR_MIN",
    "OrbitalEnergy",
    "PseudoAtomicInput",
    "PseudoAtomicResult",
    "_write_upf_for_basis",
    "compute_spread",
    "get_outermost_wavefunction",
    "read_femdvr_eigenvalues",
    "solve_and_export",
    "solve_pseudoatomic_problem",
]

DEFAULT_RC_MIN = 5.0
DEFAULT_RC_MAX = 15.0
DEFAULT_RI_FACTOR_MIN = 0.0
DEFAULT_RI_FACTOR_MAX = 0.95


class OrbitalEnergy(BaseModel):
    """Energy of a single radial orbital from a femdvr solve.

    Parameters
    ----------
    l
        Angular momentum channel.
    n_radial
        Zero-based index within the l channel (0 = lowest energy in that channel).
    energy
        Orbital energy in Hartree.
    """

    l: AngularMomentum
    n_radial: int
    energy: float


# Per-element overrides for the inner pseudoatomic SCF. Keep these alongside
# the solver they parameterise so consumers (pareto, rc_search, animate)
# don't have to import from heavier modules to reach them.
ATOMIC_FEMDVR_PATCHES: dict[str, PseudoAtomicInput] = {
    "Cr": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Cu": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Pd": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "At": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Sb": PseudoAtomicInput(dft={"alpha_mix": 0.3}),
    "Zn": PseudoAtomicInput(dft={"alpha_mix": 0.3}),
}


def read_femdvr_eigenvalues(result: PseudoAtomicResult) -> list[OrbitalEnergy]:
    """Extract per-orbital energies from a :class:`PseudoAtomicResult`.

    Prefers the ``"nscf"`` task (which uses the confinement potential) and
    falls back to ``"scf"`` if ``"nscf"`` is absent.  Channel keys in the
    result are string integers (``"0"`` = s, ``"1"`` = p, ...).

    Parameters
    ----------
    result
        Result object returned by :func:`solve_pseudoatomic_problem`.

    Returns
    -------
    list[OrbitalEnergy]
        One entry per (l, n_radial) pair, in l-then-n order.
    """
    task = "nscf" if "nscf" in result.eigenvalues else "scf"
    channel_energies = result.eigenvalues[task]

    orbitals: list[OrbitalEnergy] = []
    for l in AngularMomentum:
        key = str(l.value)
        if key not in channel_energies:
            break
        for n_radial, energy in enumerate(channel_energies[key]):
            orbitals.append(OrbitalEnergy(l=l, n_radial=n_radial, energy=energy))
    return orbitals


def _write_upf_for_basis(
    working_dir: Path,
    src_upf_path: Path,
    dst_upf_path: Path,
    pseudo_basis: PseudoatomicBasis,
    eigenvalues: list["OrbitalEnergy"],
) -> None:
    """Write a UPF file with ``PP_PSWFC`` replaced by *pseudo_basis*.

    The source UPF's mesh and every block other than ``PP_PSWFC`` and
    ``header.number_of_wfc`` is preserved verbatim. Each new ``PP_CHI``
    is built by reading the radial wavefunction from the latest
    ``*_qe.dat`` in *working_dir* and cubic-spline-interpolating it onto
    the source UPF's ``PP_MESH/PP_R`` grid; values outside the femdvr
    range are zero-padded.

    Pseudo-energies are converted from Hartree (femdvr) to Rydberg (UPF).

    Parameters
    ----------
    working_dir
        Directory holding the femdvr ``*_qe.dat`` output.
    src_upf_path
        Source UPF file, used as the template.
    dst_upf_path
        Destination path for the augmented UPF.
    pseudo_basis
        Basis whose :attr:`~PseudoatomicBasis.l_values` selects which
        orbitals are kept (in S, P, D, ... channel order).
    eigenvalues
        All :class:`OrbitalEnergy` records returned by
        :func:`read_femdvr_eigenvalues`. Used to assign each kept
        orbital its ``pseudo_energy``.
    """
    from scipy.interpolate import CubicSpline

    upf = UPFDict.from_upf(src_upf_path)
    upf_r = np.asarray(upf["mesh"]["r"], dtype=float)

    qe_dat_file = max(
        working_dir.glob("*_qe.dat"),
        key=lambda f: f.stat().st_mtime,
    )
    _, fem_r_list, fem_l_values, fem_orbitals = read_wannier90_dat_file(qe_dat_file)
    fem_r = np.asarray(fem_r_list, dtype=float)

    # Map (l, n_radial) -> femdvr-orbital index in the qe.dat file.
    fem_orbital_index: dict[tuple[int, int], int] = {}
    fem_count_per_l: dict[int, int] = {}
    for i, l in enumerate(fem_l_values):
        n_r = fem_count_per_l.get(l, 0)
        fem_orbital_index[(l, n_r)] = i
        fem_count_per_l[l] = n_r + 1

    # Map (l, n_radial) -> femdvr eigenvalue (Hartree).
    energy_lookup = {(o.l.value, o.n_radial): o.energy for o in eigenvalues}

    # Source-UPF baseline (n, l) per channel: used to keep baseline labels
    # consistent and to continue the n sequence for added orbitals.
    baseline_ns_per_l: dict[int, list[int]] = {}
    src_occ_lookup: dict[tuple[int, int], float] = {}
    for chi in upf["pswfc"]["chi"]:
        l_val, n_val = int(chi["l"]), int(chi["n"])
        baseline_ns_per_l.setdefault(l_val, []).append(n_val)
        src_occ_lookup[(l_val, n_val)] = float(chi.get("occupation", 0.0))
    for ns in baseline_ns_per_l.values():
        ns.sort()

    spdf = "SPDFG"
    new_chi: list[dict] = []
    seen_per_l: dict[int, int] = {}
    for l_int in pseudo_basis.l_values:
        n_radial = seen_per_l.get(l_int, 0)
        seen_per_l[l_int] = n_radial + 1

        if (l_int, n_radial) not in fem_orbital_index:
            raise ValueError(
                f"femdvr output is missing orbital (l={l_int}, n_radial={n_radial}); "
                f"available: {sorted(fem_orbital_index.keys())}"
            )
        wf = np.asarray(fem_orbitals[fem_orbital_index[(l_int, n_radial)]], dtype=float)
        spline = CubicSpline(fem_r, wf, extrapolate=False)
        wf_on_upf = spline(upf_r)
        wf_on_upf = np.where(np.isnan(wf_on_upf), 0.0, wf_on_upf)

        energy_ha = energy_lookup.get((l_int, n_radial), 0.0)
        pseudo_energy_ry = 2.0 * energy_ha

        baseline_ns = baseline_ns_per_l.get(l_int, [])
        if n_radial < len(baseline_ns):
            n_q = baseline_ns[n_radial]
        elif baseline_ns:
            n_q = baseline_ns[-1] + (n_radial - len(baseline_ns) + 1)
        else:
            # No baseline orbital in this channel: start from n = l + 1
            # (the lowest principal quantum number permitting this l).
            n_q = l_int + 1 + n_radial

        new_chi.append(
            {
                "index": len(new_chi) + 1,
                "occupation": src_occ_lookup.get((l_int, n_q), 0.0),
                "pseudo_energy": pseudo_energy_ry,
                "label": f"{n_q}{spdf[l_int]}",
                "l": l_int,
                "n": n_q,
                "content": wf_on_upf,
            }
        )

    upf["pswfc"]["chi"] = new_chi
    upf["header"]["number_of_wfc"] = len(new_chi)
    upf.to_upf(dst_upf_path)


def _write_dat_for_basis(
    working_dir: Path,
    dat_filename: Path,
    pseudo_basis: PseudoatomicBasis,
) -> None:
    """Write a filtered Wannier90 dat file for the given basis.

    Finds the most recently modified ``*_qe.dat`` in *working_dir*, reads
    it, selects only the orbitals matching *pseudo_basis*, and writes the
    result to *dat_filename*.

    Parameters
    ----------
    working_dir
        Directory containing the raw ``*_qe.dat`` output from atomic-femdvr.
    dat_filename
        Destination path for the filtered dat file.
    pseudo_basis
        Basis whose :attr:`~PseudoatomicBasis.l_values` determines which
        orbitals are kept.
    """
    tmp_dat_file = max(
        working_dir.glob("*_qe.dat"),
        key=lambda f: f.stat().st_mtime,
    )
    x, r, l_values, orbitals = read_wannier90_dat_file(tmp_dat_file)
    selected_orbitals = [orbitals[i] for i in _find_matches(l_values, pseudo_basis.l_values)]
    write_wannier90_dat_file(
        dat_filename,
        x,
        r,
        pseudo_basis.l_values,
        np.array(selected_orbitals),
    )


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

    # Remove any pre-existing hdf5 files in the working dir so a stale
    # density/potential file from a previous run isn't picked up.
    for hdf5_file in working_dir.glob("*_density_potential.h5"):
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
    _write_dat_for_basis(working_dir, working_dir / dat_filename, pseudo_basis)

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


def get_outermost_wavefunction(
    dat_file: Path,
    atomic_basis: AtomicBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the outermost radial wavefunction from a Wannier90 .dat file.

    The "outermost" orbital is the highest-energy occupied subshell of
    *atomic_basis* under Madelung ordering. The solver may produce more
    orbitals per l channel than the basis declares (``nmax+1`` per l),
    so we use the basis orbital count to pick the correct radial
    function for that channel.

    Parameters
    ----------
    dat_file
        Path to the Wannier90 .dat file containing the radial wavefunctions.
    atomic_basis
        Atomic basis identifying which orbital is the outermost.

    Returns
    -------
    r, r_nl
        The radial grid and the corresponding radial wavefunction.
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


def compute_spread(
    dat_file: Path,
    atomic_basis: AtomicBasis,
) -> float:
    r"""Compute the spatial spread of the outermost radial wavefunction.

    Uses :math:`\Omega = \int dr\, r^4 |R_{nl}(r)|^2`.

    Parameters
    ----------
    dat_file
        Path to the Wannier90 .dat file containing the radial wavefunctions.
    atomic_basis
        Atomic basis, used to identify the outermost subshell via
        :func:`get_outermost_wavefunction`.

    Returns
    -------
    float
        The spread of the outermost orbital.
    """
    r_arr, r_nl = get_outermost_wavefunction(dat_file, atomic_basis)
    integrand = r_arr**4 * r_nl**2
    return float(np.trapezoid(integrand, r_arr))


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
