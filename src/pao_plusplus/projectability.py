"""Projectability module for pao_plusplus."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from koopmans.utils import Spin
from ase_koopmans.calculators.calculator import CalculationFailed

from pao_plusplus.io import read_wannier90_amn_file
from pao_plusplus.workflows import run_wannierize_workflow


def projectability_score(amn: npt.NDArray[np.complex128], nw: int) -> float:
    """Calculate the projectability score from the amn array."""
    amn_abs2 = np.absolute(amn) ** 2
    projectability = np.sum(amn_abs2, axis=1)

    # The scoring function is the average of the projectability on the nw bands with the highest
    # projectability.
    sorted_projectability = np.sort(projectability, axis=0)
    _, nk = projectability.shape

    return np.sum(sorted_projectability[-nw:, :], dtype=float) / nw / nk


def compute_projectability(
    tag: str,
    pwi_file: Path,
    proj_dir: Path,
    working_dir: Path,
    pseudo_files: Iterator[Path],
    qe_bin: Path | None = None,
) -> float:
    """Compute the projectability of a set of PAOs against a set of bands.

    The PAOs come from the `.dat` file and the bands correspond to the system described in
    the `.pwi` file.
    """
    # Run the wannierize workflow up until the wannier90 step
    workflow_kwargs = {
        'pwi_file': pwi_file,
        'proj_dir': proj_dir,
        'w90_working_dir': working_dir / "w90" / tag / pwi_file.stem,
        'pw_working_dir': working_dir / "pw" / pwi_file.stem,
        'pseudo_files': pseudo_files,
        'qe_bin': qe_bin,
    }

    # Try davidson, then paro, then cg
    for diagonalization in ['david', 'paro', 'cg']:
        workflow_kwargs['diagonalization'] = diagonalization
        if diagonalization != 'david':
            workflow_kwargs['pw_working_dir'] = working_dir / "pw" / (pwi_file.stem + "-" + diagonalization)

        try:
            workflow = run_wannierize_workflow(**workflow_kwargs)
            break
        except CalculationFailed:
            # Re-attempt using the next diagonalisation method
            continue
    else:
        raise CalculationFailed("All diagonalisation methods failed.")

    # Extract the projectability from the amn file
    pw2wannier_step = workflow.steps[-1]
    amn_file = (
        workflow.absolute_directory
        / pw2wannier_step.directory
        / (pw2wannier_step.parameters.seedname + ".amn")
    )
    if not amn_file.exists():
        raise FileNotFoundError(f"Expected amn file at {amn_file} not found.")

    amn = read_wannier90_amn_file(amn_file)
    num_wann = workflow.projections.num_wann(Spin.NONE)
    return projectability_score(amn, num_wann)
