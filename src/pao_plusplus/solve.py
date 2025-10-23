"""Solve the pseudoatomic problem."""

from pathlib import Path
from atomic_femdvr.pseudo_atomic import solve_pseudo_atomic, PseudoAtomicInput
from upf_tools import UPFDict
import shutil
from pao_plusplus.io import read_wannier90_dat_file, write_wannier90_dat_file
import numpy as np
from pao_plusplus.basis import AtomicBasis, Subshell
from pao_plusplus.extend import BasisExtension


def solve_pseudoatomic_problem(upf_path: Path,
                               rc: float | None = None,
                               ri_factor: float | None = None,
                               extension: BasisExtension | None = None,
                               dat_file: Path | None = None) -> None:
    
    # Construct the atomic basis
    upf_dict = UPFDict.from_upf(upf_path)
    atomic_basis = AtomicBasis(subshells=[Subshell(n=chi['n'], l=chi['l']) for chi in upf_dict['pswfc']['chi']])

    # Extend the basis if requested
    if extension is not None:
        pseudo_basis = extension.extend(atomic_basis)
    else:
        pseudo_basis = atomic_basis.to_pseudoatomic_basis()
    
    # Construct the settings for atomic-femdvr
    config = PseudoAtomicInput(sysparams={'file_upf': str(upf_path),
                                          'nmax': pseudo_basis.n_max,
                                          'lmax': pseudo_basis.l_max.value,
                                          'element': upf_dict['header']['element']},
                               confinement={"type": "softstep",
                                            "polarization_mode": "softcoul",
                                            "Vbarrier": 10.0,})
    if rc is not None:
        config.confinement.rc = rc
    if ri_factor is not None:
        config.confinement.ri_factor = ri_factor

    # Solve the pseudoatomic problem
    dat_file = Path(upf_path.name).with_suffix('.dat') if dat_file is None else dat_file
    tmp_dir = Path('tmp')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    solve_pseudo_atomic(config, task_list=("scf", "optimize", "nscf"), export_dir=str(tmp_dir))

    # Regenerate the dat file to only include the desired orbitals
    tmp_dat_file = next(tmp_dir.glob("*.dat"))
    r, l_values, orbitals = read_wannier90_dat_file(tmp_dat_file)
    selected_orbitals = [orbitals[i] for i in _find_matches(l_values, pseudo_basis.l_values)]
    write_wannier90_dat_file(dat_file, r, pseudo_basis.l_values, np.array(selected_orbitals))

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





