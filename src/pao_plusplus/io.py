"""Input/output module for pao_plusplus."""

import numpy as np
import numpy.typing as npt
from pathlib import Path

def read_wannier90_dat_file(filename: Path) -> tuple[list[float], list[int], npt.NDArray[np.float64]]:
    """Read a Wannier90 .dat file and return the radial grid, angular momentum values, and orbitals."""
    with open(filename, "r") as f:
        lines = f.readlines()
    
    l_values = [int(x) for x in lines[1].split()]

    r = [float(line.split()[1]) for line in lines[2:]]

    orbitals = np.array([line.split()[2:] for line in lines[2:]], dtype=float).T

    return r, l_values, orbitals

def write_wannier90_dat_file(filename: Path, r: list[float], l_values: list[int], orbitals: npt.NDArray[np.float64]) -> None:
    """Write a Wannier90 .dat file given the radial grid, angular momentum values, and orbitals."""
    with open(filename, "w") as f:
        f.write(f"{len(r)}\n")
        f.write(" ".join(str(l) for l in l_values) + "\n")
        for i in range(len(r)):
            f.write(f"{i+1} {r[i]:.8e} " + " ".join(f"{orbital[i]:.8e}" for orbital in orbitals) + "\n")
