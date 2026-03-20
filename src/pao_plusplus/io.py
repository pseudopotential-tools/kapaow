"""Input/output module for pao_plusplus."""

from pathlib import Path

import numpy as np
import numpy.typing as npt


def read_wannier90_dat_file(
    filename: Path,
) -> tuple[list[float], list[float], list[int], npt.NDArray[np.float64]]:
    """Read a projector file and return the radial grid, angular momentum values, and orbitals."""
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    l_values = [int(x) for x in lines[1].split()]

    x = [float(line.split()[0]) for line in lines[2:]]
    r = [float(line.split()[1]) for line in lines[2:]]

    orbitals = np.array([line.split()[2:] for line in lines[2:]], dtype=float).T

    return x, r, l_values, orbitals


def format_wannier90_dat(
    x: list[float],
    r: list[float],
    l_values: list[int],
    orbitals: npt.NDArray[np.float64],
) -> str:
    """Format Wannier90 .dat content as a string."""
    lines = [f"{len(r)} {len(l_values)}"]
    lines.append(" ".join(str(l) for l in l_values))
    for x_value, r_value, orbital_values in zip(x, r, orbitals.T, strict=True):
        lines.append(
            f"{x_value:11.8e} {r_value:11.8e} "
            + " ".join(f"{o:11.8e}" for o in orbital_values)
        )
    return "\n".join(lines) + "\n"


def write_wannier90_dat_file(
    filename: Path,
    x: list[float],
    r: list[float],
    l_values: list[int],
    orbitals: npt.NDArray[np.float64],
) -> None:
    """Write a Wannier90 .dat file given the radial grid, angular momentum values, and orbitals."""
    filename.write_text(format_wannier90_dat(x, r, l_values, orbitals), encoding="utf-8")


def read_wannier90_amn_file(filename: Path) -> npt.NDArray[np.complex128]:
    """Read a Wannier90 .amn file and return the amn array."""
    with open(filename, encoding="utf-8") as fd:
        lines = fd.readlines()

    nbnd, nk, nw = [int(x) for x in lines[1].split()]

    amn = np.zeros((nbnd, nw, nk), dtype=np.complex128)

    for line in lines[2:]:
        m, n, k, re, im = [float(x) for x in line.split()]
        amn[int(m) - 1, int(n) - 1, int(k) - 1] = re + 1j * im

    return amn
