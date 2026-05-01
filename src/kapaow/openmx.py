"""Read and convert OpenMX pseudoatomic orbital (.pao) files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

L_LETTERS = "spdfgh"


@dataclass
class OpenMXPAO:
    """Parsed contents of an OpenMX .pao file.

    Attributes
    ----------
    lmax
        Maximum angular momentum quantum number + 1 (i.e. number of l channels).
    num_mul
        Number of multiplicities (orbitals) per l channel.
    x
        Logarithmic grid, shape ``(num_grid,)``.
    r
        Radial grid ``exp(x)``, shape ``(num_grid,)``.
    orbitals
        Per-channel orbital data.  ``orbitals[l]`` has shape
        ``(num_grid, num_mul)``.
    """

    lmax: int
    num_mul: int
    x: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    orbitals: dict[int, npt.NDArray[np.float64]] = field(default_factory=dict)


def read_openmx_pao(path: Path) -> OpenMXPAO:
    """Parse an OpenMX ``.pao`` file.

    Parameters
    ----------
    path
        Path to the ``.pao`` file.

    Returns
    -------
    OpenMXPAO
        Parsed orbital data.
    """
    text = path.read_text()

    lmax = _extract_param(text, "PAO.Lmax")
    num_mul = _extract_param(text, "PAO.Mul")

    orbitals: dict[int, npt.NDArray[np.float64]] = {}
    x: npt.NDArray[np.float64] | None = None
    r: npt.NDArray[np.float64] | None = None

    for l_val in range(lmax):
        block = _extract_tag(text, f"pseudo.atomic.orbitals.L={l_val}")
        data = np.loadtxt(block.splitlines())
        if x is None:
            x = data[:, 0]
            r = data[:, 1]
        # Columns 2 onwards are the orbital multiplicities
        orbitals[l_val] = data[:, 2:]

    if x is None or r is None:
        raise ValueError("No orbital data found in file.")
    return OpenMXPAO(lmax=lmax, num_mul=num_mul, x=x, r=r, orbitals=orbitals)


def parse_select(select: list[str]) -> list[int]:
    """Parse a selection list like ``["s", "s", "p", "d"]`` into counts per l channel.

    Each entry is an angular momentum letter (s, p, d, f, g, h).
    The count per channel is the number of occurrences.

    Returns
    -------
    list[int]
        Counts per l channel, indexed by l.  For example ``["s", "s", "p", "d"]``
        returns ``[2, 1, 1]``.
    """
    counts: dict[int, int] = {}
    for ch in select:
        ch = ch.lower()
        if ch not in L_LETTERS:
            msg = f"Unknown angular momentum letter '{ch}' in select. Use: {L_LETTERS}"
            raise ValueError(msg)
        l_val = L_LETTERS.index(ch)
        counts[l_val] = counts.get(l_val, 0) + 1

    if not counts:
        raise ValueError("select list is empty.")

    max_l = max(counts)
    return [counts.get(l, 0) for l in range(max_l + 1)]


def convert_to_wannier90(
    pao: OpenMXPAO,
    selected: list[int] | None = None,
) -> tuple[list[float], list[float], list[int], npt.NDArray[np.float64]]:
    """Convert OpenMX PAO data to Wannier90 .dat format.

    Parameters
    ----------
    pao
        Parsed OpenMX PAO data.
    selected
        Number of orbitals to include per l channel.  If ``None``, include
        all available orbitals (one per channel).

    Returns
    -------
    x, r, l_values, orbitals
        Tuple matching :func:`~kapaow.io.write_wannier90_dat_file`.
    """
    if selected is None:
        # Default: one orbital per l channel
        selected = [1] * pao.lmax

    # Validate
    for l_val, count in enumerate(selected):
        if l_val >= pao.lmax:
            available_channels = L_LETTERS[: pao.lmax]
            raise ValueError(
                f"Requested l={l_val} ({L_LETTERS[l_val]}) but file only has "
                f"channels up to l={pao.lmax - 1} ({available_channels})."
            )
        available = pao.orbitals[l_val].shape[1]
        if count > available:
            raise ValueError(
                f"Requested {count} {L_LETTERS[l_val]}-orbitals but only {available} available."
            )

    # Build output arrays
    l_values: list[int] = []
    orbital_columns: list[npt.NDArray[np.float64]] = []
    for l_val, count in enumerate(selected):
        for i in range(count):
            l_values.append(l_val)
            orbital_columns.append(pao.orbitals[l_val][:, i])

    orbitals = np.array(orbital_columns)  # shape (num_orbitals, num_grid)

    return pao.x.tolist(), pao.r.tolist(), l_values, orbitals


def pao_to_bessel(
    pao: OpenMXPAO,
    output_path: Path,
    selected: list[int] | None = None,
    qmax: float = 50.0,
    nq: int = 201,
    bessel_npoints: int = 41,
) -> Path:
    """Convert OpenMX PAO radial orbitals to a Bessel HDF5 file.

    Performs a spherical Bessel transform of each selected orbital using
    the ``atomic_femdvr`` library, producing the ``.h5`` format expected
    by ``qe_wavefunctions``.

    Parameters
    ----------
    pao
        Parsed OpenMX PAO data.
    output_path
        Where to write the Bessel HDF5 file.
    selected
        Orbitals per l channel (from :func:`parse_select`).  If ``None``,
        one orbital per channel.
    qmax
        Maximum q value for the Bessel grid (Bohr^-1).
    nq
        Number of q-grid points.
    bessel_npoints
        Quadrature points per finite element for the Bessel integral.

    Returns
    -------
    Path
        The path to the written ``.h5`` file.
    """
    from atomic_femdvr.bessel_transform import bessel_integral
    from atomic_femdvr.femdvr import FEDVR_Basis
    from atomic_femdvr.projector_output import write_bessel_hdf5
    from scipy.interpolate import interp1d

    _, r_list, l_values, orbitals = convert_to_wannier90(pao, selected)
    r = np.array(r_list)

    # Count orbitals per l channel to determine lmax and nmax
    l_counts: dict[int, int] = {}
    for l_val in l_values:
        l_counts[l_val] = l_counts.get(l_val, 0) + 1
    lmax = max(l_counts)
    nmax = max(l_counts.values()) - 1

    # Set up FEDVR basis covering the radial domain
    r_min = max(r[0], 1e-6)
    r_max = r[-1]
    ne = 20
    ng = 11
    xp = np.linspace(r_min, r_max, ne + 1)
    basis = FEDVR_Basis(ne=ne, ng=ng, xp=xp)

    # Set up q-grid
    qgrid = np.linspace(0.0, qmax, nq)

    # Bessel-transform each orbital
    phi_bessel = np.zeros((lmax + 1, nmax + 1, nq))
    l_index: dict[int, int] = {}  # tracks next n index per l
    for i, l_val in enumerate(l_values):
        n = l_index.get(l_val, 0)
        l_index[l_val] = n + 1

        # Interpolate the radial orbital onto the FEDVR grid
        f_interp = interp1d(r, orbitals[i], kind="cubic", bounds_error=False, fill_value=0.0)
        r_fedvr = basis.get_gridpoints()
        phi_on_grid = f_interp(r_fedvr)

        phi_bessel[l_val, n, :] = bessel_integral(
            basis, l_val, 1, qgrid, phi_on_grid, npoints=bessel_npoints, method="simpson"
        )

    write_bessel_hdf5(str(output_path.parent), output_path.stem, "", phi_bessel, qgrid)

    # write_bessel_hdf5 uses f"{elem}_{tag}_bessel.h5" naming; rename to output_path
    generated = output_path.parent / f"{output_path.stem}__bessel.h5"
    if generated.exists() and generated != output_path:
        generated.rename(output_path)
    elif not output_path.exists():
        # Try without the underscore separator if tag is empty
        for candidate in output_path.parent.glob(f"{output_path.stem}*bessel.h5"):
            candidate.rename(output_path)
            break

    return output_path


def _extract_param(text: str, key: str) -> int:
    """Extract an integer parameter from OpenMX file text."""
    for line in text.splitlines():
        if key in line:
            # e.g. "PAO.Lmax   3" or "grid.num.output  500  # default=2000"
            after_key = line[line.index(key) + len(key) :]
            # Strip comments
            if "#" in after_key:
                after_key = after_key[: after_key.index("#")]
            return int(after_key.strip())
    raise ValueError(f"'{key}' not found in file.")


def _extract_tag(text: str, tag: str) -> str:
    """Extract the text between ``<tag`` and ``tag>`` markers."""
    pattern = rf"<{re.escape(tag)}\s*\n(.*?)\n\s*{re.escape(tag)}>"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise ValueError(f"Tag '{tag}' not found in file.")
    return match.group(1)
