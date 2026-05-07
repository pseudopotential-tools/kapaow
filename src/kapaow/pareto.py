"""Pareto front analysis of spread vs energy shift for varying rc and ri_factor."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from atomic_femdvr.pseudo_atomic import ConvergenceError
from upf_tools import UPFDict

from kapaow.basis import AngularMomentum, AtomicBasis
from kapaow.extend import BasisExtension, BasisExtensionViaAddition
from kapaow.plotting import COLORMAP, REVTEX_COLUMN_WIDTH
from kapaow.solve import (
    ATOMIC_FEMDVR_PATCHES,
    DEFAULT_RC_MAX,
    DEFAULT_RC_MIN,
    DEFAULT_RI_FACTOR_MAX,
    DEFAULT_RI_FACTOR_MIN,
    compute_spread,
    get_outermost_wavefunction,
    solve_pseudoatomic_problem,
)

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "compute_pareto_front",
    "dump_pareto_json",
    "extract_pareto_front",
    "find_kink_triplets",
    "plot_pareto",
]


def _evaluate_point(
    upf_path: Path,
    rc: float,
    ri_factor: float,
    extension: BasisExtension | None,
    element: str,
    atomic_basis: AtomicBasis,
    original_basis: AtomicBasis,
    working_dir: Path,
) -> tuple[float, float, Path] | None:
    """Evaluate a single (rc, ri_factor) point and return (spread, max_shift) or None.

    The spread is computed from ``atomic_basis`` (which may include added
    shells).  The maximum energy shift is taken over only the (n, l) pairs
    present in ``original_basis``.
    """
    dat_filename = f"{element}_rc_{rc:.4f}_ri_{ri_factor:.4f}.dat"
    point_dir = working_dir / f"rc_{rc:.4f}_ri_{ri_factor:.4f}"
    point_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = solve_pseudoatomic_problem(
            upf_path,
            rc=rc,
            ri_factor=ri_factor,
            extension=extension,
            working_dir=point_dir,
            dat_filename=dat_filename,
            atomic_femdvr_config=ATOMIC_FEMDVR_PATCHES.get(element),
            output_wfc_bessel=False,
        )
    except ConvergenceError:
        logger.info(
            "  SCF did not converge for rc=%.4f, ri_factor=%.4f",
            rc,
            ri_factor,
        )
        return None

    energy_shifts = result.energy_shifts
    if energy_shifts is None:
        logger.info(
            "  No energy shifts for rc=%.4f, ri_factor=%.4f",
            rc,
            ri_factor,
        )
        return None

    # Collect shifts for only the original basis orbitals
    original_n_per_l = original_basis.to_pseudoatomic_basis().number_of_orbitals
    original_shifts = []
    for l_str, l_shifts in energy_shifts.items():
        l_val = int(l_str)
        n_orig = original_n_per_l.get(AngularMomentum(l_val), 0)
        original_shifts.extend(l_shifts[:n_orig])
    if not original_shifts:
        return None
    max_shift = max(abs(e) for e in original_shifts)

    dat_files = list(point_dir.glob(f"{element}_*_qe.dat"))
    if not dat_files:
        return None
    dat_file = max(dat_files, key=lambda f: f.stat().st_mtime)

    spread = compute_spread(dat_file, atomic_basis)
    return spread, max_shift, dat_file


def _accumulate_result(
    out: tuple[float, float, Path] | None,
    rc: float,
    ri_factor: float,
    spreads: list[float],
    max_energy_shifts: list[float],
    dat_files: list[Path],
    metadata: list[dict[str, Any]],
    refined: bool = False,
) -> None:
    """Append a successful evaluation result to the accumulator lists."""
    if out is None:
        return
    spread, max_shift, dat_file = out
    spreads.append(spread)
    max_energy_shifts.append(max_shift)
    dat_files.append(dat_file)
    metadata.append({"rc": rc, "ri_factor": ri_factor, "refined": refined})
    if refined:
        logger.info(
            "    -> spread=%.4f, max_energy_shift=%.6f",
            spread,
            max_shift,
        )
    else:
        logger.info(
            "  rc=%.4f, ri_factor=%.4f: spread=%.4f, max_energy_shift=%.6f",
            rc,
            ri_factor,
            spread,
            max_shift,
        )


def _annotate_confinement(
    metadata: list[dict[str, Any]],
    dat_files: list[Path],
    atomic_basis: AtomicBasis,
) -> None:
    """Annotate each metadata entry with whether its outermost wfc is modified by confinement."""
    ref_idx = max(
        range(len(metadata)),
        key=lambda i: (metadata[i]["rc"], metadata[i]["ri_factor"]),
    )
    ref_r, ref_wfc = get_outermost_wavefunction(dat_files[ref_idx], atomic_basis)
    for i in range(len(metadata)):
        r_i, wfc_i = get_outermost_wavefunction(dat_files[i], atomic_basis)
        ref_wfc_interp = np.interp(r_i, ref_r, ref_wfc)
        overlap = float(np.trapezoid(wfc_i * ref_wfc_interp * r_i**2, r_i))
        norm_i = float(np.trapezoid(wfc_i**2 * r_i**2, r_i))
        norm_ref = float(np.trapezoid(ref_wfc_interp**2 * r_i**2, r_i))
        if norm_i > 0 and norm_ref > 0:
            fidelity = overlap**2 / (norm_i * norm_ref)
            metadata[i]["modified_by_confinement"] = fidelity < 0.9999
        else:
            metadata[i]["modified_by_confinement"] = True


def compute_pareto_front(
    upf_path: Path,
    extension: BasisExtension | None = None,
    rc_values: list[float] | None = None,
    ri_factor_values: list[float] | None = None,
    working_dir: Path = Path("tmp/optimize/spread"),
    loglog: bool = False,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    """Scan rc and ri_factor and collect spread and max energy shift for each.

    Parameters
    ----------
    upf_path
        Path to the UPF pseudopotential file.
    extension
        Optional basis extension.
    rc_values
        List of rc values to scan.
    ri_factor_values
        List of ri_factor values to scan.
    working_dir
        Directory for intermediate files.

    Returns
    -------
    spreads, max_energy_shifts, metadata
        Lists of spread values, max absolute energy shifts, and per-point metadata.
    """
    if rc_values is None:
        rc_values = np.linspace(DEFAULT_RC_MIN, DEFAULT_RC_MAX, 11).tolist()
    if ri_factor_values is None:
        ri_factor_values = np.linspace(
            DEFAULT_RI_FACTOR_MIN,
            DEFAULT_RI_FACTOR_MAX,
            20,
        ).tolist()

    upf_dict = UPFDict.from_upf(upf_path)
    element = upf_dict["header"]["element"].strip()
    original_basis = AtomicBasis.from_upf(upf_path)
    if extension is not None and isinstance(extension, BasisExtensionViaAddition):
        atomic_basis = extension.extend_atomic(original_basis)
    else:
        atomic_basis = original_basis

    working_dir.mkdir(parents=True, exist_ok=True)

    spreads: list[float] = []
    max_energy_shifts: list[float] = []
    dat_files: list[Path] = []
    metadata: list[dict[str, Any]] = []

    # Initial grid scan
    for rc in rc_values:
        for ri_factor in ri_factor_values:
            out = _evaluate_point(
                upf_path,
                rc,
                ri_factor,
                extension,
                element,
                atomic_basis,
                original_basis,
                working_dir,
            )
            _accumulate_result(
                out,
                rc,
                ri_factor,
                spreads,
                max_energy_shifts,
                dat_files,
                metadata,
            )

    # Refinement pass: for each kink, mix the two endpoints for a new trial
    triplets = find_kink_triplets(spreads, max_energy_shifts, loglog=loglog)
    if triplets:
        logger.info("Refining %d kink(s)...", len(triplets))
    for i_a, _i_b, i_c in triplets:
        rc_new = 0.5 * (metadata[i_a]["rc"] + metadata[i_c]["rc"])
        ri_new = 0.5 * (metadata[i_a]["ri_factor"] + metadata[i_c]["ri_factor"])
        logger.info(
            "  Trial: rc=%.4f, ri_factor=%.4f (mix of rc=%.4f/%.4f, ri=%.4f/%.4f)",
            rc_new,
            ri_new,
            metadata[i_a]["rc"],
            metadata[i_c]["rc"],
            metadata[i_a]["ri_factor"],
            metadata[i_c]["ri_factor"],
        )
        out = _evaluate_point(
            upf_path,
            rc_new,
            ri_new,
            extension,
            element,
            atomic_basis,
            original_basis,
            working_dir,
        )
        _accumulate_result(
            out,
            rc_new,
            ri_new,
            spreads,
            max_energy_shifts,
            dat_files,
            metadata,
            refined=True,
        )

    # Determine which points have outermost wavefunctions modified by confinement.
    _annotate_confinement(metadata, dat_files, atomic_basis)

    return spreads, max_energy_shifts, metadata


def dump_pareto_json(
    spreads: list[float],
    max_energy_shifts: list[float],
    metadata: list[dict[str, Any]],
    path: Path,
    upf_path: Path | None = None,
) -> None:
    """Write all grid points to a JSON file, with a ``pareto`` flag for front membership."""
    pareto_idx_set = set(extract_pareto_front(spreads, max_energy_shifts))
    points = [
        {
            "rc": metadata[i]["rc"],
            "ri_factor": metadata[i]["ri_factor"],
            "spread": spreads[i],
            "max_energy_shift": max_energy_shifts[i],
            "modified_by_confinement": metadata[i]["modified_by_confinement"],
            "pareto": i in pareto_idx_set,
        }
        for i in range(len(spreads))
    ]
    output: dict[str, Any] = {}
    if upf_path is not None:
        output["upf_path"] = str(upf_path)
    output["points"] = points
    n_pareto = sum(1 for p in points if p["pareto"])
    path.write_text(json.dumps(output, indent=2))
    logger.info(
        "Grid (%d points, %d on Pareto front) saved to %s",
        len(points),
        n_pareto,
        path,
    )


def find_kink_triplets(
    spreads: list[float],
    max_energy_shifts: list[float],
    threshold: float = np.radians(30),
    gap_factor: float = 2.0,
    loglog: bool = False,
) -> list[tuple[int, int, int]]:
    """Find Pareto front triplets where the middle point is a kink.

    For each triplet (a, b, c) of adjacent Pareto points sorted by spread,
    compute the turning angle between segments (a→b) and (b→c).  Only
    flag points where the front bows outward (negative derivative change,
    detected via cross product sign), where the turning angle exceeds
    *threshold* (in radians), and where the distance between a and c
    exceeds *gap_factor* times the average spacing between adjacent
    Pareto points.

    When *loglog* is True the geometry is computed in log-space so that
    the detected kinks match what is visible on a log-log plot.

    Returns a list of (i_a, i_b, i_c) index triplets.
    """
    pareto_idx = extract_pareto_front(spreads, max_energy_shifts)
    if len(pareto_idx) < 3:
        return []

    # Work in log-space when the plot uses log axes
    if loglog:
        xs = [np.log(s) for s in spreads]
        ys = [np.log(e) for e in max_energy_shifts]
    else:
        xs = list(spreads)
        ys = list(max_energy_shifts)

    pareto_sorted = sorted(pareto_idx, key=lambda i: xs[i])
    n_pareto = len(pareto_sorted)

    # Average spacing between adjacent Pareto points
    total_arc = 0.0
    for j in range(n_pareto - 1):
        dx = xs[pareto_sorted[j + 1]] - xs[pareto_sorted[j]]
        dy = ys[pareto_sorted[j + 1]] - ys[pareto_sorted[j]]
        total_arc += np.sqrt(dx**2 + dy**2)
    avg_spacing = total_arc / (n_pareto - 1)

    triplets: list[tuple[int, int, int]] = []

    for k in range(1, n_pareto - 1):
        i_a, i_b, i_c = pareto_sorted[k - 1], pareto_sorted[k], pareto_sorted[k + 1]

        # Vectors along the two segments
        dx_ab = xs[i_b] - xs[i_a]
        dy_ab = ys[i_b] - ys[i_a]
        dx_bc = xs[i_c] - xs[i_b]
        dy_bc = ys[i_c] - ys[i_b]

        len_ab = np.sqrt(dx_ab**2 + dy_ab**2)
        len_bc = np.sqrt(dx_bc**2 + dy_bc**2)
        if len_ab == 0 or len_bc == 0:
            continue

        # Skip if a and c are too close (less than gap_factor × average spacing)
        dx_ac = xs[i_c] - xs[i_a]
        dy_ac = ys[i_c] - ys[i_a]
        len_ac = np.sqrt(dx_ac**2 + dy_ac**2)
        if len_ac < gap_factor * avg_spacing:
            continue

        # Cross product (a→b) × (b→c): negative (into page) means a
        # right turn, i.e. the front bows outward away from the origin
        cross = dx_ab * dy_bc - dy_ab * dx_bc
        if cross < 0:
            # Turning angle from the dot product (scale-invariant)
            dot = dx_ab * dx_bc + dy_ab * dy_bc
            cos_angle = np.clip(dot / (len_ab * len_bc), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle >= threshold:
                triplets.append((i_a, i_b, i_c))

    return triplets


def extract_pareto_front(
    spreads: list[float],
    max_energy_shifts: list[float],
) -> list[int]:
    """Return indices of points on the Pareto front (minimizing both quantities).

    A point is Pareto-optimal if no other point has both a smaller spread
    and a smaller max energy shift.
    """
    n = len(spreads)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if spreads[j] <= spreads[i] and max_energy_shifts[j] <= max_energy_shifts[i]:
                if spreads[j] < spreads[i] or max_energy_shifts[j] < max_energy_shifts[i]:
                    is_pareto[i] = False
                    break
    return [i for i in range(n) if is_pareto[i]]


_SIESTA_TOLERANCE_HA = 0.01  # 0.02 Ry in Hartree


def plot_pareto(
    json_path: Path,
    filename: Path | None = None,
    loglog: bool = False,
    logy: bool = False,
) -> None:
    """Plot spread vs max energy shift with the Pareto front highlighted.

    Reads all data from the JSON file produced by :func:`dump_pareto_json`.
    """
    with open(json_path) as f:
        raw = json.load(f)
    points = raw["points"]

    spreads_arr = np.array([p["spread"] for p in points])
    shifts_arr = np.array([p["max_energy_shift"] for p in points])
    rc_arr = np.array([p["rc"] for p in points])
    ri_factor_arr = np.array([p["ri_factor"] for p in points])
    r_half_arr = (ri_factor_arr + 1) / 2 * rc_arr
    pareto_mask = np.array([p["pareto"] for p in points])

    fig, ax = plt.subplots(
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.75), layout="tight"
    )

    point_size = 5
    sc = ax.scatter(
        spreads_arr,
        shifts_arr,
        c=r_half_arr,
        cmap=COLORMAP,
        s=point_size,
        zorder=2,
    )
    fig.colorbar(sc, ax=ax, label=r"$r_{1/2}$ (Bohr)")

    # Pareto front: truncate at the first point that uses rc_max, since
    # the front beyond that is uncertain (a larger rc might dominate).
    pareto_indices = np.where(pareto_mask)[0]
    pareto_spreads = spreads_arr[pareto_indices]
    order = np.argsort(pareto_spreads)
    sorted_pareto = pareto_indices[order]

    rc_max = rc_arr.max()
    cutoff = len(sorted_pareto)
    for k, idx in enumerate(sorted_pareto):
        if rc_arr[idx] == rc_max:
            cutoff = k + 1
            break
    trusted_pareto = sorted_pareto[:cutoff]

    # Line through trusted portion of the front
    ax.plot(
        spreads_arr[trusted_pareto],
        shifts_arr[trusted_pareto],
        "-",
        color="crimson",
        linewidth=0.5,
        zorder=3,
    )
    # Coloured dots with crimson edge on Pareto points
    cmap = plt.get_cmap(COLORMAP)
    norm = mcolors.Normalize(vmin=r_half_arr.min(), vmax=r_half_arr.max())
    ax.scatter(
        spreads_arr[sorted_pareto],
        shifts_arr[sorted_pareto],
        s=point_size,
        c=cmap(norm(r_half_arr[sorted_pareto])),
        edgecolors="crimson",
        linewidths=0.4,
        zorder=4,
    )

    # Reference line at 0.02 Ry (default energy shift tolerance in SIESTA)
    ax.axhline(_SIESTA_TOLERANCE_HA, color="grey", linewidth=0.5, linestyle="--", zorder=1)

    if loglog or logy:
        import matplotlib.ticker as mticker

        ax.set_yscale("log")
        ax.set_ylim(bottom=_SIESTA_TOLERANCE_HA / 10, top=1.0)

    if loglog or logy:
        # Set xmin to 0.9 * spread at the Pareto front where energy shift = 1 Ha
        pareto_at_top = [spreads_arr[idx] for idx in sorted_pareto if shifts_arr[idx] <= 1.0]
        if pareto_at_top:
            ax.set_xlim(left=0.9 * min(pareto_at_top))

        # Set xmax to 1.1 * spread of the Pareto front point at tolerance / 10
        y_cutoff = _SIESTA_TOLERANCE_HA / 10
        pareto_at_cutoff = [
            spreads_arr[idx] for idx in sorted_pareto if shifts_arr[idx] >= y_cutoff
        ]
        if pareto_at_cutoff:
            ax.set_xlim(right=1.1 * max(pareto_at_cutoff))

    if loglog:
        ax.set_xscale("log")

        fmt = mticker.FuncFormatter(lambda x, _: f"{x:g}")
        ax.xaxis.set_minor_formatter(fmt)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=5))

    ax.set_xlabel(r"spread of added PAO (Bohr$^2$)")
    ax.set_ylabel("maximum energy shift (Ha)")

    if filename is not None:
        from kapaow.plotting import savefig

        savefig(plt, filename)
        plt.close(fig)
    else:
        plt.show()
