"""Cross-element analysis of Pareto scan grid data."""

import json
import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from pao_plusplus.plotting import COLORMAP, REVTEX_COLUMN_WIDTH

logger = logging.getLogger(__name__)

_DEFAULT_RI_FACTORS = [0.5, 0.6, 0.7, 0.8, 0.9]


def _load_element_grid_data(grid_directory: Path) -> dict[str, list[dict]]:
    """Load per-element JSON grid data from a directory."""
    element_data: dict[str, list[dict]] = {}
    for json_file in sorted(grid_directory.glob("*.json")):
        element = json_file.stem
        with open(json_file) as f:
            raw = json.load(f)
        element_data[element] = raw["points"]
    return element_data


def _compute_rc_fractions(
    element_data: dict[str, list[dict]],
    ri_factor: float,
    ri_factor_tol: float,
    threshold_ha: float,
) -> tuple[list[float], list[float]]:
    """Compute the cumulative fraction of elements below threshold for a given ri_factor.

    Returns (rc_values, fractions) sorted by rc, or empty lists if no data matches.
    """
    element_rc_ok: dict[str, dict[float, bool]] = {}
    all_rc_values: set[float] = set()

    for element, points in element_data.items():
        matching = [p for p in points if abs(p["ri_factor"] - ri_factor) < ri_factor_tol]

        rc_ok: dict[float, bool] = {}
        for p in matching:
            rc = p["rc"]
            all_rc_values.add(rc)
            ok = p["max_energy_shift"] < threshold_ha
            if rc in rc_ok:
                rc_ok[rc] = rc_ok[rc] and ok
            else:
                rc_ok[rc] = ok

        if rc_ok:
            element_rc_ok[element] = rc_ok

    if not element_rc_ok:
        return [], []

    rc_plot = []
    fractions = []
    for rc in sorted(all_rc_values):
        n_with_data = sum(1 for el_data in element_rc_ok.values() if rc in el_data)
        if n_with_data == 0:
            continue
        n_ok = sum(1 for el_data in element_rc_ok.values() if el_data.get(rc, False))
        rc_plot.append(rc)
        fractions.append(n_ok / n_with_data)

    return rc_plot, fractions


def plot_cumulative_below_threshold(
    grid_directory: Path,
    threshold_ry: float = 0.02,
    ri_factors: list[float] = _DEFAULT_RI_FACTORS,
    ri_factor_tol: float = 0.01,
    filename: Path | None = None,
) -> None:
    """Plot the cumulative fraction of elements below the energy shift threshold vs rc.

    One line is drawn per ri_factor value.

    Parameters
    ----------
    grid_directory
        Directory containing per-element JSON files with grid data.
    threshold_ry
        Energy shift threshold in Rydberg.
    ri_factors
        The ri_factor values to plot.
    ri_factor_tol
        Tolerance for matching ri_factor values in the grid.
    filename
        Output file path; if None, show interactively.
    """
    threshold_ha = threshold_ry / 2

    element_data = _load_element_grid_data(grid_directory)

    if not element_data:
        logger.info("No data found.")
        return

    fig, ax = plt.subplots(
        figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.65),
        layout="tight",
    )

    cmap = cm.get_cmap(COLORMAP)
    sorted_ri = sorted(ri_factors, reverse=True)
    colors = [cmap(x) for x in np.linspace(0, 1, len(sorted_ri))]

    for i, ri_factor in enumerate(sorted_ri):
        rc_plot, fractions = _compute_rc_fractions(
            element_data, ri_factor, ri_factor_tol, threshold_ha
        )
        if not rc_plot:
            continue
        ax.fill_between(rc_plot, fractions, color=colors[i], label=f"$r_i/r_c = {ri_factor:.1f}$")

    ax.set_xlabel(r"$r_c$ (Bohr)")
    ax.set_ylim([0, 1])
    ax.margins(x=0)
    ax.legend()

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()
