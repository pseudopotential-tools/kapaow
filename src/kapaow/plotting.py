"""Plotting functionality."""

from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from kapaow.io import read_wannier90_dat_file

# RevTeX column widths in inches (1 pt = 1/72 inch)
REVTEX_COLUMN_WIDTH = 246 / 72  # single column
REVTEX_DOUBLE_COLUMN_WIDTH = 510 / 72  # double column

COLORMAP = "viridis_r"
COLOR_ALERT = "tab:red"

RASTER_DPI = 300


def savefig(fig_or_plt: Any, filename: Path | str, **kwargs: Any) -> None:
    """Save a figure with appropriate DPI for raster formats."""
    path = Path(filename)
    if path.suffix.lower() != ".svg":
        kwargs.setdefault("dpi", RASTER_DPI)
    fig_or_plt.savefig(filename, **kwargs)


sns.set_context("paper", font_scale=0.7)


def get_unique_l_values(dat_path: Path) -> set[int]:
    """Get the unique l values from a Wannier90 .dat file."""
    with open(dat_path, encoding="utf-8") as f:
        lines = f.readlines()
    return {int(x) for x in lines[1].split()}


def _plot_wannier90_dat_file(
    dat_path: Path,
    axes: list[Axes],
    fix_sign: float = False,
    colors: list[str | None] | None = None,
    reference_orbitals: np.ndarray | None = None,
    **kwargs: Any,
) -> None:
    _, r, l_values, orbitals = read_wannier90_dat_file(dat_path)

    for i, (l_value, orbital) in enumerate(zip(l_values, orbitals, strict=True)):
        if fix_sign:
            if reference_orbitals is not None:
                # Match sign to reference via overlap
                if np.dot(orbital, reference_orbitals[i]) < 0:
                    orbital *= -1
            elif orbital[np.argmax(np.abs(orbital))] < 0:
                orbital *= -1
        kw = dict(kwargs)
        if colors is not None and colors[i] is not None:
            kw["color"] = colors[i]
        axes[l_value].plot(r, orbital, **kw)


def plot_wannier90_dat_files(
    dat_paths: list[Path],
    filename: Path | None = None,
    axes: Axes | None = None,
    fix_sign: bool = False,
    colors: list[str | None] | None = None,
    reference_orbitals: np.ndarray | None = None,
    **kwargs,
) -> Axes:
    """Plot the pseudoatomic orbitals stored in multiple Wannier90 .dat files."""
    unique_l_values: set[int] = set()
    for dat_path in dat_paths:
        unique_l_values.update(get_unique_l_values(dat_path))

    if axes is None:
        n_panels = len(unique_l_values)
        _, axes = plt.subplots(
            n_panels,
            sharex=True,
            figsize=(REVTEX_COLUMN_WIDTH, REVTEX_COLUMN_WIDTH * 0.6 * n_panels),
        )

    linestyles = cycle(["-", "--", "-.", ":"])

    for dat_path, linestyle in zip(dat_paths, linestyles, strict=False):
        # Reset the colour cycle
        for ax in axes:
            ax.set_prop_cycle(None)
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = linestyle
        _plot_wannier90_dat_file(
            dat_path,
            axes=axes,
            fix_sign=fix_sign,
            colors=colors,
            reference_orbitals=reference_orbitals,
            **kwargs,
        )

    for ax, l_value in zip(axes, sorted(unique_l_values), strict=False):
        ax.text(
            0.95,
            0.9,
            f"$l={l_value}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.set_ylim(-2, 2)

    plt.tight_layout()

    axes[-1].set_xlim(0, 20)

    if filename is not None:
        savefig(plt, filename)

    return axes


def plot_wannier90_dat_file(
    dat_path: Path,
    filename: Path | None = None,
    axes: Axes | None = None,
    fix_sign: bool = False,
    colors: list[str | None] | None = None,
    reference_orbitals: np.ndarray | None = None,
    **kwargs,
) -> Axes:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    return plot_wannier90_dat_files(
        [dat_path],
        filename,
        axes=axes,
        fix_sign=fix_sign,
        colors=colors,
        reference_orbitals=reference_orbitals,
        **kwargs,
    )
