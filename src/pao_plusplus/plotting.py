"""Plotting functionality."""

from itertools import cycle
from pathlib import Path
from typing import Any
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from pao_plusplus.io import read_wannier90_dat_file


def get_unique_l_values(dat_path: Path) -> set[int]:
    """Get the unique l values from a Wannier90 .dat file."""
    with open(dat_path, encoding="utf-8") as f:
        lines = f.readlines()
    return {int(x) for x in lines[1].split()}


def _plot_wannier90_dat_file(dat_path: Path, axes: list[Axes], fix_sign: float=False, **kwargs: Any) -> None:
    _, r, l_values, orbitals = read_wannier90_dat_file(dat_path)

    for l_value, orbital in zip(l_values, orbitals, strict=True):
        if fix_sign and orbital[1] < 0:
            orbital *= -1
        axes[l_value].plot(r, orbital, **kwargs)


def plot_wannier90_dat_files(dat_paths: list[Path], filename: Path | None = None, axes: Axes | None = None, fix_sign: bool = False,
                             **kwargs) -> Axes:
    """Plot the pseudoatomic orbitals stored in multiple Wannier90 .dat files."""
    unique_l_values: set[int] = set()
    for dat_path in dat_paths:
        unique_l_values.update(get_unique_l_values(dat_path))

    if axes is None:
        _, axes = plt.subplots(
            len(unique_l_values), sharex=True, figsize=(6, 3 * len(unique_l_values) - 1)
        )

    linestyles = cycle(["-", "--", "-.", ":"])

    for dat_path, linestyle in zip(dat_paths, linestyles, strict=False):
        # Reset the colour cycle
        for ax in axes:
            ax.set_prop_cycle(None)
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = linestyle
        _plot_wannier90_dat_file(dat_path, axes=axes, fix_sign=fix_sign, **kwargs)

    for ax, l_value in zip(axes, sorted(unique_l_values), strict=False):
        ax.text(0.95, 0.9, f"$l={l_value}$", transform=ax.transAxes, ha="right", va="top")
        ax.set_ylim(-2, 2)

    plt.tight_layout()

    axes[-1].set_xlim(0, 20) #max(axes[-1].get_lines()[0].get_xdata()))

    if filename is not None:
        plt.savefig(filename)
    
    return axes


def plot_wannier90_dat_file(dat_path: Path, filename: Path | None = None, axes: Axes | None = None, fix_sign: bool = False, **kwargs) -> Axes:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    return plot_wannier90_dat_files([dat_path], filename, axes=axes, fix_sign=fix_sign, **kwargs)