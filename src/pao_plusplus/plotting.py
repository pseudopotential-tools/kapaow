"""Plotting functionality."""

from matplotlib import pyplot as plt
from pathlib import Path
from pao_plusplus.io import read_wannier90_dat_file
from itertools import cycle


def get_unique_l_values(dat_path: Path) -> set[int]:
    """Get the unique l values from a Wannier90 .dat file."""
    with open(dat_path, "r") as f:
        lines = f.readlines()
    return set(int(x) for x in lines[1].split())

def _plot_wannier90_dat_file(dat_path: Path, axes: list[plt.Axes], **kwargs) -> None:

    r, l_values, orbitals = read_wannier90_dat_file(dat_path)

    for l, orbital in zip(l_values, orbitals, strict=True):
        axes[l].plot(r, orbital, **kwargs)

def plot_wannier90_dat_files(dat_paths: list[Path], filename: Path | None = None) -> None:
    """Plot the pseudoatomic orbitals stored in multiple Wannier90 .dat files."""

    unique_l_values: set[int] = set()
    for dat_path in dat_paths:
        unique_l_values.update(get_unique_l_values(dat_path))
    
    _, axes = plt.subplots(len(unique_l_values), sharex=True, figsize=(6, 3 * len(unique_l_values) - 1))

    linestyles = cycle(['-', '--', '-.', ':'])

    for dat_path, linestyle in zip(dat_paths, linestyles):
        # Reset the colour cycle
        for ax in axes:
            ax.set_prop_cycle(None)
        _plot_wannier90_dat_file(dat_path, axes=axes, linestyle=linestyle)
    
    for ax, l in zip(axes, sorted(unique_l_values)):
        ax.text(0.95, 0.9, f"$l={l}$", transform=ax.transAxes, ha="right", va="top")

    plt.tight_layout()

    axes[-1].set_xlim(0, max(axes[-1].get_lines()[0].get_xdata()))

    if filename is not None:
        plt.savefig(filename)

def plot_wannier90_dat_file(dat_path: Path, filename: Path | None = None) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    plot_wannier90_dat_files([dat_path], filename)
