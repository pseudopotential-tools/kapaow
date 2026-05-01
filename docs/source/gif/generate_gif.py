"""Generate a GIF demonstrating the PAO++ functionality."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from kapaow.data.pseudodojo import fetch_pseudopotential
from kapaow.extend import BasisExtensionViaAddition
from kapaow.plotting import plot_wannier90_dat_file
from kapaow.solve import solve_pseudoatomic_problem

BARRIER_HEIGHT = 10.0
ELEMENT = "Li"
SUBPATH_LENGTH = 10
MAX_R_MID = 15.0
MIN_R_MID = 5.0
MIN_SMEAR = 0.1
MAX_SMEAR = 5.0


def plot_confining_potential(ax: Axes, rc: float, ri_factor: float):
    """Plot the confining potential on the given axes."""
    r_start = rc * ri_factor
    r_end = rc
    [_, rmax] = ax.get_xlim()

    r = np.linspace(r_start, r_end, 100)
    v_conf = BARRIER_HEIGHT * np.sin((r - r_start) / (r_end - r_start) * (np.pi / 2)) ** 2
    ax.fill_between(
        [*r.tolist(), rmax],
        0,
        [*v_conf.tolist(), BARRIER_HEIGHT],
        color="red",
        alpha=0.5,
        label=f"$r_c$ = {rc:.2f}, $r_i$ = {r_start:.2f}",
    )


def generate_still(ri_factor: float, rc: float, axes: Axes | None = None) -> Axes:
    """Generate a single still frame of the PAO++ basis."""
    upf_path = fetch_pseudopotential(ELEMENT)

    working_dir = Path("tmp")
    dat_filename = f"rc_{rc:.2f}_ri-factor_{ri_factor:.2f}.dat"

    solve_pseudoatomic_problem(
        upf_path,
        rc,
        ri_factor,
        BasisExtensionViaAddition(increment=1),
        working_dir=working_dir,
        dat_filename=dat_filename,
    )

    return plot_wannier90_dat_file(working_dir / dat_filename, axes=axes, fix_sign=True)


def ri_factor_and_rc_from_mid_and_smear(mid: float, smear: float) -> tuple[float, float]:
    """Convert mid and smear parameters to ri_factor and rc."""
    rc = mid + smear
    ri_factor = (mid - smear) / rc
    return ri_factor, rc


if __name__ == "__main__":
    # Create a path where...
    # 1) hold mid at max, sweep smear from min to max
    mid = [MAX_R_MID for _ in range(SUBPATH_LENGTH)]
    smear = np.linspace(MIN_SMEAR, MAX_SMEAR, SUBPATH_LENGTH).tolist()
    # 2) sweep mid from max to min, hold smear at max
    mid += np.linspace(MAX_R_MID, MIN_R_MID, SUBPATH_LENGTH).tolist()
    smear += [MAX_SMEAR for _ in range(SUBPATH_LENGTH)]
    # 3) hold mid at min, sweep smear from max to min
    mid += [MIN_R_MID for _ in range(SUBPATH_LENGTH)]
    smear += np.linspace(MAX_SMEAR, MIN_SMEAR, SUBPATH_LENGTH).tolist()
    # 4) sweep mid from min to max, hold smear at min
    mid += np.linspace(MIN_R_MID, MAX_R_MID, SUBPATH_LENGTH).tolist()
    smear += [MIN_SMEAR for _ in range(SUBPATH_LENGTH)]

    potential_file = Path(f"{ELEMENT}_density_potential.h5")

    for i, (m, s) in enumerate(zip(mid, smear, strict=True)):
        if potential_file.exists():
            potential_file.unlink()

        ri_factor, rc = ri_factor_and_rc_from_mid_and_smear(m, s)

        axes = generate_still(ri_factor, rc)
        axes = plot_wannier90_dat_file(Path(f"{ELEMENT}.dat"), axes=axes, linestyle="--")
        # Add a seconday y-axis to each subplot
        for ax in axes:
            ax.set_xlim([0, 20])
            ax.set_ylim([-2, 2])
            ax_secondary = ax.twinx()
            ax_secondary.set_ylabel("Secondary axis")
            plot_confining_potential(ax_secondary, rc, ri_factor)
            ax_secondary.set_ylim([0, BARRIER_HEIGHT])
        plt.savefig(f"frame{i:03}.png")
        plt.close("all")
