"""Generate GIF animations showing PAOs under varying confinement."""

from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from upf_tools import UPFDict

from kapaow.basis import AngularMomentum, AtomicBasis, Subshell
from kapaow.extend import BasisExtension
from kapaow.io import read_wannier90_dat_file
from kapaow.plotting import COLOR_ALERT, COLORMAP, plot_wannier90_dat_file
from kapaow.solve import (
    ATOMIC_FEMDVR_PATCHES,
    DEFAULT_RC_MAX,
    DEFAULT_RI_FACTOR_MAX,
    solve_and_export,
)


def generate_animation(
    upf_path: Path,
    extension: BasisExtension | None = None,
    frames_per_segment: int = 10,
    max_r_mid: float = 15.0,
    min_r_mid: float = 5.0,
    min_width: float = 0.1,
    max_width: float = 5.0,
    energy_shift_threshold: float = 0.02,
    output: Path = Path("output.gif"),
    working_dir: Path = Path("tmp") / "animate",
    on_frame: Callable[[int, int, float, float], None] | None = None,
) -> Path:
    """Generate a GIF showing PAOs under varying confinement.

    Parameters
    ----------
    upf_path
        Path to the UPF pseudopotential file.
    extension
        Optional basis extension.
    frames_per_segment
        Number of frames per path segment in the (mid, width) loop.
    max_r_mid, min_r_mid
        Range for the midpoint of the confining potential.
    min_width, max_width
        Range for the half-width of the confining potential.
    energy_shift_threshold
        Energy shift threshold in Ry; orbitals above this are plotted in red.
    output
        Path to save the output GIF.
    working_dir
        Directory for intermediate files.
    on_frame
        Optional callback ``(frame_index, total_frames, rc, ri_factor) -> None``
        called after each frame is rendered.

    Returns
    -------
    Path
        The path to the saved GIF.
    """
    cmap = mpl.colormaps[COLORMAP]
    color_original = cmap(0.2)
    color_added = cmap(0.6)
    color_potential = cmap(0.8)

    upf_dict = UPFDict.from_upf(upf_path)
    element = upf_dict["header"]["element"].strip()
    original_basis = AtomicBasis(
        subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
    )
    original_n_per_l = original_basis.to_pseudoatomic_basis().number_of_orbitals
    barrier_height = 10.0
    threshold_ha = energy_shift_threshold / 2  # Convert Ry to Hartree
    atomic_femdvr_config = ATOMIC_FEMDVR_PATCHES.get(element)

    # Build a closed loop in (mid, width) space
    n = frames_per_segment
    mid = (
        [max_r_mid] * n
        + np.linspace(max_r_mid, min_r_mid, n).tolist()
        + [min_r_mid] * n
        + np.linspace(min_r_mid, max_r_mid, n).tolist()
    )
    width = (
        np.linspace(min_width, max_width, n).tolist()
        + [max_width] * n
        + np.linspace(max_width, min_width, n).tolist()
        + [min_width] * n
    )

    working_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    # Generate a reference (unconfined) dat file
    solve_and_export(
        upf_path,
        rc=DEFAULT_RC_MAX,
        ri_factor=DEFAULT_RI_FACTOR_MAX,
        extension=extension,
        working_dir=working_dir,
        dat_filename=f"{element}_reference.dat",
        atomic_femdvr_config=atomic_femdvr_config,
    )
    ref_dat = working_dir / f"{element}_reference.dat"
    _, _, _, ref_orbitals = read_wannier90_dat_file(ref_dat)

    for i, (m, w) in enumerate(zip(mid, width, strict=True)):
        rc = m + w
        ri_factor = (m - w) / rc
        dat_filename = f"{element}_frame_{i:03d}.dat"

        result, _ = solve_and_export(
            upf_path,
            rc=rc,
            ri_factor=ri_factor,
            extension=extension,
            working_dir=working_dir,
            dat_filename=dat_filename,
            atomic_femdvr_config=atomic_femdvr_config,
        )

        # Determine per-orbital colors: blue for original, tab:orange for added,
        # red if shift too large
        _, _, l_values, _ = read_wannier90_dat_file(working_dir / dat_filename)
        energy_shifts = result.energy_shifts
        confined_colors = []
        ref_colors = []
        l_counter: dict[int, int] = {}
        for l in l_values:
            n_l = l_counter.get(l, 0)
            l_shifts = energy_shifts.get(str(l)) if energy_shifts else None
            n_orig = original_n_per_l.get(AngularMomentum(l), 0)
            is_added = n_l >= n_orig
            if is_added:
                confined_colors.append(color_added)
                ref_colors.append(color_added)
            elif abs(l_shifts[n_l]) > threshold_ha:
                confined_colors.append(COLOR_ALERT)
                ref_colors.append(color_original)
            else:
                confined_colors.append(color_original)
                ref_colors.append(color_original)
            l_counter[l] = n_l + 1

        axes = plot_wannier90_dat_file(
            working_dir / dat_filename,
            fix_sign=True,
            colors=confined_colors,
            reference_orbitals=ref_orbitals,
        )
        plot_wannier90_dat_file(
            ref_dat,
            axes=axes,
            linestyle="--",
            colors=ref_colors,
            reference_orbitals=ref_orbitals,
        )

        for j, ax in enumerate(axes):
            ax.set_xlim([0, 20])
            ax.set_ylim([-2, 2])
            ax.set_ylabel("PAOs")
            ax2 = ax.twinx()
            r_start = rc * ri_factor
            r = np.linspace(r_start, rc, 100)
            v_conf = (
                barrier_height
                * np.sin(
                    (r - r_start) / (rc - r_start) * (np.pi / 2),
                )
                ** 2
            )
            rmax = ax.get_xlim()[1]
            ax2.fill_between(
                [*r.tolist(), rmax],
                0,
                [*v_conf.tolist(), barrier_height],
                color=color_potential,
                alpha=0.3,
            )
            ax2.set_ylim([0, barrier_height])
            ax2.set_ylabel("confining potential (Ha)")
            # Legend on top subplot only
            if j == 0:
                from matplotlib.legend_handler import HandlerBase
                from matplotlib.lines import Line2D

                class _SolidDashedHandler(HandlerBase):
                    """Draw a solid line above a dashed line, like an equals sign."""

                    def __init__(self, color: str):
                        super().__init__()
                        self._color = color

                    def create_artists(
                        self,
                        legend,
                        orig_handle,
                        xdescent,
                        ydescent,
                        width,
                        height,
                        fontsize,
                        trans,
                    ):
                        spacing = height * 0.35
                        solid = Line2D(
                            [xdescent, xdescent + width],
                            [height / 2 + spacing, height / 2 + spacing],
                            color=self._color,
                            linestyle="-",
                            lw=1.5,
                            transform=trans,
                        )
                        dashed = Line2D(
                            [xdescent, xdescent + width],
                            [height / 2 - spacing, height / 2 - spacing],
                            color=self._color,
                            linestyle="--",
                            lw=1.5,
                            transform=trans,
                        )
                        return [solid, dashed]

                dummy_original = Line2D(
                    [],
                    [],
                    label="preexisting PAOs (confined / unconfined)",
                )
                dummy_added = Line2D(
                    [],
                    [],
                    label="added PAO (confined / unconfined)",
                )
                ax.legend(
                    handles=[dummy_original, dummy_added],
                    handler_map={
                        dummy_original: _SolidDashedHandler(color_original),
                        dummy_added: _SolidDashedHandler(color_added),
                    },
                    loc="lower right",
                    bbox_to_anchor=(1.0, 1.0),
                    ncol=1,
                    fontsize=6,
                    handlelength=4,
                    frameon=False,
                )

        axes[-1].set_xlabel("$r$ (Bohr)")
        frame_path = working_dir / f"frame_{i:03d}.png"
        plt.subplots_adjust(top=0.93, left=0.15, right=0.875, bottom=0.075)
        plt.savefig(frame_path, dpi=150)
        plt.close("all")
        frames.append(frame_path)

        if on_frame is not None:
            on_frame(i, len(mid), rc, ri_factor)

    # Assemble GIF
    images = [Image.open(f) for f in frames]
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
    )
    return output
