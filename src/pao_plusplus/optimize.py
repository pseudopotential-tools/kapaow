"""Optimize module for pao_plusplus."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from matplotlib.axes import Axes
from upf_tools import UPFDict

from pao_plusplus.data.sssp.espresso import input_files
from pao_plusplus.extend import BasisExtension
from pao_plusplus.projectability import compute_projectability
from pao_plusplus.solve import PseudoAtomicInput, solve_pseudoatomic_problem

RI_LOWER = 0.05
RI_UPPER = 0.95
RC_LOWER = 5.0
RC_UPPER = 15.0


ATOMIC_FEMDVR_PATCHES = {
    "Zn": PseudoAtomicInput(dft={"alpha_mix": 0.3}),
}


def optimize(upf_path: Path, extension: BasisExtension | None = None, qe_bin: Path | None = None) -> None:
    """Optimize the confining potential to maximise projectability."""
    upf_dict = UPFDict.from_upf(upf_path)

    element = upf_dict["header"]["element"].strip()
    if not isinstance(element, str):
        raise ValueError("Element symbol in UPF file is not a string.")

    tmp_dir = Path("tmp")
    projector_dir = tmp_dir / "projectors"
    calculation_dir = tmp_dir / "calculations"

    projector_dir.mkdir(parents=True, exist_ok=True)
    calculation_dir.mkdir(parents=True, exist_ok=True)

    def parameters_to_score(rc: float, ri_factor: float) -> float:
        """Compute a score based on the parameters."""
        # Rescaling
        rc = RC_LOWER + (RC_UPPER - RC_LOWER) * rc
        ri_factor = RI_LOWER + (RI_UPPER - RI_LOWER) * ri_factor

        tag = f"{element}_rc_{rc:.10f}_ri-factor_{ri_factor:.10f}"
        dat_filename = f"{tag}.dat"

        # Splve the pseudoatomic problem
        solve_pseudoatomic_problem(
            upf_path,
            rc,
            ri_factor,
            extension=extension,
            working_dir=projector_dir,
            dat_filename=dat_filename,
            atomic_femdvr_config=ATOMIC_FEMDVR_PATCHES.get(element, None)
        )

        # Create a symlink of the .dat file that has the same name as the .upf file
        target = projector_dir / upf_path.with_suffix(".dat").name
        if target.exists():
            target.unlink()
        target.symlink_to(dat_filename)

        scores: dict[str, float] = {}
        for pw_input_file in input_files(element):
            score = compute_projectability(
                tag,
                pw_input_file,
                proj_dir=projector_dir,
                working_dir=calculation_dir,
                pseudo_files=upf_path.parent.glob("*.upf"),
                qe_bin=qe_bin,
            )
            scores[pw_input_file.stem] = score
        return sum(scores.values()) / len(scores)

    optimizer = create_optimizer(parameters_to_score)

    log_file = tmp_dir / f"{element}.log.json"

    if log_file.exists():
        optimizer.load_state(log_file)

    optimizer.maximize(init_points=2, n_iter=40)

    optimizer.save_state(log_file)


def create_optimizer(func: Callable[[float, float], float] | None = None) -> BayesianOptimization:
    """Create a Bayesian optimizer for the PAO optimization problem.

    Note that we use rescaled coordinates in [0, 1] for both rc and ri_factor.
    """
    optimizer = BayesianOptimization(
        f=func,
        pbounds={"rc": (0, 1), "ri_factor": (0, 1)},
        verbose=2,
        random_state=1,
    )
    return optimizer


def plot_optimizer(log_file: Path, filename: Path | None = None) -> None:
    """Plot the results of an optimizer log file and optionally save it to disk."""
    optimizer = create_optimizer()
    optimizer.load_state(log_file)
    _plot(optimizer, filename=filename)


def _plot(
    optimizer: BayesianOptimization,
    contourf_kwargs: dict[str, Any] | None = None,
    filename: Path | None = None,
    plot_uncertainty: bool = False,
    plot_acquisition: bool = False,
) -> None:
    contourf_kwargs = {} if contourf_kwargs is None else contourf_kwargs

    n_axes = 1 + int(plot_uncertainty) + int(plot_acquisition)
    fig, axarr = plt.subplots(n_axes, 1, figsize=(6, 1 + 3 * n_axes), sharex=True)

    if not plot_uncertainty and not plot_acquisition:
        axarr = [axarr]
        [ax] = axarr
    elif plot_uncertainty and plot_acquisition:
        [ax, ax2, ax3] = axarr
    elif plot_uncertainty and not plot_acquisition:
        [ax, ax2] = axarr
    else:
        [ax, ax3] = axarr

    x = np.linspace(*optimizer.space.bounds[0])
    y = np.linspace(*optimizer.space.bounds[1])

    x_grid, y_grid = np.meshgrid(x, y)
    grid = np.array([[x, y] for x, y in zip(np.ravel(x_grid), np.ravel(y_grid), strict=True)])
    x_grid_rescaled = RC_LOWER + (RC_UPPER - RC_LOWER) * x_grid
    y_grid_rescaled = RI_LOWER + (RI_UPPER - RI_LOWER) * y_grid

    def _plot_contourf_with_colorbar(
        ax: Axes, z: np.ndarray, colorbar_label: str, nlevels: int = 20, **kwargs: Any
    ) -> None:
        # Countour plot
        vmin = z.min()
        vmax = z.max()
        levels = np.linspace(vmin, vmax, nlevels)
        surf = ax.contourf(
            x_grid_rescaled,
            y_grid_rescaled,
            z,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            levels=levels,
            extend="both",
            **kwargs,
        )

        # Colorbar
        axc = fig.colorbar(surf, aspect=8)
        axc.set_label(colorbar_label)

        # White crosses for the trials
        for res in optimizer.res:
            [x, y] = res["params"].values()
            x_rescaled = RC_LOWER + (RC_UPPER - RC_LOWER) * x
            y_rescaled = RI_LOWER + (RI_UPPER - RI_LOWER) * y
            ax.scatter(x_rescaled, y_rescaled, marker="x", color="w")

        # Red cross for the maximum
        [x, y] = optimizer.max["params"].values()
        x_rescaled = RC_LOWER + (RC_UPPER - RC_LOWER) * x
        y_rescaled = RI_LOWER + (RI_UPPER - RI_LOWER) * y
        ax.scatter(x_rescaled, y_rescaled, marker="x", color="r")

    mu, sigma = optimizer._gp.predict(grid, return_std=True)

    # Plot the fitted function across the 2D grid
    z_grid = 1 - mu.reshape(x_grid.shape)
    log_z = np.log(z_grid)
    _plot_contourf_with_colorbar(ax, log_z, r"$\log(1 - F)$")

    # Plot the uncertainty across the 2D grid
    if plot_uncertainty:
        z_grid = sigma.reshape(x_grid.shape)
        _plot_contourf_with_colorbar(ax2, z_grid, "uncertainty in fit")

    # Plot the acquisition function across the 2D grid
    if plot_acquisition:
        utility = optimizer.acquisition_function._get_acq(optimizer._gp)(grid)
        z_grid = utility.reshape(x_grid.shape)
        _plot_contourf_with_colorbar(ax3, z_grid, "utility function")

    res = optimizer.res[0]["params"].keys()

    [p1, p2] = [p.replace("rc", "$r_c$").replace("ri_factor", "$r_i/r_c$") for p in res]
    axarr[-1].set_xlabel(p1)
    for a in axarr:
        a.set_ylabel(p2)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
