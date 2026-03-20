"""Optimize module for pao_plusplus."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from upf_tools import UPFDict

from pao_plusplus.basis import AtomicBasis
from pao_plusplus.data.sssp.structures import input_files
from pao_plusplus.extend import BasisExtension
from pao_plusplus.projectability import compute_projectability_cached, preload_material
from pao_plusplus.solve import PseudoAtomicInput, compute_spread, solve_and_export
from pao_plusplus.workflows import run_bands_workflow, run_qe_workflow

RI_LOWER = 0.0
RI_UPPER = 0.95
RC_LOWER = 5.0
RC_UPPER = 15.0


ATOMIC_FEMDVR_PATCHES = {
    "Cr": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Cu": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Pd": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "At": PseudoAtomicInput(dft={"alpha_mix": 0.1, "max_iter": 200}),
    "Sb": PseudoAtomicInput(dft={"alpha_mix": 0.3}),
    "Zn": PseudoAtomicInput(dft={"alpha_mix": 0.3}),
}


def compute_num_target_bands(
    structure_file: Path,
    target_element: str,
    target_orbitals_per_atom: int,
    upf_by_element: dict[str, Path],
) -> int:
    """Compute the number of target bands for a given material.

    This is the total number of PAO orbitals across all atoms in the unit cell.
    """
    atoms = ase.io.read(str(structure_file))
    ntb = 0
    for sym in atoms.get_chemical_symbols():
        if sym == target_element:
            ntb += target_orbitals_per_atom
        else:
            other_basis = AtomicBasis.from_upf(upf_by_element[sym]).to_pseudoatomic_basis()
            ntb += other_basis.total_number_of_orbitals
    return ntb


def _extract_element(upf_path: Path) -> str:
    """Extract and validate the element symbol from a UPF file."""
    upf_dict = UPFDict.from_upf(upf_path)
    element = upf_dict["header"]["element"].strip()
    if not isinstance(element, str):
        raise ValueError("Element symbol in UPF file is not a string.")
    return element


def _index_upf_files(pseudo_dir: Path) -> dict[str, Path]:
    """Index all available UPF files in a directory by element symbol."""
    upf_by_element: dict[str, Path] = {}
    for f in list(pseudo_dir.glob("*.upf")) + list(pseudo_dir.glob("*.UPF")):
        d = UPFDict.from_upf(f)
        upf_by_element[d["header"]["element"].strip()] = f
    return upf_by_element


def _run_qe_for_materials(
    structure_file_list: list[Path],
    calculation_dir: Path,
    num_target_bands_per_material: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run QE scf+nscf and bands once per test material."""
    material_atoms: dict[str, Any] = {}
    qe_results: dict[str, Any] = {}
    bands_results: dict[str, Any] = {}
    for structure_file in structure_file_list:
        working_dir = calculation_dir / structure_file.stem
        working_dir.mkdir(parents=True, exist_ok=True)
        ntb = num_target_bands_per_material[structure_file.stem]
        min_nbnd = max(int(1.5 * ntb), ntb + 4)
        result = run_qe_workflow(
            structure_file,
            working_dir=working_dir,
            min_nbnd=min_nbnd,
        )
        qe_results[structure_file.stem] = result
        material_atoms[structure_file.stem] = result.atoms

        bands_result = run_bands_workflow(
            structure_file=structure_file,
            working_dir=working_dir,
            min_nbnd=min_nbnd,
        )
        bands_results[structure_file.stem] = bands_result
    return material_atoms, qe_results, bands_results


def _generate_other_bessel_files(
    element: str,
    material_atoms: dict[str, Any],
    upf_by_element: dict[str, Path],
    projector_dir: Path,
) -> dict[str, Path]:
    """Generate unconfined Bessel files for non-target species."""
    all_other_species = {
        atom.symbol for atoms in material_atoms.values() for atom in atoms if atom.symbol != element
    }
    other_bessel_files: dict[str, Path] = {}
    for other_elem in all_other_species:
        other_upf = upf_by_element[other_elem]
        _, bessel_path = solve_and_export(other_upf, working_dir=projector_dir)
        other_bessel_files[other_elem] = bessel_path
    return other_bessel_files


def _preload_all_materials(
    structure_file_list: list[Path],
    qe_results: dict[str, Any],
) -> dict[str, Any]:
    """Pre-load QE wavefunctions for all materials."""
    cached_materials = {}
    for structure_file in structure_file_list:
        result = qe_results[structure_file.stem]
        cached_materials[structure_file.stem] = preload_material(
            pwi_file=result.nscf_input_file,
            wfc_dir=result.nscf_wfc_dir,
            kpoint_weights=result.kpoint_weights,
        )
    return cached_materials


def optimize(
    upf_path: Path,
    extension: BasisExtension | None = None,
    spread_weight: float = 0.00,
) -> None:
    """Optimize the confining potential to maximise projectability."""
    import warnings

    warnings.filterwarnings("ignore", message="invalid value encountered", module="atomic_femdvr")

    element = _extract_element(upf_path)

    tmp_dir = Path("tmp") / "optimize" / "projectability"
    projector_dir = tmp_dir / "projectors"
    calculation_dir = tmp_dir / "calculations"

    projector_dir.mkdir(parents=True, exist_ok=True)
    calculation_dir.mkdir(parents=True, exist_ok=True)

    # Determine orbitals per atom for the target element
    atomic_basis = AtomicBasis.from_upf(upf_path)
    if extension is not None:
        pseudo_basis = extension.extend(atomic_basis)
    else:
        pseudo_basis = atomic_basis.to_pseudoatomic_basis()
    target_orbitals_per_atom = pseudo_basis.total_number_of_orbitals

    upf_by_element = _index_upf_files(upf_path.parent)
    structure_file_list = list(input_files(element))

    # Compute per-material num_target_bands before running QE
    num_target_bands_per_material: dict[str, int] = {
        sf.stem: compute_num_target_bands(sf, element, target_orbitals_per_atom, upf_by_element)
        for sf in structure_file_list
    }

    material_atoms, qe_results, _bands_results = _run_qe_for_materials(
        structure_file_list, calculation_dir, num_target_bands_per_material
    )

    other_bessel_files = _generate_other_bessel_files(
        element, material_atoms, upf_by_element, projector_dir
    )

    cached_materials = _preload_all_materials(structure_file_list, qe_results)

    step_counter = 1

    def parameters_to_score(rc: float, ri_factor: float) -> float:
        """Compute a score based on the parameters."""
        nonlocal step_counter

        # Rescaling
        rc = RC_LOWER + (RC_UPPER - RC_LOWER) * rc
        ri_factor = RI_LOWER + (RI_UPPER - RI_LOWER) * ri_factor

        dat_filename = f"{element}_rc_{rc:.10f}_ri-factor_{ri_factor:.10f}.dat"

        _, bessel_file = solve_and_export(
            upf_path,
            rc,
            ri_factor,
            extension=extension,
            working_dir=projector_dir,
            dat_filename=dat_filename,
            atomic_femdvr_config=ATOMIC_FEMDVR_PATCHES.get(element, None),
        )

        # Compute the spread penalty for the outermost subshell
        spread = compute_spread(projector_dir / dat_filename, atomic_basis)

        scores = _compute_material_scores(
            structure_file_list,
            material_atoms,
            element,
            bessel_file,
            other_bessel_files,
            cached_materials,
            num_target_bands_per_material,
        )

        projectability = sum(scores.values()) / len(scores)
        combined_score = projectability - spread_weight * spread

        step_counter += 1
        return combined_score

    optimizer = create_optimizer(parameters_to_score)

    log_file = tmp_dir / f"{element}.log.json"

    if log_file.exists():
        optimizer.load_state(log_file)

    # Always start with the unconfined limit (rc=1, ri_factor=1 in rescaled coords)
    optimizer.probe(params={"rc": 1.0, "ri_factor": 1.0}, lazy=True)

    optimizer.maximize(init_points=2, n_iter=40)

    optimizer.save_state(log_file)


def _compute_material_scores(
    structure_file_list: list[Path],
    material_atoms: dict[str, Any],
    element: str,
    bessel_file: Path,
    other_bessel_files: dict[str, Path],
    cached_materials: dict[str, Any],
    num_target_bands_per_material: dict[str, int],
) -> dict[str, float]:
    """Compute projectability scores across all test materials."""
    scores: dict[str, float] = {}
    for structure_file in structure_file_list:
        material_species = {atom.symbol for atom in material_atoms[structure_file.stem]}
        bessel_map: dict[str, Path] = {element: bessel_file}
        for other_elem in material_species - {element}:
            bessel_map[other_elem] = other_bessel_files[other_elem]

        score = compute_projectability_cached(
            cached_materials[structure_file.stem],
            bessel_files=bessel_map,
            num_target_bands=num_target_bands_per_material[structure_file.stem],
        )
        scores[structure_file.stem] = score
    return scores


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
    filename: Path | None = None,
) -> None:
    import matplotlib.ticker as mticker
    from matplotlib.colors import LogNorm

    from pao_plusplus.plotting import REVTEX_COLUMN_WIDTH

    # Layout in inches — then convert to figure fractions
    left_in = 0.55
    right_in = 0.55
    cbar_w_in = 0.1
    cbar_gap_in = 0.08
    bottom_in = 0.45
    top_in = 0.1

    fig_w = REVTEX_COLUMN_WIDTH
    axes_size_in = fig_w - left_in - right_in - cbar_w_in - cbar_gap_in
    fig_h = bottom_in + top_in + axes_size_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [left_in / fig_w, bottom_in / fig_h, axes_size_in / fig_w, axes_size_in / fig_h]
    )
    cax = fig.add_axes(
        [
            (left_in + axes_size_in + cbar_gap_in) / fig_w,
            bottom_in / fig_h,
            cbar_w_in / fig_w,
            axes_size_in / fig_h,
        ]
    )

    x = np.linspace(*optimizer.space.bounds[0])
    y = np.linspace(*optimizer.space.bounds[1])
    x_grid, y_grid = np.meshgrid(x, y)
    grid = np.array(
        [
            [x, y]
            for x, y in zip(
                np.ravel(x_grid),
                np.ravel(y_grid),
                strict=True,
            )
        ]
    )
    x_grid_rescaled = RC_LOWER + (RC_UPPER - RC_LOWER) * x_grid
    y_grid_rescaled = RI_LOWER + (RI_UPPER - RI_LOWER) * y_grid

    mu, _ = optimizer._gp.predict(grid, return_std=True)
    z_grid = 1 - mu.reshape(x_grid.shape)  # 1 - F for log scale
    z_grid = np.clip(z_grid, 1e-10, None)

    levels = np.logspace(
        np.log10(z_grid.min()),
        np.log10(z_grid.max()),
        20,
    )
    norm = LogNorm(vmin=z_grid.min(), vmax=z_grid.max())
    surf = ax.contourf(
        x_grid_rescaled,
        y_grid_rescaled,
        z_grid,
        cmap="viridis",
        levels=levels,
        norm=norm,
        extend="both",
    )

    axc = fig.colorbar(surf, cax=cax)
    axc.ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10))
    axc.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{1 - x:g}"))
    axc.ax.yaxis.set_minor_locator(mticker.NullLocator())
    axc.ax.invert_yaxis()
    axc.set_label(r"$F$")

    # White crosses for trials, red cross for the best
    for res in optimizer.res:
        [xv, yv] = res["params"].values()
        ax.scatter(
            RC_LOWER + (RC_UPPER - RC_LOWER) * xv,
            RI_LOWER + (RI_UPPER - RI_LOWER) * yv,
            marker="x",
            color="w",
        )
    [xv, yv] = optimizer.max["params"].values()
    ax.scatter(
        RC_LOWER + (RC_UPPER - RC_LOWER) * xv,
        RI_LOWER + (RI_UPPER - RI_LOWER) * yv,
        marker="x",
        color="r",
    )

    res = optimizer.res[0]["params"].keys()
    [p1, p2] = [p.replace("rc", "$r_c$").replace("ri_factor", "$r_i/r_c$") for p in res]
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close(fig)
    else:
        plt.show()
