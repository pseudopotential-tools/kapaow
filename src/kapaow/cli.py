"""Command line interface for :mod:`kapaow`.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to
import things from ``__main__`` later, but that will cause problems--the code will get
executed twice:

- When you run ``python3 -m kapaow`` python will execute``__main__.py`` as a
  script. That means there won't be any ``kapaow.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because there's no
  ``kapaow.__main__`` in ``sys.modules``.

.. seealso::

    https://click.palletsprojects.com/en/8.1.x/setuptools/#setuptools-integration
"""

import logging
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any

import click
from upf_tools import UPFDict

from kapaow.extend import (
    BasisExtension,
    BasisExtensionType,
    BasisExtensionViaAddition,
    BasisExtensionViaPolarization,
    parse_extension,
)
from kapaow.periodic_table import plot_periodic_table
from kapaow.plotting import plot_wannier90_dat_files
from kapaow.solve import DEFAULT_RC_MAX, DEFAULT_RI_FACTOR_MAX, solve_and_export

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


def add_option(func: Callable) -> click.Command:
    """Reusable Click option for adding basis functions."""
    return click.option(
        "--add",
        type=click.Choice([t.value for t in BasisExtensionType]),
        multiple=True,
        help=(
            "Add basis functions to the PAO basis (repeatable, e.g. --add subshell --add subshell)."
        ),
    )(func)


def symmetrize_option(func: Callable) -> click.Command:
    """Reusable Click option toggling symmetry-adapted projector rotation."""
    return click.option(
        "--symmetrize",
        is_flag=True,
        default=False,
        help=(
            "Rotate projectors into a symmetry-adapted, bond-oriented basis "
            "(sp-n hybrids + non-bonding irrep orbitals) before use."
        ),
    )(func)


def experimental(func: Callable) -> Callable:
    """Mark a CLI command as requiring the ``[experimental]`` extras.

    Two effects:

    * Prepends ``[experimental]`` to the docstring so the marker shows
      in ``kapaow --help`` listings, signalling to users that the
      command will only work with the AiiDA-backed stack installed.
    * Wraps the command body to catch :class:`ImportError` originating
      in :mod:`kapaow._experimental` and reraise it as
      :class:`click.UsageError`, so users without the extras see a
      one-line install hint instead of a Python traceback.
    """
    import functools

    if func.__doc__:
        func.__doc__ = f"[experimental] {func.__doc__}"
    else:
        func.__doc__ = "[experimental]"

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ImportError as exc:
            if "kapaow._experimental" in str(exc):
                raise click.UsageError(str(exc)) from exc
            raise

    return wrapper


def get_extension(add: tuple[str, ...]) -> BasisExtension | None:
    """Click-aware wrapper around :func:`kapaow.extend.parse_extension`.

    Translates the parser's :class:`ValueError` into a
    :class:`click.UsageError` so the CLI prints the standard usage hint.
    """
    try:
        return parse_extension(add)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


def _describe_extension(upf: Path, extension: BasisExtension) -> str:
    """Return a filename-friendly description of what an extension adds.

    For subshell additions, compares the original and extended bases to
    produce e.g. ``"_with_2p_and_3s"``.  For polarization, returns
    ``"_polarized"`` (or ``"_polarized_x2"`` etc.).
    """
    from kapaow.basis import AtomicBasis, Subshell

    l_letters = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}

    if isinstance(extension, BasisExtensionViaAddition):
        upf_dict = UPFDict.from_upf(upf)
        original = AtomicBasis(
            subshells=[Subshell(n=chi["n"], l=chi["l"]) for chi in upf_dict["pswfc"]["chi"]]
        )
        extended = extension.extend_atomic(original)
        added = [s for s in extended.subshells if s not in original.subshells]
        names = [f"{s.n}{l_letters[s.l]}" for s in added]
        return "_with_" + "_and_".join(names)
    elif isinstance(extension, BasisExtensionViaPolarization):
        if extension.increment == 1:
            return "_polarized"
        return f"_polarized_x{extension.increment}"
    return ""


def enable_postmortem_debugger() -> None:
    """Replace sys.excepthook to drop into a post-mortem debugger.

    Prefers :mod:`ipdb` for tab completion and syntax highlighting if it
    is installed, otherwise falls back to the stdlib :mod:`pdb`. Either
    way, this only fires when ``--debug`` is passed.
    """
    try:
        import ipdb as _debugger
    except ImportError:
        import pdb as _debugger

    def _excepthook(
        exc_type: type[BaseException], exc_value: BaseException, exc_tb: TracebackType | None
    ) -> None:
        traceback.print_exception(exc_type, exc_value, exc_tb)
        click.echo(click.style("\nEntering post-mortem debugging...", fg="yellow"))
        _debugger.post_mortem(exc_tb)

    sys.excepthook = _excepthook


@click.group()
@click.version_option()
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.option("-l", "--log", is_flag=True, help="Enable logging to kapaow.log.")
@click.pass_context
def main(ctx: click.Context, debug: bool, log: bool) -> None:
    """CLI for kapaow."""
    if debug:
        enable_postmortem_debugger()
    if log:
        import logging

        file_handler = logging.FileHandler("kapaow.log", mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        pkg_logger = logging.getLogger("kapaow")
        pkg_logger.setLevel(logging.INFO)
        pkg_logger.addHandler(file_handler)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@main.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--select",
    "select_str",
    type=str,
    default=None,
    help="Orbitals to extract from an OpenMX .pao file (e.g. 'sspd' for 2s, 1p, 1d). "
    "Only valid for .pao files.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output .dat file path (default: <stem>.dat).",
)
def convert(input_file: Path, select_str: str | None, output: Path | None) -> None:
    r"""Convert a pseudopotential file to Wannier90 .dat format.

    INPUT_FILE is a UPF pseudopotential or an OpenMX .pao file. The format
    is detected from the file extension.

    If -o/--output is not provided, the .dat content is printed to stdout.

    \b
    Examples:
        kapaow convert Li.upf
        kapaow convert Li.upf -o Li.dat
        kapaow convert Li8.0.pao --select sspd
    """
    if input_file.suffix == ".pao":
        from kapaow.io import format_wannier90_dat
        from kapaow.openmx import convert_to_wannier90, parse_select, read_openmx_pao

        pao = read_openmx_pao(input_file)
        selected = parse_select(list(select_str)) if select_str else None
        x, r, l_values, orbitals = convert_to_wannier90(pao, selected)
        dat_content = format_wannier90_dat(x, r, l_values, orbitals)
    else:
        if select_str is not None:
            raise click.UsageError("--select is only valid for OpenMX .pao files.")
        upf_dict = UPFDict.from_upf(input_file)
        dat_content = upf_dict.to_dat()

    if output is not None:
        output.write_text(dat_content)
        click.echo(f"Written to {output}")
    else:
        click.echo(dat_content, nl=False)


# ---------------------------------------------------------------------------
# confine
# ---------------------------------------------------------------------------


@main.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option("--rc", type=float, default=None, help="Confinement radius.")
@click.option(
    "--ri-factor", type=float, default=None, help="Inner radius factor for the confining potential."
)
@add_option
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output .dat file path (default: auto-generated from UPF name and parameters).",
)
def confine(
    upf: Path, rc: float | None, ri_factor: float | None, add: tuple[str, ...], output: Path | None
) -> None:
    r"""Solve for PAOs under a confining potential, optionally adding orbitals.

    If -o/--output is not provided, the .dat content is printed to stdout.

    \b
    Examples:
        kapaow confine Li.upf --add subshell --rc 8 --ri-factor 0.9
        kapaow confine Li.upf --add subshell --add subshell
    """
    import tempfile

    extension = get_extension(add)
    if extension is None:
        raise click.UsageError("confine requires --add to specify which orbitals to add.")

    if rc is None:
        rc = DEFAULT_RC_MAX
    if ri_factor is None:
        ri_factor = DEFAULT_RI_FACTOR_MAX

    if output is not None:
        solve_and_export(upf, rc=rc, ri_factor=ri_factor, extension=extension, dat_filename=output)
        click.echo(f"Written to {output}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            dat_file = Path("output.dat")
            solve_and_export(
                upf,
                rc=rc,
                ri_factor=ri_factor,
                extension=extension,
                dat_filename=dat_file,
                working_dir=Path(tmpdir),
            )
            click.echo((Path(tmpdir) / dat_file).read_text(), nl=False)


# ---------------------------------------------------------------------------
# optimize (group with projectability and spread subcommands)
# ---------------------------------------------------------------------------


@main.group()
def optimize() -> None:
    """Optimize PAO parameters."""
    pass


@optimize.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@add_option
@experimental
def projectability(upf: Path, add: tuple[str, ...]) -> None:
    """Optimize PAOs to maximise their projectability via Bayesian optimization."""
    from kapaow._experimental.optimize import optimize as internal_optimize

    extension = get_extension(add)
    internal_optimize(upf, extension)


@optimize.command()
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--rc",
    type=float,
    multiple=True,
    default=None,
    help="Confinement radii to scan (can be specified multiple times).",
)
@click.option(
    "--ri-factor",
    type=float,
    multiple=True,
    default=None,
    help="Inner radius factors to scan (can be specified multiple times).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="spread.svg",
    help="Save the Pareto plot to this file.",
)
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the grid data to a JSON file.",
)
@add_option
def spread(
    upf: Path,
    rc: tuple[float, ...],
    ri_factor: tuple[float, ...],
    output: Path,
    loglog: bool,
    logy: bool,
    json_path: Path | None,
    add: tuple[str, ...],
) -> None:
    """Optimize PAO spread by scanning rc and ri_factor, producing a Pareto front."""
    from kapaow.pareto import compute_pareto_front, dump_pareto_json, plot_pareto

    extension = get_extension(add)
    rc_values = list(rc) if rc else None
    ri_factor_values = list(ri_factor) if ri_factor else None

    spreads, max_energy_shifts, metadata = compute_pareto_front(
        upf,
        extension=extension,
        rc_values=rc_values,
        ri_factor_values=ri_factor_values,
        loglog=loglog,
    )

    # Always dump JSON (to a default path if not specified), then plot from it
    effective_json = (
        json_path
        if json_path is not None
        else Path("tmp/optimize/spread") / output.with_suffix(".json").name
    )
    dump_pareto_json(spreads, max_energy_shifts, metadata, effective_json, upf_path=upf)
    plot_pareto(effective_json, filename=output, loglog=loglog, logy=logy)
    click.echo(f"Pareto plot saved to {output}")


@optimize.command("rc")
@click.argument("upf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--ri-factor",
    type=float,
    default=DEFAULT_RI_FACTOR_MAX,
    show_default=True,
    help="Fixed inner radius factor.",
)
@click.option(
    "--threshold",
    type=float,
    required=True,
    help="Maximum allowed energy shift in Ha.",
)
@click.option(
    "--rc-min",
    type=float,
    default=None,
    help="Lower bound of the rc bracket.",
)
@click.option(
    "--rc-max",
    type=float,
    default=None,
    help="Upper bound of the rc bracket.",
)
@click.option(
    "--tol",
    type=float,
    default=0.05,
    show_default=True,
    help="Absolute tolerance on rc at which to stop bisecting.",
)
@click.option(
    "--json",
    "json_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the search result to this JSON file "
    "(default: tmp/optimize/rc_search/<upf_stem>.json).",
)
@add_option
def rc(
    upf: Path,
    ri_factor: float,
    threshold: float,
    rc_min: float | None,
    rc_max: float | None,
    tol: float,
    json_path: Path | None,
    add: tuple[str, ...],
) -> None:
    """Bisect over rc to find the smallest value satisfying an energy-shift threshold."""
    from kapaow.rc_search import dump_rc_search_json, find_smallest_rc
    from kapaow.solve import DEFAULT_RC_MAX, DEFAULT_RC_MIN

    extension = get_extension(add)
    rc_value, points = find_smallest_rc(
        upf,
        ri_factor=ri_factor,
        threshold=threshold,
        extension=extension,
        rc_min=rc_min if rc_min is not None else DEFAULT_RC_MIN,
        rc_max=rc_max if rc_max is not None else DEFAULT_RC_MAX,
        tol=tol,
    )
    effective_json = (
        json_path if json_path is not None else Path("tmp/optimize/rc_search") / f"{upf.stem}.json"
    )
    dump_rc_search_json(
        rc_value, points, ri_factor, threshold, effective_json, upf_path=upf, add=add
    )
    click.echo(f"Smallest rc: {rc_value:.4f}")
    click.echo(f"Result saved to {effective_json}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.argument("config_path", metavar="CONFIG", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory for benchmark outputs (default: tmp/benchmark/<config_stem>).",
)
@symmetrize_option
@experimental
def benchmark(  # noqa: C901  # CLI command orchestrates multiple AiiDA workflow stages
    config_path: Path,
    output: Path | None,
    symmetrize: bool,
) -> None:
    r"""Benchmark rival projectors by wannierizing a structure.

    Runs wannierization (as a metal) for each rival projector .dat file and
    compares disentanglement difficulty, convergence, and Wannier function
    spreads.

    CONFIG is a TOML file specifying the structure and projectors per element:

    \b
        structure = "LiF.cif"

        [Li]
        "Optimal PAO" = "projectors/Li_optimal.dat"
        "PseudoDojo"  = "projectors/Li_pseudodojo.dat"

        [O]
        "Standard" = "projectors/O.dat"

    \b
    Example:
        kapaow benchmark benchmark.toml
    """
    from kapaow._experimental.benchmark import (
        format_benchmark_table,
        generate_dat_files,
        plot_bands_comparison,
        plot_convergence,
        run_benchmark,
    )
    from kapaow._experimental.config import BenchmarkConfig, OptimizeDisThresholds
    from kapaow._experimental.workflows import run_bands_workflow

    cfg = BenchmarkConfig.from_toml(config_path)
    structure = cfg.structure

    dis_proj_max_value = cfg.wannier90.get("dis_proj_max")
    dis_proj_min_value = cfg.wannier90.get("dis_proj_min")
    optimize_mode = cfg.optimize_dis_thresholds

    working_dir = output or Path("tmp") / "benchmark" / config_path.stem

    # Generate .dat and bessel files from element configs
    combinations, bessel_combinations = generate_dat_files(cfg, working_dir)

    # Run DFT bands first (cached if already done). This gives us the exact
    # k-points used by QE so Wannier90 can interpolate on the same grid.
    bands_result = run_bands_workflow(
        structure,
        working_dir,
        min_nbnd=cfg.num_bands,
        kpath=cfg.kpath,
        periodic=cfg.periodic,
    )
    click.echo("DFT bands computed.")

    # The aiida-wannier90-workflows base workchain runs with
    # shift_energy_windows=True, which adds the LUMO (insulator) or Fermi
    # energy (metal) to dis_froz_max before writing the .win file. So we pass
    # the user's CBM-relative offset through unchanged.
    dis_froz_max_value: float | None = cfg.dis_froz_max_wrt_cbm
    if dis_froz_max_value is not None:
        click.echo(
            f"dis_froz_max = CBM + {dis_froz_max_value} eV "
            "(shifted to absolute by Wannier90 workflow)"
        )

    optimize_strategy = optimize_mode.value if optimize_mode != OptimizeDisThresholds.NONE else None

    click.echo(f"Running {len(combinations)} benchmark combination(s)...")
    effective_symmetrize = cfg.symmetrize or symmetrize
    if effective_symmetrize:
        click.echo("Symmetrizing projectors into bond-oriented hybrid basis.")
    results = run_benchmark(
        structure_file=structure,
        combinations=combinations,
        bessel_combinations=bessel_combinations,
        working_dir=working_dir,
        bands_kpoints_pk=bands_result.bands_kpoints_pk,
        dis_proj_max=dis_proj_max_value,
        dis_proj_min=dis_proj_min_value,
        dis_froz_max=dis_froz_max_value,
        extra_w90_params=cfg.wannier90 or None,
        optimize_strategy=optimize_strategy,
        otsu_bins=cfg.otsu_bins,
        reference_bands_pk=bands_result.reference_bands_pk,
        min_nbnd=cfg.num_bands,
        fermi_energy=bands_result.fermi_energy,
        periodic=cfg.periodic,
        symmetrize=effective_symmetrize,
        bond_cutoff=cfg.bond_cutoff,
    )

    click.echo(format_benchmark_table(results))

    # Compute fat bands if there is a single set of projectors
    channel_projectabilities = None
    if len(combinations) == 1:
        from kapaow._experimental.fat_bands import (
            build_atoms_dict,
            compute_amn_from_wfc,
            compute_projectability_per_channel,
        )
        from kapaow._experimental.projectability import _make_qe_input_wfc

        bessel_files = bessel_combinations[0]
        bands_calc_dir = bands_result.bands_calc_dir
        pwi_file = bands_calc_dir / "inputs" / "aiida.in"
        wfc_dir = bands_calc_dir / "outputs"
        atoms_dict, lattice_vectors = build_atoms_dict(pwi_file)
        qe_wfc = _make_qe_input_wfc(wfc_dir, lattice_vectors)
        num_kpoints = bands_result.band_plot_data.energies.shape[0]
        _smn, amn, cmn, channel_indices = compute_amn_from_wfc(
            qe_wfc=qe_wfc,
            bessel_files=bessel_files,
            atoms_dict=atoms_dict,
            lattice_vectors=lattice_vectors,
            num_kpoints=num_kpoints,
        )
        if effective_symmetrize:
            from kapaow._experimental.benchmark import _prepare_proj_dir
            from kapaow._experimental.symmetrize import (
                apply_rotation_to_amn,
                group_indices_by_label,
                symmetry_adapted_rotation,
            )

            # Stage the single combination's .dat files as ``{element}.dat``
            # under working_dir/fat_bands_projectors, matching the layout
            # symmetry_adapted_rotation expects.
            dat_map = combinations[0][1]
            proj_dir = _prepare_proj_dir(dat_map, dest_dir=working_dir / "fat_bands")
            rotation_matrix, labels = symmetry_adapted_rotation(
                structure_file=structure,
                proj_dir=proj_dir,
                atoms_dict=atoms_dict,
                lattice_vectors=lattice_vectors,
                hybridize=True,
                bond_cutoff=cfg.bond_cutoff,
                with_l_padding=True,
            )
            amn = apply_rotation_to_amn(amn, rotation_matrix)
            cmn = apply_rotation_to_amn(cmn, rotation_matrix)
            channel_indices = group_indices_by_label(labels)
        channel_projectabilities = compute_projectability_per_channel(amn, cmn, channel_indices)
        click.echo("Fat bands computed for single projector set.")

    seed = config_path.stem

    # Include optimization mode in figure names to avoid overwriting
    if optimize_strategy is not None:
        dpm_tag = f"_{optimize_strategy}"
    elif dis_proj_max_value is not None:
        dpm_tag = f"_dpm{dis_proj_max_value:.2f}".replace(".", "_")
    else:
        dpm_tag = ""

    convergence_plot = working_dir / f"{seed}_wannierization_trajectory{dpm_tag}.svg"
    plot_convergence(results, filename=convergence_plot)
    click.echo(f"Convergence plot saved to {convergence_plot}")

    # Plot Wannier-interpolated bands comparison against DFT bands
    bands_plot = working_dir / f"{seed}_interpolated_bands{dpm_tag}.svg"
    plot_bands_comparison(
        results,
        dft_band_plot_data=bands_result.band_plot_data,
        channel_projectabilities=channel_projectabilities,
        filename=bands_plot,
    )
    click.echo(f"Bands comparison plot saved to {bands_plot}")

    # If optimization was used, plot the trajectory
    if isinstance(dis_proj_max_value, str) and dis_proj_max_value == "optimize":
        from kapaow._experimental.benchmark import (
            extract_optimize_trajectory,
            plot_optimize_trajectory,
        )

        # Re-load the process node from the most recent optimize run
        # (stored as the last workgraph in working_dir/run_000/wannierize_optimize)
        try:
            from koopmans.aiida.setup import load_koopmans_profile

            load_koopmans_profile()
            from aiida.orm import QueryBuilder, WorkChainNode

            # Find the most recent Wannier90OptimizeWorkChain
            qb = QueryBuilder()
            qb.append(
                WorkChainNode,
                filters={"attributes.process_label": "Wannier90OptimizeWorkChain"},
                project=["*"],
            )
            qb.order_by({"*": {"ctime": "desc"}})
            qb.limit(1)
            optimize_nodes = qb.all()
            if optimize_nodes:
                opt_node = optimize_nodes[0][0]
                trials = extract_optimize_trajectory(opt_node)
                if trials:
                    traj_plot = working_dir / f"{seed}_optimize_trajectory.svg"
                    plot_optimize_trajectory(trials, filename=traj_plot)
                    click.echo(f"Optimize trajectory plot saved to {traj_plot}")
        except Exception as exc:
            # Best-effort enhancement: the rest of the benchmark output is
            # still valid even if we can't pull the optimization trajectory
            # back out of AiiDA. Log the full traceback (visible with
            # ``--log``) and surface a one-line note to the user.
            logger.exception("Failed to extract optimize trajectory")
            click.echo(
                f"Skipping optimize-trajectory plot ({type(exc).__name__}: {exc}). "
                "Run with --log for the full traceback."
            )


# ---------------------------------------------------------------------------
# animate
# ---------------------------------------------------------------------------


@main.command()
@click.argument("element_or_upf", type=str)
@click.option(
    "--frames-per-segment",
    type=int,
    default=10,
    help="Number of frames per path segment.",
)
@click.option(
    "--max-r-mid",
    type=float,
    default=15.0,
    help="Maximum midpoint of confining potential.",
)
@click.option(
    "--min-r-mid",
    type=float,
    default=5.0,
    help="Minimum midpoint of confining potential.",
)
@click.option(
    "--min-width",
    type=float,
    default=0.1,
    help="Minimum half-width of confining potential.",
)
@click.option(
    "--max-width",
    type=float,
    default=5.0,
    help="Maximum half-width of confining potential.",
)
@click.option(
    "--energy-shift-threshold",
    type=float,
    default=0.01,
    help="Energy shift threshold in Ha; orbitals above this are plotted in red.",
)
@add_option
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="output.gif",
    help="Save the output to this file.",
)
def animate(
    element_or_upf: str,
    frames_per_segment: int,
    max_r_mid: float,
    min_r_mid: float,
    min_width: float,
    max_width: float,
    energy_shift_threshold: float,
    output: Path,
    add: tuple[str, ...],
) -> None:
    """Generate a GIF showing PAOs under varying confinement.

    ELEMENT_OR_UPF is either an element symbol (e.g. Li) to use the bundled
    PseudoDojo pseudopotential, or a path to a UPF file.
    """
    from kapaow.animate import generate_animation

    upf_candidate = Path(element_or_upf)
    if upf_candidate.is_file():
        upf_path = upf_candidate
    else:
        from kapaow.data.pseudodojo import fetch_pseudopotential

        upf_path = fetch_pseudopotential(element_or_upf)

    extension = get_extension(add)

    def _on_frame(i: int, total: int, rc: float, ri_factor: float) -> None:
        click.echo(f"  Frame {i + 1}/{total}: rc={rc:.2f}, ri_factor={ri_factor:.2f}")

    generate_animation(
        upf_path=upf_path,
        extension=extension,
        frames_per_segment=frames_per_segment,
        max_r_mid=max_r_mid,
        min_r_mid=min_r_mid,
        min_width=min_width,
        max_width=max_width,
        energy_shift_threshold_ha=energy_shift_threshold,
        output=output,
        on_frame=_on_frame,
    )
    click.echo(f"GIF saved to {output}")


@main.group()
def plot() -> None:
    """Plot PAO-related data."""


def with_output_option(default_format: str, from_config: str | None = None) -> Callable:
    """Reusable Click option for output file paths.

    Parameters
    ----------
    default_format
        File extension including the dot (e.g. ``".svg"``).
    from_config
        If set, the name of the Click parameter that holds a config file
        path.  When ``--output`` is not given, the default becomes
        ``<config_stem><default_format>`` instead of ``output<default_format>``.
    """

    def decorator(func: Callable) -> click.Command:
        if from_config is None:
            return click.option(
                "--output",
                "-o",
                type=click.Path(path_type=Path),
                default=f"output{default_format}",
                help="Save the output to this file.",
            )(func)

        # Wrap the function to resolve the default from the config file
        import functools

        @click.option(
            "--output",
            "-o",
            type=click.Path(path_type=Path),
            default=None,
            help=f"Save the output to this file (default: <config_stem>{default_format}).",
        )
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("output") is None:
                config_path = kwargs[from_config]
                kwargs["output"] = Path(config_path).with_suffix(default_format)
            return func(*args, **kwargs)

        return wrapper

    return decorator


@plot.command()
@click.argument("dat", nargs=-1, type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".png")
def paos(dat: list[Path], output: Path) -> None:
    """Plot the pseudoatomic orbitals stored in a Wannier90 .dat file."""
    plot_wannier90_dat_files(dat, filename=output)


@plot.command(name="projectability-optimization")
@click.argument("log_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".png")
@experimental
def projectability_optimization(log_file: Path, output: Path | None) -> None:
    """Plot the results of a projectability optimization."""
    from kapaow._experimental.optimize import plot_optimizer

    plot_optimizer(log_file, filename=output)


@plot.command(name="spread-optimization")
@click.argument("json_file", type=click.Path(exists=True, path_type=Path))
@with_output_option(default_format=".svg")
@click.option("--loglog", is_flag=True, default=False, help="Use log-log axes.")
@click.option("--logy", is_flag=True, default=False, help="Use log scale for the y-axis only.")
def spread_optimization(json_file: Path, output: Path, loglog: bool, logy: bool) -> None:
    """Plot a spread optimization Pareto front from an existing JSON file."""
    from kapaow.pareto import plot_pareto

    plot_pareto(json_file, filename=output, loglog=loglog, logy=logy)
    click.echo(f"Pareto plot saved to {output}")


@plot.command(name="periodic-table")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--color-by",
    type=click.Choice(["projectability", "spread", "rc"]),
    required=True,
    help="Metric to color the periodic table by.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.01,
    show_default=True,
    help="Energy shift threshold in Ha (only used with --color-by spread).",
)
@with_output_option(default_format=".html")
def periodic_table_cmd(directory: Path, color_by: str, threshold: float, output: Path) -> None:
    """Plot a periodic table colored by a PAO quality metric.

    DIRECTORY contains per-element data files: optimizer log JSONs for
    --color-by projectability, Pareto front JSONs for --color-by spread,
    or rc-search JSONs for --color-by rc.
    """
    if color_by == "projectability":
        plot_periodic_table(directory, output=output)
    elif color_by == "spread":
        from kapaow.periodic_table import plot_pareto_periodic_table

        plot_pareto_periodic_table(directory, output=output, threshold_ha=threshold)
    elif color_by == "rc":
        from kapaow.periodic_table import plot_rc_periodic_table

        plot_rc_periodic_table(directory, output=output)


@plot.command(name="fat-bands")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--working-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory for intermediate files (default: tmp/fat_bands/<config_stem>).",
)
@click.option(
    "--emin",
    type=float,
    default=None,
    help="Minimum energy relative to Fermi level (eV).",
)
@click.option(
    "--emax",
    type=float,
    default=None,
    help="Maximum energy relative to Fermi level (eV).",
)
@click.option(
    "--num-bands",
    type=int,
    default=None,
    help="Manual override for the number of bands in the DFT calculation.",
)
@symmetrize_option
@with_output_option(default_format=".svg", from_config="config_file")
@experimental
def fat_bands(
    config_file: Path,
    working_dir: Path | None,
    emin: float | None,
    emax: float | None,
    num_bands: int | None,
    symmetrize: bool,
    output: Path,
) -> None:
    r"""Plot fat bands colored by projectability.

    CONFIG_FILE is a TOML file specifying the structure and per-element
    pseudopotentials. Example:

    \b
        structure = "LiF.cif"
    \b
        [Li]
        upf = "Li.upf"
        rc = 15.0
        ri_factor = 0.3
        extension = "subshell"
    \b
        [F]
        upf = "F.upf"
    """
    from kapaow._experimental.fat_bands import generate_fat_bands_from_config

    output_files = generate_fat_bands_from_config(
        config_path=config_file,
        working_dir=working_dir,
        emin=emin,
        emax=emax,
        filename=output,
        num_bands=num_bands,
        symmetrize=symmetrize,
    )
    for f in output_files:
        click.echo(f"Fat bands plot saved to {f}")


@plot.command(name="compare-projectability")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--working-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: tmp/proj_comparison/<config_stem>).",
)
@click.option(
    "--emin",
    type=float,
    default=None,
    help="Minimum energy relative to Fermi level (eV).",
)
@click.option(
    "--emax",
    type=float,
    default=None,
    help="Maximum energy relative to Fermi level (eV).",
)
@click.option(
    "--num-bands",
    type=int,
    default=None,
    help="Manual override for the number of bands in the DFT calculation.",
)
@with_output_option(default_format=".svg", from_config="config_file")
@experimental
def compare_projectability(
    config_file: Path,
    working_dir: Path | None,
    emin: float | None,
    emax: float | None,
    num_bands: int | None,
    output: Path,
) -> None:
    r"""Compare total projectability across different basis sets.

    CONFIG_FILE is a TOML file where elements can have multiple entries
    (using [[Element]] syntax) to define comparison sets. Example:

    \b
        structure = "LiF.cif"
    \b
        [[Li]]
        openmx = true
        rc = 8.0
        label = "OpenMX rc=8"
    \b
        [[Li]]
        upf = "Li.upf"
        rc = 15.0
        label = "UPF rc=15"
    \b
        [F]
        upf = "F.upf"
        rc = 15.0
    """
    from kapaow._experimental.fat_bands import generate_projectability_comparison

    generate_projectability_comparison(
        config_path=config_file,
        working_dir=working_dir,
        emin=emin,
        emax=emax,
        filename=output,
        num_bands=num_bands,
    )
    click.echo(f"Projectability comparison saved to {output}")


@plot.command(name="compare-gauge-matrices")
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--working-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory (default: tmp/matrix_comparison/<config_stem>).",
)
@click.option(
    "--num-bands",
    type=int,
    default=None,
    help="Manual override for the number of bands in the DFT calculation.",
)
@experimental
def compare_gauge_matrices(
    config_file: Path,
    working_dir: Path | None,
    num_bands: int | None,
) -> None:
    r"""Compare A_mn^k and U_mn^k matrices across different PAO choices.

    Computes the projection matrix A and its unitary polar factor U (from
    SVD) for each PAO set defined in the TOML config, then prints a
    Cartesian product table of pairwise distances.  The distance metric is
    the square root of the BZ-averaged squared Frobenius norm of the
    difference.

    CONFIG_FILE uses the same ``[[Element]]`` syntax as compare-projectability.

    \b
    Example:
        kapaow plot compare-gauge-matrices LiF_projectability_comparison.toml
    """
    import numpy as np

    from kapaow._experimental.gauge import compare_matrices, format_distance_table

    result = compare_matrices(config_file, working_dir=working_dir, num_bands=num_bands)
    click.echo(format_distance_table(result.matrix_labels, result.distance_table))

    def _fmt_angles(a: Any) -> str:
        return "  ".join(f"{v:5.2f}" for v in np.degrees(a))

    for label_i, label_j, angles_a, angles_u in result.principal_angle_comparisons:
        click.echo(f"\nPrincipal angles: {label_i} vs {label_j}")
        click.echo(f"  from A†A: {_fmt_angles(angles_a)} deg")
        click.echo(f"  from U†U: {_fmt_angles(angles_u)} deg")


@plot.command(name="unshifted-vs-rc")
@click.argument("grid_directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--threshold",
    type=float,
    default=0.01,
    show_default=True,
    help="Energy shift threshold in Ha.",
)
@with_output_option(default_format=".svg")
@experimental
def unshifted_vs_rc(grid_directory: Path, threshold: float, output: Path) -> None:
    """Plot cumulative fraction of elements below energy shift threshold vs rc.

    GRID_DIRECTORY is a directory containing per-element grid JSON files.
    """
    from kapaow._experimental.analysis import plot_cumulative_below_threshold

    plot_cumulative_below_threshold(
        grid_directory,
        threshold_ha=threshold,
        filename=output,
    )
    click.echo(f"Unshifted-vs-rc plot saved to {output}")


@plot.command(name="optimize-trajectory")
@click.argument("pk", type=int)
@with_output_option(default_format=".svg")
@experimental
def optimize_trajectory(pk: int, output: Path) -> None:
    """Plot bands distance vs dis_proj_max from a Bayesian optimization run.

    PK is the AiiDA PK of the Wannier90OptimizeWorkChain or its parent
    workgraph process node.
    """
    import logging

    from kapaow._experimental.benchmark import extract_optimize_trajectory, plot_optimize_trajectory

    logging.basicConfig(level=logging.INFO)

    from koopmans.aiida.setup import load_koopmans_profile

    load_koopmans_profile()

    from aiida import orm

    process_node = orm.load_node(pk)
    trials = extract_optimize_trajectory(process_node)

    if not trials:
        click.echo("No optimization trials found.")
        return

    click.echo(f"Found {len(trials)} trial(s):")
    for i, t in enumerate(trials):
        dist_str = f"{t.bands_distance:.4f} eV" if t.bands_distance is not None else "failed"
        click.echo(f"  {i + 1}. dis_proj_max={t.dis_proj_max:.4f}  bands_distance={dist_str}")

    plot_optimize_trajectory(trials, filename=output)
    click.echo(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
