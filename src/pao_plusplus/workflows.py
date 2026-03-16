"""Projectability module for pao_plusplus."""

import warnings
from collections.abc import Iterator
from contextlib import redirect_stdout
from os.path import relpath
from pathlib import Path
from typing import Any

from koopmans.io import read as koopmans_read
from koopmans.kpoints import Kpoints
from koopmans.utils import Spin, chdir
from koopmans.utils.warnings import CalculatorNotConvergedWarning
from koopmans.workflows import WannierizeWorkflow

from pao_plusplus.engine import (
    BandsCompletedError,
    LocalhostEngineThatStopsEarly,
    PW2WannierCompletedError,
    Wannier90PPCompletedError,
    commands_from_qe_bin,
    stop_after_bands,
    stop_after_pw2wannier,
    stop_after_wannier90pp,
)

PSEUDO_LIBRARY = "pao_plusplus"


KPOINT_PATCHES: dict[str, list[int]] = {
    "Zn-SC.pwi": [16, 16, 16],
    "In.pwi": [18, 18, 18],
    "Ir.pwi": [19, 19, 19],
    "Sb-FCC.pwi": [16, 16, 16],
    "In-XO2.pwi": [10, 10, 10],
    "In-X205.pwi": [8, 8, 8],
}

def pwi_to_workflow(
    pwi_file: Path, proj_dir: Path, engine: LocalhostEngineThatStopsEarly, diagonalization: str = 'david',
    calculate_bands: bool = True, min_nbnd: int | None = None,
) -> WannierizeWorkflow:
    """Construct a Wannierize workflow from a pw.x input file."""
    calculator = koopmans_read(pwi_file)
    atoms = calculator.atoms
    atoms.calc = None
    pw_params = calculator.parameters
    pw_params.prefix = "kc"
    pw_params.electron_maxstep = 2000
    pw_params.pop("pseudo_dir")
    kpoints = Kpoints(grid=KPOINT_PATCHES.get(pwi_file.name, calculator.parameters["kpts"]))
    pw_params.diagonalization = diagonalization
    ecutwfc = pw_params.pop("ecutwfc")
    ecutrho = pw_params.pop("ecutrho")

    calculator_parameters = {
        "pw": pw_params,
        "w90": {"auto_projections": True},
        "pw2wannier": {
            "atom_proj_ext": True,
            "atom_proj_dir": proj_dir.resolve(),
            "write_mmn": False,
        },
    }

    # Ensure .dat files exist for all species (needed by koopmans to count projectors)
    for element in {atom.symbol for atom in atoms}:
        dst = proj_dir / f"{element}.dat"
        if not dst.exists():
            pseudo = engine.get_pseudopotential(PSEUDO_LIBRARY, element)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(pseudo.to_dat())
              
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        workflow = WannierizeWorkflow(
            atoms=atoms,
            engine=engine,
            pseudo_library=PSEUDO_LIBRARY,
            kpoints=kpoints,
            calculator_parameters=calculator_parameters,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            init_orbitals="mlwfs",
            init_empty_orbitals="mlwfs",
            name=pwi_file.stem,
        )

    # Make sure we include (more than) enough bands to ensure we get all the
    # atomic-like bands
    num_wann = workflow.projections.num_bands(spin=Spin.NONE)
    nbnd = int(1.5 * num_wann)
    if min_nbnd is not None:
        nbnd = max(nbnd, min_nbnd)
    workflow.calculator_parameters["pw"]["nbnd"] = nbnd

    workflow.parameters.calculate_bands = calculate_bands

    return workflow


def run_qe_workflow(
    pwi_file: Path,
    pw_working_dir: Path,
    pseudo_files: Iterator[Path],
    diagonalization: str = 'david',
    qe_bin: Path | None = None,
    min_nbnd: int | None = None,
) -> WannierizeWorkflow:
    """Run scf + nscf + bands, stopping before wannier90 -pp.

    Uses the koopmans WannierizeWorkflow infrastructure with early stopping.
    Results are cached: if the working directory already contains completed
    calculations, they will be reused.
    """
    commands = commands_from_qe_bin(qe_bin)

    engine = LocalhostEngineThatStopsEarly(
        commands=commands,
        stop_condition=stop_after_bands,
        stop_exception=BandsCompletedError,
        from_scratch=False,
    )
    for f in pseudo_files:
        engine.install_pseudopotential(f, library=PSEUDO_LIBRARY)
    workflow = pwi_to_workflow(pwi_file, proj_dir=pw_working_dir, engine=engine, diagonalization=diagonalization, min_nbnd=min_nbnd)
    with chdir(pw_working_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CalculatorNotConvergedWarning)
            try:
                workflow.run()
            except engine.stop_exception:
                pass

    return workflow


def run_wannierize_workflow(
    pwi_file: Path,
    proj_dir: Path,
    w90_working_dir: Path,
    pw_working_dir: Path,
    pseudo_files: Iterator[Path],
    diagonalization: str = 'david',
    qe_bin: Path | None = None,
) -> WannierizeWorkflow:
    """Run the Wannierize workflow, using pre-computed qe results where available."""
    commands = commands_from_qe_bin(qe_bin)

    # First, run the parts of the workflow that don't need to be re-evaluated
    # if the projector changes
    # Run the qe part of the workflow
    engine = LocalhostEngineThatStopsEarly(
        commands=commands,
        stop_condition=stop_after_wannier90pp,
        stop_exception=Wannier90PPCompletedError,
        from_scratch=False,
    )
    for f in pseudo_files:
        engine.install_pseudopotential(f, library=PSEUDO_LIBRARY)
    workflow = pwi_to_workflow(pwi_file, proj_dir, engine=engine, diagonalization=diagonalization)
    with chdir(pw_working_dir):
        # with open("koopmans.md", "w", encoding="utf-8") as koopmans_output:
        #     with redirect_stdout(koopmans_output):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CalculatorNotConvergedWarning)
            try:
                workflow.run()
            except engine.stop_exception:
                pass

    # Link all the files from the pw_working_dir to the w90_working_dir
    w90_working_dir.mkdir(parents=True, exist_ok=True)
    for f in pw_working_dir.rglob("*"):
        if f.is_dir():
            continue
        target = w90_working_dir / f.relative_to(pw_working_dir)
        if target.exists():
            continue
        relative_path = relpath(f, target.parent)
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(relative_path)

    # Run the projector-dependent part of the workflow
    engine = LocalhostEngineThatStopsEarly(
        commands=commands,
        stop_condition=stop_after_pw2wannier,
        stop_exception=PW2WannierCompletedError,
        from_scratch=False,
    )
    for f in pseudo_files:
        engine.install_pseudopotential(f, library=PSEUDO_LIBRARY)

    workflow = pwi_to_workflow(pwi_file, proj_dir, engine=engine)
    with chdir(w90_working_dir):
        with open("koopmans.md", "w", encoding="utf-8") as koopmans_output:
            with redirect_stdout(koopmans_output):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", CalculatorNotConvergedWarning)
                        workflow.run()
                except engine.stop_exception:
                    pass

    return workflow
