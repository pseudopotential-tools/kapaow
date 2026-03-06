"""A customised engine for running koopmans workflows that can stop early."""

from collections.abc import Callable
from pathlib import Path

from koopmans.calculators import PW2WannierCalculator, Wannier90Calculator
from koopmans.commands import CommandConfigs
from koopmans.engines.localhost import LocalhostEngine
from koopmans.processes import ProcessProtocol


# A custom Exception to catch when we want the workflow to stop early
class PW2WannierCompletedError(Exception):
    """Custom exception to indicate pw2wannier step completion."""


class Wannier90PPCompletedError(Exception):
    """Custom exception to indicate Wannier90 preprocessing step completion."""


def stop_after_wannier90pp(step: ProcessProtocol, _: list[str]) -> bool:
    """Stop the workflow after the Wannier90 preprocessing step i.e. before the pw2wannier step."""
    return isinstance(step, PW2WannierCalculator)


def stop_after_pw2wannier(step: ProcessProtocol, additional_flags: list[str]) -> bool:
    """Stop the workflow after the pw2wannier step i.e. before the Wannier90 step."""
    return isinstance(step, Wannier90Calculator) and additional_flags == []


class LocalhostEngineThatStopsEarly(LocalhostEngine):
    """A LocalhostEngine that is patched to stop the full Wannierize workflow prematurely."""

    stop_condition: Callable[[ProcessProtocol, list[str]], bool]
    stop_exception: type[Exception]

    def run(self, step: ProcessProtocol, additional_flags: list[str] | None = None) -> None:
        """Run a process step, stopping early if the stop condition is met."""
        additional_flags = additional_flags or []

        if self.stop_condition(step, additional_flags):
            raise self.stop_exception()
        return super().run(step, additional_flags)


def commands_from_qe_bin(qe_bin: Path | None) -> CommandConfigs:
    """Create a CommandConfigs with QE executables from the given bin directory."""
    if qe_bin is None:
        return CommandConfigs()
    return CommandConfigs(
        pw={"executable": str(qe_bin / "pw.x")},
        pw2wannier90={"executable": str(qe_bin / "pw2wannier90.x")},
    )
