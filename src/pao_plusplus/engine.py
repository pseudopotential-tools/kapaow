"""A customised engine for running koopmans workflows that can stop early."""

from collections.abc import Callable

from koopmans.calculators import PW2WannierCalculator, Wannier90Calculator
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
