"""Quantum ESPRESSO input files."""

from collections.abc import Generator
from pathlib import Path

all_input_files = Path(__file__).parent.glob("*.pwi")


def input_files(element: str) -> Generator[Path, None, None]:
    """Return all input files for the provided element."""
    elemental_match = Path(__file__).parent / f"{element}.pwi"
    if elemental_match.exists():
        yield elemental_match
    else:
        # For some elements, the ground state is not elemental
        yield from Path(__file__).parent.glob(f"{element}*.pwi")
