"""Ground-state structure files for the SSSP benchmark set."""

from collections.abc import Generator
from pathlib import Path

STRUCTURE_EXTENSIONS = ("*.cif", "*.xsf")


def input_files(element: str) -> Generator[Path, None, None]:
    """Return all structure files for the provided element."""
    parent = Path(__file__).parent
    for ext in STRUCTURE_EXTENSIONS:
        elemental_match = parent / f"{element}{ext[1:]}"
        if elemental_match.exists():
            yield elemental_match
            return
    # For some elements, the ground state is not elemental
    for ext in STRUCTURE_EXTENSIONS:
        yield from parent.glob(f"{element}*{ext[1:]}")
