"""Quantum ESPRESSO input files for the SSSP benchmark set."""

from pathlib import Path
from typing import Generator
from .gs import input_files as gs_input_files
from .oxides import input_files as oxides_input_files
from .unaries import input_files as unaries_input_files

def input_files(element: str) -> Generator[Path, None, None]:
    """Return all input files for the provided element."""
    yield from gs_input_files(element)
    yield from oxides_input_files(element)
    yield from unaries_input_files(element)
    

