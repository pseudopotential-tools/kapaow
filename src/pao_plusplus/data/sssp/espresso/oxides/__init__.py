"""Quantum ESPRESSO input files."""

from pathlib import Path
from typing import Generator

all_input_files = Path(__file__).parent.glob("*.pwi")

def input_files(element: str) -> Generator[Path, None, None]:
    yield from Path(__file__).parent.glob(f"{element}-*.pwi")
    return
