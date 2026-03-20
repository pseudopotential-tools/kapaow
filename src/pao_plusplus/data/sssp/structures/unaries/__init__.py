"""Unary structure files for the SSSP benchmark set."""

from collections.abc import Generator
from pathlib import Path


def input_files(element: str) -> Generator[Path, None, None]:
    """Return all unary structure files for the provided element."""
    yield from Path(__file__).parent.glob(f"{element}-*.xsf")
