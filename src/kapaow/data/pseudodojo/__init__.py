"""PseudoDojo pseudopotential data access."""

from pathlib import Path


def fetch_pseudopotential(
    element: str,
    version: str = "0.5",
    functional: str = "pbe",
    protocol: str = "standard",
) -> Path:
    """Return the path to a PseudoDojo pseudopotential file."""
    upf = (
        Path(__file__).parent
        / f"nc-sr-{version.replace('.', '')}_{functional}_{protocol}_upf"
        / f"{element}.upf"
    )
    if not upf.exists():
        raise FileNotFoundError(f"{upf} does not exist.")
    return upf
