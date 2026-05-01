"""OpenMX pseudoatomic orbital data access."""

from pathlib import Path

PAO_DIR = Path(__file__).parent / "PAO"


def fetch_pao(element: str, rc: float) -> Path:
    """Return the path to a bundled OpenMX .pao file.

    Parameters
    ----------
    element
        Element symbol (e.g. ``"Li"``).
    rc
        Cutoff radius in Bohr (e.g. ``8.0``).

    Returns
    -------
    Path
        Path to the ``.pao`` file.

    Raises
    ------
    FileNotFoundError
        If no matching .pao file is found. Lists available files for the element.
    """
    # Format rc to match OpenMX naming: "8.0", "10.0", etc.
    rc_str = f"{rc:.1f}" if rc == int(rc) else f"{rc:g}"
    pao = PAO_DIR / f"{element}{rc_str}.pao"
    if not pao.exists():
        available = sorted(p.name for p in PAO_DIR.glob(f"{element}*.pao"))
        raise FileNotFoundError(
            f"No .pao file found for {element} with rc={rc_str}. "
            f"Available: {available}"
        )
    return pao
