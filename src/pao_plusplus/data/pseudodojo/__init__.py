from pathlib import Path

def fetch_pseudopotential(element: str,
                          version: str = "0.5",
                          functional: str = "pbe",
                          protocol: str = "standard") -> Path:
    upf = Path(__file__).parent / f"nc-sr-{version.replace('.', '')}" \
        f"_{functional}_{protocol}_upf/{element}.upf"
    if not upf.exists():
        raise FileNotFoundError(f"{upf} does not exist.")
    return upf
