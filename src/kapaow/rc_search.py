"""Bisection search for the smallest rc satisfying an energy-shift threshold."""

import json
import logging
from pathlib import Path

from upf_tools import UPFDict

from kapaow.basis import AtomicBasis
from kapaow.extend import BasisExtension, BasisExtensionViaAddition
from kapaow.pareto import _evaluate_point
from kapaow.solve import DEFAULT_RC_MAX, DEFAULT_RC_MIN

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "dump_rc_search_json",
    "find_smallest_rc",
]


def find_smallest_rc(
    upf_path: Path,
    ri_factor: float,
    threshold: float,
    extension: BasisExtension | None = None,
    rc_min: float = DEFAULT_RC_MIN,
    rc_max: float = DEFAULT_RC_MAX,
    tol: float = 0.05,
    working_dir: Path = Path("tmp/optimize/rc_search"),
) -> tuple[float, list[dict]]:
    """Find the smallest rc such that all energy shifts are below ``threshold``.

    Performs a bisection search over rc at fixed ``ri_factor``.  At each
    candidate rc the pseudoatomic problem is solved and the maximum
    absolute energy shift (taken over the original-basis orbitals, as in
    :func:`kapaow.pareto._evaluate_point`) is compared against
    ``threshold``.  Failures to converge are treated as not satisfying
    the threshold.

    Parameters
    ----------
    upf_path
        Path to the UPF pseudopotential file.
    ri_factor
        Fixed inner-radius factor.
    threshold
        Energy-shift threshold (in Hartree).  The search returns the
        smallest rc whose maximum energy shift is strictly less than
        this value.
    extension
        Optional basis extension.
    rc_min, rc_max
        Bracket for the bisection.
    tol
        Absolute tolerance on rc at which to stop bisecting.
    working_dir
        Directory for intermediate files.

    Returns
    -------
    rc_value, points
        The smallest rc (to within ``tol``) satisfying the threshold,
        and a list of per-probe dicts with ``rc``, ``max_energy_shift``,
        and ``satisfies`` keys (in evaluation order).

    Raises
    ------
    RuntimeError
        If even ``rc_max`` does not satisfy the threshold.
    """
    upf_dict = UPFDict.from_upf(upf_path)
    element = upf_dict["header"]["element"].strip()
    original_basis = AtomicBasis.from_upf(upf_path)
    if extension is not None and isinstance(extension, BasisExtensionViaAddition):
        atomic_basis = extension.extend_atomic(original_basis)
    else:
        atomic_basis = original_basis

    working_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []

    def satisfies(rc: float) -> bool:
        out = _evaluate_point(
            upf_path,
            rc,
            ri_factor,
            extension,
            element,
            atomic_basis,
            original_basis,
            working_dir,
        )
        if out is None:
            logger.info("  rc=%.4f: failed (no result)", rc)
            points.append({"rc": rc, "max_energy_shift": None, "satisfies": False})
            return False
        _, max_shift, _ = out
        ok = max_shift < threshold
        logger.info(
            "  rc=%.4f: max_energy_shift=%.6f %s threshold=%.6f",
            rc,
            max_shift,
            "<" if ok else ">=",
            threshold,
        )
        points.append({"rc": rc, "max_energy_shift": max_shift, "satisfies": ok})
        return ok

    logger.info(
        "Searching for smallest rc in [%.4f, %.4f] with ri_factor=%.4f, threshold=%.6f",
        rc_min,
        rc_max,
        ri_factor,
        threshold,
    )

    if not satisfies(rc_max):
        raise RuntimeError(
            f"rc_max={rc_max} does not satisfy the energy-shift threshold "
            f"{threshold} at ri_factor={ri_factor}; widen the bracket."
        )

    if satisfies(rc_min):
        logger.info("rc_min=%.4f already satisfies threshold", rc_min)
        return rc_min, points

    lo, hi = rc_min, rc_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if satisfies(mid):
            hi = mid
        else:
            lo = mid

    logger.info("Smallest rc satisfying threshold: %.4f", hi)
    return hi, points


def dump_rc_search_json(
    rc_value: float,
    points: list[dict],
    ri_factor: float,
    threshold: float,
    path: Path,
    upf_path: Path | None = None,
    add: tuple[str, ...] = (),
) -> None:
    """Write the rc search result and all probed points to a JSON file.

    The output also records ``kapaow_version`` (the running CLI's
    package version) so downstream tooling can stamp provenance from
    the JSON rather than guessing from its own ``importlib.metadata``.
    """
    from importlib.metadata import version as _pkg_version

    output: dict = {}
    if upf_path is not None:
        output["upf_path"] = str(upf_path)
    output["kapaow_version"] = _pkg_version("kapaow")
    output["ri_factor"] = ri_factor
    output["threshold"] = threshold
    output["add"] = list(add)
    output["rc"] = rc_value
    output["points"] = points
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    logger.info("rc search result (%d probes) saved to %s", len(points), path)
