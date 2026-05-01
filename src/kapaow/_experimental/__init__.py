"""Experimental, AiiDA-backed extensions to kapaow.

The modules under :mod:`kapaow._experimental` orchestrate AiiDA + Quantum
ESPRESSO + Wannier90 workflows (benchmarks, fat bands, gauge analysis,
Bayesian projectability optimisation, etc.). They depend on a heavy
external stack (``aiida-core``, ``aiida-quantumespresso``,
``aiida-wannier90-workflows``, ``aiida-koopmans``, ``qe-wavefunctions``,
``koopmans``, ``bayes_opt``) that is not required for the core
``kapaow`` install.

Importing anything from this subpackage triggers the import-time check
below: if AiiDA is unavailable, a clear :class:`ImportError` tells the
user how to pull in the extras. Code in the core ``kapaow`` namespace
must never import from here unconditionally — defer imports inside CLI
command bodies (or other narrow entry points) so the core package stays
installable on its own.
"""

from __future__ import annotations

try:
    import aiida  # noqa: F401  -- presence check only
except ImportError as exc:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "kapaow._experimental requires the 'workflows' extras. "
        "Install with: pip install 'kapaow[workflows]'"
    ) from exc
