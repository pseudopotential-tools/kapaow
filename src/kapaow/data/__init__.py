"""Reference data bundled with kapaow.

* :mod:`kapaow.data.sssp.structures` — CIF files used by the experimental
  projectability optimiser.
* :mod:`kapaow.data.openmx` — OpenMX PAO/VPS reference data used by the
  experimental fat-bands and benchmark commands.

These directories are vendored in git for the repo checkout but excluded
from the published sdist/wheel via ``[tool.uv.build-backend]`` so users
who ``pip install kapaow`` get a small package and the experimental
analysis tools fall back to user-supplied paths.

UPF/pseudopotential generation data lives in the separate kapaow_datasets
repo.
"""
