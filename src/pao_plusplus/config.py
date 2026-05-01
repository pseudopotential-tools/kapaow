"""Pydantic models for TOML workflow configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

import enum


class OptimizeDisThresholds(enum.Enum):
    """Strategy for choosing dis_proj_min / dis_proj_max."""

    NONE = "none"
    BAYESIAN = "bayesian"
    GRID = "grid"
    OTSU = "otsu"


from pao_plusplus.extend import (
    BasisExtension,
    BasisExtensionType,
    BasisExtensionViaAddition,
    BasisExtensionViaChannel,
    BasisExtensionViaPolarization,
)
from pao_plusplus.pydantic import BaseModel
from pao_plusplus.solve import DEFAULT_RC_MAX, DEFAULT_RI_FACTOR_MAX


class UpfConfig(BaseModel):
    """Configuration for an element using a UPF pseudopotential."""

    upf: Path
    """Path to the UPF pseudopotential file."""
    label: str | None = Field(default=None, description="Label for comparison plots.")
    rc: float = Field(default=DEFAULT_RC_MAX, description="Confinement radius (Bohr).")
    ri_factor: float = Field(
        default=DEFAULT_RI_FACTOR_MAX,
        description="Inner confinement radius as fraction of rc.",
    )
    extension: BasisExtensionType | None = Field(
        default=None,
        description="Basis extension type.",
    )
    extension_increment: int = Field(
        default=1,
        description="Number of orbitals to add via the extension.",
    )

    def get_extension(self) -> BasisExtension | None:
        """Convert the extension type to a BasisExtension object."""
        if self.extension is None:
            return None
        if self.extension == BasisExtensionType.SUBSHELL:
            return BasisExtensionViaAddition(increment=self.extension_increment)
        if self.extension == BasisExtensionType.POLARIZATION:
            return BasisExtensionViaPolarization(increment=self.extension_increment)
        channel = self.extension.angular_momentum
        if channel is not None:
            return BasisExtensionViaChannel(channel=channel, increment=self.extension_increment)
        raise ValueError(f"Unknown extension type: {self.extension}")


class PaoConfig(BaseModel):
    """Configuration for an element using a bundled OpenMX .pao file.

    Set ``openmx = true`` and specify ``rc`` to use the bundled .pao file
    for the element.
    """

    openmx: Literal[True]
    """Must be ``true`` to trigger OpenMX .pao lookup."""
    label: str | None = Field(default=None, description="Label for comparison plots.")
    rc: float = Field(description="Cutoff radius (Bohr) for the .pao file.")
    select: list[str] | None = Field(
        default=None,
        description="Orbital selection list (e.g. ['s', 's', 'p', 'd'] for 2s, 1p, 1d).",
    )


ElementConfig = Annotated[
    Union[UpfConfig, PaoConfig],
    Field(discriminator=None),
]


# ---- Shared fields for all workflow TOML configs ----

_NON_ELEMENT_KEYS = {
    "structure", "num_bands", "kpath", "periodic",
    "dis_froz_max_wrt_cbm",
    "optimize_dis_thresholds", "otsu_bins",
    "wannier90",
    "symmetrize", "bond_cutoff",
}


class _WorkflowConfigBase(BaseModel, extra="allow"):
    """Common fields shared by fat-bands, projectability comparison, and benchmark configs."""

    structure: Path
    """Path to the structure file (CIF, XSF, etc.)."""

    num_bands: int | None = Field(
        default=None, description="Manual override for the number of bands in the DFT calculation."
    )

    kpath: list[list[str]] | None = Field(
        default=None,
        description='Manual k-path as continuous segments, e.g. [["GAMMA", "M", "K", "GAMMA"]]. '
        "Use multiple inner lists for discontinuities.",
    )

    periodic: tuple[bool, bool, bool] = Field(
        default=(True, True, True),
        description="Periodic boundary conditions along (a, b, c). "
        "Set to [true, true, false] for 2D systems with vacuum along c; "
        "QE's `assume_isolated = '2D'` is then applied automatically by "
        "aiida-quantumespresso. Defaults to fully periodic.",
    )

    otsu_bins: int = Field(
        default=5,
        description="Number of Otsu classes for disentanglement threshold detection.",
    )

    wannier90: dict[str, object] = Field(
        default_factory=dict,
        description="Extra Wannier90 parameters passed directly to the .win file. "
        "Keys are Wannier90 input keywords (e.g. dis_conv_tol, num_iter).",
    )

    @classmethod
    def from_toml(cls, path: Path):
        """Load from a TOML file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls.model_validate(raw)


class FatBandsConfig(_WorkflowConfigBase):
    """TOML configuration for fat bands computation.

    Top-level keys:
    - ``structure``: path to a CIF/XSF structure file.
    - ``num_bands``: optional manual override for the number of bands.
    - Any other key is treated as an element name with a ``UpfConfig`` or
      ``PaoConfig``.

    Example TOML::

        structure = "LiF.cif"

        [Li]
        openmx = true
        rc = 8.0
        select = "ssp"

        [F]
        upf = "F.upf"
        rc = 15.0
    """

    elements: dict[str, UpfConfig | PaoConfig] = Field(default_factory=dict)
    """Per-element configurations, keyed by element symbol."""

    @model_validator(mode="before")
    @classmethod
    def _extract_elements(cls, data: dict) -> dict:
        """Pull element sections out of the flat TOML namespace."""
        elements = {}
        for key in list(data.keys()):
            if key not in _NON_ELEMENT_KEYS and isinstance(data[key], dict):
                elements[key] = data.pop(key)
        data["elements"] = elements
        return data


class ProjectabilityComparisonConfig(_WorkflowConfigBase):
    """TOML configuration for comparing projectability across basis sets.

    Each element can have a single config (``[Element]``) or a list
    (``[[Element]]``).  Elements with a single entry are shared across
    all comparison sets; elements with multiple entries define the sets.

    Example TOML::

        structure = "LiF.cif"

        [[Li]]
        openmx = true
        rc = 8.0
        label = "OpenMX rc=8"

        [[Li]]
        upf = "Li.upf"
        rc = 15.0
        label = "UPF rc=15"

        [F]
        upf = "F.upf"
        rc = 15.0
    """

    elements: dict[str, list[UpfConfig | PaoConfig]] = Field(default_factory=dict)
    """Per-element configurations. Each element maps to a list of configs."""

    @model_validator(mode="before")
    @classmethod
    def _extract_elements(cls, data: dict) -> dict:
        """Pull element sections out of the flat TOML namespace."""
        elements: dict[str, list] = {}
        for key in list(data.keys()):
            if key in _NON_ELEMENT_KEYS:
                continue
            val = data.pop(key)
            if isinstance(val, dict):
                elements[key] = [val]
            elif isinstance(val, list):
                elements[key] = val
        data["elements"] = elements
        return data

    @property
    def num_sets(self) -> int:
        """Number of comparison sets (max list length across elements)."""
        if not self.elements:
            return 0
        return max(len(v) for v in self.elements.values())


class BenchmarkConfig(_WorkflowConfigBase):
    """TOML configuration for benchmarking rival projectors.

    Uses the same ``[[Element]]`` syntax as
    :class:`ProjectabilityComparisonConfig`.  Elements with a single entry
    are fixed; elements with multiple entries define the comparison sets.

    Example TOML::

        structure = "LiF.cif"
        dis_proj_max = "optimize"

        [[Li]]
        upf = "Li.upf"
        rc = 15.0
        ri_factor = 0.3
        extension = "subshell"
        label = "PseudoDojo with confinement"

        [[Li]]
        openmx = true
        rc = 8.0
        select = ["s", "s", "p"]
        label = "OpenMX"

        [F]
        upf = "F.upf"
    """

    optimize_dis_thresholds: OptimizeDisThresholds = Field(
        default=OptimizeDisThresholds.NONE,
        description="Strategy for choosing dis_proj_min/max: "
        "'none' (use explicit values), 'bayesian', or 'otsu'.",
    )

    dis_froz_max_wrt_cbm: float | None = Field(
        default=None,
        description="Frozen window upper bound relative to the CBM (eV). "
        "When set, uses projectability + energy disentanglement.",
    )

    symmetrize: bool = Field(
        default=False,
        description="Rotate the projectors into a symmetry-adapted, "
        "bond-oriented basis (e.g. sp2 + pz on planar D_3h sites) "
        "before Wannierisation.",
    )

    bond_cutoff: float | None = Field(
        default=None,
        description="Distance cutoff (Angstrom) for nearest-neighbour "
        "detection used when symmetrize=true. If None, auto-detected.",
    )

    elements: dict[str, list[UpfConfig | PaoConfig]] = Field(default_factory=dict)
    """Per-element configurations. Each element maps to a list of configs."""

    @model_validator(mode="after")
    def _validate_optimize_thresholds(self) -> BenchmarkConfig:
        """Check that Otsu mode doesn't have explicit dis_proj_max."""
        if self.optimize_dis_thresholds == OptimizeDisThresholds.OTSU:
            if "dis_proj_max" in self.wannier90:
                raise ValueError(
                    "optimize_dis_thresholds = 'otsu' determines dis_proj_max "
                    "automatically; dis_proj_max must not be set in [wannier90]. "
                    "dis_proj_min may be set to override the Otsu lower threshold."
                )
        return self

    @model_validator(mode="before")
    @classmethod
    def _extract_elements(cls, data: dict) -> dict:
        """Pull element sections out of the flat TOML namespace."""
        elements: dict[str, list] = {}
        for key in list(data.keys()):
            if key in _NON_ELEMENT_KEYS:
                continue
            val = data.pop(key)
            if isinstance(val, dict):
                elements[key] = [val]
            elif isinstance(val, list):
                elements[key] = val
        data["elements"] = elements
        return data

    @property
    def num_sets(self) -> int:
        """Number of comparison sets (max list length across elements)."""
        if not self.elements:
            return 0
        return max(len(v) for v in self.elements.values())
