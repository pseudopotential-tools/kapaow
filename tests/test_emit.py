"""Tests for the emit module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kapaow.basis import AngularMomentum, PseudoatomicBasis
from kapaow.emit import _build_rank_k_basis, emit_ranks
from kapaow.solve import OrbitalEnergy

# ---------------------------------------------------------------------------
# _build_rank_k_basis
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_s1p1() -> PseudoatomicBasis:
    """Baseline with 1 s and 1 p orbital."""
    return PseudoatomicBasis(number_of_orbitals={AngularMomentum.S: 1, AngularMomentum.P: 1})


@pytest.fixture
def extras_spd() -> list[OrbitalEnergy]:
    """Three extras pre-sorted by raw eigenvalue ascending (most bound first)."""
    return [
        OrbitalEnergy(l=AngularMomentum.P, n_radial=1, energy=-0.9),  # most bound
        OrbitalEnergy(l=AngularMomentum.D, n_radial=0, energy=-0.5),
        OrbitalEnergy(l=AngularMomentum.S, n_radial=1, energy=-0.1),  # least bound
    ]


def test_build_rank0_equals_baseline(
    baseline_s1p1: PseudoatomicBasis, extras_spd: list[OrbitalEnergy]
) -> None:
    """Rank 0 should reproduce the baseline counts exactly."""
    basis = _build_rank_k_basis(baseline_s1p1, extras_spd, k=0)
    assert basis.number_of_orbitals[AngularMomentum.S] == 1
    assert basis.number_of_orbitals[AngularMomentum.P] == 1
    assert basis.number_of_orbitals[AngularMomentum.D] == 0


def test_build_rank1_adds_first_extra(
    baseline_s1p1: PseudoatomicBasis, extras_spd: list[OrbitalEnergy]
) -> None:
    """Rank 1 should add the first (most-bound) extra (p, n_radial=1 → p count becomes 2)."""
    basis = _build_rank_k_basis(baseline_s1p1, extras_spd, k=1)
    assert basis.number_of_orbitals[AngularMomentum.S] == 1
    assert basis.number_of_orbitals[AngularMomentum.P] == 2
    assert basis.number_of_orbitals[AngularMomentum.D] == 0


def test_build_rank2_adds_d_channel(
    baseline_s1p1: PseudoatomicBasis, extras_spd: list[OrbitalEnergy]
) -> None:
    """Rank 2 should also include the d extra (d count becomes 1)."""
    basis = _build_rank_k_basis(baseline_s1p1, extras_spd, k=2)
    assert basis.number_of_orbitals[AngularMomentum.S] == 1
    assert basis.number_of_orbitals[AngularMomentum.P] == 2
    assert basis.number_of_orbitals[AngularMomentum.D] == 1


def test_build_rank_k_numerical_counts() -> None:
    """Regression: verify per-channel counts for a concrete rank-K call."""
    # Baseline: s=2, d=1; Extra: d at n_radial=1 (needs d count >= 2)
    baseline = PseudoatomicBasis(number_of_orbitals={AngularMomentum.S: 2, AngularMomentum.D: 1})
    extras = [OrbitalEnergy(l=AngularMomentum.D, n_radial=1, energy=-0.3)]
    basis = _build_rank_k_basis(baseline, extras, k=1)
    np.testing.assert_array_equal(
        [basis.number_of_orbitals[AngularMomentum.S], basis.number_of_orbitals[AngularMomentum.D]],
        [2, 2],
    )


def test_build_rank_k_already_covered() -> None:
    """An extra at n_radial already covered by baseline should not change the count."""
    # Baseline already has s=2; extra s at n_radial=0 is already covered
    baseline = PseudoatomicBasis(number_of_orbitals={AngularMomentum.S: 2})
    extras = [OrbitalEnergy(l=AngularMomentum.S, n_radial=0, energy=-1.5)]
    basis = _build_rank_k_basis(baseline, extras, k=1)
    assert basis.number_of_orbitals[AngularMomentum.S] == 2  # unchanged


# ---------------------------------------------------------------------------
# emit_ranks — input validation
# ---------------------------------------------------------------------------


def test_emit_ranks_raises_without_rc(tmp_path: Path) -> None:
    """emit_ranks should raise when neither rc nor rc_search_json is given."""
    with pytest.raises(ValueError, match="exactly one"):
        emit_ranks(Path("dummy.upf"), output_dir=tmp_path)


def test_emit_ranks_raises_with_both_rc_and_json(tmp_path: Path) -> None:
    """emit_ranks should raise when both rc and rc_search_json are given."""
    fake_json = tmp_path / "rc.json"
    fake_json.write_text(json.dumps({"rc": 7.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        emit_ranks(Path("dummy.upf"), rc=7.0, rc_search_json=fake_json, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# emit_ranks — happy path (mocked femdvr)
# ---------------------------------------------------------------------------


def _make_fake_result(eigenvalues: dict) -> MagicMock:
    """Build a minimal mock PseudoAtomicResult."""
    result = MagicMock()
    result.eigenvalues = eigenvalues
    return result


@pytest.fixture
def minimal_upf(tmp_path: Path) -> Path:
    """Write a trivial UPF file stub accepted by upf_tools."""
    # Use a real minimal UPF from the test data when available; otherwise
    # we patch UPFDict.from_upf so any path works.
    return tmp_path / "H.upf"


def test_emit_ranks_happy_path(tmp_path: Path, minimal_upf: Path) -> None:
    """emit_ranks should write rank0 and rank1 dat files and a JSON index."""
    # Fake eigenvalues: s channel has 2 states, p has 1
    fake_eigenvalues = {
        "nscf": {
            "0": [-2.0, -0.1],  # s: n_radial=0 (baseline), n_radial=1 (extra)
            "1": [-1.0],  # p: n_radial=0 (baseline)
        }
    }
    fake_result = _make_fake_result(fake_eigenvalues)

    # Fake UPF: baseline = 1s + 1p
    fake_upf_data = {
        "header": {"element": "H"},
        "pswfc": {"chi": [{"n": 1, "l": 0}, {"n": 2, "l": 1}]},
    }

    # Fake dat file content: 2 grid points, 3 orbitals (s, s, p)
    qe_dat_content = "2 3\n0 0 1\n1.0e-01 1.0e-01 1.0 0.5 0.2\n2.0e-01 2.0e-01 0.9 0.4 0.1\n"

    def fake_solve(*args, working_dir: Path, **kwargs):
        """Stub the femdvr solve: drop a canned dat file and return ``fake_result``."""
        (working_dir / "H_qe.dat").write_text(qe_dat_content, encoding="utf-8")
        return fake_result

    with (
        patch("kapaow.emit.UPFDict.from_upf", return_value=fake_upf_data),
        patch("kapaow.emit.solve_pseudoatomic_problem", side_effect=fake_solve),
        patch("kapaow.emit._write_upf_for_basis"),
    ):
        records = emit_ranks(
            minimal_upf,
            rc=7.5,
            ri_factor=0.95,
            max_rank=1,
            output_dir=tmp_path,
        )

    assert len(records) == 2  # rank 0 and rank 1

    # rank 0
    assert records[0].rank == 0
    assert records[0].added_orbital is None
    assert records[0].dat_path.exists()

    # rank 1: only one extra exists (s, n_radial=1, energy=-0.1)
    assert records[1].rank == 1
    added = records[1].added_orbital
    assert added is not None
    assert added.l == AngularMomentum.S
    assert added.n_radial == 1
    np.testing.assert_allclose(added.energy, -0.1, rtol=1e-10)

    # JSON index
    json_path = tmp_path / "H_ranks.json"
    assert json_path.exists()
    index = json.loads(json_path.read_text())
    assert index["rc"] == pytest.approx(7.5)
    assert index["ri_factor"] == pytest.approx(0.95)
    assert len(index["ranks"]) == 2
    assert index["ranks"][0]["added"] is None
    assert index["ranks"][1]["added"]["l"] == 0
    assert index["ranks"][1]["added"]["n_radial"] == 1


def test_emit_ranks_reads_rc_from_json(tmp_path: Path, minimal_upf: Path) -> None:
    """emit_ranks should extract rc from a rc_search_json file."""
    rc_json = tmp_path / "rc.json"
    rc_json.write_text(json.dumps({"rc": 8.25, "ri_factor": 0.9}), encoding="utf-8")

    fake_eigenvalues = {"scf": {"0": [-2.0]}}  # only baseline s
    fake_result = _make_fake_result(fake_eigenvalues)
    fake_upf_data = {
        "header": {"element": "H"},
        "pswfc": {"chi": [{"n": 1, "l": 0}]},
    }
    qe_dat_content = "2 1\n0\n1.0e-01 1.0e-01 1.0\n2.0e-01 2.0e-01 0.9\n"

    captured_rc: list[float] = []

    def fake_solve(upf_path, *, rc, working_dir: Path, **kwargs):
        """Stub the femdvr solve and capture the ``rc`` value passed in."""
        captured_rc.append(rc)
        (working_dir / "H_qe.dat").write_text(qe_dat_content, encoding="utf-8")
        return fake_result

    with (
        patch("kapaow.emit.UPFDict.from_upf", return_value=fake_upf_data),
        patch("kapaow.emit.solve_pseudoatomic_problem", side_effect=fake_solve),
        patch("kapaow.emit._write_upf_for_basis"),
    ):
        emit_ranks(minimal_upf, rc_search_json=rc_json, output_dir=tmp_path)

    assert captured_rc[0] == pytest.approx(8.25)


# ---------------------------------------------------------------------------
# CLI emit-ranks command
# ---------------------------------------------------------------------------


def test_cli_emit_ranks_usage_error(tmp_path: Path) -> None:
    """CLI should error when neither --rc nor --rc-search-json is given."""
    from click.testing import CliRunner

    from kapaow.cli import main

    runner = CliRunner()
    # Create a dummy UPF so the path-exists check passes
    upf = tmp_path / "dummy.upf"
    upf.write_text("", encoding="utf-8")

    result = runner.invoke(main, ["emit-ranks", str(upf)])
    assert result.exit_code != 0
    assert "exactly one" in (result.output + str(result.exception)).lower()
