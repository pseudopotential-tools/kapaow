"""Tests for the openmx module."""

from pathlib import Path

import numpy as np
import pytest

from pao_plusplus.openmx import (
    OpenMXPAO,
    _extract_param,
    _extract_tag,
    convert_to_wannier90,
    parse_select,
    read_openmx_pao,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pao_file(data_path: Path) -> Path:
    return data_path / "pao_files" / "test_element.pao"


@pytest.fixture
def pao_text(pao_file: Path) -> str:
    return pao_file.read_text()


@pytest.fixture
def pao(pao_file: Path) -> OpenMXPAO:
    return read_openmx_pao(pao_file)


# ---------------------------------------------------------------------------
# _extract_param
# ---------------------------------------------------------------------------


def test_extract_param_lmax(pao_text: str) -> None:
    assert _extract_param(pao_text, "PAO.Lmax") == 2


def test_extract_param_mul(pao_text: str) -> None:
    assert _extract_param(pao_text, "PAO.Mul") == 3


def test_extract_param_missing(pao_text: str) -> None:
    with pytest.raises(ValueError, match="not found"):
        _extract_param(pao_text, "PAO.Nonexistent")


def test_extract_param_with_comment() -> None:
    text = "grid.num.output  500  # default=2000\n"
    assert _extract_param(text, "grid.num.output") == 500


# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------


def test_extract_tag_l0(pao_text: str) -> None:
    block = _extract_tag(pao_text, "pseudo.atomic.orbitals.L=0")
    lines = block.strip().splitlines()
    assert len(lines) == 5
    # First line should start with -5.0
    assert lines[0].strip().startswith("-5.0")


def test_extract_tag_l1(pao_text: str) -> None:
    block = _extract_tag(pao_text, "pseudo.atomic.orbitals.L=1")
    lines = block.strip().splitlines()
    assert len(lines) == 5


def test_extract_tag_missing(pao_text: str) -> None:
    with pytest.raises(ValueError, match="not found"):
        _extract_tag(pao_text, "pseudo.atomic.orbitals.L=99")


# ---------------------------------------------------------------------------
# read_openmx_pao
# ---------------------------------------------------------------------------


def test_read_openmx_pao_lmax(pao: OpenMXPAO) -> None:
    assert pao.lmax == 2


def test_read_openmx_pao_num_mul(pao: OpenMXPAO) -> None:
    assert pao.num_mul == 3


def test_read_openmx_pao_grid(pao: OpenMXPAO) -> None:
    assert len(pao.x) == 5
    assert len(pao.r) == 5
    np.testing.assert_allclose(pao.x[0], -5.0)
    np.testing.assert_allclose(pao.r, np.exp(pao.x), rtol=1e-4)


def test_read_openmx_pao_orbitals_shape(pao: OpenMXPAO) -> None:
    assert 0 in pao.orbitals
    assert 1 in pao.orbitals
    # 5 grid points, 3 multiplicities per channel
    assert pao.orbitals[0].shape == (5, 3)
    assert pao.orbitals[1].shape == (5, 3)


def test_read_openmx_pao_orbital_values(pao: OpenMXPAO) -> None:
    # First grid point, first orbital of L=0 should be 0.5
    np.testing.assert_allclose(pao.orbitals[0][0, 0], 0.5)
    # First grid point, second orbital of L=0 should be 0.1
    np.testing.assert_allclose(pao.orbitals[0][0, 1], 0.1)


def test_read_openmx_pao_real_file() -> None:
    """Read a real OpenMX .pao file from bundled data."""
    real_pao = Path(__file__).parent.parent / "src" / "pao_plusplus" / "data" / "openmx" / "PAO" / "Si7.0.pao"
    if not real_pao.exists():
        pytest.skip("Real PAO file not available")
    pao = read_openmx_pao(real_pao)
    assert pao.lmax == 3
    assert pao.num_mul == 15
    assert len(pao.x) > 0
    assert len(pao.orbitals) == 3


# ---------------------------------------------------------------------------
# parse_select
# ---------------------------------------------------------------------------


def test_parse_select_single() -> None:
    assert parse_select("s") == [1]


def test_parse_select_sspd() -> None:
    assert parse_select("sspd") == [2, 1, 1]


def test_parse_select_case_insensitive() -> None:
    assert parse_select("SPD") == [1, 1, 1]


def test_parse_select_multiple_p() -> None:
    assert parse_select("ppp") == [0, 3]


def test_parse_select_gap_in_channels() -> None:
    # "sd" -> l=0 has 1, l=1 has 0, l=2 has 1
    assert parse_select("sd") == [1, 0, 1]


def test_parse_select_all_letters() -> None:
    result = parse_select("spdfgh")
    assert result == [1, 1, 1, 1, 1, 1]


def test_parse_select_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_select("")


def test_parse_select_invalid_char() -> None:
    with pytest.raises(ValueError, match="Unknown angular momentum"):
        parse_select("x")


# ---------------------------------------------------------------------------
# convert_to_wannier90
# ---------------------------------------------------------------------------


def test_convert_default_selection(pao: OpenMXPAO) -> None:
    """Default: one orbital per l channel."""
    x, r, l_values, orbitals = convert_to_wannier90(pao)
    assert len(x) == 5
    assert len(r) == 5
    assert l_values == [0, 1]
    assert orbitals.shape == (2, 5)


def test_convert_custom_selection(pao: OpenMXPAO) -> None:
    """Select 2 s-orbitals and 1 p-orbital."""
    selected = [2, 1]
    x, r, l_values, orbitals = convert_to_wannier90(pao, selected)
    assert l_values == [0, 0, 1]
    assert orbitals.shape == (3, 5)


def test_convert_single_channel(pao: OpenMXPAO) -> None:
    """Select only s-orbitals."""
    selected = [3]
    x, r, l_values, orbitals = convert_to_wannier90(pao, selected)
    assert l_values == [0, 0, 0]
    assert orbitals.shape == (3, 5)


def test_convert_too_many_requested(pao: OpenMXPAO) -> None:
    """Requesting more orbitals than available should raise."""
    selected = [4, 1]  # only 3 s-orbitals available
    with pytest.raises(ValueError, match="only"):
        convert_to_wannier90(pao, selected)


def test_convert_l_channel_out_of_range(pao: OpenMXPAO) -> None:
    """Requesting a channel beyond lmax should raise."""
    selected = [1, 1, 1]  # l=2 doesn't exist (lmax=2 means channels 0,1)
    with pytest.raises(ValueError, match="only has"):
        convert_to_wannier90(pao, selected)
