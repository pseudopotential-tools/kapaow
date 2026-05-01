"""Tests for the openmx module."""

from pathlib import Path

import numpy as np
import pytest

from kapaow.openmx import (
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
    """Return path to the test .pao file."""
    return data_path / "pao_files" / "test_element.pao"


@pytest.fixture
def pao_text(pao_file: Path) -> str:
    """Return the text content of the test .pao file."""
    return pao_file.read_text()


@pytest.fixture
def pao(pao_file: Path) -> OpenMXPAO:
    """Return a parsed OpenMXPAO from the test file."""
    return read_openmx_pao(pao_file)


# ---------------------------------------------------------------------------
# _extract_param
# ---------------------------------------------------------------------------


def test_extract_param_lmax(pao_text: str) -> None:
    """PAO.Lmax should be extracted as 2."""
    assert _extract_param(pao_text, "PAO.Lmax") == 2


def test_extract_param_mul(pao_text: str) -> None:
    """PAO.Mul should be extracted as 3."""
    assert _extract_param(pao_text, "PAO.Mul") == 3


def test_extract_param_missing(pao_text: str) -> None:
    """Missing key should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        _extract_param(pao_text, "PAO.Nonexistent")


def test_extract_param_with_comment() -> None:
    """Inline comments after the value should be stripped."""
    text = "grid.num.output  500  # default=2000\n"
    assert _extract_param(text, "grid.num.output") == 500


# ---------------------------------------------------------------------------
# _extract_tag
# ---------------------------------------------------------------------------


def test_extract_tag_l0(pao_text: str) -> None:
    """L=0 block should contain 5 data lines."""
    block = _extract_tag(pao_text, "pseudo.atomic.orbitals.L=0")
    lines = block.strip().splitlines()
    assert len(lines) == 5
    assert lines[0].strip().startswith("-5.0")


def test_extract_tag_l1(pao_text: str) -> None:
    """L=1 block should contain 5 data lines."""
    block = _extract_tag(pao_text, "pseudo.atomic.orbitals.L=1")
    lines = block.strip().splitlines()
    assert len(lines) == 5


def test_extract_tag_missing(pao_text: str) -> None:
    """Missing tag should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        _extract_tag(pao_text, "pseudo.atomic.orbitals.L=99")


# ---------------------------------------------------------------------------
# read_openmx_pao
# ---------------------------------------------------------------------------


def test_read_openmx_pao_lmax(pao: OpenMXPAO) -> None:
    """Parsed lmax should be 2."""
    assert pao.lmax == 2


def test_read_openmx_pao_num_mul(pao: OpenMXPAO) -> None:
    """Parsed num_mul should be 3."""
    assert pao.num_mul == 3


def test_read_openmx_pao_grid(pao: OpenMXPAO) -> None:
    """Grid x and r should have 5 points with r = exp(x)."""
    assert len(pao.x) == 5
    assert len(pao.r) == 5
    np.testing.assert_allclose(pao.x[0], -5.0)
    np.testing.assert_allclose(pao.r, np.exp(pao.x), rtol=1e-4)


def test_read_openmx_pao_orbitals_shape(pao: OpenMXPAO) -> None:
    """Each l channel should have shape (5, 3)."""
    assert 0 in pao.orbitals
    assert 1 in pao.orbitals
    assert pao.orbitals[0].shape == (5, 3)
    assert pao.orbitals[1].shape == (5, 3)


def test_read_openmx_pao_orbital_values(pao: OpenMXPAO) -> None:
    """Spot-check specific orbital values from the test file."""
    np.testing.assert_allclose(pao.orbitals[0][0, 0], 0.5)
    np.testing.assert_allclose(pao.orbitals[0][0, 1], 0.1)


def test_read_openmx_pao_real_file() -> None:
    """Read a real OpenMX .pao file from bundled data."""
    real_pao = (
        Path(__file__).parent.parent
        / "src"
        / "kapaow"
        / "data"
        / "openmx"
        / "PAO"
        / "Si7.0.pao"
    )
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
    """Single 's' should return [1]."""
    assert parse_select(["s"]) == [1]


def test_parse_select_sspd() -> None:
    """['s', 's', 'p', 'd'] should return [2, 1, 1]."""
    assert parse_select(["s", "s", "p", "d"]) == [2, 1, 1]


def test_parse_select_case_insensitive() -> None:
    """Uppercase letters should work the same as lowercase."""
    assert parse_select(["S", "P", "D"]) == [1, 1, 1]


def test_parse_select_multiple_p() -> None:
    """['p', 'p', 'p'] should return [0, 3] (no s, three p)."""
    assert parse_select(["p", "p", "p"]) == [0, 3]


def test_parse_select_gap_in_channels() -> None:
    """['s', 'd'] should return [1, 0, 1] with a zero-count p channel."""
    assert parse_select(["s", "d"]) == [1, 0, 1]


def test_parse_select_all_letters() -> None:
    """All angular momentum letters should parse correctly."""
    result = parse_select(["s", "p", "d", "f", "g", "h"])
    assert result == [1, 1, 1, 1, 1, 1]


def test_parse_select_empty() -> None:
    """Empty list should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        parse_select([])


def test_parse_select_invalid_char() -> None:
    """Invalid character should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown angular momentum"):
        parse_select(["x"])


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
    _x, _r, l_values, orbitals = convert_to_wannier90(pao, selected)
    assert l_values == [0, 0, 1]
    assert orbitals.shape == (3, 5)


def test_convert_single_channel(pao: OpenMXPAO) -> None:
    """Select only s-orbitals."""
    selected = [3]
    _x, _r, l_values, orbitals = convert_to_wannier90(pao, selected)
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
