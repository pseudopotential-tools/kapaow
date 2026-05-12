"""Extended tests for the io module."""

from pathlib import Path

import numpy as np

from kapaow.io import (
    format_wannier90_dat,
    read_wannier90_dat_file,
    write_wannier90_dat_file,
)


def test_format_wannier90_dat_header() -> None:
    """format_wannier90_dat should produce the correct header."""
    x = [0.0, 1.0, 2.0]
    r = [1.0, 2.718, 7.389]
    l_values = [0, 1]
    orbitals = np.array([[0.5, 0.3, 0.1], [0.4, 0.2, 0.05]])

    result = format_wannier90_dat(x, r, l_values, orbitals)
    lines = result.strip().splitlines()

    # First line: num_grid num_orbitals
    assert lines[0] == "3 2"
    # Second line: l values
    assert lines[1] == "0 1"
    # Data lines
    assert len(lines) == 5  # header + l_values + 3 data lines


def test_format_wannier90_dat_roundtrip(tmp_path: Path) -> None:
    """Data written with format_wannier90_dat should be readable."""
    x = [0.0, 0.5, 1.0, 1.5]
    r = [1.0, 1.649, 2.718, 4.482]
    l_values = [0, 0, 1]
    orbitals = np.random.default_rng(42).standard_normal((3, 4))

    content = format_wannier90_dat(x, r, l_values, orbitals)
    dat_file = tmp_path / "test.dat"
    dat_file.write_text(content)

    x2, r2, l2, orb2 = read_wannier90_dat_file(dat_file)
    np.testing.assert_allclose(x, x2, atol=1e-6)
    np.testing.assert_allclose(r, r2, atol=1e-6)
    assert l_values == l2
    np.testing.assert_allclose(orbitals, orb2, atol=1e-6)


def test_write_read_single_orbital(tmp_path: Path) -> None:
    """Write and read back a file with a single orbital."""
    x = [0.0, 1.0]
    r = [1.0, 2.718]
    l_values = [0]
    orbitals = np.array([[0.5, 0.3]])

    write_wannier90_dat_file(tmp_path / "single.dat", x, r, l_values, orbitals)
    _x2, _r2, l2, orb2 = read_wannier90_dat_file(tmp_path / "single.dat")
    assert l2 == [0]
    assert orb2.shape == (1, 2)


def test_read_wannier90_dat_file_values(data_path: Path) -> None:
    """Verify specific properties of the bundled Mo.dat test file."""
    x, r, l_values, orbitals = read_wannier90_dat_file(data_path / "dat_files" / "Mo.dat")
    assert len(x) == len(r)
    assert len(l_values) == orbitals.shape[0]
    # r should be positive (it's exp(x))
    assert all(ri > 0 for ri in r)
