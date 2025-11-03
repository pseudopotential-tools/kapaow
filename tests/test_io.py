"""Testing the io module."""

from pathlib import Path

import numpy as np
import pytest

from pao_plusplus.io import read_wannier90_dat_file, write_wannier90_dat_file


@pytest.mark.parametrize("filename", ["Mo.dat", "Mo_TZ_rc10.0.dat"])
def test_read_write_wannier90_dat_file(filename: str, data_path: Path, tmp_path: Path) -> None:
    """Test reading and writing a Wannier90 .dat file."""
    # Read the original file
    x, r, l_values, orbitals = read_wannier90_dat_file(data_path / "dat_files" / filename)

    # Write to a temporary file
    write_wannier90_dat_file(tmp_path / filename, x, r, l_values, orbitals)

    # Read back the temporary file
    x_new, r_new, l_values_new, orbitals_new = read_wannier90_dat_file(tmp_path / filename)

    # Check that the data matches
    assert np.allclose(x, x_new), "Radial grid does not match after write/read."
    assert np.allclose(r, r_new), "Radial grid does not match after write/read."
    assert l_values == l_values_new, "Angular momentum values do not match after write/read."
    assert np.allclose(orbitals, orbitals_new), "Orbital values do not match after write/read."
