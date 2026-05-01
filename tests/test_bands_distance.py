"""Tests for the bands_distance module."""

import numpy as np
import pytest

from kapaow.bands_distance import BandDistanceResult, bands_distance


def test_identical_bands() -> None:
    """Identical bands should have zero distance."""
    energies = np.random.default_rng(42).standard_normal((10, 5))
    result = bands_distance(energies, energies)
    assert result.eta == pytest.approx(0.0)
    assert result.max_dist == pytest.approx(0.0)
    np.testing.assert_allclose(result.per_band_eta, 0.0, atol=1e-15)


def test_constant_offset() -> None:
    """A constant offset should give eta = abs(offset)."""
    rng = np.random.default_rng(42)
    dft = rng.standard_normal((20, 4))
    offset = 0.5
    wannier = dft + offset
    result = bands_distance(dft, wannier)
    assert result.eta == pytest.approx(offset, abs=1e-10)
    assert result.max_dist == pytest.approx(offset, abs=1e-10)
    np.testing.assert_allclose(result.per_band_eta, offset, atol=1e-10)


def test_kpoint_mismatch() -> None:
    """Different number of k-points should raise ValueError."""
    dft = np.zeros((10, 3))
    wannier = np.zeros((8, 3))
    with pytest.raises(ValueError, match="K-point count mismatch"):
        bands_distance(dft, wannier)


def test_band_trimming() -> None:
    """When band counts differ, only the minimum is compared."""
    rng = np.random.default_rng(42)
    dft = rng.standard_normal((10, 6))
    wannier = dft[:, :4].copy()  # only 4 bands
    result = bands_distance(dft, wannier)
    assert result.eta == pytest.approx(0.0)
    assert result.per_band_eta.shape == (4,)


def test_max_dist_location() -> None:
    """max_dist_k and max_dist_band should point to the largest difference."""
    dft = np.zeros((5, 3))
    wannier = np.zeros((5, 3))
    wannier[2, 1] = 3.0  # largest difference at k=2, band=1
    result = bands_distance(dft, wannier)
    assert result.max_dist == pytest.approx(3.0)
    assert result.max_dist_k == 2
    assert result.max_dist_band == 1


def test_per_band_eta_shape() -> None:
    """per_band_eta should have one entry per band."""
    dft = np.random.default_rng(42).standard_normal((15, 7))
    wannier = dft + 0.1
    result = bands_distance(dft, wannier)
    assert result.per_band_eta.shape == (7,)


def test_result_is_dataclass() -> None:
    """Result should be a BandDistanceResult."""
    dft = np.zeros((3, 2))
    result = bands_distance(dft, dft)
    assert isinstance(result, BandDistanceResult)


def test_single_kpoint_single_band() -> None:
    """Edge case: 1 k-point, 1 band."""
    dft = np.array([[1.0]])
    wannier = np.array([[1.5]])
    result = bands_distance(dft, wannier)
    assert result.eta == pytest.approx(0.5)
    assert result.max_dist == pytest.approx(0.5)
    assert result.max_dist_k == 0
    assert result.max_dist_band == 0
