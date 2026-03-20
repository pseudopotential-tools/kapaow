"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from pao_plusplus.cli import _describe_extension, get_extension, main
from pao_plusplus.extend import BasisExtensionViaAddition, BasisExtensionViaPolarization


# ---------------------------------------------------------------------------
# get_extension
# ---------------------------------------------------------------------------


def test_get_extension_empty() -> None:
    assert get_extension(()) is None


def test_get_extension_single_subshell() -> None:
    ext = get_extension(("subshell",))
    assert isinstance(ext, BasisExtensionViaAddition)
    assert ext.increment == 1


def test_get_extension_double_subshell() -> None:
    ext = get_extension(("subshell", "subshell"))
    assert isinstance(ext, BasisExtensionViaAddition)
    assert ext.increment == 2


def test_get_extension_single_polarization() -> None:
    ext = get_extension(("polarization",))
    assert isinstance(ext, BasisExtensionViaPolarization)
    assert ext.increment == 1


def test_get_extension_triple_polarization() -> None:
    ext = get_extension(("polarization", "polarization", "polarization"))
    assert isinstance(ext, BasisExtensionViaPolarization)
    assert ext.increment == 3


def test_get_extension_mixed_raises() -> None:
    with pytest.raises(click.UsageError, match="Cannot mix"):
        get_extension(("subshell", "polarization"))


# ---------------------------------------------------------------------------
# CLI convert command
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def pao_file(data_path: Path) -> Path:
    return data_path / "pao_files" / "test_element.pao"


def test_convert_pao_to_stdout(runner: CliRunner, pao_file: Path) -> None:
    """convert should print .dat content to stdout when no -o given."""
    result = runner.invoke(main, ["convert", str(pao_file)])
    assert result.exit_code == 0
    # Output should contain the number of grid points and orbitals
    lines = result.output.strip().splitlines()
    header = lines[0].split()
    assert header[0] == "5"  # num_grid
    assert header[1] == "2"  # 2 orbitals (default: 1 per l channel)


def test_convert_pao_with_select(runner: CliRunner, pao_file: Path) -> None:
    """convert with --select should select specific orbitals."""
    result = runner.invoke(main, ["convert", str(pao_file), "--select", "ssp"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    # l_values line should be "0 0 1"
    assert lines[1].strip() == "0 0 1"


def test_convert_pao_to_file(runner: CliRunner, pao_file: Path, tmp_path: Path) -> None:
    """convert with -o should write to file."""
    output = tmp_path / "output.dat"
    result = runner.invoke(main, ["convert", str(pao_file), "-o", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    assert "Written to" in result.output


def test_convert_select_on_non_pao_raises(runner: CliRunner, data_path: Path) -> None:
    """--select on a non-.pao file should raise UsageError."""
    dat_file = data_path / "dat_files" / "Mo.dat"
    result = runner.invoke(main, ["convert", str(dat_file), "--select", "sp"])
    assert result.exit_code != 0
    assert "only valid for OpenMX" in result.output


def test_convert_pao_invalid_select(runner: CliRunner, pao_file: Path) -> None:
    """Invalid --select characters should produce an error."""
    result = runner.invoke(main, ["convert", str(pao_file), "--select", "xyz"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI confine command
# ---------------------------------------------------------------------------


def test_confine_requires_add(runner: CliRunner, data_path: Path) -> None:
    """confine without --add should error."""
    # Use any existing file as the UPF argument (it won't actually run the solver)
    dat_file = data_path / "dat_files" / "Mo.dat"
    result = runner.invoke(main, ["confine", str(dat_file)])
    assert result.exit_code != 0
    assert "requires --add" in result.output


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_main_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output
    assert "confine" in result.output
    assert "optimize" in result.output
    assert "animate" in result.output
    assert "plot" in result.output


def test_convert_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--select" in result.output


def test_plot_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["plot", "--help"])
    assert result.exit_code == 0
    assert "paos" in result.output
    assert "periodic-table" in result.output


def test_optimize_help(runner: CliRunner) -> None:
    result = runner.invoke(main, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "projectability" in result.output
    assert "spread" in result.output
