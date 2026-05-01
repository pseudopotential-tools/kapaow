"""Tests for the CLI module."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from kapaow.cli import get_extension, main
from kapaow.extend import BasisExtensionViaAddition, BasisExtensionViaPolarization

# ---------------------------------------------------------------------------
# get_extension
# ---------------------------------------------------------------------------


def test_get_extension_empty() -> None:
    """Empty add tuple should return None."""
    assert get_extension(()) is None


def test_get_extension_single_subshell() -> None:
    """Single subshell should produce addition with increment 1."""
    ext = get_extension(("subshell",))
    assert isinstance(ext, BasisExtensionViaAddition)
    assert ext.increment == 1


def test_get_extension_double_subshell() -> None:
    """Two subshells should produce addition with increment 2."""
    ext = get_extension(("subshell", "subshell"))
    assert isinstance(ext, BasisExtensionViaAddition)
    assert ext.increment == 2


def test_get_extension_single_polarization() -> None:
    """Single polarization should produce polarization with increment 1."""
    ext = get_extension(("polarization",))
    assert isinstance(ext, BasisExtensionViaPolarization)
    assert ext.increment == 1


def test_get_extension_triple_polarization() -> None:
    """Three polarizations should produce polarization with increment 3."""
    ext = get_extension(("polarization", "polarization", "polarization"))
    assert isinstance(ext, BasisExtensionViaPolarization)
    assert ext.increment == 3


def test_get_extension_mixed_raises() -> None:
    """Mixing subshell and polarization should raise UsageError."""
    with pytest.raises(click.UsageError, match="Cannot mix"):
        get_extension(("subshell", "polarization"))


# ---------------------------------------------------------------------------
# CLI convert command
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click test runner."""
    return CliRunner()


@pytest.fixture
def pao_file(data_path: Path) -> Path:
    """Return path to the test .pao file."""
    return data_path / "pao_files" / "test_element.pao"


def test_convert_pao_to_stdout(runner: CliRunner, pao_file: Path) -> None:
    """Convert should print .dat content to stdout when no -o given."""
    result = runner.invoke(main, ["convert", str(pao_file)])
    assert result.exit_code == 0
    # Output should contain the number of grid points and orbitals
    lines = result.output.strip().splitlines()
    header = lines[0].split()
    assert header[0] == "5"  # num_grid
    assert header[1] == "2"  # 2 orbitals (default: 1 per l channel)


def test_convert_pao_with_select(runner: CliRunner, pao_file: Path) -> None:
    """Convert with --select should select specific orbitals."""
    result = runner.invoke(main, ["convert", str(pao_file), "--select", "ssp"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    # l_values line should be "0 0 1"
    assert lines[1].strip() == "0 0 1"


def test_convert_pao_to_file(runner: CliRunner, pao_file: Path, tmp_path: Path) -> None:
    """Convert with -o should write to file."""
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
    """Confine without --add should error."""
    # Use any existing file as the UPF argument (it won't actually run the solver)
    dat_file = data_path / "dat_files" / "Mo.dat"
    result = runner.invoke(main, ["confine", str(dat_file)])
    assert result.exit_code != 0
    assert "requires --add" in result.output


# ---------------------------------------------------------------------------
# CLI help
# ---------------------------------------------------------------------------


def test_main_help(runner: CliRunner) -> None:
    """Main help should list all top-level commands."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "convert" in result.output
    assert "confine" in result.output
    assert "optimize" in result.output
    assert "animate" in result.output
    assert "plot" in result.output


def test_convert_help(runner: CliRunner) -> None:
    """Convert help should list --select option."""
    result = runner.invoke(main, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--select" in result.output


def test_plot_help(runner: CliRunner) -> None:
    """Plot help should list subcommands."""
    result = runner.invoke(main, ["plot", "--help"])
    assert result.exit_code == 0
    assert "paos" in result.output
    assert "periodic-table" in result.output


def test_optimize_help(runner: CliRunner) -> None:
    """Optimize help should list subcommands."""
    result = runner.invoke(main, ["optimize", "--help"])
    assert result.exit_code == 0
    assert "projectability" in result.output
    assert "spread" in result.output
