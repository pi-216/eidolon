import subprocess
import sys


def test_cli_subcommands_are_wired():
    """Every subcommand is a real, documented command, not a stub."""
    for cmd in ["extract-faces", "review"]:
        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", cmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        assert "not yet implemented" not in result.stdout


def test_cli_subcommands_require_arguments():
    """Invoking a subcommand bare fails with argparse usage, not a traceback."""
    for cmd in ["extract-faces", "review"]:
        result = subprocess.run(
            [sys.executable, "-m", "tools.hegre_dataset", cmd],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "usage:" in result.stderr
