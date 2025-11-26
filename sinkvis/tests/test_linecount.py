import os
from pathlib import Path

# Get absolute path to sinkvis directory (relative to this test file)
SINKVIS_DIR = Path(__file__).parent.parent.resolve()


def test_linecount():
    """
    Counts lines in sinkvis/*.py files.
    Test passes if total lines < 1000.
    Note: Formatting is checked separately to avoid modifying files during tests.
    """
    num_lines = 0
    for file in os.listdir(SINKVIS_DIR):
        if file.endswith(".py"):
            with open(SINKVIS_DIR / file, "r") as f:
                num_lines += len(f.readlines())
    print(f"Number of lines: {num_lines}")

    assert num_lines < 1000


def test_formatting_check():
    """Check that code is formatted (without modifying files)."""
    import subprocess

    # Check black formatting (--check mode, no modification)
    result = subprocess.run(
        ["black", "--check", str(SINKVIS_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0, "Code is not formatted with black"

    # Check isort (--check mode, no modification)
    result = subprocess.run(
        ["isort", "--check", str(SINKVIS_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0, "Imports are not sorted with isort"


def test_flake8():
    """
    Check Flake8 + Cyclomatic Complexity.
    --max-complexity ensures functions aren't too dense/nested.
    --select=E,F,W,C901 enables standard errors + complexity checks.
    --max-line-length=86 sets E501 line length limit to 86 characters.
    """
    import glob
    import subprocess

    files = glob.glob(str(SINKVIS_DIR / "*.py"))

    result = subprocess.run(
        [
            "flake8",
            "--max-complexity=10",
            "--max-line-length=86",
            "--select=E,F,W,C901",
        ]
        + files,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)

    assert result.returncode == 0
