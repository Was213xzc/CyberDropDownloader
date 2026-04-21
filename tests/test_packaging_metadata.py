from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
RELEASE_SCRIPTS = (
    ROOT / "scripts/release/start_windows.bat",
    ROOT / "scripts/release/start_linux.sh",
    ROOT / "scripts/release/start_macOS.command",
)


def load_pyproject() -> dict:
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)


def test_project_exposes_both_cli_entry_points() -> None:
    scripts = load_pyproject()["project"]["scripts"]

    assert scripts["cyberdrop-dl"] == "cyberdrop_dl.main:main"
    assert scripts["cyberdrop-dl-patched"] == "cyberdrop_dl.main:main"


def test_apprise_is_available_as_optional_dependency() -> None:
    optional_dependencies = load_pyproject()["project"]["optional-dependencies"]

    assert "apprise" in optional_dependencies
    assert any(requirement.startswith("apprise ") for requirement in optional_dependencies["apprise"])


@pytest.mark.parametrize("script_path", RELEASE_SCRIPTS)
def test_release_scripts_follow_supported_python_range(script_path: Path) -> None:
    contents = script_path.read_text(encoding="utf8")

    assert "cyberdrop-dl-patched" in contents
    assert ">=3.11,<4" in contents
    assert ">=8.0,<9.0" not in contents
