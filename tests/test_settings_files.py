"""Only the known settings files may exist in the project folder (tracked or not)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "settings_files.py"


_spec = importlib.util.spec_from_file_location("settings_files", SCRIPT)
settings_files = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(settings_files)

EXPECTED_NAMES = {
    ".ignore",
    ".pytest.ini",
    ".pytest.toml",
    ".ruff.toml",
    "conftest.py",
    "pyproject.toml",
    "pytest.ini",
    "pytest.toml",
    "ruff.toml",
    "setup.cfg",
    "tox.ini",
    "uv.toml",
}


def make(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def run_script(root: Path) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done.returncode, done.stdout


def test_clean_layout_passes(tmp_path):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", "tests/test_a.py", "src/pkg/__init__.py")
    status, out = run_script(tmp_path)
    assert status == 0
    assert len(out.splitlines()) == 1


@pytest.mark.parametrize(
    "stray",
    [
        "conftest.py",
        "tests/ruff.toml",
        ".ruff.toml",
        "tests/sub/conftest.py",
        "pytest.ini",
        ".ignore",
        ".pytest.toml",
        "tests/data/conftest.py",
        "src/reference/ruff.toml",
        "tests/.venv/conftest.py",
    ],
)
def test_stray_settings_file_fails_and_is_named(tmp_path, stray):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", stray)
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == [stray]


def test_every_stray_is_named(tmp_path):
    make(tmp_path, "pyproject.toml", "conftest.py", "tox.ini", "sub/setup.cfg")
    status, out = run_script(tmp_path)
    assert status == 1
    assert sorted(out.splitlines()) == ["conftest.py", "sub/setup.cfg", "tox.ini"]


def test_allowed_names_at_other_paths_fail(tmp_path):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", "sub/pyproject.toml")
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == ["sub/pyproject.toml"]


@pytest.mark.parametrize("folder", [".venv", "reference", "data", ".git", "runs.noindex"])
def test_skipped_folders_do_not_fail(tmp_path, folder):
    make(tmp_path, "pyproject.toml", "tests/conftest.py")
    make(tmp_path, f"{folder}/conftest.py", f"{folder}/sub/ruff.toml", f"{folder}/pytest.ini")
    status, _ = run_script(tmp_path)
    assert status == 0


def make_repo_with_exclude(root: Path, exclude: str | None) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-q", "--template="], check=True)
    if exclude is not None:
        (root / ".git" / "info").mkdir()
        (root / ".git" / "info" / "exclude").write_text(exclude)


@pytest.mark.parametrize("pattern", ["local-tools/", "local-tools", "**/local-tools/"])
def test_folders_in_the_exclude_file_are_skipped_at_the_top_level(tmp_path, pattern):
    make_repo_with_exclude(tmp_path, f"# local\n\n{pattern}\n")
    make(tmp_path, "pyproject.toml", "tests/conftest.py")
    make(tmp_path, "local-tools/conftest.py", "local-tools/sub/ruff.toml")
    status, _ = run_script(tmp_path)
    assert status == 0


def test_an_exclude_folder_of_the_same_name_deeper_down_is_still_walked(tmp_path):
    make_repo_with_exclude(tmp_path, "local-tools/\n")
    make(tmp_path, "pyproject.toml", "tests/conftest.py", "src/local-tools/ruff.toml")
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == ["src/local-tools/ruff.toml"]


def test_a_folder_missing_from_the_exclude_file_is_walked(tmp_path):
    make_repo_with_exclude(tmp_path, "local-tools/\n")
    make(tmp_path, "pyproject.toml", "tests/conftest.py", "other-tools/conftest.py")
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == ["other-tools/conftest.py"]


def test_no_exclude_file_and_no_repository_skip_nothing_extra(tmp_path):
    make_repo_with_exclude(tmp_path, None)
    make(tmp_path, "pyproject.toml", "tests/conftest.py", "local-tools/conftest.py")
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == ["local-tools/conftest.py"]


def test_an_unreadable_exclude_file_exits_2(tmp_path):
    make_repo_with_exclude(tmp_path, None)
    (tmp_path / ".git" / "info").mkdir()
    (tmp_path / ".git" / "info" / "exclude").mkdir()  # a folder where the file should be
    make(tmp_path, "pyproject.toml", "tests/conftest.py")
    status, out = run_script(tmp_path)
    assert status == 2
    assert out.startswith("COULD NOT RUN")


def test_watched_names_are_exactly_the_expected_set():
    assert set(settings_files.SETTINGS_NAMES) == EXPECTED_NAMES


@pytest.mark.parametrize("name", sorted(settings_files.SETTINGS_NAMES))
def test_every_watched_name_fails_in_a_subfolder(tmp_path, name):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", f"sub/{name}")
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == [f"sub/{name}"]


@pytest.mark.parametrize("name", sorted(settings_files.SETTINGS_NAMES - {"pyproject.toml"}))
def test_every_watched_name_fails_at_the_top_level(tmp_path, name):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", name)
    status, out = run_script(tmp_path)
    assert status == 1
    assert out.splitlines() == [name]


@pytest.mark.parametrize("folder", sorted(settings_files.CACHE_FOLDERS))
def test_cache_folders_are_skipped_at_any_depth(tmp_path, folder):
    make(tmp_path, "pyproject.toml", "tests/conftest.py", f"tests/{folder}/conftest.py")
    status, _ = run_script(tmp_path)
    assert status == 0
