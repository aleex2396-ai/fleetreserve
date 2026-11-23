"""Check 8 of the gated opener for the entry module `sys.modules["__main__"]` (reading H5).

In this process, with a made-up entry module put in place of the test runner's:
1. An entry module whose `__spec__` names a module of the package is compared as a package
   module: it opens when its file equals the tag's, and is refused, by name, when its file
   differs from the tag's or is not at the tag, and when its file lies outside the package
   folder.
2. An entry module that is not of the package and whose file lies inside the root (links
   followed) is compared with the file at the same path at the tag: a script at the tag opens,
   also when reached through a link; an ignored script that is not at the tag is refused by its
   path relative to the root.
3. Not compared: an entry module with no file, a script outside the root and an entry module of
   another package whose name begins the same.

In a new Python process, as the sealed run's commands are run (`python -m fleetreserve.<name>`):
the made-up command at the tag, loaded from the root's own `src`, opens and logs one line whose
command is the module's file and its arguments; loaded from a copy of the package outside the
root, the command is refused when it is not at the tag and when it differs from the tag's, with
nothing logged; a script in an ignored folder of the root run as `python <path>` is refused.
"""

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest
from fake_heldout import ASR_FILE, commit, copy_package, git, run_python, sealed_repo, write_command

import fleetreserve
from fleetreserve import heldout_loader
from fleetreserve.heldout_loader import SealRefused

CONTENT = b"made-up sealed bytes\n"
FOLDER = Path(fleetreserve.__file__).resolve().parent
COMMAND = "made_up_command"


def _retag(root, message):
    commit(root, message)
    git(root, "tag", "-f", "frozen-plan")


def _opens(root):
    before = len(heldout_loader.logged_openings(root))
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    assert len(heldout_loader.logged_openings(root)) == before + 1


def _refused(root, match):
    before = heldout_loader.logged_openings(root)
    result = None
    with pytest.raises(SealRefused, match=match):
        result = heldout_loader.open_sealed(root, *ASR_FILE)
    assert result is None
    assert heldout_loader.logged_openings(root) == before


def _entry(monkeypatch, file, spec_name=None):
    """Put a made-up entry module in place of this process's own."""
    entry = types.ModuleType("__main__")
    entry.__spec__ = None if spec_name is None else importlib.machinery.ModuleSpec(spec_name, None)
    if file is not None:
        entry.__file__ = str(file)
    monkeypatch.setitem(sys.modules, "__main__", entry)


def _unloaded_module() -> Path:
    """A module file of the running package that this process has not loaded."""
    for path in sorted(FOLDER.glob("*.py")):
        if path.stem != "__init__" and f"fleetreserve.{path.stem}" not in sys.modules:
            return path
    raise AssertionError("every module of the package is loaded")


# --- in this process -------------------------------------------------------------------------


def test_a_package_module_run_as_the_entry_that_differs_at_the_tag_is_refused(
    tmp_path, monkeypatch
):
    module = _unloaded_module()
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    with (root / "src" / "fleetreserve" / module.name).open("a", encoding="utf-8") as handle:
        handle.write("# a change at the tag\n")
    _retag(root, "the entry module changed at the tag")
    _entry(monkeypatch, module, f"fleetreserve.{module.stem}")
    _refused(root, rf"\(reading H5\): src/fleetreserve/{module.stem}\.py$")


def test_a_package_module_run_as_the_entry_missing_at_the_tag_is_refused(tmp_path, monkeypatch):
    module = _unloaded_module()
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "src" / "fleetreserve" / module.name).unlink()
    _retag(root, "the entry module removed at the tag")
    _entry(monkeypatch, module, f"fleetreserve.{module.stem}")
    _refused(root, rf"\(reading H5\): src/fleetreserve/{module.stem}\.py$")


def test_a_package_module_run_as_the_entry_equal_to_the_tags_opens(tmp_path, monkeypatch):
    module = _unloaded_module()
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _entry(monkeypatch, module, f"fleetreserve.{module.stem}")
    _opens(root)


def test_a_package_entry_module_outside_the_package_folder_is_refused(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    outside = write_command(tmp_path / "elsewhere" / f"{COMMAND}.py")
    _entry(monkeypatch, outside, f"fleetreserve.{COMMAND}")
    _refused(root, rf"modules outside the package folder: .*elsewhere/{COMMAND}\.py$")


def test_a_script_of_the_root_at_the_tag_opens_also_through_a_link(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    script = write_command(root / "tools" / "made_up_script.py")
    _retag(root, "a script at the tag")
    _entry(monkeypatch, script)
    _opens(root)
    (tmp_path / "linked").symlink_to(root, target_is_directory=True)
    _entry(monkeypatch, tmp_path / "linked" / "tools" / "made_up_script.py")
    _opens(root)


def test_a_script_of_the_root_that_is_not_at_the_tag_is_refused(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("scratch/\n")
    _retag(root, "an ignored folder")
    script = write_command(root / "scratch" / "made_up_script.py")
    assert git(root, "status", "--porcelain") == ""
    _entry(monkeypatch, script)
    _refused(root, r"\(reading H5\): scratch/made_up_script\.py$")
    (tmp_path / "linked").symlink_to(root, target_is_directory=True)
    _entry(monkeypatch, tmp_path / "linked" / "scratch" / "made_up_script.py")
    _refused(root, r"\(reading H5\): scratch/made_up_script\.py$")


@pytest.mark.parametrize("kind", ["no-file", "outside", "other-package"])
def test_an_entry_module_outside_the_root_or_without_a_file_is_not_compared(
    tmp_path, monkeypatch, kind
):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    outside = write_command(tmp_path / "elsewhere" / "made_up_script.py")
    if kind == "no-file":
        _entry(monkeypatch, None)
    elif kind == "outside":
        _entry(monkeypatch, outside)
    else:
        _entry(monkeypatch, outside, f"fleetreservex.{COMMAND}")
    _opens(root)


# --- in a new process ------------------------------------------------------------------------


def _run_command(root, path, cwd):
    return run_python(["-m", f"fleetreserve.{COMMAND}", str(root), *ASR_FILE], path=path, cwd=cwd)


def test_the_command_at_the_tag_run_with_m_from_the_roots_own_code_opens(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    command = write_command(root / "src" / "fleetreserve" / f"{COMMAND}.py")
    _retag(root, "the made-up command at the tag")
    done = _run_command(root, root / "src", tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, f"opened {len(CONTENT)}\n", "")
    [line] = heldout_loader.logged_openings(root)
    assert line["command"] == [str(command), str(root), *ASR_FILE]
    assert git(root, "status", "--porcelain") == ""


def test_the_command_run_with_m_that_is_not_at_the_tag_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    copy_package(tmp_path / "copy")
    write_command(tmp_path / "copy" / "src" / "fleetreserve" / f"{COMMAND}.py")
    done = _run_command(root, tmp_path / "copy" / "src", tmp_path)
    assert done.returncode == 3, done.stderr
    assert done.stdout.startswith("refused: the running code is not the tag's: ")
    assert done.stdout.endswith(f"(reading H5): src/fleetreserve/{COMMAND}.py\n")
    assert heldout_loader.logged_openings(root) == []


def test_the_command_run_with_m_that_differs_from_the_tags_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    write_command(root / "src" / "fleetreserve" / f"{COMMAND}.py")
    _retag(root, "the made-up command at the tag")
    copy_package(tmp_path / "copy")
    write_command(tmp_path / "copy" / "src" / "fleetreserve" / f"{COMMAND}.py", extra="# x\n")
    done = _run_command(root, tmp_path / "copy" / "src", tmp_path)
    assert done.returncode == 3, done.stderr
    assert done.stdout.endswith(f"(reading H5): src/fleetreserve/{COMMAND}.py\n")
    assert heldout_loader.logged_openings(root) == []
    write_command(tmp_path / "copy" / "src" / "fleetreserve" / f"{COMMAND}.py")
    done = _run_command(root, tmp_path / "copy" / "src", tmp_path)
    assert (done.returncode, done.stdout) == (0, f"opened {len(CONTENT)}\n"), done.stderr


def test_a_script_in_an_ignored_folder_of_the_root_run_as_a_file_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("scratch/\n")
    _retag(root, "an ignored folder")
    script = write_command(root / "scratch" / "made_up_script.py")
    done = run_python([str(script), str(root), *ASR_FILE], path=root / "src", cwd=tmp_path)
    assert done.returncode == 3, done.stderr
    assert done.stdout.endswith("(reading H5): scratch/made_up_script.py\n")
    assert heldout_loader.logged_openings(root) == []
