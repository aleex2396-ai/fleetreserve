"""The gated opener's git calls are not redirected by the environment, the current folder or
settings files outside the repository.

1. A `GIT_DIR` or `GIT_COMMON_DIR` that names another repository, one that has the tag, does not
   make a folder without the tag open (the folder is a repository without the tag, or a plain
   copy of the files with no repository at all).
2. Each variable git reads to find its repository, working tree, index, objects, namespace or
   settings (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_COMMON_DIR`,
   `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`,
   `GIT_CEILING_DIRECTORIES`, `GIT_CONFIG_COUNT` with `GIT_CONFIG_KEY_0` and
   `GIT_CONFIG_VALUE_0`, `GIT_CONFIG_PARAMETERS`), set to a value that would make git read
   another place, leaves an opening at the tag as it is without it.
3. An ignore rule given outside the repository (through `GIT_CONFIG_*` variables, a global or
   system settings file, `HOME`, `XDG_CONFIG_HOME`) or by the repository's own setting
   `core.excludesFile` does not hide an untracked file; neither does `status.showUntrackedFiles`.
4. A file-system monitor named in the repository's settings that reports no change does not hide
   a changed tracked file.
5. Every git call of an opening runs as `git -C <root> --no-replace-objects
   -c core.fsmonitor=false -c core.excludesFile=<null device> ...`, with no `GIT_` variable in
   its environment other than `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=<null device>`, and
   every other variable kept.
6. The current folder plays no part: inside a tagged repository a root without the tag is
   refused, inside a folder without the tag a tagged root opens, and a root given relative to the
   current folder opens as the same root given in full.
7. A `core.worktree` setting in the repository that names another folder is refused.
8. A replace ref (`git replace`) that puts another hash list in place of the committed one,
   which `git show` itself then prints, changes nothing: a file whose committed hash differs is
   refused although the replacement lists its hash, and a file whose committed hash matches
   opens although the replacement lists another.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fake_heldout import ASR_FILE, git, sealed_repo

from fleetreserve import heldout_loader
from fleetreserve.heldout_loader import SealRefused

CONTENT = b"made-up sealed bytes\n"


def _opens(root):
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    assert [(line["set"], line["file"]) for line in heldout_loader.logged_openings(root)] == [
        ASR_FILE
    ]


def _refused(root, match):
    result = None
    with pytest.raises(SealRefused, match=match):
        result = heldout_loader.open_sealed(root, *ASR_FILE)
    assert result is None
    assert heldout_loader.logged_openings(root) == []


def test_a_git_dir_with_the_tag_does_not_open_a_repository_without_it(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    bad = sealed_repo(tmp_path / "bad", {ASR_FILE: CONTENT}, tag=False)
    monkeypatch.setenv("GIT_DIR", str(good / ".git"))
    _refused(bad, "no tag frozen-plan")


def test_a_git_dir_with_the_tag_does_not_open_a_plain_copy(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    shutil.copytree(good, tmp_path / "copy", ignore=shutil.ignore_patterns(".git"))
    monkeypatch.setenv("GIT_DIR", str(good / ".git"))
    _refused(tmp_path / "copy", "not the top folder of a git repository")


def test_a_common_dir_with_the_tag_does_not_open_a_repository_without_it(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    bad = sealed_repo(tmp_path / "bad", {ASR_FILE: CONTENT}, tag=False)
    monkeypatch.setenv("GIT_COMMON_DIR", str(good / ".git"))
    _refused(bad, "no tag frozen-plan")


def _redirects(tmp_path, other):
    """Each variable or group of variables with a value that names another place."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return {
        "GIT_DIR": {"GIT_DIR": str(other / ".git")},
        "GIT_WORK_TREE": {"GIT_WORK_TREE": str(elsewhere)},
        "GIT_INDEX_FILE": {"GIT_INDEX_FILE": str(elsewhere / "index")},
        "GIT_COMMON_DIR": {"GIT_COMMON_DIR": str(other / ".git")},
        "GIT_OBJECT_DIRECTORY": {"GIT_OBJECT_DIRECTORY": str(elsewhere)},
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": {
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(other / ".git" / "objects")
        },
        "GIT_NAMESPACE": {"GIT_NAMESPACE": "elsewhere"},
        "GIT_CEILING_DIRECTORIES": {"GIT_CEILING_DIRECTORIES": str(tmp_path)},
        "GIT_CONFIG_COUNT": {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.worktree",
            "GIT_CONFIG_VALUE_0": str(elsewhere),
        },
        "GIT_CONFIG_PARAMETERS": {"GIT_CONFIG_PARAMETERS": f"'core.worktree'='{elsewhere}'"},
    }


REDIRECTS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
)


@pytest.mark.parametrize("name", REDIRECTS)
def test_a_variable_naming_another_place_leaves_an_opening_as_it_is(tmp_path, monkeypatch, name):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    other = sealed_repo(tmp_path / "other", {ASR_FILE: b"other bytes\n"}, tag=False)
    for key, value in _redirects(tmp_path, other)[name].items():
        monkeypatch.setenv(key, value)
    _opens(good)


def test_the_redirect_list_covers_every_named_variable(tmp_path):
    other = sealed_repo(tmp_path / "other", {ASR_FILE: CONTENT}, tag=False)
    assert tuple(_redirects(tmp_path, other)) == REDIRECTS


def _stray(tmp_path, root):
    """An untracked file `stray.txt` in `root`, and an ignore file outside it naming it."""
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    ignore = tmp_path / "ignore-stray"
    ignore.write_text("stray.txt\n", encoding="utf-8")
    return ignore


def _settings_file(path, ignore):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[core]\n\texcludesFile = {ignore}\n", encoding="utf-8")
    return path


def _hide_by_config_count(tmp_path, root, ignore, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(ignore))


def _hide_by_config_parameters(tmp_path, root, ignore, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", f"'core.excludesfile'='{ignore}'")


def _hide_by_global_file(tmp_path, root, ignore, monkeypatch):
    settings = _settings_file(tmp_path / "settings" / "global", ignore)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(settings))


def _hide_by_system_file(tmp_path, root, ignore, monkeypatch):
    settings = _settings_file(tmp_path / "settings" / "system", ignore)
    monkeypatch.delenv("GIT_CONFIG_NOSYSTEM", raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(settings))


def _hide_by_home(tmp_path, root, ignore, monkeypatch):
    home = tmp_path / "home"
    _settings_file(home / ".gitconfig", ignore)
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))


def _hide_by_xdg_ignore(tmp_path, root, ignore, monkeypatch):
    settings = tmp_path / "xdg"
    (settings / "git").mkdir(parents=True)
    shutil.copyfile(ignore, settings / "git" / "ignore")
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(settings))


def _hide_by_local_setting(tmp_path, root, ignore, monkeypatch):
    git(root, "config", "core.excludesFile", str(ignore))


def _hide_by_untracked_setting(tmp_path, root, ignore, monkeypatch):
    git(root, "config", "status.showUntrackedFiles", "no")


@pytest.mark.parametrize(
    "hide",
    [
        _hide_by_config_count,
        _hide_by_config_parameters,
        _hide_by_global_file,
        _hide_by_system_file,
        _hide_by_home,
        _hide_by_xdg_ignore,
        _hide_by_local_setting,
        _hide_by_untracked_setting,
    ],
    ids=lambda hide: hide.__name__.removeprefix("_hide_by_"),
)
def test_an_ignore_rule_from_outside_the_tree_does_not_hide_an_untracked_file(
    tmp_path, monkeypatch, hide
):
    root = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    ignore = _stray(tmp_path, root)
    hide(tmp_path, root, ignore, monkeypatch)
    _refused(root, "uncommitted changes")


def test_a_file_system_monitor_that_reports_no_change_does_not_hide_a_change(tmp_path):
    root = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    monitor = tmp_path / "monitor.sh"
    monitor.write_text('#!/bin/sh\nprintf "token\\0"\n', encoding="utf-8")
    monitor.chmod(0o755)
    git(root, "config", "core.fsmonitor", str(monitor))
    git(root, "status", "--porcelain")
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("# a change the monitor does not report\n")
    assert git(root, "status", "--porcelain") == ""
    _refused(root, "uncommitted changes")


def test_every_git_call_runs_at_the_root_without_git_variables(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    trace = tmp_path / "trace.txt"
    monkeypatch.setenv("GIT_TRACE", str(trace))
    monkeypatch.setenv("GIT_MADE_UP_VARIABLE", "1")
    monkeypatch.setenv("MADE_UP_PLAIN_VARIABLE", "kept")
    calls = []
    real_run = subprocess.run

    def spy(args, **kwargs):
        calls.append((list(args), kwargs.get("env")))
        return real_run(args, **kwargs)

    monkeypatch.setattr(heldout_loader.subprocess, "run", spy)
    _opens(root)
    assert len(calls) >= 6
    settings = [
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.excludesFile={os.devnull}",
    ]
    expected = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    expected.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    for args, environment in calls:
        assert args[:8] == ["git", "-C", str(root), *settings]
        assert environment == expected
    assert expected["MADE_UP_PLAIN_VARIABLE"] == "kept"
    assert not trace.exists()


def _replace_hash_list(root, content):
    """A replace ref that puts, in place of the committed hash list, one listing `content`."""
    listed = git(root, "rev-parse", "HEAD:data-hashes.json")
    other = root.parent / "other-hashes.json"
    entry = {
        "path": f"data/heldout/{ASR_FILE[0]}/{ASR_FILE[1]}",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "role": "held-out",
    }
    other.write_text(json.dumps({"files": [entry]}) + "\n", encoding="utf-8")
    git(root, "replace", listed, git(root, "hash-object", "-w", str(other)))
    assert json.loads(git(root, "show", "frozen-plan:data-hashes.json")) == {"files": [entry]}


def test_a_replace_ref_does_not_change_the_hash_list_read(tmp_path):
    root = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT}, listed={ASR_FILE: b"other\n"})
    _replace_hash_list(root, CONTENT)
    assert git(root, "status", "--porcelain") == ""
    _refused(root, "SHA-256 differs from data-hashes.json at the tag; not parsed$")


def test_a_replace_ref_does_not_stop_an_opening_at_the_tag(tmp_path):
    root = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    _replace_hash_list(root, b"other\n")
    _opens(root)


def test_the_current_folder_inside_a_tagged_repository_opens_nothing_else(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    bad = sealed_repo(tmp_path / "bad", {ASR_FILE: CONTENT}, tag=False)
    monkeypatch.chdir(good)
    _refused(bad, "no tag frozen-plan")


def test_the_current_folder_without_the_tag_does_not_stop_an_opening(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    bad = sealed_repo(tmp_path / "bad", {ASR_FILE: CONTENT}, tag=False)
    monkeypatch.chdir(bad)
    _opens(good)


def test_a_plain_folder_as_the_current_folder_and_the_root_is_refused(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    shutil.copytree(good, tmp_path / "copy", ignore=shutil.ignore_patterns(".git"))
    monkeypatch.chdir(tmp_path / "copy")
    _refused(tmp_path / "copy", "not the top folder of a git repository")


def test_a_root_relative_to_the_current_folder_opens_as_the_same_root(tmp_path, monkeypatch):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    monkeypatch.chdir(tmp_path / "good")
    _opens(good.relative_to(tmp_path / "good"))
    assert good.relative_to(tmp_path / "good") == Path("repo")
    assert len(heldout_loader.logged_openings(good)) == 1


def test_a_worktree_setting_naming_another_folder_is_refused(tmp_path):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    shutil.copytree(good, tmp_path / "copy", ignore=shutil.ignore_patterns(".git"))
    git(good, "config", "core.worktree", str(tmp_path / "copy"))
    _refused(good, "not the top folder of a git repository")
