"""The gated opener's checks one by one, and in their order, in made-up repositories.

Tag and HEAD: a branch named `frozen-plan` is not the tag, beside the tag or instead of it; a
lightweight, an annotated and a nested annotated tag open, loose or packed; a tag that names a
tree is no tag; a tag whose name differs only in letter case (`Frozen-Plan`, `FROZEN-PLAN`) and
a tag named below the tag's name (`frozen-plan/x`), loose or packed, are no tag, the first named
in the refusal; beside a tag of other letter case, packed or loose, the tag's commit is the one
git lists for the exact name; HEAD on a branch and HEAD detached at the tag's commit both open.

Working copy (reading H4): a linked working copy of the repository at the tag, and a `.git` file
that points to the repository's git folder elsewhere, are refused.

Clean: a file ignored by the tree's `.gitignore`, by `.git/info/exclude` or by a folder's own
`.gitignore` does not count; an untracked file in a subfolder, a deleted tracked file and a
staged new file do; a hidden (assume-unchanged or skip-worktree) file that is not the first
entry of the index is named.

Code (reading H5): a loaded module that differs at the tag, one missing at the tag, a package
missing at the tag, a module of a subpackage that differs, a module loaded from outside the
package folder and a module stored at the tag through a clean filter that changes it are
refused, every differing file named in sorted order; a file at the tag that is not loaded, a
module without a file and a module of another package whose name begins the same do not count.

Hash list: an entry listed twice, a hash that is not 64 lower-case hexadecimal digits, a list
that is not an object of entries with `path`, `role` and `sha256`, and a set name in other
letter case are refused; another file's entry does not count.

Bytes, paths and log: the bytes returned are the bytes read and hashed, read once, also when the
file changes after the read; a root given as text opens; the log line is on disk when it is
synced; a dangling link leading outside is not present; names "." and "a\\b" are refused; a
sealed folder that is itself a link opens; a `data` that is a link is an untracked file, so the
tree is unclean; a set folder linked inside the sealed folder opens.

Order: with every check failing at once, each opening is refused by the first failing check of
the order 1 to 13, and as the faults are mended one at a time the refusal moves to the next
check, until the file opens with one log line.

Several openings: two files opened three times with a refusal between give three log lines in
order, each with its own file and hash.
"""

import hashlib
import json
import pathlib
import shutil
import sys
import time
import types
from datetime import UTC, datetime, timedelta

import pytest
from fake_heldout import ASR_FILE, NORWAY_FILE, commit, copy_package, git, sealed_repo

import fleetreserve
from fleetreserve import heldout_loader
from fleetreserve.heldout_loader import SealRefused

CONTENT = b"made-up sealed bytes\n"
OTHER = b"other made-up sealed bytes\n"


def _sealed_file(root, names=ASR_FILE):
    return root / "data" / "heldout" / names[0] / names[1]


def _opens(root, names=ASR_FILE, content=CONTENT):
    before = len(heldout_loader.logged_openings(root))
    assert heldout_loader.open_sealed(root, *names) == content
    openings = heldout_loader.logged_openings(root)
    assert len(openings) == before + 1
    assert (openings[-1]["set"], openings[-1]["file"]) == names


def _refused(root, match, names=ASR_FILE):
    before = heldout_loader.logged_openings(root)
    result = None
    with pytest.raises(SealRefused, match=match):
        result = heldout_loader.open_sealed(root, *names)
    assert result is None
    assert heldout_loader.logged_openings(root) == before


def _retag(root, message):
    commit(root, message)
    git(root, "tag", "-f", "frozen-plan")


def _hash_list(root, entries):
    lines = ",\n".join(json.dumps(entry) for entry in entries)
    (root / "data-hashes.json").write_text('{"files": [\n' + lines + "\n]}\n", encoding="utf-8")


def _entry(names=ASR_FILE, content=CONTENT, **changes):
    entry = {
        "path": f"data/heldout/{names[0]}/{names[1]}",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "role": "held-out",
    }
    entry.update(changes)
    return entry


# --- tag and HEAD ----------------------------------------------------------------------------


def test_a_branch_named_like_the_tag_is_not_the_tag(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    git(root, "branch", "frozen-plan")
    _refused(root, "no tag frozen-plan")


def test_a_branch_named_like_the_tag_beside_the_tag_is_not_read(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    git(root, "checkout", "-q", "-b", "frozen-plan")
    (root / "later.txt").write_text("a later commit\n", encoding="utf-8")
    commit(root, "a later commit on the branch frozen-plan")
    _refused(root, "after the tag")
    git(root, "checkout", "-q", "--detach", "refs/tags/frozen-plan")
    _opens(root)


@pytest.mark.parametrize("kind", ["lightweight", "annotated", "nested"])
def test_every_kind_of_tag_on_the_commit_opens(tmp_path, kind):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    if kind == "lightweight":
        git(root, "tag", "frozen-plan")
    elif kind == "annotated":
        git(root, "tag", "-a", "-m", "the tag", "frozen-plan")
    else:
        git(root, "tag", "-a", "-m", "inner", "inner")
        git(root, "tag", "-a", "-m", "outer", "frozen-plan", "inner")
        assert "\ntype tag\n" in git(root, "cat-file", "-p", "refs/tags/frozen-plan")
    _opens(root)


def test_a_tag_that_names_a_tree_is_no_tag(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    git(root, "tag", "frozen-plan", "HEAD^{tree}")
    _refused(root, "no tag frozen-plan")


def _make_tag(root, kind):
    if kind == "lightweight":
        git(root, "tag", "frozen-plan")
    elif kind == "annotated":
        git(root, "tag", "-a", "-m", "the tag", "frozen-plan")
    else:
        git(root, "tag", "-a", "-m", "inner", "inner")
        git(root, "tag", "-a", "-m", "outer", "frozen-plan", "inner")


def _packed(root, name):
    """Pack every ref; the tag `name` is then in `packed-refs` and has no loose file."""
    git(root, "pack-refs", "--all")
    assert not (root / ".git" / "refs" / "tags" / name).exists()
    packed = (root / ".git" / "packed-refs").read_text(encoding="utf-8").split()
    assert f"refs/tags/{name}" in packed


@pytest.mark.parametrize("kind", ["lightweight", "annotated", "nested"])
def test_every_kind_of_tag_on_the_commit_opens_with_the_refs_packed(tmp_path, kind):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    _make_tag(root, kind)
    _packed(root, "frozen-plan")
    _opens(root)


@pytest.mark.parametrize("packed", [False, True], ids=["loose", "packed"])
@pytest.mark.parametrize("name", ["Frozen-Plan", "FROZEN-PLAN"])
def test_a_tag_whose_name_differs_only_in_letter_case_is_no_tag(tmp_path, name, packed):
    # Where the file system ignores letter case, `git rev-parse refs/tags/frozen-plan` reads the
    # loose file of `Frozen-Plan` as the tag, and no longer once the refs are packed; git's
    # listing holds each name as written, so the refusal is the same on either kind of file
    # system, loose or packed, and names the tag of other letter case.
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    git(root, "tag", name)
    if packed:
        _packed(root, name)
    assert git(root, "for-each-ref", "--format=%(refname)", "refs/tags/") == f"refs/tags/{name}"
    _refused(
        root,
        r"^no tag frozen-plan: sealed files open only at the tag frozen-plan, or at a logged "
        r"parsing fix of it \(names that differ only in letter case are not the tag: "
        rf"refs/tags/{name}\)$",
    )


@pytest.mark.parametrize("packed", [False, True], ids=["loose", "packed"])
def test_a_tag_named_below_the_tags_name_is_no_tag(tmp_path, packed):
    # git's listing for the pattern `refs/tags/frozen-plan` holds `refs/tags/frozen-plan/x` (a
    # pattern matches up to a "/"): only a name equal to the tag's counts.
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    git(root, "tag", "frozen-plan/x")
    if packed:
        _packed(root, "frozen-plan/x")
    listed = git(root, "for-each-ref", "--format=%(refname)", "refs/tags/frozen-plan")
    assert listed == "refs/tags/frozen-plan/x"
    _refused(
        root,
        r"^no tag frozen-plan: sealed files open only at the tag frozen-plan, or at a logged "
        r"parsing fix of it$",
    )


@pytest.mark.parametrize("packed", ["frozen-plan", "Frozen-Plan"])
def test_beside_a_tag_of_other_letter_case_the_exact_names_commit_is_the_tags(tmp_path, packed):
    # Two loose files whose names differ only in letter case are one file where the file system
    # ignores letter case (`git tag frozen-plan` beside a loose `Frozen-Plan` stops with "already
    # exists"), but a packed name and a loose one coexist. With `frozen-plan` packed on the first
    # commit and `Frozen-Plan` loose on a later one, `git rev-parse refs/tags/frozen-plan` gives
    # the later commit there; the tag's commit is the first, as git's listing gives it.
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    tagged = git(root, "rev-parse", "HEAD")
    later = commit(root, "a later commit, the same tree")
    on = {"frozen-plan": tagged, "Frozen-Plan": later}
    loose = next(name for name in on if name != packed)
    git(root, "tag", packed, on[packed])
    _packed(root, packed)
    git(root, "tag", loose, on[loose])
    assert (root / ".git" / "refs" / "tags" / loose).is_file()
    listed = git(root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/tags/")
    assert listed.splitlines() == [
        f"refs/tags/Frozen-Plan {later}",
        f"refs/tags/frozen-plan {tagged}",
    ]
    _refused(root, rf"^HEAD {later} is a commit after the tag frozen-plan \({tagged}\)")
    git(root, "checkout", "-q", "--detach", tagged)
    _opens(root)
    assert heldout_loader.logged_openings(root)[-1]["tag_commit"] == tagged


def test_head_on_a_branch_and_head_detached_at_the_tag_both_open(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert git(root, "symbolic-ref", "HEAD") == "refs/heads/main"
    _opens(root)
    git(root, "checkout", "-q", "--detach", "refs/tags/frozen-plan")
    _opens(root)
    assert len(heldout_loader.logged_openings(root)) == 2


# --- working copy (reading H4) ---------------------------------------------------------------


def test_a_linked_working_copy_at_the_tag_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    git(root, "worktree", "add", "-q", "--detach", str(tmp_path / "linked"), "frozen-plan")
    shutil.copytree(root / "data", tmp_path / "linked" / "data")
    assert git(tmp_path / "linked", "status", "--porcelain") == ""
    _refused(tmp_path / "linked", "not the repository's main working copy")
    _opens(root)


def test_a_git_file_pointing_elsewhere_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    shutil.move(root / ".git", tmp_path / "moved.git")
    (root / ".git").write_text(f"gitdir: {tmp_path / 'moved.git'}\n", encoding="utf-8")
    git(root, "config", "core.worktree", str(root))
    assert git(root, "status", "--porcelain") == ""
    _refused(root, "not the repository's main working copy")


def test_a_git_folder_whose_common_folder_is_elsewhere_is_refused(tmp_path):
    good = sealed_repo(tmp_path / "good", {ASR_FILE: CONTENT})
    root = sealed_repo(tmp_path / "bad", {ASR_FILE: CONTENT}, tag=False)
    (root / ".git" / "commondir").write_text(f"{good / '.git'}\n", encoding="utf-8")
    _refused(root, "not the repository's main working copy")


# --- clean -----------------------------------------------------------------------------------


def test_ignored_files_do_not_count(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "data" / "notes.txt").write_text("ignored by .gitignore\n", encoding="utf-8")
    exclude = root / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("private.md\n")
    (root / "private.md").write_text("ignored by the local exclude file\n", encoding="utf-8")
    (root / ".cache-of-a-tool").mkdir()
    (root / ".cache-of-a-tool" / ".gitignore").write_text("*\n", encoding="utf-8")
    (root / ".cache-of-a-tool" / "entry").write_text("ignored by its folder\n", encoding="utf-8")
    _opens(root)


def test_an_untracked_file_in_a_subfolder_is_unclean(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "src" / "fleetreserve" / "stray.py").write_text("x = 1\n", encoding="utf-8")
    _refused(root, "uncommitted changes")


def test_a_deleted_tracked_file_is_unclean(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / ".gitignore").rename(tmp_path / "moved-gitignore")
    (root / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "info" / "exclude").write_text("data/\n", encoding="utf-8")
    _refused(root, "uncommitted changes")


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_a_hidden_file_after_the_first_entry_is_named(tmp_path, flag):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    name = "src/fleetreserve/sessions.py"
    git(root, "update-index", flag, name)
    with (root / name).open("a", encoding="utf-8") as handle:
        handle.write("# a change git status does not show\n")
    assert git(root, "status", "--porcelain") == ""
    _refused(root, f"marked assume-unchanged or skip-worktree: {name}$")


def test_a_staged_new_file_is_unclean(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "new.txt").write_text("staged\n", encoding="utf-8")
    git(root, "add", "new.txt")
    _refused(root, "uncommitted changes")


# --- code (reading H5) -----------------------------------------------------------------------


def _package_file(root, name):
    return root / "src" / "fleetreserve" / name


def test_loaded_modules_that_differ_at_the_tag_are_named_in_order(tmp_path):
    # `sessions` is in `sys.modules` before `load_heldout_norway` (the opener imports
    # `load_korea`, which loads `sessions`, first; the line below checks it), so the names come
    # out in sorted order only if they are sorted.
    loaded = list(sys.modules)
    assert loaded.index("fleetreserve.sessions") < loaded.index("fleetreserve.load_heldout_norway")
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    for name in ("sessions.py", "load_heldout_norway.py"):
        with _package_file(root, name).open("a", encoding="utf-8") as handle:
            handle.write("# a change at the tag\n")
    _retag(root, "two modules changed")
    _refused(
        root,
        r"not the tag's: these module files differ from the tag or are not at it \(reading H5\): "
        r"src/fleetreserve/load_heldout_norway\.py, src/fleetreserve/sessions\.py$",
    )


def test_a_clean_filter_does_not_make_other_code_look_like_the_tags(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    git(root, "config", "filter.drop-last.clean", "sed '$d'")
    git(root, "config", "filter.drop-last.smudge", "cat")
    (root / ".gitattributes").write_text("load_asr.py filter=drop-last\n", encoding="utf-8")
    git(root, "rm", "-q", "--cached", "src/fleetreserve/load_asr.py")
    commit(root, "load_asr.py stored without its last line")
    git(root, "tag", "frozen-plan")
    assert git(root, "status", "--porcelain") == ""
    _refused(root, r"\(reading H5\): src/fleetreserve/load_asr\.py$")


def test_a_loaded_module_missing_at_the_tag_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _package_file(root, "load_heldout_norway.py").unlink()
    _retag(root, "a module removed")
    _refused(root, r"\(reading H5\): src/fleetreserve/load_heldout_norway\.py$")


def test_a_package_missing_at_the_tag_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, package=False)
    _refused(root, r"\(reading H5\): src/fleetreserve/__init__\.py, ")


def test_a_differing_module_of_a_subpackage_is_refused(tmp_path):
    import fleetreserve.models.km  # noqa: F401  (loaded, so that it is checked)

    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _opens(root)
    with (root / "src" / "fleetreserve" / "models" / "km.py").open("a", encoding="utf-8") as f:
        f.write("# a change at the tag\n")
    _retag(root, "a module of a subpackage changed")
    _refused(root, r"\(reading H5\): src/fleetreserve/models/km\.py$")


def test_a_module_loaded_from_outside_the_package_folder_is_refused(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    outside = types.ModuleType("fleetreserve.made_up")
    outside.__file__ = str(tmp_path / "made_up.py")
    monkeypatch.setitem(sys.modules, "fleetreserve.made_up", outside)
    _refused(root, "modules outside the package folder: .*made_up\\.py$")


def test_modules_reached_through_a_link_are_compared_where_the_link_leads(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    package = pathlib.Path(fleetreserve.__file__).resolve().parent
    (tmp_path / "linked-package").symlink_to(package, target_is_directory=True)
    monkeypatch.setattr(fleetreserve, "__file__", str(tmp_path / "linked-package" / "__init__.py"))
    linked = types.ModuleType("fleetreserve.linked_copy")
    linked.__file__ = str(tmp_path / "linked-package" / "heldout_seal.py")
    monkeypatch.setitem(sys.modules, "fleetreserve.linked_copy", linked)
    _opens(root)


def test_an_unloaded_file_at_the_tag_and_a_module_without_a_file_do_not_count(
    tmp_path, monkeypatch
):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _package_file(root, "not_loaded.py").write_text("x = 1\n", encoding="utf-8")
    _retag(root, "a module nobody loads")
    monkeypatch.setitem(sys.modules, "fleetreserve.no_file", types.ModuleType("no_file"))
    monkeypatch.setitem(sys.modules, "fleetreservex", types.ModuleType("fleetreservex"))
    sys.modules["fleetreservex"].__file__ = str(tmp_path / "fleetreservex.py")
    _opens(root)


def test_a_restored_package_opens_again(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    with _package_file(root, "load_asr.py").open("a", encoding="utf-8") as handle:
        handle.write("# a change at the tag\n")
    _retag(root, "a module changed")
    _refused(root, "not the tag's")
    copy_package(root)
    _retag(root, "the module restored")
    _opens(root)


# --- hash list -------------------------------------------------------------------------------


def test_an_entry_listed_twice_is_refused(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _hash_list(root, [_entry(), _entry()])
    _retag(root, "an entry listed twice")
    _refused(root, "is listed 2 times")


@pytest.mark.parametrize(
    "sha256",
    [
        hashlib.sha256(CONTENT).hexdigest()[:63],
        hashlib.sha256(CONTENT).hexdigest() + "0",
        hashlib.sha256(CONTENT).hexdigest().upper(),
        "g" * 64,
        7,
        None,
    ],
    ids=["63", "65", "upper", "not-hex", "number", "null"],
)
def test_an_ill_formed_hash_is_refused(tmp_path, sha256):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _hash_list(root, [_entry(sha256=sha256)])
    _retag(root, "an ill-formed hash")
    _refused(root, "has an ill-formed hash for data/heldout/asr/")


@pytest.mark.parametrize(
    "text",
    [
        '["files"]',
        '{"files": {"path": "x"}}',
        '{"files": ["x"]}',
        '{"entries": []}',
        '{"files": [{"path": "data/heldout/asr/x", "sha256": "0"}]}',
        '{"files": [{"path": "data/heldout/asr/x", "role": "held-out"}]}',
        "not json",
    ],
    ids=["list", "files-object", "files-text", "no-files", "no-role", "no-sha256", "text"],
)
def test_an_ill_formed_hash_list_is_refused(tmp_path, text):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    good = json.dumps(_entry())
    (root / "data-hashes.json").write_text(text.replace("]}", ", " + good + "]}"), encoding="utf-8")
    _retag(root, "an ill-formed hash list")
    with pytest.raises(SealRefused, match="data-hashes.json at the tag is ill-formed") as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert refused.value.__cause__ is None
    assert heldout_loader.logged_openings(root) == []


def test_a_hash_list_missing_at_the_tag_is_a_failed_git_call(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "data-hashes.json").unlink()
    _retag(root, "the hash list removed")
    _refused(root, f"git show {git(root, 'rev-parse', 'HEAD')}:data-hashes.json failed: ")


def test_another_files_entry_does_not_count(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _hash_list(root, [_entry(NORWAY_FILE, role="train", sha256="bad"), _entry()])
    _retag(root, "another file's entry is ill-formed")
    _opens(root)


def test_a_set_name_in_other_letter_case_is_not_listed(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _refused(root, "not listed as held-out", names=("ASR", ASR_FILE[1]))


# --- bytes, paths and log --------------------------------------------------------------------


def test_the_bytes_returned_are_the_bytes_read_once_and_hashed(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    sealed = _sealed_file(root)
    reads = []
    real_read = pathlib.Path.read_bytes

    def read_then_change(self):
        content = real_read(self)
        if self == sealed:
            reads.append(content)
            sealed.write_bytes(OTHER)
        return content

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_then_change)
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    assert reads == [CONTENT]
    assert sealed.read_bytes() == OTHER
    [line] = heldout_loader.logged_openings(root)
    assert line["sha256"] == hashlib.sha256(CONTENT).hexdigest()


@pytest.mark.parametrize("names", [(".", ASR_FILE[1]), (ASR_FILE[0], "."), ("a\\b", "x")])
def test_more_names_that_are_not_plain_are_refused(tmp_path, names):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _refused(root, "not a plain name", names=names)


def test_a_sealed_folder_that_is_a_link_opens(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (tmp_path / "kept").mkdir()
    shutil.move(root / "data" / "heldout", tmp_path / "kept" / "heldout")
    (root / "data" / "heldout").symlink_to(tmp_path / "kept" / "heldout")
    _opens(root)


def test_a_data_folder_that_is_a_link_is_an_untracked_file(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    shutil.move(root / "data", tmp_path / "kept-data")
    (root / "data").symlink_to(tmp_path / "kept-data")
    _refused(root, "uncommitted changes")


def test_a_dangling_link_leading_outside_is_not_present(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _sealed_file(root).unlink()
    _sealed_file(root).symlink_to(tmp_path / "gone.csv")
    _refused(root, "listed but not present")


def test_a_root_given_as_text_opens(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert isinstance(root.as_posix(), str)
    assert heldout_loader.open_sealed(root.as_posix(), *ASR_FILE) == CONTENT
    assert len(heldout_loader.logged_openings(root)) == 1


def test_the_log_has_its_stated_name(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert heldout_loader.log_path(root) == root / "data" / "heldout-openings.log"
    _opens(root)
    assert (root / "data" / "heldout-openings.log").is_file()


def test_the_log_time_is_utc_whatever_the_local_zone(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        assert time.localtime().tm_gmtoff == 9 * 3600
        _opens(root)
    finally:
        monkeypatch.undo()
        time.tzset()
    [line] = heldout_loader.logged_openings(root)
    when = datetime.strptime(line["time_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert abs(datetime.now(UTC) - when) < timedelta(minutes=10)


def test_the_log_line_is_on_disk_when_it_is_synced(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    seen = []
    real_fsync = heldout_loader.os.fsync

    def spy(descriptor):
        seen.append(heldout_loader.log_path(root).read_text(encoding="utf-8"))
        real_fsync(descriptor)

    monkeypatch.setattr(heldout_loader.os, "fsync", spy)
    _opens(root)
    assert len(seen) == 1 and seen[0].count("\n") == 1
    assert json.loads(seen[0])["file"] == ASR_FILE[1]


def test_a_set_folder_linked_inside_the_sealed_folder_opens(tmp_path):
    root = sealed_repo(tmp_path, {("other-set", ASR_FILE[1]): CONTENT}, listed={ASR_FILE: CONTENT})
    _sealed_file(root).parent.symlink_to(root / "data" / "heldout" / "other-set")
    _opens(root)


# --- order -----------------------------------------------------------------------------------


def test_each_check_refuses_in_its_order_as_faults_are_mended(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: OTHER}, listed={}, tag=False)
    with _package_file(root, "load_asr.py").open("a", encoding="utf-8") as handle:
        handle.write("# a change at the commit to be tagged\n")
    tagged = commit(root, "a module changed")
    git(root, "worktree", "add", "-q", "--detach", str(tmp_path / "linked"))
    (root / "later.txt").write_text("a later commit\n", encoding="utf-8")
    commit(root, "a later commit")
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    log = heldout_loader.log_path(root)
    log.mkdir()
    _stopped(root, "not a plain name", log, names=("..", ASR_FILE[1]))
    _stopped(root / "src", "not the top folder", log)
    _stopped(tmp_path / "linked", "not the repository's main working copy", log)
    _stopped(root, "no tag frozen-plan", log)
    git(root, "tag", "frozen-plan", tagged)
    _stopped(root, "after the tag", log)
    git(root, "checkout", "-q", "--detach", tagged)
    git(root, "update-index", "--skip-worktree", ".gitignore")
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("# hidden from git status\n")
    _stopped(root, "uncommitted changes", log)
    (root / "stray.txt").unlink()
    _stopped(root, "marked assume-unchanged or skip-worktree: .gitignore", log)
    git(root, "update-index", "--no-skip-worktree", ".gitignore")
    git(root, "checkout", "--", ".gitignore")
    _stopped(root, "the running code is not the tag's", log)
    copy_package(root)
    _retag(root, "the module restored")
    _stopped(root, "not listed as held-out", log)
    _hash_list(root, [_entry()])
    _retag(root, "the file listed")
    sealed = _sealed_file(root)
    sealed.unlink()
    _stopped(root, "listed but not present", log)
    (tmp_path / "outside.csv").write_bytes(CONTENT)
    sealed.symlink_to(tmp_path / "outside.csv")
    _stopped(root, "leads outside the sealed folder", log)
    sealed.unlink()
    sealed.write_bytes(OTHER)
    _stopped(root, "SHA-256 differs", log)
    sealed.write_bytes(CONTENT)
    _stopped(root, "openings log could not be written", log)
    log.rmdir()
    _opens(root)


def _stopped(at, match, log, names=ASR_FILE):
    """`at` is refused by the check that `match` names; the log, a folder, stays empty."""
    with pytest.raises(SealRefused, match=match):
        heldout_loader.open_sealed(at, *names)
    assert list(log.iterdir()) == []


# --- several openings ------------------------------------------------------------------------


def test_two_files_opened_three_times_give_three_lines_in_order(tmp_path):
    root = sealed_repo(
        tmp_path,
        {ASR_FILE: CONTENT, NORWAY_FILE: OTHER, ("asr", "unlisted.csv"): CONTENT},
        listed={ASR_FILE: CONTENT, NORWAY_FILE: OTHER},
    )
    head = git(root, "rev-parse", "HEAD")
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    assert heldout_loader.open_sealed(root, *NORWAY_FILE) == OTHER
    _refused(root, "not listed as held-out", names=("asr", "unlisted.csv"))
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    openings = heldout_loader.logged_openings(root)
    first = heldout_loader.log_path(root).read_text(encoding="utf-8").splitlines()[0]
    fields = ["time_utc", "set", "file", "sha256", "commit", "tag_commit", "command"]
    assert list(json.loads(first)) == fields
    assert [(o["set"], o["file"]) for o in openings] == [ASR_FILE, NORWAY_FILE, ASR_FILE]
    assert [o["sha256"] for o in openings] == [
        hashlib.sha256(content).hexdigest() for content in (CONTENT, OTHER, CONTENT)
    ]
    assert {(o["commit"], o["tag_commit"]) for o in openings} == {(head, head)}
    times = [o["time_utc"] for o in openings]
    assert times == sorted(times)


def test_a_log_that_stops_being_writable_refuses_the_next_opening(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT, NORWAY_FILE: OTHER})
    _opens(root)
    heldout_loader.log_path(root).chmod(0o444)
    _refused(root, "openings log could not be written", names=NORWAY_FILE)
    assert len(heldout_loader.logged_openings(root)) == 1
