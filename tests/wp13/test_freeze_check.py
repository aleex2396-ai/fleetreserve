"""`tools/freeze_check.py`, the three items of WP13's pass mark, on made-up repositories with
made-up sealed folders in the temporary folder (`tests/wp2/fake_heldout.py`). No test reads the
project's own data folder.

1. `tag`: no tag fails; a branch named frozen-plan, alone and in other letter case, fails and is
   named as git lists it; a tag named in other letter case, with loose and packed refs, fails,
   and so does one beside the tag; a tag that names a tree or a file fails; a lightweight and an
   annotated tag at HEAD in a clean tree pass, also with packed refs; HEAD on a commit after the
   tag or before it could not run; an untracked file, a modified tracked file and a tracked file
   hidden from `git status` could not run.
2. `hashes`: everything equal passes; a held-out entry changed in a later commit and changed back
   fails at the commit that changed it; an entry differing at HEAD fails; a merge whose first
   parent did not have its change fails; the list deleted at HEAD fails; a byte changed in a
   sealed file fails; an unlisted file in the sealed folder fails; a listed file missing could
   not run; no commit holding the list could not run; a change to an entry that is not held-out
   passes, and the list added in a second commit makes that commit the one of download and split;
   a change found in the full history could not run in a shallow clone of it, with a grafts file
   and with a replacement ref; a sealed file or set folder leading outside the sealed folder
   through a link could not run, whatever bytes it leads to; a link that stays inside is followed,
   and a sealed folder that is itself a link passes.
3. `openings`: an absent log with the sealed files present passes; an empty log passes; a line
   written by the real opener at a made-up tag fails, the opening printed; a line that is not a
   log line could not run, and so do several, counted; openings before and after an unreadable
   line (not JSON, other fields, not UTF-8, cut short) fail, each printed; a readable log read
   one line at a time gives the loader's own openings; a link at the log's path could not run;
   no sealed folder, and a sealed folder missing one listed file, could not run. The owner's
   approvals record beside the log leaves the three items as they were, at the tag and at an
   approved parsing fix.
4. Nothing sealed is opened for parsing (`tests/wp6gate/sealed_watch.py`): `tag` and `openings`
   open, list or scan nothing in the made-up sealed folder; `hashes` opens each listed file once,
   inside the seal module's `file_digest` (bytes hashed, never decoded), and otherwise only scans
   the folder for names.
5. The tool runs git as the held-out loader runs it (the same settings and environment), and its
   three exit statuses reach a caller that runs the script as the pass mark does (a separate
   process); a failed import is could not run, never Python's own 1.
"""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from sealed_watch import watch

from fleetreserve import heldout_loader, heldout_seal, sealed_route
from fleetreserve.sessions import LayoutError

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "freeze_check.py"
sys.path.insert(0, str(TOP / "tools"))
sys.path.insert(0, str(TOP / "tests" / "wp2"))
freeze_check = importlib.import_module("freeze_check")
fake = importlib.import_module("fake_heldout")

CONTENT = b"made-up sealed bytes\n"
OTHER = b"other made-up sealed bytes\n"
FILES = {fake.ASR_FILE: CONTENT, fake.NORWAY_FILE: OTHER}
SEALED = "data/heldout"
ASR_PATH = f"{SEALED}/{fake.ASR_FILE[0]}/{fake.ASR_FILE[1]}"
NORWAY_PATH = f"{SEALED}/{fake.NORWAY_FILE[0]}/{fake.NORWAY_FILE[1]}"


def run(item, root, capsys):
    status = freeze_check.main([item, "--root", str(root)])
    return status, capsys.readouterr().out.splitlines()


def head(root):
    return fake.git(root, "rev-parse", "HEAD")


def made(tmp_path, *, tag=True, package=False):
    return fake.sealed_repo(tmp_path, FILES, tag=tag, package=package)


def rewrite_list(root, change):
    """Rewrite the working copy's hash list after `change` has edited its entries in place."""
    path = root / "data-hashes.json"
    entries = json.loads(path.read_text(encoding="utf-8"))["files"]
    change(entries)
    lines = ",\n".join(json.dumps(entry) for entry in entries)
    path.write_text('{"files": [\n' + lines + "\n]}\n", encoding="utf-8")


def set_field(path, field, value):
    def change(entries):
        for entry in entries:
            if entry["path"] == path:
                entry[field] = value

    return change


def add_entry(path, role, size=3, sha="0" * 64):
    def change(entries):
        entries.append({"path": path, "bytes": size, "sha256": sha, "role": role})

    return change


# --- 1. tag ------------------------------------------------------------------------------------


def test_no_tag_fails(tmp_path, capsys):
    root = made(tmp_path, tag=False)
    assert run("tag", root, capsys) == (
        1,
        ["FAILED: no tag frozen-plan: refs/tags/frozen-plan does not exist (reading F2)"],
    )


def test_a_branch_named_like_the_tag_is_not_the_tag(tmp_path, capsys):
    root = made(tmp_path, tag=False)
    fake.git(root, "branch", "frozen-plan")
    assert run("tag", root, capsys) == (
        1,
        [
            "FAILED: no tag frozen-plan: refs/tags/frozen-plan does not exist; a branch named "
            "frozen-plan exists, which is not the tag (reading F2)"
        ],
    )


def test_a_branch_named_like_the_tag_in_other_letter_case_is_named_as_git_lists_it(
    tmp_path, capsys
):
    root = made(tmp_path, tag=False)
    fake.git(root, "branch", "Frozen-Plan")
    assert run("tag", root, capsys) == (
        1,
        [
            "FAILED: no tag frozen-plan: refs/tags/frozen-plan does not exist; a branch named "
            "Frozen-Plan exists, which is not the tag (reading F2)"
        ],
    )


def tag_pass(kind, commit):
    return (
        f"the tag frozen-plan ({kind}) names commit {commit}; HEAD is that commit and the tree is "
        "clean"
    )


@pytest.mark.parametrize("packed", [False, True], ids=["loose", "packed"])
@pytest.mark.parametrize("name", ["Frozen-Plan", "FROZEN-PLAN"])
def test_a_tag_named_in_other_letter_case_is_not_the_tag(tmp_path, capsys, name, packed):
    root = made(tmp_path, tag=False)
    fake.git(root, "tag", name)
    if packed:
        fake.git(root, "pack-refs", "--all")
    assert run("tag", root, capsys) == (
        1,
        [
            "FAILED: no tag frozen-plan: refs/tags/frozen-plan does not exist; a tag named "
            f"{name} exists, which is not the tag: its name differs in letter case (reading F2)"
        ],
    )


@pytest.mark.parametrize("packed", [False, True], ids=["loose", "packed"])
def test_the_tag_by_its_exact_name_passes_with_loose_and_packed_refs(tmp_path, capsys, packed):
    root = made(tmp_path, tag=False)
    fake.git(root, "tag", "frozen-plan")
    if packed:
        fake.git(root, "pack-refs", "--all")
    assert run("tag", root, capsys) == (0, [tag_pass("lightweight", head(root))])


def test_a_tag_differing_from_the_tag_in_letter_case_alone_fails_beside_it(tmp_path, capsys):
    root = made(tmp_path, tag=False)
    fake.git(root, "tag", "frozen-plan")
    fake.git(root, "pack-refs", "--all")
    fake.git(root, "tag", "Frozen-Plan")  # loose, beside the packed tag
    listed = fake.git(root, "for-each-ref", "--format=%(refname)", "refs/tags/").split()
    assert sorted(listed) == ["refs/tags/Frozen-Plan", "refs/tags/frozen-plan"]
    assert run("tag", root, capsys) == (
        1,
        [
            "FAILED: the tag frozen-plan is not the only tag of its name: Frozen-Plan differs "
            "from it in letter case alone, and where the file system ignores letter case git may "
            "resolve one name to the other tag (reading F2)"
        ],
    )


@pytest.mark.parametrize("kind, target", [("tree", "HEAD^{tree}"), ("blob", "HEAD:.gitignore")])
def test_a_tag_that_names_no_commit_fails(tmp_path, capsys, kind, target):
    root = made(tmp_path, tag=False)
    fake.git(root, "tag", "frozen-plan", fake.git(root, "rev-parse", target))
    assert run("tag", root, capsys) == (
        1,
        [f"FAILED: the tag frozen-plan names a {kind}, not a commit"],
    )


@pytest.mark.parametrize("kind", ["lightweight", "annotated"])
def test_a_tag_at_head_in_a_clean_tree_passes(tmp_path, capsys, kind):
    root = made(tmp_path, tag=False)
    if kind == "lightweight":
        fake.git(root, "tag", "frozen-plan")
    else:
        fake.git(root, "tag", "-a", "-m", "the freeze", "frozen-plan")
    assert run("tag", root, capsys) == (0, [tag_pass(kind, head(root))])


def test_head_after_the_tag_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    tagged = head(root)
    (root / "later.txt").write_text("a later commit\n", encoding="utf-8")
    later = fake.commit(root, "a later commit")
    assert run("tag", root, capsys) == (
        2,
        [
            f"COULD NOT RUN: HEAD {later} is a commit after the tag frozen-plan ({tagged}); the "
            "tag is checked with HEAD at its commit (reading F1)"
        ],
    )


def test_head_before_the_tag_could_not_run(tmp_path, capsys):
    root = made(tmp_path, tag=False)
    earlier = head(root)
    (root / "later.txt").write_text("the tagged commit\n", encoding="utf-8")
    tagged = fake.commit(root, "the tagged commit")
    fake.git(root, "tag", "frozen-plan")
    fake.git(root, "checkout", "-q", "--detach", earlier)
    assert run("tag", root, capsys) == (
        2,
        [
            f"COULD NOT RUN: HEAD {earlier} is not a commit after the tag frozen-plan ({tagged}); "
            "the tag is checked with HEAD at its commit (reading F1)"
        ],
    )


def test_an_untracked_file_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    assert run("tag", root, capsys) == (
        2,
        ["COULD NOT RUN: the tree is not clean: git status lists 1 path, first ?? stray.txt"],
    )


def test_a_modified_tracked_file_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("# a change not committed\n")
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    assert run("tag", root, capsys) == (
        2,
        ["COULD NOT RUN: the tree is not clean: git status lists 2 paths, first M .gitignore"],
    )


def test_a_tracked_file_hidden_from_git_status_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    fake.git(root, "update-index", "--assume-unchanged", "data-hashes.json")
    assert run("tag", root, capsys) == (
        2,
        [
            "COULD NOT RUN: tracked files are marked assume-unchanged or skip-worktree, which "
            "hides their changes from git status: data-hashes.json"
        ],
    )


def test_a_folder_that_is_not_the_top_of_a_working_copy_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    status, lines = run("tag", root / "data", capsys)
    assert status == 2
    assert lines == [f"COULD NOT RUN: {root / 'data'} is not the top folder of a git working copy"]


# --- 2. hashes ---------------------------------------------------------------------------------


def pass_line(first, later, files=2):
    return (
        f"the {files} held-out entries of data-hashes.json committed at download and split "
        f"({first}) are unchanged in the {later} later commit(s) on HEAD's first-parent path that "
        f"change the list; the {files} files have those sizes and SHA-256 hashes and the sealed "
        "folder holds no other file outside any folder in it that is a symbolic link, which is "
        "not looked into"
    )


def test_hashes_all_equal_pass(tmp_path, capsys):
    root = made(tmp_path)
    assert run("hashes", root, capsys) == (0, [pass_line(head(root), 0)])


def test_an_entry_changed_in_a_later_commit_and_changed_back_fails(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    rewrite_list(root, set_field(ASR_PATH, "sha256", "f" * 64))
    changed = fake.commit(root, "a held-out hash changed")
    rewrite_list(root, set_field(ASR_PATH, "sha256", hashlib.sha256(CONTENT).hexdigest()))
    fake.commit(root, "the held-out hash changed back")
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: commit {changed} changes the held-out entries of data-hashes.json committed "
            f"at download and split ({first}): {ASR_PATH}: SHA-256 changed"
        ],
    )


def test_an_entry_differing_at_head_fails(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    rewrite_list(root, set_field(NORWAY_PATH, "bytes", 999))
    changed = fake.commit(root, "a held-out size changed")
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: commit {changed} changes the held-out entries of data-hashes.json committed "
            f"at download and split ({first}): {NORWAY_PATH}: {len(OTHER)} bytes became 999"
        ],
    )


def test_a_merge_whose_first_parent_lacked_its_change_fails(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    fake.git(root, "switch", "-q", "-c", "side")
    rewrite_list(root, set_field(ASR_PATH, "role", "document"))
    fake.commit(root, "a held-out role changed on a side branch")
    fake.git(root, "switch", "-q", "main")
    (root / "later.txt").write_text("a commit on the main line\n", encoding="utf-8")
    fake.commit(root, "a commit on the main line")
    fake.git(root, "merge", "-q", "--no-ff", "--no-edit", "side")
    merge = head(root)
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: commit {merge} changes the held-out entries of data-hashes.json committed "
            f"at download and split ({first}): {ASR_PATH}: role held-out became document"
        ],
    )


def test_the_list_deleted_at_head_fails(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    (root / "data-hashes.json").unlink()
    deleted = fake.commit(root, "the hash list deleted")
    status, lines = run("hashes", root, capsys)
    assert status == 1
    assert lines == [
        f"FAILED: commit {deleted} changes the held-out entries of data-hashes.json committed at "
        f"download and split ({first}): {ASR_PATH}: removed, {NORWAY_PATH}: removed"
    ]


def test_a_changed_byte_in_a_sealed_file_fails(tmp_path, capsys):
    root = made(tmp_path)
    (root / ASR_PATH).write_bytes(b"made-up sealed bytes.\n"[: len(CONTENT)])
    assert run("hashes", root, capsys) == (
        1,
        [f"FAILED: {ASR_PATH}: SHA-256 differs from the one committed at download and split"],
    )


def test_a_sealed_file_of_another_size_fails(tmp_path, capsys):
    root = made(tmp_path)
    (root / NORWAY_PATH).write_bytes(OTHER + b"x")
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: {NORWAY_PATH}: {len(OTHER) + 1} bytes, {len(OTHER)} committed at download "
            "and split"
        ],
    )


def test_an_unlisted_file_in_the_sealed_folder_fails(tmp_path, capsys):
    root = made(tmp_path)
    extra = root / SEALED / "korea" / "extra.csv"
    extra.parent.mkdir()
    extra.write_bytes(b"not listed\n")
    (root / SEALED / ".DS_Store").write_bytes(b"folder metadata\n")
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: {SEALED}/korea/extra.csv: in the sealed folder but not among the entries "
            "committed at download and split"
        ],
    )


def test_a_listed_file_missing_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    (root / ASR_PATH).unlink()
    assert run("hashes", root, capsys) == (
        2,
        [f"COULD NOT RUN: {ASR_PATH}: committed at download and split but missing"],
    )


def test_a_difference_outranks_a_missing_file(tmp_path, capsys):
    root = made(tmp_path)
    (root / ASR_PATH).unlink()
    (root / NORWAY_PATH).write_bytes(OTHER.upper())
    status, lines = run("hashes", root, capsys)
    assert status == 1
    assert lines == [
        f"COULD NOT RUN: {ASR_PATH}: committed at download and split but missing",
        f"FAILED: {NORWAY_PATH}: SHA-256 differs from the one committed at download and split",
    ]


def test_no_commit_holding_the_list_could_not_run(tmp_path, capsys):
    root = tmp_path / "repo"
    root.mkdir()
    fake.git(root, "init", "-q")
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    fake.commit(root, "no hash list")
    assert run("hashes", root, capsys) == (
        2,
        [
            "COULD NOT RUN: no commit on HEAD's first-parent path holds data-hashes.json, so "
            "nothing was committed at download and split (reading F4)"
        ],
    )


def test_a_change_to_an_entry_that_is_not_held_out_passes(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    rewrite_list(root, add_entry("data/raw/acn/caltech/2019-01.json.gz", "train"))
    fake.commit(root, "a training file listed")
    rewrite_list(root, set_field("data/raw/acn/caltech/2019-01.json.gz", "sha256", "1" * 64))
    fake.commit(root, "its hash changed")
    assert run("hashes", root, capsys) == (0, [pass_line(first, 2)])


def test_the_list_added_in_a_second_commit_is_the_commit_of_download_and_split(tmp_path, capsys):
    root = made(tmp_path, tag=False)
    text = (root / "data-hashes.json").read_text(encoding="utf-8")
    fake.git(root, "rm", "-q", "data-hashes.json")
    fake.git(root, "commit", "-q", "--amend", "-m", "no hash list yet")
    (root / "data-hashes.json").write_text(text, encoding="utf-8")
    added = fake.commit(root, "the hash list added")
    assert run("hashes", root, capsys) == (0, [pass_line(added, 0)])


REPLACED = b"replaced made-up sealed bytes\n"
FULL_HISTORY = (
    "; the item needs the full first-parent history back to the commit that first added "
    "data-hashes.json (reading F4)"
)


def replace_a_sealed_file(root):
    """A later commit that changes the a.s.r. entry, with the file on disk changed to match it, so
    that only the history shows the change."""
    (root / ASR_PATH).write_bytes(REPLACED)
    rewrite_list(root, set_field(ASR_PATH, "bytes", len(REPLACED)))
    rewrite_list(root, set_field(ASR_PATH, "sha256", hashlib.sha256(REPLACED).hexdigest()))
    return fake.commit(root, "the a.s.r. entry replaced")


def test_a_shallow_clone_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    first = head(root)
    replaced = replace_a_sealed_file(root)
    assert run("hashes", root, capsys) == (
        1,
        [
            f"FAILED: commit {replaced} changes the held-out entries of data-hashes.json committed "
            f"at download and split ({first}): {ASR_PATH}: {len(CONTENT)} bytes became "
            f"{len(REPLACED)}, SHA-256 changed",
            f"FAILED: {ASR_PATH}: {len(REPLACED)} bytes, {len(CONTENT)} committed at download and "
            "split",
        ],
    )
    clone = tmp_path / "shallow"
    fake.git(tmp_path, "clone", "-q", "--depth", "1", root.as_uri(), str(clone))
    shutil.copytree(root / SEALED, clone / SEALED)
    assert fake.git(clone, "rev-parse", "--is-shallow-repository") == "true"
    assert run("hashes", clone, capsys) == (
        2,
        [
            "COULD NOT RUN: the repository is shallow (git rev-parse --is-shallow-repository), so "
            "git holds part of its history only" + FULL_HISTORY
        ],
    )


def test_a_grafts_file_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    replaced = replace_a_sealed_file(root)
    grafts = root / ".git" / "info" / "grafts"
    grafts.parent.mkdir(exist_ok=True)
    grafts.write_text(f"{replaced}\n", encoding="utf-8")  # the replacing commit made a root
    assert run("hashes", root, capsys) == (
        2,
        [
            "COULD NOT RUN: a grafts file (.git/info/grafts) gives commits other parents than "
            "their own" + FULL_HISTORY
        ],
    )


def test_a_replacement_ref_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    replaced = replace_a_sealed_file(root)
    fake.git(root, "replace", "--graft", replaced)  # the replacing commit stood in by a root
    assert run("hashes", root, capsys) == (
        2,
        [
            "COULD NOT RUN: 1 replacement ref under refs/replace/ can stand another commit in for "
            "one of the history" + FULL_HISTORY
        ],
    )


LEADS_OUTSIDE = (
    "COULD NOT RUN: {}: leads outside the sealed folder through a symbolic link, so the held-out "
    "loader would refuse to open it; its bytes are not read (reading F8)"
)


@pytest.mark.parametrize(
    "content", [CONTENT, b"different\n"], ids=["identical copy", "other bytes"]
)
def test_a_sealed_file_leading_outside_the_sealed_folder_through_a_link_could_not_run(
    tmp_path, capsys, content
):
    root = made(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(content)
    (root / ASR_PATH).unlink()
    (root / ASR_PATH).symlink_to(outside)
    assert run("hashes", root, capsys) == (2, [LEADS_OUTSIDE.format(ASR_PATH)])


def test_a_set_folder_leading_outside_through_a_link_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    shutil.move(root / SEALED / fake.ASR_FILE[0], elsewhere)
    (root / SEALED / fake.ASR_FILE[0]).symlink_to(elsewhere, target_is_directory=True)
    assert run("hashes", root, capsys) == (2, [LEADS_OUTSIDE.format(ASR_PATH)])


def test_a_link_that_stays_inside_the_sealed_folder_is_followed(tmp_path, capsys):
    root = made(tmp_path)
    (root / ASR_PATH).unlink()
    (root / ASR_PATH).symlink_to(root / NORWAY_PATH)
    assert run("hashes", root, capsys) == (
        1,
        [f"FAILED: {ASR_PATH}: {len(OTHER)} bytes, {len(CONTENT)} committed at download and split"],
    )


def test_a_sealed_folder_that_is_itself_a_link_passes(tmp_path, capsys):
    root = made(tmp_path)
    elsewhere = tmp_path / "sealed-elsewhere"
    shutil.move(root / SEALED, elsewhere)
    (root / SEALED).symlink_to(elsewhere, target_is_directory=True)
    assert run("hashes", root, capsys) == (0, [pass_line(head(root), 0)])


# --- 3. openings -------------------------------------------------------------------------------


def openings_pass(state, files=2):
    return (
        f"no opening logged: the openings log data/heldout-openings.log is {state}, and the "
        f"working copy holds the {files} held-out files of its hash list"
    )


def test_an_absent_log_with_the_sealed_files_passes(tmp_path, capsys):
    root = made(tmp_path)
    assert run("openings", root, capsys) == (0, [openings_pass("absent")])


def test_an_empty_log_passes(tmp_path, capsys):
    root = made(tmp_path)
    (root / "data" / "heldout-openings.log").write_bytes(b"")
    assert run("openings", root, capsys) == (0, [openings_pass("empty")])


def test_a_line_written_by_the_opener_at_a_made_up_tag_fails(tmp_path, capsys):
    root = fake.sealed_repo(tmp_path, FILES)
    assert heldout_loader.open_sealed(root, *fake.ASR_FILE) == CONTENT
    [line] = heldout_loader.logged_openings(root)
    status, lines = run("openings", root, capsys)
    assert status == 1
    assert lines == [
        "FAILED: the openings log data/heldout-openings.log holds 1 opening",
        f"FAILED: opening at {line['time_utc']}: set asr, file {fake.ASR_FILE[1]}, commit "
        f"{head(root)}, tag commit {head(root)}, command {json.dumps(line['command'])}",
    ]


def test_a_line_that_is_not_a_log_line_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    (root / "data" / "heldout-openings.log").write_text("not a log line\n", encoding="utf-8")
    assert run("openings", root, capsys) == (
        2,
        [
            "COULD NOT RUN: the openings log cannot be read: ValueError: heldout-openings.log "
            "line 1 is not a log line; read again one line at a time by the same reader, line 1 "
            "cannot be read, and no line is an opening (reading F7)"
        ],
    )


def opening_line(line):
    return (
        f"FAILED: opening at {line['time_utc']}: set {line['set']}, file {line['file']}, commit "
        f"{line['commit']}, tag commit {line['tag_commit']}, command {json.dumps(line['command'])}"
    )


@pytest.mark.parametrize(
    "bad, problem",
    [
        (b"not a log line\n", "ValueError: heldout-openings.log line 2 is not a log line"),
        (b'{"time_utc": "made up"}\n', "ValueError: heldout-openings.log line 2 does not have"),
        (b"\xff\xfe made-up bytes\n", "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff"),
        (b'{"time_utc": "2026-', "ValueError: heldout-openings.log line 2 is not a log line"),
    ],
    ids=["not JSON", "other fields", "not UTF-8", "a line cut short"],
)
def test_openings_beside_an_unreadable_line_fail_and_are_printed(tmp_path, capsys, bad, problem):
    root = fake.sealed_repo(tmp_path, FILES)
    assert heldout_loader.open_sealed(root, *fake.ASR_FILE) == CONTENT
    with heldout_loader.log_path(root).open("ab") as log:
        log.write(bad)
    assert heldout_loader.open_sealed(root, *fake.NORWAY_FILE) == OTHER  # an opening after it
    with pytest.raises(ValueError):
        heldout_loader.logged_openings(root)
    lines = heldout_loader.log_path(root).read_bytes().split(b"\n")
    first, last = json.loads(lines[0]), json.loads(lines[2])
    status, printed = run("openings", root, capsys)
    assert status == 1
    assert printed[0].startswith(f"COULD NOT RUN: the openings log cannot be read: {problem}")
    assert printed[0].endswith(
        "; read again one line at a time by the same reader, line 2 cannot be read (reading F7)"
    )
    assert printed[1:] == [
        "FAILED: the openings log data/heldout-openings.log holds 2 openings",
        opening_line(first),
        opening_line(last),
    ]


def test_a_readable_log_read_one_line_at_a_time_gives_the_loaders_openings(tmp_path):
    root = fake.sealed_repo(tmp_path, FILES)
    for name in (fake.ASR_FILE, fake.NORWAY_FILE, fake.ASR_FILE):
        heldout_loader.open_sealed(root, *name)
    log, openings, unreadable = freeze_check._read_by_line(sealed_route, root)
    assert log == heldout_loader.log_path(root)
    assert (openings, unreadable) == (heldout_loader.logged_openings(root), [])
    assert len(openings) == 3


def test_unreadable_lines_are_counted_from_the_first(tmp_path, capsys):
    root = made(tmp_path)
    (root / "data" / "heldout-openings.log").write_bytes(b"\n{}\nnot a log line")
    assert run("openings", root, capsys) == (
        2,
        [
            "COULD NOT RUN: the openings log cannot be read: ValueError: heldout-openings.log "
            "line 1 is not a log line; read again one line at a time by the same reader, 3 lines "
            "cannot be read, the first line 1, and no line is an opening (reading F7)"
        ],
    )


def test_a_link_at_the_log_path_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    target = tmp_path / "elsewhere.log"
    target.write_bytes(b"")
    (root / "data" / "heldout-openings.log").symlink_to(target)
    status, lines = run("openings", root, capsys)
    assert status == 2
    assert len(lines) == 1
    assert lines[0].startswith("COULD NOT RUN: the openings log cannot be read: OSError: ")


def test_no_sealed_folder_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    shutil.rmtree(root / SEALED)
    assert run("openings", root, capsys) == (
        2,
        [
            "COULD NOT RUN: 2 of the 2 held-out files of the hash list are not in this working "
            f"copy, first {ASR_PATH}; an opening is logged only beside the sealed files "
            "(reading F5)"
        ],
    )


def test_a_sealed_folder_missing_one_listed_file_could_not_run(tmp_path, capsys):
    root = made(tmp_path)
    (root / ASR_PATH).unlink()
    assert run("openings", root, capsys) == (
        2,
        [
            "COULD NOT RUN: 1 of the 2 held-out files of the hash list are not in this working "
            f"copy, first {ASR_PATH}; an opening is logged only beside the sealed files "
            "(reading F5)"
        ],
    )


def test_an_approval_record_leaves_the_three_items_as_they_were(tmp_path, capsys):
    """The owner's approvals record (the opener's reading H7), written by hand beside the
    openings log: at the tag before any opening the three items pass; after a refusal at the
    tag and an approved parsing fix opened there, `tag` could not run (HEAD after the tag) and
    `openings` fails, as they would without the record, and `hashes` passes."""
    root = fake.sealed_repo(tmp_path, FILES, tag=False)
    tagged = fake.freeze(root)
    fake.approve(root, "0" * 40)
    assert [run(item, root, capsys)[0] for item in ("tag", "hashes", "openings")] == [0, 0, 0]
    with pytest.raises(LayoutError):
        heldout_loader.asr(root)
    fix = fake.entry(root, fake.correction(root, "asr"))
    fake.approve(root, fix)
    assert heldout_loader.open_sealed(root, *fake.ASR_FILE) == CONTENT
    status, lines = run("tag", root, capsys)
    assert status == 2
    assert lines[0].startswith(
        f"COULD NOT RUN: HEAD {fix} is a commit after the tag frozen-plan ({tagged})"
    )
    assert run("hashes", root, capsys)[0] == 0
    status, lines = run("openings", root, capsys)
    assert status == 1
    assert lines[0] == "FAILED: the openings log data/heldout-openings.log holds 2 openings"


# --- 4. nothing sealed opened for parsing ----------------------------------------------------


def test_tag_and_openings_open_list_or_scan_nothing_sealed(tmp_path, capsys):
    root = made(tmp_path)
    with watch(root / SEALED) as seen:
        assert run("tag", root, capsys)[0] == 0
        assert run("openings", root, capsys)[0] == 0
    assert seen == []


def test_hashes_opens_each_sealed_file_once_inside_the_hash_function(tmp_path, capsys, monkeypatch):
    root = made(tmp_path)
    real = heldout_seal.file_digest
    windows = []
    with watch(root / SEALED) as seen:

        def digest(path):
            start = len(seen)
            found = real(path)
            windows.append((Path(path).resolve(), start, len(seen)))
            return found

        monkeypatch.setattr(heldout_seal, "file_digest", digest)
        assert run("hashes", root, capsys)[0] == 0
    sealed = sorted((root / path).resolve() for path in (ASR_PATH, NORWAY_PATH))
    assert sorted(path for path, _, _ in windows) == sealed
    for path, start, end in windows:
        assert [(event, Path(name).resolve()) for event, name in seen[start:end]] == [
            ("open", path)
        ]
    inside = {index for _, start, end in windows for index in range(start, end)}
    outside = [event for index, event in enumerate(seen) if index not in inside]
    assert outside
    assert all(event == "os.scandir" and Path(name).is_dir() for event, name in outside)


# --- 5. git, the script and its statuses -------------------------------------------------------


def test_git_is_run_as_the_held_out_loader_runs_it(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "elsewhere")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "a settings file")
    assert freeze_check.GIT_SETTINGS == heldout_loader.GIT_SETTINGS
    assert freeze_check.git_environment() == heldout_loader.git_environment()


def script(*args, extra=None):
    environment = {**fake.tool_environment(), **(extra or {})}
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )


def test_the_script_gives_each_exit_status(tmp_path):
    passing = made(tmp_path / "passing")
    failing = made(tmp_path / "failing", tag=False)
    cannot = made(tmp_path / "cannot")
    (cannot / "stray.txt").write_text("not in git\n", encoding="utf-8")
    statuses = [
        script("tag", "--root", str(root)).returncode for root in (passing, failing, cannot)
    ]
    assert statuses == [0, 1, 2]


def test_a_wrong_call_could_not_run():
    assert script("everything").returncode == 2


def test_a_failed_import_could_not_run(tmp_path):
    shadow = tmp_path / "shadow" / "fleetreserve"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")  # a package without the modules the tool imports
    root = made(tmp_path)
    done = script("openings", "--root", str(root), extra={"PYTHONPATH": str(shadow.parent)})
    assert done.returncode == 2
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ")


def test_an_unexpected_error_could_not_run(tmp_path, capsys, monkeypatch):
    root = made(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("made-up fault")

    monkeypatch.setitem(freeze_check.ITEMS, "tag", broken)
    assert run("tag", root, capsys) == (2, ["COULD NOT RUN: RuntimeError: made-up fault"])


def test_git_variables_of_the_caller_change_nothing(tmp_path, capsys, monkeypatch):
    root = made(tmp_path)
    other = made(tmp_path / "other", tag=False)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert run("tag", root, capsys)[0] == 0
    assert os.environ["GIT_DIR"] == str(other / ".git")
