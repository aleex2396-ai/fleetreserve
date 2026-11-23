"""The check of work package 13's pass mark (plan section 13, WP13: "Pass: tag exists, and is the
commit checked out; held-out hashes match those committed at download and split; held-out list,
run matrix, seeds (one check item ...); no opening logged"), one sub-command per clause, in the
plan's order, and of the second item of work package 11's ("Pass: closed-form worst case
reproduced on 20 days; run matrix frozen (one check item ...)").

Run from the project's top folder as
    python tools/freeze_check.py tag [--root <folder>]
    python tools/freeze_check.py hashes [--root <folder>]
    python tools/freeze_check.py lists [--root <folder>]
    python tools/freeze_check.py openings [--root <folder>]
    python tools/freeze_check.py matrix [--root <folder>]
`passmarks/wp13.sh` runs the first four, `passmarks/wp11.sh` the last. `--root` is the
project's top folder (default: the folder above `tools/`). Exit status, as every check here: 0
passed, 1 failed, 2 could not run, which is never a pass. An item that finds a failure and also
something that stops it fails (1). Each finding prints one line starting "FAILED: " or "COULD
NOT RUN: "; a pass prints one line saying what passed. Any other error, a failed import among
them, is could not run (2), never Python's own 1; a wrong call is 2 (`argparse`).

What it never does: create, move or delete a tag; open a sealed file for parsing, or decode or
print its content; write a file, apart from temporary ones: a copy of a committed hash list
(`hashes`, below), one line of the openings log at a time for a log the held-out loader's reader
cannot read (`openings`, below), and the matrices `sealed_matrix` writes (`matrix`, below). It
decides nothing about the freeze: the owner decides the tag (plan section 13).

How it reaches the sealed data. The repository rule that guards the sealed data
(`tools/repo_rules.py`, the last of its rules) lets only the files it lists name that data, and
this file is not among them. It goes through two modules that are. `data_hashes`
(`tools/data_hashes.py`, the check of WP1's first item, unchanged), whose functions read a hash
list (`load_list`), give a listed path's role from its folder (`role_of`: "held-out" for every
path in the sealed folder), list the names of the files under `data/raw/` and the sealed folder
(`data_files`, folder metadata skipped) and hash one file's raw bytes (`entry_for`, through the
seal module's `file_digest`, which never decodes them). `fleetreserve.sealed_route`, whose
`openings_log`, the one function added there for this check, gives the openings log's path and
its openings by the held-out loader's own reader. A held-out entry of a hash list is an entry
whose role is "held-out" or whose path lies in the sealed folder. Where `hashes` needs the
sealed folder itself (reading F8), it takes the shortest leading part of a listed path in it
that `role_of`, given that part followed by "/", calls held-out.

`tag` (the tag exists). Reads git only. Passed: git's own list of tags (`git for-each-ref`)
holds `refs/tags/frozen-plan`, compared letter for letter, and no other tag whose name differs
from it in letter case alone; that ref names a commit (a lightweight tag, or an annotated one,
peeled through any annotated tags); HEAD is that commit (on a branch or detached); and the tree
is clean: `git status --porcelain --untracked-files=normal` prints nothing and every entry of
`git ls-files -v` carries the tag "H" (no tracked file is marked assume-unchanged or
skip-worktree). These are the held-out loader's checks 4 to 7, with the tag's name compared
letter for letter as well: on a file system that ignores letter case, `git rev-parse
refs/tags/frozen-plan` also finds a loose tag named `Frozen-Plan` and stops finding it once the
refs are packed (`git pack-refs`, which `git gc` runs), and where two such tags exist, one
loose, it finds the loose one under either name. Failed: no such tag (a tag whose name differs
in letter case alone, or a branch named frozen-plan in any letter case, is not the tag, and is
named as git lists it when it exists), a tag whose name differs from the tag's in letter case
alone beside it, or the tag names a tree or a file. Could not run: HEAD is another commit (the
line says whether it is a commit after the tag) or names none, the tree is not clean, `--root`
is not the top folder of a git working copy, or git cannot be run. Git runs as the held-out
loader runs it:
`git -C <root> --no-replace-objects -c core.fsmonitor=false -c core.excludesFile=<null device>
...`, with every variable whose name starts with `GIT_` removed and no global or system settings
file (`GIT_SETTINGS`, `git_environment`; tests/wp13 compares them with the loader's), so the tree
is clean here when the loader finds it clean, and no variable points git at another repository.

`hashes` (held-out hashes match those committed at download and split). Two parts, both run and
both required (reading F3):
(a) From git's objects alone. The commits on HEAD's first-parent path that change
    `data-hashes.json` against their first parent, merges included, oldest first (`git rev-list
    --first-parent --reverse HEAD -- data-hashes.json`): the oldest is the commit of download
    and split (reading F4). Its held-out entries must be at least one, and each later commit of
    that list must hold the same held-out entries, path by path, with the same size, SHA-256 and
    role, so HEAD's are those too. An entry changed and changed back fails at the commit that
    changed it; the list deleted fails at the commit that deleted it; a change to any other
    entry does not count. Each version is read by WP1's own reader (`data_hashes.load_list`,
    from a temporary copy of the committed text, read with `git cat-file blob`), so a version it
    cannot read could not run. The part needs HEAD's full first-parent history back to the
    commit that first added the list, so it could not run, before reading any version, in a
    shallow repository (`git rev-parse --is-shallow-repository` prints true), with a grafts file
    (`info/grafts` in the repository's git folder) or with any ref under `refs/replace/`: each
    can hide or replace commits of that path. Git runs with replacement objects off throughout.
(b) The bytes on disk. Each held-out entry committed at download and split names a file in the
    sealed folder of the working copy with that size and SHA-256, and the sealed folder holds no
    file outside those entries. The working copy's `data-hashes.json` is not read (reading F6).
    A listed file that, with symbolic links followed, lies outside the sealed folder (itself
    with links followed) is not read (reading F8), as the held-out loader's check 11 refuses
    it; a link that stays inside is followed, and the sealed folder may itself be a link. Limit:
    the names of the files in the sealed folder come from WP1's lister (`data_files`, through the
    seal module's `sealed_files`), which does not enter a folder in the sealed folder that is a
    symbolic link, so an unlisted file inside such a folder is not seen and the item can pass
    with it there; the pass line says so.
Failed: any difference in (a) or (b), an entry committed at download and split with the role
"held-out" outside the sealed folder, or one in the sealed folder with another role. Could not
run: the history is shallow, grafted or replaced; no commit on HEAD's first-parent path holds
the list; the list as first committed names no held-out entry; a version cannot be read; a
held-out file is missing or leads outside the sealed folder through a link; git cannot be run.

`openings` (no opening logged). Reads the openings log through `sealed_route.openings_log` and,
to show that the working copy holds the sealed files, its hash list (`data_hashes.load_list`)
and whether each held-out file it names is present (`Path.is_file`, which asks the file system
for the entry alone; nothing in the sealed folder is opened, listed or read). Passed: the log is
absent or empty (the loader's reading H2: "no opening logged" means the file is absent or empty)
and the working copy holds every held-out file of its hash list, at least one (reading F5), so a
clone without the sealed files cannot pass. A log for which the loader's reader raises
`ValueError` (a line that is not JSON with the log's fields, or bytes that are not UTF-8) is read
again one line at a time (reading F7): its bytes are read, opened without following a link, and
cut at each newline byte; each line, with a newline, is written alone to the log's place under a
temporary top folder and read there by the same reader (`sealed_route.openings_log`, which also
gives that place). Failed: the log holds an opening, read whole or one line at a time; each is
printed with its time, set, file, commit, tag commit and command. Could not run: the loader's
reader raises (the lines it cannot read one at a time are counted; a link or a folder at the
log's path, its `OSError`, is not read again), the hash list is missing, empty or ill-formed
(`data_hashes.load_list`), it names no held-out file, or not every held-out file it names is
present.

`matrix` (run matrix frozen, WP11). Reads git, the run list and the matrices it names as HEAD
holds them, and the kappa-choice results each matrix records as its source; nothing in the data
folder (every path is first held to `gate_run.check_files`, which refuses it). Passed, all of:
(a) the run list (`run-lists/run-list.json`), every matrix it names by `--matrix` in either
    purpose (reading F9) and the gate matrix (`gate/matrix.json`, every matrix's other source)
    are committed at HEAD and unchanged in the working copy (not listed by `git status`, not
    marked assume-unchanged or skip-worktree), and no other path is changed, untracked or
    hidden;
(b) the run list at HEAD is well formed by its own checker (`run_list.parse`: its form and the
    rules of the sealed run that the list alone decides);
(c) each matrix, read from git's objects as the run list's checker reads it
    (`run_list.read_matrix_document`), passes `sealed_matrix.check` (its `sealed_choice` holds
    against its configurations and gate units);
(d) the kappa-choice results each matrix records (`sealed_choice.source`: a run folder of
    `runs.noindex/` and `kappa-choice.json`) are verified by their run's manifest and have the
    SHA-256 and commit the matrix records, read by the matrices' own writer's reader
    (`sealed_matrix.read_choice`, which calls `run_verify.require_verified` before it reads the
    file; reading F10); one results file per model;
(e) each matrix is exactly what `sealed_matrix.write_all` writes from those results and the
    committed gate matrix, in the same folder under the same name, so a kappa, dial, unit or
    shared value edited by hand fails even when the matrix still holds against itself;
(f) the run list's fixed names `dial-workplaces` and `dial-others` equal `sealed_choice.dials`
    of every named matrix of the main model `lgbm_tail` (the run list's `run_list.DIAL_NAMES`).
Failed: the run list or a named matrix not committed (in the working copy only), changed or
hidden; a named matrix missing while another is there; the list faulty; a check, source, writer
or dial difference; a `--matrix` that is no file a run may open. Could not run: no run list;
none of the named matrices committed or present (reading F11); the list names none; the gate
matrix missing, or not the one of the running code; a source run missing or unreadable
(`run_verify.CannotCheck`); another path of the tree not clean; git cannot be run.

`lists` (held-out list, run matrix, seeds, WP13). The checks of `matrix`, and, on every file
under `gate/sealed-matrix/` (in a folder inside it too) that the run list does not name, the
checks (a) and (c) to (e) as on a named matrix: it is committed at HEAD and unchanged in the
working copy (one untracked, ignored, staged but not committed, changed or hidden from git status
fails, where for `matrix` it is another path of a tree not clean, could not run, or not looked
at), and, read from git's objects, it passes `sealed_matrix.check`, its source is verified and is
the one its model's other matrices record, and it is what `sealed_matrix.write_all` writes (a
file there that the writer writes under no name fails). (f) stays with the named matrices: the
dials are those the run list's entries run with.
(g) the held-out list is `sealed_route.HELD_OUT_UNITS` (nine distinct units): the gate units of
    every matrix the sealed run names, and of every matrix under `gate/sealed-matrix/`, named by
    the run list or not, are exactly those units (one added, dropped, swapped or repeated fails),
    and the run list's sealed-run groups name no other unit and together name each of them;
(h) seeds (reading F12): in every entry, for each purpose it runs in, each option of the command
    whose name holds "seed" is given once, as one of the run list's fixed seeds
    (`{seed:<name>}`), never a literal number; an argument shortening such an option fails, and
    so does a seed option the command has (`run_list.command_options`) that the entry leaves at
    the command's own default, unless the command itself refuses that seed on that step (the
    owner's decision 80 of 19 November 2026): the command's module declares the options with
    which it refuses a seed option (`SEED_REFUSED_WITH`, `{seed option: (options, ...)}`, found
    in its source text before it is imported), the step gives one of them in full, and, on the
    step's own arguments as the purpose instantiates them, with the seed option and each of the
    run list's fixed seeds added in turn, the command's own parser (`parser()`) or its own check
    of the parsed arguments (`seed_refusal`) refuses every one. A declaration that neither
    enforces fails, and so does a step whose own arguments its parser refuses; a declaring
    command that cannot be imported could not run. Today this is `ceiling_report --seed` with
    `--every-day`, which draws no sample. The run list's own rules on seeds (each fixed seed
    used, rule L2; one value per option, L12; one interval seed for the scoring runs and their
    intervals, L13) are those of `run_list.parse` in (b), called, not written again.
It opens nothing in the sealed folder.

This item can pass only BEFORE the sealed run. The run from the tag opens every sealed set
through the loader, which logs each opening, so after it this item fails by design: WP13's pass
is recorded once, at the tag, before the run from the tag (WP14).

Readings of the plan's wording, each with its alternative in brackets:
  F1. "Tag exists" is checked with HEAD at the tag's commit, as the held-out loader requires
      (its reading H1). [Alternative: the tag is an ancestor of HEAD.]
  F2. No tag is failed: the pass mark's "tag exists" is false. A tag whose name differs from
      frozen-plan in letter case alone is not the tag, and beside the tag it fails too, since
      git may resolve one name to the other tag. [Alternative: could not run.]
  F3. "Held-out hashes match those committed at download and split" needs both (a), the
      committed entries unchanged, and (b), the bytes on disk. [Alternative: (a) alone.]
  F4. "Those committed at download and split" are the held-out entries of the oldest commit on
      HEAD's first-parent path that holds `data-hashes.json`, found by git in the full history
      (a shallow, grafted or replaced one could not run). [Alternative: the entries of a commit
      whose id is written in this tool.]
  F5. "The sealed folder exists in the working copy" is read as: every held-out file of the
      working copy's hash list is present, at least one. [Alternative: the sealed folder exists
      as a folder, whatever it holds; not built: this file may not name that folder (the
      repository rule above), and `sealed_route` does not take its name (the check in tests/wp2
      that no module of the package but the loader, the seal module and the held-out truth build
      imports the seal module).]
  F6. (b) compares the bytes with the held-out entries committed at download and split, read
      from git, not with the working copy's hash list. [Alternative: the working copy's list, as
      `tools/data_hashes.py --check` reads it.]
  F7. A log that the loader's reader cannot read whole is read again one line at a time by the
      same reader: it fails when any of its lines is an opening, before or after an unreadable
      line (a write cut short leaves a line without an ending, and the loader's next opening
      starts a new line after it), and could not run when none is. [Alternative: could not run
      whatever its lines hold, which leaves an opening beside an unreadable line unreported.]
  F8. A held-out file that leads outside the sealed folder through a symbolic link could not
      run and its bytes are not read, as for a missing file: the held-out loader refuses it
      before reading (its check 11), as it refuses one not present (its check 10).
      [Alternative: failed; or the bytes it leads to compared, which passes a link to an
      identical copy.]
  F9. "Run matrix" is the run list and every matrix it names by `--matrix`, of the rehearsal
      and of the sealed run; a command's default matrix that the list does not write is not
      named. [Alternative: the sealed run's matrices alone.]
  F10. The kappa-choice results are read by `sealed_matrix.read_choice`, after
      `run_verify.require_verified` passes on them and `gate_run.check_files` on their path:
      reading them is what lets (e) catch a matrix edited by hand and consistent with itself.
      [Alternative: the SHA-256 compared with the run's manifest alone, never reading the
      file, which passes such a matrix.]
  F11. A run list whose matrices are none of them written yet could not run (the state before
      `sealed_matrix` is run); one of several missing fails. [Alternative: failed in both.]
  F12. "Seeds" are the options whose name holds "seed", as the commands of the list name them;
      a seed held in a matrix (`split_seed`) is frozen with the matrix by (e), not by the run
      list. [Alternative: also a list of options that are seeds under another name.] A seed
      option is not required on a step whose command refuses it there, as (h) decides from the
      command's own parser and check, never from a list kept in this tool (the owner's decision
      80). A seed given on such a step is checked as any other (literal, shortened, twice).
      [Alternatives not taken by that decision: the command accepting a fixed seed with
      `--every-day` and ignoring it; no `--every-day` in outcome O2's steps.]
"""

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TOP = Path(__file__).resolve().parents[1]
TAG = "frozen-plan"
TAG_REF = f"refs/tags/{TAG}"
BRANCH_REF = f"refs/heads/{TAG}"
LIST_NAME = "data-hashes.json"
HELD = "held-out"  # the role of a sealed file in the hash list
GIT_SETTINGS = (
    "--no-replace-objects",
    "-c",
    "core.fsmonitor=false",
    "-c",
    f"core.excludesFile={os.devnull}",
)
GIT_SECONDS = 120
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2


class CannotRun(Exception):
    """The item could not run; the text follows "COULD NOT RUN: "."""


class Findings:
    """The lines an item prints and the status they give: a failure outranks could not run."""

    def __init__(self, say):
        self.say = say
        self.failed = False
        self.cannot = False

    def fail(self, text: str) -> None:
        self.say(f"FAILED: {text}")
        self.failed = True

    def cannot_run(self, text: str) -> None:
        self.say(f"COULD NOT RUN: {text}")
        self.cannot = True

    def status(self) -> int:
        if self.failed:
            return FAILED
        return COULD_NOT_RUN if self.cannot else PASSED


def git_environment() -> dict[str, str]:
    """The environment of every git call: the caller's without any `GIT_` variable, and with no
    global or system settings file (as the held-out loader's)."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return environment


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *GIT_SETTINGS, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=git_environment(),
            timeout=GIT_SECONDS,
        )
    except (OSError, UnicodeDecodeError, subprocess.TimeoutExpired) as problem:
        raise CannotRun(f"git {' '.join(args)} could not be run: {problem}") from problem


def _git_out(root: Path, *args: str) -> str:
    done = _git(root, *args)
    if done.returncode != 0:
        raise CannotRun(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _require_top(root: Path) -> None:
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != Path(root).resolve():
        raise CannotRun(f"{root} is not the top folder of a git working copy")


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


# --- tag -------------------------------------------------------------------------------------


def _named(kind: str, kinds: str, names: list[str]) -> str:
    if len(names) == 1:
        return f"a {kind} named {names[0]} exists, which is not the tag"
    return f"{kinds} named {', '.join(names)} exist, which are not the tag"


def check_tag(root: Path, say=print) -> int:
    """The item `tag` (module text)."""
    found = Findings(say)
    try:
        _require_top(root)
        listed = _git_out(
            root, "for-each-ref", "--format=%(refname)", "refs/tags/", "refs/heads/"
        ).splitlines()
        variants = [
            ref.removeprefix("refs/tags/")
            for ref in listed
            if ref != TAG_REF and ref.casefold() == TAG_REF.casefold()
        ]
        branches = [
            ref.removeprefix("refs/heads/")
            for ref in listed
            if ref.casefold() == BRANCH_REF.casefold()
        ]
        if TAG_REF not in listed:
            also = ""
            if variants:
                differs = "its name differs" if len(variants) == 1 else "their names differ"
                also += f"; {_named('tag', 'tags', variants)}: {differs} in letter case"
            if branches:
                also += f"; {_named('branch', 'branches', branches)}"
            found.fail(f"no tag {TAG}: {TAG_REF} does not exist{also} (reading F2)")
            return found.status()
        if variants:
            differ = "differs" if len(variants) == 1 else "differ"
            found.fail(
                f"the tag {TAG} is not the only tag of its name: {', '.join(variants)} {differ} "
                "from it in letter case alone, and where the file system ignores letter case git "
                "may resolve one name to the other tag (reading F2)"
            )
            return found.status()
        peeled = _git(root, "rev-parse", "--verify", "--quiet", f"{TAG_REF}^{{commit}}")
        if peeled.returncode != 0:
            kind = _git_out(root, "cat-file", "-t", f"{TAG_REF}^{{}}").strip()
            found.fail(f"the tag {TAG} names a {kind}, not a commit")
            return found.status()
        tag_commit = peeled.stdout.strip()
        annotated = _git_out(root, "cat-file", "-t", TAG_REF).strip() == "tag"
        head = _git_out(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if head != tag_commit:
            later = _git(root, "merge-base", "--is-ancestor", tag_commit, head).returncode == 0
            where = "a commit after" if later else "not a commit after"
            found.cannot_run(
                f"HEAD {head} is {where} the tag {TAG} ({tag_commit}); the tag is checked with "
                "HEAD at its commit (reading F1)"
            )
        changed = [
            line
            for line in _git_out(
                root, "status", "--porcelain", "--untracked-files=normal"
            ).splitlines()
            if line
        ]
        if changed:
            paths = "path" if len(changed) == 1 else "paths"
            found.cannot_run(
                f"the tree is not clean: git status lists {len(changed)} {paths}, first "
                f"{changed[0].strip()}"
            )
        hidden = [
            entry[2:]
            for entry in _git_out(root, "ls-files", "-v", "-z").split("\0")
            if entry and entry[:2] != "H "
        ]
        if hidden:
            found.cannot_run(
                "tracked files are marked assume-unchanged or skip-worktree, which hides their "
                "changes from git status: " + ", ".join(hidden)
            )
        if found.status() == PASSED:
            kind = "annotated" if annotated else "lightweight"
            say(
                f"the tag {TAG} ({kind}) names commit {tag_commit}; HEAD is that commit and the "
                "tree is clean"
            )
    except CannotRun as problem:
        found.cannot_run(str(problem))
    except Exception as problem:  # never Python's own 1; earlier failures still decide
        found.cannot_run(f"{type(problem).__name__}: {problem}")
    return found.status()


# --- hashes ----------------------------------------------------------------------------------


def _import_hash_list():
    """WP1's hash list tool, `tools/data_hashes.py` (module text); a failure is could not run."""
    try:
        import data_hashes
    except Exception as problem:
        raise CannotRun(f"the check cannot start: {type(problem).__name__}: {problem}") from problem
    return data_hashes


def held_entries(data_hashes, listed: dict) -> dict:
    """The held-out entries of a hash list (module text), by path."""
    return {
        path: entry
        for path, entry in listed.items()
        if entry.role == HELD or data_hashes.role_of(path) == HELD
    }


def _committed_entries(data_hashes, root: Path, commit: str) -> dict | None:
    """The held-out entries of the hash list in `commit`, or None when it holds no list."""
    if _git(root, "cat-file", "-e", f"{commit}:{LIST_NAME}").returncode != 0:
        return None
    text = _git_out(root, "cat-file", "blob", f"{commit}:{LIST_NAME}")
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / LIST_NAME).write_text(text, encoding="utf-8")
        try:
            listed = data_hashes.load_list(Path(folder), allow_missing=True)
        except data_hashes.CouldNotRun as problem:
            raise CannotRun(f"{LIST_NAME} at commit {commit} cannot be read: {problem}") from None
    return held_entries(data_hashes, listed)


def _changes(first: dict, later: dict | None) -> list[str]:
    """Each difference of the held-out entries `later` from `first`, path by path."""
    later = later or {}
    found = []
    for path in sorted(set(first) | set(later)):
        if path not in later:
            found.append(f"{path}: removed")
        elif path not in first:
            found.append(f"{path}: added")
        else:
            one, other = first[path], later[path]
            parts = []
            if one.bytes != other.bytes:
                parts.append(f"{one.bytes} bytes became {other.bytes}")
            if one.sha256 != other.sha256:
                parts.append("SHA-256 changed")
            if one.role != other.role:
                parts.append(f"role {one.role} became {other.role}")
            if parts:
                found.append(f"{path}: {', '.join(parts)}")
    return found


def _history_cut(root: Path) -> list[str]:
    """Each way the repository at `root` has that can hide or replace commits of HEAD's history:
    shallow, a grafts file, replacement refs (module text)."""
    cuts = []
    if _git_out(root, "rev-parse", "--is-shallow-repository").strip() == "true":
        cuts.append(
            "the repository is shallow (git rev-parse --is-shallow-repository), so git holds "
            "part of its history only"
        )
    grafts = Path(
        _git_out(root, "rev-parse", "--path-format=absolute", "--git-path", "info/grafts").strip()
    )
    if grafts.exists() or grafts.is_symlink():
        shown = _relative(grafts.parent.resolve() / grafts.name, Path(root).resolve())
        cuts.append(f"a grafts file ({shown}) gives commits other parents than their own")
    replaced = _git_out(root, "for-each-ref", "--format=%(refname)", "refs/replace/").split()
    if len(replaced) == 1:
        cuts.append(
            "1 replacement ref under refs/replace/ can stand another commit in for one of the "
            "history"
        )
    elif replaced:
        cuts.append(
            f"{len(replaced)} replacement refs under refs/replace/ can stand other commits in "
            "for those of the history"
        )
    return cuts


def _committed_part(data_hashes, root: Path, found: Findings) -> tuple[str, dict, int] | None:
    """Part (a) of `hashes` (module text): (the commit of download and split, its held-out
    entries, the number of later commits that change the list), or None when it could not
    run."""
    _require_top(root)
    cuts = _history_cut(root)
    for cut in cuts:
        found.cannot_run(
            f"{cut}; the item needs the full first-parent history back to the commit that first "
            f"added {LIST_NAME} (reading F4)"
        )
    if cuts:
        return None
    commits = _git_out(
        root, "rev-list", "--first-parent", "--reverse", "HEAD", "--", LIST_NAME
    ).split()
    if not commits:
        found.cannot_run(
            f"no commit on HEAD's first-parent path holds {LIST_NAME}, so nothing was committed "
            "at download and split (reading F4)"
        )
        return None
    first_commit, *later_commits = commits
    first = _committed_entries(data_hashes, root, first_commit)
    if not first:
        found.cannot_run(
            f"{LIST_NAME} as first committed ({first_commit}) names no held-out entry (reading F4)"
        )
        return None
    for commit in later_commits:
        changes = _changes(first, _committed_entries(data_hashes, root, commit))
        if changes:
            found.fail(
                f"commit {commit} changes the held-out entries of {LIST_NAME} committed at "
                f"download and split ({first_commit}): {', '.join(changes)}"
            )
    return first_commit, first, len(later_commits)


def _sealed_folder(data_hashes, path: str) -> str:
    """The sealed folder of a `path` in it, found without naming it (module text): the shortest
    leading part of `path` that `data_hashes.role_of`, given that part and "/", calls held-out."""
    parts = path.split("/")
    for count in range(1, len(parts)):
        lead = "/".join(parts[:count])
        if data_hashes.role_of(f"{lead}/") == HELD:
            return lead
    raise CannotRun(f"{path}: no leading part of it is the sealed folder")


def _disk_part(data_hashes, root: Path, first: dict, found: Findings) -> None:
    """Part (b) of `hashes` (module text)."""
    for path, entry in sorted(first.items()):
        if data_hashes.role_of(path) != HELD:
            found.fail(f"{path}: listed as {HELD} at download and split, outside the sealed folder")
            continue
        if entry.role != HELD:
            found.fail(
                f"{path}: in the sealed folder, listed as {entry.role} at download and split"
            )
        if not (Path(root) / path).is_file():
            found.cannot_run(f"{path}: committed at download and split but missing")
            continue
        folder = (Path(root) / _sealed_folder(data_hashes, path)).resolve()
        if folder not in (Path(root) / path).resolve().parents:
            found.cannot_run(
                f"{path}: leads outside the sealed folder through a symbolic link, so the "
                "held-out loader would refuse to open it; its bytes are not read (reading F8)"
            )
            continue
        on_disk = data_hashes.entry_for(root, path, entry.role)
        if on_disk.bytes != entry.bytes:
            found.fail(
                f"{path}: {on_disk.bytes} bytes, {entry.bytes} committed at download and split"
            )
        elif on_disk.sha256 != entry.sha256:
            found.fail(f"{path}: SHA-256 differs from the one committed at download and split")
    for path in data_hashes.data_files(root):
        if data_hashes.role_of(path) == HELD and path not in first:
            found.fail(
                f"{path}: in the sealed folder but not among the entries committed at download "
                "and split"
            )


def check_hashes(root: Path, say=print) -> int:
    """The item `hashes` (module text)."""
    found = Findings(say)
    try:
        data_hashes = _import_hash_list()
        committed = _committed_part(data_hashes, root, found)
        if committed is not None:
            first_commit, first, later = committed
            _disk_part(data_hashes, root, first, found)
            if found.status() == PASSED:
                say(
                    f"the {len(first)} held-out entries of {LIST_NAME} committed at download and "
                    f"split ({first_commit}) are unchanged in the {later} later commit(s) on "
                    f"HEAD's first-parent path that change the list; the {len(first)} files have "
                    "those sizes and SHA-256 hashes and the sealed folder holds no other file "
                    "outside any folder in it that is a symbolic link, which is not looked into"
                )
    except CannotRun as problem:
        found.cannot_run(str(problem))
    except Exception as problem:  # never Python's own 1; earlier failures still decide
        found.cannot_run(f"{type(problem).__name__}: {problem}")
    return found.status()


# --- openings --------------------------------------------------------------------------------


def _read_by_line(sealed_route, root: Path) -> tuple[Path, list[dict], list[int]]:
    """Reading F7: (the openings log's path under `root`, every opening among its lines, the
    numbers of the lines that cannot be read), each line read alone by the held-out loader's own
    reader (module text)."""
    with tempfile.TemporaryDirectory() as folder:
        scratch = Path(folder)
        place, _ = sealed_route.openings_log(scratch)  # nothing there yet: the log's place only
        log = Path(root) / place.relative_to(scratch)
        no_link = lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)  # noqa: E731
        with open(log, "rb", opener=no_link) as handle:
            lines = handle.read().split(b"\n")
        if lines[-1] == b"":
            lines.pop()  # the piece after the last newline byte is no line
        place.parent.mkdir(parents=True)
        openings, unreadable = [], []
        for number, line in enumerate(lines, start=1):
            place.write_bytes(line + b"\n")
            try:
                openings += sealed_route.openings_log(scratch)[1]
            except ValueError:
                unreadable.append(number)
    return log, openings, unreadable


def check_openings(root: Path, say=print) -> int:
    """The item `openings` (module text)."""
    found = Findings(say)
    try:
        data_hashes = _import_hash_list()
        try:
            from fleetreserve import sealed_route
        except Exception as problem:
            raise CannotRun(
                f"the check cannot start: {type(problem).__name__}: {problem}"
            ) from problem
        root = Path(root)
        state = log = None
        try:
            log, openings = sealed_route.openings_log(root)
        except OSError as problem:
            found.cannot_run(
                f"the openings log cannot be read: {type(problem).__name__}: {problem}"
            )
        except ValueError as problem:
            log, openings, unreadable = _read_by_line(sealed_route, root)
            if not unreadable:
                which = "every line can be read"
            elif len(unreadable) == 1:
                which = f"line {unreadable[0]} cannot be read"
            else:
                which = f"{len(unreadable)} lines cannot be read, the first line {unreadable[0]}"
            none = "" if openings else ", and no line is an opening"
            found.cannot_run(
                f"the openings log cannot be read: {type(problem).__name__}: {problem}; read again "
                f"one line at a time by the same reader, {which}{none} (reading F7)"
            )
        if log is not None:
            name = _relative(log, root)
            if openings:
                count = "opening" if len(openings) == 1 else "openings"
                found.fail(f"the openings log {name} holds {len(openings)} {count}")
                for line in openings:
                    found.fail(
                        f"opening at {line['time_utc']}: set {line['set']}, file {line['file']}, "
                        f"commit {line['commit']}, tag commit {line['tag_commit']}, command "
                        f"{json.dumps(line['command'])}"
                    )
            else:
                state = "empty" if log.exists() or log.is_symlink() else "absent"
        try:
            listed = data_hashes.load_list(root)
        except data_hashes.CouldNotRun as problem:
            raise CannotRun(f"the sealed files cannot be found: {problem}") from None
        held = sorted(held_entries(data_hashes, listed))
        missing = [path for path in held if not (root / path).is_file()]
        if not held:
            found.cannot_run(
                f"{LIST_NAME} names no held-out file, so nothing shows that the sealed files are "
                "in this working copy (reading F5)"
            )
        elif missing:
            found.cannot_run(
                f"{len(missing)} of the {len(held)} held-out files of the hash list are not in "
                f"this working copy, first {missing[0]}; an opening is logged only beside the "
                "sealed files (reading F5)"
            )
        if found.status() == PASSED:
            say(
                f"no opening logged: the openings log {name} is {state}, and the working copy "
                f"holds the {len(held)} held-out files of its hash list"
            )
    except CannotRun as problem:
        found.cannot_run(str(problem))
    except Exception as problem:  # never Python's own 1; earlier failures still decide
        found.cannot_run(f"{type(problem).__name__}: {problem}")
    return found.status()


# --- matrix and lists --------------------------------------------------------------------------

RUN_LIST = "run-lists/run-list.json"  # the run list's one fixed path (`run_list`, module text)
MATRIX_OPTION = "--matrix"
_SEED_VALUE = re.compile(r"\{seed:([A-Za-z0-9_-]+)\}")  # a fixed seed of the run list


def _import_route():
    """The modules `matrix` and `lists` reuse (module text); a failure is could not run."""
    try:
        from fleetreserve import (
            gate_config,
            gate_run,
            kappa_choice,
            run_list,
            run_record,
            run_verify,
            sealed_matrix,
            sealed_route,
        )
        from fleetreserve.sessions import TOP as package_top
    except Exception as problem:
        raise CannotRun(f"the check cannot start: {type(problem).__name__}: {problem}") from problem
    return SimpleNamespace(
        gate_config=gate_config,
        gate_run=gate_run,
        kappa_choice=kappa_choice,
        run_list=run_list,
        run_record=run_record,
        run_verify=run_verify,
        sealed_matrix=sealed_matrix,
        sealed_route=sealed_route,
        template=sealed_matrix.TEMPLATE.relative_to(package_top).as_posix(),
    )


def _tree_state(root: Path) -> tuple[dict[str, str], set[str]]:
    """(each path `git status` lists, untracked files one by one, with its two-letter code;
    the tracked paths marked assume-unchanged or skip-worktree)."""
    parts = _git_out(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0")
    changed: dict[str, str] = {}
    k = 0
    while k < len(parts):
        if parts[k]:
            code = parts[k][:2]
            changed[parts[k][3:]] = code
            if code[0] in "RC" and k + 1 < len(parts):  # a rename or copy: then its source
                k += 1
                changed[parts[k]] = code
        k += 1
    hidden = {
        entry[2:]
        for entry in _git_out(root, "ls-files", "-v", "-z").split("\0")
        if entry and entry[:2] != "H "
    }
    return changed, hidden


def _committed_blob(root: Path, rel: str) -> bool:
    found = _git(root, "cat-file", "-t", f"HEAD:{rel}")
    return found.returncode == 0 and found.stdout.strip() == "blob"


def _frozen_state(root: Path, rel: str, changed: dict, hidden: set) -> tuple[str, str] | None:
    """None when the file `rel` is committed at HEAD and unchanged in the working copy; else
    ("missing", why) when it is neither committed nor present, or ("changed", why)."""
    if not _committed_blob(root, rel):
        if (Path(root) / rel).exists() or (Path(root) / rel).is_symlink():
            return "changed", f"{rel} is not committed at HEAD: it is in the working copy only"
        return "missing", f"{rel} is neither committed at HEAD nor in the working copy"
    if rel in changed:
        return "changed", f"{rel} is changed in the working copy (git status {changed[rel]!r})"
    if rel in hidden:
        return "changed", (
            f"{rel} is marked assume-unchanged or skip-worktree, which hides its changes from "
            "git status"
        )
    return None


def _where(k: int, label: str) -> str:
    return f"entry {k + 1} ({label})"


def _named_matrices(route, root: Path, run_list, found: Findings) -> dict[str, set[str]]:
    """The matrices the run list names by `--matrix`, in either purpose, as paths from the top
    folder: path -> the purposes that name it (reading F9). A value that is no path of the
    working copy a run may open fails, before anything at it is looked at."""
    rl, named = route.run_list, {}
    for purpose in rl.PURPOSES:
        for step in rl.instantiate(run_list, purpose):
            where = f"{_where(step.position, step.label)}, {purpose}"
            for option in rl.step_options(step, (MATRIX_OPTION,))[MATRIX_OPTION]:
                if option.value is None or dict(step.runs).get(option.place) is not None:
                    found.fail(f"{where}: its {MATRIX_OPTION} names no committed file")
                    continue
                try:
                    if option.value.startswith("~"):
                        raise route.gate_run.CouldNotRun("it is written with ~")
                    text = route.gate_run.argument_path(option.value)
                    full = Path(os.path.normpath(Path(root) / text))
                    rel = full.relative_to(Path(os.path.normpath(root))).as_posix()
                    route.gate_run.check_files([full])
                except route.gate_run.CouldNotRun as problem:
                    found.fail(f"{where}: its {MATRIX_OPTION} {option.value}: {problem}")
                    continue
                except ValueError:
                    found.fail(
                        f"{where}: its {MATRIX_OPTION} {option.value} lies outside the working copy"
                    )
                    continue
                named.setdefault(rel, set()).add(purpose)
    return named


def _plain_name(name) -> bool:
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and not any(mark in name for mark in ("/", "\\", "\0"))
    )


def _source_choice(route, root: Path, rel: str, document: dict, found: Findings, read: dict):
    """The kappa-choice results the matrix `rel` records as its source, read by
    `sealed_matrix.read_choice` after `gate_run.check_files`, once per run folder (`read`); None
    when they cannot be used (each reason printed; reading F10)."""
    source = document["sealed_choice"]["source"]
    fields = ("commit", "results", "run_folder", "sha256")
    if not isinstance(source, dict) or sorted(source) != list(fields):
        found.fail(f"{rel}: sealed_choice.source has not the fields {', '.join(fields)}")
        return None
    folder, name = source["run_folder"], source["results"]
    results = route.kappa_choice.RESULTS_NAME
    if not _plain_name(folder) or name != results:
        found.fail(
            f"{rel}: sealed_choice.source names {folder!r}/{name!r}, not a file {results} in a "
            f"run folder of {route.run_record.RUNS_FOLDER}/; nothing read there"
        )
        return None
    if folder not in read:
        path = Path(root) / route.run_record.RUNS_FOLDER / folder / name
        read[folder] = None
        shown = f"{route.run_record.RUNS_FOLDER}/{folder}/{name}"
        try:
            route.gate_run.check_files([path, path.parent / route.run_record.MANIFEST_NAME])
            read[folder] = route.sealed_matrix.read_choice(path)
        except route.gate_run.CouldNotRun as problem:
            found.fail(f"{rel}: its source {shown}: {problem}")
        except route.run_verify.CannotCheck as problem:
            found.cannot_run(f"{rel}: its source {shown} cannot be verified: {problem}")
        except route.run_verify.NotVerified as problem:
            found.fail(f"{rel}: its source {shown} is refused by its run's manifest: {problem}")
        except route.sealed_matrix.CouldNotRun as problem:
            found.fail(f"{rel}: its source {shown}: {problem}")
    choice = read[folder]
    if choice is None:
        return None
    differ = [f for f in ("sha256", "commit") if choice["source"][f] != source[f]]
    if differ:
        found.fail(
            f"{rel}: sealed_choice.source records another {' and '.join(differ)} than its "
            f"kappa-choice results {route.run_record.RUNS_FOLDER}/{folder}/{name} have now"
        )
        return None
    return choice


def _first_keys(one: dict, other: dict) -> str:
    keys = sorted(k for k in set(one) | set(other) if one.get(k) != other.get(k))
    return ", ".join(keys[:3]) + (", ..." if len(keys) > 3 else "")


def _same_number(text, number) -> bool:
    try:
        return float(text) == float(number)
    except (TypeError, ValueError):
        return False


def _sealed_folder_files(route, root: Path, changed: dict, hidden: set) -> set[str]:
    """Every file under the sealed matrices' folder (`gate/sealed-matrix/`): tracked at HEAD,
    in the index, untracked (ignored ones too) or listed by `git status`, as paths from the top
    folder."""
    folder = f"{Path(route.template).parent.as_posix()}/{route.sealed_matrix.SEALED_FOLDER}/"
    tracked = _git_out(root, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", folder)
    others = _git_out(root, "ls-files", "--others", "-z", "--", folder)
    staged = _git_out(root, "ls-files", "--cached", "-z", "--", folder)
    files = {p for text in (tracked, others, staged) for p in text.split("\0") if p}
    return files | {p for p in (set(changed) | hidden) if p.startswith(folder)}


def _frozen_matrices(route, root: Path, found: Findings, every_sealed: bool = False) -> dict | None:
    """The checks of `matrix` (module text), each finding printed; what `lists` reads further:
    {"run_list", "named", "unnamed", "documents", "head"}, or None when the run list cannot be
    read. With `every_sealed` (the item `lists`), the checks (a) and (c) to (e) are also those of
    every file under `gate/sealed-matrix/` that the run list does not name ("unnamed")."""
    rl, sm = route.run_list, route.sealed_matrix
    _require_top(root)
    head = _git_out(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    changed, hidden = _tree_state(root)
    reported = set()
    state = _frozen_state(root, RUN_LIST, changed, hidden)
    reported.add(RUN_LIST)
    if state is not None:
        kind, why = state
        if kind == "missing":
            found.cannot_run(f"no run list: {why}")
            return None
        found.fail(why)
        if why.endswith("in the working copy only"):
            return None
    try:
        text = _git_out(root, "cat-file", "blob", f"HEAD:{RUN_LIST}")
        run_list = rl.parse(rl.decode(text.encode("utf-8")), name_max=rl.name_limit(Path(root)))
    except rl.Faulty as problem:
        for line in problem.faults:
            found.fail(f"{RUN_LIST} at HEAD is faulty by its own checker (run_list): {line}")
        return None
    except rl.CannotDecide as problem:
        found.cannot_run(f"{RUN_LIST} at HEAD cannot be checked by run_list: {problem}")
        return None
    named = _named_matrices(route, root, run_list, found)
    if not named:
        found.cannot_run(f"{RUN_LIST} at HEAD names no matrix by {MATRIX_OPTION}")
    states = {rel: _frozen_state(root, rel, changed, hidden) for rel in sorted(named)}
    reported.update(states)
    missing = [rel for rel, s in states.items() if s is not None and s[0] == "missing"]
    if named and len(missing) == len(named):
        found.cannot_run(
            f"none of the {len(named)} matrices {RUN_LIST} names is committed or in the working "
            f"copy, first {missing[0]} (sealed_matrix writes them; reading F11)"
        )
    else:
        for rel in missing:
            found.fail(states[rel][1])
    for s in states.values():
        if s is not None and s[0] == "changed":
            found.fail(s[1])
    unnamed: list[str] = []
    if every_sealed:
        unnamed = sorted(_sealed_folder_files(route, root, changed, hidden) - set(named))
        for rel in unnamed:
            state = _frozen_state(root, rel, changed, hidden)
            if state is not None:
                found.fail(state[1])
        reported.update(unnamed)
    template = route.template
    template_state = _frozen_state(root, template, changed, hidden)
    reported.add(template)
    if template_state is not None:
        if template_state[0] == "missing":
            found.cannot_run(f"the gate matrix, every matrix's other source: {template_state[1]}")
        else:
            found.fail(f"the gate matrix, every matrix's other source: {template_state[1]}")
    elif _git_out(root, "cat-file", "blob", f"HEAD:{template}") != sm.TEMPLATE.read_text():
        found.cannot_run(
            f"{template} committed at HEAD is not the gate matrix of the running code "
            f"({sm.TEMPLATE}), from which sealed_matrix writes the matrices"
        )
    others = sorted((set(changed) | hidden) - reported)
    if others:
        paths = "path" if len(others) == 1 else "paths"
        found.cannot_run(
            f"the tree is not clean: {len(others)} other {paths} changed, untracked or hidden "
            f"from git status, first {others[0]}"
        )
    documents = {}
    for rel in [*states, *unnamed]:
        if not _committed_blob(root, rel):
            continue
        read, why = rl.read_matrix_document(Path(root), rel)
        if read is None:
            found.fail(f"{rel}: {why}")
            continue
        documents[rel] = read[1]
    choices: dict[str, dict | None] = {}
    by_model: dict[str, dict[str, dict]] = {}
    sources: dict[str, dict] = {}
    for rel, document in documents.items():
        try:
            problems = sm.check(document)
        except route.gate_config.MatrixError as problem:
            problems = [f"not a matrix: {problem}"]
        for problem in problems:
            found.fail(f"{rel}: its sealed_choice does not hold (sealed_matrix.check): {problem}")
        if problems:
            continue
        choice = _source_choice(route, root, rel, document, found, choices)
        if choice is not None:
            sources[rel] = choice
            by_model.setdefault(choice["model"], {})[choice["source"]["run_folder"]] = choice
    for model, runs in sorted(by_model.items()):
        if len(runs) > 1:
            found.fail(
                f"the matrices of the model {model} name {len(runs)} kappa-choice results: "
                f"{', '.join(sorted(runs))}"
            )
    one = [next(iter(runs.values())) for runs in by_model.values() if len(runs) == 1]
    if one:
        with tempfile.TemporaryDirectory() as folder:
            written = sm.write_all(one, Path(folder))
            gate = str(Path(template).parent.as_posix())
            expected = {
                f"{gate}/{path.relative_to(folder).as_posix()}": json.loads(path.read_text())
                for path in written
            }
        for rel in sorted(sources):
            if len(by_model[sources[rel]["model"]]) > 1:
                continue
            want = expected.get(rel)
            if want is None:
                found.fail(
                    f"{rel}: sealed_matrix writes no matrix of that path from its kappa-choice "
                    f"results (it writes {', '.join(sorted(expected))})"
                )
            elif want != documents[rel]:
                found.fail(
                    f"{rel}: differs from what sealed_matrix writes from its kappa-choice results "
                    f"and the committed gate matrix, in {_first_keys(want, documents[rel])}"
                )
    names = run_list.fixed["names"]
    main = [
        rel
        for rel, d in documents.items()
        if rel in named and d["sealed_choice"].get("model") == sm.MAIN_MODEL
    ]
    if any(rel in named for rel in documents) and not main:
        found.fail(
            f"{RUN_LIST} names no committed matrix of the main model {sm.MAIN_MODEL}, whose "
            "dials its fixed names hold"
        )
    for rel in main:
        dials = documents[rel]["sealed_choice"].get("dials")
        for name, site_type in rl.DIAL_NAMES.items():
            wanted = dials.get(site_type) if isinstance(dials, dict) else None
            if not _same_number(names.get(name), wanted):
                found.fail(
                    f"{RUN_LIST}: its fixed name {name} is {names.get(name)!r}, not the dial "
                    f"{wanted} that sealed_choice.dials gives {site_type} in {rel}"
                )
    return {
        "run_list": run_list,
        "named": named,
        "unnamed": unnamed,
        "documents": documents,
        "head": head,
    }


def check_matrix(root: Path, say=print) -> int:
    """The item `matrix` (module text)."""
    found = Findings(say)
    try:
        route = _import_route()
        frozen = _frozen_matrices(route, root, found)
        if frozen is None and found.status() == PASSED:
            found.cannot_run("the run list was not read, so nothing was checked")
        if frozen is not None and found.status() == PASSED:
            documents = frozen["documents"]
            sealed_run = route.run_list.SEALED_RUN
            sealed = sum(1 for p in frozen["named"].values() if sealed_run in p)
            say(
                f"run matrix frozen: {RUN_LIST} and the {len(documents)} matrices it names "
                f"({sealed} named by the sealed run) are committed at HEAD {frozen['head']} and "
                "unchanged; each passes sealed_matrix.check, its kappa-choice results are "
                "verified by their runs' manifests with the SHA-256 it records, and it is what "
                "sealed_matrix writes from them and the committed gate matrix; the run list's "
                f"dials are those of the main model {route.sealed_matrix.MAIN_MODEL}"
            )
    except CannotRun as problem:
        found.cannot_run(str(problem))
    except Exception as problem:  # never Python's own 1; earlier failures still decide
        found.cannot_run(f"{type(problem).__name__}: {problem}")
    return found.status()


def _held_out_list(route, frozen: dict, found: Findings) -> int:
    """The held-out list's part of `lists` (module text); the number of sealed matrices read."""
    nine = list(route.sealed_route.HELD_OUT_UNITS)
    if not nine or len(set(nine)) != len(nine):
        found.cannot_run(f"sealed_route.HELD_OUT_UNITS is empty or repeats a unit: {nine}")
        return 0
    folder = f"{Path(route.template).parent.as_posix()}/{route.sealed_matrix.SEALED_FOLDER}/"
    sealed = [
        rel
        for rel, purposes in sorted(frozen["named"].items())
        if route.run_list.SEALED_RUN in purposes or rel.startswith(folder)
    ] + list(frozen["unnamed"])
    read = 0
    for rel in sealed:
        document = frozen["documents"].get(rel)
        if document is None:
            continue  # reported by the run matrix's part
        read += 1
        units = document.get("gate_units")
        units = list(units) if isinstance(units, list) else []
        added = sorted(set(units) - set(nine))
        dropped = sorted(set(nine) - set(units))
        twice = sorted({u for u in units if units.count(u) > 1})
        said = [f"adds {', '.join(added)}"] if added else []
        said += [f"drops {', '.join(dropped)}"] if dropped else []
        said += [f"names {', '.join(twice)} more than once"] if twice else []
        if said:
            found.fail(
                f"{rel}: its gate units are not the {len(nine)} held-out units of "
                f"sealed_route.HELD_OUT_UNITS: it {'; '.join(said)}"
            )
    groups = frozen["run_list"].units[route.run_list.SEALED_RUN]
    for name, group in sorted(groups.items()):
        other = [u for u in group if u not in nine]
        if other:
            found.fail(
                f"{RUN_LIST}: its sealed-run group {name!r} names {', '.join(other)}, not a "
                "held-out unit of sealed_route.HELD_OUT_UNITS"
            )
    unnamed = sorted(set(nine) - {u for group in groups.values() for u in group})
    if unnamed:
        found.fail(
            f"{RUN_LIST}: no sealed-run group names the held-out unit(s) {', '.join(unnamed)}"
        )
    return read


def _command_start(parts) -> int:
    """The place after a run-list command's `-m fleetreserve.<module>` (well formed: `run_list`)."""
    return next(k + 2 for k in range(len(parts) - 1) if parts[k] == "-m")


SEED_DECLARATION = "SEED_REFUSED_WITH"  # a command's seed options and the options refusing them
SEED_CHECK = "seed_refusal"  # a command's check of its parsed arguments: a refusal text or None
_LEFT = "a seed left at the command's own default, not fixed in the run list"


def _declares_refusal(module: str) -> bool:
    """Whether the source text of the package's `module` assigns `SEED_DECLARATION` at its top
    level, read with `ast` before anything is imported (module text, (h))."""
    spec = importlib.util.find_spec(f"fleetreserve.{module}")
    if spec is None or spec.origin is None:
        return False
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == SEED_DECLARATION for t in targets):
            return True
    return False


def _parses(command, argv: list[str]):
    """The command's own parser (`parser()`) on `argv`: the parsed arguments, or None when it
    refuses them (its message, written to the error stream, kept out of the item's lines)."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            return command.parser().parse_args(argv)
        except SystemExit:
            return None


def _refused_seed(command, option: str, arguments, seeds) -> tuple[bool, str | None]:
    """(whether the imported `command` refuses its seed option `option` on a step whose own
    arguments, after its module, are `arguments`, a fault text or None) (module text, (h)): only
    when its `SEED_DECLARATION` names, for `option`, an option `arguments` give in full, and its
    parser or its `SEED_CHECK` refuses `arguments` with `option` and each of `seeds` added."""
    name = command.__name__.rsplit(".", 1)[-1]
    declared = getattr(command, SEED_DECLARATION, {})
    excluders = declared.get(option, ()) if isinstance(declared, dict) else ()
    keys = []
    for part in arguments:
        if part == "--":
            break
        keys.append(part.partition("=")[0])
    given = [o for o in excluders if o in keys]
    if not given:
        return False, None
    shown = " or ".join(given)
    if _parses(command, list(arguments)) is None:
        return False, (
            f"{name}'s parser refuses the step's own arguments, so whether it refuses {option} "
            f"with {shown} is not shown: {_LEFT}"
        )
    check = getattr(command, SEED_CHECK, None)
    for seed in seeds:
        parsed = _parses(command, [*arguments, option, str(seed)])
        if parsed is None:
            continue  # the parser refuses it
        if callable(check) and check(parsed) is not None:
            continue  # its own check refuses it
        return False, (
            f"{name} declares {option} refused with {shown} ({SEED_DECLARATION}), but neither "
            f"its parser nor its {SEED_CHECK} refuses {option} {seed} with the step's other "
            f"arguments: {_LEFT}"
        )
    return True, None


def _seeds(route, run_list, found: Findings) -> tuple[int, int]:
    """The seeds' part of `lists` (module text, reading F12): (the number of seed options read,
    the number of seed options left out where the step's command refuses them)."""
    rl, given, refused = route.run_list, 0, 0
    seeds = sorted(set(run_list.fixed["seeds"].values())) or [0]
    steps = {
        (step.position, purpose): step
        for purpose in rl.PURPOSES
        for step in rl.instantiate(run_list, purpose)
    }
    for k, entry in enumerate(run_list.entries):
        module = entry.command[_command_start(entry.command) - 1].split(".", 1)[1]
        options, _ = rl.command_options(module)
        seed_options = sorted(o for o in options if "seed" in o.lower())
        for purpose in rl.PURPOSES:
            if not entry.runs_in(purpose):
                continue
            where = f"{_where(k, entry.label)}, {purpose}"
            parts = (*entry.command, *entry.extra.get(purpose, ()))
            counts: dict[str, int] = {}
            j = _command_start(parts)
            while j < len(parts):
                part = parts[j]
                if part == "--":
                    break
                key, equals, value = part.partition("=")
                shortens = [o for o in seed_options if o.startswith(key) and o != key]
                is_seed = key.startswith("--") and len(key) > 2 and "seed" in key.lower()
                if key.startswith("--") and len(key) > 2 and shortens and key not in options:
                    found.fail(f"{where}: its argument {part!r} shortens {' or '.join(shortens)}")
                    is_seed = True
                if is_seed:
                    if not equals:
                        value = parts[j + 1] if j + 1 < len(parts) else None
                        j += 1
                    counts[key] = counts.get(key, 0) + 1
                    given += 1
                    if value is None or not _SEED_VALUE.fullmatch(value):
                        found.fail(
                            f"{where}: its {key} is {value!r}, not one of the run list's fixed "
                            "seeds {seed:<name>}: a literal seed"
                        )
                j += 1
            for key, count in sorted(counts.items()):
                if count > 1:
                    found.fail(f"{where}: gives {key} {count} times, two seeds for one draw")
            for option in seed_options:
                if option in counts:
                    continue
                if _declares_refusal(module):
                    try:
                        command = importlib.import_module(f"{rl.PACKAGE}.{module}")
                    except Exception as problem:
                        found.cannot_run(
                            f"{where}: whether {module} refuses {option} there is not decided: "
                            f"it cannot be imported: {type(problem).__name__}: {problem}"
                        )
                        continue
                    step = steps[(k, purpose)]
                    arguments = step.command[_command_start(step.command) :]
                    is_refused, fault = _refused_seed(command, option, arguments, seeds)
                    if is_refused:
                        refused += 1
                        continue
                    if fault is not None:
                        found.fail(f"{where}: {fault}")
                        continue
                found.fail(
                    f"{where}: leaves {module}'s {option} at the command's own default, a "
                    "seed not fixed in the run list"
                )
    return given, refused


def check_lists(root: Path, say=print) -> int:
    """The item `lists` (module text)."""
    found = Findings(say)
    try:
        route = _import_route()
        frozen = _frozen_matrices(route, root, found, every_sealed=True)
        if frozen is None and found.status() == PASSED:
            found.cannot_run("the run list was not read, so nothing was checked")
        if frozen is not None:
            sealed = _held_out_list(route, frozen, found)
            given, refused = _seeds(route, frozen["run_list"], found)
            if found.status() == PASSED:
                nine = len(route.sealed_route.HELD_OUT_UNITS)
                fixed = len(frozen["run_list"].fixed["seeds"])
                say(
                    f"held-out list, run matrix, seeds: the {nine} held-out units of "
                    f"sealed_route.HELD_OUT_UNITS are the gate units of each of the {sealed} "
                    f"sealed matrices ({len(frozen['unnamed'])} of them under gate/sealed-matrix/ "
                    f"not named by {RUN_LIST}, each committed, unchanged and checked as a named "
                    "matrix is) and the units of the run list's sealed-run groups; the run "
                    "matrix frozen as the item matrix checks it; every one of the "
                    f"{given} seed options of the run list's entries is one of its {fixed} fixed "
                    "seeds, none literal, none given twice or left at a command's default, and "
                    "run_list finds no fault in the list (its rule on the interval seed, L13, "
                    "and one value per option, L12, among them); the "
                    f"{refused} seed options left out of steps whose own command refuses them "
                    "there are each refused by that command's parser or seed_refusal (the "
                    "owner's decision 80)"
                )
    except CannotRun as problem:
        found.cannot_run(str(problem))
    except Exception as problem:  # never Python's own 1; earlier failures still decide
        found.cannot_run(f"{type(problem).__name__}: {problem}")
    return found.status()


ITEMS = {
    "tag": check_tag,
    "hashes": check_hashes,
    "openings": check_openings,
    "matrix": check_matrix,
    "lists": check_lists,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check the items of WP13's pass mark.")
    parser.add_argument("item", choices=sorted(ITEMS), help="the item to check")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)  # a wrong call exits with status 2
    try:
        return ITEMS[args.item](args.root)
    except Exception as problem:
        print(f"COULD NOT RUN: {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
