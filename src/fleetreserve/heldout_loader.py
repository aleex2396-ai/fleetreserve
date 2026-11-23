"""The held-out loader: the gated opening of a sealed file for parsing, at the tag (plan section 3).

The plan's seal rule: "Only the held-out loader (tested on the published schema and a fake file)
opens them, at the tag `frozen-plan` with a clean tree, logging each opening. After the tag:
parsing fixes only, each logged with its diff before any result is seen." `open_sealed` is that
opening, at the tag's commit or at a logged parsing fix after it (check 5, reading H6) whose
commit the owner has approved (check 8a, reading H7).

"A logged parsing fix" of the tag, in every text of the package that names one (the sealed
routes' labels, their checks text and their refusals among them), is a commit after the tag's
commit that passes every rule of reading H6 below: one line of single-parent commits from the
tag, each a correction of one held-out set's loader and its tests or an entry naming the
corrections in plan section 14, HEAD an entry, and each corrected set's layout refusal logged in
the refusal log `data/heldout-refusals.log` before its correction, a later correction of a set
with a refusal of its own, logged after the set's previous correction. This is the one place that
says what it is; the other texts point here.

Scope: the checks defend against accidents under an honest caller: stray environment variables
and git settings, a replace ref (`git replace`), a wrong folder or working copy, a moved HEAD, a
changed or extra file, code that is not the tag's (the entry module run with `python -m` or as a
script among it), a tag that is missing or elsewhere, a path that leaves the sealed folder, an
openings log that is a symbolic link, and a log line that cannot be written. They do not defend
against deliberate tampering with git itself (a rigged git program, hand-edited objects, refs or
index) or with the code (edits made to get round a check, modules changed after they are
loaded); that is left to review of the change. Not covered either:
- A tag deleted and made again on another commit, with HEAD there: git keeps no record of the
  tag's earlier commit, so the remade tag cannot be told from the first one here. Moving the tag
  is the owner's act; only a record kept outside this folder (the tag's commit written in plan
  section 14, or pushed) can show it.
- "One sealed folder and one openings log" (reading H4) holds per copy of the folder: a full
  copy or a clone of the whole folder, sealed files included, is the main working copy of its
  own repository, so it passes check 3 and opens with its own log beside its own sealed folder.
- A refusal counts for a parsing fix only when the openings log holds the opening it follows
  (reading H6(e)): a refusal line written by hand alone does not count, at a commit where nothing
  was opened (a correction commit among them) or with a time before its opening. Lines written
  by hand into both logs, consistent with each other (an opening and a refusal of the same set,
  file, SHA-256, commit and tag commit, the refusal's time not before the opening's, at a
  commit of the window), are read as a real opening and its refusal: the two logs are records
  kept under an honest caller, and nothing here tells a hand-written pair from one the routes
  wrote. A refusal line written by hand after a real opening whose bytes parsed counts as well,
  as a parse that succeeds leaves no record, and at the tag the sealed run opens every set's
  file, so this one edit of the refusal log can serve a first fix of any set.

`open_sealed` returns a sealed file's bytes only when every check below passes, in this order:

1. The set name and the file name are plain names: not empty, not "." or "..", without "/" or
   "\\" (`heldout_seal.sealed_path`).
2. `root` is the top folder of a git working copy: `git rev-parse --show-toplevel` names it
   (both with symbolic links followed).
3. That working copy is the repository's main one: git's common folder, which holds the
   repository's refs and objects, is the folder `root/.git` (reading H4). A linked working copy
   (`git worktree add`), a `.git` file that points to a git folder elsewhere, a submodule and a
   `.git` folder whose `commondir` file names another folder are refused.
4. The tag exists: `refs/tags/frozen-plan` names a commit (directly, or through one or more
   annotated tags). Compared: the full name `refs/tags/frozen-plan`, byte for byte, with each
   name in git's listing of the tags (`git for-each-ref refs/tags/`, loose and packed refs
   alike), the commit then reached from the object listed under it, so a tag whose name differs
   only in letter case (`Frozen-Plan`), which `git rev-parse` finds under the full name while it
   is a loose ref on a file system that ignores letter case, is no tag and is named in the
   refusal. A branch named `frozen-plan` is not read, and a tag that names a tree or a file is
   refused as no tag.
5. The checked-out commit (HEAD) is the tag's commit (reading H1), with HEAD on a branch or
   detached, or a logged parsing fix of it: a commit after the tag that passes every rule of
   reading H6. A commit that does not descend from the tag is refused as before; one after the
   tag that is not such a fix is refused with the first rule of H6 it breaks. Checks 6 to 8
   then apply at HEAD, the fix commit included, and check 9 reads the hash list at the tag's
   commit, as before. Before check 5 reads the openings log and the refusal log (only when HEAD
   is a later commit that descends from the tag), `check_no_link` looks at the data folder, the
   sealed folder and the two logs, in that order: a ".." part in a path, or a symbolic link
   (dangling or not) at one, is refused naming it, with nothing opened (`lstat` only, after the
   folder above was seen not to be a link). The same look is made by the sealed routes' gate, so
   a caller of `open_sealed` without the gate (the held-out truth build) has it too. The sealed
   routes' gate (`sealed_route.require_gate`) makes checks 2 to 8 by these same functions, so it
   passes at the same commits; check 8a is the opener's alone.
6. The tree is clean: `git status --porcelain --untracked-files=normal` prints nothing (see
   "Clean" below).
7. No tracked file is marked assume-unchanged or skip-worktree (`git update-index`), which would
   hide its changes from check 6: every entry of `git ls-files -v` carries the tag "H"; the
   refusal names the files that do not.
8. The running code is the checked-out commit's (reading H5): the tag's, or the fix commit's.
   Compared, each by the git object id of the file its `__file__` names (read with
   `git hash-object --no-filters`) against the id at that commit:
   - every module in `sys.modules` under the name `fleetreserve` or a name starting with
     `fleetreserve.` that has a `__file__`, against the file at the same place under
     `src/fleetreserve/`, where the package folder is the folder of `fleetreserve.__file__`;
   - the entry module `sys.modules["__main__"]`, when its `__spec__` names a module of the
     package (a module run with `python -m fleetreserve.<name>`, as the sealed run's commands
     are), in the same way as a package module;
   - otherwise the entry module, when it has a `__file__` that lies inside `root` (links
     followed), against the file at the same path relative to `root`: a script of the
     repository run as `python <path>`. A script inside `root` that is not at the tag (an
     ignored or untracked script, an installed tool's entry script under `.venv/`) is refused.
   Not compared: an entry module with no `__file__` (`python -c`, an interactive session, a
   program read from standard input) and an entry module whose file lies outside `root` (a test
   runner's or an installed tool's own entry script, a script of another folder): code outside
   the package and the folder, as reading H5 states. A module of the package without a
   `__file__` (a namespace package) is not compared. The refusal names each compared file that
   differs, is not at the tag or, for a module of the package, lies outside the package folder.
8a. At a parsing fix only (not at the tag): the owner's approval of HEAD is recorded (reading
   H7). The approvals record `data/heldout-fix-approvals.log` is read whole, by
   `logged_openings`; a record that is absent, that cannot be read (a folder or a symbolic link
   at its path among them) or that holds any malformed line refuses, and so does a well-formed
   record with no line whose `fix_commit` is HEAD's full id.
   Once check 8 (and at a fix 8a) has passed, the gate marks the recorded run going in this
   repository, if there is one (`gate_mark`, its readings G1 to G3); a mark that cannot be
   written refuses.
9. The file is listed exactly once in the hash list `data-hashes.json` as committed at the tag's
   commit (read with `git show`, not from the working folder), under its path
   `data/heldout/<set>/<file>`, with the role "held-out" and a SHA-256 written as 64 lower-case
   hexadecimal digits. A hash list that is not a JSON object whose `files` is a list of entries
   each with `path`, `role` and `sha256` is refused as ill-formed.
10. The file is present at that path.
11. With symbolic links followed, the file lies inside the sealed folder `data/heldout/` (itself
    with links followed), as `sessions.require_raw_path` requires under `data/raw`: a link of
    the file or of its set folder that leads outside is refused, a link that stays inside is
    followed. The sealed folder may itself be a link: where it leads is the sealed folder, and
    check 12 still compares the bytes with the committed hash.
12. The SHA-256 of its bytes equals the listed one.
13. One line is appended to the openings log (reading H2), a JSON object with the fields
    `time_utc`, `set`, `file`, `sha256`, `commit` (HEAD), `tag_commit` (the tag's commit) and
    `command` (`sys.argv`), and, at a parsing fix only, last, `fix_of_tag` (the tag's commit
    again, so a line says by itself that it was made at a fix), and flushed to disk. The log is
    opened with `O_NOFOLLOW`, so a symbolic link at the log's path,
    dangling or not, is refused and nothing is written where it leads; a missing log is made
    with the mode `open` gives a new file (0o666 less the process's umask). A log whose last
    line has no line ending gets one before the new line, so the two never run together. If the
    line cannot be written, nothing is returned.

How git is run. Every git call is `git -C <root> --no-replace-objects -c core.fsmonitor=false
-c core.excludesFile=<null device> ...`, in the caller's environment with every variable whose
name starts with `GIT_` removed and `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=<null device>`
added (`git_environment`). So no variable can point git at another repository, working tree,
index, object store, namespace or setting (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`,
`GIT_COMMON_DIR`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`,
`GIT_CEILING_DIRECTORIES`, `GIT_CONFIG_*`, `GIT_REPLACE_REF_BASE` and the rest), no global or
system settings file is read, and the current folder plays no part. `--no-replace-objects`
makes git read every object as stored, never the object a replace ref (`git replace`) puts in
its place, so the tag's tree, hash list and module files are the ones committed. The
repository's own `.git/config` is read, apart from the two settings given on the command line.

Clean (check 6). A modified, staged or deleted tracked file and an untracked file that is not
ignored make the tree unclean. Ignored means ignored by a `.gitignore` file in the tree or by the
repository's own `.git/info/exclude`, and by nothing else: not by a global ignore file, a system
or global settings file, `core.excludesFile` in the repository's settings or a variable. An
ignored file does not count, by design: everything under `data/` is ignored by the project's
`.gitignore`, and files kept only on one machine are kept out of git by `.git/info/exclude`.
A folder that holds its own `.gitignore` ignoring everything in it, as tools' cache folders do,
counts as ignored. A `data` that is a symbolic link and not a folder is not ignored by the rule
`data/`, which names folders only, so it counts as an untracked file. The file-system monitor is
off, so git looks at every file itself.

Errors. A failed check raises `SealRefused` with that check's own message and no cause. A hash
list that cannot be read as check 9 states is such a failed check: text that is not JSON (a
`ValueError`), JSON nested too deep to read (a `RecursionError`) and a list of the wrong shape
(a `KeyError` or `TypeError`) are refused as "ill-formed", the type's name in the message, with
no cause and the original exception suppressed (`from None`). Any other exception raised while
`open_sealed` runs, that is any `Exception` (an `OSError` when git cannot be run or a file
cannot be read; a `UnicodeDecodeError` when git's output, the hash list among it, is not UTF-8;
a `ValueError`, for example for a root holding a NUL character; a `TypeError` for a root or a
name of the wrong type; a `RuntimeError`; and any other), is raised as `SealRefused` with that
exception as its cause. A `BaseException` that is not an `Exception` (`KeyboardInterrupt`,
`SystemExit`) passes unchanged. In every case no bytes are returned, and a log line is written
only by check 13, after every other check has passed.

Checks 1 to 11 read no byte of the sealed file (checks 10 and 11 read the file system's entries
and links only). Check 12 reads the bytes once and hashes them (reading H3); on a mismatch the
bytes are dropped unparsed. The bytes returned are the bytes that were hashed, so a file that
changes after its read cannot change what is returned. Hashing is done here on those bytes, not
with `heldout_seal.file_digest`, which reads the file itself.

`logged_openings` reads the log back without touching a sealed file: an absent log, or one that
is empty, gives no opening; a line that is not a JSON object with exactly the fields above, in
that order (`fix_of_tag` last or absent), raises `ValueError` naming the line; an `OSError` (a
folder at the log's path, and a symbolic link there, dangling or not, which is refused as check
13 refuses it, among them) and a `UnicodeDecodeError` for bytes that are not UTF-8 are raised
unchanged. The bytes are decoded as UTF-8 whatever the locale. Given `log=REFUSALS_NAME`, it
reads the refusal log in the same way, each line with exactly the fields `REFUSAL_FIELDS`;
`logged_refusals(root)` is that call. Given `log=APPROVALS_NAME`, it reads the approvals record
(reading H7) in the same way, each line with exactly the fields `APPROVAL_FIELDS`. (The refusal
log and the approvals record are read through `logged_openings` because the test of the routes,
`tests/wp2/test_heldout_routes.py`, allows a file read in this module only in `open_sealed`,
`logged_openings` and `_append`.) `log_lines(content, log)` parses bytes already read as
`logged_openings` parses the file's, and reads no file: the WP14 check
(`tools/sealed_run_check.py`) reads the openings and refusal logs through its own reader, which
refuses a symbolic link at any folder above them too.

The refusal log (reading H6) is the append-only text file `data/heldout-refusals.log`, beside
the openings log. `norway_sites`, `asr` and `korea` write one line to it when their parser
raises `LayoutError` on the bytes just opened: a JSON object with the fields `time_utc`, `set`,
`file`, `sha256`, `commit` and `tag_commit` of that opening (the commit is HEAD: the tag's, or a
fix commit's) and `layout`, the parser's message. It is written by `_append`, as the openings
log is. The `LayoutError` is then raised unchanged; if the line cannot be written, it is raised
with a note saying so (`add_note`), and no later fix of that set can open (H6(e)).

Readings of the plan's wording, each with its alternative in brackets:

  H1. "At the tag `frozen-plan`" means the checked-out commit is exactly the tag's commit, or a
      logged parsing fix after it as reading H6 states. [Alternative: any descendant of the
      tag.]
  H2. The openings log is the one append-only text file `data/heldout-openings.log`, beside the
      sealed folder `data/heldout/` and not inside it, so the hash list's check of the sealed
      folder never sees it; the project's `.gitignore` ignores `data/`, so the log does not make
      the tree unclean. "No opening logged" (WP13's pass mark) means the file is absent or empty.
  H3. A refusal never reads content: the hash check, which reads bytes but does not parse them,
      comes last, after every other check has passed.
  H4. The opening is made only in the repository's main working copy, the one that holds the
      sealed folder, so that there is one sealed folder and one openings log. That holds per
      copy of the whole folder (see "Not covered" above): the sealed run is made in the
      project's own folder at the tag. [Alternative: any working copy of the repository at the
      tag, each with its own log.]
  H5. "At the tag" covers the code that opens and parses: the package's modules loaded when the
      file is opened, and the entry module as check 8 states, must be the tag's. Modules loaded
      later, and other code outside the package, are not checked. [Alternative: only the
      working copy at `root` is checked.]
  H6. A parsing fix. Plan section 13 (WP14): "when a loader refuses a sealed file's layout at
      the opening, the refusal and the layout fact behind it are logged, that loader alone is
      corrected in a commit entered in section 14, the owner decides the tag for that commit".
      The owner's decisions of 17 November 2026: the tag never moves, and the opener also
      accepts a logged fix commit that differs from the tag only in the allowed files; a fix may
      touch only that data set's loader, its tests and plan section 14. HEAD, a commit after the
      tag's commit T, is a logged parsing fix of it when all of these hold, checked in this
      order:
      (a) Every commit after T up to HEAD (`git rev-list --parents T..HEAD`) has exactly one
          parent, and they form one line from T to HEAD: no merge lies between.
      (b) Every path that differs between T and HEAD is a held-out set's loader, a test file of
          one, or `docs/plan.md`; the paths are those `git diff --name-only -z --no-renames
          --no-relative --ignore-submodules=none` lists, as for the earlier-run rule of
          `run_evidence`. A set's loader is the module whose `from_bytes` its route calls
          (`LOADERS`): `src/fleetreserve/load_heldout_norway.py`, `load_asr.py` and
          `load_korea.py`. A test file of it is a file under `tests/`, at any depth, named
          `test_<loader>.py` or `test_<loader>_<anything>.py`. [Alternatives: the test files
          at T only; any file under `tests/`.]
      (c) Each commit of the line is, by the same paths against its parent, either a correction
          of one set (it changes files of that set's loader and tests and nothing else) or an
          entry (it changes `docs/plan.md` only, and only inside section 14: the lines after
          the line `## 14. Changes after the tag`, which must appear exactly once at both
          commits, up to the next line that starts a heading of level 1 or 2). Refused: a
          commit that changes nothing, one that changes two sets' files, one that changes the
          plan with other files. HEAD is an entry, and at least one correction lies on the
          line: a fix is two commits, the correction, then the entry naming it, since a commit
          cannot hold its own id. Every commit is checked, not only the difference from T to
          HEAD, so a path changed and changed back inside the line is refused, and the plan
          differs from T's only inside section 14. [Alternatives: a correction and its entry in
          one commit, named by a later one; the difference from T to HEAD alone; an entry with
          no correction before it, whose code is the tag's, opening too.]
      (d) Section 14 at HEAD names each correction by its full id (not as part of a longer run
          of hexadecimal digits).
      (e) For each correction, the refusal log holds a line of its set and file with
          `tag_commit` T and `commit` a commit at which that set's loader was as the correction
          found it: for the set's first correction on the line, T or a commit of the line
          before the correction (a layout refusal at an opening at the tag, or at an earlier fix
          of the same line, as when the held-out truth build, stopped at the tag by one set, is
          run again at that set's fix and stopped by the next); for a later correction of the
          same set, a commit of the line after the set's previous correction and before this
          one (a refusal at an opening at the fix commit that the later correction corrects).
          A refusal counts only when the openings log holds the opening it follows
          (`_opened_before`): an opening of the same set, file, SHA-256, commit and tag commit,
          with a time not after the refusal's, as every route writes them (`open_sealed` logs
          the opening before it returns the bytes, and `_parsed` logs the refusal after the
          parser raised). [Alternative: the refusal log alone, under which a refusal line
          written by hand at a commit where nothing was opened counts.] When the window holds a
          refusal line of the set with no such opening, the refusal message says so.
          So a refusal of a set counts for one correction of it only, and one excuse never
          opens a set twice: a refusal logged at T, or one of another set, never covers a
          second correction of the set; two corrections of one set with no fix commit between
          them are refused, since no refusal can be logged between them. The reading for a
          set's first correction was confirmed on 18 November 2026. [Alternative: at T
          only, under which a set first refused at a fix could not be corrected without moving
          the tag.] The rule for a later correction is the owner's decision that a second fix of
          a set needs its own logged refusal, read as stated here. [Alternatives: only at the
          fix commit just before the later correction; two corrections of one set with no fix
          commit between them read as one.]
      The rules look at commits, paths, the plan's section 14 and the refusal log, never at what
      a correction changes inside the loader; since one set's loader can be imported by another
      set's (`load_heldout_norway` imports `load_asr`), a fix's diff is read by a person before
      the run continues. The tag stays on T; an opening at a fix logs
      `fix_of_tag`, and the sealed routes record the fix commit (`sealed_route.fix_record`), so
      the results can state it.
  H7. The owner's approval of a fix commit. The owner's decision 65, answer 11 (18 November
      2026): each parsing-fix commit is approved in writing before the loader opens files at it.
      Its record is the text file `data/heldout-fix-approvals.log`, beside the openings log and,
      like it, outside git (the project's `.gitignore` ignores `data/`) and outside the sealed
      folder. It is written by a person, by hand, after the owner's yes: no function of the
      package or of `tools/` writes it (`tests/wp2/test_heldout_fix_approval.py` reads their
      text for that). Each line is a JSON object with exactly the fields `time_utc` (the UTC
      time of the yes, written YYYY-MM-DDThh:mmZ, seconds and up to six decimals of them
      allowed), `fix_commit` (the approved commit's full id, 40 or 64 lower-case hexadecimal
      digits) and `words` (the owner's words, not empty), in that order, for example
      `{"time_utc": "2026-11-25T09:30Z", "fix_commit": "<full id>", "words": "yes to the fix"}`.
      An approval names one commit: it does not carry to another fix, to a later correction of
      the same set, or to a commit that differs from the named one in any byte (its message
      among them), since each of these has another id. At the tag no record is needed or read.
      A record that cannot be read, or that holds a malformed line, refuses whatever its other
      lines say, so a mistyped line is never passed over; the refusal is a `SealRefused`, as
      every other check's (a command's "could not run", exit 2). [Alternatives: a line in plan
      section 14 of a further commit naming the entry commit by its full id, which keeps the
      approval in git's history but makes HEAD a commit other than the one approved (the entry
      commit cannot hold its own id); and the sealed routes' gate requiring the approval as
      well, which would also refuse scoring runs made again at an unapproved fix commit that
      open no sealed file.]

`norway_sites` and `asr` open their data set's sealed file through `open_sealed` and hand the
bytes to the parsing modules `load_heldout_norway` and `load_asr`, which read no file; when
`open_sealed` raises, no parser is called. A parser's own error comes after the opening and its
log line, and is raised unchanged: for `load_heldout_norway` and `load_asr` a `LayoutError` for
every file they cannot read (their module texts state how); `load_korea` states its own errors.
A `LayoutError` is logged in the refusal log first (above).
No check can be switched off: there is no test mode and no environment variable; tests pass the
top folder of a made-up repository as `root`.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import fleetreserve
import fleetreserve.load_korea
from fleetreserve import gate_mark, heldout_seal, load_asr, load_heldout_norway
from fleetreserve.sessions import LayoutError

TAG = "frozen-plan"
HASH_LIST = "data-hashes.json"
HELD_OUT_ROLE = "held-out"
LOG_NAME = "heldout-openings.log"
LOG_FIELDS = ("time_utc", "set", "file", "sha256", "commit", "tag_commit", "command")
FIX_FIELD = "fix_of_tag"  # last field of an opening at a parsing fix (check 13, reading H6)
REFUSALS_NAME = "heldout-refusals.log"
REFUSAL_FIELDS = ("time_utc", "set", "file", "sha256", "commit", "tag_commit", "layout")
APPROVALS_NAME = "heldout-fix-approvals.log"  # written by a person only (reading H7)
APPROVAL_FIELDS = ("time_utc", "fix_commit", "words")
APPROVAL_TIME_FORM = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?Z"
)
COMMIT_FORM = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
TIME_PROBLEM = "has a time_utc that is not a UTC time written YYYY-MM-DDThh:mm[:ss[.ffffff]]Z"
NORWAY = ("norway", "Dataset1_charging_reports-held-out.csv")
ASR = ("asr", "202410DatasetEVOfficeParking_v0.csv")
KOREA = ("korea", "ChargingRecords.csv")
PACKAGE = "fleetreserve"
PACKAGE_PATH = "src/fleetreserve"
# Each set's sealed file and the loader module its route hands the bytes to (reading H6(b)).
LOADERS = {
    NORWAY: load_heldout_norway.__name__.rpartition(".")[2],
    ASR: load_asr.__name__.rpartition(".")[2],
    KOREA: fleetreserve.load_korea.__name__.rpartition(".")[2],
}
PLAN = "docs/plan.md"
SECTION_14 = b"## 14. Changes after the tag"
NOT_ALLOWED = f"not a held-out set's loader, a test file of one or {PLAN}"
DIFF = ("diff", "--name-only", "-z", "--no-renames", "--no-relative", "--ignore-submodules=none")
ENTRY = "__main__"
GIT_SETTINGS = (
    "--no-replace-objects",
    "-c",
    "core.fsmonitor=false",
    "-c",
    f"core.excludesFile={os.devnull}",
)
NEW_FILE_MODE = 0o666  # the mode `open` gives a new file, less the umask
SHA256_FORM = re.compile(r"[0-9a-f]{64}")


class SealRefused(PermissionError):
    """No bytes were returned: a check of `open_sealed` failed, or an error stopped it."""


def log_path(root: Path) -> Path:
    """The openings log of a project top folder: `data/heldout-openings.log` (reading H2)."""
    return Path(root) / "data" / LOG_NAME


def refusals_path(root: Path) -> Path:
    """The refusal log of a project top folder: `data/heldout-refusals.log` (reading H6)."""
    return Path(root) / "data" / REFUSALS_NAME


def approvals_path(root: Path) -> Path:
    """The approvals record of a project top folder: `data/heldout-fix-approvals.log` (reading
    H7); read here, never written."""
    return Path(root) / "data" / APPROVALS_NAME


def git_environment() -> dict[str, str]:
    """The environment of every git call: the caller's without any `GIT_` variable, and with no
    global or system settings file."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return environment


def _git(root: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *GIT_SETTINGS, *args],
            capture_output=True,
            text=text,
            check=False,
            env=git_environment(),
        )
    except (OSError, UnicodeDecodeError) as problem:
        raise SealRefused(f"git {' '.join(args)} could not be run: {problem}") from problem


def _git_out(root: Path, *args: str) -> str:
    done = _git(root, *args)
    if done.returncode != 0:
        raise SealRefused(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _check_working_copy(root: Path) -> None:
    """Checks 2 and 3: `root` is the top folder of the repository's main working copy."""
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
        raise SealRefused(f"{root} is not the top folder of a git repository")
    common = _git_out(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    if Path(common).resolve() != (root / ".git").resolve():
        raise SealRefused(
            f"{root} is not the repository's main working copy: git's common folder is not "
            f"{root / '.git'} (reading H4)"
        )


def _tag_commit(root: Path) -> str:
    """Check 4: the commit the tag names, by its exact name in git's listing of the tags."""
    name = f"refs/tags/{TAG}"
    listing = _git(root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/tags/")
    lines = listing.stdout.splitlines() if listing.returncode == 0 else []
    listed = [line.partition(" ")[::2] for line in lines]
    named = [object_id for refname, object_id in listed if refname == name and object_id]
    tag = None
    if named:
        tag = _git(root, "rev-parse", "--verify", "--quiet", f"{named[0]}^{{commit}}")
    if tag is None or tag.returncode != 0:
        cased = sorted(r for r, _ in listed if r != name and r.casefold() == name.casefold())
        note = ""
        if cased:
            note = f" (names that differ only in letter case are not the tag: {', '.join(cased)})"
        raise SealRefused(
            f"no tag {TAG}: sealed files open only at the tag {TAG}, or at a logged parsing fix "
            f"of it{note}"
        )
    return tag.stdout.strip()


def _head_after_tag(root: Path) -> tuple[str, str, bool]:
    """(HEAD, the tag's commit, whether HEAD is a later commit that descends from it), after
    check 4."""
    tag_commit = _tag_commit(root)
    head = _git_out(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if head == tag_commit:
        return head, tag_commit, False
    later = _git(root, "merge-base", "--is-ancestor", tag_commit, head).returncode == 0
    return head, tag_commit, later


def _linked_places(root: Path) -> tuple[Path, ...]:
    """The places under `root` that check 5 reads through, in the order they are looked at: the
    data folder, the sealed folder, the openings log and the refusal log; reads nothing."""
    return (
        log_path(root).parent,
        heldout_seal.folder(root),
        log_path(root),
        refusals_path(root),
    )


def check_no_link(root: Path) -> None:
    """The look for links before check 5 reads a log: raises `SealRefused`, with nothing opened,
    when a place of `_linked_places` holds a ".." part or is a symbolic link, each looked at with
    `lstat` after the folder above it was seen not to be one; a place that is not there passes.
    `open_sealed` makes it, and the sealed routes' gate (`sealed_route.require_gate`), when HEAD
    is a later commit that descends from the tag (module text)."""
    for place in _linked_places(root):
        if ".." in place.parts:
            raise SealRefused(
                f"{place} holds a '..' part: the gate never reads the data folder, the sealed "
                "folder or a log of it by such a path; nothing opened"
            )
        try:
            found = os.lstat(place)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISLNK(found.st_mode):
            raise SealRefused(
                f"{place} is a symbolic link: the gate never follows a link at the data folder, "
                "the sealed folder or a log of it; nothing opened"
            )


def _check_at_tag_or_fix(root: Path) -> tuple[str, str, bool]:
    """Checks 4 and 5 of `open_sealed` and of the sealed routes' gate: (HEAD, tag commit,
    whether HEAD is a parsing fix)."""
    head, tag_commit, later = _head_after_tag(root)
    if head == tag_commit:
        return head, tag_commit, False
    if not later:
        raise SealRefused(
            f"HEAD {head} does not descend from the tag {TAG} ({tag_commit}); sealed files open "
            "only at the tag's commit or at a logged parsing fix after it (readings H1 and H6)"
        )
    reason = _not_a_fix(root, head, tag_commit)
    if reason:
        raise SealRefused(
            f"HEAD {head} is a commit after the tag {TAG} ({tag_commit}) and not a logged "
            f"parsing fix of it: {reason} (reading H6)"
        )
    return head, tag_commit, True


# --- a parsing fix after the tag (reading H6) ------------------------------------------------


def _changed(root: Path, before: str, after: str) -> list[str]:
    """The paths that differ between two commits, in git's order (reading H6(b))."""
    return [path for path in _git_out(root, *DIFF, before, after, "--").split("\0") if path]


def _set_of(path: str) -> tuple[str, str] | None:
    """The set (and sealed file) whose loader or test file `path` is (reading H6(b))."""
    for names, module in LOADERS.items():
        test = rf"tests/(?:[^/]+/)*test_{re.escape(module)}(?:_[^/]*)?\.py"
        if path == f"{PACKAGE_PATH}/{module}.py" or re.fullmatch(test, path):
            return names
    return None


def _plan_parts(root: Path, commit: str) -> tuple[bytes, bytes, bytes] | None:
    """(the plan up to section 14's heading, section 14, the plan after it) at `commit`; None
    when the plan is not there or holds the heading other than once (reading H6(c))."""
    done = _git(root, "cat-file", "blob", f"{commit}:{PLAN}", text=False)
    if done.returncode != 0:
        return None
    lines = done.stdout.splitlines(keepends=True)
    at = [number for number, line in enumerate(lines) if line.rstrip(b"\n") == SECTION_14]
    if len(at) != 1:
        return None
    start = at[0] + 1
    ends = [n for n in range(start, len(lines)) if lines[n].startswith((b"# ", b"## "))]
    end = ends[0] if ends else len(lines)
    return b"".join(lines[:start]), b"".join(lines[start:end]), b"".join(lines[end:])


def _kind(root: Path, parent: str, commit: str) -> tuple[tuple[str, str] | None, str]:
    """(the set `commit` corrects, or None for an entry; why it is neither, or "")."""
    paths = _changed(root, parent, commit)
    if not paths:
        return None, f"commit {commit} changes nothing"
    if PLAN in paths:
        if len(paths) > 1:
            return None, (
                f"commit {commit} changes {PLAN} together with other files; a correction and "
                "its section-14 entry are two commits"
            )
        parts = {}
        for at in (parent, commit):
            parts[at] = _plan_parts(root, at)
            if parts[at] is None:
                return None, (
                    f"{PLAN} at commit {at} does not hold the line '{SECTION_14.decode()}' "
                    "exactly once"
                )
        (before, _, after), (now_before, _, now_after) = parts[parent], parts[commit]
        if (before, after) != (now_before, now_after):
            return None, f"commit {commit} changes {PLAN} outside section 14"
        return None, ""
    sets = []
    for path in paths:
        names = _set_of(path)
        if names is None:
            return None, f"commit {commit} changes {path}, which is {NOT_ALLOWED}"
        sets.append(names)
    if len(set(sets)) > 1:
        named = ", ".join(sorted({names[0] for names in sets}))
        return None, f"commit {commit} changes the files of more than one held-out set ({named})"
    return sets[0], ""


def _not_a_fix(root: Path, head: str, tag_commit: str) -> str:
    """Empty when HEAD, a commit after the tag's, is a logged parsing fix of it (reading H6);
    otherwise the first rule of H6 it breaks."""
    listing = _git_out(
        root, "rev-list", "--parents", "--topo-order", "--reverse", f"{tag_commit}..{head}"
    )
    rows = [line.split() for line in listing.splitlines()]
    for row in rows:
        if len(row) != 2:
            return f"commit {row[0]} between the tag and HEAD has {len(row) - 1} parents, not one"
    chain = [row[0] for row in rows]
    if not chain or [row[1] for row in rows] != [tag_commit, *chain[:-1]] or chain[-1] != head:
        return "the commits between the tag and HEAD are not one line from the tag to HEAD"
    for path in _changed(root, tag_commit, head):
        if path != PLAN and _set_of(path) is None:
            return f"{path} differs from the tag's commit and is {NOT_ALLOWED}"
    corrections = []
    parent, corrected = tag_commit, None
    for number, commit in enumerate(chain):
        corrected, reason = _kind(root, parent, commit)
        if reason:
            return reason
        if corrected is not None:
            corrections.append((number, commit, corrected))
        parent = commit
    if corrected is not None:
        return (
            f"HEAD is a loader correction ({corrected[0]}) without its entry in plan section 14 "
            "after it"
        )
    if not corrections:
        return "no loader correction lies between the tag and HEAD"
    section = _plan_parts(root, head)[1]
    for _, commit, _ in corrections:
        if not re.search(rb"(?<![0-9a-fA-F])" + commit.encode() + rb"(?![0-9a-fA-F])", section):
            return (
                f"plan section 14 at HEAD does not name the loader correction {commit} by its "
                "full id"
            )
    refusals = logged_refusals(root)
    openings = logged_openings(root)
    corrected_at = {number for number, _, _ in corrections}
    previous = {}  # each set's latest correction so far: its place on the line
    for number, commit, names in corrections:
        last = previous.get(names)
        # the commits at which the set's loader was as this correction found it (H6(e))
        since = {tag_commit, *chain[:number]} if last is None else set(chain[last + 1 : number])
        previous[names] = number
        found = [
            refusal
            for refusal in refusals
            if (refusal["set"], refusal["file"]) == names
            and refusal["tag_commit"] == tag_commit
            and refusal["commit"] in since
        ]
        if any(_opened_before(refusal, openings) for refusal in found):
            continue
        # a refusal line in the window with no matching opening before it does not count (H6(e))
        unopened = (
            f" (a refusal line of it there follows no matching opening in data/{LOG_NAME}, so "
            "it does not count)"
            if found
            else ""
        )
        if last is None:
            return (
                f"no layout refusal of the {names[0]} file is logged in data/{REFUSALS_NAME} at "
                f"the tag's commit or at a commit of the fix before the correction {commit}"
                f"{unopened}"
            )
        fixes = [chain[n] for n in range(last + 1, number) if n not in corrected_at]
        if fixes:
            where = (
                f"open the {names[0]} file at the fix commit {' or '.join(fixes)} (the held-out "
                "truth build there), so that its layout refusal is logged there"
            )
            before = chain[number - 1]
            loader = f"{PACKAGE_PATH}/{LOADERS[names]}.py"
            if number - 1 not in corrected_at and loader not in _changed(root, before, commit):
                # a test-only correction after an entry: the loader there may parse the file
                where += f", or leave out the later correction {commit}, which changes tests only"
        else:
            start = chain[last - 1] if last else tag_commit
            where = (
                "no fix commit lies between the two, so none can be logged: start a new line "
                f"from {start} with the corrections as one"
            )
        return (
            f"no layout refusal of the {names[0]} file is logged in data/{REFUSALS_NAME} after "
            f"its previous correction {chain[last]} and before the correction {commit}{unopened}; "
            f"a later correction of a set needs a refusal of its own (reading H6(e)): {where}"
        )
    return ""


OPENING_MATCH = ("set", "file", "sha256", "commit", "tag_commit")


def _opened_before(refusal: dict, openings: list[dict]) -> bool:
    """Whether the openings log holds an opening of the refused bytes that the refusal follows:
    the same set, file, SHA-256, commit and tag commit, at a time not after the refusal's
    (reading H6(e))."""
    return any(
        all(opening[key] == refusal[key] for key in OPENING_MATCH)
        and opening["time_utc"] <= refusal["time_utc"]
        for opening in openings
    )


def _approval_problem(record: dict) -> str:
    """Why a line of the approvals record is malformed in its values, or "" (reading H7)."""
    time, fix, words = (record[name] for name in APPROVAL_FIELDS)
    if not (isinstance(time, str) and APPROVAL_TIME_FORM.fullmatch(time)):
        return TIME_PROBLEM
    try:
        datetime.fromisoformat(time.removesuffix("Z"))
    except ValueError:
        return TIME_PROBLEM
    if not (isinstance(fix, str) and COMMIT_FORM.fullmatch(fix)):
        return "has a fix_commit that is not a full commit id in lower-case hexadecimal"
    if not (isinstance(words, str) and words.strip()):
        return "has no words: the owner's words of the approval, as he gave them"
    return ""


def _check_approval(root: Path, head: str) -> None:
    """Check 8a, at a parsing fix only: every line of the approvals record is well formed and
    one names HEAD's full id (reading H7)."""
    where = f"the approvals record data/{APPROVALS_NAME}"
    after = f"nothing opened at the fix commit {head} (reading H7)"
    try:
        records = logged_openings(root, APPROVALS_NAME)
    except (OSError, UnicodeDecodeError) as error:  # before ValueError, a base of the second
        problem = type(error).__name__
        raise SealRefused(f"{where} cannot be read ({problem}): {error}; {after}") from error
    except ValueError as error:
        raise SealRefused(f"{where} is malformed: {error}; {after}") from None
    for number, record in enumerate(records, start=1):
        problem = _approval_problem(record)
        if problem:
            raise SealRefused(
                f"{where} is malformed: {APPROVALS_NAME} line {number} {problem}; {after}"
            )
    if not any(record["fix_commit"] == head for record in records):
        path = approvals_path(root)
        absent = "" if path.exists() or path.is_symlink() else " (the record is absent)"
        raise SealRefused(
            f"no approval of the fix commit {head} is recorded in data/{APPROVALS_NAME}{absent}: "
            "sealed files open at a parsing fix only after the owner's approval of that exact "
            "commit is written there by hand (reading H7)"
        )


def _check_clean(root: Path) -> None:
    """Checks 6 and 7: nothing uncommitted, and no tracked file hidden from `git status`."""
    changed = _git_out(root, "status", "--porcelain", "--untracked-files=normal")
    if changed.strip():
        raise SealRefused("the git tree has uncommitted changes; sealed files open only clean")
    hidden = [
        entry[2:]
        for entry in _git_out(root, "ls-files", "-v", "-z").split("\0")
        if entry and entry[:2] != "H "
    ]
    if hidden:
        raise SealRefused(
            "tracked files marked assume-unchanged or skip-worktree: " + ", ".join(hidden)
        )


def _of_package(name: object) -> bool:
    return isinstance(name, str) and (name == PACKAGE or name.startswith(PACKAGE + "."))


def _check_code(root: Path, commit: str, whose: str = "the tag") -> None:
    """Check 8: every loaded module of the package, and the entry module as the module text
    states, is the file of `commit`, HEAD (reading H5); `whose` names it in a refusal."""
    package = Path(fleetreserve.__file__).resolve().parent
    top = root.resolve()
    loaded: dict[str, str] = {}
    outside = []
    modules = [(name, module) for name, module in list(sys.modules.items()) if _of_package(name)]
    entry = sys.modules.get(ENTRY)
    if _of_package(getattr(getattr(entry, "__spec__", None), "name", None)):
        modules.append((ENTRY, entry))
    else:
        script = getattr(entry, "__file__", None)
        if script is not None and top in Path(script).resolve().parents:
            path = Path(script).resolve()
            loaded[path.relative_to(top).as_posix()] = str(path)
    for _, module in modules:
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        if package not in path.parents:
            outside.append(str(path))
            continue
        loaded[f"{PACKAGE_PATH}/{path.relative_to(package).as_posix()}"] = str(path)
    if outside:
        raise SealRefused(
            f"the running code is not {whose}'s: modules outside the package folder: "
            + ", ".join(sorted(outside))
        )
    listing = _git_out(root, "ls-tree", "-r", "-z", "--full-tree", commit)
    at_commit = {}
    for entry in listing.split("\0"):
        if entry:
            info, path_at_commit = entry.split("\t", 1)
            at_commit[path_at_commit] = info.split()[2]
    names = sorted(loaded)
    ids = _git_out(root, "hash-object", "--no-filters", "--", *(loaded[n] for n in names)).split()
    differing = [name for name, oid in zip(names, ids, strict=True) if at_commit.get(name) != oid]
    if differing:
        raise SealRefused(
            f"the running code is not {whose}'s: these module files differ from {whose} or are "
            "not at it (reading H5): " + ", ".join(differing)
        )


def _listed_sha256(root: Path, tag_commit: str, path: str) -> str:
    """Check 9: the hash listed for `path` in the hash list at the tag."""
    text = _git_out(root, "show", f"{tag_commit}:{HASH_LIST}")
    try:
        entries = [(f["path"], f["role"], f["sha256"]) for f in json.loads(text)["files"]]
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        problem = type(error).__name__
        raise SealRefused(f"{HASH_LIST} at the tag is ill-formed ({problem})") from None
    matches = [(role, sha256) for listed, role, sha256 in entries if listed == path]
    if len(matches) > 1:
        raise SealRefused(f"{path} is listed {len(matches)} times in {HASH_LIST} at the tag")
    role, sha256 = matches[0] if matches else (None, None)
    if role != HELD_OUT_ROLE:
        raise SealRefused(f"{path} is not listed as held-out in {HASH_LIST} at the tag")
    if not (isinstance(sha256, str) and SHA256_FORM.fullmatch(sha256)):
        raise SealRefused(f"{HASH_LIST} at the tag has an ill-formed hash for {path}")
    return sha256


def _append(path: Path, line: dict) -> None:
    """Check 13's write, and a refusal's: one line appended to the log at `path`, never through
    a symbolic link; a last line without an ending is ended."""
    text = json.dumps(line) + "\n"
    with open(
        path,
        "a+b",
        opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW, NEW_FILE_MODE),
    ) as handle:
        size = handle.seek(0, os.SEEK_END)
        if size:
            handle.seek(size - 1)
            if handle.read(1) != b"\n":
                text = "\n" + text
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def open_sealed(root: Path, set_name: str, file_name: str, logged: dict | None = None) -> bytes:
    """The bytes of `data/heldout/<set_name>/<file_name>` under `root`, if every check passes.
    When `logged` is a dict, the opening's log line is copied into it."""
    try:
        root = Path(root)
        path = heldout_seal.sealed_path(root, set_name, file_name)
        relative = path.relative_to(root).as_posix()
        _check_working_copy(root)
        head, tag_commit, later = _head_after_tag(root)
        if head != tag_commit and later:
            check_no_link(root)
        head, tag_commit, fix = _check_at_tag_or_fix(root)
        _check_clean(root)
        _check_code(root, head, "the fix commit" if fix else "the tag")
        if fix:
            _check_approval(root, head)
        gate_mark.mark_run(root, tag_commit, head if fix else None)
        listed = _listed_sha256(root, tag_commit, relative)
        present = path.is_file()
        inside = heldout_seal.folder(root).resolve() in path.resolve().parents
        if not present:
            raise SealRefused(f"{relative} is listed but not present")
        if not inside:
            raise SealRefused(f"{relative} leads outside the sealed folder through a symbolic link")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise SealRefused(f"{relative} could not be read") from error
        if hashlib.sha256(content).hexdigest() != listed:
            raise SealRefused(
                f"{relative}: SHA-256 differs from {HASH_LIST} at the tag; not parsed"
            )
        when = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        values = (when, set_name, file_name, listed, head, tag_commit, list(sys.argv))
        line = dict(zip(LOG_FIELDS, values, strict=True))
        if fix:
            line[FIX_FIELD] = tag_commit
        try:
            _append(log_path(root), line)
        except OSError as error:
            raise SealRefused(
                f"the openings log could not be written; {relative} not returned"
            ) from error
        if logged is not None:
            logged.update(line)
        return content
    except SealRefused:
        raise
    except Exception as error:
        raise SealRefused(f"refused: {type(error).__name__}: {error}") from error


def logged_openings(root: Path, log: str = LOG_NAME) -> list[dict]:
    """Every opening in the openings log, in order; an absent or empty log gives an empty list.
    With `log=REFUSALS_NAME`, every layout refusal in the refusal log, and with
    `log=APPROVALS_NAME`, every line of the approvals record, in the same way."""
    shapes = None
    if log == LOG_NAME:
        path = log_path(root)
    elif log == REFUSALS_NAME:
        path = refusals_path(root)
    elif log == APPROVALS_NAME:
        path, shapes = approvals_path(root), (APPROVAL_FIELDS,)
    else:
        raise ValueError(f"no log {log!r}")
    if not path.exists() and not path.is_symlink():
        return []
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as file:
        content = file.read()
    return log_lines(content, log, shapes)


def log_lines(content: bytes, log: str = LOG_NAME, shapes: tuple | None = None) -> list[dict]:
    """The lines of the openings log's bytes `content` (with `log=REFUSALS_NAME`, the refusal
    log's), parsed as `logged_openings` parses the file; reads no file. `shapes`, the field
    tuples a line may have, is given by `logged_openings` for the approvals record."""
    if shapes is None:
        if log == LOG_NAME:
            shapes = (LOG_FIELDS, (*LOG_FIELDS, FIX_FIELD))
        elif log == REFUSALS_NAME:
            shapes = (REFUSAL_FIELDS,)
        else:
            raise ValueError(f"no log {log!r}")
    content = content.decode("utf-8")
    lines = []
    for number, text in enumerate(content.splitlines(), start=1):
        try:
            line = json.loads(text)
        except (ValueError, RecursionError):
            raise ValueError(f"{log} line {number} is not a log line") from None
        if not isinstance(line, dict) or tuple(line) not in shapes:
            raise ValueError(f"{log} line {number} does not have the log's fields")
        lines.append(line)
    return lines


def logged_refusals(root: Path) -> list[dict]:
    """Every layout refusal in the refusal log, in order (reading H6)."""
    return logged_openings(root, REFUSALS_NAME)


def _parsed(root: Path, parser, content: bytes, opening: dict):
    """`parser.from_bytes(content)`; a `LayoutError` is logged in the refusal log with the
    opening's fields, then raised unchanged (with a note if the line cannot be written)."""
    try:
        return parser.from_bytes(content)
    except LayoutError as problem:
        when = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        values = (when, *(opening[name] for name in REFUSAL_FIELDS[1:6]), str(problem))
        try:
            _append(refusals_path(root), dict(zip(REFUSAL_FIELDS, values, strict=True)))
        except OSError as error:
            problem.add_note(
                f"the layout refusal could not be logged in data/{REFUSALS_NAME}: "
                f"{type(error).__name__}: {error}"
            )
        raise


def norway_sites(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The held-out Norwegian sites OSL_T and TRO_R: session table and dropped rows."""
    opening: dict = {}
    content = open_sealed(root, *NORWAY, logged=opening)
    return _parsed(root, load_heldout_norway, content, opening)


def asr(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The a.s.r. Utrecht car park: session table and dropped rows."""
    opening: dict = {}
    content = open_sealed(root, *ASR, logged=opening)
    return _parsed(root, load_asr, content, opening)


def korea(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The five Korean portfolios: session table and dropped rows. The sealed file is opened
    through `open_sealed`, as for `asr`, and its bytes go to `load_korea`, which reads no file."""
    opening: dict = {}
    content = open_sealed(root, *KOREA, logged=opening)
    return _parsed(root, fleetreserve.load_korea, content, opening)
