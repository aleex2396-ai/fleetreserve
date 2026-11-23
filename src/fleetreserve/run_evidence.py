"""The newest recorded run of a label, read as evidence for a pass-mark item.

`newest_run(top, label)` is the newest folder `runs.noindex/<UTC start>-<label>` (by name, so by
start time). An older run is never used in its place: a newer run that failed or broke off must
not be hidden by an older one that passed.

`read_evidence(run, results_name, head)` reads one results file of that run as JSON and says how
far it can be trusted:

- `verified`: the file passes `run_verify.require_verified` (the run ended with exit 0, was not
  stopped, and the file is the one it wrote). Only such evidence can pass an item.
- `intact`: the run ended by itself with exit code 1 (a check failed) or 2 (a check could not
  run), was not stopped or timed out, and the file is listed in its manifest with the SHA-256 it
  has now. Such a run is not verified, so it can never pass an item; a caller may read from it
  only that an item failed, or why it could not run.

Anything else raises `CannotCheck`: no run, a manifest that cannot be read, a run that hit its
time limit or was interrupted, a file missing, changed or not listed, a file that is not a JSON
object, and a stale run (below).

Which commit. A run whose manifest `commit` is `head` (the commit checked out now) is read as it
is, with `origin` `AT_CHECKED_OUT`. A run recorded at another commit C is read, with `origin`
`FROM_EARLIER`, only when all four of these hold; otherwise it is stale and raises
`CannotCheck` with the first that fails:
1. C is a full commit id (40 or 64 lowercase hexadecimal digits) that git, run in the run folder,
   types as a commit (`git cat-file -t`; the id of an annotated tag is a tag, not a commit), and C
   is an ancestor of `head` (`git merge-base --is-ancestor`).
2. The run's recorded command (the manifest's `command`) is a list of words, none of which names
   an entry of `UNREAD_BY_RUNS` (`names_kept_entry`, reading E1).
3. Every path that differs between the committed trees of C and `head` lies under an entry of
   `UNREAD_BY_RUNS`, and none is a path the caller names in `also_compare`. The paths are those
   `git diff --name-only -z --no-renames --no-relative --ignore-submodules=none C head` lists: a
   file added, deleted, renamed, or changed in content or in mode, and a submodule moved; the
   three options keep the repository's settings for renames, for listing only the folder git runs
   in (`diff.relative`) and for ignoring submodules (`diff.ignoreSubmodules`,
   `submodule.<name>.ignore`) from hiding a path. The message of a stale run names the first path,
   in git's order, that is off the keep-list or named in `also_compare`. The working tree is not
   looked at.
4. The run did not go through the sealed-data gate: its run folder holds no mark of the gate and
   its manifest lists none (the file `gate_mark.MARK_NAME`), and its results file is not a JSON
   object holding the route's record, the key `gate_summary.ROUTE_KEY` (`sealed_route`), with any
   value. The mark is written by the gate itself (`sealed_route.require_gate` and the held-out
   loader's opener, after their checks of the commit, the tree and the code) into the folder of
   the recorded run going in the repository, which the recorder then lists in the run's manifest
   with its SHA-256 (`gate_mark`, readings G1 to G3). So, outside the limits below, every file
   of a recorded run whose command passes the gate is refused here, also a second output file
   the route's record is not in; once the run has ended, deleting the mark the gate wrote leaves
   it listed in the manifest. The limits, named by kind (`gate_mark`'s module text, 'Limits',
   says each in full, with what the commands under `src/` do): kind 1, a command or any other
   process that, before the recorder writes the manifest, changes the run folder (deleting or
   replacing the mark, writing a file named `manifest.json`, putting an entry at the mark's path
   before the gate passes) or changes or removes `FLEETRESERVE_RUN_DIR`; kind 2, a gate passed
   in a process started without the variable, or with it naming another folder; kind 3, a
   recorder started in another working copy or repository than the gate's root; kind 4, a name
   the gate or the mark uses, changed in the command's own process; kind 5, sealed content read
   by a run that passes no gate itself (the sealed files read without the loader, or what a
   marked run wrote); kind 6, an edit or a replacement of the manifest after the run; kind 7, a
   command run outside the recorder (no run folder, and nothing it writes is verified). No
   command under `src/` makes a run of kinds 1, 2 or 4, as read there with `git grep`: none
   writes to its own environment, deletes or moves a file (the recorder's own manifest aside),
   writes a file named `manifest.json` (the recorder alone does) or assigns to a name of another
   module, and the processes started there are git, `sysctl`, the recorder, the recorded command
   (with the variable set) and worker pools. The route's record is kept as a
   second sign, written by the commands. Every
   command that passes the sealed-data gate writes that key (`sealed_route.mark` for `learn_run`,
   `compare_run` and `corner_run`; `drift_run`, `o3_run`, `ledger_run`, the cold start and the
   held-out truth build write it themselves; `gate_run` refuses a held-out run before writing
   any result, and the sealed run's drive writes only its drive log, which `read_evidence` is
   never asked to read: the one tool that reads drive logs reads the rehearsal's, a drive of
   development units that passes no gate). `tests/test_run_evidence_gate_callers.py` checks it
   for the accidents of an honest change; its module text states the rule, what it counts too
   cautiously and what it does not see. In short, it sees every module under `src/`, subfolders
   too, with any top-level name (a function, a class with its methods, a module-level name
   holding a function, a table or a partial, an import, a package's re-export, the module's own
   code) that refers, through any chain of such names in any modules, to the opener
   `open_sealed` or to `sealed_route.require_gate`, and asks each such module to write the
   record (assign `[ROUTE_KEY]`, hold it as a dict key, or refer to `sealed_route.mark`) and
   each of its entries (the module's own code, a function or class nothing else names, a call
   at import) to reach a name of that module that writes it; a helper that writes it does not
   excuse its callers. It does not see: whether the record is written on every path below an
   entry (a record on a dead branch, in a scratch dict or in one branch of a dispatcher
   counts) or with the right content; a function held in a module-level name that nothing
   under `src/` names (only its module is asked); a name built at run time other than the text
   of an `importlib.import_module` or `__import__` call (`globals()`, `sys.modules`, `exec`,
   `getattr` on an object not taken from this package); other import machinery; anything
   outside `src/`; the three modules that define the gate. Such a run is evidence only at the commit
   it was recorded at, whatever differs (the premise, below; reading E3); it comes last so that a
   run stale for one of the reasons above is named by that reason, as before. A caller on the
   sealed route that needs a run of another commit (the tag's runs at a logged parsing fix) names
   the commits it accepts itself. A results file that cannot be read as JSON is not read from an
   earlier commit either: it raises `CannotCheck` below.

The note. `Evidence.note` gives the words a tool prints after the name of a run read from an
earlier commit (empty for a run at `head`). Every line a pass-mark tool under `tools/` writes that
names such a run carries them after the run's name, whatever the line then says (passed, failed,
could not run); `with_note` puts them into a message raised after the run was read. The messages
`read_evidence` itself raises once it has decided to read a run from an earlier commit (its
results file cannot be read, or is not a JSON object) carry them too. Not written by a tool, so
without them: Python's own text of an unexpected error, which can name a file in a run's folder.

The keep-list `UNREAD_BY_RUNS` is `docs/`, `tests/`, `tools/`, `passmarks/`, `check.sh` and
`passmark.sh`, each with its reason beside it below. A change to any other path (under `src/`,
`gate/` or `learn/`, `pyproject.toml`, `uv.lock`, `.python-version`, `data-hashes.json`,
`.gitignore`, a new file or folder at the top) makes every earlier run stale. There is no
exemption per module or per model.

The premise. This is the one statement of it. The rule rests on one premise, which it does not
check: no value a recorded command computes depends on a committed file under an entry of
`UNREAD_BY_RUNS`. For the commands of this package (`python -m fleetreserve.<module>`) that is:
no module under `src/` reads such a file at run time for a value it computes. Where the premise
holds, every committed file a run read at C for its values is byte-identical at `head`. One
decision does depend on such files: whether the sealed-data gate passes at a logged parsing fix
of the tag depends on plan section 14 in `docs/plan.md` and on the names of the test files
changed since the tag (the held-out loader's check 5, reading H6). That is why a run through the
sealed-data gate is outside this rule (point 4 above). A file under an entry that a module under
`src/` comes to read for a value must be taken out of the keep-list by name (none is at this
commit). `tests/test_run_evidence_premise.py` guards the premise against accidents, not as a
proof: every file git tracks under `src/` is a `.py` file; no string, bytes constant (docstrings
left out) or import of those files names an entry (`names_kept_entry`), except those its list
`REVIEWED` names one by one, each with its reason (among them the held-out loader's two strings
read for the gate's decision alone); no path git tracks outside the keep-list is a symbolic
link. Its module text says what it does not see. These checks defend against the accidents of an
honest operator, not against deliberate tricks (the owner's decision 8 of 16 November 2026).

Not compared, for a run at `head` as for one from an earlier commit: files that are not committed
(the data under `data/`, other runs under `runs.noindex/`, an uncommitted edit).

Readings (the alternative in brackets):
  E1. A recorded command can name a place of the keep-list itself, by running a script under
      `tools/` or passing a file under `passmarks/` as an argument; such a run from an earlier
      commit is stale whatever changed. Each word is tested as text, with its letters in lower
      case (a file system that ignores case reads `TOOLS/x.py` as `tools/x.py`). It names a
      folder entry in either of two ways: (a) it contains the entry with its "/" anywhere
      (`tools/x.py`, `python tools/x.py` held in one word, `PYTHONPATH=src:tools/`,
      `--files=a.json,tools/x.json`, `'tools/x.py'`, `@tests/x`, and also `contests/x`, which
      names no place of the list); (b) cut into parts at each "/", "\\" and "=", a part equals
      the entry's name without its "/" or begins with that name and a dot (`tools`, `../tools`,
      `--dir=passmarks`, `PYTHONPATH=tools`, `docs\\notes.md`, `tools.x`). It names a file entry
      when it contains it (`check.sh`, `./passmark.sh`). A word that names a place otherwise
      (`-mtools.x`, `cd tools && python x.py` held in one word, a command that reads a folder of
      the list without naming it) is not seen. [Alternatives: (a) only where the entry starts
      the word or follows a character other than a letter or digit, which reads `contests/x` but
      also `-Itools/x.py`, an option with its value attached; any word holding an entry's name
      anywhere, which also marks words such as "corner tests"; or the premise alone covers the
      commands, and such a run is read when only paths of the keep-list differ.]
  E2. The repository compared is the one holding the run folder (git is run in it), where
      `run_verify` also looks up the recorded commit. [Alternative: the repository of the
      caller's top folder.]
  E3. A reading confirmed on 17 November 2026 (it makes acceptance stricter,
      never easier): the premise is about values, and a run through the sealed-data gate is never
      read from an earlier commit (point 4), since whether the gate passes at a fix depends on
      `docs/plan.md` and on test file names. [Alternatives: (i) `docs/plan.md` and the held-out
      loaders' test files taken out of the keep-list by name, so that any change to them makes
      every earlier run stale; (ii) the premise restated alone, with a run through the gate read
      from an earlier commit like any other run.]
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fleetreserve.gate_mark import MARK_NAME
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.run_record import MANIFEST_NAME, RUNS_FOLDER, sha256_file
from fleetreserve.run_verify import NotVerified, require_verified

VERIFIED = "verified"
INTACT = "intact"
INTACT_EXIT_CODES = (1, 2)
AT_CHECKED_OUT = "at the checked-out commit"
FROM_EARLIER = "from an earlier commit"

# The keep-list: the places no recorded command of this package reads (module text, the
# premise). An entry ending in "/" is a folder and covers every path under it; any other entry is
# one file at the top folder.
UNREAD_BY_RUNS = (
    "docs/",  # the plan and the record of looks at the data: text for readers
    "tests/",  # the tests and their made-up data, run by pytest
    "tools/",  # the pass-mark tools and the repository's checks, which read runs once they ended
    "passmarks/",  # each work package's pass-mark items and the T8 run list, read by the tools
    "check.sh",  # the full check of a change
    "passmark.sh",  # runs the items of one work package's pass mark
)
FULL_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class CannotCheck(Exception):
    """The item cannot be decided from the recorded runs (exit status 2)."""


@dataclass(frozen=True)
class Evidence:
    run: Path
    manifest: dict
    data: dict
    status: str  # VERIFIED or INTACT
    origin: str = AT_CHECKED_OUT  # AT_CHECKED_OUT or FROM_EARLIER

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    @property
    def note(self) -> str:
        """Empty for a run at the checked-out commit; else what a tool prints after its name."""
        if self.origin == AT_CHECKED_OUT:
            return ""
        return earlier_note(self.manifest.get("commit"))


def earlier_note(commit) -> str:
    """The note printed after the name of a run read from the earlier commit `commit`."""
    return (
        f" (accepted from the earlier commit {str(commit)[:12]}: every committed path that "
        "differs between it and the checked-out commit is on the keep-list of run_evidence; "
        "uncommitted files were not compared)"
    )


def with_note(problem: Exception, evidence: Evidence) -> CannotCheck:
    """`CannotCheck` with the message of `problem` and `evidence.note` after the first place the
    message names the run (module text, the note). The message is kept as it is for a run at the
    checked-out commit, when it does not name the run, or when the note is there already."""
    text, name = str(problem), evidence.run.name
    if not evidence.note or name not in text or f"{name}{evidence.note}" in text:
        return CannotCheck(text)
    return CannotCheck(text.replace(name, f"{name}{evidence.note}", 1))


def newest_run(top: Path, label: str) -> Path:
    pattern = re.compile(r"\d{8}T\d{6}Z-" + re.escape(label))
    runs = top / RUNS_FOLDER
    folders = (
        sorted(path for path in runs.iterdir() if path.is_dir() and pattern.fullmatch(path.name))
        if runs.is_dir()
        else []
    )
    if not folders:
        raise CannotCheck(f"no run named <UTC start>-{label} in {RUNS_FOLDER}/")
    return folders[-1]


def head_commit(top: Path) -> str:
    done = subprocess.run(
        ["git", "-C", str(top), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck("git rev-parse HEAD failed")
    return done.stdout.strip()


def _manifest(run: Path) -> dict:
    try:
        manifest = json.loads((run / MANIFEST_NAME).read_text())
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"{MANIFEST_NAME} of {run.name} cannot be read: {problem}") from problem
    if not isinstance(manifest, dict):
        raise CannotCheck(f"{MANIFEST_NAME} of {run.name} is not a JSON object")
    return manifest


def _intact(run: Path, manifest: dict, file: Path) -> bool:
    code = manifest.get("exit_code")
    if isinstance(code, bool) or code not in INTACT_EXIT_CODES:
        return False
    if manifest.get("timed_out", False) or "interrupted" in manifest:
        return False
    outputs = manifest.get("outputs")
    relative = file.relative_to(run).as_posix()
    if not isinstance(outputs, dict) or relative not in outputs or not file.is_file():
        return False
    return sha256_file(file) == outputs[relative]


def _on_keep_list(path: str) -> bool:
    """Whether a path from the top folder lies under an entry of `UNREAD_BY_RUNS`."""
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry for entry in UNREAD_BY_RUNS
    )


def names_kept_entry(text: str) -> str | None:
    """The first entry of `UNREAD_BY_RUNS` that `text` names, else None (reading E1). With its
    letters in lower case, the text names a folder entry when it contains the entry with its "/",
    or when, cut into parts at each "/", "\\" and "=", a part equals the entry's name without its
    "/" or begins with that name and a dot; it names a file entry when it contains it."""
    low = text.casefold()
    parts = re.split(r"[/\\=]", low)
    for entry in UNREAD_BY_RUNS:
        name = entry.rstrip("/")
        if entry.endswith("/"):
            if entry in low or any(part == name or part.startswith(f"{name}.") for part in parts):
                return entry
        elif entry in low:
            return entry
    return None


def _git(run: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(run), *args], capture_output=True, check=False)
    except OSError as problem:
        raise CannotCheck(f"git could not be started: {problem}") from problem


def _through_the_gate(data) -> str:
    """Why a run whose results are `data` is never read from an earlier commit (module text,
    point 4); empty when they do not hold the route's record."""
    if isinstance(data, dict) and ROUTE_KEY in data:
        return (
            f"it went through the sealed-data gate (its results hold the route's record "
            f"{ROUTE_KEY}), and such a run is evidence only at the commit it was recorded at "
            "(reading E3)"
        )
    return ""


def _gate_marked(run: Path, manifest: dict) -> str:
    """Why a run whose folder holds the gate's mark, or whose manifest lists it, is never read
    from an earlier commit (module text, point 4); empty when neither holds."""
    outputs = manifest.get("outputs")
    if (isinstance(outputs, dict) and MARK_NAME in outputs) or os.path.lexists(run / MARK_NAME):
        return (
            f"the sealed-data gate marked its run ({MARK_NAME} in its folder or its manifest), "
            "and such a run is evidence only at the commit it was recorded at (reading E3)"
        )
    return ""


def _not_earlier(run: Path, commit, command, head: str, also_compare) -> str:
    """Empty when the run at `commit` is read from an earlier commit (module text); else why not."""
    if not isinstance(commit, str) or not FULL_COMMIT.fullmatch(commit):
        return f"its recorded commit {commit!r} is not a full commit id"
    if _git(run, "cat-file", "-t", commit).stdout.strip() != b"commit":
        return f"its recorded commit {commit[:12]} is not a commit of the repository"
    ancestor = _git(run, "merge-base", "--is-ancestor", commit, head).returncode
    if ancestor == 1:
        return f"its recorded commit {commit[:12]} is not an ancestor of the checked-out commit"
    if ancestor != 0:
        return "git cannot tell whether its recorded commit is an ancestor of the checked-out one"
    if not isinstance(command, list) or not all(isinstance(word, str) for word in command):
        return "its recorded command is not a list of words"
    for word in command:
        entry = names_kept_entry(word)
        if entry is not None:
            return f"its recorded command names {entry} (the word {word!r}; reading E1)"
    done = _git(
        run,
        "diff",
        "--name-only",
        "-z",
        "--no-renames",
        "--no-relative",
        "--ignore-submodules=none",
        commit,
        head,
        "--",
    )
    if done.returncode != 0:
        return "git cannot compare the two commits"
    for raw in done.stdout.split(b"\0"):
        path = raw.decode("utf-8", "surrogateescape")
        if not path:
            continue
        if not _on_keep_list(path):
            return f"{path} differs between them"
        if path in also_compare:
            return f"{path} differs between them (a path the reading tool compares too)"
    return ""


def read_evidence(
    run: Path, results_name: str, head: str, *, also_compare: tuple[str, ...] = ()
) -> Evidence:
    """The evidence of one results file of `run` (module text). `also_compare` names committed
    paths of the keep-list that must not differ either, for a run from an earlier commit."""
    if not run.is_dir():
        raise CannotCheck(f"the run folder {run.name} does not exist")
    manifest = _manifest(run)
    file = run / results_name
    try:
        require_verified(file)
        status = VERIFIED
    except NotVerified as problem:
        if not _intact(run, manifest, file):
            raise CannotCheck(f"{run.name} is not verified: {problem}") from problem
        status = INTACT
    try:
        data, unreadable = json.loads(file.read_text()), None
    except (OSError, ValueError) as problem:
        data, unreadable = None, problem
    commit = manifest.get("commit")
    origin = AT_CHECKED_OUT
    if commit != head:
        reason = (
            _not_earlier(run, commit, manifest.get("command"), head, also_compare)
            or _through_the_gate(data)
            or _gate_marked(run, manifest)
        )
        if reason:
            raise CannotCheck(
                f"{run.name} is stale: recorded at commit {str(commit)[:12]}, not at the "
                f"checked-out commit {head[:12]}, and {reason}"
            )
        origin = FROM_EARLIER
    named = run.name if origin == AT_CHECKED_OUT else f"{run.name}{earlier_note(commit)}"
    if unreadable is not None:
        raise CannotCheck(f"{results_name} of {named} cannot be read: {unreadable}") from unreadable
    if not isinstance(data, dict):
        raise CannotCheck(f"{results_name} of {named} is not a JSON object")
    return Evidence(run, manifest, data, status, origin)
