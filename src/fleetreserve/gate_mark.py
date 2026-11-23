"""The sealed-data gate's own mark on the recorded run it passes in.

`mark_run(root, tag_commit, fix_commit)` is called by the gate itself, once its checks of the
commit, the tree and the running code have passed: by `sealed_route.require_gate` after its last
check, and by the held-out loader's opener `open_sealed` after its check 8, before it reads the
hash list or any sealed file; a gate refused at any of those checks writes no mark. No command
calls it. It writes the file `MARK_NAME` into the run folder of the recorded run the command
runs in, when there is one (below), so that `run_evidence.read_evidence` refuses every results
file of that run from an earlier commit (its point 4, reading E3), whether or not the file
itself holds the route's record `sealed_route`, within the limits below. The run recorder
(`run_record`) lists the mark in the run's manifest with its SHA-256, as it lists every file of
the run folder; nothing of what the recorder writes changes. Once the run has ended, deleting
the mark the gate wrote leaves it listed in the manifest.

Limits, named by kind. Like the route's checks (`sealed_route`'s module text), the mark guards
against an honest operator's accidents, not against deliberate evasion. It is not written, or
not kept, in a run of any of these kinds:
1. The run folder or the variable changed while the run goes: a command, or any other process,
   that before the recorder writes the manifest changes the run folder (for example: deletes,
   moves or replaces the mark; writes a file named `manifest.json` into it, after which the gate
   takes the run as finished, step 3 below; puts an entry at the mark's path before the gate
   passes, such as a link, a folder or a file under another spelling the file system takes for
   that name, which the gate takes for its mark and does not write over, and which the recorder
   lists under the mark's name only when it is a file of exactly that name), or that changes or
   removes `FLEETRESERVE_RUN_DIR` in its own environment before the gate passes.
2. A gate passed in a process started without the variable, or with it naming another folder: a
   child process the command starts with such an environment, or a recorded command whose first
   words are `env -u FLEETRESERVE_RUN_DIR`.
3. A recorder started in another working copy or repository than the gate's root: the gate's
   root is the top folder its caller names (`require_gate`'s default is `sessions.TOP`, the one
   the package's code is loaded from), the recorder's is the git top folder it is started in;
   when the two differ, the run folder does not lie in the root's runs folder (step 2 below),
   and the run is recorded at the commit of the working copy the recorder was started in.
4. A name the gate or the mark uses, changed in the command's own process (an assignment to
   `gate_mark.mark_run` or `gate_mark.RUN_DIR_VARIABLE`, for example): the gate's check 8 compares
   the module files with the commit, not the names in memory.
5. Sealed content read by a run that passes no gate itself: a command that reads the sealed files
   without the loader, or reads what a run that passed the gate wrote (its results, the held-out
   tables built from the sealed files). Only the run whose own process passed the gate is marked;
   what a command may read is that command's own check (for the route, `sealed_route`), not the
   mark's.
6. After the run has ended: an edit or a replacement of the manifest; with the mark also deleted,
   nothing of the gate is left.
7. A command run outside the recorder: it has no run folder to mark (below), and nothing it
   writes is read as evidence.
A gate that passes only after the manifest is written (in a process of the command that outlives
the recorder) marks nothing (step 3 below); nothing written after the manifest is listed in it.

No command under `src/` makes a run of kinds 1, 2 or 4, as read there with `git grep`: none
writes to its own environment (the environments built there are the recorder's, which sets the
variable, and git's, the held-out loader's `git_environment`, which drops only variables whose
names start with `GIT_`); the processes started there are git, `sysctl` (`timing_marks`), the
recorder (by `run_driver`, in its own environment), the recorded command (by the recorder, with
the variable set) and worker pools (which inherit the environment); none deletes or moves a file
(the one `os.replace` is the recorder's, of its own manifest); none but the recorder writes a
file named `manifest.json`, and a path a caller names on a command line (an `--out` file) is the
caller's choice, of kind 1; none assigns to a name of another module. Kinds 3 and 7 are in how a
command is started, kind 6 is an edit by hand, and kind 5 is in what a command reads.

Which run folder. The folder named by the variable `FLEETRESERVE_RUN_DIR`
(`run_record.RUN_DIR_VARIABLE`), only when all three hold (reading G2):
1. the variable is set and not empty, and names an existing folder;
2. that folder, with links followed, lies directly in the runs folder `runs.noindex` of `root`
   (links followed), the top folder the gate passed for: the recorder makes every run folder
   there, under the git top folder it is started in, which is `root` unless the recorder was
   started in another working copy or repository (kind 3 above);
3. the folder holds no manifest yet: the recorder writes the manifest only once the command has
   ended, so a run that is going has none unless its command wrote one (kind 1 above), and a
   finished run is never written to.
Otherwise nothing is written and the gate goes on as before: a command run outside the
recorder has no run folder to mark, and what it writes is not evidence anyway, since
`run_verify` verifies only a file listed in the manifest of the run folder that holds it.

What is written. When the mark is already there (a second pass of the gate in the same run:
`gate_again`, the cold start's second gate, a second sealed file opened), nothing: the first
pass's mark stays. Otherwise the file is made new (`O_CREAT | O_EXCL | O_NOFOLLOW`, so a link
or any other entry already at its path is never written through and counts as the mark being
there), holding one JSON object: `gate` "passed", the tag's commit, the fix commit at a logged
parsing fix (absent at the tag) and `TEXT`; and flushed to disk. Its content is for readers:
`read_evidence` reads only whether the mark is in the run folder or listed in the manifest.

If the mark cannot be written (an `OSError`), the gate refuses: `require_gate` and
`open_sealed` raise their `SealRefused` with that error as its cause, as for any other error
while they run, and no sealed file is opened (reading G3).

Readings, each with its alternative in brackets:
  G1. The mark is a file of its own in the run folder, written by the gate, which the recorder
      hashes into the manifest like every other file; `read_evidence` refuses a run from an
      earlier commit when the mark is in its folder or its manifest lists it. [Alternative: a
      field of the manifest itself, written by the recorder from a signal the gate leaves
      (for example a file outside the run folder): a change to what `run_record` writes for
      every run.]
  G2. Only a run folder of the recorder in `root`'s runs folder that has no manifest yet is
      marked. [Alternative: any folder `FLEETRESERVE_RUN_DIR` names, which would also mark
      folders the tests set it to and a finished run's folder.]
  G3. A mark that cannot be written refuses the gate, as an openings-log line that cannot be
      written refuses an opening (the loader's check 13). [Alternative: go on without the mark,
      leaving the run to the route's record in its results alone.]
"""

import json
import os
from pathlib import Path

from fleetreserve.run_record import MANIFEST_NAME, RUN_DIR_VARIABLE, RUNS_FOLDER

MARK_NAME = "sealed-gate-passed.json"
TEXT = (
    "This run passed the sealed-data gate; its results are evidence only at the commit it was "
    "recorded at (run_evidence, reading E3)."
)
NEW_FILE_MODE = 0o666  # the mode `open` gives a new file, less the umask


def run_folder(root: Path) -> Path | None:
    """The run folder the gate marks for `root` (module text, steps 1 to 3), or None."""
    named = os.environ.get(RUN_DIR_VARIABLE)
    if not named:
        return None
    folder = Path(named)
    if not folder.is_dir():
        return None
    folder = folder.resolve()
    if folder.parent != (Path(root).resolve() / RUNS_FOLDER).resolve():
        return None
    if os.path.lexists(folder / MANIFEST_NAME):
        return None
    return folder


def mark_run(root: Path, tag_commit: str, fix_commit: str | None) -> Path | None:
    """Writes the mark into the run folder of `root`'s run going now, when there is one and it
    has no mark yet (module text); the mark's path, or None. Raises `OSError`."""
    folder = run_folder(root)
    if folder is None:
        return None
    path = folder / MARK_NAME
    record = {"gate": "passed", "tag_commit": str(tag_commit)}
    if fix_commit is not None:
        record["fix_commit"] = str(fix_commit)
    record["text"] = TEXT
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, NEW_FILE_MODE)
    except FileExistsError:
        return path
    with os.fdopen(descriptor, "wb") as handle:
        handle.write((json.dumps(record, indent=2) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return path
