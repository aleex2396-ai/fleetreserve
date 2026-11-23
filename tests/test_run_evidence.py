"""`run_evidence.read_evidence` on made-up repositories: a run recorded at the checked-out commit
is read as before; a run recorded at an earlier commit is read only when its commit is a full id
of a commit that is an ancestor of the checked-out one, its recorded command names no place of
the keep-list `UNREAD_BY_RUNS`, and every committed path that differs between the two commits is
on the keep-list (and is none of the paths the caller compares too). Otherwise it is stale and the
message names the reason, the first differing path off the keep-list when there is one. A run
whose results hold the route's record (the key `sealed_route`, with any value) went through the
sealed-data gate and is never read from an earlier commit, whatever differs (point 4, reading
E3); a development run beside it, at the same commits, is read as before.

Hand-worked case. A run of `python -m fleetreserve.command --matrix gate/matrix.json` is recorded
at commit C. Commit H changes only `docs/notes.md`: `git diff --name-only --no-renames C H` lists
`docs/notes.md`, under the entry `docs/`, so the run is read from the earlier commit C. A further
commit that changes `src/fleetreserve/deep.py`, which nothing in the made-up package imports,
lists that path too; it is under no entry, so the run is stale and the message names
`src/fleetreserve/deep.py`: the rule compares paths and follows no import.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve import run_evidence
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.run_evidence import INTACT, VERIFIED, CannotCheck, read_evidence

FILES = {
    "pyproject.toml": '[project]\nname = "made-up"\n',
    "uv.lock": "version = 1\n",
    ".python-version": "3.12\n",
    ".gitignore": "runs.noindex/\n",
    "data-hashes.json": "{}\n",
    "check.sh": "exit 0\n",
    "passmark.sh": "exit 0\n",
    "docs/notes.md": "Notes.\n",
    "tests/test_made_up.py": "def test_nothing():\n    pass\n",
    "tools/made_up_check.py": "VALUE = 1\n",
    "passmarks/wp0.sh": 'check "made up" true\n',
    "passmarks/runs.json": "[]\n",
    "gate/matrix.json": '{"units": []}\n',
    "learn/settings.json": '{"eta": 1}\n',
    "src/fleetreserve/__init__.py": '"""Made-up package."""\n',
    "src/fleetreserve/command.py": "VALUE = 1\n",
    "src/fleetreserve/deep.py": "VALUE = 2\n",
    "src/fleetreserve/models/__init__.py": "",
    "src/fleetreserve/models/m_one.py": "VALUE = 3\n",
    "src/fleetreserve/settings.json": "{}\n",
}
COMMAND = [
    "uv",
    "run",
    "--frozen",
    "python",
    "-m",
    "fleetreserve.command",
    "--matrix",
    "gate/matrix.json",
]
RESULTS_NAME = "results.json"
STAMPS = itertools.count(1)


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def commit_files(repo: Path, files: dict) -> str:
    """Write each file (path to text; None removes it), commit, and return the new commit."""
    for path, text in files.items():
        target = repo / path
        if text is None:
            git(repo, "rm", "-q", "--", path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        git(repo, "add", "--", path)
    git(repo, "commit", "-q", "-m", "change")
    return head(repo)


def recorded(
    repo: Path, *, command=None, exit_code: int = 0, commit=None, drop=(), results=None
) -> Path:
    """A run folder at the repository's commit (or `commit`), its manifest in the recorder's form;
    the manifest fields in `drop` are left out; its results `{"value": 7}`, or `results`."""
    n = next(STAMPS)
    folder = repo / "runs.noindex" / f"20000101T{n:06d}Z-made-up"
    folder.mkdir(parents=True)
    document = {"value": 7} if results is None else results
    results = folder / RESULTS_NAME
    results.write_text(json.dumps(document))
    manifest = {
        "commit": head(repo) if commit is None else commit,
        "command": list(COMMAND if command is None else command),
        "start": "2000-01-01T00:00:00Z",
        "end": "2000-01-01T00:00:01Z",
        "exit_code": exit_code,
        "outputs": {RESULTS_NAME: hashlib.sha256(results.read_bytes()).hexdigest()},
        "long": False,
        "max_seconds": None,
        "timed_out": False,
    }
    for key in drop:
        manifest.pop(key)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder


@pytest.fixture(autouse=True)
def clean_git_environment(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    commit_files(root, FILES)
    return root


def stale_reason(run: Path, later: str, earlier: str) -> str:
    with pytest.raises(CannotCheck) as raised:
        read_evidence(run, RESULTS_NAME, later)
    prefix = (
        f"{run.name} is stale: recorded at commit {earlier[:12]}, not at the checked-out commit "
        f"{later[:12]}, and "
    )
    assert str(raised.value).startswith(prefix)
    return str(raised.value).removeprefix(prefix)


# --- a run at the checked-out commit ---------------------------------------------------------


def test_a_run_at_the_checked_out_commit_is_read_as_before(repo):
    run = recorded(repo, command=["python", "tools/made_up_check.py"])  # reading E1 not applied
    evidence = read_evidence(run, RESULTS_NAME, head(repo))
    assert (evidence.status, evidence.data, evidence.run) == (VERIFIED, {"value": 7}, run)
    assert evidence.origin == run_evidence.AT_CHECKED_OUT and evidence.note == ""


# --- accepted from an earlier commit ---------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "docs/notes.md",
        "tests/test_made_up.py",
        "tools/made_up_check.py",
        "passmarks/wp0.sh",
        "check.sh",
        "passmark.sh",
    ],
)
def test_a_change_under_one_entry_of_the_keep_list_only_is_accepted(repo, path):
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {path: "changed\n"})
    evidence = read_evidence(run, RESULTS_NAME, later)
    assert (evidence.status, evidence.data) == (VERIFIED, {"value": 7})
    assert evidence.origin == run_evidence.FROM_EARLIER
    assert evidence.note == (
        f" (accepted from the earlier commit {earlier[:12]}: every committed path that differs "
        "between it and the checked-out commit is on the keep-list of run_evidence; uncommitted "
        "files were not compared)"
    )


def test_files_added_deleted_and_moved_inside_the_keep_list_are_accepted(repo):
    run = recorded(repo)
    commit_files(repo, {"docs/new.md": "new\n", "tests/test_made_up.py": None})
    git(repo, "mv", "tools/made_up_check.py", "passmarks/moved.py")
    git(repo, "commit", "-q", "-m", "move")
    evidence = read_evidence(run, RESULTS_NAME, head(repo))
    assert evidence.origin == run_evidence.FROM_EARLIER


def test_an_intact_run_from_an_earlier_commit_is_read_as_intact(repo):
    run = recorded(repo, exit_code=1)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    evidence = read_evidence(run, RESULTS_NAME, later)
    assert evidence.status == INTACT and evidence.origin == run_evidence.FROM_EARLIER


# --- a run through the sealed-data gate (point 4, reading E3) ------------------------------------

THROUGH_THE_GATE = (
    f"it went through the sealed-data gate (its results hold the route's record {ROUTE_KEY}), and "
    "such a run is evidence only at the commit it was recorded at (reading E3)"
)
ROUTE_RESULTS = {"value": 7, ROUTE_KEY: {"kind": "held-out", "tag_commit": "0" * 40}}


@pytest.mark.parametrize("path", ["docs/plan.md", "docs/notes.md", "tests/test_made_up.py"])
def test_a_run_through_the_gate_from_an_earlier_commit_is_stale_where_a_development_run_is_read(
    repo, path
):
    earlier = head(repo)
    route = recorded(repo, results=ROUTE_RESULTS)
    development = recorded(repo)
    later = commit_files(repo, {path: "changed\n"})
    assert stale_reason(route, later, earlier) == THROUGH_THE_GATE
    assert read_evidence(development, RESULTS_NAME, later).origin == run_evidence.FROM_EARLIER


@pytest.mark.parametrize("record", [{"kind": "held-out"}, {}, None, "made up"])
def test_the_routes_key_with_any_value_marks_a_run_through_the_gate(repo, record):
    earlier = head(repo)
    run = recorded(repo, results={"value": 7, ROUTE_KEY: record})
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert stale_reason(run, later, earlier) == THROUGH_THE_GATE


def test_a_run_through_the_gate_stale_for_an_earlier_reason_is_named_by_it_as_before(repo):
    earlier = head(repo)
    run = recorded(repo, results=ROUTE_RESULTS)
    later = commit_files(repo, {"src/fleetreserve/deep.py": "VALUE = 3\n"})
    assert stale_reason(run, later, earlier) == "src/fleetreserve/deep.py differs between them"


def test_an_intact_run_through_the_gate_from_an_earlier_commit_is_stale(repo):
    earlier = head(repo)
    run = recorded(repo, exit_code=1, results=ROUTE_RESULTS)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert stale_reason(run, later, earlier) == THROUGH_THE_GATE


def test_a_run_through_the_gate_at_the_checked_out_commit_is_read_as_before(repo):
    run = recorded(repo, results=ROUTE_RESULTS)
    evidence = read_evidence(run, RESULTS_NAME, head(repo))
    assert (evidence.status, evidence.data) == (VERIFIED, ROUTE_RESULTS)
    assert evidence.origin == run_evidence.AT_CHECKED_OUT


def test_the_rule_compares_commits_not_the_working_tree(repo):
    run = recorded(repo)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    (repo / "src/fleetreserve/deep.py").write_text("VALUE = 20\n")  # not committed
    assert read_evidence(run, RESULTS_NAME, later).origin == run_evidence.FROM_EARLIER


# --- stale: a path off the keep-list differs ---------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/fleetreserve/command.py",  # the module the command runs
        "src/fleetreserve/deep.py",  # a module nothing imports
        "src/fleetreserve/models/m_one.py",  # a model
        "src/fleetreserve/settings.json",  # a file under src/ that is not Python
        "gate/matrix.json",
        "learn/settings.json",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "data-hashes.json",
        ".gitignore",
    ],
)
def test_a_change_off_the_keep_list_is_stale_and_named(repo, path):
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {"docs/notes.md": "changed\n", path: "changed\n"})
    assert stale_reason(run, later, earlier) == f"{path} differs between them"


@pytest.mark.parametrize("path", ["new.txt", "uv.toml", "tools.py", "docs.md", "testsx/a.py"])
def test_a_new_file_at_the_top_is_stale_whatever_its_name_begins_with(repo, path):
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {path: "new\n"})
    assert stale_reason(run, later, earlier) == f"{path} differs between them"


def test_a_deleted_file_is_stale(repo):
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {"src/fleetreserve/deep.py": None})
    assert stale_reason(run, later, earlier) == "src/fleetreserve/deep.py differs between them"


def test_a_file_renamed_into_or_out_of_the_keep_list_is_stale(repo):
    earlier = head(repo)
    run = recorded(repo)
    git(repo, "mv", "src/fleetreserve/deep.py", "docs/deep.py")
    git(repo, "commit", "-q", "-m", "move into docs")
    assert stale_reason(run, head(repo), earlier) == "src/fleetreserve/deep.py differs between them"
    run = recorded(repo)
    earlier = head(repo)
    git(repo, "mv", "docs/notes.md", "notes.md")
    git(repo, "commit", "-q", "-m", "move out of docs")
    assert stale_reason(run, head(repo), earlier) == "notes.md differs between them"


def test_a_change_of_mode_alone_is_stale(repo):
    earlier = head(repo)
    run = recorded(repo)
    git(repo, "update-index", "--chmod=+x", "src/fleetreserve/deep.py")
    git(repo, "commit", "-q", "-m", "mode")
    assert git(repo, "diff", "--raw", earlier, "HEAD").startswith(":100644 100755 ")
    assert stale_reason(run, head(repo), earlier) == "src/fleetreserve/deep.py differs between them"


def test_the_setting_to_list_only_the_folder_git_runs_in_hides_no_path(repo):
    """`diff.relative` would list only the run folder's own paths, none of them committed."""
    git(repo, "config", "diff.relative", "true")
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {"src/fleetreserve/deep.py": "VALUE = 20\n"})
    assert stale_reason(run, later, earlier) == "src/fleetreserve/deep.py differs between them"


def test_the_settings_to_ignore_submodules_hide_no_moved_submodule(repo):
    """A submodule entry under src/ moved from one commit to another, with the repository set to
    ignore every submodule."""
    first, second = head(repo), commit_files(repo, {"docs/notes.md": "second\n"})
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},src/fleetreserve/vendor")
    git(repo, "commit", "-q", "-m", "a submodule")
    earlier = head(repo)
    run = recorded(repo)
    git(repo, "update-index", "--cacheinfo", f"160000,{second},src/fleetreserve/vendor")
    git(repo, "commit", "-q", "-m", "the submodule moved")
    git(repo, "config", "diff.ignoreSubmodules", "all")
    git(repo, "config", "submodule.src/fleetreserve/vendor.ignore", "all")
    assert git(repo, "diff", "--name-only", earlier, "HEAD") == ""  # what the settings hide
    assert stale_reason(run, head(repo), earlier) == "src/fleetreserve/vendor differs between them"


def test_a_message_raised_after_the_run_is_read_from_an_earlier_commit_carries_the_note(repo):
    earlier = head(repo)
    run = recorded(repo)
    (run / RESULTS_NAME).write_text("[7]")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["outputs"][RESULTS_NAME] = hashlib.sha256(b"[7]").hexdigest()
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(CannotCheck) as raised:
        read_evidence(run, RESULTS_NAME, earlier)
    assert str(raised.value) == f"{RESULTS_NAME} of {run.name} is not a JSON object"
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    with pytest.raises(CannotCheck) as raised:
        read_evidence(run, RESULTS_NAME, later)
    assert str(raised.value) == (
        f"{RESULTS_NAME} of {run.name}{run_evidence.earlier_note(earlier)} is not a JSON object"
    )


def test_with_note_puts_the_note_after_the_first_place_the_message_names_the_run(repo):
    earlier = head(repo)
    run = recorded(repo)
    at_head = read_evidence(run, RESULTS_NAME, earlier)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    from_earlier = read_evidence(run, RESULTS_NAME, later)
    note = run_evidence.earlier_note(earlier)
    problem = CannotCheck(f"{run.name} ran on another matrix; {run.name} again")
    assert str(run_evidence.with_note(problem, from_earlier)) == (
        f"{run.name}{note} ran on another matrix; {run.name} again"
    )
    assert str(run_evidence.with_note(problem, at_head)) == str(problem)
    once = run_evidence.with_note(problem, from_earlier)
    assert str(run_evidence.with_note(once, from_earlier)) == str(once)
    other = CannotCheck("the run has 2 solves per programme, not 3")
    assert str(run_evidence.with_note(other, from_earlier)) == str(other)


def test_a_path_the_reading_tool_compares_too_is_stale_when_it_differs(repo):
    earlier = head(repo)
    run = recorded(repo)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    also = ("passmarks/runs.json",)
    assert read_evidence(run, RESULTS_NAME, later, also_compare=also).origin == (
        run_evidence.FROM_EARLIER
    )
    later = commit_files(repo, {"passmarks/runs.json": '["made-up"]\n'})
    assert read_evidence(run, RESULTS_NAME, later).origin == run_evidence.FROM_EARLIER
    with pytest.raises(CannotCheck) as raised:
        read_evidence(run, RESULTS_NAME, later, also_compare=also)
    assert str(raised.value) == (
        f"{run.name} is stale: recorded at commit {earlier[:12]}, not at the checked-out commit "
        f"{later[:12]}, and passmarks/runs.json differs between them (a path the reading tool "
        "compares too)"
    )


# --- stale: the command names the keep-list (reading E1) ------------------------------------


@pytest.mark.parametrize(
    ("command", "entry", "word"),
    [
        (
            ["uv", "run", "--frozen", "python", "tools/made_up_check.py"],
            "tools/",
            "tools/made_up_check.py",
        ),
        (
            ["uv", "run", "--frozen", "python", "-m", "tools.made_up_check"],
            "tools/",
            "tools.made_up_check",
        ),
        ([*COMMAND, "--list", "passmarks/runs.json"], "passmarks/", "passmarks/runs.json"),
        ([*COMMAND, "--notes=docs/notes.md"], "docs/", "--notes=docs/notes.md"),
        ([*COMMAND, "--folder", "tests"], "tests/", "tests"),
        (["bash", "check.sh"], "check.sh", "check.sh"),
        (["python", "TOOLS/made_up_check.py"], "tools/", "TOOLS/made_up_check.py"),
        ([*COMMAND, "--out", "../tests"], "tests/", "../tests"),
        ([*COMMAND, "--dir=passmarks"], "passmarks/", "--dir=passmarks"),
        (["env", "PYTHONPATH=tools", *COMMAND], "tools/", "PYTHONPATH=tools"),
        (["bash", "./Passmark.sh"], "passmark.sh", "./Passmark.sh"),
        ([*COMMAND, "--notes", "docs\\notes.md"], "docs/", "docs\\notes.md"),
    ],
)
def test_a_command_that_names_a_place_of_the_keep_list_is_stale(repo, command, entry, word):
    earlier = head(repo)
    run = recorded(repo, command=command)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert stale_reason(run, later, earlier) == (
        f"its recorded command names {entry} (the word {word!r}; reading E1)"
    )


def test_a_command_word_that_only_holds_a_name_of_the_keep_list_names_no_place(repo):
    run = recorded(repo, command=[*COMMAND, '{"note": "two-way corner tests"}', "toolset"])
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert read_evidence(run, RESULTS_NAME, later).origin == run_evidence.FROM_EARLIER


@pytest.mark.parametrize(
    ("command", "entry", "word"),
    [
        # a folder entry with its "/" anywhere in the word
        (["sh", "-c", "python tools/x.py"], "tools/", "python tools/x.py"),
        (["env", "PYTHONPATH=src:tools/", *COMMAND], "tools/", "PYTHONPATH=src:tools/"),
        ([*COMMAND, "--files=a.json,tools/x.json"], "tools/", "--files=a.json,tools/x.json"),
        ([*COMMAND, "'tools/x.py'"], "tools/", "'tools/x.py'"),
        ([*COMMAND, "--out", "@tests/x"], "tests/", "@tests/x"),
        ([*COMMAND, "--out", "contests/x"], "tests/", "contests/x"),
        # a part, cut at "/", "\" and "=", equal to a folder entry's name; letters in lower case
        ([*COMMAND, "--dir=passmarks"], "passmarks/", "--dir=passmarks"),
        ([*COMMAND, "TOOLS/x.py"], "tools/", "TOOLS/x.py"),
        ([*COMMAND, "--out", "../tools"], "tools/", "../tools"),
    ],
    ids=[
        "a shell line in one word",
        "a path list",
        "a comma list",
        "quotes inside the word",
        "an at sign",
        "contests",
        "an option with =",
        "upper case",
        "a parent folder",
    ],
)
def test_each_form_of_a_command_word_naming_a_folder_of_the_keep_list_is_stale(
    repo, command, entry, word
):
    """Each form is refused (reading E1): the word contains the folder with its "/" (`contests/x`
    too, which names no place of the list), or a part of it equals the folder's name."""
    earlier = head(repo)
    run = recorded(repo, command=command)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert stale_reason(run, later, earlier) == (
        f"its recorded command names {entry} (the word {word!r}; reading E1)"
    )


EVERY_ENTRY = {
    "docs/notes.md": "later\n",
    "tests/test_made_up.py": "x = 2\n",
    "tools/made_up_check.py": "VALUE = 2\n",
    "passmarks/wp0.sh": "true\n",
    "check.sh": "exit 1\n",
    "passmark.sh": "exit 1\n",
}
PY = ["uv", "run", "--frozen", "python"]


@pytest.mark.parametrize(
    ("changed", "command", "recorded_as", "reason"),
    [
        ({"docs/notes.md": "later\n"}, COMMAND, "earlier", None),
        ({"tests/test_made_up.py": "x = 2\n"}, COMMAND, "earlier", None),
        ({"tools/made_up_check.py": "VALUE = 2\n"}, COMMAND, "earlier", None),
        (EVERY_ENTRY, COMMAND, "earlier", None),
        (
            {"src/fleetreserve/command.py": "VALUE = 2\n"},
            COMMAND,
            "earlier",
            "src/fleetreserve/command.py differs between them",
        ),
        (
            {"docs/notes.md": "later\n"},
            COMMAND,
            "side",
            "its recorded commit {commit} is not an ancestor of the checked-out commit",
        ),
        ({"docs/notes.md": "later\n"}, COMMAND, "short", "not verified"),
        (
            {"docs/notes.md": "later\n"},
            [*PY, "tools/x.py"],
            "earlier",
            "its recorded command names tools/ (the word 'tools/x.py'; reading E1)",
        ),
        (
            {"docs/notes.md": "later\n"},
            [*PY, "-m", "tools.x"],
            "earlier",
            "its recorded command names tools/ (the word 'tools.x'; reading E1)",
        ),
        (
            {"docs/notes.md": "later\n"},
            [*COMMAND, "--list=passmarks/x.json"],
            "earlier",
            "its recorded command names passmarks/ (the word '--list=passmarks/x.json'; "
            "reading E1)",
        ),
        (
            {"docs/notes.md": "later\n"},
            ["bash", "check.sh"],
            "earlier",
            "its recorded command names check.sh (the word 'check.sh'; reading E1)",
        ),
    ],
    ids=[
        "docs only",
        "tests only",
        "tools only",
        "every entry",
        "a src change",
        "a sibling branch",
        "a short id",
        "tools/x.py",
        "-m tools.x",
        "--list=passmarks/x.json",
        "check.sh",
    ],
)
def test_runs_decided_alike_by_every_form_of_reading_e1(
    repo, changed, command, recorded_as, reason
):
    """Eleven cases that every form of reading E1 decides alike: read from the earlier commit when
    only keep-list paths differ and the command names none, else stale or not verified."""
    commit = head(repo)
    if recorded_as == "side":
        git(repo, "switch", "-q", "-c", "side")
        commit = commit_files(repo, {"docs/notes.md": "side\n"})
        git(repo, "switch", "-q", "-")
    elif recorded_as == "short":
        commit = commit[:12]
    run = recorded(repo, command=command, commit=commit)
    later = commit_files(repo, changed)
    if reason is None:
        evidence = read_evidence(run, RESULTS_NAME, later)
        assert evidence.origin == run_evidence.FROM_EARLIER
        assert evidence.note == run_evidence.earlier_note(commit)
    elif reason == "not verified":
        with pytest.raises(CannotCheck) as raised:
            read_evidence(run, RESULTS_NAME, later)
        assert str(raised.value) == (
            f"{run.name} is not verified: the recorded commit is not a full hexadecimal SHA"
        )
    else:
        assert stale_reason(run, later, commit) == reason.format(commit=commit[:12])


# --- stale: the commit -------------------------------------------------------------------------


def test_a_commit_that_is_not_an_ancestor_is_stale(repo):
    base = head(repo)
    git(repo, "switch", "-q", "-c", "side")
    side = commit_files(repo, {"docs/notes.md": "side\n"})
    run = recorded(repo)
    git(repo, "switch", "-q", "-")
    later = commit_files(repo, {"docs/notes.md": "main\n"})
    assert base != later
    assert stale_reason(run, later, side) == (
        f"its recorded commit {side[:12]} is not an ancestor of the checked-out commit"
    )


def test_the_id_of_an_annotated_tag_is_stale(repo):
    git(repo, "tag", "-a", "made-up-tag", "-m", "a tag")
    tag = git(repo, "rev-parse", "made-up-tag")
    assert tag != head(repo)
    run = recorded(repo, commit=tag)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    assert stale_reason(run, later, tag) == (
        f"its recorded commit {tag[:12]} is not a commit of the repository"
    )


@pytest.mark.parametrize("how", ["short", "upper", "unknown"])
def test_a_short_malformed_or_unknown_commit_is_stale(repo, how):
    commit = {"short": head(repo)[:12], "upper": head(repo).upper(), "unknown": "b" * 40}[how]
    run = recorded(repo, commit=commit, exit_code=1)  # intact: the verifier would refuse it
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    reason = stale_reason(run, later, commit)
    if how == "unknown":
        assert reason == f"its recorded commit {commit[:12]} is not a commit of the repository"
    else:
        assert reason == f"its recorded commit {commit!r} is not a full commit id"


def test_a_checked_out_commit_git_does_not_know_is_stale(repo):
    run = recorded(repo)
    with pytest.raises(CannotCheck, match="git cannot tell whether its recorded commit is an"):
        read_evidence(run, RESULTS_NAME, "c" * 40)


def test_a_manifest_without_a_commit_cannot_be_checked_as_before(repo):
    verified_shape = recorded(repo, drop=("commit",))
    with pytest.raises(CannotCheck, match="is not verified: manifest.json lacks: commit"):
        read_evidence(verified_shape, RESULTS_NAME, head(repo))
    intact_shape = recorded(repo, exit_code=1, drop=("commit",))
    with pytest.raises(CannotCheck) as raised:
        read_evidence(intact_shape, RESULTS_NAME, head(repo))
    assert str(raised.value) == (
        f"{intact_shape.name} is stale: recorded at commit None, not at the checked-out commit "
        f"{head(repo)[:12]}, and its recorded commit None is not a full commit id"
    )


def test_a_run_folder_outside_the_repository_is_stale(tmp_path, repo):
    run = recorded(repo, exit_code=1)
    earlier = head(repo)
    later = commit_files(repo, {"docs/notes.md": "changed\n"})
    moved = tmp_path / "elsewhere" / run.name
    moved.parent.mkdir()
    run.rename(moved)
    assert stale_reason(moved, later, earlier) == (
        f"its recorded commit {earlier[:12]} is not a commit of the repository"
    )
