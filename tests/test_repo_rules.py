"""Repository rules: what git tracks must follow the project's rules (tools/repo_rules.py).

Every test works in a throwaway git repository inside pytest's temporary folder, with git's own
configuration switched off. Strings that look like a user path or a live key are built in pieces
so that this file does not trip the rules it tests.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
from git_run import run_git

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "repo_rules.py"

_spec = importlib.util.spec_from_file_location("repo_rules", SCRIPT)
repo_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repo_rules)

EXPECTED_RULE_NAMES = [
    "tracked-private",
    "large-file",
    "absolute-path",
    "secret-text",
    "solver-door",
    "heldout-seal",
]

# Pieces of strings that would trip the rules if written out whole.
SLASH = "/"
USERS = SLASH + "Users"
HOME = SLASH + "home"
BACKSLASH = "\\"
ANT_PREFIX = "sk-ant" + "-"
PEM_OPEN = "-----BEGIN" + " "
PEM_CLOSE = "-----"
IDENTITY = ("-c", "user.name=test", "-c", "user.email=test@example.invalid")


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


def write(root: Path, files: dict[str, str | bytes]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)


def make_repo(
    tmp_path: Path,
    committed: dict[str, str | bytes] | None = None,
    staged: dict[str, str | bytes] | None = None,
    untracked: dict[str, str | bytes] | None = None,
    exclude: str | None = None,
) -> Path:
    """A repository holding `committed` files, with `staged` files added but not committed.

    `exclude`, when given, is the text of the repository's own `.git/info/exclude` file.
    """
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "--template=")
    if exclude is not None:
        (root / ".git" / "info").mkdir()
        (root / ".git" / "info" / "exclude").write_text(exclude)
    write(root, {"README.txt": "plain\n", **(committed or {})})
    git(root, "add", "-f", "--", *["README.txt", *(committed or {})])
    git(root, *IDENTITY, "commit", "-q", "-m", "first")
    if staged:
        write(root, staged)
        git(root, "add", "-f", "--", *staged)
    if untracked:
        write(root, untracked)
    return root


def run_script(root: Path, *extra: str, env: dict[str, str] | None = None) -> tuple[int, list[str]]:
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(root), *extra],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )
    return done.returncode, done.stdout.splitlines()


def violations(tmp_path: Path, committed: dict[str, str | bytes]) -> list[str]:
    status, lines = run_script(make_repo(tmp_path, committed))
    assert status == 1, lines
    return lines


def passes(tmp_path: Path, committed: dict[str, str | bytes]) -> None:
    status, lines = run_script(make_repo(tmp_path, committed))
    assert status == 0, lines
    assert len(lines) == 1


def each_reported_once(tmp_path: Path, cases: list[tuple[str, str | bytes, str, str]]) -> None:
    """Commit every case as its own file; each must give exactly one line, as expected.

    A case is (path, content, expected start of the line, text the line must contain).
    """
    root = make_repo(tmp_path, {path: content for path, content, _, _ in cases})
    status, lines = run_script(root)
    assert status == 1, lines
    by_path: dict[str, list[str]] = {}
    for line in lines:
        by_path.setdefault(line.split(":")[0], []).append(line)
    assert sorted(by_path) == sorted(path for path, *_ in cases), lines
    for path, content, start, needle in cases:
        assert len(by_path[path]) == 1, (content, by_path[path])
        assert by_path[path][0].startswith(start), (content, by_path[path])
        assert needle in by_path[path][0], (content, by_path[path])


def none_reported(tmp_path: Path, cases: list[tuple[str, str | bytes]]) -> None:
    """Commit every case as its own file; the run must pass and say which case failed if not."""
    root = make_repo(tmp_path, dict(cases))
    status, lines = run_script(root)
    assert status == 0, (lines, [content for path, content in cases if path in "".join(lines)])
    assert len(lines) == 1


def test_rule_names_are_exactly_the_expected_list():
    assert [rule.name for rule in repo_rules.RULES] == EXPECTED_RULE_NAMES


def test_constants_name_the_expected_files():
    assert repo_rules.SOLVER_DOOR_FILES == ["src/fleetreserve/lp.py", "tools/repo_rules.py"]
    assert repo_rules.HELDOUT_FILES == [
        "src/fleetreserve/heldout_loader.py",
        "src/fleetreserve/heldout_seal.py",
        "src/fleetreserve/heldout_truth_build.py",
        "src/fleetreserve/sealed_route.py",
        "tools/data_hashes.py",
        "tools/fetch_data.py",
        "tools/norway_split.py",
        "tools/repo_rules.py",
    ]
    assert repo_rules.LARGE_FILE_BYTES == 1_000_000


# --- whole-run behaviour -------------------------------------------------------------------------


def test_clean_repository_passes_with_one_line(tmp_path):
    passes(tmp_path, {"src/a.py": "x = 1\n", "docs/notes.md": "text\n"})


def test_staged_but_uncommitted_violation_is_caught(tmp_path):
    root = make_repo(tmp_path, staged={"data/new.csv": "1,2\n"})
    status, lines = run_script(root)
    assert status == 1
    assert [line.split(": ")[:2] for line in lines] == [["data/new.csv:-", "tracked-private"]]


def test_untracked_violation_is_not_reported(tmp_path):
    root = make_repo(
        tmp_path,
        untracked={"data/new.csv": "1,2\n", "notes.txt": "cd " + USERS + "/someone/x\n"},
    )
    status, lines = run_script(root)
    assert status == 0, lines


def test_ignored_untracked_violation_is_not_reported(tmp_path):
    root = make_repo(tmp_path, committed={".gitignore": "data/\n"}, untracked={"data/a.csv": "1\n"})
    status, lines = run_script(root)
    assert status == 0, lines


def test_folder_that_is_not_a_git_repository_exits_2(tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "a.txt").write_text("x\n")
    status, lines = run_script(folder)
    assert status == 2
    assert len(lines) == 1


def test_folder_inside_a_repository_but_not_its_top_exits_2(tmp_path):
    root = make_repo(tmp_path, committed={"sub/a.txt": "x\n"})
    status, _ = run_script(root / "sub")
    assert status == 2


def test_missing_folder_exits_2(tmp_path):
    status, _ = run_script(tmp_path / "nowhere")
    assert status == 2


def test_missing_git_exits_2(tmp_path):
    root = make_repo(tmp_path)
    empty = tmp_path / "empty-path"
    empty.mkdir()
    status, lines = run_script(root, env={"PATH": str(empty)})
    assert status == 2
    assert len(lines) == 1


def test_extra_arguments_exit_2(tmp_path):
    root = make_repo(tmp_path)
    status, _ = run_script(root, "extra")
    assert status == 2


def test_several_violations_are_all_listed_in_order(tmp_path):
    lines = violations(
        tmp_path,
        {
            "reference/old.py": "x = 1\n",
            "src/fleetreserve/other.py": "x = 1\nfrom scipy.optimize import linprog\n",
            "notes.txt": "one\ncd " + USERS + "/someone/work\n",
            "keys.txt": "k = " + ANT_PREFIX + "a" * 30 + "\n",
            "src/fleetreserve/seal.py": "p = 'data/heldout/x'\n",
        },
    )
    assert [line.split(": ")[:2] for line in lines] == [
        ["keys.txt:1", "secret-text"],
        ["notes.txt:2", "absolute-path"],
        ["reference/old.py:-", "tracked-private"],
        ["src/fleetreserve/other.py:2", "solver-door"],
        ["src/fleetreserve/seal.py:1", "heldout-seal"],
    ]


def test_two_findings_on_one_file_are_both_listed(tmp_path):
    body = "a\n" + USERS + "/x/y\n" + "b\n" + HOME + "/z/w\n"
    lines = violations(tmp_path, {"notes.txt": body})
    assert [line.split(": ")[:2] for line in lines] == [
        ["notes.txt:2", "absolute-path"],
        ["notes.txt:4", "absolute-path"],
    ]


def test_binary_files_are_skipped_by_text_rules(tmp_path):
    path_bytes = (USERS + "/someone/x " + SLASH + "tmp/y").encode()
    key_bytes = (ANT_PREFIX + "a" * 30).encode()
    passes(
        tmp_path,
        {
            "blob_nul.bin": b"\x00" + path_bytes + key_bytes,
            "src/fleetreserve/blob.dat": b"\x00heldout linprog",
        },
    )


def test_nul_after_the_first_8192_bytes_is_still_text(tmp_path):
    body = b"a" * 8192 + b"\x00\n" + (USERS + "/someone/x\n").encode()
    lines = violations(tmp_path, {"late_nul.txt": body})
    assert len(lines) == 1
    assert lines[0].startswith("late_nul.txt:2: absolute-path: ")


def test_a_latin1_file_is_text_and_its_path_is_found(tmp_path):
    body = ("caf\xe9\ncaf\xe9 " + USERS + "/someone/x\n").encode("latin-1")
    assert b"\xe9" in body
    lines = violations(tmp_path, {"latin1.txt": body})
    assert len(lines) == 1
    assert lines[0].startswith("latin1.txt:2: absolute-path: ")


def test_a_latin1_file_with_a_key_is_found(tmp_path):
    body = ("caf\xe9 " + ANT_PREFIX + "a" * 30 + "\n").encode("latin-1")
    lines = violations(tmp_path, {"latin1.txt": body})
    assert lines[0].startswith("latin1.txt:1: secret-text: ")


def test_binary_files_still_get_path_and_size_rules(tmp_path):
    lines = violations(
        tmp_path,
        {"data/x.bin": b"\x00\x01", "big.bin": b"\x00" * (repo_rules.LARGE_FILE_BYTES + 1)},
    )
    assert [line.split(": ")[:2] for line in lines] == [
        ["big.bin:-", "large-file"],
        ["data/x.bin:-", "tracked-private"],
    ]


def test_staged_content_is_what_is_checked(tmp_path):
    root = make_repo(tmp_path, committed={"notes.txt": "fine\n"})
    (root / "notes.txt").write_text("cd " + USERS + "/someone/x\n")
    git(root, "add", "notes.txt")
    (root / "notes.txt").write_text("fine again\n")  # worktree differs from what is staged
    status, lines = run_script(root)
    assert status == 1
    assert lines[0].startswith("notes.txt:1: absolute-path: ")


# --- tracked-private -----------------------------------------------------------------------------

PRIVATE_PATHS = [
    "data/x.csv",
    "data/sub/deep/x.csv",
    "runs.noindex/run1/manifest.json",
    "reference/old.py",
    ".venv/bin/python",
    "src/.venv/lib.py",
    "sub/runs.noindex/x.txt",
    "STATUS.md",
    "docs/STATUS.md",
    ".env",
    "sub/.env",
    "grid.token",
    "sub/my_api_key.txt",
    "acn_api_key",
    "anthropic_key",
    "sub/anthropic_key.txt",
    ".key/k.txt",
    "sub/.key/k.txt",
    "API_KEY.txt",
    "sub/Anthropic_Key",
    "sub/Grid.TOKEN",
    "other/.ENV",
    ".env.local",
    ".env.production",
    "sub/.env.example",
]

PRIVATE_LOOK_ALIKES = [
    "src/fleetreserve/data/table.py",
    "docs/data.md",
    "tests/reference_values.py",
    "database/a.txt",
    "notes/status.md",
    "docs/side_notes.md",
    "tests/test_environment.py",
    "src/keyboard.py",
    "tests/key_values.json",
    "tools/envelope.py",
    "src/token_counter.py",
    "data.py",
    ".envrc.example",
    "environment.yml",
    "my_env.py",
    "docs/env.md",
    "src/token_bucket.py",
]


def test_tracked_private_paths_are_reported(tmp_path):
    cases = [(path, "x\n", f"{path}:-: tracked-private: ", "") for path in PRIVATE_PATHS]
    each_reported_once(tmp_path, cases)


def test_private_look_alikes_pass(tmp_path):
    none_reported(tmp_path, [(path, "x = 1\n") for path in PRIVATE_LOOK_ALIKES])


# The checkout's own exclude file lists the names that stay on this machine; the rule adds every
# plain name and folder pattern in it to the built-in private names.
LOCAL_EXCLUDE = (
    """\
# comment lines and blank lines add nothing

notes.md
private/
**/scratch-notes/
**/hidden.cfg
  spaced.txt"""
    + "  \n"
)

LOCAL_PRIVATE_PATHS = [
    "notes.md",
    "sub/notes.md",
    "private/x",
    "private/sub/deep/x.txt",
    "sub/private/x",
    "scratch-notes/x",
    "sub/scratch-notes/deep/x",
    "hidden.cfg",
    "sub/hidden.cfg",
    "spaced.txt",
]

LOCAL_LOOK_ALIKES = [
    "notes.md.txt",
    "notesx.md",
    "docs/my-notes.md",
    "private.md",
    "docs/private_data/x.txt",
    "scratch-notes.txt",
    "hidden.cfg.bak",
    "docs/spaced.txt.old",
]


def test_names_in_the_exclude_file_are_reported(tmp_path):
    root = make_repo(tmp_path, {path: "x\n" for path in LOCAL_PRIVATE_PATHS}, exclude=LOCAL_EXCLUDE)
    status, lines = run_script(root)
    assert status == 1, lines
    assert sorted(line.split(": ")[0] for line in lines) == sorted(
        f"{path}:-" for path in LOCAL_PRIVATE_PATHS
    )
    assert all(": tracked-private: " in line for line in lines), lines


def test_look_alikes_of_the_exclude_names_pass(tmp_path):
    root = make_repo(tmp_path, {p: "x = 1\n" for p in LOCAL_LOOK_ALIKES}, exclude=LOCAL_EXCLUDE)
    status, lines = run_script(root)
    assert status == 0, lines


def test_a_folder_pattern_does_not_match_a_file_of_that_name(tmp_path):
    root = make_repo(tmp_path, {"private": "x\n", "sub/private": "x\n"}, exclude="private/\n")
    status, lines = run_script(root)
    assert status == 0, lines


def test_exclude_forms_that_are_not_read_add_nothing(tmp_path):
    exclude = "*.tmp\nsub/inner.md\n/rooted.txt\n!kept.md\nback\\slash\n[ab].md\nq?.md\n"
    files = ["x.tmp", "sub/inner.md", "rooted.txt", "kept.md", "a.md", "qq.md"]
    root = make_repo(tmp_path, {path: "x\n" for path in files}, exclude=exclude)
    status, lines = run_script(root)
    assert status == 0, lines


def test_no_exclude_file_adds_nothing(tmp_path):
    root = make_repo(tmp_path, {"notes.md": "x\n", "private/x": "x\n"})
    assert not (root / ".git" / "info" / "exclude").exists()
    status, lines = run_script(root)
    assert status == 0, lines


def test_exclude_names_are_read_in_a_linked_working_copy(tmp_path):
    root = make_repo(tmp_path, {"notes.md": "x\n"}, exclude="notes.md\n")
    copy = tmp_path / "copy"
    git(root, *IDENTITY, "worktree", "add", "-q", "--detach", str(copy))
    status, lines = run_script(copy)
    assert status == 1, lines
    assert [line.split(": ")[:2] for line in lines] == [["notes.md:-", "tracked-private"]]


def test_an_unreadable_exclude_file_exits_2(tmp_path):
    root = make_repo(tmp_path)
    (root / ".git" / "info").mkdir()
    (root / ".git" / "info" / "exclude").mkdir()  # a folder where the file should be
    status, lines = run_script(root)
    assert status == 2
    assert len(lines) == 1


def test_built_in_private_names_are_the_expected_set():
    names = repo_rules.PRIVATE_TOP_FOLDERS + repo_rules.PRIVATE_FILE_NAMES
    assert sorted(names) == sorted(["data", "runs.noindex", "reference", ".venv", "STATUS.md"])


# --- large-file ----------------------------------------------------------------------------------


def test_file_over_the_limit_is_reported(tmp_path):
    size = repo_rules.LARGE_FILE_BYTES + 1
    lines = violations(tmp_path, {"docs/big.txt": "a" * size})
    assert len(lines) == 1
    assert lines[0].startswith("docs/big.txt:-: large-file: ")
    assert str(size) in lines[0]


def test_files_at_or_under_the_limit_pass(tmp_path):
    sizes = [0, 1000, repo_rules.LARGE_FILE_BYTES]
    none_reported(tmp_path, [(f"docs/size{size}.txt", "a" * size) for size in sizes])


# --- absolute-path -------------------------------------------------------------------------------

ABSOLUTE_PATH_CASES = [
    ("cd " + USERS + "/someone/work", USERS + "/someone/work"),
    ("see " + HOME + "/someone/f.csv now", HOME + "/someone/f.csv"),
    (SLASH + "private/var/folders/x", SLASH + "private/var/folders/x"),
    ('path = "' + SLASH + 'tmp/out.csv"', SLASH + "tmp/out.csv"),
    ("(" + SLASH + "var/log/x)", SLASH + "var/log/x"),
    (SLASH + "Volumes/Disk/x", SLASH + "Volumes/Disk/x"),
    ("PATH=" + SLASH + "opt/homebrew/bin", SLASH + "opt/homebrew/bin"),
    ("cat " + SLASH + "etc/hosts", SLASH + "etc/hosts"),
    ("open(" + SLASH + "dev/null)", SLASH + "dev/null"),
    ("run " + SLASH + "usr/local/bin/tool", SLASH + "usr/local/bin/tool"),
    ("file://" + USERS + "/someone/x", USERS + "/someone/x"),
    ("a: " + SLASH + "tmp/x", SLASH + "tmp/x"),
    ("see `" + USERS + "/someone/x` here", USERS + "/someone/x"),
    ("saved to <" + HOME + "/someone/x>", HOME + "/someone/x"),
    ("in *" + SLASH + "tmp/x* only", SLASH + "tmp/x"),
    ("path:" + USERS + "/someone/x", USERS + "/someone/x"),
]

PATH_LOOK_ALIKES = [
    "https://example.org/home/page and https://example.org/tmp/x",
    "http://host:8080/var/log",
    "ssh://git@example.org/opt/repo.git",
    "git@example.org:user/home/x",
    "and/or, either/or, I/O, km/h",
    "a fraction 1/2 and 3/4, ratio 10/etc",
    "a relative path src/tmp/x.py and ./tmp/x and ../var/x and ~/work",
    "double slash a//tmp/b and a//home/b",
    "dollar " + "${DIR}" + SLASH + "var/log and " + "$HOME" + SLASH + "tmp/x",
    "#!" + SLASH + "usr/bin/env bash",
    SLASH + "usr/bin/env -i PATH=x",
    "words " + SLASH + "tmpfile and " + SLASH + "optional/x and " + SLASH + "developer/x",
    SLASH + "etcetera and " + SLASH + "homepage and " + SLASH + "variable",
    "bare " + SLASH + "tmp and " + SLASH + "home without a trailing slash",
    "time 10:30, ratio 3:1, Figure 1: results, Note: this",
    "C:" + " " + "not a drive path",
    "x = {'key': 1}",
    "the colon in key:value pairs",
    r"$f:\mathbb{R}^n\to\mathbb{R}$",
    r"$g:\R$",
    r"$h:\{0,1\}\to\R$",
    r"f: \mathbb{R}",
    r"a map \(f:\mathbb{R}\to\mathbb{R}\)",
    r"{f:\R}, \f:\R, and ef:\R",
    "C:" + BACKSLASH + "Users" + BACKSLASH + "name" + BACKSLASH + "file.csv",
    '"D:' + BACKSLASH * 2 + "data" + BACKSLASH * 2 + 'x"',
    "the C:" + BACKSLASH + " drive, path=E:" + BACKSLASH + "work" + BACKSLASH + "x",
    "open(F:" + BACKSLASH + "out" + BACKSLASH + "x) and [G:" + BACKSLASH + "data" + BACKSLASH + "x",
    r"\[ h:\Omega\to[0,1] \]",
    r"F:\mathcal{X}\to\mathbb{R}",
    r"$ f:\mathbb{R}\to\mathbb{R} $",
    r"$$ c:\{1,\dots,T\}\to\R $$",
]


def test_absolute_paths_are_reported_with_line(tmp_path):
    cases = [
        (
            f"case{i}.txt",
            "first\nsecond\n" + text + "\nlast\n",
            f"case{i}.txt:3: absolute-path: ",
            token,
        )
        for i, (text, token) in enumerate(ABSOLUTE_PATH_CASES)
    ]
    each_reported_once(tmp_path, cases)


def test_path_look_alikes_pass(tmp_path):
    none_reported(
        tmp_path, [(f"case{i}.txt", f"first\n{t}\n") for i, t in enumerate(PATH_LOOK_ALIKES)]
    )


def test_the_allowed_absolute_path_is_exactly_one_token(tmp_path):
    lines = violations(tmp_path, {"run.sh": "#!" + SLASH + "usr/bin/env.old bash\n"})
    assert lines[0].startswith("run.sh:1: absolute-path: ")


# --- secret-text ---------------------------------------------------------------------------------

KEY_CASES = [
    "token = " + ANT_PREFIX + "api03-" + "x" * 40,
    ANT_PREFIX + "A_b-" * 8,
    "id = AKIA" + "ABCDEFGHIJKLMNOP",
    "key=AKIA" + "0123456789ABCDEF;",
    PEM_OPEN + "RSA PRIVATE KEY" + PEM_CLOSE,
    PEM_OPEN + "PRIVATE KEY" + PEM_CLOSE,
    PEM_OPEN + "OPENSSH PRIVATE KEY" + PEM_CLOSE,
    PEM_OPEN + "EC PRIVATE KEY" + PEM_CLOSE,
]

KEY_LOOK_ALIKES = [
    "keys start with " + ANT_PREFIX + " and a long string",
    ANT_PREFIX + "short",
    "AKIA" + "SHORT",
    "AKIA" + "A" * 17,
    "xAKIA" + "ABCDEFGHIJKLMNOP",
    "akia" + "abcdefghijklmnop",
    PEM_OPEN + "CERTIFICATE" + PEM_CLOSE,
    PEM_OPEN + "PUBLIC KEY" + PEM_CLOSE,
    "a private key is never committed",
    "sk-learn and ask-ant-farm",
]


def test_live_key_shapes_are_reported(tmp_path):
    cases = [
        (f"cfg{i}.txt", "a\n" + text + "\n", f"cfg{i}.txt:2: secret-text: ", "")
        for i, text in enumerate(KEY_CASES)
    ]
    each_reported_once(tmp_path, cases)


def test_the_report_does_not_repeat_the_key(tmp_path):
    secret = ANT_PREFIX + "api03-" + "Zq9" * 12
    lines = violations(tmp_path, {"cfg.txt": "token = " + secret + "\n"})
    assert secret not in "\n".join(lines)
    assert "Zq9" not in "\n".join(lines)
    assert "Anthropic" not in "\n".join(lines)


def test_key_look_alikes_pass(tmp_path):
    none_reported(tmp_path, [(f"cfg{i}.txt", f"a\n{t}\n") for i, t in enumerate(KEY_LOOK_ALIKES)])


# --- solver-door ---------------------------------------------------------------------------------

SOLVER_LINES = [
    "from scipy.optimize import linprog",
    "from scipy.optimize import milp, LinearConstraint",
    "import highspy",
    "import cvxpy as cp",
    "import pulp",
    "import gurobipy as gp",
    "from scipy.optimize._highs import _highs_wrapper",
    "result = scipy.optimize.linprog(c)",
    "x = optimize.linprog(c, A_ub=a)",
]


def test_direct_solver_use_is_reported(tmp_path):
    cases = []
    for i, line in enumerate(SOLVER_LINES):
        for folder in ("src/fleetreserve", "tools"):
            path = f"{folder}/other{i}.py"
            cases.append((path, "import os\n" + line + "\n", f"{path}:2: solver-door: ", ""))
    each_reported_once(tmp_path, cases)


def test_a_same_named_file_in_another_folder_is_not_exempt(tmp_path):
    lines = violations(
        tmp_path, {"src/fleetreserve/sub/lp.py": "from scipy.optimize import linprog\n"}
    )
    assert lines[0].startswith("src/fleetreserve/sub/lp.py:1: solver-door: ")


def test_solver_look_alikes_pass(tmp_path):
    none_reported(
        tmp_path,
        [
            ("src/fleetreserve/a.py", "from fleetreserve.lp import solve\n"),
            ("src/fleetreserve/b.py", "import scipy.optimize\nscipy.optimize.minimize(f)\n"),
            ("src/fleetreserve/c.py", "linprogress = 1\nmilpa = 2\npulpit = 3\n"),
            ("src/fleetreserve/d.py", "# solved through the one timed solve function\n"),
            ("src/fleetreserve/notes.md", "The solver is linprog.\n"),
            ("docs/plan.md", "linprog is named here\n"),
        ],
    )


def test_exempt_files_may_use_the_solver(tmp_path):
    text = "from scipy.optimize import linprog, milp\nimport highspy\n"
    paths = ["src/fleetreserve/lp.py", "tests/test_lp.py", "tests/sub/check_solver.py"]
    none_reported(tmp_path, [(path, text) for path in paths])


def test_the_rule_file_itself_may_name_the_solvers(tmp_path):
    passes(tmp_path, {"tools/repo_rules.py": "NAMES = ['linprog', 'milp']\n"})


# --- heldout-seal --------------------------------------------------------------------------------

HELDOUT_LINES = [
    "open('data/heldout/x.csv')",
    "HELDOUT_DIR = 1",
    "Heldout = 1",
    "from fleetreserve import heldout_loader",
    "p = 'data/held_out/x.csv'",
    "p = 'data/held-out/x.csv'",
    "p = 'held_out/x.csv'",
    "p = 'data\\\\held-out\\\\x.csv'",
    "p = base + '/held_out'",
    "p = 'x' + '/held-out'",
]

HELDOUT_LOOK_ALIKES = [
    "held_out_fraction = 0.2",
    "# the held-out fold is scored once",
    "HELD_OUT_SCORE = 1",
    "held out data stays unseen",
    "x = 'held-out'",
    "held_out_value=3",
    "theld_out/x",
]


def test_other_code_naming_heldout_is_reported(tmp_path):
    cases = []
    for i, line in enumerate(HELDOUT_LINES):
        for path in (f"src/fleetreserve/data{i}.py", f"tools/run{i}.sh", f"src/notes{i}.txt"):
            cases.append((path, "a\nb\n" + line + "\n", f"{path}:3: heldout-seal: ", ""))
    each_reported_once(tmp_path, cases)


def test_the_name_of_the_limits_module_passes(tmp_path):
    none_reported(
        tmp_path,
        [
            ("src/fleetreserve/load_x.py", "from fleetreserve.heldout_limits import LIMIT_KW\n"),
            ("src/fleetreserve/load_y.py", "from fleetreserve import heldout_limits\n"),
        ],
    )


def test_a_longer_name_that_starts_with_it_is_reported(tmp_path):
    each_reported_once(
        tmp_path,
        [
            (path, f"a\nb\n{line}\n", f"{path}:3: heldout-seal: ", "")
            for path, line in (
                ("src/fleetreserve/a.py", "import heldout_limits_extra"),
                ("src/fleetreserve/b.py", "p = 'data/heldout/limits'"),
                ("src/fleetreserve/c.py", "from fleetreserve import heldout_loader"),
            )
        ],
    )


def test_held_out_wording_that_is_not_a_path_passes(tmp_path):
    none_reported(
        tmp_path,
        [(f"src/fleetreserve/model{i}.py", f"{t}\n") for i, t in enumerate(HELDOUT_LOOK_ALIKES)],
    )


def test_allow_listed_files_may_name_heldout(tmp_path):
    text = "open('data/heldout/x.csv')\nHELDOUT = 1\n"
    none_reported(tmp_path, [(path, text) for path in repo_rules.HELDOUT_FILES])


def test_documentation_tests_and_other_folders_are_exempt(tmp_path):
    paths = ["docs/plan.md", "tests/test_heldout.py", "tests/sub/heldout_cases.py", "check.sh"]
    none_reported(tmp_path, [(path, "data/heldout/x.csv and held_out/y\n") for path in paths])


def test_a_similar_file_name_is_not_allow_listed(tmp_path):
    lines = violations(tmp_path, {"src/fleetreserve/heldout_other.py": "HELDOUT = 1\n"})
    assert lines[0].startswith("src/fleetreserve/heldout_other.py:1: heldout-seal: ")


def test_a_file_name_alone_is_not_text(tmp_path):
    passes(tmp_path, {"src/fleetreserve/heldout_other.py": "x = 1\n"})
