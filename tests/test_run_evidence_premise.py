"""The premise of `run_evidence`'s rule for runs from an earlier commit, looked at on this tree.
The premise is stated in one place, the module text of `run_evidence` ("The premise"); here it is
looked at through the places of the keep-list `UNREAD_BY_RUNS` that modules under `src/` name.

This is a check against accidents, not a proof. It looks at exactly these, and fails on a hit
not listed in `REVIEWED`:
1. Every file git tracks under `src/` is a `.py` file (`git ls-files src`, the index, which is
   the checked-out commit in a clean working copy), so that the next look covers every file
   there.
2. In every `.py` file under `src/`: its string and bytes constants, docstrings left out (reading
   P1), bytes read as Latin-1, the literal parts of f-strings included. A constant names an entry
   as `run_evidence.names_kept_entry` says (reading E1 there): letters in lower case, the text
   contains a folder entry with its "/" (`"a.json,tools/x.json"`), or, cut into parts at each
   "/", "\\" and "=", a part equals a folder entry's name or begins with it and a dot
   (`"tools"`, `"TOOLS/x.json"`, `"../docs"`, `"tools.x"`), or the text contains a file entry
   (`"check.sh"`).
   Its import statements: one names an entry when its first part, in lower case, is a folder
   entry's name.
3. No path git tracks outside the keep-list is a symbolic link (`git ls-files -s`, mode 120000),
   so that no file outside it can be the content of a file inside it.
Not seen: a name built from parts at run time (`"to" + "ols"`), read from a file or a docstring,
or found by listing a folder; a file opened by a program that is not under `src/`.

On a hit: if the module reads that file for a value a run computes, the file is taken out of the
keep-list by name (module text of `run_evidence`); if the string names no file the module reads,
or a file or names it reads only for the sealed-data gate's decision at a logged parsing fix
(whose runs `run_evidence` never reads from an earlier commit, its point 4 and reading E3), it is
added to `REVIEWED` with the reason: one entry per module and exact string, never a pattern or a
whole module. A third string of the keep-list in the held-out loader, or one of its two reviewed
strings in another module, is a new hit.

  P1. Docstrings are left out: they are the modules' own text for readers and name the tests and
      tools they point to (`tests/wp1`, `docs/prior-looks.md`, ...); a module that opened a path
      read from its docstring would be a deliberate trick. [Alternative: docstrings looked at
      too, with each of their mentions listed in `REVIEWED`.]
"""

import ast
import os
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve.run_evidence import UNREAD_BY_RUNS, names_kept_entry

TOP = Path(__file__).resolve().parents[1]
OPENER = "src/fleetreserve/heldout_loader.py"
# Read only for the sealed-data gate's decision (module text, on a hit).
FOR_THE_GATE = (
    "read only for the sealed-data gate's decision at a logged parsing fix of the tag (the "
    "held-out loader's check 5, reading H6), never for a value a run computes; a run through the "
    "gate is never read from an earlier commit (run_evidence, point 4 and reading E3)"
)
# (module path from the top folder, the string or import): the reason it is reviewed.
REVIEWED = {
    **{
        ("src/fleetreserve/run_evidence.py", entry): (
            "the keep-list's own entry, which run_evidence compares with the paths git lists and "
            "with the words of a recorded command, and opens none of"
        )
        for entry in UNREAD_BY_RUNS
    },
    (OPENER, "docs/plan.md"): (
        "the plan, whose section 14 the opener reads with git at the commits of a fix: "
        f"{FOR_THE_GATE}"
    ),
    (OPENER, "tests/(?:[^/]+/)*test_"): (
        "the start of the pattern of a held-out loader's test file names, matched against the "
        f"paths a fix changes; no file under tests/ is opened: {FOR_THE_GATE}"
    ),
}


def unreviewed(source: Path, top: Path) -> list[tuple[str, str]]:
    """The hits under `source` that `REVIEWED` does not list (module text, on a hit)."""
    return [hit for hit in hits(source, top) if hit not in REVIEWED]


def _docstrings(tree: ast.Module) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def hits(source: Path, top: Path) -> list[tuple[str, str]]:
    """(module path from `top`, the constant or the imported name) for every `.py` file under
    `source` that names an entry of the keep-list (module text, look 2)."""
    folders = {entry.rstrip("/") for entry in UNREAD_BY_RUNS if entry.endswith("/")}
    found = []
    for path in sorted(source.rglob("*.py")):
        module = path.relative_to(top).as_posix()
        tree = ast.parse(path.read_text(), filename=module)
        skip = _docstrings(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
                text = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
                if id(node) not in skip and names_kept_entry(text) is not None:
                    found.append((module, text))
            elif isinstance(node, ast.Import):
                found += [
                    (module, a.name)
                    for a in node.names
                    if a.name.split(".")[0].casefold() in folders
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.split(".")[0].casefold() in folders
            ):
                found.append((module, node.module))
    return found


def _git(top: Path, *args: str) -> list[str]:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = run_git(
        ["git", "-C", str(top), *args], capture_output=True, text=True, check=True, env=environment
    )
    return [line for line in done.stdout.split("\0") if line]


def not_python_under_src(top: Path) -> list[str]:
    """Committed files under `src/` that are not `.py` files (module text, look 1)."""
    return [path for path in _git(top, "ls-files", "-z", "--", "src") if not path.endswith(".py")]


def links_off_the_keep_list(top: Path) -> list[str]:
    """Committed symbolic links outside the keep-list (module text, look 3)."""
    found = []
    for line in _git(top, "ls-files", "-s", "-z"):
        info, path = line.split("\t", 1)
        kept = any(
            path.startswith(entry) if entry.endswith("/") else path == entry
            for entry in UNREAD_BY_RUNS
        )
        if info.split()[0] == "120000" and not kept:
            found.append(path)
    return found


def test_no_module_under_src_names_a_place_of_the_keep_list():
    assert not_python_under_src(TOP) == []
    assert unreviewed(TOP / "src", TOP) == []
    assert links_off_the_keep_list(TOP) == []


def test_the_reviewed_list_is_the_keep_lists_own_entries_and_the_openers_two_strings():
    """Nothing wider: one entry per module and exact string, each with a reason."""
    assert set(REVIEWED) == {
        *(("src/fleetreserve/run_evidence.py", entry) for entry in UNREAD_BY_RUNS),
        (OPENER, "docs/plan.md"),
        (OPENER, "tests/(?:[^/]+/)*test_"),
    }
    assert all(isinstance(reason, str) and reason for reason in REVIEWED.values())


# The opener's two reviewed strings, as its module holds them (a constant and an f-string's start).
OPENER_LINES = (
    'PLAN = "docs/plan.md"\n'
    "def pattern(module):\n"
    '    return rf"tests/(?:[^/]+/)*test_{module}(?:_[^/]*)?\\.py"\n'
)


def test_a_third_string_of_the_keep_list_in_the_opener_is_a_hit(tmp_path):
    opener = tmp_path / OPENER
    opener.parent.mkdir(parents=True)
    opener.write_text(OPENER_LINES)
    assert unreviewed(tmp_path / "src", tmp_path) == []  # the two reviewed strings alone
    opener.write_text(OPENER_LINES + 'NOTES = "docs/notes.md"\n')
    assert unreviewed(tmp_path / "src", tmp_path) == [(OPENER, "docs/notes.md")]


@pytest.mark.parametrize("string", ["docs/plan.md", "tests/(?:[^/]+/)*test_"])
def test_a_reviewed_string_of_the_opener_in_another_module_is_a_hit(tmp_path, string):
    other = tmp_path / "src" / "fleetreserve" / "sealed_route.py"
    other.parent.mkdir(parents=True)
    other.write_text(f"WHERE = {string!r}\n")
    assert unreviewed(tmp_path / "src", tmp_path) == [("src/fleetreserve/sealed_route.py", string)]


def test_the_look_finds_each_form_and_leaves_docstrings_out(tmp_path):
    package = tmp_path / "src" / "made_up"
    package.mkdir(parents=True)
    (package / "reads.py").write_text(
        '"""Docstring naming tests/wp1 and docs/plan.md."""\n'
        "import json\n"
        "import tools.helper\n"
        "import Docs.helper\n"
        "from passmarks import lists\n"
        "from . import tools\n"
        "\n"
        "def f(top, name):\n"
        '    """Also left out: passmarks/wp9-t8-runs.json."""\n'
        '    a = top / "tools" / "x.json"\n'
        '    b = top / "passmarks/wp9-t8-runs.json"\n'
        '    c = f"{top}/docs/{name}"\n'
        '    d = "tests.helpers"\n'
        '    e = "bash check.sh"\n'
        '    g = ("the goal tests", "toolset", "docsx", "testing")\n'
        '    h = (b"tools/x.json", "TOOLS/x.json", "../docs", "/a/tests")\n'
        "    return json, a, b, c, d, e, g, h\n"
    )
    assert sorted(hits(tmp_path / "src", tmp_path)) == sorted(
        [
            ("src/made_up/reads.py", "tools.helper"),
            ("src/made_up/reads.py", "Docs.helper"),
            ("src/made_up/reads.py", "passmarks"),
            ("src/made_up/reads.py", "tools"),
            ("src/made_up/reads.py", "passmarks/wp9-t8-runs.json"),
            ("src/made_up/reads.py", "/docs/"),
            ("src/made_up/reads.py", "tests.helpers"),
            ("src/made_up/reads.py", "bash check.sh"),
            ("src/made_up/reads.py", "tools/x.json"),
            ("src/made_up/reads.py", "TOOLS/x.json"),
            ("src/made_up/reads.py", "../docs"),
            ("src/made_up/reads.py", "/a/tests"),
        ]
    )


def test_the_looks_find_a_file_under_src_that_is_not_python_and_a_link_off_the_list(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "made_up").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "gate").mkdir()
    (repo / "src" / "made_up" / "x.py").write_text("X = 1\n")
    (repo / "src" / "made_up" / "where.json").write_text('{"file": "tools/x.json"}\n')
    (repo / "tests" / "matrix.json").write_text("{}\n")
    (repo / "tests" / "link.json").symlink_to("../gate/matrix.json")
    (repo / "gate" / "matrix.json").symlink_to("../tests/matrix.json")
    (repo / "src" / "made_up" / "link.py").symlink_to("../../tests/helper.py")
    for args in (("init", "-q"), ("add", ".")):
        run_git(["git", "-C", str(repo), *args], capture_output=True, check=True)
    assert not_python_under_src(repo) == ["src/made_up/where.json"]
    assert links_off_the_keep_list(repo) == ["gate/matrix.json", "src/made_up/link.py"]
