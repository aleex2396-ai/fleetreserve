"""No route around the gated opener, read from the modules' syntax trees.

1. The three parsing modules (`load_heldout_norway`, `load_asr`, `load_korea`) read no file: no
   call of `open`, `Path.open`, `read_text`, `read_bytes` or `io.open`; `pandas.read_csv` only
   on an in-memory buffer `io.BytesIO(...)`; no import of `heldout_seal` or of its
   `sealed_path`.
2. In `heldout_loader`, the module that may name the sealed folder, `open_sealed` reads bytes
   and is the only function that uses `sealed_path`; no function other than `open_sealed`,
   `logged_openings` and `_append` (the last two read and write the openings log) calls a file
   read; the functions that load a data set call `open_sealed` and read nothing themselves.
3. No module of the package other than `heldout_loader` and `heldout_seal` imports
   `heldout_seal` or uses `sealed_path`, with one exception: `heldout_truth_build`, listed in
   `tools/repo_rules.py`'s `HELDOUT_FILES`, imports `heldout_seal` once and reads only its
   `FOLDER`, to refuse an output folder inside the sealed folder; it never uses `sealed_path`.
4. Test files that call `open_sealed`, `norway_sites`, `asr`, `korea` or `logged_openings` give
   it, as its first positional argument, a root made by `sealed_repo` (a made-up repository in the
   temporary folder) or built from pytest's `tmp_path` (`tmp_path / "copy"`), directly, through
   a name bound only to such roots, through a method called on such a root
   (`root.relative_to(...)`), through a helper's parameter that every caller in the file
   fills that way, or through a test's parameter named after a fixture of the same file
   (decorated `pytest.fixture`) whose every `return` gives such a root; or the constant `None`,
   which names no folder. A fixture of a `conftest.py` is not seen. Calls are seen when written
   through the module (`heldout_loader.asr`, under any name the file imports it as, or
   `fleetreserve.heldout_loader.asr`) or through a name imported
   from it (`from fleetreserve.heldout_loader import asr`, with or without `as`). A call with
   no positional argument counts as a call with a root not made up. Not seen: a function
   reached in another way, such as `getattr`, a name assigned from the module's function, or a
   call from a module that is not a test file. Test files are found by the text
   "heldout_loader"; the calls in this file's own checks are inside strings and are parsed only
   by those checks.
"""

import ast
from pathlib import Path

import pytest

TOP = Path(__file__).resolve().parents[2]
PACKAGE = TOP / "src" / "fleetreserve"
PARSERS = ("load_heldout_norway.py", "load_asr.py", "load_korea.py")
OPENER = "heldout_loader.py"
READS = {"open", "read_text", "read_bytes"}
LOADER_CALLS = {"open_sealed", "norway_sites", "asr", "korea", "logged_openings"}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_buffer(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) == "BytesIO"


def file_reads(tree: ast.AST) -> list[str]:
    """Each call in `tree` that reads a file, as text with its line."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        if name in READS:
            found.append(f"line {node.lineno}: {name}")
        if name == "read_csv" and not (node.args and _is_buffer(node.args[0])):
            found.append(f"line {node.lineno}: read_csv without an in-memory buffer")
    return found


def seal_imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            if (
                "sealed_path" in names
                or "heldout_seal" in names
                or "heldout_seal" in (node.module or "")
            ):
                found.append(f"line {node.lineno}: from {node.module} import {sorted(names)}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "heldout_seal" in alias.name:
                    found.append(f"line {node.lineno}: import {alias.name}")
        if isinstance(node, ast.Attribute) and node.attr == "sealed_path":
            found.append(f"line {node.lineno}: .sealed_path")
    return found


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


@pytest.mark.parametrize("name", PARSERS)
def test_the_parsing_modules_read_no_file(name):
    tree = _tree(PACKAGE / name)
    assert file_reads(tree) == []
    assert seal_imports(tree) == []


def test_the_opener_reads_bytes_only_inside_open_sealed():
    tree = _tree(PACKAGE / OPENER)
    opener = _function(tree, "open_sealed")
    assert any("read_bytes" in found for found in file_reads(opener))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("norway_sites", "asr", "korea"):
            assert file_reads(node) == []
            called = {_name(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}
            assert "open_sealed" in called
    readers = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name not in ("open_sealed", "logged_openings", "_append")
        and file_reads(node)
    ]
    assert readers == []
    seal_uses = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef) and seal_imports(node)
    ]
    assert seal_uses == ["open_sealed"]


# The one module that may import `heldout_seal` beside the opener: the held-out truth build,
# which `tools/repo_rules.py` lists in `HELDOUT_FILES` (the owner's confirmation of 15 November
# 2026). It may read the folder's name `FOLDER` only, to refuse an output folder inside it, and
# never `sealed_path`.
FOLDER_ONLY = "heldout_truth_build.py"


def seal_attributes(tree: ast.AST) -> set[str]:
    """The attributes read from the name `heldout_seal` in `tree`."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "heldout_seal"
    }


def test_no_other_module_of_the_package_takes_the_sealed_path():
    users = sorted(
        path.name
        for path in PACKAGE.glob("*.py")
        if path.name not in (OPENER, "heldout_seal.py", FOLDER_ONLY) and seal_imports(_tree(path))
    )
    assert users == []
    assert folder_only(_tree(PACKAGE / FOLDER_ONLY))


def folder_only(tree: ast.AST) -> bool:
    """True when `tree` imports the module `heldout_seal` once, as `from fleetreserve import`,
    and reads nothing from it but `heldout_seal.FOLDER`."""
    found = seal_imports(tree)
    return (
        len(found) == 1
        and ": from fleetreserve import " in found[0]
        and "sealed_path" not in found[0]
        and seal_attributes(tree) == {"FOLDER"}
    )


@pytest.mark.parametrize(
    "source, allowed",
    [
        ("from fleetreserve import heldout_seal, truth\nx = heldout_seal.FOLDER\n", True),
        ("from fleetreserve import heldout_seal\nx = heldout_seal.sealed_path\n", False),
        ("from fleetreserve import heldout_seal\nx = heldout_seal.FOLDER, heldout_seal.X\n", False),
        ("from fleetreserve import heldout_seal\n", False),
        ("from fleetreserve.heldout_seal import FOLDER\nx = FOLDER\n", False),
        ("import fleetreserve.heldout_seal as heldout_seal\nx = heldout_seal.FOLDER\n", False),
        (
            "from fleetreserve import heldout_seal\nfrom fleetreserve import heldout_seal as s\n"
            "x = heldout_seal.FOLDER\n",
            False,
        ),
    ],
)
def test_the_folder_only_rule_sees_any_other_use(source, allowed):
    assert folder_only(ast.parse(source)) is allowed


def _assignments(function: ast.FunctionDef) -> dict[str, list[ast.expr]]:
    assigned: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)
    return assigned


def _is_fixture(decorator: ast.expr) -> bool:
    """`@pytest.fixture` or `@pytest.fixture(...)`."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (
        isinstance(target, ast.Attribute)
        and target.attr == "fixture"
        and isinstance(target.value, ast.Name)
        and target.value.id == "pytest"
    )


def _fixture(module: ast.Module, name: str) -> ast.FunctionDef | None:
    """The fixture function `name` defined at the top of `module`, or None."""
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node if any(_is_fixture(d) for d in node.decorator_list) else None
    return None


def _root_is_made_up(arg: ast.expr, function: ast.FunctionDef, module: ast.Module, depth=0) -> bool:
    """True when `arg`, inside `function`, is `tmp_path`, a path built from it, a name bound
    only to `sealed_repo(...)`, or a parameter every caller in the module fills that way."""
    if depth > 5:
        return False
    if isinstance(arg, ast.Name) and arg.id == "tmp_path":
        return True
    if isinstance(arg, ast.Constant) and arg.value is None:
        return True
    if isinstance(arg, ast.BinOp):
        return _root_is_made_up(arg.left, function, module, depth)
    if isinstance(arg, ast.Call) and _name(arg.func) == "sealed_repo":
        return True
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        return _root_is_made_up(arg.func.value, function, module, depth + 1)
    if not isinstance(arg, ast.Name):
        return False
    assigned = _assignments(function).get(arg.id)
    if assigned:
        return all(_root_is_made_up(value, function, module, depth + 1) for value in assigned)
    parameters = [a.arg for a in function.args.args]
    if arg.id not in parameters:
        return False
    fixture = _fixture(module, arg.id)
    if function.name.startswith("test_") and fixture is not None:
        returned = [n.value for n in ast.walk(fixture) if isinstance(n, ast.Return)]
        # a bare `return` gives None, which is no made-up root
        return bool(returned) and all(
            _root_is_made_up(value, fixture, module, depth + 1) for value in returned
        )
    position = parameters.index(arg.id)
    callers = []
    for caller in ast.walk(module):
        if not isinstance(caller, ast.FunctionDef):
            continue
        for node in ast.walk(caller):
            if isinstance(node, ast.Call) and _name(node.func) == function.name:
                callers.append((caller, node))
    return bool(callers) and all(
        len(call.args) > position
        and _root_is_made_up(call.args[position], caller, module, depth + 1)
        for caller, call in callers
    )


def _loader_names(tree: ast.Module) -> tuple[set[str], dict[str, str]]:
    """Names bound to the `heldout_loader` module, and names bound to its loading functions
    (`from fleetreserve.heldout_loader import asr as load`), by any import in `tree`."""
    modules: set[str] = set()
    functions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "fleetreserve.heldout_loader" and alias.asname:
                    modules.add(alias.asname)
        if isinstance(node, ast.ImportFrom) and node.module == "fleetreserve":
            for alias in node.names:
                if alias.name == "heldout_loader":
                    modules.add(alias.asname or alias.name)
        if isinstance(node, ast.ImportFrom) and node.module == "fleetreserve.heldout_loader":
            for alias in node.names:
                if alias.name in LOADER_CALLS:
                    functions[alias.asname or alias.name] = alias.name
    return modules, functions


def _loader_call(node: ast.Call, modules: set[str], functions: dict[str, str]) -> str | None:
    """The loading function `node` calls, or None if it calls none."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in functions:
        return functions[func.id]
    if isinstance(func, ast.Attribute) and func.attr in LOADER_CALLS:
        owner = func.value
        if isinstance(owner, ast.Name) and owner.id in modules:
            return func.attr
        dotted = isinstance(owner, ast.Attribute) and owner.attr == "heldout_loader"
        if dotted and isinstance(owner.value, ast.Name) and owner.value.id == "fleetreserve":
            return func.attr
    return None


def _calls_with_unmade_roots(tree: ast.Module) -> list[str]:
    modules, functions = _loader_names(tree)
    modules.add("heldout_loader")
    bad = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            called = _loader_call(node, modules, functions)
            if called is None:
                continue
            if not (node.args and _root_is_made_up(node.args[0], function, tree)):
                bad.append(f"{function.name} line {node.lineno}: {called}")
    return bad


def test_tests_give_the_opener_only_made_up_roots():
    users = [
        path
        for path in sorted((TOP / "tests").rglob("test_*.py"))
        if "heldout_loader" in path.read_text(encoding="utf-8")
    ]
    assert len(users) >= 3
    bad = {path.name: _calls_with_unmade_roots(_tree(path)) for path in users}
    assert {name: found for name, found in bad.items() if found} == {}


def test_the_root_check_sees_a_real_root():
    source = (
        "def test_x():\n"
        "    root = sessions.TOP\n"
        "    heldout_loader.asr(root)\n"
        "    heldout_loader.open_sealed(TOP, 'asr', 'x.csv')\n"
        "def _helper(root):\n"
        "    heldout_loader.norway_sites(root)\n"
        "def test_y(tmp_path):\n"
        "    _helper(TOP)\n"
        "def _made(root):\n"
        "    heldout_loader.logged_openings(root)\n"
        "def test_z(tmp_path):\n"
        "    _made(sealed_repo(tmp_path, {}))\n"
        "    heldout_loader.logged_openings(tmp_path / 'repo')\n"
        "    copy = tmp_path / 'copy'\n"
        "    heldout_loader.asr(copy)\n"
        "    heldout_loader.asr(copy.relative_to(tmp_path))\n"
        "def test_w(tmp_path):\n"
        "    real = TOP / 'repo'\n"
        "    heldout_loader.asr(real)\n"
        "    heldout_loader.asr(TOP.relative_to(tmp_path))\n"
        "    mixed = sealed_repo(tmp_path, {})\n"
        "    mixed = TOP\n"
        "    heldout_loader.asr(mixed)\n"
        "    heldout_loader.asr(sessions.TOP.absolute())\n"
        "    heldout_loader.asr(None)\n"
        "    heldout_loader.asr('.')\n"
    )
    found = _calls_with_unmade_roots(ast.parse(source))
    assert [line.split(" line ")[0] for line in found] == [
        "test_x",
        "test_x",
        "_helper",
        "test_w",
        "test_w",
        "test_w",
        "test_w",
        "test_w",
    ]


def test_the_root_check_reads_a_fixture_of_the_same_file():
    source = (
        "@pytest.fixture\n"
        "def made(tmp_path):\n"
        "    return sealed_repo(tmp_path, {})\n"
        "@pytest.fixture(scope='module')\n"
        "def made_once(tmp_path_factory):\n"
        "    folder = tmp_path_factory.mktemp('x')\n"
        "    root = sealed_repo(folder, {})\n"
        "    return root\n"
        "@pytest.fixture\n"
        "def real():\n"
        "    return TOP\n"
        "@pytest.fixture\n"
        "def mixed(tmp_path):\n"
        "    if tmp_path:\n"
        "        return sealed_repo(tmp_path, {})\n"
        "    return TOP\n"
        "@pytest.fixture\n"
        "def yielded(tmp_path):\n"
        "    yield sealed_repo(tmp_path, {})\n"
        "def plain(tmp_path):\n"
        "    return sealed_repo(tmp_path, {})\n"
        "def test_a(made, made_once):\n"
        "    heldout_loader.asr(made)\n"
        "    heldout_loader.logged_openings(made_once)\n"
        "def test_b(real, mixed, yielded, plain):\n"
        "    heldout_loader.asr(real)\n"
        "    heldout_loader.asr(mixed)\n"
        "    heldout_loader.asr(yielded)\n"
        "    heldout_loader.asr(plain)\n"
        "def _helper(made):\n"
        "    heldout_loader.asr(made)\n"
        "@pytest.fixture\n"
        "def bare(tmp_path):\n"
        "    return\n"
        "def test_c(bare):\n"
        "    heldout_loader.asr(bare)\n"
    )
    found = _calls_with_unmade_roots(ast.parse(source))
    assert [line.split(": ")[0] for line in found] == [
        "test_b line 26",
        "test_b line 27",
        "test_b line 28",
        "test_b line 29",
        "_helper line 31",
        "test_c line 36",
    ]


def test_the_root_check_sees_functions_imported_directly():
    source = (
        "from fleetreserve.heldout_loader import asr, open_sealed as opener\n"
        "from fleetreserve import heldout_loader as hl\n"
        "import fleetreserve.heldout_loader as loader\n"
        "import fleetreserve.heldout_loader\n"
        "def test_x(tmp_path):\n"
        "    asr(TOP)\n"
        "    opener(TOP, 'asr', 'x.csv')\n"
        "    hl.norway_sites(TOP)\n"
        "    loader.logged_openings(TOP)\n"
        "    fleetreserve.heldout_loader.asr(TOP)\n"
        "    asr(sealed_repo(tmp_path, {}))\n"
        "    hl.norway_sites(tmp_path / 'repo')\n"
        "    print(TOP)\n"
    )
    found = _calls_with_unmade_roots(ast.parse(source))
    assert [line.split(": ")[1] for line in found] == [
        "asr",
        "open_sealed",
        "norway_sites",
        "logged_openings",
        "asr",
    ]
