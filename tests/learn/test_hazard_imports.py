"""PyTorch must be imported after LightGBM in every file of this project that imports it
(`hazard_net`, module text: a process that imports torch first ends with a segmentation fault
the first time LightGBM trains). The files under `src/` and `tests/` are read as text, nothing is
imported. A line imports torch when it is an `import torch` or `from torch` statement, or calls
`import_module` or `__import__` with the text "torch" as the first argument (a call is found
whether or not it is written as `importlib.import_module`).

1. In a file with an unindented line that imports torch, the first such line is preceded by an
   unindented `import lightgbm` line.
2. A file that imports torch only inside a function (an indented line) must be named in `LAZY`
   with the reason that is safe, and the reason is checked as far as text allows: LightGBM is
   imported earlier in the same file. A new such file fails the test until it is named.
3. The scan finds the files it is meant to guard (it is not empty).
4. The scan itself is tested on files made under a temporary folder: each way of importing torch
   is found, and a file that imports LightGBM first passes.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = re.compile(r"^(import torch\b|from torch\b)")
INDENTED = re.compile(r"^\s+(import torch\b|from torch\b)")
DYNAMIC = re.compile(r"(import_module|__import__)\(\s*[\"']torch\b")
LIGHTGBM = re.compile(r"^\s*import lightgbm\b")
LIGHTGBM_TOP = re.compile(r"^import lightgbm\b")
LAZY = {
    "tests/learn/test_learn_lane.py": (
        "imports torch inside its last test, after the test above it has imported LightGBM; "
        "pytest runs the tests of a file in order, and that test runs torch on one thread"
    ),
}
GUARDED = (
    "src/fleetreserve/hazard_net.py",
    "src/fleetreserve/hazard_fit.py",
    "src/fleetreserve/models/hazard.py",
    "tests/learn/test_hazard_net.py",
    "tests/learn/test_hazard_likelihood.py",
)


def _files(root):
    for folder in ("src", "tests"):
        yield from sorted((root / folder).rglob("*.py"))


def _imports_torch_at_the_start(line):
    return bool(TOP.match(line) or (not line[:1].isspace() and DYNAMIC.search(line)))


def _imports_torch_indented(line):
    return bool(INDENTED.match(line) or (line[:1].isspace() and DYNAMIC.search(line)))


def _first(test, lines):
    return next((k for k, line in enumerate(lines) if test(line)), None)


def scan(root):
    """(files that import torch at the start of a line, files that import it only in an indented
    line, a description of every file of the first kind and of the second kind that does not
    import LightGBM first)."""
    guarded, lazy, problems_top, problems_lazy = [], [], [], []
    for path in _files(root):
        lines = path.read_text().splitlines()
        name = path.relative_to(root).as_posix()
        top = _first(_imports_torch_at_the_start, lines)
        indented = _first(_imports_torch_indented, lines)
        if top is not None:
            guarded.append(name)
            before = [k for k, line in enumerate(lines[:top]) if LIGHTGBM_TOP.match(line)]
            if not before:
                problems_top.append(f"{name}: torch is imported at line {top + 1} with no LightGBM")
        elif indented is not None:
            lazy.append(name)
            earlier = [k for k, line in enumerate(lines) if LIGHTGBM.match(line)]
            if not (earlier and earlier[0] < indented):
                problems_lazy.append(
                    f"{name}: torch is imported at line {indented + 1} before LightGBM"
                )
    return guarded, lazy, problems_top, problems_lazy


def test_torch_is_imported_after_lightgbm_wherever_it_is_imported_at_the_top():
    guarded, _, problems, _ = scan(ROOT)
    assert not problems, problems
    assert set(GUARDED) <= set(guarded)


def test_a_file_that_imports_torch_inside_a_function_is_named_with_its_reason():
    _, lazy, _, problems = scan(ROOT)
    assert not problems, problems
    assert sorted(lazy) == sorted(LAZY)


# --- the scan, tested on files made under a temporary folder -------------------------------------

IMPORT_MODULE = "importlib.import_module(" + '"torch")'
DUNDER = "__import__(" + "'torch')"
BARE_IMPORT_MODULE = "import_module(" + '"torch.nn")'


def _root(tmp_path, files):
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_the_scan_finds_every_way_of_importing_torch_before_lightgbm(tmp_path):
    root = _root(
        tmp_path,
        {
            "src/plain.py": "import torch\n",
            "src/from_form.py": "from torch import nn\n",
            "src/dynamic_top.py": f"import importlib\nmodule = {IMPORT_MODULE}\n",
            "src/dunder_top.py": f"torch = {DUNDER}\n",
            "src/bare_top.py": f"from importlib import import_module\nx = {BARE_IMPORT_MODULE}\n",
            "src/dynamic_lazy.py": f"def f():\n    return {IMPORT_MODULE}\n",
            "src/dunder_lazy.py": f"def f():\n    return {DUNDER}\n",
            "src/lazy_plain.py": "def f():\n    import torch\n    return torch\n",
        },
    )
    guarded, lazy, problems_top, problems_lazy = scan(root)
    assert sorted(guarded) == [
        "src/bare_top.py",
        "src/dunder_top.py",
        "src/dynamic_top.py",
        "src/from_form.py",
        "src/plain.py",
    ]
    assert sorted(lazy) == ["src/dunder_lazy.py", "src/dynamic_lazy.py", "src/lazy_plain.py"]
    assert len(problems_top) == 5 and len(problems_lazy) == 3, (problems_top, problems_lazy)


def test_the_scan_passes_files_that_import_lightgbm_first(tmp_path):
    root = _root(
        tmp_path,
        {
            "src/top.py": "import lightgbm\nimport torch\n",
            "src/dynamic_top.py": f"import lightgbm\nmodule = {IMPORT_MODULE}\n",
            "src/dunder_lazy.py": f"import lightgbm\n\n\ndef f():\n    return {DUNDER}\n",
            "tests/not_torch.py": "import lightgbm\ntorchlike = 1\nimport torchvision_like\n",
        },
    )
    guarded, lazy, problems_top, problems_lazy = scan(root)
    assert sorted(guarded) == ["src/dynamic_top.py", "src/top.py"]
    assert lazy == ["src/dunder_lazy.py"]
    assert problems_top == [] and problems_lazy == []
