"""Plan section 12 and section 13 (WP1): the earlier analysis fixed full per-car matrices.

The plan says the earlier analysis "fixed affine shares (full Gamma_i, which look ahead)" at
10:00 in its programme (24). Full means that car i's matrix Gamma_i, which maps the base-set
profile u0 to the car's power as g_i + Gamma_i u0, is not restricted to be lower-triangular or
diagonal, so the car's power in one step may depend on u0 in later steps. The causal policy of
plan section 2 is the lower-triangular case.

This file checks the earlier code itself, `reference/part1/src/fleetoffer/offer.py`, which is
kept on this machine only (never in git). Three kinds of evidence:

1. The file is the one stated in `reference/part1/PROVENANCE.txt` (commit text and SHA-256).
2. Static: the syntax tree of `solve` declares each car's matrix as free variables with no bound
   on either side, one for every pair of an output step and a base-set step, and nothing in the
   file restricts it to a triangle or a diagonal.
3. Behavioural: on a made-up fleet of three cars, programme (24) returns matrices with entries
   above the diagonal, and the cars' power differs when those entries are removed.

Negative tests build modified copies of the file in a temporary folder (matrix made lower-
triangular in three different ways) and check that the checks above reject each copy. The
static check misses the third copy and the behavioural check catches it, which is why both run.

If the earlier file is missing, this file ends the run with status 2 (could not run); see
`tests/conftest.py` for how a whole-suite run treats it.
"""

import ast
import hashlib
import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

TOP = Path(__file__).resolve().parents[2]
EARLIER = TOP / "reference" / "part1"
OFFER_PATH = EARLIER / "src" / "fleetoffer" / "offer.py"
PROVENANCE_PATH = EARLIER / "PROVENANCE.txt"

# SHA-256 of the exported file (commit 05a6cf9 of the earlier analysis). A changed file fails.
OFFER_SHA256 = "89086786ad4f08ad57b7f2e79568b1c71bf1dd046ae36ff74ac6995e97feff45"
PROVENANCE_TEXT = ("exported unchanged", "commit 05a6cf9")

if not OFFER_PATH.is_file():
    pytest.exit(
        "COULD NOT RUN: the earlier analysis's offer code is not in this checkout "
        "(reference/part1/src/fleetoffer/offer.py)",
        returncode=2,
    )

# Names that would restrict a matrix to a triangle or a diagonal.
TRIANGLE_NAMES = {
    "tri",
    "tril",
    "triu",
    "tril_indices",
    "triu_indices",
    "tril_indices_from",
    "triu_indices_from",
    "diag",
    "diagonal",
    "diag_indices",
    "diag_indices_from",
    "fill_diagonal",
}
# A made-up fleet: deadlines (steps), energy ceilings (kWh), four steps of a quarter hour.
FLEET = {"deadline": [2, 4, 3], "energy": [1.0, 2.0, 1.5], "steps": 4}
# The smallest effect of the above-diagonal entries that counts as "matters", in kW. The solved
# value on this fleet is about 1.4 (and 0 for a triangular copy).
MIN_UPPER_EFFECT_KW = 0.1


# ---- the checks, written to run on any copy of the file --------------------------------------


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_np_inf(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "inf"
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
    )


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def static_findings(source: str) -> list[str]:
    """What is wrong with the claim 'each car's matrix is full', read from the syntax tree."""
    tree = ast.parse(source)
    out: list[str] = []
    solve = _function(tree, "solve")
    if solve is None:
        return ["no function `solve`"]

    # 1. The matrix variables: add_var(m * q, -inf, inf), the entries Gamma_i[a, cols[b]] for
    # every output step a in 0..m-1 and base-set step b in 0..q-1, free in both directions.
    declared = [
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Assign)
        and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["big0"]
    ]
    if len(declared) != 1:
        out.append(f"{len(declared)} declarations of `big0`, expected 1")
    for node in declared:
        call = node.value
        ok = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "add_var"
            and len(call.args) == 3
            and not call.keywords
        )
        if ok:
            count, low, high = call.args
            ok = (
                isinstance(count, ast.BinOp)
                and isinstance(count.op, ast.Mult)
                and [_names(count.left), _names(count.right)] == [{"m"}, {"q"}]
                and isinstance(low, ast.UnaryOp)
                and isinstance(low.op, ast.USub)
                and _is_np_inf(low.operand)
                and _is_np_inf(high)
            )
        if not ok:
            out.append("`big0` is not add_var(m * q, -np.inf, np.inf): matrix not free")

    # 2. Nothing in the file restricts a matrix to a triangle or a diagonal.
    for node in ast.walk(tree):
        found = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else None
        )
        if found in TRIANGLE_NAMES:
            out.append(f"line {node.lineno}: `{found}` restricts a matrix to a triangle")

    # 3. In `solve`, no condition looks at the row step `a` or the column step `b` of a matrix
    # entry (a condition such as `cols[b] > a` is how an entry would be left out).
    for node in ast.walk(solve):
        tests = []
        if isinstance(node, ast.If | ast.IfExp | ast.While):
            tests.append(node.test)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)
        elif isinstance(node, ast.Compare):
            tests.append(node)
        for test in tests:
            finding = f"line {test.lineno}: condition on a matrix row or column step"
            if {"a", "b"} & _names(test) and finding not in out:
                out.append(finding)

    # 4. The result: big_gamma has shape (cars, steps, steps); car i's rows 0..m-1 (up to its
    # deadline) and every column of J are written back, with no mask.
    shape_ok = write_ok = False
    for node in ast.walk(solve):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and target.id == "big_gamma":
            shape_ok = (
                isinstance(value, ast.Call)
                and ast.unparse(value.func) == "np.zeros"
                and ast.unparse(value.args[0]) == "(n, steps, steps)"
            )
        if isinstance(target, ast.Subscript) and ast.unparse(target) == "big_gamma[i, :m, cols]":
            write_ok = True
    if not shape_ok:
        out.append("big_gamma is not np.zeros((n, steps, steps))")
    if not write_ok:
        out.append("big_gamma[i, :m, cols] is not written back as one block")

    # 5. The allocation uses the whole matrix and the whole profile: gamma + big_gamma @ u0.
    allocate = next(
        (
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "Offer"
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "allocate"
        ),
        None,
    )
    returns = [n for n in ast.walk(allocate) if isinstance(n, ast.Return)] if allocate else []
    if len(returns) != 1 or ast.unparse(returns[0].value) != "self.gamma + self.big_gamma @ u0":
        out.append("Offer.allocate does not return self.gamma + self.big_gamma @ u0")
    return out


def load_module(path: Path):
    """Import a copy of the earlier file under a throw-away name (it imports numpy and scipy
    only). The name is registered while the file runs, which its dataclasses need."""
    name = f"earlier_offer_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def upper_effect_kw(module) -> tuple[float, float]:
    """Solve programme (24) on the made-up fleet. Returns two numbers:

    the largest change (kW) in any car's power, over the checked base-set profiles and the
    reversed fill, when the entries of the matrices above the diagonal are set to zero; and the
    largest gap (kW) between the cars' total and the offered profile, which must be about zero.
    """
    fleet = module.Fleet.of(**FLEET, cap=6.7, step_h=0.25)
    base = module.BaseSet.average(fleet)
    offer = module.solve(fleet, base)
    assert offer.status.startswith("optimal"), offer.status
    assert offer.alpha > 0
    steps = FLEET["steps"]
    assert offer.big_gamma.shape == (len(fleet), steps, steps)
    lower = np.tril(np.ones((steps, steps)))
    profiles = list(module.checked_profiles(base, checkpoints=(1, 2, 3)).values())
    profiles.append(base.fill(np.arange(steps)[::-1]))
    effect = gap = 0.0
    for u0 in profiles:
        full = offer.allocate(u0)
        causal = offer.gamma + (offer.big_gamma * lower) @ u0
        effect = max(effect, float(np.abs(full - causal).max()))
        gap = max(gap, float(np.abs(full.sum(axis=0) - offer.profile(u0)).max()))
    return effect, gap


def behaviour_findings(module) -> list[str]:
    effect, gap = upper_effect_kw(module)
    out = []
    if effect <= MIN_UPPER_EFFECT_KW:
        out.append(f"entries above the diagonal change a car's power by only {effect:.3g} kW")
    if gap > 1e-6:
        out.append(f"the cars' total misses the offered profile by {gap:.3g} kW")
    return out


# ---- the earlier file, as exported ---------------------------------------------------------


def test_file_is_the_stated_export():
    provenance = PROVENANCE_PATH.read_text()
    for text in PROVENANCE_TEXT:
        assert text in provenance
    assert hashlib.sha256(OFFER_PATH.read_bytes()).hexdigest() == OFFER_SHA256


def test_matrices_are_free_in_the_syntax_tree():
    assert static_findings(OFFER_PATH.read_text()) == []


def test_solved_matrices_have_entries_above_the_diagonal_that_matter():
    module = load_module(OFFER_PATH)
    assert behaviour_findings(module) == []
    effect, _ = upper_effect_kw(module)
    assert effect > MIN_UPPER_EFFECT_KW


# ---- copies with the matrix made lower-triangular: each must be rejected -----------------------

DECLARATION = (
    "        big0 = add_var(m * q, -np.inf, np.inf)  # Gamma_i[a, cols[b]] at big0 + a q + b\n"
)
WRITE_BACK = "        big_gamma[i, :m, cols] = x[big0 : big0 + m * q].reshape(m, q).T\n"

MUTANTS = {
    # Entries above the diagonal fixed at zero through the variable bounds.
    "bounds": (
        DECLARATION,
        DECLARATION
        + "        for a in range(m):\n"
        + "            for b in range(q):\n"
        + "                if cols[b] > a:\n"
        + "                    lower[big0 + a * q + b] = 0.0\n"
        + "                    upper[big0 + a * q + b] = 0.0\n",
    ),
    # The solved matrix cut to its lower triangle afterwards.
    "tril": (
        WRITE_BACK,
        WRITE_BACK + "        big_gamma[i] = np.tril(big_gamma[i])\n",
    ),
    # The same cut written without any triangle function or condition on a loop variable.
    "mask": (
        WRITE_BACK,
        WRITE_BACK
        + "        big_gamma[i] *= np.arange(steps)[:, None] >= np.arange(steps)[None, :]\n",
    ),
}


@pytest.fixture
def mutant_path(request, tmp_path):
    old, new = MUTANTS[request.param]
    source = OFFER_PATH.read_text()
    assert source.count(old) == 1, "the line to change is not in the earlier file"
    path = tmp_path / f"offer_{request.param}.py"
    path.write_text(source.replace(old, new))
    return request.param, path


@pytest.mark.parametrize("mutant_path", list(MUTANTS), indirect=True)
def test_triangular_copy_is_rejected(mutant_path):
    name, path = mutant_path
    static = static_findings(path.read_text())
    behaviour = behaviour_findings(load_module(path))
    assert static or behaviour, f"copy '{name}' was accepted"
    # Each check has its own copy it must catch on its own.
    if name in ("bounds", "tril"):
        assert static
    if name in ("bounds", "tril", "mask"):
        assert behaviour
    assert (name == "mask") == (not static)
