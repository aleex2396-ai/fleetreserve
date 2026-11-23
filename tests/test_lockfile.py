"""The dependency groups, the lock file and the default install agree, and stay that way.

`uv lock --check` already fails when the lock file is out of date with `pyproject.toml`. These
tests add what it does not say: every package the plan needs is named, the lock holds each one at
a version its range allows, the heavy groups stay out of the default install, and the core
packages load. The comparison is a function over the two files' text, so the tests can also
run it on damaged copies and see it report the damage.
"""

import importlib
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]

# Packages the plan's work packages need (plan section in the comment), by group.
# "" is the project's own dependency list.
PLAN_PACKAGES = {
    "": {"numpy", "scipy", "pandas"},  # sections 2-6, 13 (HiGHS), 3 (loaders)
    "learn": {"torch", "lightgbm", "scikit-learn", "holidays", "lifelines"},  # section 7
    "foundation": {"chronos-forecasting", "tabpfn"},  # section 7, Chronos-2 and TabPFN v2
    "language": {"anthropic"},  # section 8
    "figures": {"matplotlib"},  # section 13, WP16
    "baselines": {"acnportal"},  # section 10, ASA
}
HEAVY_GROUPS = {"learn", "foundation", "language", "figures", "baselines"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_problems(pyproject_text: str, lock_text: str) -> list[str]:
    """Every disagreement between the two files, one line each; empty when they agree."""
    project = tomllib.loads(pyproject_text)
    lock = tomllib.loads(lock_text)
    locked = {norm(p["name"]): p for p in lock["package"]}
    own = locked.get(norm(project["project"]["name"]))
    if own is None:
        return ["the project itself is missing from the lock file"]

    groups = {"": project["project"]["dependencies"], **project["dependency-groups"]}
    lock_groups = {"": [d["name"] for d in own.get("dependencies", [])]}
    for group, deps in own.get("dev-dependencies", {}).items():
        lock_groups[group] = [d["name"] for d in deps]

    problems = []
    for group, specs in groups.items():
        label = group or "project dependencies"
        if group not in lock_groups:
            problems.append(f"{label}: group is missing from the lock file")
        for spec in specs:
            req = Requirement(spec)
            name = norm(req.name)
            if name not in locked:
                problems.append(f"{label}: {name} is not in the lock file")
                continue
            if name not in {norm(n) for n in lock_groups.get(group, [])}:
                problems.append(f"{label}: {name} is in the lock file but not in this group there")
            version = Version(locked[name]["version"])
            if not req.specifier.contains(version):
                problems.append(f"{label}: locked {name} {version} is outside '{req.specifier}'")
    for group in lock_groups.keys() - groups.keys():
        problems.append(f"{group or 'project dependencies'}: in the lock file but not in pyproject")
    return problems


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_lock_file_agrees_with_pyproject():
    assert lock_problems(read("pyproject.toml"), read("uv.lock")) == []


def test_plan_packages_are_named_in_their_groups():
    project = tomllib.loads(read("pyproject.toml"))
    groups = {"": project["project"]["dependencies"], **project["dependency-groups"]}
    for group, needed in PLAN_PACKAGES.items():
        named = {norm(Requirement(spec).name) for spec in groups[group]}
        assert needed <= named, f"group '{group}' lacks {sorted(needed - named)}"


def test_acnportal_is_pinned_to_the_version_the_plan_names():
    project = tomllib.loads(read("pyproject.toml"))
    specs = [Requirement(s) for s in project["dependency-groups"]["baselines"]]
    (acn,) = [r for r in specs if norm(r.name) == "acnportal"]
    assert str(acn.specifier) == "==0.3.3"


def test_only_dev_installs_by_default():
    uv = tomllib.loads(read("pyproject.toml"))["tool"]["uv"]
    assert uv["default-groups"] == ["dev"]
    assert not HEAVY_GROUPS & set(uv["default-groups"])


def test_the_heavy_packages_are_not_project_dependencies():
    project = tomllib.loads(read("pyproject.toml"))
    core = {norm(Requirement(s).name) for s in project["project"]["dependencies"]}
    heavy = set().union(*(PLAN_PACKAGES[g] for g in HEAVY_GROUPS))
    assert not core & heavy


@pytest.mark.parametrize("module", ["numpy", "scipy.optimize", "pandas"])
def test_core_packages_import(module):
    importlib.import_module(module)


def test_a_package_missing_from_the_lock_is_reported():
    lock = read("uv.lock")
    block = re.search(r'\[\[package\]\]\nname = "torch"\n.*?(?=\n\[\[package\]\])', lock, re.S)
    assert block is not None
    damaged = lock.replace(block.group(0) + "\n", "")
    assert damaged != lock
    assert "learn: torch is not in the lock file" in lock_problems(read("pyproject.toml"), damaged)


def test_a_dropped_group_is_reported():
    pyproject = read("pyproject.toml")
    damaged = pyproject.replace('language = ["anthropic>=2.8,<3"]\n', "")
    assert damaged != pyproject
    problems = lock_problems(damaged, read("uv.lock"))
    assert "language: in the lock file but not in pyproject" in problems


def test_a_locked_version_outside_its_range_is_reported():
    pyproject = read("pyproject.toml")
    damaged = pyproject.replace('"scipy>=1.18,<2"', '"scipy>=9,<10"')
    assert damaged != pyproject
    problems = lock_problems(damaged, read("uv.lock"))
    assert any(p.startswith("project dependencies: locked scipy") for p in problems)
