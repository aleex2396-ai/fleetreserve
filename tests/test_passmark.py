"""The pass-mark runner `passmark.sh`: one result per listed item, a count line, one exit status.

Most tests run copies of `passmark.sh` and `tools/check_lib.sh` in a temporary folder with a
made-up list, so they do not depend on which real items exist. Two tests read the real
`passmarks/wp1.sh` and `passmarks/wp2.sh` and compare them with the items of each work package's
pass mark written out below. The real runner is never started from here.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

TOP = Path(__file__).resolve().parents[1]
RESULTS_NAME = "passmark-demo.txt"

# The twelve items of the WP1 pass mark (plan section 13) in the plan's order: name, command.
PYTEST = "uv run --frozen pytest -c pyproject.toml"
WP1_ITEMS = [
    (
        "hashes committed (plan 13 WP1)",
        "uv run --frozen python tools/data_hashes.py --check",
    ),
    (
        "T1 one-way box check against vertex LPs, 400 of 400 (plan 4)",
        f"{PYTEST} tests/wp1/test_t1_one_way.py",
    ),
    (
        "T1 two-way box check against vertex LPs, 300 of 300 (plan 4)",
        f"{PYTEST} tests/wp1/test_t1_two_way.py",
    ),
    (
        "T1 floors and free recovery hours, 600 of 600 (plan 4)",
        f"{PYTEST} tests/wp1/test_t1_floors_recovery.py",
    ),
    (
        "T1 band scale (c), 400 of 400 (plan 4)",
        f"{PYTEST} tests/wp1/test_t1_band.py",
    ),
    (
        "T3 one-car two-hour example: hindsight 0.10, upper-corner path 0.15 (plan 4)",
        f"{PYTEST} tests/wp1/test_t3_example.py",
    ),
    (
        "T4 proposition on toy fleets, 300 of 300 (plan 4)",
        f"{PYTEST} tests/wp1/test_t4_proposition.py",
    ),
    (
        "T4 conjecture: no gap on 22400 seeded toy fleets (plan 4)",
        f"{PYTEST} tests/wp1/test_t4_conjecture.py",
    ),
    (
        "T4 gap example: hindsight 1.0, tree and affine 0.556 (plan 4)",
        f"{PYTEST} tests/wp1/test_t4_gap_example.py",
    ),
    (
        "T5 discharge-after-failure case: 0.5 kWh two-way, 1.0 under the plan (plan 5)",
        f"{PYTEST} tests/wp1/test_t5_case.py",
    ),
    (
        "T10 drift-guard cases: clipped 0.500, unclipped 0.103 at t=116 (plan 6)",
        f"{PYTEST} tests/wp1/test_t10_cases.py",
    ),
    (
        "full per-car matrices confirmed in the earlier analysis's offer code (plan 13 WP1)",
        f"{PYTEST} tests/wp1/test_earlier_offer_matrices.py",
    ),
]

# The three items of the WP2 pass mark (plan section 13: "training and development truth sets;
# fallback shares logged") in the plan's order: name, command.
VERIFY = "uv run --frozen python -m fleetreserve.truth_verify"
WP2_ITEMS = [
    ("training truth sets built and consistent (plan 13 WP2)", f"{VERIFY} train"),
    ("development truth sets built and consistent (plan 13 WP2)", f"{VERIFY} development"),
    (
        "fallback shares logged for every unit (plan 13 WP2)",
        "uv run --frozen python tools/wp2_fallback_check.py",
    ),
]


@pytest.fixture
def sandbox(tmp_path):
    """A folder holding copies of the runner and its library, and a `passmarks` folder."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "passmarks").mkdir()
    shutil.copy(TOP / "passmark.sh", tmp_path / "passmark.sh")
    shutil.copy(TOP / "tools" / "check_lib.sh", tmp_path / "tools" / "check_lib.sh")
    return tmp_path


def write_list(sandbox: Path, lines: list[str], name: str = "demo") -> None:
    (sandbox / "passmarks" / f"{name}.sh").write_text("\n".join(lines) + "\n")


def run(sandbox: Path, *args: str, run_dir: Path | None = None):
    environment = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "")}
    if run_dir is not None:
        environment["FLEETRESERVE_RUN_DIR"] = str(run_dir)
    done = subprocess.run(
        ["bash", str(sandbox / "passmark.sh"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
        cwd=sandbox.parent,  # not the sandbox: the runner must change to its own folder
        check=False,
    )
    return done


def test_all_items_pass(sandbox):
    write_list(sandbox, ['check "one" true', 'check "two" true'])
    done = run(sandbox, "demo")
    assert done.returncode == 0
    assert done.stdout.splitlines() == [
        "passed: one",
        "passed: two",
        "2 passed, 0 failed, 0 could not run of 2 items",
        "all 2 checks passed",
    ]


def test_one_item_fails(sandbox):
    write_list(sandbox, ['check "one" true', 'check "two" false'])
    done = run(sandbox, "demo")
    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "passed: one",
        "FAILED: two",
        "1 passed, 1 failed, 0 could not run of 2 items",
    ]


def test_missing_file_cannot_run(sandbox):
    write_list(sandbox, ['check "one" true', 'check "two" bash tests/no_such_file.sh'])
    done = run(sandbox, "demo")
    assert done.returncode == 2
    lines = done.stdout.splitlines()
    assert lines[0] == "passed: one"
    assert lines[-2:] == [
        "COULD NOT RUN: two (exit 127)",
        "1 passed, 0 failed, 1 could not run of 2 items",
    ]


def test_missing_pytest_style_exit_status_cannot_run(sandbox):
    # pytest ends with status 4 for a file that is not there and 5 for no tests: both are "could
    # not run", never a pass.
    write_list(sandbox, ['check "one" bash -c "exit 4"', 'check "two" bash -c "exit 5"'])
    done = run(sandbox, "demo")
    assert done.returncode == 2
    assert done.stdout.splitlines() == [
        "COULD NOT RUN: one (exit 4)",
        "COULD NOT RUN: two (exit 5)",
        "0 passed, 0 failed, 2 could not run of 2 items",
    ]


def test_fail_and_cannot_run_together_is_failed(sandbox):
    write_list(
        sandbox,
        ['check "one" bash -c "exit 2"', 'check "two" false', 'check "three" true'],
    )
    done = run(sandbox, "demo")
    assert done.returncode == 1
    assert done.stdout.splitlines() == [
        "COULD NOT RUN: one (exit 2)",
        "FAILED: two",
        "passed: three",
        "1 passed, 1 failed, 1 could not run of 3 items",
    ]


def test_empty_list_cannot_run(sandbox):
    write_list(sandbox, ["# no items yet"])
    done = run(sandbox, "demo")
    assert done.returncode == 2
    lines = done.stdout.splitlines()
    assert lines[-1] == "0 passed, 0 failed, 0 could not run of 0 items"
    assert any(line.startswith("COULD NOT RUN:") for line in lines)


def test_item_output_is_kept_and_counted_even_without_a_final_newline(sandbox):
    write_list(
        sandbox,
        ["check \"one\" bash -c 'printf partial'", "check \"two\" bash -c 'echo hello; exit 1'"],
    )
    done = run(sandbox, "demo")
    assert done.returncode == 1
    lines = done.stdout.splitlines()
    assert "hello" in lines
    assert lines[-1] == "1 passed, 1 failed, 0 could not run of 2 items"


def test_no_argument_prints_usage_and_cannot_run(sandbox):
    write_list(sandbox, ['check "one" true'])
    done = run(sandbox)
    assert done.returncode == 2
    assert "usage" in (done.stdout + done.stderr).lower()
    assert "passed:" not in done.stdout


def test_unknown_name_prints_usage_and_cannot_run(sandbox):
    write_list(sandbox, ['check "one" true'])
    done = run(sandbox, "nosuch")
    assert done.returncode == 2
    assert "usage" in (done.stdout + done.stderr).lower()
    assert "passed:" not in done.stdout


def test_name_with_path_parts_is_unknown(sandbox):
    write_list(sandbox, ['check "one" true'])
    (sandbox / "other.sh").write_text('check "escaped" true\n')
    done = run(sandbox, "../other")
    assert done.returncode == 2
    assert "escaped" not in done.stdout


def test_run_directory_set_writes_results_file(sandbox, tmp_path_factory):
    write_list(sandbox, ['check "one" true', 'check "two" false', 'check "three" bash -c "exit 2"'])
    run_dir = tmp_path_factory.mktemp("run")
    done = run(sandbox, "demo", run_dir=run_dir)
    assert done.returncode == 1
    written = (run_dir / RESULTS_NAME).read_text().splitlines()
    assert written == [
        "passed: one",
        "FAILED: two",
        "COULD NOT RUN: three (exit 2)",
        "1 passed, 1 failed, 1 could not run of 3 items",
    ]
    assert written == done.stdout.splitlines()


def test_run_directory_set_writes_results_file_for_empty_list(sandbox, tmp_path_factory):
    write_list(sandbox, ["# nothing"])
    run_dir = tmp_path_factory.mktemp("run")
    done = run(sandbox, "demo", run_dir=run_dir)
    assert done.returncode == 2
    written = (run_dir / RESULTS_NAME).read_text().splitlines()
    assert written == done.stdout.splitlines()
    assert written[-1] == "0 passed, 0 failed, 0 could not run of 0 items"


def test_run_directory_not_set_writes_no_file(sandbox):
    write_list(sandbox, ['check "one" true'])
    done = run(sandbox, "demo")
    assert done.returncode == 0
    assert sorted(path.name for path in sandbox.iterdir()) == ["passmark.sh", "passmarks", "tools"]
    assert not list(sandbox.glob("passmark-*.txt"))


def test_run_directory_missing_cannot_run(sandbox, tmp_path_factory):
    write_list(sandbox, ['check "one" true'])
    missing = tmp_path_factory.mktemp("run") / "not_made"
    done = run(sandbox, "demo", run_dir=missing)
    assert done.returncode == 2


def _listed_items(name: str) -> list[tuple[str, str]]:
    text = (TOP / "passmarks" / f"{name}.sh").read_text()
    code_lines = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    found = []
    for line in code_lines:
        match = re.fullmatch(r'check "([^"]+)" (.+)', line)
        assert match, f"not a check line: {line!r}"
        found.append((match.group(1), match.group(2)))
    return found


def test_real_wp1_list_is_the_twelve_items_of_the_pass_mark():
    found = _listed_items("wp1")
    assert len(found) == 12
    assert found == WP1_ITEMS


def test_real_wp2_list_is_the_three_items_of_the_pass_mark():
    found = _listed_items("wp2")
    assert len(found) == 3
    assert found == WP2_ITEMS


# The three items of the WP3 pass mark (plan section 13: "45.6 kWh ± 0.1 kWh, 0.4% and 3.9% ± 0.1
# point") in the plan's order: name, command.
WP3_ITEMS = [
    (
        "re-run perfect-knowledge offer is 45.6 kWh, 0.1 kWh either way (plan 13 WP3)",
        "uv run --frozen python tools/earlier_rerun_check.py --kwh",
    ),
    (
        "re-run learned share is 0.4%, 0.1 point either way (plan 13 WP3)",
        "uv run --frozen python tools/earlier_rerun_check.py --learned",
    ),
    (
        "re-run population share is 3.9%, 0.1 point either way (plan 13 WP3)",
        "uv run --frozen python tools/earlier_rerun_check.py --population",
    ),
]


def test_real_wp3_list_is_the_three_items_of_the_pass_mark():
    text = (TOP / "passmarks" / "wp3.sh").read_text()
    code_lines = [
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    ]
    found = []
    for line in code_lines:
        match = re.fullmatch(r'check "([^"]+)" (.+)', line)
        assert match, f"not a check line: {line!r}"
        found.append((match.group(1), match.group(2)))
    assert len(found) == 3
    assert found == WP3_ITEMS


# The ten items of the WP4 pass mark (plan section 13: "T1, flexitroid, Frank 2011 projection
# check, timing. Pass: T1's checks; solve times of the box LP and the tree (100 cars, 8 hours) and
# of LP (24) (50 cars, 14 hours) logged with the machine they ran on") in the plan's order: name,
# command.
TIMING = "uv run --frozen python tools/timing_marks_check.py"
WP4_ITEMS = [
    *WP1_ITEMS[1:5],
    (
        "T1 on the day model: box check, scale and ceilings from a day's set functions against "
        "linear programmes (plan 2, 4)",
        f"{PYTEST} tests/wp4/test_t1_exact.py tests/wp4/test_ceilings.py",
    ),
    (
        "flexitroid matches T1(b) on 50 development days (plan 4)",
        "uv run --frozen python tools/flexitroid_check.py",
    ),
    (
        "Frank 2011 projection check: the fleet set on S has only the faces p(A) <= u(A) <= b(A) "
        "(plan 4)",
        f"{PYTEST} tests/wp4/test_projection.py",
    ),
    (
        "box LP solve times (100 cars, 8 hours) logged with the machine they ran on (plan 13 WP4)",
        f"{TIMING} --box",
    ),
    (
        "tree solve times (100 cars, 8 hours) logged with the machine they ran on (plan 13 WP4)",
        f"{TIMING} --tree",
    ),
    (
        "LP (24) solve times (50 cars, 14 hours) logged with the machine they ran on (plan 13 WP4)",
        f"{TIMING} --lp24",
    ),
]


def test_real_wp4_list_is_the_ten_items_of_the_pass_mark():
    found = _listed_items("wp4")
    assert len(found) == 10
    assert found == WP4_ITEMS


# The four items of the WP5 pass mark (plan section 13: "diag <= affine <= tree <= hindsight every
# development day; T2's check; two-way corner tests match the LP; T4 log started, split by
# in-window departure") in the plan's order: name, command. In the T2 configuration, rate and
# energy known was decided by the project owner on 16 Nov 2026 and a car with no usable forecast
# left unsteered on 18 Nov 2026; kappa 0.1 and the departure quantile as planned are this check's
# stated choice (the kappa for the gate is the plan's rule, section 7; the departure floor has no
# recorded decision); the sample size 100 is the plan's (section 5).
DEV = "uv run --frozen python tools/dev_day_check.py"
WP5_ITEMS = [
    (
        "diag <= affine <= tree <= hindsight on every development day, hourly and block "
        "products (plan 13 WP5)",
        f"{DEV} --order",
    ),
    (
        "T2's check on 100 development days: closed-form worst case equals replay of all corner "
        "paths, no violation where E' holds, share of days in E' shown (plan 5)",
        f"{DEV} --t2 --sample-size 100 --energy-bounds known --uncertifiable-car "
        "leave-unsteered --kappa 0.1 --departure-floor as-planned",
    ),
    (
        "two-way corner tests match the LP at B = 5 kWh on 100 development days (plan 9, 13 WP5)",
        f"{DEV} --two-way --sample-size 100",
    ),
    (
        "T4 log started on every development day, split by in-window departure (plan 4)",
        f"{DEV} --t4-log",
    ),
]


def test_real_wp5_list_is_the_four_items_of_the_pass_mark():
    found = _listed_items("wp5")
    assert len(found) == 4
    assert found == WP5_ITEMS


# The one item of the WP6 pass mark (plan section 13: "T5 on 100% of day-paths"): name, command.
WP6_ITEMS = [
    (
        "T5 on 100% of day-paths in the recorded gate run (plan 13 WP6)",
        "uv run --frozen python tools/gate_t5_check.py",
    ),
]


def test_real_wp6_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp6")
    assert len(found) == 1
    assert found == WP6_ITEMS


# The one item of the WP8 pass mark (plan section 13: "development pinball loss, concordance index
# and offer (WP6 calibration) logged per model"): name, command. The models are the list held in
# tools/learn_logged_check.py, pinned by tests/wp8/test_learn_logged_check.py.
WP8_ITEMS = [
    (
        "development pinball loss, concordance index and offer (WP6 calibration) logged per model "
        "of plan section 7 in its recorded learn run (plan 13 WP8)",
        "uv run --frozen python tools/learn_logged_check.py",
    ),
]


def test_real_wp8_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp8")
    assert len(found) == 1
    assert found == WP8_ITEMS


# The one item of the WP10 pass mark (plan section 13: "Baselines, no-pause sensitivity. Pass:
# shared offers, paths, output files"): name, command. The methods, models and runs it requires are
# those of tools/compare_shared_check.py, pinned by tests/wp10/test_compare_shared_check.py.
WP10_ITEMS = [
    (
        "baselines and comparison methods on shared offers, paths and output files, with the "
        "no-pause sensitivity, in the recorded comparison runs (plan 13 WP10)",
        "uv run --frozen python tools/compare_shared_check.py",
    ),
]


def test_real_wp10_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp10")
    assert len(found) == 1
    assert found == WP10_ITEMS


# The two items of the WP11 pass mark, one per clause of plan section 13's WP11 line ("closed-form
# worst case reproduced on 20 days; run matrix frozen"), in its order: name, command. The checks
# are tools/corner_closed_form_check.py, pinned by tests/wp11/test_corner_closed_form_check.py,
# and tools/freeze_check.py matrix, pinned by tests/wp13/test_freeze_items.py.
WP11_ITEMS = [
    (
        "fixed plan's closed-form worst case reproduced on 20 days in the recorded corner search "
        "(plan 13 WP11)",
        "uv run --frozen python tools/corner_closed_form_check.py",
    ),
    (
        "run matrix frozen: the committed run list run-lists/run-list.json and every matrix it "
        "names, committed and unchanged at HEAD, each what sealed_matrix writes from its verified "
        "kappa-choice results, the run list's dials the main model's (plan 13 WP11)",
        "uv run --frozen python tools/freeze_check.py matrix",
    ),
]


def test_real_wp11_list_is_the_two_items_of_the_pass_mark():
    found = _listed_items("wp11")
    assert len(found) == 2
    assert found == WP11_ITEMS


# The one item of the WP12 pass mark (plan section 13: "Dress rehearsal of held-out scripts on
# development units. Pass: done, its duration logged"): name, command. The check is
# tools/rehearsal_check.py, pinned by tests/wp12/test_rehearsal_check.py.
WP12_ITEMS = [
    (
        "dress rehearsal of the held-out scripts on development units done, its duration logged, "
        "in the driver log of the rehearsal of the committed run list run-lists/run-list.json, "
        "from which the run from the tag is derived (plan 13 WP12; the differences between the "
        "two are listed, not judged)",
        "uv run --frozen python tools/rehearsal_check.py",
    ),
]


def test_real_wp12_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp12")
    assert len(found) == 1
    assert found == WP12_ITEMS


# The driver command of the rehearsal that writes the log the item reads, as passmarks/wp12.sh
# and the check's module text write it: with the learn and baselines groups, so the start check
# of the rehearsal's learning steps can load their models (the owner's decision of 19 November
# 2026; what counts as a pass is unchanged).
WP12_DRIVER = (
    "caffeinate -i uv run --frozen --group learn --group baselines python -m "
    "fleetreserve.run_driver run-lists/run-list.json --purpose rehearsal"
)


def _one_line(text: str) -> str:
    """`text` with each line's leading comment mark dropped and every run of spaces made one."""
    return " ".join(" ".join(line.strip().removeprefix("#") for line in text.splitlines()).split())


@pytest.mark.parametrize("path", ["passmarks/wp12.sh", "tools/rehearsal_check.py"])
def test_the_wp12_driver_command_carries_the_learn_and_baselines_groups(path):
    """Each file writes the driver command at least once, and every driver command of the
    rehearsal it writes is the one with both groups."""
    text = _one_line((TOP / path).read_text())
    assert WP12_DRIVER in text, path
    assert text.count("--purpose rehearsal") == text.count(WP12_DRIVER), path


# The four items of the WP9 pass mark, one per clause of plan section 13's WP9 line, in its
# order: name, command. The check is tools/wp9_check.py, pinned by the tests in tests/wp9/ named
# test_wp9_*.py and by tests/learn/test_wp9_check_names.py.
WP9_ITEMS = [
    (
        "loss 0 at abstain in both tiers, in the recorded gate run (plan 13 WP9)",
        "uv run --frozen python tools/wp9_check.py abstain",
    ),
    (
        "development loss within its 200-split spread: the mean test loss over the 200 "
        "development splits at most the level plus two standard errors, slot loss in both tiers "
        "and driver loss in tier A, per gate unit, in the recorded comparison runs (plan 13 WP9)",
        "uv run --frozen python tools/wp9_check.py spread",
    ),
    (
        "T8's check in the recorded soft-scale checks: a passed comparison on 100 days in every "
        "direction, no failed comparison, an MPS batch timed, tracking logged (plan 13 WP9)",
        "uv run --frozen python tools/wp9_check.py t8",
    ),
    (
        "conformal training kept or not kept, decided from the recorded sizing comparison on the "
        "development units (plan 13 WP9: kept only if it raises the development offer)",
        "uv run --frozen python tools/wp9_check.py conformal",
    ),
]


def test_real_wp9_list_is_the_four_items_of_the_pass_mark():
    found = _listed_items("wp9")
    assert len(found) == 4
    assert found == WP9_ITEMS


# The four items of the WP13 pass mark, one per clause of plan section 13's WP13 line ("tag
# exists, and is the commit checked out; held-out hashes match those committed at download and
# split; held-out list, run matrix, seeds; no opening logged"), in its order: name, command. The
# check is tools/freeze_check.py, pinned by tests/wp13/test_freeze_check.py and, for the third
# item, tests/wp13/test_freeze_items.py.
FREEZE = "uv run --frozen python tools/freeze_check.py"
WP13_ITEMS = [
    (
        "tag exists: refs/tags/frozen-plan names a commit, and HEAD is that commit in a clean "
        "tree (plan 13 WP13)",
        f"{FREEZE} tag",
    ),
    (
        "held-out hashes match those committed at download and split: the held-out entries of "
        "data-hashes.json unchanged on HEAD's first-parent path since the commit that first holds "
        "it, and the sealed files of those sizes and SHA-256 hashes, with no other file in the "
        "sealed folder outside any folder in it that is a symbolic link (plan 13 WP13)",
        f"{FREEZE} hashes",
    ),
    (
        "held-out list, run matrix, seeds: the nine held-out units of "
        "sealed_route.HELD_OUT_UNITS exactly the gate units of every sealed matrix and the units "
        "of the run list's sealed-run groups; the run matrix frozen as WP11's item checks it; "
        "every seed of the run list's entries one of its fixed seeds, none literal or given "
        "twice, and none left at a command's default unless that command's own parser or check "
        "refuses it on that step (the owner's decision of 19 November 2026), and the run list's "
        "own seed rules (L12, L13) holding (plan 13 WP13)",
        f"{FREEZE} lists",
    ),
    (
        "no opening logged: the openings log absent or empty, with the sealed files present in "
        "this working copy (plan 13 WP13; passes only before the run from the tag)",
        f"{FREEZE} openings",
    ),
]


def test_real_wp13_list_is_the_four_items_of_the_pass_mark():
    found = _listed_items("wp13")
    assert len(found) == 4
    assert found == WP13_ITEMS


# The one item of the WP15 pass mark, plan section 13's WP15 line ("Sensitivities: rho 0.1, 0.2
# for the main model only; chronological calibration (T10, at OSL_T); regimes (no split beyond the
# two a.s.r. units). Pass: all reported."), read by the readings W1 to W3 that the committed run
# list states in its about: name, command. The check is tools/wp15_check.py, pinned by
# tests/wp15/test_wp15_check.py.
WP15_ITEMS = [
    (
        "sensitivities all reported, from the run from the tag by the readings W1 to W3 that the "
        "committed run list states in its about: rho 0.1 and 0.2 for the main model, its learn "
        "and interval entries at each rho holding the offer and both ceilings; the chronological "
        "calibration, the drift guard's entry on OSL_T alone holding its per-day losses and "
        "offers; the regimes, the two a.s.r. units scored as units of their own by the main "
        "model's learn entry (plan 13 WP15)",
        "uv run --frozen python tools/wp15_check.py",
    ),
]


def test_real_wp15_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp15")
    assert len(found) == 1
    assert found == WP15_ITEMS


# The one item of the WP14 pass mark, plan section 13's WP14 line whole ("one logged run plus
# logged parsing fixes, the scoring runs made again at a fix commit belonging to that one run; logs
# hashed"): name, command. The check is tools/sealed_run_check.py, pinned by
# tests/wp14/test_sealed_run_check.py; the command passes the run recorder's folder on.
WP14_ITEMS = [
    (
        "one logged run plus logged parsing fixes, the scoring runs made again at a fix commit "
        "belonging to that one run; logs hashed: exactly one complete sealed-run drive of the "
        "committed run list at the checked-out commit (the tag's or a logged parsing fix the gate "
        "accepts), every opening inside a step, every earlier drive that ran a step an operational "
        "stop with no opening or stopped at a logged layout refusal, every run's manifest and "
        "log.txt hashed as logged; results in wp14-passmark.json (plan 13 WP14)",
        'env FLEETRESERVE_RUN_DIR="${FLEETRESERVE_RUN_DIR-}" uv run --frozen python '
        "tools/sealed_run_check.py",
    ),
]


def test_real_wp14_list_is_the_one_item_of_the_pass_mark():
    found = _listed_items("wp14")
    assert len(found) == 1
    assert found == WP14_ITEMS
