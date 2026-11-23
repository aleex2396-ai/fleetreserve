"""The sealed run's rules of the run list (`run_list`, module text, THE SEALED RUN'S RULES), on
made-up lists, by the checker's command line and by the driver (`run_driver`) in a throwaway git
repository (`drive_fake`), whichever purpose is driven.

1. An honest list (the sealed truth build first, each sealed scoring entry naming it by
   `{run:<label>}`, the interval seed carried, the nine held-out units scored in one `learn_run`
   whose run one `day_intervals` entry reads) is well formed (exit 0) and drives each purpose to
   its end (exit 0).
2. Rule 1, truth builds first: a literal path as the sealed run's `--held-out-truth-run`; a truth
   build after the first entry that reads the sealed units' tables (a second build, of the held
   setting, put after the scoring entry; a build named by an entry that comes after it); a
   shortened option `--held-out-truth`; the option given twice. Each is faulty by the command
   line (exit 1, a line naming the entry and the rule) and refused by the driver for either
   purpose (exit 2, no log, no run).
3. Rule 3, one run for the pooled figure: a `day_intervals` entry that reads its run by a literal
   path; a `learn_run` on the sealed route, read by `day_intervals`, whose matrix scores five of
   the nine held-out units, or the nine and a development unit; the nine split over two
   `learn_run` entries (each named); a `learn_run` naming no matrix; a matrix that cannot be read.
   Each faulty by the command line (exit 1) and refused by the driver (exit 2, no log, no run). A
   `learn_run` off the sealed route is not checked. A list outside any git working copy whose
   rule 3 must read a matrix cannot be decided (exit 2).
4. After a logged parsing fix every entry runs again at the fix commit: with the REAL gate, in a
   throwaway repository at the tag `frozen-plan` (`fake_heldout`), a sealed-run drive runs every
   entry; after a logged refusal, a correction and its entry in section 14, a second drive at the
   fix commit runs every entry again, each in a run folder of its own whose manifest names the
   fix commit, and each `{run:<label>}` is filled with a run of the second drive.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import pytest
from drive_fake import (
    LIST_PATH,
    SEALED_MATRIX,
    SEALED_TRUTH,
    commit_list,
    document,
    drive_logs,
    entry,
    git,
    make_repo,
    no_start_check,
    read_log,
    run_folders,
    step,
    write_matrix,
)

from fleetreserve import cold_start_gate, heldout_truth_build, run_driver, run_list, sealed_route

_WP2 = Path(__file__).resolve().parents[1] / "wp2"  # the made-up held-out repositories
if str(_WP2) not in sys.path:
    sys.path.insert(0, str(_WP2))
import fake_heldout  # noqa: E402

_WP6GATE = Path(__file__).resolve().parents[1] / "wp6gate"  # the watch of sealed files
if str(_WP6GATE) not in sys.path:
    sys.path.insert(0, str(_WP6GATE))
import sealed_watch  # noqa: E402

TOP = Path(__file__).resolve().parents[2]
NINE = list(sealed_route.HELD_OUT_UNITS)
REHEARSE = ["--purpose", "rehearsal"]
SEAL = ["--purpose", "sealed-run"]
SEED = 20261117


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    write_matrix(root, "gate/five.json", NINE[:5])
    write_matrix(root, "gate/four.json", NINE[5:])
    write_matrix(root, "gate/ten.json", [*NINE, "nrel-before-may-2019"])
    write_matrix(root, "gate/development.json", ["nrel-before-may-2019", "nrel-from-may-2019"])
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "made-up matrices")
    no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    monkeypatch.chdir(root)
    return root


def scoring(label: str, matrix: str = SEALED_MATRIX, *, build: str = "sealed-truth") -> dict:
    """A `learn_run` of both purposes: the development matrix in the rehearsal, `matrix` and the
    sealed route in the sealed run; the interval seed carried."""
    return entry(
        label,
        step("learn_run", "--interval-seed", "{seed:intervals}", "--interval-resamples", "2000"),
        after=["wp2-truth"],
        extra={
            "rehearsal": ["--matrix", "learn/matrix.json"],
            "sealed-run": [
                "--matrix",
                matrix,
                "--held-out-tables",
                "data/derived/truth-sealed",
                "--held-out-truth-run",
                f"{{run:{build}}}",
            ],
        },
    )


def intervals(label: str, of: str) -> dict:
    return entry(
        label,
        step("day_intervals", "--run", f"{{run:{of}}}", "--seed", "{seed:intervals}"),
        after=[of],
    )


def honest() -> dict:
    """The honest list of the module text, item 1."""
    return document(
        [
            entry("wp2-truth", step("truth_build")),
            entry("sealed-truth", step(SEALED_TRUTH), only="sealed-run"),
            entry(
                "sealed-truth-held",
                step(SEALED_TRUTH, "--setting", "held"),
                only="sealed-run",
            ),
            scoring("scored-learn"),
            intervals("scored-intervals", "scored-learn"),
        ],
        seeds={"intervals": SEED},
    )


def check(path: Path, capsys) -> tuple[int, list[str]]:
    code = run_list.main([str(path), "--no-starts"])  # stand-in commands: no start check
    return code, capsys.readouterr().out.splitlines()


def passing_gate(monkeypatch) -> None:
    """Make the sealed run's gate pass, as at the tag."""
    monkeypatch.setattr(
        cold_start_gate, "require_gate", lambda root: git(root, "rev-parse", "HEAD").strip()
    )


def logged(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r["record"] == kind]


# --- 1. the honest list --------------------------------------------------------------------------


def test_the_honest_list_is_well_formed(repo, capsys):
    path = commit_list(repo, honest())
    code, lines = check(path, capsys)
    assert code == 0, lines
    assert lines[-1].startswith("well formed: ")


@pytest.mark.parametrize("purpose", [REHEARSE, SEAL], ids=["rehearsal", "sealed-run"])
def test_the_honest_list_drives_to_its_end(repo, monkeypatch, purpose):
    commit_list(repo, honest())
    passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *purpose]) == 0
    (path,) = drive_logs(repo)
    records = read_log(path)
    expected = ["wp2-truth", "scored-learn", "scored-intervals"]
    if purpose == SEAL:
        expected = ["wp2-truth", "sealed-truth", "sealed-truth-held", *expected[1:]]
    assert [r["label"] for r in logged(records, "entry")] == expected
    assert records[-1]["completed"] is True


# --- the faulty lists, each checked by the command line and refused by the driver ----------------


def literal_truth_run(d):
    d["entries"][3]["extra"]["sealed-run"][-1] = "runs.noindex/20261122T000000Z-sealed-truth"


def held_build_after_the_scoring(d):
    # the build of the held setting after the first scoring entry, named by a later one
    held = d["entries"].pop(2)
    d["entries"].insert(3, held)
    d["entries"].insert(4, scoring("scored-learn-held", build="sealed-truth-held"))


def build_named_after_a_reader(d):
    # an entry reads the sealed tables before the build the scoring entry names
    d["entries"].insert(
        1,
        entry(
            "sealed-cold",
            step("learn_run", "--held-out-tables", "data/derived/truth-sealed"),
            only="sealed-run",
        ),
    )


def shortened_truth_run(d):
    d["entries"][3]["extra"]["sealed-run"][4] = "--held-out-truth"


def truth_run_twice(d):
    d["entries"][3]["extra"]["sealed-run"] += ["--held-out-truth-run", "{run:sealed-truth}"]


def build_of_another_module(d):
    d["entries"][1]["command"] = step("compare_run", "--sealed")  # the earlier stand-in


def unnamed_build_after_the_scoring(d):
    d["entries"].append(entry("late-truth", step(SEALED_TRUTH), only="sealed-run"))


def literal_intervals_run(d):
    d["entries"][4]["command"][4] = "runs.noindex/20261122T000000Z-scored-learn"
    d["entries"][4]["after"] = []


def five_of_the_nine(d):
    d["entries"][3]["extra"]["sealed-run"][1] = "gate/five.json"


def the_nine_and_another(d):
    d["entries"][3]["extra"]["sealed-run"][1] = "gate/ten.json"


def nine_split_over_two_runs(d):
    d["entries"][3:5] = [
        scoring("scored-five", "gate/five.json"),
        intervals("intervals-five", "scored-five"),
        scoring("scored-four", "gate/four.json"),
        intervals("intervals-four", "scored-four"),
    ]


def no_matrix(d):
    del d["entries"][3]["extra"]["sealed-run"][0:2]
    del d["entries"][3]["extra"]["rehearsal"]


def mixed_matrix_read_by_no_intervals(d):
    # the review's (a): the nine and a development unit, with the truth run, no day_intervals
    d["entries"].append(scoring("scored-mixed", "gate/ten.json"))


def development_matrix_with_the_truth_run(d):
    # the review's (b): the sealed route's options with a matrix of development units
    d["entries"][3]["extra"]["sealed-run"][1] = "gate/development.json"


def route_command(module: str, *arguments: str) -> list[str]:
    """A command of the sealed route (no stand-in is needed: the driver refuses the list)."""
    return [sys.executable, "-m", f"fleetreserve.{module}", *arguments]


def held_out_command_without_the_truth_run(module: str, *arguments: str):
    # the review's (c): a command of the sealed route on the held-out units, no truth run
    def change(d):
        d["entries"].append(
            entry("sealed-other", route_command(module, *arguments), only="sealed-run")
        )

    change.__name__ = f"{module}_on_the_nine_without_the_truth_run"
    return change


def gate_run_on_the_route(d):
    # the review's (d): a sealed gate_run with the truth run and the seed
    seeded = ["--interval-seed", "{seed:intervals}", "--interval-resamples", "2000"]
    truth = ["--held-out-truth-run", "{run:sealed-truth}"]
    d["entries"].append(
        entry(
            "sealed-gate",
            route_command("gate_run", "--matrix", SEALED_MATRIX, *truth, *seeded),
            only="sealed-run",
            after=["sealed-truth"],
        )
    )


def shortened_matrix(d):
    # the review's list: argparse reads `--mat` as `--matrix` and keeps the last, five units
    d["entries"][3]["extra"]["sealed-run"] += ["--mat", "gate/five.json"]


def matrix_twice(d):
    d["entries"][3]["extra"]["sealed-run"] += ["--matrix", "gate/five.json"]


def shortened_option_no_rule_reads(d):
    d["entries"][3]["command"] += ["--work", "2"]  # learn_run's --workers


def nine_without_the_truth_run(d):
    del d["entries"][3]["extra"]["sealed-run"][2:]
    del d["entries"][4]  # no intervals: the matrix alone puts it on the sealed route


def matrix_written_with_a_tilde(d):
    d["entries"][3]["extra"]["sealed-run"][1] = "~/sealed-matrix.json"


def unreadable_matrix(d):
    d["entries"][3]["extra"]["sealed-run"][1] = "gate/no-such-matrix.json"


NEEDS = "refused at its start: held-out units need --held-out-truth-run"
OTHER = "entry 6 (sealed-other)"
OTHER_COMMANDS = [
    *((m, ("--matrix", SEALED_MATRIX)) for m in ("compare_run", "corner_run", "ledger_run")),
    *((m, ("--units", *NINE)) for m in ("cold_start", "drift_run", "o3_run")),
]
RULE_1 = "(rule 1, reading L11)"
RULE_3 = "(rule 3, reading L14)"
LEARN = "entry 4 (scored-learn)"
FAULTS = [
    (literal_truth_run, LEARN, "a literal path, where {run:<label>} of the sealed truth build"),
    (held_build_after_the_scoring, "entry 4 (sealed-truth-held)", RULE_1),
    (build_named_after_a_reader, "entry 3 (sealed-truth)", "after entry 2 (sealed-cold)"),
    (build_of_another_module, "entry 2 (sealed-truth)", "runs fleetreserve.compare_run, not"),
    (unnamed_build_after_the_scoring, "entry 6 (late-truth)", RULE_1),
    (shortened_truth_run, LEARN, "shortens --held-out-truth-run"),
    (truth_run_twice, LEARN, "gives --held-out-truth-run 2 times"),
    (literal_intervals_run, "entry 5 (scored-intervals)", "the run whose units it pools cannot"),
    (
        five_of_the_nine,
        LEARN,
        f"scores 5 of the 9 held-out units (lacks {', '.join(sorted(NINE[5:]))})",
    ),
    (the_nine_and_another, LEARN, "scores 9 of the 9 held-out units (also names nrel-before"),
    (nine_split_over_two_runs, "entry 4 (scored-five)", "scores 5 of the 9 held-out units"),
    (nine_split_over_two_runs, "entry 6 (scored-four)", "scores 4 of the 9 held-out units"),
    (no_matrix, LEARN, f"names no --matrix file whose units can be read {RULE_3}"),
    (unreadable_matrix, LEARN, "its matrix gate/no-such-matrix.json cannot be read"),
    (shortened_matrix, LEARN, "its argument '--mat' shortens --matrix"),
    (matrix_twice, LEARN, "gives --matrix 2 times"),
    (shortened_option_no_rule_reads, LEARN, "its argument '--work' shortens --workers"),
    (nine_without_the_truth_run, LEARN, NEEDS),
    (matrix_written_with_a_tilde, LEARN, "it is written with ~"),
    (mixed_matrix_read_by_no_intervals, "entry 6 (scored-mixed)", "mix held-out units"),
    (development_matrix_with_the_truth_run, LEARN, "are for held-out units, and none of the"),
    (no_matrix, LEARN, "of its matrix learn/matrix.json, refused at its start by the sealed"),
    *(
        (held_out_command_without_the_truth_run(module, *arguments), OTHER, NEEDS)
        for module, arguments in OTHER_COMMANDS
    ),
    (gate_run_on_the_route, "entry 6 (sealed-gate)", "builds a scored unit's forecasts"),
]
IDS = [f"{change.__name__}-{where.split()[1]}" for change, where, _ in FAULTS]


def faulty(change) -> dict:
    listed = copy.deepcopy(honest())
    change(listed)
    return listed


@pytest.mark.parametrize(("change", "where", "what"), FAULTS, ids=IDS)
def test_each_fault_of_the_sealed_run_is_named_by_the_checker(repo, capsys, change, where, what):
    path = commit_list(repo, faulty(change))
    code, lines = check(path, capsys)
    assert code == 1, lines
    assert lines[-1].startswith("faulty: ")
    named = [line for line in lines[:-1] if line.startswith(f"FAULT: {where}: ") and what in line]
    assert named, lines


@pytest.mark.parametrize(("change", "where", "what"), FAULTS, ids=IDS)
@pytest.mark.parametrize("purpose", [REHEARSE, SEAL], ids=["rehearsal", "sealed-run"])
def test_the_driver_refuses_each_fault_before_anything_runs(
    repo, monkeypatch, capsys, purpose, change, where, what
):
    commit_list(repo, faulty(change))
    passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *purpose]) == 2
    out = capsys.readouterr().out
    assert out.startswith(f"refused: the run list {LIST_PATH} is faulty"), out
    assert any(
        line.startswith(f"FAULT: {where}: ") and what in line for line in out.splitlines()
    ), out
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


def test_a_learn_run_off_the_sealed_route_is_not_checked(repo, capsys):
    listed = honest()
    route = listed["entries"][3]["extra"]["sealed-run"]
    route[1] = "gate/development.json"
    del route[2:]  # no sealed tables, no sealed build and no held-out unit: off the route
    path = commit_list(repo, listed)
    code, lines = check(path, capsys)
    assert code == 0, lines


def test_a_matrix_to_read_outside_a_git_working_copy_cannot_be_decided(tmp_path, capsys):
    path = tmp_path / "list.json"
    path.write_text(json.dumps(honest()))
    code, lines = check(path, capsys)
    assert code == 2, lines
    assert lines == [
        f"COULD NOT RUN: entry 4 (scored-learn) names the matrix {SEALED_MATRIX}, which cannot be "
        "found: the list is not in a git working copy, the folder its commands run from"
    ]


# --- 4. after a logged parsing fix every entry runs again at the fix commit ----------------------


def test_after_a_logged_parsing_fix_every_entry_runs_again_at_the_fix_commit(repo):
    commit_list(repo, honest())
    with (repo / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("data/\n")
    fake_heldout.copy_package(repo)
    tagged = fake_heldout.freeze(repo)
    assert git(repo, "rev-parse", "HEAD").strip() == tagged
    assert run_driver.main([LIST_PATH, *SEAL]) == 0  # the REAL gate, at the tag
    (first_log,) = drive_logs(repo)
    first_runs = set(run_folders(repo))
    fake_heldout.log_refusal(repo, fake_heldout.ASR_FILE, b"made-up sealed bytes\n", tagged)
    fixed = fake_heldout.entry(repo, fake_heldout.correction(repo, "asr"))
    time.sleep(1.1)  # the second drive's log and runs start in a later UTC second
    assert fixed != tagged
    assert git(repo, "rev-parse", "HEAD").strip() == fixed  # HEAD truly at the fix commit
    assert run_driver.main([LIST_PATH, *SEAL]) == 0  # the REAL gate, at the fix
    (second_log,) = [p for p in drive_logs(repo) if p != first_log]
    records = read_log(second_log)
    assert (records[0]["commit"], records[0]["tag_commit"], records[0]["fix_commit"]) == (
        fixed,
        tagged,
        fixed,
    )
    entries = logged(records, "entry")
    labels = ["wp2-truth", "sealed-truth", "sealed-truth-held", "scored-learn", "scored-intervals"]
    assert [r["label"] for r in entries] == labels  # every entry, from the first
    runs = {r["label"]: r["run"] for r in entries}
    for record in entries:
        folder = repo / record["run"]
        assert folder not in first_runs  # a run of its own, not one recorded at the tag
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["commit"] == fixed
    # each {run:<label>} is filled with a run of this drive
    learn = (repo / runs["scored-learn"] / "step.txt").read_text().split()
    assert learn[learn.index("--held-out-truth-run") + 1] == runs["sealed-truth"]
    told = (repo / runs["scored-intervals"] / "step.txt").read_text().split()
    assert told[told.index("--run") + 1] == runs["scored-learn"]


# --- 2. rule 2, the interval seed (held back: it refuses lists of the earlier tests) -------------


def no_interval_seed(d):
    del d["entries"][3]["command"][3:5]


def resamples_not_2000(d):
    d["entries"][3]["command"][6] = "200"


def seed_differs(d):
    d["entries"][4]["command"][6] = "7"


def two_seeds(d):
    other = intervals("other-intervals", "scored-learn")
    other["command"][6] = "7"
    d["entries"].append(other)


def no_day_intervals(d):
    del d["entries"][4]


def intervals_of_a_build(d):
    d["entries"].append(
        entry(
            "build-intervals",
            step("day_intervals", "--run", "{run:sealed-truth}", "--seed", "{seed:intervals}"),
            only="sealed-run",
        )
    )


RULE_2 = "(rule 2, reading L13)"
SEED_FAULTS = [
    (no_interval_seed, LEARN, "scores without --interval-seed S --interval-resamples 2000"),
    (resamples_not_2000, LEARN, "scores without --interval-seed S --interval-resamples 2000"),
    (seed_differs, LEARN, f"its --interval-seed is {SEED}, not the seed 7"),
    (two_seeds, "the sealed run", "use more than one --seed"),
    (no_day_intervals, LEARN, "no day_intervals entry of the sealed run gives the seed S"),
    (intervals_of_a_build, "entry 6 (build-intervals)", "which is no run of gate_run or learn_run"),
]


@pytest.mark.parametrize(
    ("change", "where", "what"), SEED_FAULTS, ids=[f[0].__name__ for f in SEED_FAULTS]
)
def test_each_fault_of_the_interval_seed_is_named_by_the_checker(repo, capsys, change, where, what):
    path = commit_list(repo, faulty(change))
    code, lines = check(path, capsys)
    assert code == 1, lines
    named = [line for line in lines[:-1] if line.startswith(f"FAULT: {where}: ") and what in line]
    assert named, lines


@pytest.mark.parametrize(
    ("change", "where", "what"), SEED_FAULTS, ids=[f[0].__name__ for f in SEED_FAULTS]
)
def test_the_driver_refuses_each_fault_of_the_interval_seed(
    repo, monkeypatch, capsys, change, where, what
):
    commit_list(repo, faulty(change))
    passing_gate(monkeypatch)
    assert run_driver.main([LIST_PATH, *REHEARSE]) == 2
    out = capsys.readouterr().out
    assert any(
        line.startswith(f"FAULT: {where}: ") and what in line for line in out.splitlines()
    ), out
    assert drive_logs(repo) == []
    assert run_folders(repo) == []


def test_the_route_names_the_sealed_truth_build_module():
    assert f"fleetreserve.{sealed_route.TRUTH_BUILD_MODULE}" == heldout_truth_build.__name__


def test_a_matrix_in_the_working_copy_only_is_not_read(repo, capsys):
    path = commit_list(repo, faulty(lambda d: None))
    write_matrix(repo, SEALED_MATRIX, NINE[:5])  # the working copy's file: five, not committed
    code, lines = check(path, capsys)
    assert code == 0, lines  # the committed matrix of the nine is the one read
    (repo / "gate" / "loose.json").write_text((repo / SEALED_MATRIX).read_text())
    listed = honest()
    listed["entries"][3]["extra"]["sealed-run"][1] = "gate/loose.json"
    (repo / "loose-list.json").write_text(json.dumps(listed))
    code, lines = check(repo / "loose-list.json", capsys)
    assert code == 1, lines
    assert any("it is not a file committed at the checked-out commit" in line for line in lines)


def test_a_matrix_named_in_the_sealed_folder_is_refused_and_never_opened(repo, capsys):
    with (repo / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("data/\n")
    git(repo, "commit", "-q", "-am", "data ignored")
    sealed = repo / "data" / "heldout"
    write_matrix(repo, "data/heldout/x.json", NINE)
    listed = honest()
    listed["entries"][3]["extra"]["sealed-run"][1] = "data/heldout/x.json"
    path = commit_list(repo, listed)
    with sealed_watch.watch(sealed) as seen:
        code, lines = check(path, capsys)
    assert code == 1, lines
    assert seen == []  # no open, listing or scan of the sealed folder or its file
    assert any(
        line.startswith(f"FAULT: {LEARN}: its matrix data/heldout/x.json cannot be read")
        and "refused before it is opened" in line
        for line in lines
    ), lines
