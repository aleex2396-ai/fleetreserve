"""`tools/freeze_check.py matrix` (WP11's "run matrix frozen") and `lists` (WP13's "held-out
list, run matrix, seeds"), on made-up repositories in the temporary folder. Each holds the
project's run list (`run-lists/run-list.json`, its dials set to the made-up main model's; its
own fixed seed for `compare_run --no-pause-seed` kept), the committed gate matrix, four made-up
kappa-choice runs made by `kappa_choice.joint` itself (the hand case of
`tests/wp6gate/test_kappa_choice.py`) with their manifests, and the matrices `sealed_matrix`
writes from them, committed. No test reads the project's own data or runs folder.

1. Both items pass on the repository as written.
2. `matrix` (and so `lists`) fails on: a named matrix missing while others are there; a named
   matrix in the working copy only; a committed matrix changed in the working copy or hidden from
   git status; the run list changed or in the working copy only; a matrix whose sealed_choice
   disagrees with its configurations (`sealed_matrix.check`); a matrix whose kappa is edited by
   hand everywhere, so it holds against itself (what `sealed_matrix` writes from its source
   differs); a run list whose fixed dial differs from the main model's matrix; a matrix
   recording another SHA-256 than its source has; a source changed since its run (its manifest
   refuses it); a source named outside the runs folder; a faulty run list; the gate matrix
   changed in the working copy. It could not run on:
   no run list; no named matrix written; a source run missing; another path of the tree changed;
   a committed gate matrix that is not the running code's; the run list left unread.
3. `lists` fails on: a sealed matrix with a held-out unit added, dropped or swapped, or named
   twice (no matrix); a sealed-run group of the run list naming another unit, or the groups
   leaving a held-out unit out; a literal seed in an entry's command or in a purpose's
   arguments; a seed option given twice; a seed option shortened; a command's seed option left
   at its default; the run list's own interval-seed rule (L13) broken. It reads, beside the
   matrices the run list names, every file under `gate/sealed-matrix/` (the review's break B8):
   one there that the run list does not name fails on a held-out unit swapped, on each of the
   matrix checks (c) to (e), as no matrix, and when it is untracked, ignored, staged but not
   committed, changed or hidden from git status, where `matrix` decides as before.
4. Neither item opens, lists or scans anything in the made-up sealed folder
   (`tests/wp6gate/sealed_watch.py`), passing or failing, also with a source run folder that is a
   link into it.
5. The script gives each exit status to a caller that runs it as the pass mark does.
"""

import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_kappa_choice as made_up_choice
from sealed_watch import watch

from fleetreserve import kappa_choice, run_list, sealed_matrix, sealed_route

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "freeze_check.py"
sys.path.insert(0, str(TOP / "tools"))
sys.path.insert(0, str(TOP / "tests" / "wp2"))
freeze_check = importlib.import_module("freeze_check")
fake = importlib.import_module("fake_heldout")

FILES = {fake.ASR_FILE: b"made-up sealed bytes\n", fake.NORWAY_FILE: b"other made-up bytes\n"}
SEALED = "data/heldout"
RUN_LIST = "run-lists/run-list.json"
MODELS = sealed_matrix.FULL_METHOD_MODELS
W, OTHER = "workplaces", "Norway and Korea"
MAIN = "gate/sealed-matrix/lgbm_tail.json"
REHEARSAL_MAIN = "gate/rehearsal-matrix/lgbm_tail.json"
HAZARD = "gate/sealed-matrix/hazard_tuned_inputs.json"
NAMED = 14  # the matrices the run list names: 7 sealed, 7 rehearsal


def joint(model: str, r2=None) -> dict:
    document = copy.deepcopy(kappa_choice.joint(made_up_choice.runs()))
    document["run"]["model"] = model
    if r2 is not None:
        kappa, dials = r2
        document["readings"]["R2"]["kappa"] = kappa
        for site_type, dial in dials.items():
            document["readings"]["R2"]["dials"][site_type] = {
                "dial": dial,
                "unit": "u",
                "mean_offer": 1.0,
            }
    return document


def record(root: Path, model: str, document: dict, commit: str) -> Path:
    folder = root / "runs.noindex" / f"20261120T000000Z-kappa-{model}"
    folder.mkdir(parents=True)
    path = folder / kappa_choice.RESULTS_NAME
    path.write_text(json.dumps(document))
    outputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"commit": commit, "command": ["python", "-m", "fleetreserve.kappa_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return path


def run_list_text() -> dict:
    """The project's run list with the made-up main model's dials (R2 of the hand case: 0.05 at
    both site types). The list itself gives each `compare_run` entry its fixed seed
    `--no-pause-seed {seed:no-pause}`, so none is added here (added a second time, run_list
    refuses the list, reading L12)."""
    document = json.loads((TOP / RUN_LIST).read_text())
    document["fixed"]["names"]["dial-workplaces"] = "0.05"
    document["fixed"]["names"]["dial-others"] = "0.05"
    return document


def write_json(root: Path, rel: str, document: dict) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n")


def read_json(root: Path, rel: str) -> dict:
    return json.loads((root / rel).read_text())


def build(folder: Path) -> Path:
    root = fake.sealed_repo(folder, FILES, tag=False, package=False)
    (root / ".gitignore").write_text("data/\nruns.noindex/\n", encoding="utf-8")
    (root / "gate").mkdir()
    shutil.copyfile(sealed_matrix.TEMPLATE, root / "gate" / "matrix.json")
    commit = fake.commit(root, "made-up gate matrix")
    documents = {m: joint(m) for m in MODELS}
    documents["earlier_gbm"] = joint("earlier_gbm", (0.3, {W: 0.01, OTHER: 0.05}))
    paths = [str(record(root, m, d, commit)) for m, d in documents.items()]
    assert sealed_matrix.main([*paths, "--out", str(root / "gate")]) == 0
    write_json(root, RUN_LIST, run_list_text())
    fake.commit(root, "made-up run list and matrices")
    return root


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build(tmp_path_factory.mktemp("freeze-items"))


@pytest.fixture
def repo(base, tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(base, root, symlinks=True)
    return root


def run(item, root, capsys):
    status = freeze_check.main([item, "--root", str(root)])
    return status, capsys.readouterr().out.splitlines()


def both(root, capsys):
    return run("matrix", root, capsys), run("lists", root, capsys)


def has(lines, *parts):
    """Some line holds every one of `parts`."""
    return any(all(part in line for part in parts) for line in lines)


def edit(root: Path, rel: str, change, *, commit=True) -> None:
    document = read_json(root, rel)
    change(document)
    write_json(root, rel, document)
    if commit:
        fake.commit(root, f"made-up change of {rel}")


def entry(document: dict, label: str) -> dict:
    return next(e for e in document["entries"] if e["label"] == label)


def with_other_site_type(document: dict, label: str) -> dict:
    """The entry `label` copied right after itself as `<label>-others` at `{name:dial-others}`,
    the form the list had before one step stood for both site types (the dials of the made-up
    main model are equal, so rule 11 holds): a fixed seed that the edited entry no longer names
    is still named by the copy, so the list's own rule that every fixed value is used stays
    quiet and the check under test is the one that speaks."""
    found = entry(document, label)
    copied = copy.deepcopy(found)
    copied["label"] = f"{label}-others"
    copied["command"] = replace(copied["command"], "{name:dial-workplaces}", "{name:dial-others}")
    document["entries"].insert(document["entries"].index(found) + 1, copied)
    return copied


# --- 1. a pass ----------------------------------------------------------------------------------


def test_both_items_pass_on_the_repository_as_written(repo, capsys):
    (matrix, matrix_lines), (lists, lists_lines) = both(repo, capsys)
    assert (matrix, lists) == (0, 0), matrix_lines + lists_lines
    assert matrix_lines == [
        f"run matrix frozen: {RUN_LIST} and the {NAMED} matrices it names (7 named by the "
        f"sealed run) are committed at HEAD {fake.git(repo, 'rev-parse', 'HEAD')} and unchanged; "
        "each passes sealed_matrix.check, its kappa-choice results are verified by their runs' "
        "manifests with the SHA-256 it records, and it is what sealed_matrix writes from them "
        "and the committed gate matrix; the run list's dials are those of the main model "
        "lgbm_tail"
    ]
    assert len(lists_lines) == 1
    assert lists_lines[0].startswith(
        "held-out list, run matrix, seeds: the 9 held-out units of sealed_route.HELD_OUT_UNITS "
        "are the gate units of each of the 7 sealed matrices"
    )


# --- 2. run matrix frozen -------------------------------------------------------------------------


def test_a_named_matrix_missing_while_others_are_there_fails(repo, capsys):
    fake.git(repo, "rm", "-q", HAZARD)
    fake.commit(repo, "a matrix removed")
    for status, lines in both(repo, capsys):
        assert status == 1
        assert f"FAILED: {HAZARD} is neither committed at HEAD nor in the working copy" in lines


def test_no_named_matrix_written_could_not_run(repo, capsys):
    fake.git(repo, "rm", "-q", "-r", "gate/sealed-matrix", "gate/rehearsal-matrix")
    fake.commit(repo, "every matrix removed")
    status, lines = run("matrix", repo, capsys)
    assert status == 2
    assert lines == [
        f"COULD NOT RUN: none of the {NAMED} matrices {RUN_LIST} names is committed or in the "
        "working copy, first gate/rehearsal-matrix/earlier_gbm.json (sealed_matrix writes them; "
        "reading F11)"
    ]
    assert run("lists", repo, capsys)[0] == 2


def test_a_named_matrix_in_the_working_copy_only_fails(repo, capsys):
    fake.git(repo, "rm", "-q", "--cached", HAZARD)
    fake.git(repo, "commit", "-q", "--no-verify", "-m", "a matrix no longer tracked")
    assert (repo / HAZARD).is_file()
    for status, lines in both(repo, capsys):
        assert status == 1
        assert f"FAILED: {HAZARD} is not committed at HEAD: it is in the working copy only" in lines


def test_a_committed_matrix_changed_in_the_working_copy_fails(repo, capsys):
    edit(repo, HAZARD, lambda d: d.update(rho=0.5), commit=False)
    for status, lines in both(repo, capsys):
        assert status == 1
        assert f"FAILED: {HAZARD} is changed in the working copy (git status ' M')" in lines


def test_a_matrix_hidden_from_git_status_fails(repo, capsys):
    fake.git(repo, "update-index", "--assume-unchanged", HAZARD)
    edit(repo, HAZARD, lambda d: d.update(rho=0.5), commit=False)
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(lines, f"FAILED: {HAZARD} is marked assume-unchanged or skip-worktree")


def test_the_run_list_changed_in_the_working_copy_fails(repo, capsys):
    edit(repo, RUN_LIST, lambda d: d.update(about="changed"), commit=False)
    for status, lines in both(repo, capsys):
        assert status == 1
        assert f"FAILED: {RUN_LIST} is changed in the working copy (git status ' M')" in lines


def test_a_run_list_in_the_working_copy_only_fails(repo, capsys):
    fake.git(repo, "rm", "-q", "--cached", RUN_LIST)
    fake.git(repo, "commit", "-q", "--no-verify", "-m", "the run list no longer tracked")
    for status, lines in both(repo, capsys):
        assert (status, lines) == (
            1,
            [f"FAILED: {RUN_LIST} is not committed at HEAD: it is in the working copy only"],
        )


def test_no_run_list_could_not_run(repo, capsys):
    fake.git(repo, "rm", "-q", RUN_LIST)
    fake.commit(repo, "the run list removed")
    for status, lines in both(repo, capsys):
        assert (status, lines) == (
            2,
            [
                f"COULD NOT RUN: no run list: {RUN_LIST} is neither committed at HEAD nor in the "
                "working copy"
            ],
        )


def test_a_matrix_whose_sealed_choice_disagrees_with_its_configurations_fails(repo, capsys):
    edit(repo, HAZARD, lambda d: d["sealed_choice"].update(kappa=0.2))
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(
            lines,
            f"FAILED: {HAZARD}: its sealed_choice does not hold (sealed_matrix.check)",
            "a configuration is not at the matrix's kappa",
        )


def kappa_everywhere(kappa):
    def change(document):
        document["sealed_choice"]["kappa"] = kappa
        for configuration in document["configurations"]:
            configuration["kappa"] = kappa

    return change


def test_a_kappa_edited_by_hand_everywhere_fails_against_its_source(repo, capsys):
    edit(repo, HAZARD, kappa_everywhere(0.2))
    assert sealed_matrix.check(read_json(repo, HAZARD)) == []  # it holds against itself
    for status, lines in both(repo, capsys):
        assert status == 1
        assert (
            f"FAILED: {HAZARD}: differs from what sealed_matrix writes from its kappa-choice "
            "results and the committed gate matrix, in configurations, sealed_choice"
        ) in lines


def test_a_shared_value_edited_by_hand_fails_against_the_gate_matrix(repo, capsys):
    edit(repo, HAZARD, lambda d: d.update(split_seed=7))
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(lines, f"FAILED: {HAZARD}: differs from what sealed_matrix writes", "split_seed")


def test_a_run_list_dial_other_than_the_main_models_fails(repo, capsys):
    edit(repo, RUN_LIST, lambda d: d["fixed"]["names"].update({"dial-workplaces": "0.02"}))
    for status, lines in both(repo, capsys):
        assert status == 1
        for rel in (MAIN, REHEARSAL_MAIN):
            assert (
                f"FAILED: {RUN_LIST}: its fixed name dial-workplaces is '0.02', not the dial 0.05 "
                f"that sealed_choice.dials gives workplaces in {rel}"
            ) in lines


def test_a_matrix_recording_another_sha256_than_its_source_has_fails(repo, capsys):
    edit(repo, HAZARD, lambda d: d["sealed_choice"]["source"].update(sha256="0" * 64))
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(
            lines,
            f"FAILED: {HAZARD}: sealed_choice.source records another sha256 than its "
            "kappa-choice results",
        )


def source_of(root, rel=HAZARD) -> Path:
    source = read_json(root, rel)["sealed_choice"]["source"]
    return root / "runs.noindex" / source["run_folder"] / source["results"]


def test_a_source_changed_since_its_run_fails(repo, capsys):
    path = source_of(repo)
    path.write_text(path.read_text() + " ")
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(
            lines,
            "FAILED: gate/rehearsal-matrix/hazard_tuned_inputs.json: its source runs.noindex/",
            "is refused by its run's manifest: the file has changed since the run",
        )


def test_a_source_run_missing_could_not_run(repo, capsys):
    shutil.rmtree(source_of(repo).parent)
    for status, lines in both(repo, capsys):
        assert status == 2
        assert has(
            lines,
            "COULD NOT RUN: gate/rehearsal-matrix/hazard_tuned_inputs.json: its source "
            "runs.noindex/",
            "cannot be verified: the file is missing",
        )


def test_a_source_named_outside_the_runs_folder_fails_and_nothing_is_read(repo, capsys):
    edit(
        repo,
        HAZARD,
        lambda d: d["sealed_choice"]["source"].update(run_folder="../data/heldout/asr"),
    )
    with watch(repo / SEALED) as seen:
        for status, lines in both(repo, capsys):
            assert status == 1
            assert (
                f"FAILED: {HAZARD}: sealed_choice.source names '../data/heldout/asr'/"
                "'kappa-choice.json', not a file kappa-choice.json in a run folder of "
                "runs.noindex/; nothing read there"
            ) in lines
    assert seen == []


def test_a_faulty_run_list_fails(repo, capsys):
    edit(repo, RUN_LIST, lambda d: d["fixed"]["seeds"].update(unused=3))
    for status, lines in both(repo, capsys):
        assert status == 1
        assert has(lines, f"FAILED: {RUN_LIST} at HEAD is faulty by its own checker (run_list)")


def test_a_gate_matrix_changed_in_the_working_copy_fails(repo, capsys):
    edit(repo, "gate/matrix.json", lambda d: d.update(split_seed=7), commit=False)
    for status, lines in both(repo, capsys):
        assert status == 1
        assert (
            "FAILED: the gate matrix, every matrix's other source: gate/matrix.json is changed in "
            "the working copy (git status ' M')"
        ) in lines


def test_a_committed_gate_matrix_other_than_the_running_codes_could_not_run(repo, capsys):
    edit(repo, "gate/matrix.json", lambda d: d.update(split_seed=7))
    for status, lines in both(repo, capsys):
        assert status == 2
        assert has(
            lines,
            "COULD NOT RUN: gate/matrix.json committed at HEAD is not the gate matrix of the "
            "running code",
        )


def test_a_run_list_left_unread_is_never_a_pass(repo, capsys, monkeypatch):
    monkeypatch.setattr(freeze_check, "_frozen_matrices", lambda route, root, found, **_: None)
    for status, lines in both(repo, capsys):
        assert (status, lines) == (
            2,
            ["COULD NOT RUN: the run list was not read, so nothing was checked"],
        )


def test_another_path_changed_could_not_run(repo, capsys):
    (repo / "notes.txt").write_text("made-up\n")
    for status, lines in both(repo, capsys):
        assert (status, lines) == (
            2,
            [
                "COULD NOT RUN: the tree is not clean: 1 other path changed, untracked or "
                "hidden from git status, first notes.txt"
            ],
        )


# --- 3. held-out list and seeds -------------------------------------------------------------------


NINE = list(sealed_route.HELD_OUT_UNITS)


@pytest.mark.parametrize(
    "units, said",
    [
        ([*NINE, "korea-other"], "it adds korea-other"),
        (NINE[:-1], f"it drops {NINE[-1]}"),
        ([*NINE[:-1], "korea-other"], f"it adds korea-other; drops {NINE[-1]}"),
    ],
    ids=["added", "dropped", "swapped"],
)
def test_a_sealed_matrix_whose_gate_units_are_not_the_nine_fails(repo, capsys, units, said):
    edit(repo, HAZARD, lambda d: d.update(gate_units=units))
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert (
        f"FAILED: {HAZARD}: its gate units are not the 9 held-out units of "
        f"sealed_route.HELD_OUT_UNITS: {said}"
    ) in lines


def without_o2(document: dict) -> None:
    """The run list's outcome O2 steps (`ceiling_report`) left out: they name the sealed group
    `nine` too, and the run list's own rule 10 refuses an O2 step on a group that is not the nine
    held-out units before this item reads the groups."""
    document["entries"] = [
        e for e in document["entries"] if "fleetreserve.ceiling_report" not in e["command"]
    ]


def test_a_sealed_run_group_naming_another_unit_fails(repo, capsys):
    def swap(document):
        without_o2(document)
        group = document["units"]["sealed-run"]["nine"]
        group[group.index("korea-visits")] = "korea-other"

    edit(repo, RUN_LIST, swap)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert (
        f"FAILED: {RUN_LIST}: its sealed-run group 'nine' names korea-other, not a held-out unit "
        "of sealed_route.HELD_OUT_UNITS"
    ) in lines


def test_sealed_run_groups_leaving_a_held_out_unit_out_fail(repo, capsys):
    def drop(document):
        without_o2(document)
        for group in document["units"]["sealed-run"].values():
            if "korea-visits" in group:
                group.remove("korea-visits")

    edit(repo, RUN_LIST, drop)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert (
        f"FAILED: {RUN_LIST}: no sealed-run group names the held-out unit(s) korea-visits"
    ) in lines


def replace(command, old, new):
    return [new if part == old else part for part in command]


def test_a_literal_seed_in_a_command_fails(repo, capsys):
    def literal(document):
        with_other_site_type(document, "corner")
        found = entry(document, "corner")
        found["command"] = replace(found["command"], "{seed:split}", "7")

    edit(repo, RUN_LIST, literal)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    for purpose in ("rehearsal", "sealed-run"):
        assert (
            f"FAILED: entry 24 (corner), {purpose}: its --seed is '7', not one of the "
            "run list's fixed seeds {seed:<name>}: a literal seed"
        ) in lines


def test_a_literal_seed_in_one_purposes_arguments_fails(repo, capsys):
    def literal(document):
        found = entry(document, "o3")
        found["command"] = [p for p in found["command"] if p not in ("--seed", "{seed:o3}")]
        found["extra"]["rehearsal"] = ["--seed=3"]
        found["extra"]["sealed-run"] = [*found["extra"]["sealed-run"], "--seed", "{seed:o3}"]

    edit(repo, RUN_LIST, literal)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert (
        "FAILED: entry 28 (o3), rehearsal: its --seed is '3', not one of the run list's fixed "
        "seeds {seed:<name>}: a literal seed"
    ) in lines
    assert not has(lines, "entry 28 (o3), sealed-run")


def seed_twice(document) -> None:
    """The o3 entry's rehearsal gives `--seed` a second time."""
    entry(document, "o3")["extra"]["rehearsal"] = ["--seed", "{seed:ledger}"]


def seed_shortened(document) -> None:
    """The o3 entry's rehearsal gives its seed as `--see`, the sealed run as `--seed`."""
    found = entry(document, "o3")
    found["command"] = [p for p in found["command"] if p not in ("--seed", "{seed:o3}")]
    found["extra"]["rehearsal"] = ["--see", "{seed:o3}"]
    found["extra"]["sealed-run"] = [*found["extra"]["sealed-run"], "--seed", "{seed:o3}"]


def rehearsal_forms_unchecked(monkeypatch) -> None:
    """The run list's own check of the rehearsal's option forms left out (`run_list`, reading
    L12: `option_form_faults` of the rehearsal's steps), the sealed run's kept. That check
    refuses a seed option given twice or shortened on any rehearsal step, and this item reads no
    seeds of a list `run_list.parse` refuses, so its own seeds part (h) is reached on such a step
    only without it: the next test shows the route as it runs."""
    original = run_list.option_form_faults

    def sealed_run_only(steps, said="on the sealed run", why="(reading L12)"):
        return [] if said == "in the rehearsal" else original(steps, said, why)

    monkeypatch.setattr(run_list, "option_form_faults", sealed_run_only)


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (
            seed_twice,
            "entry 28 (o3): gives --seed 2 times: in the rehearsal an option is given once",
        ),
        (
            seed_shortened,
            "entry 28 (o3): its argument '--see' shortens --seed: in the rehearsal every option "
            "is written in full",
        ),
    ],
    ids=["twice", "shortened"],
)
def test_a_doubled_or_shortened_rehearsal_seed_fails_by_the_run_lists_own_check(
    repo, capsys, change, said
):
    """With `run_list.option_form_faults` as it runs, `lists` fails on each list of the next two
    tests by the run list's own checker (part (b)), naming the step, before its seeds part (h)."""
    edit(repo, RUN_LIST, change)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert (
        f"FAILED: {RUN_LIST} at HEAD is faulty by its own checker (run_list): {said} (reading L12)"
    ) in lines
    assert not has(lines, "two seeds for one draw") and not has(lines, ", rehearsal: its argument")


def test_a_seed_given_twice_in_one_purpose_fails(repo, capsys, monkeypatch):
    rehearsal_forms_unchecked(monkeypatch)
    edit(repo, RUN_LIST, seed_twice)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert "FAILED: entry 28 (o3), rehearsal: gives --seed 2 times, two seeds for one draw" in lines


def test_a_shortened_seed_option_fails(repo, capsys, monkeypatch):
    rehearsal_forms_unchecked(monkeypatch)  # the sealed run's --seed in full: run_list's L12 kept
    edit(repo, RUN_LIST, seed_shortened)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert "FAILED: entry 28 (o3), rehearsal: its argument '--see' shortens --seed" in lines


def test_a_seed_option_left_at_its_commands_default_fails(repo, capsys):
    def left(document):
        with_other_site_type(document, "compare")
        found = entry(document, "compare")
        at = found["command"].index("--no-pause-seed")  # found by name, wherever it stands
        assert found["command"][at + 1] == "{seed:no-pause}"
        del found["command"][at : at + 2]

    edit(repo, RUN_LIST, left)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    for purpose in ("rehearsal", "sealed-run"):
        assert (
            f"FAILED: entry 23 (compare), {purpose}: leaves compare_run's "
            "--no-pause-seed at the command's own default, a seed not fixed in the run list"
        ) in lines


def test_the_run_lists_own_interval_seed_rule_holds(repo, capsys):
    def other(document):
        found = entry(document, "intervals-hazard_tuned_inputs")
        found["command"] = replace(found["command"], "{seed:intervals}", "{seed:ledger}")

    edit(repo, RUN_LIST, other)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(
        lines,
        f"FAILED: {RUN_LIST} at HEAD is faulty by its own checker (run_list)",
        "reading L13",
    )


def test_a_sealed_matrix_naming_a_held_out_unit_twice_fails(repo, capsys):
    edit(repo, HAZARD, lambda d: d.update(gate_units=[*NINE, NINE[0]]))
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, f"FAILED: {HAZARD}: its committed text is no matrix", "distinct unit names")


# --- 3b. every file under gate/sealed-matrix/, named by the run list or not ---------------------

SPARE = "gate/sealed-matrix/spare.json"  # a file the run list does not name
NO_PATH = "sealed_matrix writes no matrix of that path from its kappa-choice results"


def swapped(document):
    units = document["gate_units"]
    units[units.index("korea-visits")] = "korea-other"


def spare(root: Path, change=None, *, rel=SPARE, commit=True) -> None:
    """A copy of the hazard model's sealed matrix at `rel`, which the run list does not name,
    changed by `change`, committed unless `commit` is false."""
    document = read_json(root, HAZARD)
    if change is not None:
        change(document)
    write_json(root, rel, document)
    if commit:
        fake.commit(root, f"made-up matrix {rel}, not named by the run list")


def test_an_unnamed_committed_sealed_matrix_with_a_held_out_unit_swapped_fails(repo, capsys):
    """The review's break B8: `matrix` checks only what the run list names, `lists` every file."""
    spare(repo, swapped)
    with watch(repo / SEALED) as seen:
        (matrix, _), (lists, lines) = both(repo, capsys)
    assert seen == []
    assert (matrix, lists) == (0, 1)
    assert (
        f"FAILED: {SPARE}: its gate units are not the 9 held-out units of "
        "sealed_route.HELD_OUT_UNITS: it adds korea-other; drops korea-visits"
    ) in lines


def test_every_unnamed_sealed_matrix_is_read_not_only_the_first(repo, capsys):
    spare(repo, rel="gate/sealed-matrix/a-spare.json")
    spare(repo, swapped, rel="gate/sealed-matrix/z-spare.json")
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, f"FAILED: gate/sealed-matrix/a-spare.json: {NO_PATH}")
    assert has(lines, "FAILED: gate/sealed-matrix/z-spare.json: its gate units are not the 9")
    # the swap breaks z-spare's sealed_choice too (c), so the writer's check (e) is not reached
    assert has(lines, "FAILED: gate/sealed-matrix/z-spare.json: its sealed_choice does not hold")


def test_an_unnamed_sealed_matrix_in_a_folder_inside_is_read(repo, capsys):
    rel = "gate/sealed-matrix/inner/spare.json"
    spare(repo, swapped, rel=rel)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, f"FAILED: {rel}: its gate units are not the 9 held-out units")


@pytest.mark.parametrize(
    "change, said",
    [
        (
            lambda d: d["sealed_choice"].update(kappa=0.2),
            f"FAILED: {SPARE}: its sealed_choice does not hold (sealed_matrix.check): a "
            "configuration is not at the matrix's kappa",
        ),
        (
            lambda d: d["sealed_choice"]["source"].update(sha256="0" * 64),
            f"FAILED: {SPARE}: sealed_choice.source records another sha256 than its kappa-choice "
            "results runs.noindex/",
        ),
        (None, f"FAILED: {SPARE}: {NO_PATH}"),
    ],
    ids=["c-check", "d-source", "e-writer"],
)
def test_an_unnamed_sealed_matrix_failing_c_d_or_e_fails(repo, capsys, change, said):
    spare(repo, change)
    assert run("matrix", repo, capsys)[0] == 0
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, said), lines
    assert not has(lines, f"{SPARE}: its gate units")  # its gate units are the nine


def test_an_unnamed_committed_file_that_is_no_matrix_fails(repo, capsys):
    (repo / SPARE).write_text("not a matrix\n")
    fake.commit(repo, "a made-up file that is no matrix")
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, f"FAILED: {SPARE}: its committed text is no matrix")


def untracked(root):
    spare(root, commit=False)


def ignored(root):
    spare(root, commit=False)
    with (root / ".git" / "info" / "exclude").open("a") as exclude:
        exclude.write(f"{SPARE}\n")


def staged(root):
    spare(root, commit=False)
    fake.git(root, "add", SPARE)


def changed(root):
    spare(root)
    edit(root, SPARE, lambda d: d.update(rho=0.5), commit=False)


def hidden(root):
    spare(root)
    fake.git(root, "update-index", "--assume-unchanged", SPARE)
    edit(root, SPARE, swapped, commit=False)


@pytest.mark.parametrize(
    "make, said, matrix_status",
    [
        (untracked, f"{SPARE} is not committed at HEAD: it is in the working copy only", 2),
        (ignored, f"{SPARE} is not committed at HEAD: it is in the working copy only", 0),
        (staged, f"{SPARE} is not committed at HEAD: it is in the working copy only", 2),
        (changed, f"{SPARE} is changed in the working copy (git status ' M')", 2),
        (hidden, f"{SPARE} is marked assume-unchanged or skip-worktree", 2),
    ],
    ids=["untracked", "ignored", "staged", "changed", "hidden"],
)
def test_an_uncommitted_or_changed_file_under_the_sealed_folder_fails(
    repo, capsys, make, said, matrix_status
):
    make(repo)
    status, lines = run("lists", repo, capsys)
    assert status == 1
    assert has(lines, f"FAILED: {said}"), lines
    assert not has(lines, "COULD NOT RUN")
    assert run("matrix", repo, capsys)[0] == matrix_status  # that item's decision as before


def test_a_file_only_under_the_rehearsal_folder_is_another_path_as_before(repo, capsys):
    spare(repo, rel="gate/rehearsal-matrix/spare.json", commit=False)
    for status, lines in both(repo, capsys):
        assert (status, lines) == (
            2,
            [
                "COULD NOT RUN: the tree is not clean: 1 other path changed, untracked or "
                "hidden from git status, first gate/rehearsal-matrix/spare.json"
            ],
        )


# --- 4. nothing sealed opened ---------------------------------------------------------------------


def test_neither_item_opens_lists_or_scans_anything_sealed(repo, capsys):
    with watch(repo / SEALED) as seen:
        assert [status for status, _ in both(repo, capsys)] == [0, 0]
    assert seen == []


def test_a_source_run_folder_linked_into_the_sealed_folder_is_refused_unread(repo, capsys):
    folder = source_of(repo).parent
    shutil.rmtree(folder)
    (repo / SEALED / "asr" / kappa_choice.RESULTS_NAME).write_text("{}")
    (repo / SEALED / "asr" / "manifest.json").write_text("{}")
    os.symlink(repo / SEALED / "asr", folder)
    with watch(repo / SEALED) as seen:
        for status, lines in both(repo, capsys):
            assert status == 1
            assert has(
                lines,
                "FAILED: gate/rehearsal-matrix/hazard_tuned_inputs.json: its source runs.noindex/",
                "a file is refused before it is opened",
            )
    assert seen == []


# --- 5. the script --------------------------------------------------------------------------------


def script(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        env=fake.tool_environment(),
        timeout=300,
        check=False,
    )


def test_the_script_gives_each_exit_status(base, tmp_path):
    passing = tmp_path / "passing"
    failing = tmp_path / "failing"
    cannot = tmp_path / "cannot"
    for root in (passing, failing, cannot):
        shutil.copytree(base, root, symlinks=True)
    edit(failing, HAZARD, kappa_everywhere(0.2))
    (cannot / "notes.txt").write_text("made-up\n")
    for item in ("matrix", "lists"):
        statuses = [script(item, "--root", str(r)).returncode for r in (passing, failing, cannot)]
        assert statuses == [0, 1, 2]
