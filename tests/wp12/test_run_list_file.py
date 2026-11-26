"""The committed run list `run-lists/run-list.json` and the sealed run's rules 6 and 7 of the
checker (`run_list`, module text), in a made-up working copy holding the committed list, the
committed default matrices and the matrices `sealed_matrix` writes from made-up kappa-choice
results (`tests/wp12/test_sealed_matrix.py`).

1. The committed list, with those matrices committed and its two fixed dial names entered from
   the main model's matrix as its `about` says, keeps every rule the list alone and the
   committed matrices decide (exit 0 with `--no-starts`; the start check needs the model
   packages of the `learn` and `baselines` groups and runs in an environment that has them). Its
   sealed run starts with the four truth builds, the two sealed builds one after the other; each
   model of the full method is scored on the nine sealed units by one `learn_run` on its own
   matrix, the main model also on its comparison matrix and in the held setting; the interval
   seed is the one of the two `day_intervals` entries; `compare_run` gives `--splits 1`,
   `corner_run` the split seed and 50 days, `drift_run` neither `--etas` nor
   `--calibration-days`. The sensitivity of plan WP15 at rho 0.1 and 0.2 is one `learn_run` of
   the main model on each of its rho matrices, with a `day_intervals` entry on each, and no other
   model runs at those values.
2. Rule 6, the truth builds first and the openings of sealed data in one step: a development
   truth build moved after a `learn_run`, and one put between the two sealed builds, are each a
   fault naming rule 6 and no other rule. (A `learn_run` moved before the sealed builds is a
   fault already of the references rule, as its `{run:sealed-truth}` names a later entry.)
3. Rule 7, no model that reads other days' outcomes: `--model empirical` on a `learn_run` and
   `--reference empirical` on a `cold_start` are each a fault naming rule 7.
4. Intervals over days (plan section 11: "given for the main model `lgbm_tail` and the neural
   model `hazard_tuned_inputs` only"): each `learn_run` of those two models in the sealed run,
   in every setting the plan reports (the main family, the comparison with forecast energy, the
   held setting, rho 0.1 and 0.2), has one `day_intervals` entry reading its run, in the same
   setting; no other run has one; the list's `about` cites that sentence for the runs without.
5. Rule 8, each step's dial its matrix's (the review of 4ae83be, finding 1, probe P4): with the
   main model's matrix at workplaces dial 0.05 and Norway and Korea dial 0.02, the list's names
   left at 0.02 make each workplace step (comparison, corner search, cold start) a fault naming
   the step, 0.02 and 0.05, and no other step; the names swapped make every dial step a fault;
   the names entered pass. The drift guard given the workplaces' name is a fault (its unit
   OSL_T is of Norway and Korea); a literal dial, a dial of another fixed name, a matrix with
   no `sealed_choice` and a `--kappa` other than the matrix's are each a fault; a `--kappa`
   equal to it is not. The driver refuses the list of P4 before its log and its first step.
   The fixed names' site types are `sealed_matrix`'s.
6. Every seed fixed in the list (decision 65 answer 22): each `compare_run` entry names
   `--no-pause-seed {seed:no-pause}`, and the fixed seed is `no_pause.DEFAULT_SEED`, so the run
   computes what it computed with the code's default.
7. Outcome O2 (plan section 11; the owner's decision 79 (4) of 19 November 2026): two
   `ceiling_report` entries, rho 0 in the main setting (`data/derived/truth`, the run of
   `dev-truth`) and in the held setting (`data/derived/truth-held`, the run of
   `dev-truth-held`), each `--every-day --workers 4` on the group `nine`: the nine held-out units
   with the sealed truth run of its setting (and, held, the held setting's sealed tables) in the
   sealed run, the three development units with no held-out option in the rehearsal; time limit
   172800 s, one for every entry and stated in the `about`, whose W1 no longer puts
   `ceiling_report` off the sealed route and runs no other command of the sealed route at rho
   0.1 or 0.2. Each of the two sealed steps reaches its first read of data at a made-up tag, and
   the same start without `--every-day` is refused.
8. Rules 9 to 11 (`run_list`, module text): a held-out option in the rehearsal (rule 9); a sealed
   O2 step on development units, lacking a held-out unit, or without `--every-day` (rule 10); a
   `compare_run` or `corner_run` step giving a fixed dial name another step of its command gives,
   the two dials equal so rule 8 passes (rule 11; the review of 3d4fa78, note 2): each is one
   fault naming the step and its rule. A lone `compare_run` or `corner_run` step stands for both
   site types only where the matrix gives them one dial (rule 11, decision 103 of 22 Nov 2026): at
   workplaces 0.05 and Norway and Korea 0.02 each lone step is one fault naming both dials, and
   the list given one step per site type is not. A sealed O2 group mixing development and
   held-out units is the list's units fault, found first.
9. Rule 12 (`run_list`, module text; decision 79 (4)): an O2 step at rho 0.1 added back, the held
   step removed, the held step without `--held-out-tables` in the sealed run, the main step at
   rho 0.2, a second main step in place of the held one, and the held step without `--every-day`
   in the rehearsal are each faults naming rule 12, in each purpose where they hold; so are a
   step given a truth run of the other setting (the held step at the main sealed or development
   build, the main step at the held sealed build; the review of 2054956, finding 2) and a truth
   run given as a literal path. The two steps with no `--rho` (the command's default, 0) keep it.
10. Reading M3 of `sealed_matrix`: the three fixed-plan models (Kaplan-Meier `km`, pooled
   LightGBM `lgbm`, `cox`) are each scored by a `learn_run` on the main model's matrix, in the
   rehearsal and in the sealed run, and by no other matrix.
"""

import copy
import json
import os
from pathlib import Path

import pytest
from git_run import run_git
from test_sealed_matrix import OTHER, W, files, honest, joint

from fleetreserve import run_driver, run_list, run_verify, sealed_matrix

TOP = Path(__file__).resolve().parents[2]
LIST = TOP / "run-lists" / "run-list.json"
LIMIT = 172800  # every entry's time limit: a guard against a hang, never a planned length


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def git(folder: Path, *args: str) -> None:
    run_git(["git", "-C", str(folder), *args], capture_output=True, check=True)


def entered(document: dict, root: Path) -> dict:
    """The list with its fixed names dial-workplaces and dial-others set to the main model's
    dials per site type in its written matrix, as the list's `about` says they are entered."""
    found = copy.deepcopy(document)
    matrix = json.loads((root / "gate" / "sealed-matrix" / "lgbm_tail.json").read_text())
    for name, site_type in run_list.DIAL_NAMES.items():
        found["fixed"]["names"][name] = f"{matrix['sealed_choice']['dials'][site_type]:g}"
    return found


def copy_with(tmp_path: Path, document: dict, *, enter: bool = True, choices=None) -> Path:
    """A made-up working copy holding `document` and the matrices written from `choices` (the
    made-up kappa-choice results, `honest()` by default), its dials entered unless asked not to."""
    root = tmp_path / "copy"
    root.mkdir()
    git(root, "init", "-q")
    for path in ("gate/matrix.json", "learn/matrix.json"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((TOP / path).read_bytes())
    written = files(tmp_path, choices or honest())
    assert sealed_matrix.main([*written, "--out", str(root / "gate")]) == 0
    if enter:
        document = entered(document, root)
    path = root / "run-lists" / "run-list.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document, indent=1) + "\n")
    git(root, "add", "-A", "--", "gate", "learn", "run-lists")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "m")
    return path


def check(path: Path, capsys) -> tuple[int, list[str]]:
    code = run_list.main([str(path), "--no-starts"])
    return code, capsys.readouterr().out.splitlines()


def committed() -> dict:
    return json.loads(LIST.read_text())


def split_list() -> dict:
    """The committed list with `compare` and `corner` given once per site type, as the list held
    them before decision 103 of 22 Nov 2026: `compare-workplaces` and `corner-workplaces` at
    `{name:dial-workplaces}`, and `compare-others` and `corner-others` at `{name:dial-others}`,
    the others after `cold-start-workplaces`. Needed where the site types' dials differ, and for
    the rules on a dial name given twice."""
    found = committed()
    entries = found["entries"]
    labels = [e["label"] for e in entries]
    others = []
    for old, new in (("compare", "compare"), ("corner", "corner")):
        entry = entries[labels.index(old)]
        entry["label"] = f"{new}-workplaces"
        copied = copy.deepcopy(entry)
        copied["label"] = f"{new}-others"
        command = copied["command"]
        command[command.index("{name:dial-workplaces}")] = "{name:dial-others}"
        others.append(copied)
    labels = [e["label"] for e in entries]
    entries[
        labels.index("cold-start-workplaces") + 1 : labels.index("cold-start-workplaces") + 1
    ] = others
    return found


def test_the_committed_list_keeps_every_rule_with_the_written_matrices(tmp_path, capsys):
    code, lines = check(copy_with(tmp_path, committed()), capsys)
    assert code == 0, lines
    assert lines[-2] == "starts: the starts not made (--no-starts)"
    parsed = run_list.load(LIST)
    steps = run_list.instantiate(parsed, run_list.SEALED_RUN)
    modules = [run_list.step_module(s) for s in steps]
    assert modules[:4] == ["truth_build", "truth_build", *["heldout_truth_build"] * 2]
    learned = {}
    for s in steps:
        if run_list.step_module(s) == "learn_run":
            found = run_list.step_options(s, ("--model", "--matrix", "--interval-seed"))
            model, matrix = found["--model"][0].value, found["--matrix"][0].value
            learned.setdefault(model, []).append(matrix)
            assert found["--interval-seed"][0].value == "20261122"
    for model in sealed_matrix.FULL_METHOD_MODELS:
        assert f"gate/sealed-matrix/{model}.json" in learned[model]
    rho_matrices = sorted(m for found in learned.values() for m in found if "-rho" in m)
    assert rho_matrices == [
        f"gate/sealed-matrix/lgbm_tail-rho{r:g}.json" for r in sealed_matrix.SENSITIVITY_RHOS
    ]
    assert all(m in learned["lgbm_tail"] for m in rho_matrices)
    assert "gate/sealed-matrix/lgbm_tail-forecast-energy.json" in learned["lgbm_tail"]
    by_label = {s.label: s for s in steps}
    for r in sealed_matrix.SENSITIVITY_RHOS:
        label = f"lgbm_tail-rho{r:g}".replace(".", "_")
        intervals = by_label[f"intervals-{label}"]
        assert [found for _, found in intervals.runs] == [f"learn-{label}"]
    compare = " ".join(by_label["compare"].command)
    assert "--splits 1" in compare and "--workers 6" in compare
    corner = " ".join(by_label["corner"].command)
    assert "--days 50" in corner and "--seed 1" in corner and "--workers 6" in corner
    assert not {"compare-workplaces", "compare-others", "corner-workplaces", "corner-others"} & set(
        by_label
    )  # one step each, at the dial both site types share (decision 103)
    assert {e.max_seconds for e in parsed.entries} == {LIMIT}  # a guard against a hang only
    drift = by_label["drift-lgbm_tail"].command
    assert "--etas" not in drift and "--calibration-days" not in drift
    assert drift[drift.index("--units") + 1 :][:1] == ("OSL_T",)


def moved(document: dict, label: str, before: str) -> dict:
    found = copy.deepcopy(document)
    labels = [e["label"] for e in found["entries"]]
    entry = found["entries"].pop(labels.index(label))
    labels = [e["label"] for e in found["entries"]]
    found["entries"].insert(labels.index(before), entry)
    return found


def model(document: dict, label: str, option: str, value: str) -> dict:
    found = copy.deepcopy(document)
    (entry,) = [e for e in found["entries"] if e["label"] == label]
    command = entry["command"]
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command += [option, value]
    return found


CASES = {
    "a truth build after a learn_run (rule 6)": (
        lambda d: moved(d, "dev-truth-held", "learn-hazard_tuned_inputs"),
        "the sealed run starts with its truth builds (rule 6",
    ),
    "a development truth build between the two sealed builds (rule 6)": (
        lambda d: moved(d, "dev-truth-held", "sealed-truth-held"),
        "comes between two entries that open sealed data",
    ),
    "--model empirical on a learn_run (rule 7)": (
        lambda d: model(d, "learn-fixed-plan-km", "--model", "empirical"),
        "its --model empirical reads other days' outcomes",
    ),
    "--reference empirical on a cold_start (rule 7)": (
        lambda d: model(d, "cold-start-others", "--reference", "empirical"),
        "its --reference empirical reads other days' outcomes",
    ),
}


@pytest.mark.parametrize(("change", "line"), list(CASES.values()), ids=list(CASES))
def test_a_broken_rule_6_or_7_is_a_fault(tmp_path, capsys, change, line):
    code, lines = check(copy_with(tmp_path, change(committed())), capsys)
    assert code == 1, lines
    faults = [found for found in lines if found.startswith("FAULT: ")]
    assert len(faults) == 1 and line in faults[0], lines


def test_each_run_the_plan_gives_intervals_has_its_day_intervals_entry():
    parsed = run_list.load(LIST)
    steps = run_list.instantiate(parsed, run_list.SEALED_RUN)
    by_label = {s.label: s for s in steps}
    read = {}
    for s in steps:
        if run_list.step_module(s) == "day_intervals":
            (label,) = [found for _, found in s.runs]
            assert label not in read, label
            read[label] = s
    wanted = {
        "learn-lgbm_tail",
        "learn-hazard_tuned_inputs",
        "learn-lgbm_tail-forecast-energy",
        "learn-lgbm_tail-held",
        "learn-lgbm_tail-rho0_1",
        "learn-lgbm_tail-rho0_2",
    }
    assert set(read) == wanted
    for label, intervals in read.items():
        model = run_list.step_options(by_label[label], ("--model",))["--model"][0].value
        assert model in ("lgbm_tail", "hazard_tuned_inputs"), label
        assert intervals.setting == by_label[label].setting, label
    others = {
        s.label for s in steps if run_list.step_module(s) == "learn_run" and s.label not in wanted
    }
    assert others == {
        "learn-earlier_gbm",
        "learn-earlier_population",
        "learn-fixed-plan-km",
        "learn-fixed-plan-lgbm",
        "learn-fixed-plan-cox",
    }
    cited = '"given for the main model lgbm_tail and the neural model hazard_tuned_inputs only"'
    assert f"plan section 11, opening paragraph: intervals are {cited}" in parsed.about
    without = parsed.about[parsed.about.index("I2. ") : parsed.about.index("I3. ")]
    assert without.startswith("I2. No day_intervals entry for") and "plan section 11" in without
    for name in ("earlier_gbm", "earlier_population", "Kaplan-Meier", "pooled LightGBM", "Cox"):
        assert name in without, name


# --- 5. rule 8: each step's dial and kappa are its matrix's --------------------------------------

RULE_8 = "(rule 8, reading L19)"
WORKPLACE_STEPS = ("compare-workplaces", "corner-workplaces", "cold-start-workplaces")  # split_list
OTHER_STEPS = ("compare-others", "corner-others", "cold-start-others", "drift-lgbm_tail")


def unequal_dials() -> dict:
    """The made-up kappa-choice results with the main model's dials 0.05 (workplaces) and 0.02
    (Norway and Korea), as in probe P4."""
    found = honest()
    found["lgbm_tail"] = joint("lgbm_tail", (0.1, {W: 0.05, OTHER: 0.02}))
    return found


def dial_faults(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("FAULT: ") and RULE_8 in line]


def label_of(line: str) -> str:
    return line.split("(", 1)[1].split(")", 1)[0]


def by_label(document: dict, label: str) -> dict:
    (found,) = [e for e in document["entries"] if e["label"] == label]
    return found


def test_the_names_entered_from_the_matrix_pass_rule_8(tmp_path, capsys):
    code, lines = check(copy_with(tmp_path, split_list(), choices=unequal_dials()), capsys)
    assert code == 0, lines


def test_names_left_at_another_dial_are_a_fault_of_each_workplace_step(tmp_path, capsys):
    path = copy_with(tmp_path, split_list(), enter=False, choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    found = dial_faults(lines)
    assert found == [line for line in lines if line.startswith("FAULT: ")], lines  # rule 8 alone
    assert [label_of(line) for line in found] == list(WORKPLACE_STEPS), lines
    for line in found:
        assert "its --dial 0.02 (the fixed name dial-workplaces) is not the dial 0.05" in line
        assert "gate/sealed-matrix/lgbm_tail.json" in line, line


def test_names_swapped_are_a_fault_of_every_dial_step(tmp_path, capsys):
    document = split_list()
    document["fixed"]["names"] |= {"dial-workplaces": "0.02", "dial-others": "0.05"}
    path = copy_with(tmp_path, document, enter=False, choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    found = dial_faults(lines)
    assert sorted(label_of(line) for line in found) == sorted(WORKPLACE_STEPS + OTHER_STEPS)
    others = [line for line in found if label_of(line) in OTHER_STEPS]
    for line in others:
        assert "its --dial 0.05 (the fixed name dial-others) is not the dial 0.02" in line


def named(label: str, value: str):
    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        command = by_label(found, label)["command"]
        command[command.index("--dial") + 1] = value
        return found

    return change


def kappa(value: str):
    """A step of `dial_choice` added to the list with `--kappa value` on the main model's
    matrix."""

    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        step = copy.deepcopy(by_label(found, "learn-fixed-plan-km"))
        step["label"] = "dial-choice-kappa"
        command = step["command"]
        step["command"] = [
            *command[: command.index("-m")],
            "-m",
            "fleetreserve.dial_choice",
            "--model",
            "lgbm_tail",
            "--kappa",
            value,
        ]
        step["extra"] = {"sealed-run": ["--matrix", "gate/sealed-matrix/lgbm_tail.json"]}
        found["entries"].append(step)
        return found

    return change


RULE_8_CASES = {
    "the drift guard given the workplaces' name": (
        named("drift-lgbm_tail", "{name:dial-workplaces}"),
        "(drift-lgbm_tail): it scores OSL_T, of another site type than workplaces",
    ),
    "a literal dial on the drift guard": (
        named("drift-lgbm_tail", "0.02"),
        "(drift-lgbm_tail): its --dial 0.02 is a literal value, not one of the fixed names",
    ),
    "a dial named by another fixed name": (
        named("corner-others", "{name:base-set}"),
        "(corner-others): its --dial day-average is {name:base-set}, not one of the fixed names",
    ),
    "a kappa other than the matrix's": (
        kappa("0.3"),
        "(dial-choice-kappa): its --kappa 0.3 is not the kappa 0.1 of sealed_choice",
    ),
}


@pytest.mark.parametrize(("change", "line"), list(RULE_8_CASES.values()), ids=list(RULE_8_CASES))
def test_each_fault_of_rule_8_is_named(tmp_path, capsys, change, line):
    path = copy_with(tmp_path, change(split_list()), choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    found = dial_faults(lines)
    assert len(found) == 1 and line in found[0], lines


def test_a_kappa_equal_to_the_matrixs_is_no_fault(tmp_path, capsys):
    path = copy_with(tmp_path, kappa("0.1")(split_list()), choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 0, lines


def test_a_dial_step_on_a_matrix_with_no_sealed_choice_is_a_fault(tmp_path, capsys):
    document = split_list()
    extra = by_label(document, "compare-workplaces")["extra"]["sealed-run"]
    extra[extra.index("--matrix") + 1] = "gate/no-choice.json"
    path = copy_with(tmp_path, document, choices=unequal_dials())
    root = path.parents[1]
    matrix = json.loads((root / "gate" / "sealed-matrix" / "lgbm_tail.json").read_text())
    del matrix["sealed_choice"]
    (root / "gate" / "no-choice.json").write_text(json.dumps(matrix, indent=1) + "\n")
    git(root, "add", "gate/no-choice.json")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "n")
    code, lines = check(path, capsys)
    assert code == 1, lines
    (found,) = dial_faults(lines)
    assert "(compare-workplaces): its matrix gate/no-choice.json has no sealed_choice" in found


def test_the_driver_refuses_names_left_at_another_dial_before_its_first_step(tmp_path, capsys):
    path = copy_with(tmp_path, split_list(), enter=False, choices=unequal_dials())
    root = path.parents[1]
    listed = Path("run-lists") / "run-list.json"
    code = run_driver.drive(listed, purpose=run_list.REHEARSAL, start_dir=root)
    out = capsys.readouterr().out
    assert code == 2, out
    assert [label_of(line) for line in dial_faults(out.splitlines())] == list(WORKPLACE_STEPS)
    assert not (root / "runs.noindex").exists()  # no log, no run


def test_the_fixed_dial_names_hold_the_site_types_of_sealed_matrix():
    assert sorted(run_list.DIAL_NAMES.values()) == sorted(
        (sealed_matrix.WORKPLACES, sealed_matrix.OTHERS)
    )


def test_the_fixed_plan_models_are_scored_on_the_main_models_matrix():
    """Reading M3 of `sealed_matrix`. Red when one of the three fixed-plan entries of the
    committed list names another model's matrix in either purpose (for example
    `gate/sealed-matrix/hazard_tuned_inputs.json` in place of `.../lgbm_tail.json`), when a
    `learn_run` of `km`, `lgbm` or `cox` is added on another matrix, or when a fixed-plan entry
    is dropped."""
    parsed = run_list.load(LIST)
    fixed_plan = {
        "learn-fixed-plan-km": "km",
        "learn-fixed-plan-lgbm": "lgbm",
        "learn-fixed-plan-cox": "cox",
    }
    for purpose in run_list.PURPOSES:
        wanted = f"gate/{'sealed' if purpose == run_list.SEALED_RUN else 'rehearsal'}-matrix"
        wanted += "/lgbm_tail.json"
        steps = [
            s
            for s in run_list.instantiate(parsed, purpose)
            if run_list.step_module(s) == "learn_run"
        ]
        read = {}
        for s in steps:
            options = run_list.step_options(s, ("--model", "--matrix"))
            (model,) = [o.value for o in options["--model"]]
            (matrix,) = [o.value for o in options["--matrix"]]
            read[s.label] = (model, matrix)
        assert read["learn-lgbm_tail"] == ("lgbm_tail", wanted), purpose
        for label, model in fixed_plan.items():
            assert read[label] == (model, wanted), (purpose, label)
        for label, (model, matrix) in read.items():
            if model in fixed_plan.values():
                assert label in fixed_plan and matrix == wanted, (purpose, label)


def test_every_compare_run_entry_names_the_no_pause_seed_fixed_in_the_list():
    from fleetreserve import no_pause

    document = committed()
    compares = [e for e in document["entries"] if "fleetreserve.compare_run" in e["command"]]
    assert [e["label"] for e in compares] == ["compare"]
    for found in compares:
        command = found["command"]
        assert command.count("--no-pause-seed") == 1, found["label"]
        assert command[command.index("--no-pause-seed") + 1] == "{seed:no-pause}", found["label"]
    assert document["fixed"]["seeds"]["no-pause"] == no_pause.DEFAULT_SEED
    parsed = run_list.load(LIST)
    for purpose in run_list.PURPOSES:
        for s in run_list.instantiate(parsed, purpose):
            if run_list.step_module(s) == "compare_run":
                given = run_list.step_options(s, ("--no-pause-seed",))["--no-pause-seed"]
                assert [o.value for o in given] == [str(no_pause.DEFAULT_SEED)], s.label


# --- 7. outcome O2's six entries ----------------------------------------------------------------

O2_LABELS = ("o2-rho0", "o2-held-rho0")
REHEARSAL_UNITS = ["nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled"]
# setting -> (tables, development truth build, sealed truth build, the sealed run's own arguments)
O2_SETTINGS = {
    "main": ("data/derived/truth", "dev-truth", "sealed-truth", []),
    "held": (
        "data/derived/truth-held",
        "dev-truth-held",
        "sealed-truth-held",
        ["--held-out-tables", "data/derived/truth-heldout-held"],
    ),
}


def test_the_list_holds_o2s_two_entries_for_each_purpose():
    from fleetreserve.cold_start_gate import HELD_OUT_UNITS

    parsed = run_list.load(LIST)
    entries = [e for e in parsed.entries if "fleetreserve.ceiling_report" in e.command]
    assert tuple(e.label for e in entries) == O2_LABELS
    for purpose in run_list.PURPOSES:
        sealed = purpose == run_list.SEALED_RUN
        steps = {s.label: s for s in run_list.instantiate(parsed, purpose)}
        for label in O2_LABELS:
            setting = "held" if label.startswith("o2-held-") else "main"
            rho = label.rsplit("rho", 1)[1].replace("_", ".")
            tables, truth, sealed_truth, own = O2_SETTINGS[setting]
            units = list(HELD_OUT_UNITS) if sealed else REHEARSAL_UNITS
            step = steps[label]
            wanted = ["--units", *units, "--every-day", "--rho", rho, "--workers", "4"]
            wanted += ["--tables", tables, "--truth-run", f"{{run:{truth}}}"]
            if sealed:
                wanted += [*own, "--held-out-truth-run", f"{{run:{sealed_truth}}}"]
            assert list(step.command[run_list._module_at(step.command) + 1 :]) == wanted, label
            runs = [found for _, found in step.runs]
            assert runs == ([truth, sealed_truth] if sealed else [truth]), (purpose, label)
            assert (step.setting, step.max_seconds, step.after) == (setting, LIMIT, (truth,))
    assert sorted(HELD_OUT_UNITS) == sorted(parsed.units[run_list.SEALED_RUN]["nine"])
    assert "(ceiling_report) is not on the sealed route" not in parsed.about
    assert "no other command of the sealed route runs at those rho values" in parsed.about
    assert "decision 79 (4)" in parsed.about
    assert "172800 s (48 h), a guard against a hang only" in parsed.about


def test_the_sealed_o2_steps_reach_their_first_read_at_a_made_up_tag(tmp_path):
    from fleetreserve import run_starts

    top = copy_with(tmp_path, committed()).parents[1]
    parsed = run_list.load(top / "run-lists" / "run-list.json")
    steps = [s for s in run_list.route_steps(parsed) if s.label in O2_LABELS]
    assert [s.label for s in steps] == list(O2_LABELS)
    (tmp_path / "starts").mkdir()
    root = run_starts.made_up_repository(top, tmp_path / "starts").resolve()
    assert [run_starts.start_fault(s, root) for s in steps] == [None] * 2
    # the same start without --every-day is refused, so the check above can fail
    first = steps[0]
    place = first.command.index("--every-day")
    without = run_list.Step(
        position=first.position,
        label=first.label,
        command=first.command[:place] + first.command[place + 1 :],
        runs=tuple((k - (k > place), label) for k, label in first.runs),
        setting=first.setting,
        max_seconds=first.max_seconds,
        after=first.after,
    )
    found = run_starts.start_fault(without, root)
    assert found is not None and "(--every-day)" in found, found


# --- 8. rules 9 to 11 --------------------------------------------------------------------------


def nine() -> list[str]:
    return list(committed()["units"]["sealed-run"]["nine"])


def regroup(sealed: list[str], *, extra: bool = True):
    """`o2-rho0` given a group of its own, `o2`: the development units in the rehearsal and
    `sealed` in the sealed run, its sealed run's arguments kept or dropped."""

    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        found["units"]["rehearsal"]["o2"] = list(REHEARSAL_UNITS)
        found["units"]["sealed-run"]["o2"] = sealed
        entry = by_label(found, "o2-rho0")
        entry["units"] = "o2"
        if not extra:
            entry["extra"] = {}
        return found

    return change


def rehearsal_extra(*parts: str):
    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        by_label(found, "o2-rho0")["extra"]["rehearsal"] = list(parts)
        return found

    return change


def no_every_day(document: dict) -> dict:
    found = copy.deepcopy(document)
    by_label(found, "o2-rho0")["command"].remove("--every-day")
    return found


NEW_RULE_CASES = {
    "--held-out-tables on a rehearsal entry (rule 9)": (
        rehearsal_extra("--held-out-tables", "data/derived/truth-heldout-held"),
        "(o2-rho0): in the rehearsal it gives --held-out-tables",
        "(rule 9, reading L20)",
    ),
    "--held-out-truth-run of a development build on a rehearsal entry (rule 9)": (
        rehearsal_extra("--held-out-truth-run", "{run:dev-truth}"),
        "(o2-rho0): in the rehearsal it gives --held-out-truth-run",
        "(rule 9, reading L20)",
    ),
    "a sealed O2 step on development units with no held-out option (rule 10)": (
        regroup(list(REHEARSAL_UNITS), extra=False),
        "(o2-rho0): in the sealed run it names the development units nrel-before-may-2019, "
        "nrel-from-may-2019, norway-development-pooled",
        "(rule 10, reading L21)",
    ),
    "a sealed O2 step lacking a held-out unit (rule 10)": (
        regroup(nine()[:-1]),
        f"(o2-rho0): in the sealed run it lacks {nine()[-1]}",
        "(rule 10, reading L21)",
    ),
    "a sealed O2 step without --every-day (rule 10)": (
        no_every_day,
        "(o2-rho0): in the sealed run it gives no --every-day",
        "(rule 10, reading L21)",
    ),
    "compare_run given one dial name twice, the values equal (rule 11)": (
        named("compare-workplaces", "{name:dial-others}"),
        "(compare-others): its --dial is the fixed name dial-others, which entry 23 "
        "(compare-workplaces) of compare_run gives too",
        "(rule 11, reading L22)",
    ),
    "corner_run given one dial name twice, the values equal (rule 11)": (
        named("corner-others", "{name:dial-workplaces}"),
        "(corner-others): its --dial is the fixed name dial-workplaces, which entry 24 "
        "(corner-workplaces) of corner_run gives too",
        "(rule 11, reading L22)",
    ),
}


def equal_dials() -> dict:
    """The made-up kappa-choice results with the main model's two dials equal (0.02), so a dial
    name swapped by hand keeps rule 8 (the review of 3d4fa78, note 2)."""
    found = honest()
    found["lgbm_tail"] = joint("lgbm_tail", (0.1, {W: 0.02, OTHER: 0.02}))
    return found


def test_the_committed_list_keeps_rules_9_to_11_with_equal_dials(tmp_path, capsys):
    code, lines = check(copy_with(tmp_path, committed(), choices=equal_dials()), capsys)
    assert code == 0, lines
    (tmp_path / "split").mkdir()
    code, lines = check(copy_with(tmp_path / "split", split_list(), choices=equal_dials()), capsys)
    assert code == 0, lines  # one step per site type is still allowed


# --- 8a. a lone compare_run or corner_run step stands for both site types only at one dial -----


def test_the_committed_list_with_unequal_dials_is_refused_for_its_lone_steps(tmp_path, capsys):
    """Rule 11, decision 103: with the main model's dials 0.05 (workplaces) and 0.02 (Norway and
    Korea), one `compare` step and one `corner` step, each at the workplaces' name, leave Norway
    and Korea's units at 0.05, which is not their dial: each is one fault naming both dials."""
    path = copy_with(tmp_path, committed(), choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    faults = [line for line in lines if line.startswith("FAULT: ")]
    assert [label_of(f) for f in faults] == ["compare", "corner"], lines
    for fault, module in zip(faults, ("compare_run", "corner_run"), strict=True):
        assert f"it is the only {module} step, at the dial 0.05 of workplaces" in fault, fault
        assert "gives workplaces 0.05 and Norway and Korea 0.02" in fault, fault
        assert "gate/sealed-matrix/lgbm_tail.json" in fault, fault
        assert "(rule 11, reading L22)" in fault, fault


def test_a_lone_step_at_the_other_site_types_name_is_refused_too(tmp_path, capsys):
    document = committed()
    for label in ("compare", "corner"):
        document = named(label, "{name:dial-others}")(document)
    path = copy_with(tmp_path, document, choices=unequal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    faults = [line for line in lines if line.startswith("FAULT: ")]
    assert [label_of(f) for f in faults] == ["compare", "corner"], lines
    for fault in faults:
        assert "gives Norway and Korea 0.02 and workplaces 0.05" in fault, fault


def test_the_driver_refuses_the_lone_steps_at_unequal_dials(tmp_path, capsys):
    path = copy_with(tmp_path, committed(), choices=unequal_dials())
    root = path.parents[1]
    listed = Path("run-lists") / "run-list.json"
    code = run_driver.drive(listed, purpose=run_list.REHEARSAL, start_dir=root)
    out = capsys.readouterr().out
    assert code == 2, out
    assert out.count("(rule 11, reading L22)") == 2, out
    assert not (root / "runs.noindex").exists()  # no log, no run


@pytest.mark.parametrize(
    ("change", "line", "rule"), list(NEW_RULE_CASES.values()), ids=list(NEW_RULE_CASES)
)
def test_each_fault_of_rules_9_to_11_is_named(tmp_path, capsys, change, line, rule):
    path = copy_with(tmp_path, change(split_list()), choices=equal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    found = [f for f in lines if f.startswith("FAULT: ") and rule in f]
    assert len(found) == 1 and line in found[0], lines


def test_a_sealed_o2_group_mixing_the_kinds_is_the_units_fault(tmp_path, capsys):
    """A development unit among held-out ones is the list's units fault (module text, THE FORM),
    found before the sealed run's rules are read."""
    units = nine()
    units[units.index("OSL_T")] = "nrel-before-may-2019"
    path = copy_with(tmp_path, regroup(units)(committed()), choices=equal_dials())
    code, lines = check(path, capsys)
    assert code == 1, lines
    found = [f for f in lines if f.startswith("FAULT: ")]
    assert len(found) == 1 and found[0].startswith("FAULT: units.sealed-run.o2: mixes"), lines


# --- 9. rule 12: outcome O2's two steps -----------------------------------------------------------


def o2_added(label: str, rho: str):
    """`o2-rho0` copied as `label` at `rho`, put after the held step."""

    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        entry = copy.deepcopy(by_label(found, "o2-rho0"))
        entry["label"] = label
        entry["command"][entry["command"].index("--rho") + 1] = rho
        labels = [e["label"] for e in found["entries"]]
        found["entries"].insert(labels.index("o2-held-rho0") + 1, entry)
        return found

    return change


def o2_removed(label: str):
    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        found["entries"].remove(by_label(found, label))
        return found

    return change


def o2_changed(label: str, change_entry):
    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        change_entry(by_label(found, label))
        return found

    return change


def drop(option: str, *, purpose: str | None = None):
    """The entry with `option` and its one value removed from its command, or from the extra
    arguments of `purpose`."""

    def change_entry(entry: dict) -> None:
        parts = entry["command"] if purpose is None else entry["extra"][purpose]
        k = parts.index(option)
        del parts[k : k + 2]

    return change_entry


def rho_at(value: str):
    def change_entry(entry: dict) -> None:
        entry["command"][entry["command"].index("--rho") + 1] = value

    return change_entry


def made_main(entry: dict) -> None:
    """The held step turned into a second main step: the main setting's tables and truth runs."""
    entry["setting"] = "main"
    command = entry["command"]
    command[command.index("--tables") + 1] = "data/derived/truth"
    command[command.index("--truth-run") + 1] = "{run:dev-truth}"
    entry["extra"] = {"sealed-run": ["--held-out-truth-run", "{run:sealed-truth}"]}
    entry["after"] = ["dev-truth"]


def rehearsal_without_every_day(entry: dict) -> None:
    """`--every-day` moved from the command to the sealed run's extra arguments, so the
    rehearsal's step lacks it and the sealed run's keeps it."""
    entry["command"].remove("--every-day")
    entry["extra"]["sealed-run"] = ["--every-day", *entry["extra"]["sealed-run"]]


def run_at(old: str, new: str | None, *, literal: str | None = None):
    """The entry's `{run:<old>}`, in its command or its extra arguments, made `{run:<new>}` (or
    the `literal` path), its `after` following."""

    def change_entry(entry: dict) -> None:
        wanted = literal if new is None else f"{{run:{new}}}"
        for parts in (entry["command"], *entry["extra"].values()):
            if f"{{run:{old}}}" in parts:
                parts[parts.index(f"{{run:{old}}}")] = wanted
        entry["after"] = [new if a == old else a for a in entry["after"] if new or a != old]

    return change_entry


BOTH = ("in the rehearsal", "in the sealed run")
RULE_12 = "(rule 12, reading L23)"
O2_CASES = {
    "an O2 step at rho 0.1 added back": (
        o2_added("o2-rho0_1", "0.1"),
        [
            *(f"entry 31 (o2-rho0_1): {p} its --rho is 0.1" for p in BOTH),
            *(
                f"entry 31 (o2-rho0_1): {p} it is a second ceiling_report step of the main "
                "setting, beside entry 29 (o2-rho0)"
                for p in BOTH
            ),
        ],
    ),
    "the held step removed": (
        o2_removed("o2-held-rho0"),
        [f"entries: {p} no ceiling_report step is of the held setting" for p in BOTH],
    ),
    "the held step without --held-out-tables in the sealed run": (
        o2_changed("o2-held-rho0", drop("--held-out-tables", purpose=run_list.SEALED_RUN)),
        [
            "entry 30 (o2-held-rho0): in the sealed run its --held-out-tables is "
            "data/derived/truth-heldout (not given: the command's default), not "
            "data/derived/truth-heldout-held of its setting, held"
        ],
    ),
    "the main step at rho 0.2": (
        o2_changed("o2-rho0", rho_at("0.2")),
        [f"entry 29 (o2-rho0): {p} its --rho is 0.2" for p in BOTH],
    ),
    "a second main step in place of the held one": (
        o2_changed("o2-held-rho0", made_main),
        [
            *(
                f"entry 30 (o2-held-rho0): {p} it is a second ceiling_report step of the main "
                "setting, beside entry 29 (o2-rho0)"
                for p in BOTH
            ),
            *(f"entries: {p} no ceiling_report step is of the held setting" for p in BOTH),
        ],
    ),
    "the held step without --every-day in the rehearsal": (
        o2_changed("o2-held-rho0", rehearsal_without_every_day),
        ["entry 30 (o2-held-rho0): in the rehearsal it gives no --every-day"],
    ),
    "the held step at the main sealed truth run (review of 2054956, finding 2)": (
        o2_changed("o2-held-rho0", run_at("sealed-truth-held", "sealed-truth")),
        [
            "entry 30 (o2-held-rho0): in the sealed run its --held-out-truth-run is the run of "
            "entry 3 (sealed-truth), a build of the main setting, not of its setting, held"
        ],
    ),
    "the held step at the main development truth run": (
        o2_changed("o2-held-rho0", run_at("dev-truth-held", "dev-truth")),
        [
            f"entry 30 (o2-held-rho0): {p} its --truth-run is the run of entry 1 (dev-truth), "
            "a build of the main setting, not of its setting, held"
            for p in BOTH
        ],
    ),
    "the main step at the held sealed truth run": (
        o2_changed("o2-rho0", run_at("sealed-truth", "sealed-truth-held")),
        [
            "entry 29 (o2-rho0): in the sealed run its --held-out-truth-run is the run of "
            "entry 4 (sealed-truth-held), a build of the held setting, not of its setting, main"
        ],
    ),
    "the main step's truth run a literal path": (
        o2_changed("o2-rho0", run_at("dev-truth", None, literal="data/derived/runs/made-up")),
        [
            f"entry 29 (o2-rho0): {p} its --truth-run is 'data/derived/runs/made-up', not the "
            "run of a truth_build entry"
            for p in BOTH
        ],
    ),
}


@pytest.mark.parametrize(("change", "wanted"), list(O2_CASES.values()), ids=list(O2_CASES))
def test_each_fault_of_rule_12_is_named(tmp_path, capsys, change, wanted):
    code, lines = check(copy_with(tmp_path, change(committed()), choices=equal_dials()), capsys)
    assert code == 1, lines
    found = [f for f in lines if f.startswith("FAULT: ") and RULE_12 in f]
    assert len(found) == len(wanted), lines
    for line in wanted:
        assert sum(f.startswith(f"FAULT: {line}: ") for f in found) == 1, (line, lines)


def test_the_two_o2_steps_at_the_default_rho_keep_rule_12(tmp_path, capsys):
    document = committed()
    for label in O2_LABELS:
        document = o2_changed(label, drop("--rho"))(document)
    code, lines = check(copy_with(tmp_path, document, choices=equal_dials()), capsys)
    assert code == 0, lines


# --- 10. every rehearsal step's options in full and once, so rule 12 reads them as the command's
# parser does (reading L12; the reviews of 1644c94 and e33add3)

REHEARSAL_FORM = "in the rehearsal"
L12 = "(reading L12)"


def rehearsal_extra_of(label: str, *parts: str):
    def change_entry(entry: dict) -> None:
        entry["extra"]["rehearsal"] = list(parts)

    return o2_changed(label, change_entry)


def both_truth_names(entry: dict) -> None:
    """The held step given `--run` of the main build after its `--truth-run` (one option of
    `ceiling_report`, whose parser keeps the last)."""
    entry["command"] += ["--run", "{run:dev-truth}"]
    entry["after"] = ["dev-truth", "dev-truth-held"]


def rehearsal_both_truth_names(entry: dict) -> None:
    entry["extra"]["rehearsal"] = ["--run", "{run:dev-truth}"]
    entry["after"] = ["dev-truth", "dev-truth-held"]


TWICE = "gives --run and --truth-run, one option, 2 times"
GIVEN_ONCE = f"{REHEARSAL_FORM} an option is given once {L12}"
FORM_CASES = {  # the review of 1644c94, findings 1 and 2 (its breaks b20, b22, b09, b10)
    "b20: --rh 0.1 on the rehearsal's main step": (
        rehearsal_extra_of("o2-rho0", "--rh", "0.1"),
        [
            f"entry 29 (o2-rho0): its argument '--rh' shortens --rho: {REHEARSAL_FORM} every "
            f"option is written in full {L12}"
        ],
    ),
    "b22: --rho 0.1 after --rho 0 on the rehearsal's main step": (
        rehearsal_extra_of("o2-rho0", "--rho", "0.1"),
        [f"entry 29 (o2-rho0): gives --rho 2 times: {GIVEN_ONCE}"],
    ),
    "b09: --truth-run and --run on the held step in both purposes": (
        o2_changed("o2-held-rho0", both_truth_names),
        [
            f"entry 30 (o2-held-rho0): {TWICE}: on the sealed run an option is given once "
            "(reading L12)",
            f"entry 30 (o2-held-rho0): {TWICE}: {GIVEN_ONCE}",
        ],
    ),
    "b10: --truth-run and --run on the held step in the rehearsal": (
        o2_changed("o2-held-rho0", rehearsal_both_truth_names),
        [f"entry 30 (o2-held-rho0): {TWICE}: {GIVEN_ONCE}"],
    ),
}


@pytest.mark.parametrize(("change", "wanted"), list(FORM_CASES.values()), ids=list(FORM_CASES))
def test_each_option_form_fault_of_an_o2_step_is_named(tmp_path, capsys, change, wanted):
    code, lines = check(copy_with(tmp_path, change(committed()), choices=equal_dials()), capsys)
    assert code == 1, lines
    for line in wanted:
        assert lines.count(f"FAULT: {line}") == 1, (line, lines)
    assert len([f for f in lines if f.startswith("FAULT: ")]) == len(wanted), lines


def added_to_rehearsal(label: str, *parts: str):
    """The entry `label` given `parts` after its rehearsal's own extra arguments."""

    def change(document: dict) -> dict:
        found = copy.deepcopy(document)
        extra = by_label(found, label)["extra"]
        extra["rehearsal"] = [*extra.get("rehearsal", []), *parts]
        return found

    return change


# (entry number, label, its command, the option, a new value the parser would read, a short form
# naming that option alone): one rehearsal step of each command but `ceiling_report` (the review
# of e33add3, finding 2)
REHEARSAL_STEPS = (
    (2, "dev-truth-held", "truth_build", "--setting", "main", "--sett"),
    (5, "learn-lgbm_tail", "learn_run", "--interval-resamples", "7", "--interval-res"),
    (16, "intervals-lgbm_tail", "day_intervals", "--seed", "7", "--se"),
    (22, "ledger-lgbm_tail", "ledger_run", "--tree-days", "5", "--tree-d"),
    (23, "compare", "compare_run", "--splits", "2", "--spl"),
    (24, "corner", "corner_run", "--days", "3", "--day"),
    (25, "cold-start-workplaces", "cold_start", "--model", "earlier_gbm", "--mod"),
    (27, "drift-lgbm_tail", "drift_run", "--model", "earlier_gbm", "--mod"),
    (28, "o3", "o3_run", "--days", "3", "--day"),
)
OTHER_CASES = {}
for number, label, module, option, value, short in REHEARSAL_STEPS:
    OTHER_CASES[f"{module}: {option} given twice"] = (
        added_to_rehearsal(label, option, value),
        f"entry {number} ({label}): gives {option} 2 times: {GIVEN_ONCE}",
    )
    OTHER_CASES[f"{module}: {short} for {option}"] = (
        added_to_rehearsal(label, short, value),
        f"entry {number} ({label}): its argument '{short}' shortens {option}: {REHEARSAL_FORM} "
        f"every option is written in full {L12}",
    )


@pytest.mark.parametrize(("change", "line"), list(OTHER_CASES.values()), ids=list(OTHER_CASES))
def test_a_repeated_or_shortened_option_of_any_rehearsal_step_is_named(
    tmp_path, capsys, change, line
):
    code, lines = check(copy_with(tmp_path, change(committed()), choices=equal_dials()), capsys)
    assert code == 1, lines
    assert [f for f in lines if f.startswith("FAULT: ")] == [f"FAULT: {line}"], lines
