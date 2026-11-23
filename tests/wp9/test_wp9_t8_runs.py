"""The list of soft-scale checks that `tools/wp9_check.py t8` combines, `passmarks/wp9-t8-runs.json`
(reading W3e), on made-up recorded runs (`wp9_runs` and `test_wp9_t8` say how they are made). The
made-up repository commits the list's development runs (`wp9_runs.DEVELOPMENT_LIST_BYTES`); a test
that reads other entries commits them, and their matrices, itself.

A further run, by hand: `wp9-soft-scale-check-lgbm_tail_rotation-training`, of the model
`lgbm_tail_rotation` and the configuration `main-k0.3-empirical`, on a committed matrix
`gate/t8-training.json` whose gate units are `caltech` and `jpl`, 25 days of each (sites caltech
and jpl from 2019-03-01), every comparison passed at a resolvable step. Beside it the two listed
runs of today on 50 different development days each (as in `test_wp9_t8`), every comparison
passed: the departure is deciding only in the fractional run (50 days) and the further run (50
days), so it has 100 passed days only with the further run; cap and need have 150.

The committed training-site runs, by hand: `TRAINING_A` on `gate/t8-training-a.json` (gate units
caltech, office001, ASK and BAR) and `TRAINING_B` on `gate/t8-training-b.json` (jpl, OSL_S, TRO
and BOD), each listed at 300 days per unit; made up here with 10 days of each of their units
(sites named as the units, from 2018-05-01), every comparison passed at a resolvable step, beside
the two development runs on 50 different days each. The departure is deciding in the fractional
run (50 days) and in both training runs (40 days each): 130 passed days, the rounded run's 50 not
applicable; cap and need have 180.

1. The committed list holds the two development runs and the two training-site runs, with their
   models, configurations, matrices and days per unit; the two training matrices are committed,
   load as rotation what-ifs, score every second training unit (together each one once) and are
   equal to `gate/matrix.json` in every other field but their text.
2. A further listed run on another matrix is read like the others: its days join theirs by (site,
   date) and the item passes (0); a failed comparison in it fails the item (1); with no run of it
   the item could not run (2), naming it.
3. Days per unit (reading W3e): with the committed list and every listed run recorded, each
   training run at its listed 300 days per unit, the item passes (0); a training run at the
   command's default 100 is a trial and one at 400 cannot decide (2); an entry whose days per
   unit is 99, 150.0, true or "300" could not run (2).
4. A list that cannot decide could not run (2): a label without the prefix, with a character the
   run recorder refuses, or the prefix alone; a label listed twice; an entry without its matrix;
   an entry with a field it does not know.
5. The list is read as committed: a further entry written but not committed is not read.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_wp9_t8 as t8
import wp9_runs
from wp9_runs import record, run_item, truth

from fleetreserve import gate_config
from fleetreserve.truth_build import TRAIN_UNITS

THIRD = "wp9-soft-scale-check-lgbm_tail_rotation-training"
TRAINING = "gate/t8-training.json"
TRAINING_TEXT = json.dumps({"gate_units": ["caltech", "jpl"]})
TODAY = json.loads(wp9_runs.DEVELOPMENT_LIST_BYTES)
TRAINING_A = "wp9-soft-scale-check-lgbm_tail_rotation-training-a"
TRAINING_B = "wp9-soft-scale-check-lgbm_tail_rotation-training-b"
TRAINING_MATRICES = {TRAINING_A: "gate/t8-training-a.json", TRAINING_B: "gate/t8-training-b.json"}
HALVES = {
    TRAINING_A: ("caltech", "office001", "ASK", "BAR"),
    TRAINING_B: ("jpl", "OSL_S", "TRO", "BOD"),
}
TRAINING_PER_UNIT = 300
TRAINING_CONFIGURATION = "main-k0.3-empirical"
ENTRY = {
    "label": THIRD,
    "model": "lgbm_tail_rotation",
    "configuration": "main-k0.3-empirical",
    "matrix": TRAINING,
}
NB, NF, NP = wp9_runs.GATE_UNITS
FIRST = {NB: t8.ALL[NB][:3], NF: t8.ALL[NF][:32], NP: t8.ALL[NP][:15]}  # 50 days
SECOND = {NB: t8.ALL[NB][3:], NF: t8.ALL[NF][32:], NP: t8.ALL[NP][15:]}  # the 50 others
TRAINING_DAYS = {unit: t8._days(unit, "2019-03-01", 25) for unit in ("caltech", "jpl")}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    wp9_runs.isolate_git(monkeypatch)
    monkeypatch.setitem(t8.CONFIGURATIONS, THIRD, ENTRY["configuration"])
    for label in TRAINING_MATRICES:
        monkeypatch.setitem(t8.CONFIGURATIONS, label, TRAINING_CONFIGURATION)


@pytest.fixture
def repo(tmp_path):
    return wp9_runs.make_repo(tmp_path)


def everything(unit, k, kind, field):
    return "resolvable"


def listed(*entries) -> str:
    return json.dumps({**TODAY, "runs": [*TODAY["runs"], *entries]})


def with_third(repo):
    wp9_runs.commit_files(repo, {TRAINING: TRAINING_TEXT, wp9_runs.T8_LIST: listed(ENTRY)})


def todays_runs(repo, monkeypatch, name) -> tuple:
    one = t8.check_run(
        repo,
        monkeypatch,
        t8.FRACTIONAL,
        t8.soft(name, t8.FRACTIONAL, outcome=everything, days=FIRST),
    )
    two = t8.check_run(
        repo, monkeypatch, t8.ROUNDED, t8.soft(name, t8.ROUNDED, outcome=everything, days=SECOND)
    )
    return one, two


def third_run(repo, monkeypatch, name, outcome=everything):
    document = t8.soft(name, THIRD, outcome=outcome, days=copy.deepcopy(TRAINING_DAYS))
    document["model"] = ENTRY["model"]
    sha = hashlib.sha256(TRAINING_TEXT.encode()).hexdigest()
    document["matrix"] = {"file": "t8-training.json", "sha256": sha}
    args = ["--model", ENTRY["model"], "--configuration", ENTRY["configuration"]]
    files = {"soft-scale-check.json": json.dumps(document), "soft-scale-check-summary.txt": ""}
    code = 0 if document["passed"] else 1
    return record(
        repo, monkeypatch, THIRD, "soft_scale_check", [*args, "--matrix", TRAINING], files, code
    )


# --- 1. the committed list ------------------------------------------------------------------------


def test_the_committed_list_holds_the_development_and_training_site_runs():
    run = wp9_runs.check.T8Run
    assert wp9_runs.check.read_t8_list(wp9_runs.T8_LIST_BYTES) == (
        run(t8.FRACTIONAL, "lgbm_tail", t8.CONFIGURATIONS[t8.FRACTIONAL], "gate/matrix.json"),
        run(t8.ROUNDED, "lgbm_tail", t8.CONFIGURATIONS[t8.ROUNDED], "gate/matrix.json"),
        run(
            TRAINING_A,
            "lgbm_tail_rotation",
            TRAINING_CONFIGURATION,
            "gate/t8-training-a.json",
            TRAINING_PER_UNIT,
        ),
        run(
            TRAINING_B,
            "lgbm_tail_rotation",
            TRAINING_CONFIGURATION,
            "gate/t8-training-b.json",
            TRAINING_PER_UNIT,
        ),
    )
    assert TRAINING_PER_UNIT == 3 * wp9_runs.check.DAYS_PER_UNIT


def test_the_training_matrices_score_every_training_unit_once_in_a_rotation_whatif():
    gate = json.loads(wp9_runs.MATRIX_BYTES)
    shared = {key: value for key, value in gate.items() if key not in ("about", "gate_units")}
    scored = []
    for k, (label, path) in enumerate(TRAINING_MATRICES.items()):
        document = json.loads((wp9_runs.TOP / path).read_text())
        matrix = gate_config.load(wp9_runs.TOP / path)
        assert matrix.rotation_whatif is True
        assert matrix.gate_units == HALVES[label] == TRAIN_UNITS[k::2]
        rest = {
            key: value
            for key, value in document.items()
            if key not in ("about", "gate_units", "rotation_whatif")
        }
        assert rest == shared
        assert document["about"] != gate["about"]
        scored += matrix.gate_units
    assert sorted(scored) == sorted(TRAIN_UNITS) and len(set(scored)) == len(TRAIN_UNITS)


# --- 2. a further run -----------------------------------------------------------------------------


def test_a_further_listed_run_adds_its_days(repo, monkeypatch, capsys):
    with_third(repo)
    name = truth(repo, monkeypatch)
    one, two = todays_runs(repo, monkeypatch, name)
    three = third_run(repo, monkeypatch, name)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 0
    assert lines[1].endswith(f": {t8.FRACTIONAL}, {t8.ROUNDED}, {THIRD}")
    assert lines[4] == (
        f"  {three.name}: model lgbm_tail_rotation, configuration main-k0.3-empirical, deciding "
        "departure, cap, need; 50 days checked (caltech 25, jpl 25)"
    )
    assert lines[5].startswith("  days checked: 150 distinct days in 150 checks")
    units = f"{NB} 3, {NF} 32, {NP} 15, caltech 25, jpl 25"
    assert lines[6] == t8._line("soft departure", 100, 100, units, na=50)
    assert lines[7] == t8._line(
        "soft cap", 150, 150, f"{NB} 5, {NF} 65, {NP} 30, caltech 25, jpl 25"
    )
    assert lines[-1].endswith(f"; {one.name}, {two.name}, {three.name}")


def test_a_failure_in_a_further_run_fails(repo, monkeypatch, capsys):
    def one_failure(unit, k, kind, field):
        return "failed" if (unit, k, kind, field) == ("jpl", 2, "soft", "need") else "resolvable"

    with_third(repo)
    name = truth(repo, monkeypatch)
    todays_runs(repo, monkeypatch, name)
    three = third_run(repo, monkeypatch, name, one_failure)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[-1] == (
        f"FAILED: 1 deciding comparison failed: {three.name} | jpl | jpl 2019-03-03 | soft need"
    )


def test_a_further_run_recorded_before_its_entry_could_not_run(repo, monkeypatch, capsys):
    """Reading W3f: the matrix is committed and every run recorded before the entry is listed,
    so the list is the one path that differs."""
    wp9_runs.commit_files(repo, {TRAINING: TRAINING_TEXT})
    name = truth(repo, monkeypatch)
    todays_runs(repo, monkeypatch, name)
    three = third_run(repo, monkeypatch, name)
    earlier = wp9_runs.head(repo)
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed(ENTRY)})
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1].startswith("COULD NOT RUN: ")
    assert (
        f"{three.name} is stale: recorded at commit {earlier[:12]}, not at the checked-out commit "
        f"{wp9_runs.head(repo)[:12]}, and {wp9_runs.T8_LIST} differs between them (a path the "
        "reading tool compares too)"
    ) in lines[-1]


def test_a_further_run_not_recorded_could_not_run(repo, monkeypatch, capsys):
    with_third(repo)
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == f"NO RUN: no run named <UTC start>-{THIRD} in runs.noindex/"


# --- 3. days per unit -----------------------------------------------------------------------------


def committed_list(repo):
    """The project's list and its training matrices, committed in the made-up repository."""
    files = {path: (wp9_runs.TOP / path).read_text() for path in TRAINING_MATRICES.values()}
    files[wp9_runs.T8_LIST] = wp9_runs.T8_LIST_BYTES.decode()
    wp9_runs.commit_files(repo, files)


def training_run(repo, monkeypatch, name, label, per_unit=TRAINING_PER_UNIT):
    """A made-up run of a training-site entry (module text) at `per_unit` days per unit."""
    path = TRAINING_MATRICES[label]
    days = {unit: t8._days(unit, "2018-05-01", 10) for unit in HALVES[label]}
    document = t8.soft(name, label, outcome=everything, days=days, per_unit=per_unit)
    document["model"] = "lgbm_tail_rotation"
    sha = hashlib.sha256((wp9_runs.TOP / path).read_bytes()).hexdigest()
    document["matrix"] = {"file": Path(path).name, "sha256": sha}
    args = ["--model", "lgbm_tail_rotation", "--configuration", TRAINING_CONFIGURATION]
    args += ["--matrix", path, "--days", str(per_unit)]
    files = {"soft-scale-check.json": json.dumps(document), "soft-scale-check-summary.txt": ""}
    return record(repo, monkeypatch, label, "soft_scale_check", args, files, 0)


def test_the_committed_list_passes_with_each_run_at_its_days_per_unit(repo, monkeypatch, capsys):
    committed_list(repo)
    name = truth(repo, monkeypatch)
    one, two = todays_runs(repo, monkeypatch, name)
    a = training_run(repo, monkeypatch, name, TRAINING_A)
    b = training_run(repo, monkeypatch, name, TRAINING_B)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 0, lines
    assert lines[1].endswith(f": {t8.FRACTIONAL}, {t8.ROUNDED}, {TRAINING_A}, {TRAINING_B}")
    head = (
        "model lgbm_tail_rotation, configuration main-k0.3-empirical, deciding departure, cap, need"
    )
    assert lines[4:6] == [
        f"  {a.name}: {head}; 40 days checked (caltech 10, office001 10, ASK 10, BAR 10)",
        f"  {b.name}: {head}; 40 days checked (jpl 10, OSL_S 10, TRO 10, BOD 10)",
    ]
    assert lines[6].startswith("  days checked: 180 distinct days in 180 checks")
    training = "caltech 10, office001 10, ASK 10, BAR 10, jpl 10, OSL_S 10, TRO 10, BOD 10"
    departure = f"{NB} 3, {NF} 32, {NP} 15, {training}"
    assert lines[7] == t8._line("soft departure", 130, 130, departure, na=50)
    assert lines[8] == t8._line("soft cap", 180, 180, f"{NB} 5, {NF} 65, {NP} 30, {training}")
    assert lines[10] == t8._line("exact departure", 130, 130, departure, na=50)
    assert lines[-1].endswith(f"; {one.name}, {two.name}, {a.name}, {b.name}")


@pytest.mark.parametrize(
    ("per_unit", "says"),
    (
        (100, "is a trial: 100 days per unit, not 300"),
        (400, "checked 400 days per unit, not the 300 its entry lists"),
    ),
)
def test_a_training_run_at_other_days_per_unit_could_not_run(
    repo, monkeypatch, capsys, per_unit, says
):
    committed_list(repo)
    name = truth(repo, monkeypatch)
    todays_runs(repo, monkeypatch, name)
    a = training_run(repo, monkeypatch, name, TRAINING_A, per_unit=per_unit)
    training_run(repo, monkeypatch, name, TRAINING_B)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == f"COULD NOT RUN: {a.name} {says}"


@pytest.mark.parametrize("days", (99, 150.0, True, "300"))
def test_an_entry_whose_days_per_unit_is_not_a_whole_number_of_at_least_100_could_not_run(
    repo, monkeypatch, capsys, days
):
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed({**ENTRY, "days_per_unit": days})})
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {wp9_runs.T8_LIST}: {THIRD}'s days_per_unit {days!r} is not a whole "
        "number of at least 100"
    )


# --- 4. a list that cannot decide -----------------------------------------------------------------


@pytest.mark.parametrize(
    "label", ("wp9-other-check-lgbm_tail", "wp9-soft-scale-check-a b", "wp9-soft-scale-check-")
)
def test_a_label_not_of_the_form_could_not_run(repo, monkeypatch, capsys, label):
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed({**ENTRY, "label": label})})
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {wp9_runs.T8_LIST}: {label!r} is not a run label of the form "
        "wp9-soft-scale-check-<letters, digits, '-' or '_'>"
    )


def test_a_label_listed_twice_could_not_run(repo, monkeypatch, capsys):
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed(TODAY["runs"][0])})
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == f"COULD NOT RUN: {wp9_runs.T8_LIST} lists {t8.FRACTIONAL} more than once"


def test_an_entry_without_its_matrix_could_not_run(repo, monkeypatch, capsys):
    entry = {k: v for k, v in ENTRY.items() if k != "matrix"}
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed(entry)})
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1].startswith(
        f"COULD NOT RUN: {wp9_runs.T8_LIST}: an entry is not a label, model, configuration and "
        "matrix"
    )


def test_an_entry_with_a_field_it_does_not_know_could_not_run(repo, monkeypatch, capsys):
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: listed({**ENTRY, "seed": "2"})})
    todays_runs(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1].startswith(
        f"COULD NOT RUN: {wp9_runs.T8_LIST}: an entry is not a label, model, configuration and "
        "matrix"
    )


# --- 5. read as committed -------------------------------------------------------------------------


def test_an_entry_not_committed_is_not_read(repo, monkeypatch, capsys):
    documents = t8.passing(truth(repo, monkeypatch))
    t8.check_run(repo, monkeypatch, t8.FRACTIONAL, documents[0])
    t8.check_run(repo, monkeypatch, t8.ROUNDED, documents[1])
    (repo / wp9_runs.T8_LIST).write_text(listed(ENTRY))  # in the working tree only
    code, lines = run_item(repo, "t8", capsys)
    assert code == 0
    assert lines[1].endswith(f": {t8.FRACTIONAL}, {t8.ROUNDED}")
