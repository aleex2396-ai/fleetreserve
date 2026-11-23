"""`tools/wp9_check.py abstain`, the WP9 item "loss 0 at abstain", on made-up recorded gate runs
(`wp9_runs` says how the runs are made).

The made-up gate run (`gate`, `days`), by hand: one configuration `c1`; unit `u1` is split, with
one calibration day and two test days, day 1 offering (certified value 10 kWh) and day 2 not
offering; unit `u0` is not split. The scales chosen on `u1`: tier A fixed 0, one-way 0.5,
two-way 0.5; tier B fixed 0.3, one-way 0, two-way 0.6. Abstaining test entries (reading W1):
tier A, the fixed rung on day 1 (scale 0) and all three rungs on day 2 (the day does not offer),
4 in all; tier B, the one-way rung on day 1 and the three on day 2, 4 in all. Every one of them
has offer 0 and losses 0; the entries at a scale above 0 have losses above 0 (they are not read).

1. Every abstaining entry at offer 0 and losses 0 in both tiers: passed (0).
2. An abstaining entry with an offer or a loss that is not 0, in either tier: failed (1), from a
   run that ended with 0 and from one that ended with 1 (read as intact).
3. Could not run (2): no run; a stale run; a truth build not verified at the commit; a trial or
   a partial run; a run on another matrix, truth setting or command; a tier with no abstaining
   entry; a run that ended with 1 and shows no such entry; a days file changed after the run; an
   unexpected error.
4. The newest run under the label is the one read.
"""

import copy
import csv
import io
import json

import pytest
import wp9_runs
from wp9_runs import MATRIX_SHA, inputs, record, run_item, truth

from fleetreserve.gate_run import DAY_COLUMNS
from fleetreserve.rungs import RUNGS

LABEL = "wp6-gate"
ARGS = ["--matrix", "gate/matrix.json", "--workers", "6"]
NOT_OFFERING = "certified value at most the zero value"


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    wp9_runs.isolate_git(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return wp9_runs.make_repo(tmp_path)


SCALES = {"A": {"fixed": 0.0, "one-way": 0.5, "two-way": 0.5}}
SCALES["B"] = {"fixed": 0.3, "one-way": 0.0, "two-way": 0.6}
# (offer, slot loss, driver loss) of day 1 per tier and rung; day 2 is 0 everywhere
DAY_1 = {
    "A": {"fixed": (0.0, 0.0, 0.0), "one-way": (5.0, 0.25, 0.1), "two-way": (5.0, 0.0, 0.0)},
    "B": {"fixed": (3.0, 0.5, 0.2), "one-way": (0.0, 0.0, 0.0), "two-way": (6.0, 0.25, 0.3)},
}


def gate(truth_run, **changes) -> dict:
    """The made-up `gate.json` of the module text."""
    rungs = {
        rung: {"beta_hat": SCALES["A"][rung], "tier_b": {"beta_hat": SCALES["B"][rung]}}
        for rung in RUNGS
    }
    document = {
        "trial": False,
        "partial": False,
        "label": "gate run",
        "inputs": inputs(truth_run),
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA, "limit_days": None, "only": None},
        "configurations": {
            "c1": {
                "units": {
                    "u1": {"rungs": rungs},
                    "u0": {"reason": "not calibratable: 10 days"},
                }
            }
        },
        "failed_configurations": [],
    }
    document.update(changes)
    return document


def _row(unit, role, date, reason="") -> dict:
    row = dict.fromkeys(DAY_COLUMNS, "")
    row.update(unit=unit, configuration="c1", site="S", date=date, role=role)
    row.update(abstain_reason=reason, certified_value_kwh=repr(0.0 if reason else 10.0))
    return row


def _scored(row, figures) -> dict:
    for tier, by_rung in figures.items():
        for rung, values in by_rung.items():
            infix = "_tier_b" if tier == "B" else ""
            for field, value in zip(("offer_kwh", "slot_loss", "driver_loss"), values, strict=True):
                row[f"{rung}{infix}_{field}"] = repr(value)
    return row


def days(change=None) -> str:
    """The made-up `gate-days.csv` of the module text; `change(rows)` edits its rows first."""
    zero = {tier: dict.fromkeys(RUNGS, (0.0, 0.0, 0.0)) for tier in ("A", "B")}
    rows = [
        _row("u1", "calibration", "2020-01-01"),
        _scored(_row("u1", "test", "2020-01-02"), copy.deepcopy(DAY_1)),
        _scored(_row("u1", "test", "2020-01-03", NOT_OFFERING), zero),
        _row("u0", "not calibratable", "2020-01-04"),
    ]
    if change is not None:
        change(rows)
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(DAY_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def gate_run(repo, monkeypatch, document, table=None, *, exit_code=0, module="gate_run"):
    files = {"gate.json": json.dumps(document), "gate-days.csv": table or days()}
    return record(repo, monkeypatch, LABEL, module, ARGS, files, exit_code)


# --- 1. a pass ------------------------------------------------------------------------------------


def test_abstaining_entries_at_offer_and_losses_zero_pass(repo, monkeypatch, capsys):
    run = gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 0
    assert lines[0].startswith("WP9, loss 0 at abstain")
    assert lines[2:5] == [
        "  tier A: 4 abstaining test entries (1 at scale 0, 3 on days that do not offer) in 1 "
        "unit under 1 configuration, every one with offer 0, slot loss 0 and driver loss 0",
        "  tier B: 4 abstaining test entries (1 at scale 0, 3 on days that do not offer) in 1 "
        "unit under 1 configuration, every one with offer 0, slot loss 0 and driver loss 0",
        "  not read: 1 row of units that are not split (no scale is chosen and no loss written)",
    ]
    assert lines[-1] == (
        "passed: offer 0 and slot and driver loss exactly 0 on every abstaining test entry, 4 in "
        f"tier A and 4 in tier B; {run.name}"
    )


# --- 2. a failure ---------------------------------------------------------------------------------


def _set(rung, field, value, date="2020-01-02"):
    def change(rows):
        (row,) = [r for r in rows if r["date"] == date]
        row[f"{rung}_{field}"] = repr(value)

    return change


@pytest.mark.parametrize("exit_code", (0, 1, 2))
@pytest.mark.parametrize(
    ("change", "tier", "shown"),
    (
        (
            _set("fixed", "offer_kwh", 0.5),
            "A",
            "c1 | u1 | 2020-01-02 fixed: offer 0.5, slot loss 0, driver loss 0",
        ),
        (
            _set("one-way", "slot_loss", 0.1, "2020-01-03"),
            "A",
            "c1 | u1 | 2020-01-03 one-way: offer 0, slot loss 0.1, driver loss 0",
        ),
        (
            _set("one-way_tier_b", "driver_loss", 0.05),
            "B",
            "c1 | u1 | 2020-01-02 one-way: offer 0, slot loss 0, driver loss 0.05",
        ),
    ),
)
def test_an_abstaining_entry_with_an_offer_or_a_loss_fails(
    repo, monkeypatch, capsys, change, tier, shown, exit_code
):
    document = gate(truth(repo, monkeypatch))
    run = gate_run(repo, monkeypatch, document, days(change), exit_code=exit_code)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 1
    (line,) = [x for x in lines if x.startswith(f"  tier {tier}:")]
    assert line.endswith(f"1 not 0: {shown}")
    counts = "tier A 1, tier B 0" if tier == "A" else "tier A 0, tier B 1"
    assert lines[-1] == (
        f"FAILED: 1 abstaining test entry with an offer or a loss that is not 0 ({counts}); "
        f"{run.name}"
    )


# --- 3. could not run -----------------------------------------------------------------------------


def test_no_run_could_not_run(repo, capsys):
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2
    assert lines[-1] == "NO RUN: no run named <UTC start>-wp6-gate in runs.noindex/"


def test_a_stale_run_could_not_run(repo, monkeypatch, capsys):
    gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)))
    wp9_runs.commit_change(repo)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and "is stale" in lines[-1]


def test_a_run_from_an_earlier_commit_with_only_docs_changed_passes_and_says_so(
    repo, monkeypatch, capsys
):
    run = gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)))
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 0
    assert lines[1] == (
        f"gate run checked at commit {wp9_runs.head(repo)[:12]} (a run accepted from an earlier "
        f"commit says so on its line), on the committed gate/matrix.json ({MATRIX_SHA[:12]})"
    )
    assert lines[-1].startswith("passed: ")
    assert lines[-1].endswith(f"; {run.name}{wp9_runs.earlier_note(earlier)}")


def test_a_truth_build_not_verified_at_the_commit_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    wp9_runs.commit_change(repo)  # the truth build is now of an older commit than the gate run
    gate_run(repo, monkeypatch, gate(name))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and "is stale" in lines[-1] and name in lines[-1]
    gate_run(repo, monkeypatch, gate("20200101T000000Z-wp2-truth"))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and "20200101T000000Z-wp2-truth does not exist" in lines[-1]


@pytest.mark.parametrize(
    ("changes", "says"),
    (
        ({"trial": True}, "is a trial (--limit-days) or a partial (--only) run"),
        ({"partial": True}, "is a trial (--limit-days) or a partial (--only) run"),
        (
            {"matrix": {"file": "matrix.json", "sha256": "0" * 64}},
            "ran on another matrix",
        ),
        ({"inputs": {"setting": "held", "truth_run": None}}, "read the truth setting 'held'"),
    ),
)
def test_a_run_that_cannot_decide_could_not_run(repo, monkeypatch, capsys, changes, says):
    gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch), **changes))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]


@pytest.mark.parametrize(
    ("changes", "says"),
    (
        ({"trial": True}, "is a trial (--limit-days) or a partial (--only) run"),
        ({"matrix": {"file": "matrix.json", "sha256": "0" * 64}}, "ran on another matrix"),
        ({"inputs": {"setting": "held", "truth_run": None}}, "read the truth setting 'held'"),
        (None, "gate-days.csv of "),
    ),
    ids=["a trial", "another matrix", "held setting", "a days file changed"],
)
def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, monkeypatch, capsys, changes, says
):
    run = gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch), **(changes or {})))
    if changes is None:
        with (run / "gate-days.csv").open("a") as handle:
            handle.write("\n")
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]
    note = wp9_runs.earlier_note(earlier)
    assert f"{run.name}{note}" in lines[-1]
    assert all(f"{run.name}{note}" in line for line in lines if run.name in line)


def test_a_truth_build_from_an_earlier_commit_not_verified_is_named_with_the_note(
    repo, monkeypatch, capsys
):
    files = {"truth-summary.json": json.dumps({"setting": "main", "units": {}})}
    name = record(repo, monkeypatch, wp9_runs.TRUTH_LABEL, "truth_build", [], files, 1).name
    gate_run(repo, monkeypatch, gate(name))
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: the truth build {name}{wp9_runs.earlier_note(earlier)} is not verified"
    )


def test_a_run_of_another_command_could_not_run(repo, monkeypatch, capsys):
    gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)), module="other")
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and "does not run fleetreserve.gate_run" in lines[-1]


def test_a_tier_without_an_abstaining_entry_could_not_run(repo, monkeypatch, capsys):
    def offering(rows):
        rows[:] = [r for r in rows if r["date"] != "2020-01-03"]  # no day that does not offer
        rows[1]["one-way_tier_b_offer_kwh"] = repr(1.0)

    document = gate(truth(repo, monkeypatch))
    document["configurations"]["c1"]["units"]["u1"]["rungs"]["one-way"]["tier_b"]["beta_hat"] = 0.2
    run = gate_run(repo, monkeypatch, document, days(offering))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: no abstaining test entry in tier B, so loss 0 at abstain is not shown "
        f"there; {run.name}"
    )


def test_a_run_ended_with_exit_1_and_no_such_entry_could_not_run(repo, monkeypatch, capsys):
    document = gate(truth(repo, monkeypatch), failed_configurations=["c2 | u1: made up"])
    run = gate_run(repo, monkeypatch, document, exit_code=1)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name} ended with exit code 1 and shows no abstaining entry with an "
        "offer or a loss, so it cannot decide the item"
    )


def test_a_days_file_changed_after_the_run_could_not_run(repo, monkeypatch, capsys):
    run = gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)))
    (run / "gate-days.csv").write_text(days())  # the same rows, written again
    with (run / "gate-days.csv").open("a") as handle:
        handle.write("\n")
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2 and "gate-days.csv" in lines[-1] and "has changed" in lines[-1]


def test_an_unexpected_error_could_not_run(repo, monkeypatch, capsys):
    gate_run(repo, monkeypatch, gate(truth(repo, monkeypatch)))

    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(wp9_runs.check, "abstaining_entries", broken)
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 2
    assert lines[-1] == "COULD NOT RUN: the check stopped on RuntimeError: made up"


# --- 4. the newest run ----------------------------------------------------------------------------


def test_the_newest_run_is_the_one_read(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    gate_run(repo, monkeypatch, gate(name))
    newer = gate_run(repo, monkeypatch, gate(name), days(_set("fixed", "offer_kwh", 0.5)))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 1 and newer.name in lines[-1]
    newest = gate_run(repo, monkeypatch, gate(name))
    code, lines = run_item(repo, "abstain", capsys)
    assert code == 0 and newest.name in lines[-1]
