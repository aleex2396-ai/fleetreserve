"""`tools/wp9_check.py t8`, the WP9 item "T8's check", on made-up recorded soft-scale checks
(`wp9_runs` says how the runs are made).

The two listed runs, by hand (`fractional`, `rounded`), on the matrix's gate units, the
development units: NB, 5 days of `nrel-before-may-2019` (site NREL-A from 2019-01-01); NF, 45 days
of `nrel-from-may-2019` (site NREL-A from 2019-06-01); NP, 30 days of `norway-development-pooled`
(site OSL_1 from 2020-01-01). The fractional run checks those 80 days, deciding in departure, cap
and need; the rounded run checks the same 80 and 20 more of NF (from 2019-08-01), deciding in cap
and need (its departure is not applicable, and every one of its departure comparisons fails: they
are not read). Combined over distinct days (reading W3a) that is 100 days of 180 checks.

The comparisons ("resolvable" is passed at a resolvable step, "passed" passed at a step that is
not), and the combination per day and direction, the strongest outcome over the runs that decide
it (failed, resolvable, passed, kink, unresolved, tied):
- soft departure, fractional only: resolvable on NF's first 10 days, unresolved on the other 70;
  the rounded run's 20 extra days are not applicable;
- soft cap: resolvable everywhere, 100;
- soft need: fractional resolvable on NP, kink on NB, unresolved on NF; rounded resolvable on its
  20 extra days, unresolved on the 80 others; combined passed 50 (NF 20, NP 30), kink 5,
  unresolved 45;
- exact departure, fractional only: tied on NP, resolvable on NB, passed on NF's first 5 days,
  unresolved on NF's other 40; combined passed 10, 20 not applicable;
- exact cap: tied on NP, resolvable on the other 70;
- exact need: tied on NP; fractional resolvable on NF and unresolved on NB, rounded resolvable on
  its 20 extra days; combined passed 65, unresolved 5, tied 30.
Only soft cap has a passed comparison on 100 days, so the item cannot decide (reading W3b).

`passing` changes both runs to the same 100 days (NB 5, NF 65, NP 30), so that every direction
has a passed comparison on all 100: fractional passes everything at a resolvable step except
exact departure on NF's first 5 days (passed at a step that is not resolvable), soft need on NF's
last 20 days (unresolved) and exact need on NB (kink); rounded fails every departure comparison
(not read), passes cap on NP only, soft need on NF's last 20 days and exact need on NB, and is
unresolved or tied elsewhere. Combined: 100 passed days in every direction, exact departure with
95 of them at a resolvable step.

1. A passed comparison on 100 distinct days in each of the six directions, no deciding comparison
   failed, an MPS batch timed and the tracking logged on development units: passed (0), with the
   per-direction counts and the units they come from; the departure comes from the fractional run
   alone, the rounded run's not-applicable departure does not block it.
2. Failed (1): a deciding comparison failed, also from a run that ended with exit 1 or 2 and also
   when the other listed run cannot be read; a failed comparison in a direction that is not
   deciding in its run is not read. Also fewer than 100 days with a passed comparison in a
   direction that some run decides, with the counts and a line that the plan settles it (the
   module text's runs; one passed comparison per direction and every other one unresolved; the
   same with 99 kinks; the two runs on 50 different days each, so the departure has 50; 80 days
   passed in two runs, 160 checks), also beside a problem that alone would be a could not run
   (no MPS batch timed), and also beside a direction that no run decides (1 wins over 2).
3. Could not run (2): no run; a stale run; a truth build not verified at the commit;
   a trial (fewer days per unit) or a partial run (fewer units); a run of another command or
   configuration; no MPS batch timed; no tracking on a development unit; a run that ended with
   exit 1 and shows no failed comparison; recorded counts that do not add up; an unexpected
   error; and a direction that no listed run decides (0 days, with or without a missing MPS
   batch), when no other direction is short.
4. The newest run under each label is the one read.
"""

import copy
import json
from datetime import date, timedelta

import pytest
import wp9_runs
from wp9_runs import GATE_UNITS, MATRIX_SHA, inputs, record, run_item, truth

NB, NF, NP = GATE_UNITS  # nrel-before-may-2019, nrel-from-may-2019, norway-development-pooled
FRACTIONAL = "wp9-soft-scale-check-lgbm_tail-fractional"
ROUNDED = "wp9-soft-scale-check-lgbm_tail-rounded-eps0-02"
CONFIGURATIONS = {
    FRACTIONAL: "main-k0.3-empirical",
    ROUNDED: "main-rounded-hour-k0.3-eps0.02-empirical",
}
KINDS = ("soft", "exact")
FIELDS = ("departure", "cap", "need", "rate")
DIRECTIONS = [f"{kind} {field}" for kind in KINDS for field in ("departure", "cap", "need")]
PLAN_READING = (
    "  T8 asks 100 distinct days with a passed comparison per direction (plan section 7; the "
    "owner's decision of 18 November 2026): fewer days with a passed comparison in a direction "
    "do not satisfy it, and kink, unresolved and tied comparisons count toward none of the 100 "
    "(reading W3b)"
)


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    wp9_runs.isolate_git(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return wp9_runs.make_repo(tmp_path)


def _days(site, first, n):
    start = date.fromisoformat(first)
    return [(site, (start + timedelta(days=k)).isoformat()) for k in range(n)]


DAYS = {NB: _days("NREL-A", "2019-01-01", 5), NF: _days("NREL-A", "2019-06-01", 45)}
DAYS[NP] = _days("OSL_1", "2020-01-01", 30)
EXTRA = _days("NREL-A", "2019-08-01", 20)  # of NF, in the rounded run only
ALL = {**DAYS, NF: DAYS[NF] + EXTRA}  # 100 days


def comparison(outcome: str) -> dict:
    """A comparison record; "resolvable" is one passed at a resolvable step."""
    if outcome == "tied":
        return {"outcome": "tied"}
    resolvable = outcome == "resolvable"
    found = "passed" if resolvable else outcome
    return {
        "outcome": found,
        "non_finite": False,
        "autograd": 1.0,
        "decided_step": 1e-5 if found in ("passed", "failed") else None,
        "resolvable": resolvable,
        "relative_error": 0.0 if found in ("passed", "failed") else None,
        "steps": [],
    }


def fractional_outcome(unit, k, kind, field) -> str:
    if kind == "exact" and unit == NP:
        return "tied"
    table = {
        ("soft", "departure"): "resolvable" if unit == NF and k < 10 else "unresolved",
        ("soft", "cap"): "resolvable",
        ("soft", "need"): {NP: "resolvable", NB: "kink", NF: "unresolved"}[unit],
        ("exact", "departure"): (
            "resolvable" if unit == NB else ("passed" if k < 5 else "unresolved")
        ),
        ("exact", "cap"): "resolvable",
        ("exact", "need"): "resolvable" if unit == NF else "unresolved",
    }
    return table.get((kind, field), "unresolved")


def rounded_outcome(unit, k, kind, field) -> str:
    extra = unit == NF and k >= len(DAYS[NF])
    if field == "departure":
        return "failed"  # not applicable in this run: not read
    if kind == "exact" and unit == NP:
        return "tied"
    if field == "cap":
        return "resolvable"
    if field == "need":
        return "resolvable" if extra else "unresolved"
    return "unresolved"


def fractional_passing(unit, k, kind, field) -> str:
    last = unit == NF and k >= len(DAYS[NF])
    table = {
        ("exact", "departure"): "passed" if unit == NF and k < 5 else "resolvable",
        ("soft", "need"): "unresolved" if last else "resolvable",
        ("exact", "need"): "kink" if unit == NB else "resolvable",
    }
    return table.get((kind, field), "resolvable" if field != "rate" else "unresolved")


def rounded_passing(unit, k, kind, field) -> str:
    last = unit == NF and k >= len(DAYS[NF])
    if field == "departure":
        return "failed"  # not applicable in this run: not read
    if field == "cap":
        return "resolvable" if unit == NP else "unresolved"
    if (kind, field) == ("soft", "need"):
        return "resolvable" if last else "unresolved"
    if (kind, field) == ("exact", "need"):
        return "resolvable" if unit == NB else "tied"
    return "unresolved"


def _deciding_failed(rows, deciding) -> int:
    return sum(
        r["comparisons"][kind][field]["outcome"] == "failed"
        for r in rows
        for kind in KINDS
        for field in deciding
    )


def _not_exercised(rows, deciding) -> list:
    return [
        f"{kind} {field}"
        for kind in KINDS
        for field in FIELDS
        if field in deciding
        and not any(r["comparisons"][kind][field].get("resolvable") for r in rows)
    ]


def soft(
    truth_run, label, *, outcome=None, days=None, mps=True, per_unit=100, deciding=None
) -> dict:
    """The made-up `soft-scale-check.json` of run `label` (module text); `outcome(unit, k, kind,
    field)`, `days` (unit to (site, date) list) and `deciding` replace the module text's."""
    rounded = label == ROUNDED
    if deciding is None:
        deciding = ["cap", "need"] if rounded else ["departure", "cap", "need"]
    outcome = outcome or (rounded_outcome if rounded else fractional_outcome)
    if days is None:
        days = copy.deepcopy(ALL if rounded else DAYS)
    units, every = {}, []
    for unit, found in days.items():
        rows = [
            {
                "site": site,
                "date": day,
                "comparisons": {
                    kind: {f: comparison(outcome(unit, k, kind, f)) for f in FIELDS}
                    for kind in KINDS
                },
            }
            for k, (site, day) in enumerate(found)
        ]
        every += rows
        units[unit] = {
            "checked_days": len(rows),
            "failed_comparisons": _deciding_failed(rows, deciding),
            "tracking": {"days": len(rows)},
            "days": rows,
        }
    failed, missing = _deciding_failed(every, deciding), _not_exercised(every, deciding)
    timing = {"batch_days": 80, "seconds": 0.004} if mps else "not available"
    return {
        "label": "T8 check (plan section 7)",
        "model": "lgbm_tail",
        "configuration": CONFIGURATIONS[label],
        "settings": {
            "days_per_unit": per_unit,
            "deciding_fields": deciding,
            "not_applicable": [f for f in ("departure", "cap", "need") if f not in deciding],
        },
        "units": units,
        "checked_days": len(every),
        "batch_timing": {"cpu": {"batch_days": 80, "seconds": 0.001}, "mps": timing},
        "tracking": {"days": len(every)},
        "failed_comparisons": failed,
        "not_exercised": missing,
        "passed": failed == 0 and not missing,
        "inputs": inputs(truth_run),
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
    }


def _exit(document) -> int:
    return 0 if document["passed"] else 1


def check_run(repo, monkeypatch, label, document, *, exit_code=None, module="soft_scale_check"):
    code = _exit(document) if exit_code is None else exit_code
    args = ["--model", document["model"], "--configuration", CONFIGURATIONS[label]]
    files = {"soft-scale-check.json": json.dumps(document), "soft-scale-check-summary.txt": ""}
    return record(repo, monkeypatch, label, module, args, files, code)


def both(repo, monkeypatch, **changes) -> tuple:
    name = truth(repo, monkeypatch)
    one = check_run(repo, monkeypatch, FRACTIONAL, soft(name, FRACTIONAL, **changes))
    two = check_run(repo, monkeypatch, ROUNDED, soft(name, ROUNDED))
    return one, two


def passing(name, **changes) -> list:
    """The two documents of `passing` (module text), `changes` applied to both."""
    one = {"outcome": fractional_passing, "days": copy.deepcopy(ALL), **changes}
    two = {"outcome": rounded_passing, "days": copy.deepcopy(ALL), **changes}
    return [soft(name, FRACTIONAL, **one), soft(name, ROUNDED, **two)]


def both_passing(repo, monkeypatch, **changes) -> tuple:
    documents = passing(truth(repo, monkeypatch), **changes)
    one = check_run(repo, monkeypatch, FRACTIONAL, documents[0])
    two = check_run(repo, monkeypatch, ROUNDED, documents[1])
    return one, two


def _line(direction, passed, resolvable, units, kink=0, unresolved=0, where="none", tied=0, na=0):
    days = "day" if passed == 1 else "days"
    return (
        f"  {direction}: {passed} {days} passed ({resolvable} at a resolvable step; {units}), "
        f"{kink} kink, {unresolved} unresolved ({where}), {tied} tied, 0 failed, {na} not "
        "applicable"
    )


# --- 1. a pass ------------------------------------------------------------------------------------


def test_a_passed_comparison_on_100_days_in_every_direction_passes(repo, monkeypatch, capsys):
    one, two = both_passing(repo, monkeypatch)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 0
    assert lines[0].startswith("WP9, T8's check")
    assert lines[2:4] == [
        f"  {one.name}: model lgbm_tail, configuration main-k0.3-empirical, deciding departure, "
        f"cap, need; 100 days checked ({NB} 5, {NF} 65, {NP} 30)",
        f"  {two.name}: model lgbm_tail, configuration main-rounded-hour-k0.3-eps0.02-empirical, "
        f"deciding cap, need; 100 days checked ({NB} 5, {NF} 65, {NP} 30)",
    ]
    assert lines[4] == (
        "  days checked: 100 distinct days in 200 checks (a day checked in more than one run "
        "counts once)"
    )
    every = f"{NB} 5, {NF} 65, {NP} 30"
    assert lines[5:11] == [
        _line("soft departure", 100, 100, every),
        _line("soft cap", 100, 100, every),
        _line("soft need", 100, 100, every),
        _line("exact departure", 100, 95, every),
        _line("exact cap", 100, 100, every),
        _line("exact need", 100, 100, every),
    ]
    assert lines[11] == (
        "  a passed comparison on at least 100 distinct days in every direction (reading W3b): met"
    )
    assert PLAN_READING not in lines
    assert lines[-1] == (
        "passed: T8's check: a passed comparison on at least 100 distinct days in each of the 6 "
        "directions (soft and exact; departure, cap and need), no deciding comparison failed, an "
        f"MPS batch timed and the tracking logged on development units; {one.name}, {two.name}"
    )


# --- 2. a failure ---------------------------------------------------------------------------------


def _one_failure(unit, k, kind, field):
    if (unit, k, kind, field) == (NF, 3, "soft", "cap"):
        return "failed"
    return fractional_outcome(unit, k, kind, field)


@pytest.mark.parametrize("exit_code", (0, 1, 2))
def test_a_failed_deciding_comparison_fails(repo, monkeypatch, capsys, exit_code):
    name = truth(repo, monkeypatch)
    document = soft(name, FRACTIONAL, outcome=_one_failure)
    one = check_run(repo, monkeypatch, FRACTIONAL, document, exit_code=exit_code)
    check_run(repo, monkeypatch, ROUNDED, soft(name, ROUNDED))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[-1] == (
        f"FAILED: 1 deciding comparison failed: {one.name} | {NF} | NREL-A 2019-06-04 | soft cap"
    )


def test_a_failure_wins_over_a_run_that_cannot_be_read(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    one = check_run(repo, monkeypatch, FRACTIONAL, soft(name, FRACTIONAL, outcome=_one_failure))
    code, lines = run_item(repo, "t8", capsys)  # no rounded run
    assert code == 1 and lines[-1].startswith(f"FAILED: 1 deciding comparison failed: {one.name}")


# --- 3. could not run -----------------------------------------------------------------------------


def test_fewer_than_100_passed_days_in_a_direction_fails(repo, monkeypatch, capsys):
    one, two = both(repo, monkeypatch)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[2:4] == [
        f"  {one.name}: model lgbm_tail, configuration main-k0.3-empirical, deciding departure, "
        f"cap, need; 80 days checked ({NB} 5, {NF} 45, {NP} 30)",
        f"  {two.name}: model lgbm_tail, configuration main-rounded-hour-k0.3-eps0.02-empirical, "
        f"deciding cap, need; 100 days checked ({NB} 5, {NF} 65, {NP} 30)",
    ]
    assert lines[4] == (
        "  days checked: 100 distinct days in 180 checks (a day checked in more than one run "
        "counts once)"
    )
    assert lines[5:11] == [
        _line("soft departure", 10, 10, f"{NF} 10", 0, 70, f"{NB} 5, {NF} 35, {NP} 30", 0, 20),
        _line("soft cap", 100, 100, f"{NB} 5, {NF} 65, {NP} 30"),
        _line("soft need", 50, 50, f"{NF} 20, {NP} 30", 5, 45, f"{NF} 45"),
        _line("exact departure", 10, 5, f"{NB} 5, {NF} 5", 0, 40, f"{NF} 40", 30, 20),
        _line("exact cap", 70, 70, f"{NB} 5, {NF} 65", tied=30),
        _line("exact need", 65, 65, f"{NF} 65", 0, 5, f"{NB} 5", 30),
    ]
    assert lines[11] == (
        "  a passed comparison on at least 100 distinct days in every direction (reading W3b): not "
        "met (soft departure 10, soft need 50, exact departure 10, exact cap 70, exact need 65)"
    )
    assert lines[12] == PLAN_READING
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in soft departure 10, soft "
        "need 50, exact departure 10, exact cap 70, exact need 65"
    )


@pytest.mark.parametrize("other", ("unresolved", "kink"))
def test_one_passed_comparison_per_direction_fails(repo, monkeypatch, capsys, other):
    def once(unit, k, kind, field):  # every direction passed on NF's first day only
        return "resolvable" if (unit, k) == (NF, 0) else other

    name = truth(repo, monkeypatch)
    for label in (FRACTIONAL, ROUNDED):
        document = soft(name, label, outcome=once, days=copy.deepcopy(ALL))
        assert document["passed"]  # each run's own rule: one resolvable comparison per field
        check_run(repo, monkeypatch, label, document)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    if other == "kink":
        assert lines[6] == _line("soft cap", 1, 1, f"{NF} 1", kink=99)
    else:
        assert lines[6] == _line("soft cap", 1, 1, f"{NF} 1", 0, 99, f"{NB} 5, {NF} 64, {NP} 30")
    assert lines[12] == PLAN_READING
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in soft departure 1, soft "
        "cap 1, soft need 1, exact departure 1, exact cap 1, exact need 1"
    )


def test_two_runs_on_50_different_days_fails(repo, monkeypatch, capsys):
    first = {NB: ALL[NB][:3], NF: ALL[NF][:32], NP: ALL[NP][:15]}  # 50 days
    second = {NB: ALL[NB][3:], NF: ALL[NF][32:], NP: ALL[NP][15:]}  # the 50 others

    def everything(unit, k, kind, field):
        return "resolvable"

    name = truth(repo, monkeypatch)
    check_run(repo, monkeypatch, FRACTIONAL, soft(name, FRACTIONAL, outcome=everything, days=first))
    check_run(repo, monkeypatch, ROUNDED, soft(name, ROUNDED, outcome=everything, days=second))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[4].startswith("  days checked: 100 distinct days in 100 checks")
    assert lines[5] == _line("soft departure", 50, 50, f"{NB} 3, {NF} 32, {NP} 15", na=50)
    assert lines[6] == _line("soft cap", 100, 100, f"{NB} 5, {NF} 65, {NP} 30")
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in soft departure 50, exact "
        "departure 50"
    )


def test_a_day_passed_in_two_runs_counts_once(repo, monkeypatch, capsys):
    def everything(unit, k, kind, field):
        return "resolvable"

    name = truth(repo, monkeypatch)
    for label in (FRACTIONAL, ROUNDED):  # the same 80 days in both
        check_run(repo, monkeypatch, label, soft(name, label, outcome=everything, days=DAYS))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[4].startswith("  days checked: 80 distinct days in 160 checks")
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in "
        + ", ".join(f"{direction} 80" for direction in DIRECTIONS)
    )


def test_a_direction_deciding_in_no_run_could_not_run(repo, monkeypatch, capsys):
    both_passing(repo, monkeypatch, deciding=["cap", "need"])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[5] == _line("soft departure", 0, 0, "none", na=100)
    assert lines[-1] == (
        "COULD NOT RUN: fewer than 100 days with a passed comparison in soft departure 0 "
        "(deciding in no run), exact departure 0 (deciding in no run)"
    )


def test_a_direction_deciding_in_no_run_beside_no_mps_batch_could_not_run_and_names_both(
    repo, monkeypatch, capsys
):
    both_passing(repo, monkeypatch, deciding=["cap", "need"], mps=False)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: fewer than 100 days with a passed comparison in soft departure 0 "
        "(deciding in no run), exact departure 0 (deciding in no run); no run timed an MPS "
        "batch (MPS not available)"
    )


def test_a_direction_deciding_in_no_run_beside_a_short_one_fails_and_names_both(
    repo, monkeypatch, capsys
):
    name = truth(repo, monkeypatch)
    for label in (FRACTIONAL, ROUNDED):  # departure decides in neither run; 80 days in the others
        document = soft(
            name,
            label,
            outcome=lambda *_: "resolvable",
            days=DAYS,
            deciding=["cap", "need"],
        )
        check_run(repo, monkeypatch, label, document)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1  # 1 wins over 2 (the tool's own rule, W3b and the Exit line)
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in soft departure 0 (deciding in "
        "no run), soft cap 80, soft need 80, exact departure 0 (deciding in no run), exact cap "
        "80, exact need 80"
    )


def test_fewer_than_100_passed_days_beside_no_mps_batch_fails_and_names_both(
    repo, monkeypatch, capsys
):
    name = truth(repo, monkeypatch)
    for label in (FRACTIONAL, ROUNDED):  # the same 80 days in both, no MPS batch timed
        document = soft(name, label, outcome=lambda *_: "resolvable", days=DAYS, mps=False)
        check_run(repo, monkeypatch, label, document)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[-1] == (
        "FAILED: fewer than 100 days with a passed comparison in "
        + ", ".join(f"{direction} 80" for direction in DIRECTIONS)
        + "; no run timed an MPS batch (MPS not available)"
    )


def test_no_run_could_not_run(repo, capsys):
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        f"NO RUN: no run named <UTC start>-{FRACTIONAL} in runs.noindex/; no run named "
        f"<UTC start>-{ROUNDED} in runs.noindex/"
    )


def test_a_stale_run_could_not_run(repo, monkeypatch, capsys):
    both_passing(repo, monkeypatch)
    wp9_runs.commit_change(repo)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and "is stale" in lines[-1]


def test_runs_from_an_earlier_commit_with_only_docs_changed_pass_and_say_so(
    repo, monkeypatch, capsys
):
    one, two = both_passing(repo, monkeypatch)
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "t8", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 0
    assert lines[1] == (
        f"soft-scale checks read at commit {wp9_runs.head(repo)[:12]} (a run accepted from an "
        f"earlier commit says so on its line): {FRACTIONAL}, {ROUNDED}"
    )
    assert lines[2].startswith(f"  {one.name}{note}: model lgbm_tail, ")
    assert lines[3].startswith(f"  {two.name}{note}: model lgbm_tail, ")
    assert lines[-1].endswith(f"; {one.name}{note}, {two.name}{note}")


def test_runs_recorded_before_a_change_of_the_list_could_not_run(repo, monkeypatch, capsys):
    """Reading W3f: the list's entries unchanged, its bytes changed (indented by 2, not 1)."""
    one, two = both_passing(repo, monkeypatch)
    earlier = wp9_runs.head(repo)
    text = json.dumps(json.loads(wp9_runs.DEVELOPMENT_LIST_BYTES), indent=2)
    wp9_runs.commit_files(repo, {wp9_runs.T8_LIST: text})
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    stale = (
        f"is stale: recorded at commit {earlier[:12]}, not at the checked-out commit "
        f"{wp9_runs.head(repo)[:12]}, and {wp9_runs.T8_LIST} differs between them (a path the "
        "reading tool compares too)"
    )
    assert lines[-1].startswith("COULD NOT RUN: ")
    assert f"{one.name} {stale}" in lines[-1] and f"{two.name} {stale}" in lines[-1]


def test_a_truth_build_not_verified_at_the_commit_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    wp9_runs.commit_change(repo)
    documents = passing(name)
    check_run(repo, monkeypatch, FRACTIONAL, documents[0])
    check_run(repo, monkeypatch, ROUNDED, passing(truth(repo, monkeypatch))[1])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2 and "is stale" in lines[-1] and name in lines[-1]


def test_a_trial_or_a_partial_run_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    one = check_run(repo, monkeypatch, FRACTIONAL, soft(name, FRACTIONAL, per_unit=20))
    partial = {unit: days for unit, days in ALL.items() if unit != NB}
    two = check_run(repo, monkeypatch, ROUNDED, soft(name, ROUNDED, days=partial))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {one.name} is a trial: 20 days per unit, not 100; {two.name} is a "
        f"partial run: it checked {NF}, {NP}, not every gate unit of gate/matrix.json "
        f"({NB}, {NF}, {NP})"
    )


@pytest.mark.parametrize(
    ("change", "says"),
    (
        (lambda d: d.update(configuration="main-k0.3-oracle"), "checked the configuration"),
        (lambda d: d.update(model="lgbm"), "is of the model 'lgbm'"),
        (lambda d: d["matrix"].update(sha256="0" * 64), "ran on another matrix"),
        (lambda d: d.update(failed_comparisons=2), "the counts do not add up"),
    ),
)
def test_a_run_that_cannot_decide_could_not_run(repo, monkeypatch, capsys, change, says):
    documents = passing(truth(repo, monkeypatch))
    change(documents[0])
    check_run(repo, monkeypatch, FRACTIONAL, documents[0], exit_code=0)
    check_run(repo, monkeypatch, ROUNDED, documents[1])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]


@pytest.mark.parametrize(
    ("change", "says"),
    (
        (lambda d: d.update(configuration="main-k0.3-oracle"), "checked the configuration"),
        (lambda d: d["matrix"].update(sha256="0" * 64), "ran on another matrix"),
        (lambda d: d.update(failed_comparisons=2), "the counts do not add up"),
        (lambda d: d["settings"].update(days_per_unit=20), "is a trial: 20 days per unit"),
    ),
    ids=["another configuration", "another matrix", "counts", "a trial"],
)
def test_runs_from_an_earlier_commit_that_cannot_decide_are_named_with_the_note(
    repo, monkeypatch, capsys, change, says
):
    documents = passing(truth(repo, monkeypatch))
    change(documents[0])
    one = check_run(repo, monkeypatch, FRACTIONAL, documents[0], exit_code=0)
    two = check_run(repo, monkeypatch, ROUNDED, documents[1])
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "t8", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]
    assert f"{one.name}{note}" in lines[-1]
    for run in (one, two):
        assert all(f"{run.name}{note}" in line for line in lines if run.name in line)


def test_a_failure_in_a_run_from_an_earlier_commit_is_named_with_the_note(
    repo, monkeypatch, capsys
):
    name = truth(repo, monkeypatch)
    document = soft(name, FRACTIONAL, outcome=_one_failure)
    one = check_run(repo, monkeypatch, FRACTIONAL, document)
    check_run(repo, monkeypatch, ROUNDED, soft(name, ROUNDED))
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1
    assert lines[-1] == (
        f"FAILED: 1 deciding comparison failed: {one.name}{wp9_runs.earlier_note(earlier)} | "
        f"{NF} | NREL-A 2019-06-04 | soft cap"
    )


def test_a_run_of_another_command_could_not_run(repo, monkeypatch, capsys):
    documents = passing(truth(repo, monkeypatch))
    check_run(repo, monkeypatch, FRACTIONAL, documents[0], module="other")
    check_run(repo, monkeypatch, ROUNDED, documents[1])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2 and "does not run fleetreserve.soft_scale_check" in lines[-1]


def test_a_run_ended_with_exit_1_and_no_failure_could_not_run(repo, monkeypatch, capsys):
    def no_soft_departure(unit, k, kind, field):
        if (kind, field) == ("soft", "departure"):
            return "unresolved"
        return fractional_outcome(unit, k, kind, field)

    one, _ = both(repo, monkeypatch, outcome=no_soft_departure)  # the run itself exits 1
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {one.name} ended with exit code 1 and shows no failed comparison (not "
        "exercised: soft departure), so it cannot decide the item"
    )


def test_no_mps_batch_timed_could_not_run(repo, monkeypatch, capsys):
    both_passing(repo, monkeypatch, mps=False)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == "COULD NOT RUN: no run timed an MPS batch (MPS not available)"


def test_no_tracking_on_a_development_unit_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    documents = passing(name)
    for document in documents:
        for entry in document["units"].values():
            entry["tracking"] = {"days": 0}
    check_run(repo, monkeypatch, FRACTIONAL, documents[0])
    check_run(repo, monkeypatch, ROUNDED, documents[1])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: no run logged the tracking of the certified value on a development unit"
    )


def test_an_unexpected_error_could_not_run(repo, monkeypatch, capsys):
    both_passing(repo, monkeypatch)

    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(wp9_runs.check, "combine_days", broken)
    code, lines = run_item(repo, "t8", capsys)
    assert code == 2
    assert lines[-1] == "COULD NOT RUN: the check stopped on RuntimeError: made up"


# --- 4. the newest run ----------------------------------------------------------------------------


def test_the_newest_run_of_each_label_is_the_one_read(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    documents = passing(name)
    check_run(repo, monkeypatch, FRACTIONAL, documents[0])
    check_run(repo, monkeypatch, ROUNDED, documents[1])
    newer = check_run(repo, monkeypatch, FRACTIONAL, soft(name, FRACTIONAL, outcome=_one_failure))
    code, lines = run_item(repo, "t8", capsys)
    assert code == 1 and newer.name in lines[-1]
    newest = check_run(repo, monkeypatch, FRACTIONAL, passing(name)[0])
    code, lines = run_item(repo, "t8", capsys)
    assert code == 0 and newest.name in lines[-1]
