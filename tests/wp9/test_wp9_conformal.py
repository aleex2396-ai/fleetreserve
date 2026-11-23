"""`tools/wp9_check.py conformal`, the WP9 decision "conformal training kept only if it raises the
development offer", on made-up recorded sizing comparisons (`wp9_runs` says how the runs are
made).

The made-up comparison (`sizing`), by hand, of `lgbm_tail_rotation` at the dial 0.02 (reading
W4e), at 19 control draws: every training unit labelled (reading W4f), each shown only with the
figures of NF below, and the matrix's gate units scored as the development units
(`development_units`, reading W4a), the same figures for both arms (kWh per counting test day; the
gains are the record's own differences; the gains by draw are the smallest gain, then each 0.25
above the one before):

    unit and tier       constant  net-0.1  ct    control mean  p5 over draws  smallest gain
    NB (not split)      -         -        -     -             -              -
    NF tier A           10        11       12    10.5          +0.5           +0.4
    NF tier B           8         8.5      9     8.2           +0.3           +0.2
    NP tier A           2         2.2      2.5   2.1           +0.1           +0.05
    NP tier B           0         0        0     0             0              0

NB is not split and NP tier B offers nothing: neither is a test (reading W4d), so 3 units and tiers
are. On each of them ct beats the constant, the net it starts from and every one of its draws (no
draw at or above it): both arms are kept. `NOT_KEPT` lowers ct-scale on NF tier A to 10.8 (below
the net) and the smallest gain of ct-abstain on NP tier A, and its 5th percentile, to -0.1 (1 of 19
draws at or above it): both arms are then not kept.

1. Decided (0): both arms kept; both not kept, per unit and tier; one of each.
2. Could not run (2): a record without gains by draw (cannot tell), also beside an arm not kept;
   gains by draw that do not add up (a count or a length that differs); 10 control draws (too few,
   even with an arm not raising); training units only (a run without development units: the
   development offer is not measured); fewer training units labelled (the sizings are not the
   finished ones, reading W4f); another model or dial; no run; a stale run; a truth build not
   verified at the commit; a trial (50 splits) or a partial run (a development unit missing); a
   run of another command; a run that ended with exit 1; an entry whose draws differ from the
   run's; an unexpected error.
3. The newest run under the label is the one read.
"""

import copy
import json

import pytest
import wp9_runs
from wp9_runs import GATE_UNITS, MATRIX_SHA, inputs, record, run_item, truth

from fleetreserve.truth_build import TRAIN_UNITS

NB, NF, NP = GATE_UNITS
LABEL = "wp9-size-resolve-whatif-ct-lgbm_tail_rotation"
ARGS = ["--model", "lgbm_tail_rotation", "--matrix", "gate/matrix.json", "--dial", "0.02"]
ARMS = ("scale", "abstain")
# (constant, net, ct, control mean, p5 over draws, smallest gain) per unit and tier; None: not split
FIGURES = {
    (NB, "A"): None,
    (NB, "B"): None,
    (NF, "A"): (10.0, 11.0, 12.0, 10.5, 0.5, 0.4),
    (NF, "B"): (8.0, 8.5, 9.0, 8.2, 0.3, 0.2),
    (NP, "A"): (2.0, 2.2, 2.5, 2.1, 0.1, 0.05),
    (NP, "B"): (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}
NOT_KEPT = {(NF, "A"): {"scale": {"ct": 10.8}}, (NP, "A"): {"abstain": {"p5": -0.1, "low": -0.1}}}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    wp9_runs.isolate_git(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return wp9_runs.make_repo(tmp_path)


def _entry(offer, constant, splits) -> dict:
    gain = None if offer is None else offer - constant
    return {
        "splits": splits,
        "mean_offer_kwh_per_counting_test_day": offer,
        "difference_from_constant": {"mean_kwh": gain, "p5_kwh": gain, "p95_kwh": gain},
    }


def _by_draw(low, draws) -> dict:
    if low is None:
        return dict.fromkeys(
            ("gain_kwh", "smallest_gain_kwh", "largest_gain_kwh", "draws_at_or_above")
        )
    gains = [low + 0.25 * k for k in range(draws)]
    return {
        "gain_kwh": gains,
        "smallest_gain_kwh": gains[0],
        "largest_gain_kwh": gains[-1],
        "draws_at_or_above": sum(g <= 0 for g in gains),
    }


def _tier(figures, draws, arms, by_draw) -> dict:
    if figures is None:
        constant, net, ct, control, p5, low = (None,) * 6
    else:
        constant, net, ct, control, p5, low = figures
    splits = 0 if figures is None else 200
    out = {"constant": _entry(constant, constant, splits)}
    for arm in ARMS:
        changed = arms.get(arm, {})
        offer, p5_arm = changed.get("ct", ct), changed.get("p5", p5)
        out[f"net-0.1-{arm}"] = _entry(net, constant, splits)
        out[f"ct-{arm}"] = _entry(offer, constant, splits)
        out[f"ct-{arm}"]["difference_from_shuffled"] = {
            "draws": draws,
            "control_mean_offer_kwh_per_counting_test_day": control,
            "mean_kwh": None if offer is None else offer - control,
            "over_draws": {"p5_kwh": p5_arm, "p95_kwh": None if p5_arm is None else p5_arm + 1.0},
        }
        if by_draw:
            found = _by_draw(changed.get("low", low), draws)
            out[f"ct-{arm}"]["difference_from_shuffled"]["by_draw"] = found
    return out


def sizing(
    truth_run,
    *,
    draws=19,
    development=GATE_UNITS,
    training=TRAIN_UNITS,
    changed=None,
    splits=200,
    by_draw=True,
) -> dict:
    """The made-up `size-resolve-whatif.json` of the module text; `changed[(unit, tier)][arm]`
    replaces ct's offer ("ct"), its 5th percentile over draws ("p5") or its smallest gain
    ("low") on a development unit; `development` the gate units scored as development units (none:
    a run without them); `training` the training units labelled; `by_draw` False leaves out the
    gains by draw."""
    changed = changed or {}
    found = {}
    for unit in training:
        found[unit] = {
            "status": "calibratable",
            "tiers": {tier: _tier(FIGURES[(NF, tier)], draws, {}, by_draw) for tier in ("A", "B")},
        }
    for unit in development:
        found[unit] = {
            "status": "calibratable" if FIGURES[(unit, "A")] else "not calibratable",
            "tiers": {
                tier: _tier(FIGURES[(unit, tier)], draws, changed.get((unit, tier), {}), by_draw)
                for tier in ("A", "B")
            },
        }
    run = {
        "model": "lgbm_tail_rotation",
        "matrix_configuration": "main-rounded-hour-k0.3-eps0.02-empirical",
        "dial": 0.02,
        "splits": splits,
        "units": list(training),
        "control_draws": draws,
    }
    if development:
        run["development_units"] = list(development)
    return {
        "label": "what-if on training units",
        "units": found,
        "pooled_by_site_type": {},
        "failed_units": [],
        "failed_controls": [],
        "run": run,
        "ct_fits": {},
        "inputs": inputs(truth_run),
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
    }


def sizing_run(
    repo, monkeypatch, document, *, exit_code=0, module="size_resolve_whatif", args=ARGS
):
    files = {"size-resolve-whatif.json": json.dumps(document)}
    return record(repo, monkeypatch, LABEL, module, args, files, exit_code)


# --- 1. decided -----------------------------------------------------------------------------------


def test_both_arms_raising_everywhere_are_kept(repo, monkeypatch, capsys):
    run = sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch)))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 0
    assert lines[0].startswith("WP9, conformal training kept only if it raises")
    assert lines[2] == (
        "  rule (readings W4b, W4c): kept when, on every development unit and tier where an offer "
        "is made, ct beats the constant, the net it starts from and every one of its 19 shuffled "
        "draws (no draw at or above it in the gains by draw); a sizing whose sizes carry nothing "
        "is kept with a chance of at most 1/20 = 5.0%"
    )
    assert (
        f"  {NF} tier B, ct-scale: offer 9.000 kWh per counting test day; over constant +1.000, "
        "over net-0.1-scale +0.500, over its shuffled control: mean +0.800, p5 over 19 draws "
        "+0.300, smallest gain +0.200, 0 of 19 draws at or above it: raises"
    ) in lines
    assert lines[-3:-1] == [
        "  ct-scale: kept: raises the offer on 3 of 3 units and tiers with an offer; 3 not a test",
        "  ct-abstain: kept: raises the offer on 3 of 3 units and tiers with an offer; 3 not a "
        "test",
    ]
    assert lines[-1] == (
        "decided: ct-scale kept, ct-abstain kept (reading W4b on the development units); keeping "
        f"is recorded, only the owner cuts; {run.name}"
    )


def test_both_arms_not_raising_somewhere_are_not_kept(repo, monkeypatch, capsys):
    run = sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=NOT_KEPT))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 0
    assert f"  {NB} tier A, ct-scale: not a test (not split)" in lines
    assert f"  {NP} tier B, ct-abstain: not a test (nothing offers)" in lines
    assert (
        f"  {NF} tier A, ct-scale: offer 10.800 kWh per counting test day; over constant +0.800, "
        "over net-0.1-scale -0.200, over its shuffled control: mean +0.300, p5 over 19 draws "
        "+0.500, smallest gain +0.400, 0 of 19 draws at or above it: does not raise (not above "
        "net-0.1-scale)"
    ) in lines
    assert (
        f"  {NP} tier A, ct-abstain: offer 2.500 kWh per counting test day; over constant +0.500, "
        "over net-0.1-abstain +0.300, over its shuffled control: mean +0.400, p5 over 19 draws "
        "-0.100, smallest gain -0.100, 1 of 19 draws at or above it: does not raise (1 of 19 "
        "draws at or above it)"
    ) in lines
    assert lines[-3:-1] == [
        f"  ct-scale: not kept: does not raise the offer on 1 of 3 units and tiers with an offer "
        f"({NF} tier A); 3 not a test",
        f"  ct-abstain: not kept: does not raise the offer on 1 of 3 units and tiers with an offer "
        f"({NP} tier A); 3 not a test",
    ]
    assert lines[-1] == (
        "decided: ct-scale not kept, ct-abstain not kept (reading W4b on the development units); "
        f"keeping is recorded, only the owner cuts; {run.name}"
    )


def test_one_arm_kept_and_one_not_kept(repo, monkeypatch, capsys):
    changed = {(NF, "A"): {"scale": {"ct": 10.8}}}
    sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=changed))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 0
    assert lines[-1].startswith("decided: ct-scale not kept, ct-abstain kept ")


# --- 2. could not run -----------------------------------------------------------------------------


def test_a_record_without_gains_by_draw_cannot_be_told(repo, monkeypatch, capsys):
    run = sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), by_draw=False))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert (
        f"  {NF} tier B, ct-scale: offer 9.000 kWh per counting test day; over constant +1.000, "
        "over net-0.1-scale +0.500, over its shuffled control: mean +0.800, p5 over 19 draws "
        "+0.300: cannot tell (the record holds no gains by draw)"
    ) in lines
    assert (
        "  ct-scale: cannot tell: raises on 0 and cannot tell on 3 of 3 units and tiers with an "
        "offer; 3 not a test"
    ) in lines
    assert lines[-1] == (
        f"COULD NOT RUN: ct-scale, ct-abstain cannot be told from {run.name}: its record holds no "
        "gains by draw, so beating every draw cannot be read"
    )


def test_one_arm_not_kept_and_one_not_told_could_not_run(repo, monkeypatch, capsys):
    changed = {(NF, "A"): {"scale": {"ct": 10.8}}}
    document = sizing(truth(repo, monkeypatch), changed=changed, by_draw=False)
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-2].startswith("  ct-abstain: cannot tell: raises on 0 and cannot tell on 3 of 3")
    assert lines[-1] == (
        f"COULD NOT RUN: ct-abstain cannot be told from {run.name}: its record holds no gains by "
        "draw, so beating every draw cannot be read"
    )


@pytest.mark.parametrize(
    ("change", "says"),
    (
        (lambda f: f.update(draws_at_or_above=1), "the gains by draw do not add up"),
        (lambda f: f.update(smallest_gain_kwh=0.3), "the gains by draw do not add up"),
        (lambda f: f["gain_kwh"].pop(), "18 gains by draw, the run 19 draws"),
    ),
)
def test_gains_by_draw_that_do_not_add_up_could_not_run(repo, monkeypatch, capsys, change, says):
    document = sizing(truth(repo, monkeypatch))
    change(document["units"][NF]["tiers"]["B"]["ct-abstain"]["difference_from_shuffled"]["by_draw"])
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1].startswith(f"COULD NOT RUN: {run.name}, {NF} tier B, ct-abstain: {says}")


def test_ten_control_draws_are_too_few_to_decide(repo, monkeypatch, capsys):
    document = sizing(truth(repo, monkeypatch), draws=10, changed=NOT_KEPT)
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: too few control draws to decide: {run.name} has 10, at least 19 needed "
        "(a sizing whose sizes carry nothing is kept with a chance of at most 1/(draws + 1), 5% "
        "at 19)"
    )


def test_training_units_only_do_not_measure_the_development_offer(repo, monkeypatch, capsys):
    document = sizing(truth(repo, monkeypatch), development=())
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert (
        "  jpl tier A, ct-scale (training unit, shown only): offer 12.000 kWh per counting test "
        "day; over constant +2.000, over net-0.1-scale +1.000, over its shuffled control: mean "
        "+1.500, p5 over 19 draws +0.500, smallest gain +0.400, 0 of 19 draws at or above it: "
        "raises"
    ) in lines
    assert lines[-1] == (
        f"COULD NOT RUN: the development offer is not measured: {run.name} scores no development "
        f"unit, none of the gate's ({NB}, {NF}, {NP}; reading W4a); the comparison scores them "
        "with --development-units"
    )


def test_the_training_units_are_shown_only(repo, monkeypatch, capsys):
    sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=NOT_KEPT))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 0
    shown = [line for line in lines if "(training unit, shown only)" in line]
    assert len(shown) == len(TRAIN_UNITS) * 2 * 2
    assert {line.split(" tier ")[0].strip() for line in shown} == set(TRAIN_UNITS)
    assert lines[-1].startswith("decided: ct-scale not kept, ct-abstain not kept ")


def test_fewer_training_units_do_not_give_finished_sizings(repo, monkeypatch, capsys):
    training = ["caltech", "jpl", "OSL_S"]
    document = sizing(truth(repo, monkeypatch), training=training, changed=NOT_KEPT)
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name} labels the training units caltech, jpl, OSL_S, not every one "
        f"({', '.join(TRAIN_UNITS)}; reading W4f), so the development units' sizings are not the "
        "finished ones fitted on every training unit"
    )


@pytest.mark.parametrize(
    ("where", "value", "says"),
    (
        ("--model", "lgbm_tail", "is of the model 'lgbm_tail', 'lgbm_tail_rotation', not"),
        ("model", "hazard_tuned", "is of the model 'hazard_tuned', 'lgbm_tail_rotation', not"),
        ("--dial", "0.03", "is at the dial '0.03' (results 0.02), not 0.02 (reading W4e)"),
        ("dial", 0.03, "is at the dial '0.02' (results 0.03), not 0.02 (reading W4e)"),
    ),
)
def test_another_model_or_dial_could_not_run(repo, monkeypatch, capsys, where, value, says):
    document, args = sizing(truth(repo, monkeypatch)), list(ARGS)
    if where.startswith("--"):
        args[args.index(where) + 1] = value
    else:
        document["run"][where] = value
    run = sizing_run(repo, monkeypatch, document, args=args)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1].startswith(f"COULD NOT RUN: {run.name} {says}")


def test_no_run_could_not_run(repo, capsys):
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == f"NO RUN: no run named <UTC start>-{LABEL} in runs.noindex/"


def test_a_stale_run_could_not_run(repo, monkeypatch, capsys):
    sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=NOT_KEPT))
    wp9_runs.commit_change(repo)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and "is stale" in lines[-1]


def test_a_run_from_an_earlier_commit_with_only_docs_changed_decides_and_says_so(
    repo, monkeypatch, capsys
):
    run = sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=NOT_KEPT))
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "conformal", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 0
    assert lines[1].startswith(
        f"sizing comparison {run.name}{note}, checked at commit {wp9_runs.head(repo)[:12]}: model "
    )
    assert lines[-1] == (
        "decided: ct-scale not kept, ct-abstain not kept (reading W4b on the development units); "
        f"keeping is recorded, only the owner cuts; {run.name}{note}"
    )


@pytest.mark.parametrize(
    ("changes", "exit_code", "dial", "says"),
    (
        ({}, 1, "0.02", "ended with exit code 1 (failed: "),
        ({"splits": 50}, 0, "0.02", "is a trial: 50 splits"),
        ({"draws": 10}, 0, "0.02", "too few control draws to decide: "),
        ({}, 0, "0.03", "is at the dial '0.03'"),
    ),
    ids=["exit 1", "a trial", "ten draws", "another dial"],
)
def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, monkeypatch, capsys, changes, exit_code, dial, says
):
    document = sizing(truth(repo, monkeypatch), changed=NOT_KEPT, **changes)
    document["failed_controls"] = [f"{NF}: ValueError: made up"] if exit_code else []
    args = list(ARGS)
    args[args.index("--dial") + 1] = dial
    run = sizing_run(repo, monkeypatch, document, exit_code=exit_code, args=args)
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "conformal", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and says in lines[-1]
    assert f"{run.name}{note}" in lines[-1]
    assert all(f"{run.name}{note}" in line for line in lines if run.name in line)


def test_a_truth_build_not_verified_at_the_commit_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    wp9_runs.commit_change(repo)
    sizing_run(repo, monkeypatch, sizing(name, changed=NOT_KEPT))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2 and "is stale" in lines[-1] and name in lines[-1]


def test_a_trial_or_a_partial_run_could_not_run(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    run = sizing_run(repo, monkeypatch, sizing(name, changed=NOT_KEPT, splits=50))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == f"COULD NOT RUN: {run.name} is a trial: 50 splits, not the plan's 200"
    document = sizing(name, changed=NOT_KEPT, development=[NB, NF])
    run = sizing_run(repo, monkeypatch, document)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name} is a partial run: it scores the development units {NB}, {NF}, "
        f"not every one of the gate ({NB}, {NF}, {NP}; reading W4a)"
    )


def test_a_run_of_another_command_could_not_run(repo, monkeypatch, capsys):
    document = sizing(truth(repo, monkeypatch), changed=NOT_KEPT)
    sizing_run(repo, monkeypatch, document, module="other")
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2 and "does not run fleetreserve.size_resolve_whatif" in lines[-1]


@pytest.mark.parametrize("exit_code", (1, 2))
def test_a_run_ended_with_exit_1_or_2_could_not_run(repo, monkeypatch, capsys, exit_code):
    document = sizing(truth(repo, monkeypatch), changed=NOT_KEPT)
    document["failed_controls"] = [f"{NF}: ValueError: made up"]
    run = sizing_run(repo, monkeypatch, document, exit_code=exit_code)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name} ended with exit code {exit_code} (failed: {NF}: ValueError: "
        "made up), "
        "so it cannot decide the item"
    )


def test_an_entry_with_other_draws_could_not_run(repo, monkeypatch, capsys):
    document = sizing(truth(repo, monkeypatch), changed=NOT_KEPT)
    shuffled = document["units"][NF]["tiers"]["B"]["ct-abstain"]["difference_from_shuffled"]
    shuffled["draws"] = 18
    run = sizing_run(repo, monkeypatch, copy.deepcopy(document))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: {run.name}, {NF} tier B, ct-abstain: its shuffled control has 18 draws, "
        "the run 19"
    )


def test_an_unexpected_error_could_not_run(repo, monkeypatch, capsys):
    sizing_run(repo, monkeypatch, sizing(truth(repo, monkeypatch), changed=NOT_KEPT))

    def broken(*args, **kwargs):
        raise RuntimeError("made up")

    monkeypatch.setattr(wp9_runs.check, "unit_and_tier", broken)
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2
    assert lines[-1] == "COULD NOT RUN: the check stopped on RuntimeError: made up"


# --- 3. the newest run ----------------------------------------------------------------------------


def test_the_newest_run_is_the_one_read(repo, monkeypatch, capsys):
    name = truth(repo, monkeypatch)
    sizing_run(repo, monkeypatch, sizing(name, changed=NOT_KEPT))
    newer = sizing_run(repo, monkeypatch, sizing(name, by_draw=False))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 2 and newer.name in lines[-1]
    newest = sizing_run(repo, monkeypatch, sizing(name, changed=NOT_KEPT))
    code, lines = run_item(repo, "conformal", capsys)
    assert code == 0 and newest.name in lines[-1]
