"""`tools/wp9_check.py spread`, the WP9 item "development loss within its 200-split spread", on
made-up recorded runs (`wp9_runs` says how the runs are made): a main truth build listing the gate
units' tables, the dial-choice run of `lgbm_tail` (label `wp6-dial-choice-lgbm_tail`) and its
comparison run at each tier A dial (label `wp10-compare-lgbm_tail-eps<dial>`), whose split table
holds, for the forecast variant, both tiers and the methods `fixed-plan` (deciding, reading W2c)
and `re-solve` (shown only), seeds 1 to 200.

The made-up split losses (the same for both methods unless a test says otherwise):

    NB (nrel-before-may-2019)       every split at scale 0: offers nothing, so no loss (W2e)
    NF (nrel-from-may-2019)         seeds 1 to 10 at scale 0; seeds 11 to 105 slot 0.05, driver
                                    0.02; seeds 106 to 200 slot 0.16, driver 0.06 (both tiers)
    NP (norway-development-pooled)  every split slot 0.08, driver 0.03 (both tiers)

Worked by hand, NF tier A slot loss: the 10 splits at scale 0 are in neither the mean nor the
count, so n = 190, the mean is (95 x 0.05 + 95 x 0.16) / 190 = 0.105, every split lies 0.055 from
it, so the sample standard deviation is 0.055 x sqrt(190 / 189) and the standard error, that over
sqrt(190), is 0.055 / sqrt(189) = 0.004001 (6 places); the level plus two standard errors is
0.10 + 0.008001 = 0.108001, and 0.105 is at most that: within. Counted with the 10 at loss 0 the
mean would be 0.09975. NF tier A driver loss: n = 190, mean 0.04, standard error 0.02 / sqrt(189)
= 0.001455, bound 0.05 + 0.002910 = 0.052910: within. NP: every split the same, standard error 0,
mean 0.08 at most 0.10 and 0.03 at most 0.05: within. NB is not a test in either tier. So the
item passes (0), on 4 tests (NF and NP, tier A slot and driver; tier B slot is 2 more: 6).

`HIGH` raises NF's slot losses in both tiers to 0.06 and 0.17: the mean 0.115 is above 0.108001
(the standard error is unchanged), so the item fails (1) on NF tier A and tier B slot.

1. Passed (0): every test within, the line of each giving the mean, the standard error, the level,
   the bound and the number of splits used; the re-solve above its bound is shown only and does
   not fail the item; two dials (one per site type) read from two comparison runs, each for its
   own units.
2. Failed (1): `HIGH`, the hand case; a failure in one run beside a dial with no run.
3. Could not run (2): no dial-choice run; no comparison run; a stale run; a run with 199 splits; a
   unit and tier with 199 rows; a split that offers with no loss written; a loss written on a
   split at scale 0; one split with a loss (no standard error); no unit with a loss in a tier
   and loss; a split table changed after the run.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import wp9_runs
from wp9_runs import GATE_UNITS, MATRIX_SHA, record, run_item

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp10"))
import truth_summary_real  # noqa: E402

from fleetreserve import compare_run, compare_splits, truth_build  # noqa: E402
from fleetreserve.gate_config import parse

NB, NF, NP = GATE_UNITS
CEILINGS = list(parse(json.loads(wp9_runs.MATRIX_BYTES)).ceilings)
VARIANTS = list(compare_run.VARIANTS)
TIERS = ("A", "B")
METHODS = ("fixed-plan", "re-solve")
DIAL = 0.02
SPLITS = 200
NOTHING = None  # a split that offers nothing: scale 0, no loss


def losses(first: float, second: float, nothing: int = 10) -> list:
    """Per seed 1 to 200: `nothing` splits that offer nothing, then half the rest at `first` and
    half at `second`."""
    half = (SPLITS - nothing) // 2
    return [NOTHING] * nothing + [first] * half + [second] * (SPLITS - nothing - half)


# (unit, tier) -> (slot losses, driver losses) per seed; the same for both methods by default.
FIGURES = {
    (NB, "A"): ([NOTHING] * SPLITS, [NOTHING] * SPLITS),
    (NB, "B"): ([NOTHING] * SPLITS, [NOTHING] * SPLITS),
    (NF, "A"): (losses(0.05, 0.16), losses(0.02, 0.06)),
    (NF, "B"): (losses(0.05, 0.16), losses(0.3, 0.3)),
    (NP, "A"): ([0.08] * SPLITS, [0.03] * SPLITS),
    (NP, "B"): ([0.08] * SPLITS, [0.3] * SPLITS),
}
HIGH = {(NF, tier): {"slot": losses(0.06, 0.17)} for tier in TIERS}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    wp9_runs.isolate_git(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return wp9_runs.make_repo(tmp_path)


def table_hash(unit: str, kind: str) -> str:
    return hashlib.sha256(f"{unit}-{kind}".encode()).hexdigest()


def inputs(truth_run: str) -> dict:
    return {
        "setting": "main",
        "truth_run": truth_run,
        "table_sha256": {
            f"{unit}-{kind}.csv": table_hash(unit, kind)
            for unit in GATE_UNITS
            for kind in ("cars", "days")
        },
    }


def truth(repo, monkeypatch, setting: str = "main") -> str:
    """A recorded truth build listing the gate units' tables, its summary the text of the real
    writer (`truth_summary_real`: no `setting` key for the main setting); its folder's name."""
    written = {
        unit: {kind: table_hash(unit, kind) for kind in ("cars", "days")} for unit in GATE_UNITS
    }
    files = {truth_build.SUMMARY_NAME: truth_summary_real.summary_text(written, setting)}
    return record(repo, monkeypatch, "wp2-truth", "truth_build", [], files).name


def dials(repo, monkeypatch, truth_run: str, workplace=DIAL, norway=DIAL):
    document = {
        "model": "lgbm_tail",
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
        "inputs": inputs(truth_run),
        "run": {"splits": SPLITS},
        "dial_choice": {
            NF: {"A": {"dial": workplace}, "B": {"dial": 0.1}},
            NP: {"A": {"dial": norway}, "B": {"dial": 0.1}},
        },
    }
    args = ["--model", "lgbm_tail", "--matrix", "gate/matrix.json", "--splits", "200"]
    files = {"dial-choice.json": json.dumps(document)}
    return record(repo, monkeypatch, "wp6-dial-choice-lgbm_tail", "dial_choice", args, files)


def split_rows(changed: dict, units) -> list[dict]:
    """The rows of the forecast variant (the only one read): `changed[(unit, tier)]` replaces the
    slot or driver losses of `FIGURES` for both methods, or, under "method", for that one."""
    rows = []
    for unit in units:
        for tier in TIERS:
            for method in METHODS:
                slot, driver = FIGURES[(unit, tier)]
                change = changed.get((unit, tier), {})
                change = {**change, **change.get(method, {})}
                slot, driver = change.get("slot", slot), change.get("driver", driver)
                for seed in range(1, SPLITS + 1):
                    loss, drive = slot[seed - 1], driver[seed - 1]
                    scale = 0.0 if loss is NOTHING else 0.5
                    rows.append(
                        {
                            "unit": unit,
                            "variant": VARIANTS[0],
                            "tier": tier,
                            "method": method,
                            "seed": seed,
                            "calibration_days": 100,
                            "test_days": 50,
                            "offering_calibration_days": 90,
                            "scale": scale,
                            "scale_zero_reason": "made up" if scale == 0 else "",
                            "offer_kwh_per_counting_test_day": 3.0 * scale,
                            "offering_test_days": 45,
                            "failing_test_days": None if scale == 0 else 2,
                            "mean_test_slot_loss": loss,
                            "mean_test_driver_loss": drive,
                            "share_of_best_causal_ceiling": None if scale == 0 else 0.2,
                            **{compare_splits.ceiling_column(c): 20.0 for c in CEILINGS},
                        }
                    )
    return rows


def comparison(
    repo,
    monkeypatch,
    tmp_path,
    truth_run,
    *,
    dial=DIAL,
    units=GATE_UNITS,
    changed=None,
    splits=SPLITS,
    rows=None,
):
    """A recorded comparison run of lgbm_tail at `dial` whose split table holds `rows` (by
    default `split_rows(changed, units)`)."""
    names = compare_run.output_names(0)
    rows = split_rows(changed or {}, units) if rows is None else rows
    document = {
        "model": "lgbm_tail",
        "dial": dial,
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
        "inputs": inputs(truth_run),
        "run": {
            "splits": splits,
            "seeds": [1, splits],
            "tiers": list(TIERS),
            "variants": VARIANTS,
            "ceilings": CEILINGS,
        },
        "units": {unit: {"status": "calibratable"} for unit in GATE_UNITS},
    }
    folder = tmp_path / f"made-{len(list(tmp_path.glob('made-*')))}"
    folder.mkdir()
    for kind, text in (
        ("results", json.dumps(document)),
        ("summary", "made-up summary\n"),
        ("table", "car,departure\n1,2.0\n"),
        ("timing", "{}\n"),
    ):
        (folder / names[kind]).write_text(text)
    compare_splits.write(rows, folder / names["splits"], CEILINGS)
    label = f"wp10-compare-lgbm_tail-eps{f'{dial:g}'.replace('.', '')}"
    args = ["--model", "lgbm_tail", "--dial", f"{dial:g}", "--matrix", "gate/matrix.json"]
    args += ["--splits", "200", "--workers", "2"]
    return record(repo, monkeypatch, label, "compare_run", args, {}, copied=folder)


def everything(repo, monkeypatch, tmp_path, **kwargs):
    name = truth(repo, monkeypatch)
    dials(repo, monkeypatch, name)
    return comparison(repo, monkeypatch, tmp_path, name, **kwargs)


def line_of(lines, start: str) -> str:
    (found,) = [line for line in lines if line.startswith(start)]
    return found


def bound(se: float, level: float) -> float:
    return level + 2.0 * se


SE_NF_SLOT = 0.055 / math.sqrt(189)  # the hand case of the module text
SE_NF_DRIVER = 0.02 / math.sqrt(189)


# --- 1. passed ------------------------------------------------------------------------------------


def test_every_test_within_passes(repo, monkeypatch, tmp_path, capsys):
    run = everything(repo, monkeypatch, tmp_path)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 0
    assert lines[0].startswith("WP9, development loss within its 200-split spread")
    assert line_of(lines, f"  {NF} tier A slot loss, fixed-plan:") == (
        f"  {NF} tier A slot loss, fixed-plan: mean 0.105000 over 190 splits with a loss (10 offer "
        f"nothing), standard error {SE_NF_SLOT:.6f}, level 0.1, level plus two standard errors "
        f"{bound(SE_NF_SLOT, 0.1):.6f}: within"
    )
    assert f"{SE_NF_SLOT:.6f}" == "0.004001" and f"{bound(SE_NF_SLOT, 0.1):.6f}" == "0.108001"
    assert line_of(lines, f"  {NF} tier A driver loss, fixed-plan:") == (
        f"  {NF} tier A driver loss, fixed-plan: mean 0.040000 over 190 splits with a loss (10 "
        f"offer nothing), standard error {SE_NF_DRIVER:.6f}, level 0.05, level plus two standard "
        f"errors {bound(SE_NF_DRIVER, 0.05):.6f}: within"
    )
    assert line_of(lines, f"  {NP} tier B slot loss, fixed-plan:") == (
        f"  {NP} tier B slot loss, fixed-plan: mean 0.080000 over 200 splits with a loss (0 offer "
        "nothing), standard error 0.000000, level 0.1, level plus two standard errors 0.100000: "
        "within"
    )
    assert line_of(lines, f"  {NB} tier A slot loss, fixed-plan:") == (
        f"  {NB} tier A slot loss, fixed-plan: not a test (none of its 200 splits offers)"
    )
    assert not any(line.startswith(f"  {NB} tier B driver loss") for line in lines)
    assert lines[-1] == (
        "passed: the mean test loss over the development splits is at most the level plus two "
        "standard errors on 6 of 6 units, tiers and losses with a loss (fixed-plan; tier A slot "
        f"and driver loss, tier B slot loss), 3 not a test; {run.name}"
    )


def test_a_real_main_truth_summary_has_no_setting_key_and_the_item_passes(
    repo, monkeypatch, tmp_path, capsys
):
    """The truth build writes `setting` only for the held setting (`truth_build.summary_text`);
    the item gets past the dials on the summary the real writer makes for the main setting."""
    run = everything(repo, monkeypatch, tmp_path)
    summary = next((repo / "runs.noindex").glob("*-wp2-truth")) / truth_build.SUMMARY_NAME
    assert "setting" not in json.loads(summary.read_text())
    code, lines = run_item(repo, "spread", capsys)
    assert code == 0
    assert lines[-1].startswith("passed: ") and lines[-1].endswith(run.name)


def test_the_re_solve_above_its_bound_is_shown_only(repo, monkeypatch, tmp_path, capsys):
    changed = {(NF, "A"): {"re-solve": {"slot": losses(0.06, 0.17)}}}
    everything(repo, monkeypatch, tmp_path, changed=changed)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 0
    line = line_of(lines, f"  {NF} tier A slot loss, re-solve (shown only, reading W2c):")
    assert "mean 0.115000 over 190 splits" in line and line.endswith(": above")
    assert lines[-1].startswith("passed: ")


def test_two_dials_are_read_from_two_runs(repo, monkeypatch, tmp_path, capsys):
    name = truth(repo, monkeypatch)
    dials(repo, monkeypatch, name, workplace=0.02, norway=0.1)
    nrel = comparison(repo, monkeypatch, tmp_path, name, dial=0.02, units=(NB, NF))
    norway = comparison(repo, monkeypatch, tmp_path, name, dial=0.1, units=(NP,))
    code, lines = run_item(repo, "spread", capsys)
    assert code == 0
    assert f"  lgbm_tail at dial 0.02: {nrel.name}, read for {NB}, {NF}" in lines
    assert f"  lgbm_tail at dial 0.1: {norway.name}, read for {NP}" in lines
    assert lines[-1].endswith(f"3 not a test; {nrel.name}, {norway.name}")


# --- 2. failed ------------------------------------------------------------------------------------


def test_a_mean_above_the_level_plus_two_standard_errors_fails(repo, monkeypatch, tmp_path, capsys):
    run = everything(repo, monkeypatch, tmp_path, changed=HIGH)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 1
    assert line_of(lines, f"  {NF} tier A slot loss, fixed-plan:") == (
        f"  {NF} tier A slot loss, fixed-plan: mean 0.115000 over 190 splits with a loss (10 offer "
        f"nothing), standard error {SE_NF_SLOT:.6f}, level 0.1, level plus two standard errors "
        f"{bound(SE_NF_SLOT, 0.1):.6f}: above"
    )
    assert lines[-1] == (
        "FAILED: the mean test loss over the development splits is above the level plus two "
        f"standard errors on 2 of 6 units, tiers and losses with a loss ({NF} tier A slot loss, "
        f"{NF} tier B slot loss; fixed-plan); {run.name}"
    )


def test_a_failure_wins_over_a_dial_with_no_run(repo, monkeypatch, tmp_path, capsys):
    name = truth(repo, monkeypatch)
    dials(repo, monkeypatch, name, workplace=0.02, norway=0.1)
    comparison(repo, monkeypatch, tmp_path, name, dial=0.02, units=(NB, NF), changed=HIGH)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 1
    assert (
        "  NO RUN: lgbm_tail at dial 0.1: no run named <UTC start>-wp10-compare-lgbm_tail-eps01 "
        "in runs.noindex/"
    ) in lines
    assert lines[-1].startswith("FAILED: ")


# --- 3. could not run -----------------------------------------------------------------------------


@pytest.mark.parametrize("setting", ["held", "elsewhere"])
def test_a_truth_build_of_another_setting_could_not_run(
    repo, monkeypatch, tmp_path, capsys, setting
):
    name = truth(repo, monkeypatch, setting)
    dials(repo, monkeypatch, name)
    comparison(repo, monkeypatch, tmp_path, name)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert setting in lines[-1] and "NO RUN" not in lines[-1]


def test_no_dial_choice_run_could_not_run(repo, capsys):
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        "NO RUN: dials of lgbm_tail: no run named <UTC start>-wp6-dial-choice-lgbm_tail in "
        "runs.noindex/"
    )


def test_no_comparison_run_could_not_run(repo, monkeypatch, capsys):
    dials(repo, monkeypatch, truth(repo, monkeypatch))
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: NO RUN: lgbm_tail at dial 0.02: no run named <UTC start>-wp10-compare-"
        "lgbm_tail-eps002 in runs.noindex/"
    )


def test_a_stale_run_could_not_run(repo, monkeypatch, tmp_path, capsys):
    everything(repo, monkeypatch, tmp_path, changed=HIGH)
    wp9_runs.commit_change(repo)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: ") and "is stale" in lines[-1]


def test_runs_from_an_earlier_commit_with_only_docs_changed_pass_and_say_so(
    repo, monkeypatch, tmp_path, capsys
):
    name = truth(repo, monkeypatch)
    dial_run = dials(repo, monkeypatch, name)
    run = comparison(repo, monkeypatch, tmp_path, name)
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "spread", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 0
    assert lines[1].startswith(
        f"comparison runs checked at commit {wp9_runs.head(repo)[:12]} (a run accepted from an "
        "earlier commit says so on its line), on the committed gate/matrix.json"
    )
    assert line_of(lines, "  dials of lgbm_tail: ").startswith(
        f"  dials of lgbm_tail: {dial_run.name}{note}: "
    )
    assert line_of(lines, "  lgbm_tail at dial ").startswith(
        f"  lgbm_tail at dial {DIAL:g}: {run.name}{note}, read for "
    )
    assert lines[-1].startswith("passed: ") and lines[-1].endswith(f"; {run.name}{note}")


@pytest.mark.parametrize("change", ("fewer splits", "split table changed"))
def test_runs_from_an_earlier_commit_that_cannot_decide_are_named_with_the_note(
    repo, monkeypatch, tmp_path, capsys, change
):
    name = truth(repo, monkeypatch)
    dial_run = dials(repo, monkeypatch, name)
    splits = 199 if change == "fewer splits" else SPLITS
    run = comparison(repo, monkeypatch, tmp_path, name, splits=splits, changed=HIGH)
    if change == "split table changed":
        path = run / compare_run.output_names(0)["splits"]
        path.write_text(path.read_text().replace(",0.17,", ",0.07,"))
    earlier = wp9_runs.docs_commit(repo)
    code, lines = run_item(repo, "spread", capsys)
    note = wp9_runs.earlier_note(earlier)
    assert code == 2 and lines[-1].startswith("COULD NOT RUN: lgbm_tail at dial 0.02: ")
    assert f"{run.name}{note}" in lines[-1]
    assert line_of(lines, "  dials of lgbm_tail: ").startswith(
        f"  dials of lgbm_tail: {dial_run.name}{note}: "
    )
    for folder in (dial_run, run):
        assert all(f"{folder.name}{note}" in line for line in lines if folder.name in line)


def test_a_run_with_fewer_splits_could_not_run(repo, monkeypatch, tmp_path, capsys):
    run = everything(repo, monkeypatch, tmp_path, splits=199, changed=HIGH)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: lgbm_tail at dial 0.02: {run.name} has 199 splits, not 200"
    )


def test_a_unit_and_tier_with_fewer_rows_could_not_run(repo, monkeypatch, tmp_path, capsys):
    rows = [
        r
        for r in split_rows(HIGH, GATE_UNITS)
        if (r["unit"], r["tier"], r["method"], r["seed"]) != (NF, "A", "fixed-plan", 200)
    ]
    run = everything(repo, monkeypatch, tmp_path, rows=rows)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: lgbm_tail at dial 0.02: {run.name}: {NF} tier A, fixed-plan: 199 splits "
        "in the split table, not the seeds 1 to 200"
    )


def test_a_split_that_offers_with_no_loss_could_not_run(repo, monkeypatch, tmp_path, capsys):
    rows = split_rows(HIGH, GATE_UNITS)
    (row,) = [
        r
        for r in rows
        if (r["unit"], r["tier"], r["method"], r["seed"]) == (NP, "B", "fixed-plan", 7)
    ]
    row["mean_test_slot_loss"] = None
    run = everything(repo, monkeypatch, tmp_path, rows=rows)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: lgbm_tail at dial 0.02: {run.name}: {NP} tier B, fixed-plan, seed 7: the "
        "split offers (scale 0.5, 45 offering test days) and its slot loss is missing"
    )


def test_a_loss_on_a_split_that_offers_nothing_could_not_run(repo, monkeypatch, tmp_path, capsys):
    rows = split_rows(HIGH, GATE_UNITS)
    (row,) = [
        r
        for r in rows
        if (r["unit"], r["tier"], r["method"], r["seed"]) == (NB, "A", "re-solve", 3)
    ]
    row["mean_test_driver_loss"] = 0.0
    run = everything(repo, monkeypatch, tmp_path, rows=rows)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        f"COULD NOT RUN: lgbm_tail at dial 0.02: {run.name}: {NB} tier A, re-solve, seed 3: the "
        "split offers nothing (scale 0, 45 offering test days) and a driver loss is written"
    )


def test_one_split_with_a_loss_has_no_standard_error(repo, monkeypatch, tmp_path, capsys):
    one = [NOTHING] * (SPLITS - 1) + [0.01]
    changed = {(NP, tier): {"slot": one, "driver": one} for tier in TIERS}
    everything(repo, monkeypatch, tmp_path, changed=changed)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert line_of(lines, f"  {NP} tier A slot loss, fixed-plan:") == (
        f"  {NP} tier A slot loss, fixed-plan: mean 0.010000 over 1 split with a loss (199 offer "
        "nothing), standard error not defined (one split with a loss)"
    )
    assert lines[-1] == (
        "COULD NOT RUN: no standard error on 3 units, tiers and losses with one split with a loss "
        f"({NP} tier A slot loss, {NP} tier A driver loss, {NP} tier B slot loss; fixed-plan)"
    )


def test_no_unit_with_a_loss_in_a_tier_could_not_run(repo, monkeypatch, tmp_path, capsys):
    nothing = [NOTHING] * SPLITS
    changed = {(u, "B"): {"slot": nothing, "driver": nothing} for u in (NF, NP)}
    everything(repo, monkeypatch, tmp_path, changed=changed)
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1] == (
        "COULD NOT RUN: no unit with a loss in tier B slot loss, so the clause is not tested there"
    )


def test_a_split_table_changed_after_the_run_could_not_run(repo, monkeypatch, tmp_path, capsys):
    run = everything(repo, monkeypatch, tmp_path, changed=HIGH)
    path = run / compare_run.output_names(0)["splits"]
    path.write_text(path.read_text().replace(",0.17,", ",0.07,"))
    code, lines = run_item(repo, "spread", capsys)
    assert code == 2
    assert lines[-1].startswith(
        f"COULD NOT RUN: lgbm_tail at dial 0.02: {path.name} of {run.name} is not verified"
    )
