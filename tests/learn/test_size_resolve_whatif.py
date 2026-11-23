"""The re-solve sizing what-if command (`size_resolve_whatif`) on made-up training units
(`gate_fake.gate_cars`: 36 days of three cars each; every other unit has an early leaver on one day
in 10 to 16) and a made-up verified truth build, three units labelled (caltech, jpl, OSL_S).

1. A unit that is not a training unit is refused before any table is opened (exit 2, "COULD NOT
   RUN", no file), by the checks `size_whatif` shares.
2. `--model lgbm_tail_rotation` runs end to end with exit 0 and writes its four files: every
   sizing in both tiers (in the summary in a fixed order), evaluations counted per unit, the
   summary's first line saying it is a what-if, oracle lines that say what they are, both site
   types pooled.
3. With `constant` the offer of each split is `dial_choice`'s measurement read by
   `dial_splits.score_split` on the same prepared days and splits, for both tiers.
4. The evaluator keeps, for each (day, scale), the one-way rung's losses of `gate_day.evaluate`
   at that scale, and evaluates each pair once.
5. Raw day types are read from the labelled rows (weekday, or public holiday), None elsewhere.
6. A unit the matrix's rule does not split is not evaluated.
6b. The sizings of a unit on a hand-made loss table: a fixed sizing is divided by its largest
   size; the re-solve oracle and the multiplier get their sizes from `size_resolve_sizings` on
   the losses at size 1, each arm its own (Mondays lose above 0.4, Tuesdays never, Wednesdays at
   every scale, so "other" abstains in the arm abstain and sits at the floor in the arm scale).
7. Two runs (one worker and two workers) write byte-identical day tables, results and summaries.
8. The day-size net (`size_net`): its sizings `net-<level>-<arm>` are listed beside the boosters'
   (item 2); its predictions are the day table's `net_q<level>` columns, each equal to
   `size_net.leave_one_unit_out` on the labelled days and the cars of the labelled units, and
   its sizes are those predictions through `size_scale.sizes`; the results record its inputs, the
   record of each network fitted (per unit and level) and its pinball loss beside the booster's,
   and the summary counts the networks kept at their starting constant. Without the context
   files the command could not run (exit 2, "COULD NOT RUN", no file).
9. Conformal training (`size_ct`): its sizings `ct-<arm>` follow the net's (item 2); its outputs
   are the day table's `ct_<arm>` columns, each equal to `size_ct.leave_one_unit_out` on the
   labelled days (with their certified values) and the cars of the labelled units, and each arm's
   output becomes that arm's sizes through `size_scale.sizes`; the results record each network's
   fit (per unit and arm), and the summary counts those kept at the net they start from.
10. Shuffled controls: `shuffled` moves the relative sizes of a unit's labelled days by the
   given order, a day that does not offer under the sizing (NaN) moving with its size, by hand;
   `control_order` is one fixed permutation per unit and draw (the same twice, another for
   another unit or another draw, not the identity); on the hand-made table of 6b every net, ct
   and day-type sizing, and no other, gets its control `shuffled-<name>` with one entry per
   draw, whose relative sizes on every split and tier are the sizing's moved by that draw's
   order, and every pair they need is evaluated. In the command's run the controls are not
   sizings of the tiers (item 2), every such sizing's `difference_from_shuffled` is over
   `CONTROL_DRAWS` draws, its mean is its offer minus the control's mean offer over the draws,
   the summary gives it on the line after the sizing's, and the pairs the controls added are
   counted per unit apart from the evaluations. A draw count below 1 could not run.
11. Several draws on a made-up loss table (60 days, each failing above its own threshold, 20
   splits): a sizing whose sizes carry nothing about the days (uniform noise) is, given its
   sizes, one more random arrangement of them, so it falls inside the p5 to p95 band of its
   draws' offers with a chance of about 0.76 for 10 draws (by simulation of exchangeable values;
   0 for one draw, whose band is one point); of 20 noise sizings at least 10 sit inside their
   spread over draws (0 is between its p5 and p95), while a sizing that knows each day's
   threshold sits above it (p5 above 0) and a sizing with the same size on every day is exactly
   0 in every figure. With one draw no noise sizing sits inside (the test fails).
12. A control that fails on a unit leaves that unit's sizings, evaluations and the other units as
   they are: the unit's control is reported failed (exit 1), its sizings carry no
   `difference_from_shuffled` and neither does the pool it belongs to.
"""

import json

import numpy as np
import pandas as pd
import pytest
import size_net_fake
from gate_fake import COMMIT, gate_cars, write_build
from test_size_whatif import document

from fleetreserve import (
    dial_choice,
    dial_splits,
    gate_day,
    gate_run,
    model_inputs,
    run_verify,
    size_ct,
    size_gbm,
    size_net,
    size_net_inputs,
    size_resolve,
    size_resolve_summary,
    size_resolve_whatif,
    size_scale,
    size_whatif,
    t7_stat,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load
from fleetreserve.truth_build import TRAIN_UNITS

SPLITS = 3
DAYS = 36
UNITS = ("caltech", "jpl", "OSL_S")
NAMES = (
    "constant",
    *(f"gbm-{level:g}-{arm}" for level in (0.05, 0.1, 0.2, 0.3, 0.5) for arm in size_scale.ARMS),
    *(f"net-{level:g}-{arm}" for level in (0.05, 0.1, 0.2, 0.3, 0.5) for arm in size_scale.ARMS),
    "ct-scale",
    "ct-abstain",
    "oracle-scale",
    "oracle-abstain",
    "oracle-resolve-scale",
    "oracle-resolve-abstain",
    "day-type-scale",
    "day-type-abstain",
)
LEARNED = tuple(n for n in NAMES if n.startswith(("net-", "ct-", "day-type-")))


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        unit: gate_cars(DAYS, unit, early_every=0 if j % 2 == 0 else 9 + j)
        for j, unit in enumerate(TRAIN_UNITS)
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document()))
    return {"run": run, "tables": tables, "matrix": matrix}


def arguments(build, *extra):
    return [
        "--model",
        "lgbm_tail_rotation",
        "--dial",
        "0.03",
        "--matrix",
        str(build["matrix"]),
        "--tables",
        str(build["tables"]),
        "--truth-run",
        str(build["run"]),
        "--splits",
        str(SPLITS),
        *extra,
    ]


def files(folder) -> dict:
    names = (
        size_resolve_whatif.DAYS_NAME,
        size_resolve_whatif.RESULTS_NAME,
        size_resolve_whatif.SUMMARY_NAME,
    )
    return {name: (folder / name).read_bytes() for name in names}


FAILING = "jpl"  # the unit whose controls fail in the run "controls fail" (item 12)


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    out = {}
    real = size_resolve_whatif.control_sizings

    def failing(data, *rest):
        if data.name == FAILING:
            raise RuntimeError("made-up control failure")
        return real(data, *rest)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        patch.setattr(size_net_inputs, "load_context", size_net_fake.context)
        for workers in (1, 2, "controls fail"):
            folder = tmp_path_factory.mktemp(f"workers{workers}".replace(" ", "-"))
            patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
            count = 1 if workers == "controls fail" else workers
            argv = arguments(build, "--units", *UNITS, "--workers", str(count))
            if workers == "controls fail":
                patch.setattr(size_resolve_whatif, "control_sizings", failing)
            out[workers] = (size_resolve_whatif.main(argv), folder)
    return out


# --- 1. ----------------------------------------------------------------------------------------


def test_a_unit_that_is_not_a_training_unit_is_refused(build, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    argv = arguments(build, "--units", "nrel-from-may-2019")
    argv[argv.index("--tables") + 1] = str(tmp_path / "no-such-folder")
    code = size_resolve_whatif.main(argv)
    text = capsys.readouterr().out
    assert code == 2 and text.startswith("COULD NOT RUN") and "are not training units" in text
    assert list(folder.iterdir()) == []


# --- 2. ----------------------------------------------------------------------------------------


def test_the_command_runs_and_writes_its_files(runs):
    code, folder = runs[1]
    assert code == 0
    assert (folder / size_resolve_whatif.TIMING_NAME).exists()
    results = json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text())
    assert results["failed_units"] == [] and set(results["units"]) == set(UNITS)
    assert results["run"]["rung"] == dial_choice.RUNG
    assert results["run"]["control_draws"] == size_resolve_whatif.CONTROL_DRAWS
    assert results["failed_controls"] == []
    for entry in results["units"].values():
        assert entry["evaluations"]["day_scale_pairs"] > 0
        for tier in size_scale.TIERS:
            assert set(entry["tiers"][tier]) == set(NAMES)
            assert entry["tiers"][tier]["constant"]["splits"] == SPLITS
    assert set(results["pooled_by_site_type"]) == {"workplace", "other"}
    summary = (folder / size_resolve_whatif.SUMMARY_NAME).read_text().splitlines()
    assert summary[0] == size_resolve_summary.WHAT
    oracle = [line for line in summary if line.strip().startswith("oracle-")]
    assert len(oracle) == 4 * 2 * 5  # four oracle sizings, two tiers, three units and two pools
    assert all(size_resolve_summary.ORACLE_NOTE in line for line in oracle)
    first = summary.index("  tier A:")
    order = [
        line.split(":")[0].split(" (")[0].strip()
        for line in summary[first + 1 :]
        if not line.strip().startswith("against its shuffled control")
    ]
    assert tuple(order[: len(NAMES)]) == NAMES and order[len(NAMES)] == "tier B"


# --- 3. ----------------------------------------------------------------------------------------


def prepared_unit(build, unit):
    matrix = load(build["matrix"])
    configuration = size_whatif.find_configuration(matrix, 0.03)
    models = size_whatif.check_models("lgbm_tail_rotation", matrix, (unit,))
    loaded = {
        u: load_unit_days(
            build["tables"], u, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for u in TRAIN_UNITS
    }
    model, fit_units = models[unit]
    prepared = size_whatif.prepare_unit(
        unit, model, fit_units, matrix, configuration, {unit: loaded[unit]}, loaded, 1
    )
    return matrix, loaded, prepared


@pytest.mark.parametrize("unit", ["jpl", "OSL_S"])
def test_constant_is_dial_choice_on_every_split(runs, build, unit):
    results = json.loads((runs[1][1] / size_resolve_whatif.RESULTS_NAME).read_text())
    matrix, loaded, prepared = prepared_unit(build, unit)
    losses, _ = dial_choice.measure(
        prepared, [None] * len(prepared), lambda pairs: [gate_run._evaluate_task(p) for p in pairs]
    )
    splits = dial_choice.split_indices(unit, loaded[unit], matrix, range(1, SPLITS + 1))
    for tier in size_scale.TIERS:
        offers = [
            dial_splits.score_split(losses, seed, cal, test, tier).offer_kwh
            for seed, cal, test in splits
        ]
        found = results["units"][unit]["tiers"][tier]["constant"]
        assert found["mean_offer_kwh_per_counting_test_day"] == pytest.approx(
            np.mean(offers), abs=1e-12
        )
        assert found["mean_scalar_used"] == pytest.approx(
            np.mean([dial_splits.score_split(losses, *s, tier).scale for s in splits])
        )
    assert any(o > 0 for o in offers)


# --- 4. to 6. ----------------------------------------------------------------------------------


def test_the_evaluator_keeps_the_one_way_losses_of_each_pair_once(build):
    _, _, prepared = prepared_unit(build, "jpl")
    runner = gate_run.Runner(1, {}, load(build["matrix"]))
    calls = []
    real = runner.map
    runner.map = lambda task, jobs: (calls.append(len(jobs)), real(task, jobs))[1]
    evaluator = size_resolve_whatif.Evaluator(runner, prepared)
    days = [d for d, p in enumerate(prepared) if not p.abstains][:4]
    pairs = [(d, s) for d in days for s in (0.3, 0.75)]
    evaluator.add(pairs)
    evaluator.add(pairs[:3])
    assert calls == [8]
    for d, s in pairs:
        rung = gate_day.evaluate(prepared[d], s).rungs[dial_choice.RUNG]
        assert evaluator.losses[(d, s)] == (rung.slot_loss, rung.driver_loss)
    assert evaluator.counts()["day_scale_pairs"] == 8


def test_raw_day_types_come_from_the_labelled_rows():
    data = size_whatif.UnitData("u", None, 4, "calibratable", (), [], [], {}, 0, [])
    frame = pd.DataFrame(
        {
            "unit": ["u", "u", "v"],
            "day": [0, 2, 1],
            "date": ["2020-01-06", "2020-01-01", "2020-01-07"],
            "public_holiday": [0.0, 1.0, 0.0],
        }
    )
    found = size_resolve_whatif.raw_types(data, frame)
    assert found == ["Monday", None, t7_stat.HOLIDAY, None]


def test_a_unit_that_is_not_split_is_not_evaluated():
    data = size_whatif.UnitData("u", None, 2, "too few days", (), [], [], {}, 0, [])

    class Refuses:
        losses = {}

        def add(self, pairs):
            raise AssertionError("evaluated")

    r_of, scales = size_resolve_whatif.unit_sizings(
        data, {"constant": np.array([1.0, np.nan])}, Refuses(), [None, None]
    )
    assert set(r_of) == {"constant", *(n for n in NAMES if n.startswith(("oracle-r", "day")))}
    assert scales == {}


class TableEvaluator:
    """`Evaluator`'s interface on a loss function of (day, scale)."""

    def __init__(self, loss):
        self.loss = loss
        self.losses = {}

    def add(self, pairs):
        for d, s in pairs:
            self.losses[(d, s)] = self.loss(d, s)


def test_the_sizings_of_a_unit_on_a_hand_made_table():
    raw = ["Monday"] * 12 + ["Tuesday"] * 12 + ["Wednesday"] * 6
    # days 0, 1 and 2 are the test days, one of each type
    order = [0, 12, 24, *range(1, 12), *range(13, 24), *range(25, 30)]
    raw = [raw[k] for k in order]
    thresholds = {"Monday": 0.4, "Tuesday": 2.0, "Wednesday": 0.0}

    def loss(d, s):
        return (1.0 if s > thresholds[raw[d]] else 0.0, 0.0)

    calibration, test = list(range(3, 30)), [0, 1, 2]
    data = size_whatif.UnitData(
        "u", None, 30, "calibratable", (), [], [], {}, 0, [(1, calibration, test)]
    )
    q_all = {"constant": np.ones(30), "half": np.full(30, 0.5)}
    r_of, scales = size_resolve_whatif.unit_sizings(data, q_all, TableEvaluator(loss), raw)
    assert list(r_of["half"](1, "A")) == [1.0] * 30
    expect = {"Monday": 0.4, "Tuesday": 1.0, "Wednesday": size_scale.FLOOR}
    found = r_of["oracle-resolve-scale"](1, "B")
    assert [found[d] for d in range(3)] == [pytest.approx(expect[raw[d]]) for d in range(3)]
    assert np.isnan(r_of["oracle-resolve-abstain"](1, "B")[2])
    found = r_of["day-type-scale"](1, "B")
    assert list(found[:3]) == [pytest.approx(0.4), 1.0, pytest.approx(size_scale.FLOOR)]
    found = r_of["day-type-abstain"](1, "B")
    assert list(found[:2]) == [pytest.approx(0.4), 1.0] and np.isnan(found[2])
    assert scales["B"]["other"] == [0.0] and scales["B"]["Monday"] == [0.4]


# --- 7. ----------------------------------------------------------------------------------------


def test_one_and_two_workers_write_the_same_bytes(runs):
    assert runs[1][0] == runs[2][0] == 0
    assert files(runs[1][1]) == files(runs[2][1])


# --- 8. ----------------------------------------------------------------------------------------


def labelled_cars(build) -> pd.DataFrame:
    """The cars plugged in at the decision time of the labelled units (`size_net_inputs`)."""
    matrix = load(build["matrix"])
    loaded = {
        u: load_unit_days(
            build["tables"], u, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for u in UNITS
    }
    sources = size_net_fake.context().sources
    return pd.concat(
        [size_net_inputs.car_frame(loaded[u].cars, rho=matrix.rho, sources=sources) for u in UNITS],
        ignore_index=True,
    )


def test_the_net_sizes_are_its_leave_one_unit_out_predictions(runs, build):
    _, folder = runs[1]
    results = json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text())
    days = pd.read_csv(folder / size_resolve_whatif.DAYS_NAME, dtype={"date": str})
    columns = [f"net_q{level:g}" for level in size_net.LEVELS]
    assert list(days.columns) == [*size_whatif.DAY_COLUMNS, *columns, "ct_scale", "ct_abstain"]
    cars = labelled_cars(build)
    rows = days[["unit", "site", "date", size_gbm.LABEL]]
    records = {}
    expected = size_net.leave_one_unit_out(rows, cars, records=records)
    for level, column in zip(size_net.LEVELS, columns, strict=True):
        found = days[column].to_numpy(float)
        assert not np.isnan(found).any()
        np.testing.assert_allclose(found, expected[float(level)]["net"], rtol=1e-15, atol=0)
    assert results["net_inputs"] == size_net_inputs.record(size_net_fake.context())
    assert sorted(records) == sorted(UNITS)
    assert all(sorted(r) == sorted(f"{lv:g}" for lv in size_net.LEVELS) for r in records.values())
    assert results["net_fits"] == records
    summary = (folder / size_resolve_whatif.SUMMARY_NAME).read_text()
    kept = sum(r["epoch_kept"] == 0 for by in records.values() for r in by.values())
    assert size_resolve_summary.net_fit_line(records) in summary.splitlines()
    assert size_resolve_summary.net_fit_line(records) == (
        f"day-size net: {len(UNITS) * len(size_net.LEVELS)} networks fitted, {kept} kept at "
        "epoch 0 (the starting constant)"
    )
    for unit in UNITS:
        pinball = results["units"][unit]["pinball"]
        assert set(pinball) == {f"{level:g}" for level in size_net.LEVELS}
        assert all(set(p) == {"gbm", "constant", "net"} for p in pinball.values())
        assert all(p["net"] is not None for p in pinball.values())
    for kind in ("workplace", "other"):
        pinball = results["pooled_by_site_type"][kind]["pinball"]
        assert all(set(p) == {"gbm", "constant", "net"} for p in pinball.values())


def test_the_net_predictions_become_sizes_through_the_arms():
    rows = [{"day": 0}, {"day": 2}, {"day": 3}]
    data = size_whatif.UnitData("u", None, 4, "calibratable", (), rows, [], {}, 0, [])
    predictions = {float(level): np.array([0.02, 0.5, 1.7]) for level in size_net.LEVELS}
    found = size_whatif.predicted_sizings(data, predictions, "net")
    names = {f"net-{level:g}-{arm}" for level in size_net.LEVELS for arm in size_scale.ARMS}
    assert set(found) == names
    scale, abstain = found["net-0.1-scale"], found["net-0.1-abstain"]
    assert np.isnan(scale[1]) and np.isnan(abstain[1])
    assert scale[[0, 2, 3]].tolist() == [size_scale.FLOOR, 0.5, 1.0]
    assert np.isnan(abstain[0]) and abstain[[2, 3]].tolist() == [0.5, 1.0]


def test_without_the_context_files_the_command_could_not_run(build, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)

    def missing(*arguments, **keywords):
        raise model_inputs.ContextError("the context files under data/raw cannot be read: none")

    monkeypatch.setattr(size_net_inputs, "load_context", missing)
    code = size_resolve_whatif.main(arguments(build, "--units", "jpl"))
    text = capsys.readouterr().out
    assert code == 2 and text.startswith("COULD NOT RUN") and "context files" in text
    assert list(folder.iterdir()) == []


# --- 9. ----------------------------------------------------------------------------------------


def test_the_conformal_sizes_are_its_leave_one_unit_out_outputs(runs, build):
    _, folder = runs[1]
    results = json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text())
    days = pd.read_csv(folder / size_resolve_whatif.DAYS_NAME, dtype={"date": str})
    rows = days[["unit", "site", "date", size_gbm.LABEL, size_ct.VALUE]]
    records = {}
    expected = size_ct.leave_one_unit_out(rows, labelled_cars(build), records=records)
    for arm in size_scale.ARMS:
        found = days[f"ct_{arm}"].to_numpy(float)
        assert not np.isnan(found).any()
        np.testing.assert_array_equal(found, expected[arm])
    assert sorted(records) == sorted(UNITS)
    assert all(sorted(r) == sorted(size_scale.ARMS) for r in records.values())
    assert results["ct_fits"] == records
    kept = sum(r["epoch_kept"] == 0 for by in records.values() for r in by.values())
    summary = (folder / size_resolve_whatif.SUMMARY_NAME).read_text().splitlines()
    assert size_resolve_summary.ct_fit_line(records) in summary
    assert size_resolve_summary.ct_fit_line(records) == (
        f"conformal training: {len(UNITS) * len(size_scale.ARMS)} networks fitted, {kept} kept "
        "at epoch 0 (the day-size net it starts from)"
    )


def test_the_conformal_outputs_become_sizes_each_through_its_own_arm():
    rows = [{"day": 0}, {"day": 2}, {"day": 3}]
    data = size_whatif.UnitData("u", None, 4, "calibratable", (), rows, [], {}, 0, [])
    outputs = {"scale": np.array([0.02, 0.5, 1.7]), "abstain": np.array([0.6, 0.01, 0.3])}
    found = size_whatif.conformal_sizings(data, outputs)
    assert list(found) == ["ct-scale", "ct-abstain"]
    scale, abstain = found["ct-scale"], found["ct-abstain"]
    assert np.isnan(scale[1]) and np.isnan(abstain[1])
    assert scale[[0, 2, 3]].tolist() == [size_scale.FLOOR, 0.5, 1.0]
    assert np.isnan(abstain[2]) and abstain[[0, 3]].tolist() == [0.6, 0.3]
    alone = {arm: np.full(3, np.nan) for arm in size_scale.ARMS}
    assert size_whatif.conformal_sizings(data, alone) == {}


# --- 10. ---------------------------------------------------------------------------------------


def test_shuffled_moves_the_labelled_days_sizes_by_hand():
    nan = np.nan
    days = np.array([0, 1, 3, 4])  # day 2 does not offer under constant
    found = size_resolve_whatif.shuffled(
        np.array([0.2, 0.4, nan, 1.0, 0.6]), days, np.array([3, 0, 2, 1])
    )
    np.testing.assert_array_equal(found, [0.6, 0.2, nan, 1.0, 0.4])
    found = size_resolve_whatif.shuffled(
        np.array([nan, 0.4, nan, 1.0, 0.6]), days, np.array([1, 0, 2, 3])
    )
    np.testing.assert_array_equal(found, [0.4, nan, nan, 1.0, 0.6])


def test_the_control_order_is_one_fixed_permutation_per_unit_and_draw():
    order = size_resolve_whatif.control_order("jpl", 30, 0)
    assert sorted(order.tolist()) == list(range(30))
    assert order.tolist() != list(range(30))
    np.testing.assert_array_equal(order, size_resolve_whatif.control_order("jpl", 30, 0))
    assert order.tolist() != size_resolve_whatif.control_order("caltech", 30, 0).tolist()
    assert order.tolist() != size_resolve_whatif.control_order("jpl", 30, 1).tolist()


def test_each_learned_sizing_gets_its_control_on_a_hand_made_table():
    raw = (["Monday"] * 12 + ["Tuesday"] * 12 + ["Wednesday"] * 6)[:29] + [None]
    thresholds = {"Monday": 0.4, "Tuesday": 2.0, "Wednesday": 0.0}

    def loss(d, s):
        return (1.0 if s > thresholds[raw[d]] else 0.0, 0.0)

    splits = [(1, list(range(3, 29)), [0, 1, 2]), (2, list(range(0, 26)), [26, 27, 28])]
    data = size_whatif.UnitData("u", None, 30, "calibratable", (), [], [], {}, 0, splits)
    constant = np.append(np.ones(29), np.nan)  # day 29 does not offer
    net = np.append(np.linspace(0.1, 0.9, 29), np.nan)
    ct = net.copy()
    ct[[4, 9]] = np.nan  # abstains on two days
    q_all = {
        "constant": constant,
        "gbm-0.1-scale": net,
        "net-0.1-scale": net,
        "ct-abstain": ct,
        "oracle-scale": net,
    }
    evaluator = TableEvaluator(loss)
    r_of, _ = size_resolve_whatif.unit_sizings(data, q_all, evaluator, raw)
    controls = size_resolve_whatif.control_sizings(data, q_all, r_of, evaluator, 3)
    assert list(controls) == [
        "shuffled-net-0.1-scale",
        "shuffled-ct-abstain",
        "shuffled-day-type-scale",
        "shuffled-day-type-abstain",
    ]
    days = np.arange(29)
    for name, draws in controls.items():
        original = name.removeprefix("shuffled-")
        assert len(draws) == 3
        for draw, control in enumerate(draws):
            order = size_resolve_whatif.control_order("u", 29, draw)
            for seed in (1, 2):
                for tier in size_scale.TIERS:
                    r = r_of[original](seed, tier)
                    found = control(seed, tier)
                    expect = r.copy()
                    expect[days] = r[days][order]
                    np.testing.assert_array_equal(found, expect)
                    assert np.isnan(found[29])
                    needed = size_resolve.wanted(found)
                    assert needed <= set(evaluator.losses)
    first, second = (f(1, "A") for f in controls["shuffled-net-0.1-scale"][:2])
    assert not np.array_equal(first, second, equal_nan=True)  # day 29 is NaN in both
    assert np.isnan(controls["shuffled-ct-abstain"][0](1, "A")).sum() == 3


def test_every_learned_sizing_is_read_against_its_shuffled_control(runs):
    _, folder = runs[1]
    results = json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text())
    summary = (folder / size_resolve_whatif.SUMMARY_NAME).read_text().splitlines()
    groups = [*results["units"].values(), *results["pooled_by_site_type"].values()]
    for entry in groups:
        for tier in size_scale.TIERS:
            found = entry["tiers"][tier]
            for name in found:
                assert ("difference_from_shuffled" in found[name]) == (name in LEARNED), name
            for name in LEARNED:
                diff = found[name]["difference_from_shuffled"]
                assert diff["draws"] == size_resolve_whatif.CONTROL_DRAWS
                offer = found[name]["mean_offer_kwh_per_counting_test_day"]
                control = diff["control_mean_offer_kwh_per_counting_test_day"]
                assert diff["mean_kwh"] == pytest.approx(offer - control, abs=1e-12)
                assert diff["over_draws"]["p5_kwh"] <= diff["over_draws"]["p95_kwh"]
    against = [line for line in summary if "against its shuffled control" in line]
    assert len(against) == len(LEARNED) * len(size_scale.TIERS) * len(groups)
    for i, line in enumerate(summary):
        if "against its shuffled control" in line:
            sizing = summary[i - 1].strip().split(":")[0]
            assert sizing in LEARNED and f"shuffled-{sizing}, mean of " in line
    assert sorted(results["control_evaluations"]) == sorted(UNITS)
    definition = results["definitions"]["shuffled_control"]
    assert f"shuffled controls: {definition}" in summary
    for unit, counts in results["control_evaluations"].items():
        assert counts["day_scale_pairs"] > 0
        assert f"{unit}: the shuffled controls added {counts['day_scale_pairs']} (day, scale) " in (
            "\n".join(summary)
        )


def test_a_draw_count_below_one_could_not_run(build, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    code = size_resolve_whatif.main(arguments(build, "--units", "jpl", "--control-draws", "0"))
    text = capsys.readouterr().out
    assert code == 2 and text == "COULD NOT RUN: --control-draws must be at least 1\n"
    assert list(folder.iterdir()) == []


# --- 11. ---------------------------------------------------------------------------------------

NOISE = 20  # noise sizings
INSIDE = 10  # of them at least this many sit inside their spread over draws


def noise_figures(draws: int) -> dict:
    """Per sizing of the made-up table of item 11, its `difference_from_shuffled` in tier B."""
    n, splits = 60, 20
    rng = np.random.default_rng(11)
    threshold = rng.uniform(0.2, 1.0, n)
    value = rng.uniform(1.0, 3.0, n)

    def loss(d, s):
        return (1.0 if s > threshold[d] else 0.0, 0.0)

    split_list = []
    for seed in range(1, splits + 1):
        order = rng.permutation(n)
        split_list.append((seed, sorted(order[:30].tolist()), sorted(order[30:].tolist())))
    data = size_whatif.UnitData("u", None, n, "calibratable", (), [], [], {}, 0, split_list)
    q_all = {"constant": np.ones(n), "net-same": np.full(n, 0.5), "net-knows": threshold.copy()}
    for j in range(NOISE):
        q_all[f"net-noise-{j}"] = np.random.default_rng(100 + j).uniform(0.05, 1.0, n)
    evaluator = TableEvaluator(loss)
    r_of, _ = size_resolve_whatif.unit_sizings(data, q_all, evaluator, ["Monday"] * n)
    controls = size_resolve_whatif.control_sizings(data, q_all, r_of, evaluator, draws)
    measured = {
        name: size_resolve.measure(evaluator.losses, value, f, split_list)
        for name, f in r_of.items()
    }
    measured_controls = {
        name: [size_resolve.measure(evaluator.losses, value, f, split_list) for f in fs]
        for name, fs in controls.items()
    }
    tiers = size_resolve_summary._tiers({"u": measured}, ["u"], {"u": measured_controls})
    return {name: f["difference_from_shuffled"] for name, f in tiers["B"].items() if "net-" in name}


def inside(diff) -> bool:
    return diff["over_draws"]["p5_kwh"] <= 0.0 <= diff["over_draws"]["p95_kwh"]


def test_a_noise_sizing_sits_inside_its_spread_over_draws():
    found = noise_figures(size_resolve_whatif.CONTROL_DRAWS)
    noise = [found[f"net-noise-{j}"] for j in range(NOISE)]
    assert sum(inside(diff) for diff in noise) >= INSIDE
    assert found["net-knows"]["over_draws"]["p5_kwh"] > 0.0
    same = found["net-same"]
    assert [same[k] for k in ("mean_kwh", "p5_kwh", "p95_kwh", "mean_share")] == [0.0] * 4
    assert list(same["over_draws"].values()) == [0.0] * 4


# --- 12. ---------------------------------------------------------------------------------------


def without_controls(entry: dict) -> dict:
    """`entry`'s tiers with every `difference_from_shuffled` removed."""
    return {
        tier: {
            name: {k: v for k, v in figures.items() if k != "difference_from_shuffled"}
            for name, figures in found.items()
        }
        for tier, found in entry["tiers"].items()
    }


def test_a_failing_control_keeps_the_units_sizings(runs):
    code, folder = runs["controls fail"]
    results = json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text())
    whole = json.loads((runs[1][1] / size_resolve_whatif.RESULTS_NAME).read_text())
    assert code == 1 and results["failed_units"] == []
    assert results["failed_controls"] == [f"{FAILING}: RuntimeError: made-up control failure"]
    nothing = {"day_scale_pairs": 0, "re_solves_of_the_one_way_rung": {}}
    assert results["control_evaluations"][FAILING] == nothing  # it failed before any evaluation
    summary = (folder / size_resolve_whatif.SUMMARY_NAME).read_text().splitlines()
    assert summary[-1] == (
        "FAILED CONTROLS (the unit's other figures kept): "
        f"{FAILING}: RuntimeError: made-up control failure"
    )
    for unit, entry in results["units"].items():
        if unit == FAILING:
            assert without_controls(entry) == without_controls(whole["units"][unit])
            rest = {k: v for k, v in entry.items() if k != "tiers"}
            assert rest == {k: v for k, v in whole["units"][unit].items() if k != "tiers"}
            found = entry["tiers"]["A"]
            assert not any("difference_from_shuffled" in f for f in found.values())
        else:
            assert entry == whole["units"][unit]
    for kind, entry in results["pooled_by_site_type"].items():
        if FAILING in entry["units_scored"]:
            assert without_controls(entry) == without_controls(whole["pooled_by_site_type"][kind])
            assert not any("difference_from_shuffled" in f for f in entry["tiers"]["B"].values())
        else:
            assert entry == whole["pooled_by_site_type"][kind]
