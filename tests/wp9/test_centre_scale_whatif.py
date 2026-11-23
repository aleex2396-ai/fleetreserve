"""The centre-scale what-if command (`centre_scale_whatif`) and its summary
(`centre_scale_summary`), on the made-up gate units of `learn_fake.small_gate_units` (5, 24 and 24
site-days of three cars, one service hour) with a made-up verified truth build, two dial
configurations (0.03 and 0.1, kappa 0.3) and 20 calibration days.

1. A run with `--model empirical` and two splits ends with exit 0 and writes its four files; the
   summary starts with the what-if label, names the unit that cannot be split, counts the days
   outside the set, gives the identity counts, a floor line per unit, dial and rung, and a figure
   line per tier, unit, dial and rung with mean [p5, p95] over the two splits.
2. The unmixed figures of split 1 (the gate run's split) equal, field by field, those `gate_run`
   writes for the same build and matrix, for both rungs and both tiers; at least one compared
   scale lies strictly between 0 and the grid's top. The mixed floor is not the unmixed one.
3. A broken unmixed route (a field of its test figures changed after scoring) makes the command
   refuse: exit 2, "identity failed", no file written.
4. The calibration of the mixed kind sees the mixed policy's losses at every scale it evaluates:
   each evaluation's mean losses are the running maximum of `centre_scale_day.evaluate`'s losses
   at the scales evaluated, and at some evaluated scale they differ from the gate's.
5. A day whose front-loaded plan is outside the set is named in the results and the summary,
   counted, and replayed with the certified policy.
6. The identity at scale 1 (step 5): equal evaluations pass; a planted difference is found, and
   the command fails on it (exit 1) with no day evaluated at 1 by the calibration or the floor,
   since every offering day is replayed at 1 under every room factor, unless one of the two
   replays has a re-solve that timed out, which is counted.
7. Refusals before any day: an unknown or repeated configuration, `--splits 0`, an unknown model,
   a rotation what-if matrix (exit 2, nothing written).
8. The arithmetic of the summary by hand: the mean and the 5th and 95th percentile skip missing
   values; the paired difference is mixed minus unmixed on the same split; one split prints its
   value, two print mean [p5, p95].

Room factors (`--room-factors`, default 1 2 4 8):

9. The calibration of a factor above 1 sees that factor's losses, with the running maximum and the
   bisection, on a planted loss that is not monotone in the scale (20 offering calibration days;
   the bound on the driver loss holds while its sum is at most 0.05). Day 0 loses 0.04 while the
   weight w = min(1, c beta) lies in [0.3, 0.5], day 1 loses 0.02 from w = 0.87 on, every other
   loss is 0. Factor 4 (w = 4 beta): the grid holds at 0.05 (sum 0), 0.1 (w 0.4: 0.04) and 0.2
   (raw 0, 0.04 after the running maximum) and fails at 0.3 (w 1: 0.02 + 0.04); the bisection
   evaluates 0.25 F, 0.225 F, 0.2125 H (w 0.85), 0.21875 F (w 0.875), 0.215625 H, 0.2171875 H
   (w 0.86875), 0.21796875 F (w 0.871875), and stops with the scale 0.2171875. Without the
   running maximum every grid scale would hold (0.02 at w = 1) and the scale would be 1. Factor 1
   (w = beta): the grid holds up to 0.8 and fails at 1; the bisection ends at 0.86953125.
10. On made-up days, the calibration of factor 4 sees `centre_scale_day.evaluate(day, beta, 4)` at
   every evaluated scale (after the running maximum), which differs from factor 1's at some.
11. A run with the default factors: every split, the floor and the evaluations carry one entry per
   factor; every offering day is compared at scale 1 under each of the 4 factors; the summary's
   lines for factor 1 are as before and it ends with one table of 80 rows (2 units, 2 dials, 2
   rungs, 2 tiers, 4 factors and unmixed).
12. A factor of 20 (weight 1 from scale 0.05 on) is the unmixed policy: its floor equals the
   unmixed floor, and on every split, rung and tier where the unmixed scale is at least 0.05 (so
   every scale either search evaluates has weight 1) its scale is the unmixed one capped at 1. A
   scale below 0.05 comes from a bisection between 0 and 0.05, where 20 beta < 1 and the mix
   differs, so no identity is asked there.
13. Refusals of `--room-factors`: 1 missing, a factor below 1, one named twice, infinite or not a
   number (exit 2, nothing written); and the room table's text by hand.
"""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gate_fake import COMMIT, GATE_UNITS, matrix_document, write_build
from learn_fake import small_gate_units

from fleetreserve import (
    centre_scale_day,
    centre_scale_summary,
    centre_scale_whatif,
    gate_calibrate,
    gate_day,
    gate_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load
from fleetreserve.gate_day import DayEvaluation, RungEvaluation

LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
CONFIGURATIONS = (LOW[0], HIGH[0])
FILES = (
    centre_scale_whatif.RESULTS_NAME,
    centre_scale_whatif.SUMMARY_NAME,
    centre_scale_whatif.TABLE_NAME,
    centre_scale_whatif.TIMING_NAME,
)


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def _matrix_document() -> dict:
    document = matrix_document(LOW + HIGH)
    document["n_calibration"] = 20
    return document


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, small_gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(_matrix_document()))
    return run, tables, matrix


def _arguments(build, matrix=None) -> list[str]:
    run, tables, committed = build
    matrix = committed if matrix is None else matrix
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def _what_if(build, out: Path, *, splits="2", patch=None) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--model", "empirical", "--configurations", *CONFIGURATIONS]
        return centre_scale_whatif.main([*_arguments(build), *arguments, "--splits", splits])


def _gate(build, out: Path) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return gate_run.main(_arguments(build))


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("runs")
    what = _what_if(build, folder / "what-if")
    gate = _gate(build, folder / "gate")
    return (what, folder / "what-if"), (gate, folder / "gate")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


# --- 1. a full run -------------------------------------------------------------------------------


def test_a_run_writes_every_file_and_its_summary(runs):
    (code, out), _ = runs
    assert code == 0
    for name in FILES:
        assert (out / name).is_file()
    document = _read(out / centre_scale_whatif.RESULTS_NAME)
    assert document["failed_configurations"] == []
    assert document["label"] == centre_scale_summary.LABEL
    assert document["run"]["grid_mixed"][-1] == 1.0 and document["run"]["floor_scale"] == 0.05
    small, *usable = (document["units"][u] for u in GATE_UNITS)
    assert small["status"].startswith("not calibratable") and small["configurations"] == {}
    lines = (out / centre_scale_whatif.SUMMARY_NAME).read_text().splitlines()
    assert lines[0] == centre_scale_summary.LABEL
    assert (
        "Front-loaded plan outside the set: 0 days (each replayed unmixed at every scale)" in lines
    )
    assert f"{GATE_UNITS[0]}: not calibratable: 5 days" in lines
    assert any(line.startswith("Identity: the unmixed figures of the gate run's") for line in lines)
    floor = [line for line in lines if line.startswith("  ") and "days; slot" in line]
    assert len(floor) == 2 * 2 * 2  # units, dials, rungs
    tiers = [k for k, line in enumerate(lines) if line.startswith("tier ")]
    assert len(tiers) == 2 and tiers[0] > lines.index(floor[-1])
    figures = [line for line in lines[tiers[0] :] if line.startswith("  ")]
    assert len(figures) == 2 * 2 * 2 * 2  # tiers, units, dials, rungs
    assert all(" [" in line.split("scale ")[1].split(";")[0] for line in figures)
    for entry in usable:
        for found in entry["configurations"].values():
            assert [r["seed"] for r in found["splits"]] == [1, 2]
            assert found["over_splits"]["one-way"]["A"]["mixed"]["scale"]["splits"] == 2


# --- 2. the identity with the gate ---------------------------------------------------------------


def test_the_unmixed_figures_of_the_gates_split_are_the_gates(runs):
    (_, out), (gate_code, gate_out) = runs
    assert gate_code == 0
    document = _read(out / centre_scale_whatif.RESULTS_NAME)
    gate = _read(gate_out / gate_run.RESULTS_NAME)
    compared, inside = 0, 0
    for unit in GATE_UNITS[1:]:
        for name in CONFIGURATIONS:
            source = gate["configurations"][name]["units"][unit]["rungs"]
            splits = document["units"][unit]["configurations"][name]["splits"]
            (record,) = [r for r in splits if r["seed"] == 1]
            for rung in centre_scale_day.MIXED_RUNGS:
                for tier in ("A", "B"):
                    ours = record["rungs"][rung]["unmixed"][tier]
                    assert ours == centre_scale_summary.tier_figures(source[rung], tier)
                    compared += 1
                    inside += 0 < ours["scale"] < 2
    assert compared == 16 and inside >= 1


def test_the_mixed_floor_is_not_the_unmixed(runs):
    (_, out), _ = runs
    document = _read(out / centre_scale_whatif.RESULTS_NAME)
    floors = [
        f
        for u in GATE_UNITS[1:]
        for c in document["units"][u]["configurations"].values()
        for f in c["floor"].values()
    ]
    assert len(floors) == 8
    assert any(f["mixed"]["mean_driver_loss"] != f["unmixed"]["mean_driver_loss"] for f in floors)


# --- 3. a broken unmixed route -------------------------------------------------------------------


def test_a_broken_unmixed_route_makes_the_command_refuse(build, tmp_path, capsys):
    real = centre_scale_whatif.score

    def broken(cache, ceilings, calibration, test, calibrate):
        found = real(cache, ceilings, calibration, test, calibrate)
        if calibrate is gate_calibrate.calibrate_rung:
            found["one-way"]["test"] = {**found["one-way"]["test"], "offering_days": -1}
        return found

    def patch(monkeypatch):
        monkeypatch.setattr(centre_scale_whatif, "score", broken)

    code = _what_if(build, tmp_path / "out", splits="1", patch=patch)
    assert code == 2
    assert "COULD NOT RUN: identity failed" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


# --- 4. the calibration sees the mixed losses ----------------------------------------------------


@pytest.fixture(scope="module")
def prepared_unit(build):
    _, tables, path = build
    matrix = load(path)
    unit = GATE_UNITS[1]
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    runner = gate_run.Runner(1, {unit: days}, matrix)
    configuration = matrix.configuration(LOW[0])
    prepared = runner.map(
        gate_run._prepare_task, [(unit, i, configuration) for i in range(len(days.days))]
    )
    return prepared, gate_run.plan_unit(unit, days, matrix)


def test_the_mixed_calibration_sees_the_mixed_losses_at_every_scale(prepared_unit):
    prepared, plan = prepared_unit
    seen = []

    def batch(pairs):
        seen.extend(b for _, b in pairs)
        return [centre_scale_whatif._mixed_task(job) for job in pairs]

    cache = gate_calibrate.DayCache(prepared, batch)
    calibration = [d for d, r in enumerate(plan.roles) if r == "calibration"]
    offering = gate_calibrate.offering_days(prepared, calibration)
    result = centre_scale_whatif.calibrate_mixed(cache, calibration, "one-way")
    evaluated = [e[0] for e in result["evaluations"]]
    assert len(evaluated) >= 2 and set(evaluated) <= set(seen) and max(seen) <= 1.0
    running = np.zeros((2, len(offering)))
    differs = 0
    for beta in sorted(evaluated):
        mixed = [centre_scale_day.evaluate(prepared[d], beta).rungs["one-way"] for d in offering]
        gate = [gate_day.evaluate(prepared[d], beta).rungs["one-way"] for d in offering]
        raw = np.array([[r.slot_loss for r in mixed], [r.driver_loss for r in mixed]])
        running = np.maximum(running, raw)
        (entry,) = [e for e in result["evaluations"] if e[0] == beta]
        assert entry[1] == pytest.approx(running[0].mean(), abs=1e-12)
        assert entry[2] == pytest.approx(running[1].mean(), abs=1e-12)
        differs += any(
            (m.slot_loss, m.driver_loss) != (g.slot_loss, g.driver_loss)
            for m, g in zip(mixed, gate, strict=True)
        )
    assert differs >= 1


# --- 5. a day outside the set --------------------------------------------------------------------


def test_a_day_outside_the_set_is_named_and_counted(build, tmp_path):
    real = centre_scale_day.front_loaded_slack
    chosen = {}

    def slack(prepared):
        if not chosen and prepared.unit == GATE_UNITS[1]:
            chosen["date"] = prepared.date
        if prepared.unit == GATE_UNITS[1] and prepared.date == chosen.get("date"):
            return -1.0
        return real(prepared)

    def patch(monkeypatch):
        monkeypatch.setattr(centre_scale_day, "front_loaded_slack", slack)

    code = _what_if(build, tmp_path / "out", splits="1", patch=patch)
    assert code == 0
    document = _read(tmp_path / "out" / centre_scale_whatif.RESULTS_NAME)
    assert document["front_loaded_outside_days"] == 2  # the day, under each of two dials
    named = f"{chosen['date']} A"
    for name in CONFIGURATIONS:
        found = document["units"][GATE_UNITS[1]]["configurations"][name]["front_loaded_outside"]
        assert found == [{"day": named, "smallest_slack_kwh": -1.0}]
    lines = (tmp_path / "out" / centre_scale_whatif.SUMMARY_NAME).read_text().splitlines()
    head = "Front-loaded plan outside the set: 2 days (each replayed unmixed at every scale):"
    assert head in lines
    assert f"  {GATE_UNITS[1]} | {LOW[0]} | {named} (slack -1)" in lines


def test_a_day_outside_the_set_replays_the_certified_policy(prepared_unit, monkeypatch):
    prepared, _ = prepared_unit
    days = [p for p in prepared if not p.abstains][:3]
    monkeypatch.setattr(centre_scale_day, "front_loaded_slack", lambda p: -1.0)
    for day in days:
        for beta in (0.05, 0.5):
            found = centre_scale_day.evaluate(day, beta)
            gate = gate_day.evaluate(day, beta)
            for rung in centre_scale_day.MIXED_RUNGS:
                np.testing.assert_array_equal(found.rungs[rung].finals, gate.rungs[rung].finals)
                assert found.rungs[rung].driver_loss == gate.rungs[rung].driver_loss


# --- 6. the identity at scale 1 ------------------------------------------------------------------


def _evaluation(slot, driver, resolves=None):
    rung = RungEvaluation(slot, driver, (None,), None, resolves or {}, 0.0)
    return DayEvaluation(1.0, False, {"fixed": rung, "one-way": rung}, 1, (), 0.0)


class _Cache:
    def __init__(self, store):
        self.store = store


def test_the_identity_at_scale_1():
    gate = {(0, 1.0): _evaluation(0.0, 0.1), (1, 1.0): _evaluation(0.0, 0.2)}
    same = {(0, 1.0): _evaluation(0.0, 0.1), (0, 0.5): _evaluation(0.0, 0.0)}
    assert centre_scale_whatif.scale_one_identity(_Cache(same), _Cache(gate)) == (1, [], [])
    planted = {(0, 1.0): _evaluation(0.0, 0.3)}
    assert centre_scale_whatif.scale_one_identity(_Cache(planted), _Cache(gate)) == (1, [], [0])
    timed = {(0, 1.0): _evaluation(0.0, 0.3, {"timeout": 1})}
    assert centre_scale_whatif.scale_one_identity(_Cache(timed), _Cache(gate)) == (1, [0], [])


def test_a_difference_at_scale_1_makes_the_command_refuse(build, tmp_path, capsys):
    real = centre_scale_day.evaluate

    def shifted(prepared, beta, *factor):
        found = real(prepared, beta, *factor)
        if beta != 1.0 or found.abstained:
            return found
        fixed = found.rungs["fixed"]
        moved = dataclasses.replace(fixed, driver_loss=0.25 if fixed.driver_loss == 0.5 else 0.5)
        return dataclasses.replace(found, rungs={**found.rungs, "fixed": moved})

    def patch(monkeypatch):
        monkeypatch.setattr(centre_scale_day, "evaluate", shifted)

    code = _what_if(build, tmp_path / "out", splits="1", patch=patch)
    out = capsys.readouterr().out
    assert code == 1
    assert "FAILED: identity failed" in out and "at scale 1 the mixed replay differs" in out
    assert list((tmp_path / "out").iterdir()) == []


# --- 7. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configurations, splits, model, why",
    [
        (("no-such-configuration",), "1", "empirical", "no configuration named"),
        ((LOW[0], LOW[0]), "1", "empirical", "named twice"),
        ((LOW[0],), "0", "empirical", "--splits must be at least 1"),
        ((LOW[0],), "1", "no_such_model", "no model module"),
    ],
)
def test_refusals_before_any_day(build, tmp_path, configurations, splits, model, why, capsys):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--model", model, "--configurations", *configurations, "--splits", splits]
        code = centre_scale_whatif.main([*_arguments(build), *arguments])
    printed = capsys.readouterr().out
    assert code == 2 and printed.startswith("COULD NOT RUN") and why in printed
    assert list(out.iterdir()) == []


def test_a_rotation_matrix_is_refused(build, tmp_path, capsys):
    document = _matrix_document()
    document["gate_units"] = ["jpl"]
    document["required_units"] = 1
    document["rotation_whatif"] = True
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    arguments = ["--model", "empirical", "--configurations", LOW[0], "--splits", "1"]
    code = centre_scale_whatif.main([*_arguments(build, matrix), *arguments])
    printed = capsys.readouterr().out
    assert code == 2 and "COULD NOT RUN: a rotation what-if matrix" in printed


# --- 8. the arithmetic of the summary ------------------------------------------------------------


def test_the_figures_over_splits_by_hand():
    found = centre_scale_summary.stats([1.0, None, 2.0, 3.0])
    assert found == {"splits": 3, "mean": 2.0, "p5": pytest.approx(1.1), "p95": pytest.approx(2.9)}
    assert centre_scale_summary.stats([None]) == {
        "splits": 0,
        "mean": None,
        "p5": None,
        "p95": None,
    }


def _tier(scale, offer, slot, driver):
    return {
        "beta_hat": scale,
        "reason": "",
        "evaluations": [],
        "slot_only": {"beta_hat": scale},
        "driver_only": {"beta_hat": scale},
        "test": {
            "offering_days": 4,
            "failing_days": None if scale == 0 else 1,
            "mean_offer_kwh_per_day": offer,
            "share_of_ceiling": {"c": {"ratio_of_sums": offer / 10, "mean_of_daily_ratios": 0}},
            "mean_slot_loss_offering_days": slot,
            "mean_driver_loss_offering_days": driver,
        },
    }


def _rung(scale, offer, slot, driver):
    out = _tier(scale, offer, slot, driver)
    out["tier_b"] = {**_tier(scale, offer, slot, driver), "stay_to_bound": {}}
    return out


def _records():
    first = {"mixed": _rung(0.5, 5.0, 0.0, 0.01), "unmixed": _rung(0.25, 2.5, 0.0, 0.04)}
    second = {"mixed": _rung(0.0, 0.0, None, None), "unmixed": _rung(0.45, 4.5, 0.1, 0.02)}
    return [
        centre_scale_summary.split_record(seed, {k: {"one-way": v} for k, v in found.items()})
        for seed, found in ((1, first), (2, second))
    ]


def test_the_paired_difference_by_hand():
    records = _records()
    assert records[0]["rungs"]["one-way"]["mixed_minus_unmixed"]["A"] == {
        "offer_kwh_per_day": 2.5,
        "scale": 0.25,
    }
    over = centre_scale_summary.over_splits(records, ["c"])["one-way"]["A"]
    assert over["mixed_minus_unmixed"]["offer_kwh_per_day"]["mean"] == pytest.approx(-1.0)
    assert over["mixed"]["mean_driver_loss_offering_days"]["splits"] == 1  # split 2 offers nothing
    assert over["unmixed"]["share_of_ceiling"]["c"]["mean"] == pytest.approx(0.35)


def test_the_summary_text_by_hand():
    document = {
        "label": centre_scale_summary.LABEL,
        "model": "m",
        "matrix": {"file": "matrix.json"},
        "run": {
            "splits": 2,
            "seeds": [1, 2],
            "split_seed": 1,
            "configurations": ["x"],
            "floor_scale": 0.05,
            "gate_ceiling": "c",
        },
        "identity": {
            "seed": 1,
            "gate_checked": 1,
            "policies_checked": 9,
            "scale_one_compared": 3,
            "scale_one_differ_with_timeout": 1,
        },
        "timeouts": {"re_solves_mixed": 1},
        "failed_configurations": [],
        "units": {
            "u": {
                "status": "calibratable",
                "configurations": {
                    "x": {
                        "driver_risk": 0.02,
                        "front_loaded_outside": [],
                        "floor": {
                            "one-way": centre_scale_summary.floor_figures(
                                ([0.0, 0.5], [0.0, 0.1]), ([0.0, 0.0], [0.2, 0.2]), 2, 0.05
                            )
                        },
                        "over_splits": centre_scale_summary.over_splits(_records(), ["c"]),
                    }
                },
            }
        },
    }
    lines = centre_scale_summary.summary_lines(document)
    assert lines[0] == centre_scale_summary.LABEL
    assert "  u | dial 0.02 | one-way: 2 days; slot 0.250 / 0.000; driver 0.050 / 0.200" in lines
    assert (
        "At scale 1 (every offering day replayed there under every room factor and unmixed) the "
        "mixed replay equals the unmixed one on 2 of 3 day and room factor pairs; 1 differ, each "
        "with a re-solve that timed out."
    ) in lines
    assert (
        "Timeouts: certificates 0; re-solves mixed 1, unmixed 0; best-causal ceilings 0." in lines
    )
    line = next(x for x in lines if x.startswith("  u | dial 0.02 | one-way: scale"))
    assert line.startswith(
        "  u | dial 0.02 | one-way: scale 0.250 [0.025, 0.475] / 0.350 [0.260, 0.440]; "
        "offer 2.50 [0.25, 4.75] / 3.50 [2.60, 4.40]; mixed minus unmixed -1.00 [-4.15, 2.15]; "
        "share 0.250 [0.025, 0.475] / 0.350 [0.260, 0.440]; slot 0.000 / 0.050 [0.005, 0.095]; "
        "driver 0.010 / 0.030 [0.021, 0.039]"
    )


# --- 9. a room factor's calibration on a planted loss that is not monotone -----------------------


class _Inline:
    """A runner that maps in this process (as `gate_run.Runner` with one worker)."""

    def map(self, task, jobs):
        return [task(job) for job in jobs]


def _planted(seen):
    def evaluate(prepared, beta, factor=1.0):
        seen.append((beta, factor))
        weight = min(1.0, factor * beta)
        driver = 0.0
        if prepared.index == 0 and 0.3 <= weight <= 0.5:
            driver = 0.04
        if prepared.index == 1 and weight >= 0.87:
            driver = 0.02
        rung = RungEvaluation(0.0, driver, (None,), None, {}, 0.0)
        return DayEvaluation(beta, False, {"fixed": rung, "one-way": rung}, 0, (), 0.0)

    return evaluate


@pytest.mark.parametrize(
    "factor, scale, evaluated",
    [
        (4.0, 0.2171875, [0.05, 0.1, 0.2, 0.3, 0.25, 0.225, 0.2125, 0.21875, 0.215625, 0.2171875]),
        (1.0, 0.86953125, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 0.9, 0.85]),
    ],
)
def test_a_room_factors_calibration_on_a_loss_that_is_not_monotone(
    monkeypatch, factor, scale, evaluated
):
    seen = []
    monkeypatch.setattr(centre_scale_day, "evaluate", _planted(seen))
    days = [SimpleNamespace(abstains=False, index=k) for k in range(20)]
    cache = centre_scale_whatif.mixed_cache(_Inline(), days, factor)
    result = centre_scale_whatif.calibrate_mixed(cache, list(range(20)), "one-way")
    assert result["beta_hat"] == pytest.approx(scale, abs=1e-12)
    found = [e[0] for e in result["evaluations"]][: len(evaluated)]
    assert found == pytest.approx(evaluated, abs=1e-12)  # midpoints are rounded floats
    assert {f for _, f in seen} == {factor} and max(b for b, _ in seen) <= 1.0
    if factor == 4.0:  # the running maximum keeps day 0's 0.04 above w = 0.5
        means = {e[0]: e[2] for e in result["evaluations"]}
        assert means[0.2] == pytest.approx(0.002) and means[0.3] == pytest.approx(0.003)
        assert result["driver_only"]["beta_hat"] == result["beta_hat"]


# --- 10. on made-up days, a factor's calibration reads that factor's replay ----------------------


def test_the_calibration_of_factor_4_sees_its_own_replay(prepared_unit):
    prepared, plan = prepared_unit
    calibration = [d for d, r in enumerate(plan.roles) if r == "calibration"]
    offering = gate_calibrate.offering_days(prepared, calibration)
    cache = centre_scale_whatif.mixed_cache(_Inline(), prepared, 4.0)
    result = centre_scale_whatif.calibrate_mixed(cache, calibration, "one-way")
    evaluated = [e[0] for e in result["evaluations"]]
    assert len(evaluated) >= 2 and max(evaluated) <= 1.0
    running = np.zeros((2, len(offering)))
    differs = 0
    for beta in sorted(evaluated):
        room = [
            centre_scale_day.evaluate(prepared[d], beta, 4.0).rungs["one-way"] for d in offering
        ]
        one = [centre_scale_day.evaluate(prepared[d], beta).rungs["one-way"] for d in offering]
        raw = np.array([[r.slot_loss for r in room], [r.driver_loss for r in room]])
        running = np.maximum(running, raw)
        (entry,) = [e for e in result["evaluations"] if e[0] == beta]
        assert entry[1] == pytest.approx(running[0].mean(), abs=1e-12)
        assert entry[2] == pytest.approx(running[1].mean(), abs=1e-12)
        differs += any(
            (a.slot_loss, a.driver_loss) != (b.slot_loss, b.driver_loss)
            for a, b in zip(room, one, strict=True)
        )
    assert differs >= 1


# --- 11. a run with the default factors ----------------------------------------------------------

ROOM_KINDS = ("mixed", "mixed_room_2", "mixed_room_4", "mixed_room_8")


def test_a_run_with_the_default_factors_carries_each_factor(runs):
    (code, out), _ = runs
    assert code == 0
    document = _read(out / centre_scale_whatif.RESULTS_NAME)
    assert document["run"]["room_factors"] == [1.0, 2.0, 4.0, 8.0]
    assert "room_factor" in document["definitions"]
    compared = 0
    for unit in GATE_UNITS[1:]:
        for found in document["units"][unit]["configurations"].values():
            offering = found["offering_days"]
            assert found["identity"]["scale_one_compared"] == 4 * offering  # step 5
            compared += found["identity"]["scale_one_compared"]
            for record in found["splits"]:
                for rung in centre_scale_day.MIXED_RUNGS:
                    keys = set(record["rungs"][rung])
                    assert keys == {
                        "unmixed",
                        *ROOM_KINDS,
                        *(k + "_minus_unmixed" for k in ROOM_KINDS),
                    }
            for rung in centre_scale_day.MIXED_RUNGS:
                assert set(found["floor"][rung]) == {
                    "scale",
                    "offering_days",
                    "unmixed",
                    *ROOM_KINDS,
                }
            for kind in (*ROOM_KINDS, "unmixed"):
                assert found["evaluations"][f"{kind}_day_scale_pairs"] >= offering
    assert document["identity"]["scale_one_compared"] == compared > 0
    lines = (out / centre_scale_whatif.SUMMARY_NAME).read_text().splitlines()
    assert "Timeouts of the other room factors: re-solves room 2 0, room 4 0, room 8 0." in lines
    (head,) = [k for k, line in enumerate(lines) if line.startswith("Room factors 1, 2, 4, 8:")]
    table = lines[head + 1 :]
    assert [c.strip() for c in table[0].split("|")][:5] == ["unit", "dial", "rung", "tier", "room"]
    rows = [[c.strip() for c in line.split("|")] for line in table[1:]]
    assert len(rows) == 2 * 2 * 2 * 2 * 5 and all(len(r) == 12 for r in rows)
    assert [r[4] for r in rows[:5]] == ["1", "2", "4", "8", "unmixed"]


# --- 12. a factor of 20 is the unmixed policy from scale 0.05 on ---------------------------------


def test_a_factor_of_20_is_the_unmixed_policy(build, tmp_path):
    code = _room_what_if(build, tmp_path / "out", ["1", "20"])
    assert code == 0
    document = _read(tmp_path / "out" / centre_scale_whatif.RESULTS_NAME)
    inside, below = 0, 0
    for unit in GATE_UNITS[1:]:
        for found in document["units"][unit]["configurations"].values():
            for rung, floor in found["floor"].items():
                assert floor["mixed_room_20"] == floor["unmixed"], rung
            for record in found["splits"]:
                for rung, figures in record["rungs"].items():
                    for tier in ("A", "B"):
                        unmixed = figures["unmixed"][tier]
                        room = figures["mixed_room_20"][tier]
                        if unmixed["scale"] < 0.05:  # searched below 0.05, where 20 beta < 1
                            below += 1
                            continue
                        assert room["scale"] == min(unmixed["scale"], 1.0), (unit, rung, tier)
                        if unmixed["scale"] < 1.0:
                            inside += 1
                            for field in ("offer_kwh_per_day", "mean_driver_loss_offering_days"):
                                assert room[field] == unmixed[field]
    assert inside >= 1 and below >= 1


# --- 13. refusals of the room factors and the room table by hand ---------------------------------


def _room_what_if(build, out: Path, factors, splits="1") -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--model", "empirical", "--configurations", *CONFIGURATIONS]
        arguments += ["--splits", splits, "--room-factors", *factors]
        return centre_scale_whatif.main([*_arguments(build), *arguments])


@pytest.mark.parametrize(
    "factors, why",
    [
        (["2", "4"], "1, the mix as first built, must be among the factors"),
        (["0.5", "1"], "at least 1"),
        (["1", "1"], "named twice"),
        (["1", "inf"], "finite"),
        (["1", "nan"], "finite"),
    ],
)
def test_room_factor_refusals(build, tmp_path, factors, why, capsys):
    code = _room_what_if(build, tmp_path / "out", factors)
    printed = capsys.readouterr().out
    assert code == 2 and printed.startswith("COULD NOT RUN: --room-factors") and why in printed
    assert list((tmp_path / "out").iterdir()) == []


def test_the_room_table_by_hand():
    first = {
        "mixed": _rung(0.5, 5.0, 0.0, 0.01),
        "mixed_room_4": _rung(0.6, 6.0, 0.0, 0.0),
        "unmixed": _rung(0.25, 2.5, 0.0, 0.04),
    }
    second = {
        "mixed": _rung(0.0, 0.0, None, None),
        "mixed_room_4": _rung(0.4, 4.0, 0.05, 0.01),
        "unmixed": _rung(0.45, 4.5, 0.1, 0.02),
    }
    records = [
        centre_scale_summary.split_record(seed, {k: {"one-way": v} for k, v in found.items()})
        for seed, found in ((1, first), (2, second))
    ]
    floor = centre_scale_summary.floor_figures(
        ([0.0, 0.5], [0.0, 0.1]),
        ([0.0, 0.0], [0.2, 0.2]),
        2,
        0.05,
        {"mixed_room_4": ([0.0, 0.0], [0.0, 0.02])},
    )
    document = {
        "run": {"splits": 2, "floor_scale": 0.05, "room_factors": [1.0, 4.0]},
        "units": {
            "u": {
                "configurations": {
                    "x": {
                        "driver_risk": 0.02,
                        "floor": {"one-way": floor},
                        "over_splits": centre_scale_summary.over_splits(records, ["c"]),
                    }
                }
            }
        },
    }
    lines = centre_scale_summary._room_lines(document)
    assert lines[0].startswith("Room factors 1, 4: at scale beta the planned charging is (1 - w)")
    rows = [[c.strip() for c in line.split("|")] for line in lines[1:]]
    assert rows[0] == [
        "unit",
        "dial",
        "rung",
        "tier",
        "room",
        "floor slot",
        "floor driver",
        "scale",
        "offer",
        "slot",
        "driver",
        "offer minus unmixed",
    ]
    assert len(rows) == 1 + 2 * 3
    tier_a = ["u", "dial 0.02", "one-way", "A"]
    assert rows[1] == [*tier_a, "1", "0.250", "0.050", "0.250", "2.50", "0.000", "0.010"] + [
        "-1.00 [-4.15, 2.15]"
    ]
    assert rows[2] == [*tier_a, "4", "0.000", "0.010", "0.500", "5.00", "0.025", "0.005"] + [
        "1.50 [-0.30, 3.30]"
    ]
    assert rows[3] == [*tier_a, "unmixed", "0.000", "0.200", "0.350", "3.50", "0.050", "0.030"] + [
        "-"
    ]
    assert centre_scale_summary._room_lines({**document, "run": {"splits": 2}}) == []
