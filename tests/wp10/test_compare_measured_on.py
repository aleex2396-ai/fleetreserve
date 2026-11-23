"""The record `measured_on` of the comparison command (`compare_run`): the prepared days every
method of one unit and variant is measured on, written per unit and variant in the results so
that a run's boxes, audited paths and steered cars can be checked, also against another run.

1. `compare_run.measured_on` of made-up prepared days (three that offer and one that abstains)
   equals this file's own recomputation from the module text's rule: the number of days, the
   number of audited-path rows and the SHA-256 of the compact JSON of one entry per day for the
   boxes, the paths and the steered cars.
2. Changing one day's box, one entry of one audited path or the steered cars changes that digest
   and leaves the other two as they were.
3. In a run of the command on made-up gate units (methods `re-solve` and `fixed-plan`), every unit
   that can be split has the record under both variants, equal to `measured_on` of the days given
   to the methods (recorded through `gate_day.prepare_day`); the unit that cannot be split has
   none. The run at `--no-pause-share 1/3` has the same days, path rows and paths digest as the
   run without the option under every unit and variant, and other steered cars on some.
"""

import dataclasses
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from compare_fake import NAMES, TWO_HOURS, offer_days, two_hour_cars
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import compare_run, gate_day, gate_run, run_verify

SPLITS = 4
METHODS = ["re-solve", "fixed-plan"]
KEYS = {"days", "path_rows", "boxes_sha256", "paths_sha256", "steered_sha256"}


def _digest(items) -> str:
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def _own_record(prepared) -> dict:
    """The record recomputed by the rule of the module text of `compare_run`."""
    boxes, paths, steered = [], [], []
    for day in prepared:
        box = None
        if day.abstain_reason is None:
            policy = day.certificate.policy
            box = [[float(v) for v in policy.m], [float(v) for v in policy.w]]
        boxes.append([day.date, day.site, day.abstain_reason is None, float(day.value), box])
        paths.append([day.date, day.site, [[int(v) for v in row] for row in day.paths]])
        steered.append(
            [day.date, day.site, list(day.steered), [[int(c), why] for c, why in day.unsteered]]
        )
    return {
        "days": len(prepared),
        "path_rows": sum(len(day.paths) for day in prepared),
        "boxes_sha256": _digest(boxes),
        "paths_sha256": _digest(paths),
        "steered_sha256": _digest(steered),
    }


def _made_up_days():
    days = offer_days(3)
    abstaining = dataclasses.replace(
        days[1], date="2020-01-09", abstain_reason="no certificate (made up)", value=0.0
    )
    return [*days, abstaining]


def test_the_record_follows_the_rule_of_the_module_text():
    prepared = _made_up_days()
    found = compare_run.measured_on(prepared)
    assert set(found) == KEYS
    assert found == _own_record(prepared)
    assert found["days"] == 4 and found["path_rows"] == 4 * 8


def _changed_box(day):
    policy = day.certificate.policy
    m = np.array(policy.m, dtype=float)
    m[0] += 1e-9
    moved = dataclasses.replace(policy, m=m)
    return dataclasses.replace(day, certificate=dataclasses.replace(day.certificate, policy=moved))


def _changed_path(day):
    paths = np.array(day.paths, dtype=float)
    paths[3, 0] = -paths[3, 0]
    return dataclasses.replace(day, paths=paths)


def _changed_steered(day):
    return dataclasses.replace(
        day, steered=day.steered[1:], unsteered=((day.steered[0], gate_day.NO_PAUSE),)
    )


@pytest.mark.parametrize(
    ("change", "key"),
    [
        (_changed_box, "boxes_sha256"),
        (_changed_path, "paths_sha256"),
        (_changed_steered, "steered_sha256"),
    ],
    ids=["a box", "a path", "the steered cars"],
)
def test_a_change_moves_its_own_digest_only(change, key):
    prepared = _made_up_days()
    before = compare_run.measured_on(prepared)
    after = compare_run.measured_on([prepared[0], change(prepared[1]), *prepared[2:]])
    assert after[key] != before[key]
    assert {k: v for k, v in after.items() if k != key} == {
        k: v for k, v in before.items() if k != key
    }


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: two_hour_cars(28),
        GATE_UNITS[2]: two_hour_cars(14, "B") + two_hour_cars(14, "C"),
    }
    run, tables = write_build(folder, units, TWO_HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, *extra, prepared=None, share=0):
    run, tables, matrix = build
    folder.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        if prepared is not None:
            prepare = gate_day.prepare_day

            def recorded(truth, forecaster, configuration, settings, no_pause=()):
                found = prepare(truth, forecaster, configuration, settings, no_pause=no_pause)
                prepared.setdefault((truth.day.unit, configuration.name), []).append(found)
                return found

            monkeypatch.setattr(gate_day, "prepare_day", recorded)
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        code = compare_run.main([*arguments, "--methods", *METHODS, *extra])
    assert code == 0
    return json.loads((folder / compare_run.output_names(share)["results"]).read_text())


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    prepared: dict = {}
    plain = _run(build, top / "plain", prepared=prepared)
    third = _run(build, top / "third", "--no-pause-share", "1/3", share=Fraction(1, 3))
    return plain, third, prepared


def test_every_unit_and_variant_that_runs_has_the_record_of_its_days(runs):
    plain, _, prepared = runs
    names = plain["configurations"]
    assert plain["units"][GATE_UNITS[0]]["status"] != "calibratable"
    assert plain["units"][GATE_UNITS[0]]["variants"] == {}
    for unit in GATE_UNITS[1:]:
        for variant, configuration in names.items():
            found = plain["units"][unit]["variants"][variant]["measured_on"]
            assert found == compare_run.measured_on(prepared[(unit, configuration)])
            assert found["days"] == plain["units"][unit]["counting_days"]


def test_the_no_pause_run_has_the_same_paths_and_other_steered_cars(runs):
    plain, third, _ = runs
    moved = 0
    for unit in GATE_UNITS[1:]:
        for variant in plain["configurations"]:
            ours = third["units"][unit]["variants"][variant]["measured_on"]
            theirs = plain["units"][unit]["variants"][variant]["measured_on"]
            for key in ("days", "path_rows", "paths_sha256"):
                assert ours[key] == theirs[key]
            moved += ours["steered_sha256"] != theirs["steered_sha256"]
    assert moved > 0
