"""The intervals over days of real scoring runs on made-up days: `gate_run` (and `learn_run` with
the empirical model) on the small made-up gate units of `learn_fake` with two service hours, with
a made-up verified truth build, recorded with a manifest, without and with `--interval-seed 3
--interval-resamples 30`.

1. The option changes no other output: `gate.json`, `gate-days.csv`, `gate-summary.txt` and
   `gate-t5-breaks.csv` are the same bytes, and the per-day file is the same once each record's
   `resampled` is taken out; every record has one, and some hold pairs beyond the run's own (a
   job whose searches all stop at the run's own scales needs none).
2. Every resample's scale is exact and equals a fresh recalibration by the project's own code on
   the resampled days: `gate_calibrate.calibrate_rung` on the resample's calibration days (each as
   often as drawn), with a new `gate_calibrate.DayCache` whose evaluations are
   `gate_day.evaluate` itself, not the run's; tier A its joint scale, tier B its slot-only scale.
   All 30 resamples of every job, every rung, both tiers, some scales above 0 in each tier.
3. On the run's own calibration days, each once, the recomputed scale is exact and equals the
   run's scale in `gate.json` exactly, for every job, rung and tier.
4. `day_intervals` reads the run (exit 0), every resample exact, and writes an interval for every
   configuration, unit, rung and tier the run scored, each offer interval with the run's own offer
   as its point. The run without the option is refused (2).
5. `learn_run` with the empirical model and the option writes the same per-day file as `gate_run`.
"""

import json

import numpy as np
import pytest
from gate_fake import COMMIT, write_build
from interval_fake import write_manifest
from learn_fake import small_gate_units, small_matrix

from fleetreserve import (
    day_bootstrap,
    day_intervals,
    day_losses,
    gate_calibrate,
    gate_day,
    gate_run,
    learn_run,
    run_verify,
)
from fleetreserve.rungs import RUNGS

OPTION = ["--interval-seed", "3", "--interval-resamples", "30"]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    truth, tables = write_build(folder, small_gate_units(), service_hours=2)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(truth)]
    outs, captured = {}, {}
    original = day_losses.job_record

    def capture(unit, configurations, roles, prepared, *rest):
        captured[(configurations[0], unit)] = (list(roles), list(prepared))
        return original(unit, configurations, roles, prepared, *rest)

    for name, command, extra in (
        ("plain", gate_run.main, []),
        ("resampled", gate_run.main, OPTION),
        ("learn", learn_run.main, ["--model", "empirical", *OPTION]),
    ):
        out = folder / f"20200101T000000Z-{name}"
        out.mkdir()
        with pytest.MonkeyPatch.context() as monkeypatch:
            _verified(monkeypatch)
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            if name == "resampled":
                monkeypatch.setattr(day_losses, "job_record", capture)
            assert command([*arguments, *extra]) == 0
        module = f"fleetreserve.{'learn_run' if name == 'learn' else 'gate_run'}"
        write_manifest(out, module, commit=COMMIT)
        outs[name] = out
    outs["prepared"] = captured
    return outs


def _read(out, name):
    return json.loads((out / name).read_text())


def test_the_option_changes_no_other_output(runs):
    for name in (gate_run.RESULTS_NAME, gate_run.DAYS_NAME, gate_run.SUMMARY_NAME):
        assert (runs["plain"] / name).read_bytes() == (runs["resampled"] / name).read_bytes()
    assert (runs["plain"] / gate_run.BREAKS_NAME).read_bytes() == (
        runs["resampled"] / gate_run.BREAKS_NAME
    ).read_bytes()
    plain = _read(runs["plain"], gate_run.DAY_LOSSES_NAME)
    resampled = _read(runs["resampled"], gate_run.DAY_LOSSES_NAME)
    assert all("resampled" not in r for r in plain["jobs"])
    assert all("resampled" in r for r in resampled["jobs"])
    assert any(r["resampled"]["pairs"]["day"] for r in resampled["jobs"])
    for record in resampled["jobs"]:
        del record["resampled"]
    assert resampled == plain


def test_every_resample_scale_is_a_fresh_recalibration(runs):
    losses = _read(runs["resampled"], gate_run.DAY_LOSSES_NAME)
    above = {"A": 0, "B": 0}
    checked = 0
    for record in losses["jobs"]:
        job = day_bootstrap.Job.from_record(record, RUNGS)
        roles, prepared = runs["prepared"][(record["configurations"][0], record["unit"])]
        cache = gate_calibrate.DayCache(
            prepared, lambda pairs: [gate_day.evaluate(p, b).compact() for p, b in pairs]
        )
        calibration = [d for d, role in enumerate(roles) if role == "calibration"]
        n_test = sum(role == "test" for role in roles)
        rows, _ = day_bootstrap.draws(3, job.unit, len(calibration), n_test, 30)
        for row in rows:
            drawn = [calibration[p] for p in row]
            for rung in RUNGS:
                fresh = gate_calibrate.calibrate_rung(cache, drawn, rung)
                for tier, beta in (("A", fresh["beta_hat"]), ("B", fresh["slot_only"]["beta_hat"])):
                    found = job.scale(rung, tier, row)
                    assert found.exact and found.low == found.high == beta
                    above[tier] += beta > 0
                    checked += 1
    assert checked == len(losses["jobs"]) * 30 * len(RUNGS) * 2
    assert above["A"] > 0 and above["B"] > 0


def test_the_scale_recomputed_on_the_run_days_is_the_run_scale(runs):
    document = _read(runs["resampled"], gate_run.RESULTS_NAME)
    losses = _read(runs["resampled"], gate_run.DAY_LOSSES_NAME)
    checked = 0
    for record in losses["jobs"]:
        job = day_bootstrap.Job.from_record(record, RUNGS)
        for name in job.configurations:
            entry = document["configurations"][name]["units"][job.unit]
            for rung in RUNGS:
                for tier, scored in (
                    ("A", entry["rungs"][rung]),
                    ("B", entry["rungs"][rung]["tier_b"]),
                ):
                    found = job.run_scale(rung, tier)
                    assert found.exact and found.low == found.high == scored["beta_hat"]
                    checked += 1
    assert checked > 0


def test_the_command_reads_the_run(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: COMMIT)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--seed", "3", "--resamples", "30"]
    assert day_intervals.main(["--run", str(runs["resampled"]), *arguments]) == 0
    found = _read(out, day_intervals.RESULTS_NAME)
    document = _read(runs["resampled"], gate_run.RESULTS_NAME)
    assert found["problems"] == []
    for name, configuration in document["configurations"].items():
        for unit, entry in configuration["units"].items():
            if "rungs" not in entry:
                continue
            for rung in RUNGS:
                for tier, scored in (
                    ("A", entry["rungs"][rung]),
                    ("B", entry["rungs"][rung]["tier_b"]),
                ):
                    mine = found["configurations"][name][unit][rung][tier]
                    offer = mine["offer_kwh_per_counting_test_day"]
                    assert offer["point"] == scored["test"]["mean_offer_kwh_per_day"]
                    assert offer["interval"][0] <= offer["interval"][1]
                    assert mine["scale"]["resamples"] == mine["scale"]["resamples_exact"] == 30
    refused = tmp_path / "refused"
    refused.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(refused))
    assert day_intervals.main(["--run", str(runs["plain"]), *arguments]) == 2


def test_the_learn_run_writes_the_same_file(runs):
    gate = (runs["resampled"] / gate_run.DAY_LOSSES_NAME).read_bytes()
    learn = (runs["learn"] / learn_run.DAY_LOSSES_NAME).read_bytes()
    assert learn == gate


def test_the_pooled_block_is_the_units_figures_combined_resample_by_resample(
    runs, tmp_path, monkeypatch
):
    """Readings I13 and I15 on the real run: the pooled points are the units' own points combined
    (the sum of the offers per day; the total over the total counting test days), and the pooled
    intervals are the percentiles of the same combination of resample r of every unit, worked
    here from `Job.scale` and the days alone (every scale is exact in this run)."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: COMMIT)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--run", str(runs["resampled"]), "--seed", "3", "--resamples", "30"]
    assert day_intervals.main(arguments) == 0
    found = _read(out, day_intervals.RESULTS_NAME)
    losses = _read(runs["resampled"], gate_run.DAY_LOSSES_NAME)
    jobs = {
        (name, r["unit"]): day_bootstrap.Job.from_record(r, RUNGS)
        for r in losses["jobs"]
        for name in r["configurations"]
    }
    key = "offer_kwh_per_counting_test_day"
    checked = 0
    for name, block in found["pooled"]["configurations"].items():
        assert len(block["units"]) > 1
        for rung in RUNGS:
            for tier in ("A", "B"):
                mine = [jobs[(name, unit)] for unit in block["units"]]
                per_unit = [
                    found["configurations"][name][unit][rung][tier] for unit in block["units"]
                ]
                days = [len(job.test_days) for job in mine]
                points = [p[key]["point"] for p in per_unit]
                per_day, total = np.zeros(30), np.zeros(30)
                for job, n in zip(mine, days, strict=True):
                    cal, test = day_bootstrap.draws(3, job.unit, len(job.calibration_days), n, 30)
                    for r in range(30):
                        scale = job.scale(rung, tier, cal[r])
                        assert scale.exact
                        offered = scale.low * job.value[job.test_days[test[r]]].sum()
                        per_day[r] += offered / n
                        total[r] += offered
                got = block["rungs"][rung][tier]
                a, b = got["sum_of_unit_offers"][key], got["total_over_total_days"][key]
                assert a["point"] == pytest.approx(sum(points))
                assert b["point"] == pytest.approx(
                    sum(p * n for p, n in zip(points, days, strict=True)) / sum(days)
                )
                levels = [0.025, 0.975]
                assert a["interval"] == pytest.approx(np.quantile(per_day, levels, method="linear"))
                assert b["interval"] == pytest.approx(
                    np.quantile(total / sum(days), levels, method="linear")
                )
                checked += 1
    assert checked == len(found["pooled"]["configurations"]) * len(RUNGS) * 2
