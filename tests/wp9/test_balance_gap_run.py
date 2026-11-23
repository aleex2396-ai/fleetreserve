"""The balance-gap measurement of a recorded run (`balance_gap_run`), on made-up runs.

`balance_gap_fake.build`: a made-up build and matrix, and a recorded `gate_run` and
`learn_run --model empirical` on them (one service hour; the empirical configuration at kappa 0.1
and its oracle partner; three units, the first not split).

1. Both runs are measured: exit 0; one row per configuration, unit, offering day of the run (no
   abstain reason, certificate "optimal") and service hour, and no other; each number equal to
   the exact `B_t` of the tests of V1 to V8 (`validity_errors`, `tests/wp11`) on the certificate
   formed again here by `gate_day.prepare_day`, at 0 (the centre gap), 1, the validity limit and
   the run's scales read here from its results; the counts of the results file those of the exact
   gaps; one sentence per unit on V8's premise.
2. A recorded certified value one double above the value formed again: that day refused under that
   configuration only, named with both values in the results and the summary, no row for it, the
   other configuration's row of that day written, exit 1.
3. The first certificate solve made to time out (a stand-in for the solver answers the first call
   given the box limit as HiGHS answers at its time limit): that day counted and listed with its
   configuration, no row and no gap of 0 for it, every other offering day measured, and exit 2
   (could not run in full, reading M10) with the measured part written and labelled (the label
   leads the summary and the csv, and the results hold `measured_in_full` false). Every
   certificate solve made to time out: no row, every largest gap None, the hours 0, the sentence
   "no hour was measured" for each unit, exit 2. A refused day beside a timeout: exit 1. A run
   without a timeout carries no label and exits 0.
   The flag `held_up_to_scale_1` (and `held_up_to_validity_limit`) beside the counts of offering
   days, days measured and days timed out: true only when every offering day was measured and no
   measured hour can fail by the gap alone; null when a day is unmeasured and no measured hour
   fails; false when a measured hour fails (readings M2, M10). A premise that held says "needed,
   not enough".
   Ties: a gain gap exactly at ZERO_GAIN or ROW_TOLERANCE, or a centre gap exactly at FAILURE_KWH,
   through the command, are not above their levels (`test_balance_gap` pins the same at the
   counts). A run scale of 0 is named in the results and the summary.
4. A run whose results name a held-out unit, or that was made on the sealed-data route: could not
   run (exit 2, "no route") with nothing opened in a made-up repository holding a sealed folder
   (`fake_heldout.sealed_repo`), the matrix and made-up held-out tables, and nothing in the run
   folder but its manifest and results file (`sealed_watch.watch`).
5. Could not run (exit 2): a run that ended with exit 1, a stale run, a run of another command or
   a trial, each before anything in the run folder but its manifest and results file is opened; a
   matrix changed since the run, `--workers 0`, a path written as a URL.
6. End to end under the run recorder in a throwaway git repository: the manifest records exit 0 and
   the SHA-256 of the four outputs, each verified by `run_verify.require_verified`.
"""

import csv
import dataclasses
import json
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from balance_gap_fake import GATE_MODULE, LEARN_MODULE, build, read_rows, record, rehash, verified
from gate_fake import COMMIT
from git_run import run_git
from learn_fake import GATE_UNITS, small_matrix
from scipy.optimize import OptimizeResult

from fleetreserve import gate_day, gate_run, lp, run_verify, sealed_route
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.forecast_model import TableForecaster, read_table
from fleetreserve.gate_config import read
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN, AffinePolicy
from fleetreserve.replay import FAILURE_KWH
from fleetreserve.run_record import record_run

HERE = Path(__file__).resolve().parent
for _helpers in (HERE.parent / "wp11", HERE.parent / "wp2", HERE.parent / "wp14"):
    if str(_helpers) not in sys.path:
        sys.path.insert(0, str(_helpers))
from fake_heldout import NORWAY_FILE, sealed_repo  # noqa: E402
from sealed_watch import watch  # noqa: E402
from solver_timeout_fake import force_timeouts  # noqa: E402
from validity_errors import balance_gaps  # noqa: E402

DAYS = {"gate": "gate-days.csv", "learn": "learn-gate-days.csv"}
RESULTS = {"gate": "gate.json", "learn": "learn-gate.json"}
SCALES = {
    "fixed_tier_a": ("rungs", "fixed", "beta_hat"),
    "fixed_tier_b": ("rungs", "fixed", "tier_b", "beta_hat"),
    "every_request_tier_a": ("fixed_every_request", "beta_used"),
    "every_request_tier_b": ("fixed_every_request", "tier_b", "beta_used"),
}
LEVELS = {
    "zero_gain": Fraction(ZERO_GAIN),
    "failure_over_limit": Fraction(FAILURE_KWH) / Fraction(VALIDITY_LIMIT),
    "row_tolerance": Fraction(1e-7),
}


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    return build(tmp_path_factory.mktemp("balance"))


def measure(made, run, out: Path, monkeypatch, *extra, root=None) -> int:
    from fleetreserve import balance_gap_run

    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--gate-run", str(run), "--tables", str(made["tables"])]
    arguments += ["--truth-run", str(made["truth"]), *extra]
    return balance_gap_run.main(arguments, root=made["folder"] if root is None else root)


def copy_run(made, which: str, folder: Path, label: str) -> Path:
    run = folder / f"20260103T000000Z-{label}"
    shutil.copytree(made[which], run)
    return run


def offering(row) -> bool:
    return row["abstain_reason"] == "" and row["certificate_outcome"] == "optimal"


def at(document, *keys):
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


class FormedAgain:
    """The certificates of the made-up run formed again by `gate_day.prepare_day`."""

    def __init__(self, made, which: str):
        self.matrix, _ = read(made["matrix"])
        m = self.matrix
        self.days = {
            u: load_unit_days(
                made["tables"], u, rho=m.rho, no_fit=m.no_fit, small_fleet=m.small_fleet
            )
            for u in m.gate_units
        }
        if which == "learn":
            table = read_table(made["learn"] / "learn-forecasts.csv")
            self.forecasters = {u: TableForecaster(table[table["unit"] == u]) for u in self.days}
        else:
            self.forecasters = {
                u: EmpiricalForecaster(d, reference=m.reference, min_cell=m.min_cell)
                for u, d in self.days.items()
            }
        self.found = {}

    def policy(self, unit, configuration, site, date):
        key = (unit, configuration, site, date)
        if key not in self.found:
            (truth,) = [
                t
                for t in self.days[unit].days
                if (str(t.record.site), str(t.record.date)) == (site, date)
            ]
            prepared = gate_day.prepare_day(
                truth,
                self.forecasters[unit],
                self.matrix.configuration(configuration),
                gate_run.settings(self.matrix),
            )
            assert prepared.abstain_reason is None
            self.found[key] = prepared.certificate.policy
        return self.found[key]


@pytest.mark.parametrize("which", ["gate", "learn"])
def test_every_offering_day_of_a_run_is_measured_and_each_number_is_the_exact_gap(
    made, which, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    out = tmp_path / "out"
    assert measure(made, made[which], out, monkeypatch) == 0
    printed = capsys.readouterr().out
    days = read_rows(made[which] / DAYS[which])
    document = json.loads((made[which] / RESULTS[which]).read_text())
    rows = read_rows(out / "balance-gaps.csv")
    results = json.loads((out / "balance-gap.json").read_text())
    assert results["exit_status"] == 0
    assert results["timeouts"] == [] and results["refused_days"] == [] and results["failed"] == []
    expected = {
        (r["configuration"], r["unit"], r["site"], r["date"], r["role"])
        for r in days
        if offering(r)
    }
    assert expected, "the made-up run has offering days"
    found = [(r["configuration"], r["unit"], r["site"], r["date"], r["role"]) for r in rows]
    formed = FormedAgain(made, which)
    hours = {}
    for key in expected:
        policy = formed.policy(key[1], key[0], key[2], key[3])
        hours[key] = policy.n_service
    assert sorted(found) == sorted(k for k, n in hours.items() for _ in range(n))
    exact = {}  # (configuration, unit) -> [(centre, gain, B at each scale)]
    for row in rows:
        configuration, unit, hour = row["configuration"], row["unit"], int(row["hour"])
        policy = formed.policy(unit, configuration, row["site"], row["date"])
        centre = balance_gaps(policy, 0.0)[hour - 1]
        gain = balance_gaps(policy, 1.0)[hour - 1] - centre
        assert float(row["centre_gap_kwh"]) == float(centre)
        assert float(row["gain_gap_kwh"]) == float(gain)
        assert float(row["b_scale_1_kwh"]) == float(balance_gaps(policy, 1.0)[hour - 1])
        limit = balance_gaps(policy, VALIDITY_LIMIT)[hour - 1]
        assert float(row["b_validity_limit_kwh"]) == float(limit)
        entry = document["configurations"][configuration]["units"][unit]
        for name, keys in SCALES.items():
            scale = at(entry, *keys)
            if scale is None:
                assert row[f"b_{name}_kwh"] == ""
            else:
                b = balance_gaps(policy, float(scale))[hour - 1]
                assert float(row[f"b_{name}_kwh"]) == float(b)
        exact.setdefault((configuration, unit), []).append((centre, gain, limit))
    for configuration, entry in document["configurations"].items():
        for unit in entry["units"]:
            told = results["units"][unit]["configurations"][configuration]
            gaps = exact.get((configuration, unit), [])
            offered = [r for r in days if r["configuration"] == configuration and r["unit"] == unit]
            assert told["offering_days_in_run"] == sum(map(offering, offered))
            assert told["days_measured"] == told["offering_days_in_run"]
            assert told["days_timed_out"] == told["days_refused"] == 0
            assert told["hours"] == len(gaps)
            assert told["hours_shared_exactly"] == sum(c == 0 and g == 0 for c, g, _ in gaps)
            for name, level in LEVELS.items():
                assert told["gain_gap_above"][name] == sum(g > level for _, g, _ in gaps)
            largest = max((g for _, g, _ in gaps), default=None)
            assert told["largest_gain_gap_kwh"] == (None if largest is None else float(largest))
            failing = sum(b > Fraction(FAILURE_KWH) for _, _, b in gaps)
            assert told["at_scales"]["validity_limit"]["hours_can_fail_alone"] == failing
            for name, keys in SCALES.items():
                value = at(entry["units"][unit], *keys)
                assert told["scales"][name] == (None if value is None else float(value))
    for unit in GATE_UNITS:
        premise = results["units"][unit]["premise"]
        assert premise["sentence"].startswith(f"{unit}: ")
        assert "V8's premise" in premise["sentence"]
        assert premise["sentence"] in printed
        measured = sum(told["hours"] for told in results["units"][unit]["configurations"].values())
        assert premise["hours_measured"] == measured
        if measured:
            held = premise["hours_can_fail_alone_at_scale_1"] == 0
            assert premise["held_up_to_scale_1"] is held
    assert (out / "balance-gap-summary.txt").read_text() == printed
    timing = json.loads((out / "balance-gap-timing.json").read_text())
    assert timing["workers"] == 1


@pytest.mark.parametrize("which", ["gate", "learn"])
def test_every_scale_the_command_reads_is_a_key_the_real_writer_wrote(
    made, which, tmp_path, monkeypatch, capsys
):
    """The runs of `made` are written by `gate_run.main` and `learn_run.main`, not by hand: when
    their writer renames or drops a key this command reads, a column is empty and this fails."""
    from fleetreserve import balance_gap_run

    verified(monkeypatch)
    document = json.loads((made[which] / RESULTS[which]).read_text())
    for name, keys in balance_gap_run.SCALE_FIELDS.items():
        found = [
            at(unit, *keys)
            for entry in document["configurations"].values()
            for unit in entry["units"].values()
        ]
        assert any(isinstance(v, int | float) for v in found), name
    out = tmp_path / "out"
    assert measure(made, made[which], out, monkeypatch) == 0
    capsys.readouterr()
    rows = read_rows(out / "balance-gaps.csv")
    assert rows
    for name in balance_gap_run.SCALE_FIELDS:
        assert any(r[f"b_{name}_kwh"] != "" for r in rows), name


def _edit_every_request(run: Path, results_name: str, change) -> None:
    """`change(record)` applied to every unit entry's `fixed_every_request` of the run's results
    (written by the real writer); the run listed again in its manifest."""
    document = json.loads((run / results_name).read_text())
    for entry in document["configurations"].values():
        for unit in entry["units"].values():
            if isinstance(unit.get("fixed_every_request"), dict):
                change(unit["fixed_every_request"])
    (run / results_name).write_text(json.dumps(document))
    rehash(run)


def _drop_marker(record):
    del record["earlier_driver_bound"]


def _old_key_null(record):
    record["every_scale_driver_bound"] = None


def _old_key_present(record):
    record["every_scale_driver_bound"] = {"beta_used": 0.5}


def _old_key_and_no_marker(record):
    _drop_marker(record)
    _old_key_present(record)


@pytest.mark.parametrize(
    "change",
    [_drop_marker, _old_key_null, _old_key_present, _old_key_and_no_marker],
    ids=["neither key", "old key null", "old key present", "old key without the marker"],
)
def test_a_run_not_of_the_present_shape_is_refused_and_nothing_is_written(
    made, change, tmp_path, monkeypatch, capsys
):
    """The present shape is a positive test: tier A of the every-request route carries the key
    `earlier_driver_bound`, which the present writer always writes beside `beta_used`, and not
    the key `every_scale_driver_bound` of the runs made before 1063724, even as null."""
    verified(monkeypatch)
    run = copy_run(made, "gate", tmp_path, "gate-shape")
    _edit_every_request(run, "gate.json", change)
    out = tmp_path / "out"
    assert measure(made, run, out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: ")
    assert "earlier_driver_bound" in printed and "1063724" in printed
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("which", ["gate", "learn"])
def test_a_run_of_the_present_shape_is_accepted(made, which, tmp_path, monkeypatch, capsys):
    verified(monkeypatch)
    document = json.loads((made[which] / RESULTS[which]).read_text())
    records = [
        unit["fixed_every_request"]
        for entry in document["configurations"].values()
        for unit in entry["units"].values()
        if "fixed_every_request" in unit
    ]
    assert records
    for found in records:
        assert isinstance(found["earlier_driver_bound"], dict)
        assert "every_scale_driver_bound" not in found
    assert measure(made, made[which], tmp_path / "out", monkeypatch) == 0
    capsys.readouterr()


DISTINCT = {  # each scale of the made-up run set to a figure of its own
    "fixed_tier_a": 0.125,
    "fixed_tier_b": 0.25,
    "every_request_tier_a": 0.375,
    "every_request_tier_b": 0.5,
}


@pytest.mark.parametrize("which", ["gate", "learn"])
def test_each_scale_column_equals_its_own_figure_of_the_run(
    made, which, tmp_path, monkeypatch, capsys
):
    """The made-up runs are written by the real writers but record scales of 0 or 1, equal across
    columns, so a crossed key would pass. Here the figures of the written results, at the keys of
    `SCALES` (this file's own table), are set to four different numbers, and every certificate has
    the centre gap 1e-5 and the gain gap 1e-5: `B_t(beta) = 1e-5 (1 + beta)`, a number of its
    own for each column."""
    verified(monkeypatch)
    run = copy_run(made, which, tmp_path, f"{which}-distinct")
    document = json.loads((run / RESULTS[which]).read_text())
    set_by = {}
    for entry in document["configurations"].values():
        for unit in entry["units"].values():
            for name, value in DISTINCT.items():
                *parents, last = SCALES[name]
                holder = unit
                for key in parents:
                    holder = holder.get(key) if isinstance(holder, dict) else None
                if isinstance(holder, dict) and isinstance(holder.get(last), int | float):
                    holder[last] = value
                    set_by[name] = set_by.get(name, 0) + 1
    assert set(set_by) == set(DISTINCT), set_by
    (run / RESULTS[which]).write_text(json.dumps(document))
    rehash(run)
    _swap_policies(monkeypatch, _one_hour_gaps(1e-5, 1e-5))
    out = tmp_path / "out"
    assert measure(made, run, out, monkeypatch) == 0
    capsys.readouterr()
    rows = read_rows(out / "balance-gaps.csv")
    assert rows
    seen = set()
    for row in rows:
        for name, value in DISTINCT.items():
            cell = row[f"b_{name}_kwh"]
            if cell == "":
                continue
            seen.add(name)
            exact = Fraction(1e-5) + Fraction(value) * Fraction(1e-5)
            assert float(cell) == float(exact), (name, cell)
    assert seen == set(DISTINCT)


def _rewrite_days(path: Path, change) -> dict:
    """Apply `change` to the first offering row of a split unit; returns that row as it was."""
    lines = path.read_text().splitlines()
    rows = list(csv.DictReader(lines))
    (index,) = [
        k
        for k, r in enumerate(rows)
        if offering(r) and r["role"] == "test" and r["unit"] == GATE_UNITS[1]
    ][:1]
    before = dict(rows[index])
    change(rows[index])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return before


def test_a_recorded_value_that_differs_from_the_value_formed_again_refuses_that_day(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    run = copy_run(made, "gate", tmp_path, "gate-changed")
    changed = {}

    def change(row):
        changed["value"] = repr(float(np.nextafter(float(row["certified_value_kwh"]), np.inf)))
        row["certified_value_kwh"] = changed["value"]

    before = _rewrite_days(run / "gate-days.csv", change)
    rehash(run)
    out = tmp_path / "out"
    assert measure(made, run, out, monkeypatch) == 1
    printed = capsys.readouterr().out
    results = json.loads((out / "balance-gap.json").read_text())
    assert results["exit_status"] == 1
    (refused,) = results["refused_days"]
    day = (before["unit"], before["configuration"], before["site"], before["date"])
    assert (refused["unit"], refused["configuration"], refused["site"], refused["date"]) == day
    assert refused["recorded"]["certified_value_kwh"] == changed["value"]
    assert refused["re_formed"]["certified_value_kwh"] == before["certified_value_kwh"]
    told = results["units"][before["unit"]]["configurations"][before["configuration"]]
    assert told["days_refused"] == 1
    assert told["days_measured"] == told["offering_days_in_run"] - 1
    rows = read_rows(out / "balance-gaps.csv")
    keys = {(r["unit"], r["configuration"], r["site"], r["date"]) for r in rows}
    assert day not in keys
    others = [
        r
        for r in read_rows(made["gate"] / "gate-days.csv")
        if offering(r)
        and (r["unit"], r["site"], r["date"]) == day[0:1] + day[2:]
        and r["configuration"] != day[1]
    ]
    assert others, "the other configuration offers on that day"
    for r in others:
        assert (r["unit"], r["configuration"], r["site"], r["date"]) in keys
    assert printed.startswith("FAILED: 1 day refused")
    assert f"{day[0]} {day[2]} {day[3]} under {day[1]}" in printed
    assert changed["value"] in printed


def _first_solve_times_out(monkeypatch) -> list:
    """The first solve given the box limit answers as HiGHS at its time limit (module text)."""
    real, calls = lp.linprog, []

    def stand_in(c, *args, options=None, **kwargs):
        if options is not None and options.get("time_limit") == lp.BOX_SECONDS:
            calls.append(len(c))
            if len(calls) == 1:
                return OptimizeResult(
                    x=np.zeros(len(c)),
                    fun=0.0,
                    status=1,
                    success=False,
                    message="Time limit reached. (made-up answer, no solve)",
                    nit=0,
                )
        return real(c, *args, options=options, **kwargs)

    monkeypatch.setattr(lp, "linprog", stand_in)
    return calls


def test_a_certificate_that_times_out_is_counted_and_listed_and_never_a_gap_of_0(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    calls = _first_solve_times_out(monkeypatch)
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert len(calls) > 1
    results = json.loads((out / "balance-gap.json").read_text())
    (timed_out,) = results["timeouts"]
    assert results["refused_days"] == [] and results["failed"] == []
    day = (timed_out["unit"], timed_out["site"], timed_out["date"])
    (configuration,) = timed_out["configurations"]
    rows = read_rows(out / "balance-gaps.csv")
    keys = {(r["unit"], r["configuration"], r["site"], r["date"]) for r in rows}
    assert (day[0], configuration, day[1], day[2]) not in keys
    told = results["units"][day[0]]["configurations"][configuration]
    assert told["days_timed_out"] == 1
    assert told["days_measured"] == told["offering_days_in_run"] - 1
    offered = [r for r in read_rows(made["gate"] / "gate-days.csv") if offering(r)]
    assert len(keys) == len(offered) - 1
    assert "Timeouts (1): " in printed and f"{day[0]} {day[1]} {day[2]}" in printed
    assert "1 offering day not measured" in results["units"][day[0]]["premise"]["sentence"]
    assert results["exit_status"] == 2
    assert results["run"]["measured_in_full"] is False
    assert printed.startswith("NOT MEASURED IN FULL")
    assert (out / "balance-gap-summary.txt").read_text().startswith("NOT MEASURED IN FULL")
    assert (out / "balance-gaps.csv").read_text().startswith("# NOT MEASURED IN FULL")
    premise = results["units"][day[0]]["premise"]
    assert premise["offering_days"] == premise["days_measured"] + 1
    assert premise["days_timed_out"] == 1 and premise["offering_days_not_measured"] == 1
    assert premise["held_up_to_scale_1"] is None  # never true beside an unmeasured day
    assert premise["held_up_to_validity_limit"] is None
    for unit in GATE_UNITS:
        found = results["units"][unit]["premise"]
        assert found["hours_can_fail_alone_at_scale_1"] == 0 and found["hours_measured"] > 0
        assert (
            found["offering_days"] == found["days_measured"] + found["offering_days_not_measured"]
        )
        if unit != day[0]:
            assert found["days_timed_out"] == 0
            assert found["held_up_to_scale_1"] is True
            assert found["held_up_to_validity_limit"] is True


def test_every_certificate_timed_out_measures_no_hour_and_writes_no_gap(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    forced = force_timeouts(monkeypatch, lp.BOX_SECONDS)
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert forced
    assert read_rows(out / "balance-gaps.csv") == []
    results = json.loads((out / "balance-gap.json").read_text())
    offered = [r for r in read_rows(made["gate"] / "gate-days.csv") if offering(r)]
    listed = sum(len(t["configurations"]) for t in results["timeouts"])
    assert listed == len(offered)
    for unit in GATE_UNITS:
        for told in results["units"][unit]["configurations"].values():
            assert told["hours"] == 0 and told["days_measured"] == 0
            assert told["days_timed_out"] == told["offering_days_in_run"]
            assert told["largest_centre_gap_kwh"] is None
            assert told["largest_gain_gap_kwh"] is None
            for scale in told["at_scales"].values():
                assert scale["largest_b_kwh"] is None
        premise = results["units"][unit]["premise"]
        assert premise["held_up_to_scale_1"] is None
        assert premise["held_up_to_validity_limit"] is None
        assert "no hour was measured" in premise["sentence"]
        assert premise["sentence"] in printed


def _swap_policies(monkeypatch, swap) -> None:
    """Every offering day's certificate policy replaced by `swap(policy)` (made-up gaps)."""
    real = gate_day.prepare_day

    def stand_in(*args, **kwargs):
        prepared = real(*args, **kwargs)
        if prepared.abstain_reason is None:
            certificate = dataclasses.replace(
                prepared.certificate, policy=swap(prepared.certificate.policy)
            )
            prepared = dataclasses.replace(prepared, certificate=certificate)
        return prepared

    monkeypatch.setattr(gate_day, "prepare_day", stand_in)


def _one_hour_gaps(centre: float, gain: float):
    """Two cars, centre 0, width 1: car 1 plans `centre` with gain 1, car 2 has the gain `gain`:
    the exact gaps are `centre` and `abs(gain)`."""

    def swap(policy):
        assert policy.n_service == 1
        g = np.zeros((2, 2))
        G = np.zeros((2, 2, 1))
        g[0, 1] = centre
        G[0, 1, 0] = 1.0
        G[1, 1, 0] = gain
        return AffinePolicy([0.0], [1.0], g, G)

    return swap


def test_a_measured_failing_hour_stays_false_beside_an_unmeasured_day(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    above = float(np.nextafter(FAILURE_KWH, 1.0))
    _swap_policies(monkeypatch, _one_hour_gaps(above, 0.0))
    _first_solve_times_out(monkeypatch)
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch) == 2
    capsys.readouterr()
    results = json.loads((out / "balance-gap.json").read_text())
    assert len(results["timeouts"]) == 1
    for unit in GATE_UNITS:
        premise = results["units"][unit]["premise"]
        if not premise["hours_measured"]:
            continue
        assert premise["hours_can_fail_alone_at_scale_1"] == premise["hours_measured"]
        assert premise["held_up_to_scale_1"] is False
        assert premise["held_up_to_validity_limit"] is False
        assert "did not hold" in premise["sentence"]


def test_a_refused_day_beside_a_timeout_exits_1_and_the_timeout_is_still_listed(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    run = copy_run(made, "gate", tmp_path, "gate-changed")

    def change(row):
        row["certified_value_kwh"] = repr(
            float(np.nextafter(float(row["certified_value_kwh"]), np.inf))
        )

    _rewrite_days(run / "gate-days.csv", change)
    rehash(run)
    _first_solve_times_out(monkeypatch)
    out = tmp_path / "out"
    assert measure(made, run, out, monkeypatch) == 1
    printed = capsys.readouterr().out
    results = json.loads((out / "balance-gap.json").read_text())
    assert results["exit_status"] == 1
    assert len(results["refused_days"]) == 1 and len(results["timeouts"]) == 1
    assert "FAILED: 1 day refused" in printed and "Timeouts (1): " in printed


def test_a_run_without_a_timeout_carries_no_label_and_says_a_held_premise_is_not_enough(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch) == 0
    printed = capsys.readouterr().out
    results = json.loads((out / "balance-gap.json").read_text())
    assert results["run"]["measured_in_full"] is True
    assert "NOT MEASURED IN FULL" not in printed
    assert not (out / "balance-gaps.csv").read_text().startswith("#")
    for unit in GATE_UNITS:
        premise = results["units"][unit]["premise"]
        assert premise["days_timed_out"] == 0 and premise["offering_days_not_measured"] == 0
        assert premise["offering_days"] == premise["days_measured"]
        assert premise["held_up_to_scale_1"] is True
        assert premise["held_up_to_validity_limit"] is True
        assert "needed, not enough" in premise["sentence"]
        assert "V6" in premise["sentence"] and "V8" in premise["sentence"]


@pytest.mark.parametrize(
    "centre, gain, above_zero, above_quotient, above_row, at_1, at_limit",
    [
        (FAILURE_KWH, 0.0, 0, 0, 0, 0, 0),
        (float(np.nextafter(FAILURE_KWH, 1.0)), 0.0, 0, 0, 0, 1, 1),
        (0.0, ZERO_GAIN, 0, 0, 0, 0, 0),
        (0.0, float(np.nextafter(ZERO_GAIN, 1.0)), 1, 0, 0, 0, 0),
        (0.0, -ROW_TOLERANCE, 1, 1, 0, 0, 1),
        (0.0, float(np.nextafter(ROW_TOLERANCE, 1.0)), 1, 1, 1, 0, 1),
    ],
)
def test_a_gap_exactly_at_a_level_is_not_above_it_through_the_command(
    made,
    tmp_path,
    monkeypatch,
    capsys,
    centre,
    gain,
    above_zero,
    above_quotient,
    above_row,
    at_1,
    at_limit,
):
    verified(monkeypatch)
    _swap_policies(monkeypatch, _one_hour_gaps(centre, gain))
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch) == 0
    capsys.readouterr()
    results = json.loads((out / "balance-gap.json").read_text())
    hours = 0
    for unit in results["units"].values():
        for e in unit["configurations"].values():
            hours += e["hours"]
            assert e["gain_gap_above"] == {
                "zero_gain": above_zero * e["hours"],
                "failure_over_limit": above_quotient * e["hours"],
                "row_tolerance": above_row * e["hours"],
            }
            assert e["at_scales"]["scale_1"]["hours_can_fail_alone"] == at_1 * e["hours"]
            assert e["at_scales"]["validity_limit"]["hours_can_fail_alone"] == at_limit * e["hours"]
    assert hours > 0


def _set_fixed_scale(run: Path, value: float) -> int:
    """The fixed plan's tier A scale of every unit entry that has one set to `value`; the run
    listed again in its manifest. Returns how many entries were set."""
    document = json.loads((run / "gate.json").read_text())
    count = 0
    for entry in document["configurations"].values():
        for unit in entry["units"].values():
            fixed = at(unit, "rungs", "fixed")
            if isinstance(fixed, dict) and fixed.get("beta_hat") is not None:
                fixed["beta_hat"] = value
                count += 1
    (run / "gate.json").write_text(json.dumps(document))
    rehash(run)
    return count


def test_a_run_scale_of_0_is_named_and_its_count_is_the_centre_gap_alone(
    made, tmp_path, monkeypatch, capsys
):
    verified(monkeypatch)
    run = copy_run(made, "gate", tmp_path, "gate-zero")
    assert _set_fixed_scale(run, 0.0) > 0
    above = float(np.nextafter(FAILURE_KWH, 1.0))
    _swap_policies(monkeypatch, _one_hour_gaps(above, 0.5))
    out = tmp_path / "out"
    assert measure(made, run, out, monkeypatch) == 0
    printed = capsys.readouterr().out
    results = json.loads((out / "balance-gap.json").read_text())
    named = 0
    for unit in results["units"].values():
        for e in unit["configurations"].values():
            zero = sorted(name for name, beta in e["scales"].items() if beta == 0.0)
            assert e["scales_at_0"] == zero
            if e["scales"]["fixed_tier_a"] == 0.0:
                named += 1
                assert "fixed_tier_a" in e["scales_at_0"]
            for name in zero:
                at_0 = e["at_scales"][name]
                assert at_0["scale"] == 0.0
                # the count at 0 is the centre gap alone, though the gate does not replay at 0
                assert at_0["hours_can_fail_alone"] == e["hours"]
                assert at_0["largest_b_kwh"] == (above if e["hours"] else None)
    assert named > 0
    assert "scale 0" in results["definitions"]["scale_0"]
    assert "the gate does not replay at scale 0" in printed
    for row in read_rows(out / "balance-gaps.csv"):
        if row["b_fixed_tier_a_kwh"] != "":
            assert row["b_fixed_tier_a_kwh"] == row["centre_gap_kwh"]


@pytest.mark.parametrize("case", ["held-out unit", "sealed-data route"])
def test_a_held_out_run_could_not_run_before_anything_else_is_opened(
    case, tmp_path, monkeypatch, capsys
):
    from fleetreserve import balance_gap_run

    verified(monkeypatch)
    repo = sealed_repo(tmp_path / "made", {NORWAY_FILE: b"made-up sealed bytes\n"}, package=False)
    held = sealed_route.HELD_OUT_UNITS[0]
    unit = held if case == "held-out unit" else GATE_UNITS[1]
    tables = repo / "data" / "derived" / "truth-made-up"
    tables.mkdir(parents=True)
    for kind in ("cars", "days"):
        (tables / f"{unit}-{kind}.csv").write_text("made-up\n")
    matrix = small_matrix()
    matrix["gate_units"] = [unit]
    (repo / "matrix.json").write_text(json.dumps(matrix))
    source = tmp_path / "source"
    source.mkdir()
    document = {
        "trial": False,
        "partial": False,
        "label": "gate run",
        "units": {unit: {}},
        "configurations": {"main-k0.1-empirical": {"units": {unit: {}}}},
        "matrix": {"file": "matrix.json", "sha256": "0" * 64},
    }
    if case == "sealed-data route":
        document[ROUTE_KEY] = {"kind": "made-up"}
    (source / "learn-gate.json").write_text(json.dumps(document))
    (source / "learn-gate-days.csv").write_text("made-up\n")
    (source / "learn-forecasts.csv").write_text("made-up\n")
    run = record(tmp_path, source, LEARN_MODULE, "learn-made-up", extra=["--model", "empirical"])
    arguments = ["--gate-run", str(run), "--tables", str(tables)]
    with watch(repo) as in_repo, watch(run) as in_run:
        code = balance_gap_run.main(arguments, root=repo)
    printed = capsys.readouterr().out
    assert code == 2
    assert printed.startswith("COULD NOT RUN: ")
    assert "this command has no route to them" in printed
    assert (
        "held-out units are read only through the sealed-data path at the tag frozen-plan, or at a "
        "logged parsing fix of it, and this command has no route to them; nothing else was read"
    ) in printed
    assert len(printed.splitlines()) == 1
    assert in_repo == []
    assert {Path(path).name for _, path in in_run} <= {"manifest.json", "learn-gate.json"}


def _manifest(run: Path, **changes) -> None:
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(changes)
    (run / "manifest.json").write_text(json.dumps(manifest))


def _trial(run: Path) -> None:
    document = json.loads((run / "gate.json").read_text())
    document["trial"] = True
    (run / "gate.json").write_text(json.dumps(document))
    rehash(run)


CASES = {
    "ended with exit 1": (lambda run: _manifest(run, exit_code=1), "not verified"),
    "stale": (lambda run: _manifest(run, commit="b" * 40), "stale"),
    "another command": (
        lambda run: _manifest(run, command=["python", "-m", "fleetreserve.other"]),
        "not a recorded run of",
    ),
    "a trial": (_trial, "trial"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_run_that_cannot_be_trusted_could_not_run(made, case, tmp_path, monkeypatch, capsys):
    verified(monkeypatch)
    run = copy_run(made, "gate", tmp_path, "gate-copy")
    change, words = CASES[case]
    change(run)
    out = tmp_path / "out"
    with watch(run) as seen:
        assert measure(made, run, out, monkeypatch) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: ") and words in printed
    assert list(out.iterdir()) == []
    assert {Path(path).name for _, path in seen} <= {"manifest.json", "gate.json"}


def test_a_matrix_changed_since_the_run_could_not_run(made, tmp_path, monkeypatch, capsys):
    verified(monkeypatch)
    changed = small_matrix()
    changed["n_calibration"] = 19
    (tmp_path / "matrix.json").write_text(json.dumps(changed))
    out = tmp_path / "out"
    assert measure(made, made["gate"], out, monkeypatch, root=tmp_path) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: ") and "another matrix" in printed


@pytest.mark.parametrize(
    "extra, words",
    [(["--workers", "0"], "--workers must be at least 1")],
)
def test_bad_arguments_could_not_run(made, extra, words, tmp_path, monkeypatch, capsys):
    verified(monkeypatch)
    assert measure(made, made["gate"], tmp_path / "out", monkeypatch, *extra) == 2
    assert words in capsys.readouterr().out


def test_a_run_named_as_a_url_could_not_run(made, tmp_path, monkeypatch, capsys):
    from fleetreserve import balance_gap_run

    verified(monkeypatch)
    assert balance_gap_run.main(["--gate-run", "file:runs/x"], root=made["folder"]) == 2
    assert "written as a URL" in capsys.readouterr().out


WRAPPER = """
import sys
from pathlib import Path

from fleetreserve import balance_gap_run, gate_run, run_verify

commit = "{commit}"
run_verify._check_commit = lambda folder, recorded: None
gate_run._head_commit = lambda: commit
balance_gap_run._head_commit = lambda: commit
sys.exit(balance_gap_run.main(sys.argv[1:-1], root=Path(sys.argv[-1])))
"""


def _git(repo: Path, *args: str) -> None:
    settings = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    settings += ["-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]
    run_git(["git", *settings, "-C", str(repo), *args], capture_output=True, check=True)


def test_a_recorded_run_of_the_command_writes_verified_outputs(made, tmp_path, monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".gitignore").write_text("runs.noindex/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-q", "-m", "first")
    command = [sys.executable, "-c", WRAPPER.format(commit=COMMIT)]
    command += ["--gate-run", str(made["gate"]), "--tables", str(made["tables"])]
    command += ["--truth-run", str(made["truth"]), str(made["folder"])]
    assert record_run("wp9-balance-gap-made-up", command, start_dir=repo) == 0
    (folder,) = (repo / "runs.noindex").iterdir()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["exit_code"] == 0 and manifest["command"] == command
    names = (
        "balance-gaps.csv",
        "balance-gap.json",
        "balance-gap-summary.txt",
        "balance-gap-timing.json",
    )
    assert set(names) <= set(manifest["outputs"])
    for name in names:
        assert run_verify.require_verified(folder / name) == manifest
    assert read_rows(folder / "balance-gaps.csv")
    assert GATE_MODULE in json.loads((folder / "balance-gap.json").read_text())["run"]["command"]
