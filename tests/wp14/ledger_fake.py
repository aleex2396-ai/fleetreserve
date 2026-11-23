"""Made-up scoring runs for the tests of the ledger of plan T9 (`ledger_run`).

`record(folder, source, module, ...)`: a recorded run folder `<UTC start>-<label>` holding the
files of `source` and a manifest listing their SHA-256, as `run_record` writes it, with the
command `uv run --frozen python -m <module> ...` at `gate_fake.COMMIT` (the commit checks are
replaced by the caller). `exit_code` and `commit` can be changed to make a run that does not
verify or is stale.

`ledger_units()`: three gate units of 5, 24 and 24 days of `gate_fake.gate_cars`, with the second
car leaving 2.3 hours after τ (departure hour 2) on every second day (on every second and every
third day at the pooled unit's two sites) instead of 1.3 hours after τ on every twelfth: that car
is plugged in in hour 0 and the one service hour only and must take its whole room in those two
hours, so a hindsight plan can move its share of hour 1 after the request while a causal rule
fixes the hour-0 power first, and R + A can be above 0 on those days (as in
`tests/wp5/test_hourly_equality`; the command's test requires it on at least one day).

The hand-worked run (`write_hand_run`), one service hour, rho 0, two units of two days each, the
same three cars on both days (so whichever day the split makes the test day has the numbers
below). True departure hour d (the car is plugged in in hours 0 to d - 1), rate r (kW), room e
(kWh, all of it protected):

- `nrel-from-may-2019` (site A): car 1 d = 2, r = 1, e = 1; car 2 d = 3, r = 1, e = 1; car 3
  d = 2, r = 2, e = 1. Hour 1 is the service hour, hour 2 a recovery hour.
  V_hind = b(S) - p(S): b(S) = min(1, 1) + min(1, 1) + min(2, 1) = 3, p(S) = 0 (each car can take
  its room outside hour 1: cars 1 and 3 in hour 0, car 2 in hours 0 and 2), so V_hind = 3.
  A causal rule fixes hour 0 before the request: cars 1 and 3 need exactly 1 kWh in hours 0 and
  1, so their hour-1 power is fixed by their hour-0 power; car 2 with hour-0 power 0 can take any
  0 to 1 in hour 1 and the rest in hour 2. So V_caus = V_aff = V_diag = 1 (car 2's share
  0.5 + 0.5 zeta is diagonal), R = 3 - 1 = 2, A = 0, V_aff - V_diag = 0.
  F: each car's base-set limits are min(power, room) = (1, 1) in hours 0 and 1 and energy 1, so
  the day-average base set is U0 = {0 <= u_t <= 1, u_0 + u_1 <= 1}, a triangle. LP (24)'s set
  p + alpha U0 lies in the fleet's set {u >= 0, 2 <= u_0 + u_1 <= 3} iff its three corners do:
  p_0 + p_1 >= 2 and p_0 + p_1 + alpha <= 3, so alpha = 1, and the largest hourly box is
  alpha min(1, 1) = 1: F = 3 - 1 = 2. With one service hour the block product is the hourly one:
  block V_hind 3, V_caus = V_aff = V_diag = 1, price of the block 0.
- `norway-development-pooled` (site B): three cars d = 3, r = 1, e = 1. Each can take its room in
  hour 2, so hour 0 does not matter: V_hind = V_caus = V_aff = V_diag = 3, R = A = 0; the base set
  is the same triangle, the fleet's set {0 <= u_t <= 3, u_0 + u_1 <= 3}, alpha = 3, box 3, F = 0.

The run's own numbers (made up, as a scoring run would record them): the certified value V_cert
`HAND_CERTIFIED[(configuration, unit)]` and the scales `HAND_SCALES[(configuration, tier)]` per
rung; each day's offer is the scale times V_cert. So, for example, on `nrel-from-may-2019` under
the empirical configuration, tier A, fixed plan (scale 0.5, V_cert 0.8): offer 0.4, R 2, A 0,
U = 1 - 0.8 = 0.2, K = (1 - 0.5) 0.8 = 0.4, and 0.4 + 2 + 0 + 0.2 + 0.4 = 3; at the two-way
rung (scale 1.25) the offer is 1 and K = -0.2.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path

from gate_fake import COMMIT, GATE_UNITS, ROUNDED, gate_cars, matrix_document, write_build
from truth_tables_fake import FakeCar, date_of

from fleetreserve import gate_run
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import CEILINGS, load
from fleetreserve.rungs import RUNGS

GATE_MODULE = "fleetreserve.gate_run"
LEARN_MODULE = "fleetreserve.learn_run"


def ledger_units() -> dict:
    return {
        GATE_UNITS[0]: gate_cars(5),
        GATE_UNITS[1]: gate_cars(24, early_every=2, early_out=12.3),
        GATE_UNITS[2]: gate_cars(12, "B", early_every=2, early_out=12.3)
        + gate_cars(12, "C", early_every=3, early_out=12.3),
    }


def record(
    folder: Path,
    source: Path,
    module: str,
    label: str = "made-up",
    *,
    exit_code: int = 0,
    commit: str = COMMIT,
) -> Path:
    """A recorded run of `module` holding the files of `source` (module text)."""
    run = folder / f"20260102T000000Z-{label}"
    shutil.copytree(source, run)
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir())}
    manifest = {
        "commit": commit,
        "command": ["uv", "run", "--frozen", "python", "-m", module, "--matrix", "matrix.json"],
        "start": "2026-01-02T00:00:00Z",
        "end": "2026-01-02T00:01:00Z",
        "exit_code": exit_code,
        "timed_out": False,
        "outputs": outputs,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def rehash(run: Path) -> None:
    """List the run's files again in its manifest with the SHA-256 they have now."""
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["outputs"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run.iterdir())
        if p.name != "manifest.json"
    }
    (run / "manifest.json").write_text(json.dumps(manifest))


# --- the hand-worked run -------------------------------------------------------------------------

HAND_UNITS = ("nrel-from-may-2019", "norway-development-pooled")
HAND_FLEETS = {
    HAND_UNITS[0]: ("A", ((2, 1.0, 1.0), (3, 1.0, 1.0), (2, 2.0, 1.0))),
    HAND_UNITS[1]: ("B", ((3, 1.0, 1.0), (3, 1.0, 1.0), (3, 1.0, 1.0))),
}
# By hand (module text): V_hind, V_aff (= V_caus = V_diag), the largest box in LP (24)'s set.
HAND_TRUE = {HAND_UNITS[0]: (3.0, 1.0, 1.0), HAND_UNITS[1]: (3.0, 3.0, 3.0)}
EMPIRICAL, ORACLE = ROUNDED
HAND_CERTIFIED = {
    (EMPIRICAL, HAND_UNITS[0]): 0.8,
    (ORACLE, HAND_UNITS[0]): 1.0,
    (EMPIRICAL, HAND_UNITS[1]): 2.5,
    (ORACLE, HAND_UNITS[1]): 3.0,
}
HAND_SCALES = {
    (EMPIRICAL, "A"): {"fixed": 0.5, "one-way": 0.75, "two-way": 1.25},
    (EMPIRICAL, "B"): {"fixed": 1.0, "one-way": 1.5, "two-way": 2.0},
    (ORACLE, "A"): dict.fromkeys(RUNGS, 1.0),
    (ORACLE, "B"): dict.fromkeys(RUNGS, 1.0),
}


def hand_cars() -> dict:
    units = {}
    for unit, (site, fleet) in HAND_FLEETS.items():
        units[unit] = [
            FakeCar(site, date_of(k), 8.0, 10.5 + d, rate=r, room=e, session=f"{site}-{k}-{j}")
            for k in range(2)
            for j, (d, r, e) in enumerate(fleet)
        ]
    return units


def hand_matrix() -> dict:
    document = matrix_document(ROUNDED)
    document["gate_units"] = list(HAND_UNITS)
    document["required_units"] = 1
    document["n_calibration"] = 1
    return document


def _hand_entry(name: str, unit: str) -> dict:
    certified = HAND_CERTIFIED[(name, unit)]
    ceiling, best, _ = HAND_TRUE[unit]
    rungs = {}
    for rung in RUNGS:
        scale_a, scale_b = HAND_SCALES[(name, "A")][rung], HAND_SCALES[(name, "B")][rung]
        rungs[rung] = {
            "beta_hat": scale_a,
            "test": {"mean_offer_kwh_per_day": scale_a * certified},
            "tier_b": {
                "beta_hat": scale_b,
                "test": {"mean_offer_kwh_per_day": scale_b * certified},
            },
        }
    return {"rungs": rungs, "ceiling_sums_test_days_kwh": {CEILINGS[0]: ceiling, CEILINGS[1]: best}}


def _hand_rows(name: str, unit: str, plan) -> list[dict]:
    certified = HAND_CERTIFIED[(name, unit)]
    ceiling, best, _ = HAND_TRUE[unit]
    rows = []
    for truth, role in zip(plan.days.days, plan.roles, strict=True):
        row = dict.fromkeys(gate_run.DAY_COLUMNS, "")
        row |= {
            "unit": unit,
            "configuration": name,
            "site": truth.record.site,
            "date": str(truth.record.date),
            "role": role,
            "cars": "3",
            "steered_cars": "3",
            "unsteered_no_reference": "0",
            "unsteered_empty_set": "0",
            "certificate_outcome": "optimal",
            "certified_value_kwh": repr(certified),
            "ceiling_hindsight_kwh": repr(ceiling),
            "ceiling_best_causal_kwh": repr(best),
        }
        if role == "test":
            for rung in RUNGS:
                row[f"{rung}_offer_kwh"] = repr(HAND_SCALES[(name, "A")][rung] * certified)
                row[f"{rung}_tier_b_offer_kwh"] = repr(HAND_SCALES[(name, "B")][rung] * certified)
                for loss in ("slot_loss", "driver_loss"):
                    row[f"{rung}_{loss}"] = row[f"{rung}_tier_b_{loss}"] = "0.0"
        rows.append(row)
    return rows


def write_hand_run(folder: Path) -> dict:
    """The hand-worked build, matrix and recorded gate run (module text): paths by name."""
    truth, tables = write_build(folder, hand_cars())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(hand_matrix()))
    loaded = load(matrix)
    summary = json.loads((truth / "truth-summary.json").read_text())
    hashes = {
        f"{unit}-{kind}.csv": summary["units"][unit]["sha256"][kind]
        for unit in HAND_UNITS
        for kind in ("cars", "days")
    }
    plans = {}
    for unit in HAND_UNITS:
        days = load_unit_days(
            tables, unit, rho=loaded.rho, no_fit=loaded.no_fit, small_fleet=loaded.small_fleet
        )
        plans[unit] = gate_run.plan_unit(unit, days, loaded)
        assert plans[unit].roles.count("test") == 1, plans[unit].roles
    configurations, rows = {}, []
    for name in ROUNDED:
        configurations[name] = {"units": {}}
        for unit in HAND_UNITS:
            configurations[name]["units"][unit] = _hand_entry(name, unit)
            rows += _hand_rows(name, unit, plans[unit])
    document = {
        "trial": False,
        "partial": False,
        "label": "gate run",
        "inputs": {"setting": "main", "truth_run": truth.name, "table_sha256": hashes},
        "matrix": {"file": matrix.name, "sha256": hashlib.sha256(matrix.read_bytes()).hexdigest()},
        "configurations": configurations,
    }
    source = folder / "hand-source"
    source.mkdir()
    (source / gate_run.RESULTS_NAME).write_text(json.dumps(document))
    (source / gate_run.DAYS_NAME).write_text(gate_run._csv(rows, gate_run.DAY_COLUMNS, None))
    run = record(folder, source, GATE_MODULE, "hand-gate")
    return {"truth": truth, "tables": tables, "matrix": matrix, "run": run}


def copy_document(run: Path, name: str) -> dict:
    return copy.deepcopy(json.loads((run / name).read_text()))
