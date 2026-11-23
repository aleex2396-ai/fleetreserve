"""Made-up development tables and a faked verified truth build for the tests of `ceiling_report`.

Times are in hours from midnight of a car's date, τ at 10:00, six service hours (see
`truth_tables_fake`). A day is one of five kinds, by its number:

0. three cars leaving after the window (hours 8 to 9 after τ) with room the recovery hours can
   still supply: T4's hypotheses hold;
1. three cars leaving inside the window (hours 2 to 4 after τ);
2. three cars leaving in hour 7, room 4 kWh at 2 kW, no recovery hour: no in-window departure but
   the hypotheses fail;
3. a mix of one car leaving inside the window and two leaving after it;
4. three cars unplugged within the decision hour: they are left out, the fleet is empty.
"""

import hashlib
import json

from truth_tables_fake import FakeCar, date_of, write

from fleetreserve import ceiling_report, run_verify

COMMIT = "a" * 40
DAYS_PER_UNIT = {
    "nrel-before-may-2019": (("A", 20),),
    "nrel-from-may-2019": (("A", 25),),
    "norway-development-pooled": (("A", 15), ("B", 15)),
}


def day_cars(site: str, k: int) -> list[FakeCar]:
    date = date_of(k)
    kind = k % 5
    cars = []
    for j in range(3):
        session = f"{site}-{k}-{j}"
        if kind == 0:
            cars.append(
                FakeCar(
                    site, date, 8.0, 18.5 + 0.3 * j, rate=3.0, room=2.0 + 0.4 * j, session=session
                )
            )
        elif kind == 1:
            cars.append(
                FakeCar(
                    site,
                    date,
                    8.0,
                    12.25 + j,
                    rate=2.0 + 0.5 * j,
                    room=3.0 + 0.5 * j,
                    session=session,
                )
            )
        elif kind == 2:
            cars.append(
                FakeCar(site, date, 8.0, 17.2 + 0.1 * j, rate=2.0, room=4.0, session=session)
            )
        elif kind == 3:
            out = (12.5, 17.5, 18.5)[j]
            cars.append(FakeCar(site, date, 8.0, out, rate=2.5, room=3.0 + j, session=session))
        else:
            cars.append(FakeCar(site, date, 8.0, 10.5, rate=2.0, room=1.0, session=session))
    return cars


def unit_cars(unit: str, scale: float = 1.0, layout=None) -> list[FakeCar]:
    cars = []
    for site, days in (DAYS_PER_UNIT if layout is None else layout)[unit]:
        cars += [car for k in range(round(days * scale)) for car in day_cars(site, k)]
    return cars


def setup_build(tmp_path, monkeypatch, scale: float = 1.0, layout=None):
    """Tables and a verified truth build at `COMMIT` of the three units, or of the units of
    `layout` (unit -> its sites and their numbers of days, as `DAYS_PER_UNIT`). Returns (run,
    tables)."""
    tables = tmp_path / "tables"
    tables.mkdir()
    units = {}
    for unit in ceiling_report.UNITS if layout is None else layout:
        write(tables, unit, unit_cars(unit, scale, layout))
        units[unit] = {
            "sha256": {
                kind: hashlib.sha256((tables / f"{unit}-{kind}.csv").read_bytes()).hexdigest()
                for kind in ("cars", "days")
            }
        }
    run = tmp_path / "20260101T000000Z-wp2-truth"
    run.mkdir()
    summary = run / "truth-summary.json"
    summary.write_text(json.dumps({"units": units}))
    manifest = {
        "commit": COMMIT,
        "command": ["build"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {"truth-summary.json": hashlib.sha256(summary.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(ceiling_report, "_head_commit", lambda: COMMIT)
    return run, tables
