"""`models.hazard` and `models.hazard_tuned` give the forecast tables they gave before the hook the
seed ensemble (`models.hazard_ens`) reads (`HazardModel.set_curves`) was added.

`PINNED` holds the SHA-256 of each table as `forecast_model.write_table` writes it (every number
written so that reading gives the same float back), computed by `table_hashes` at the commit
before the hook (3d838bf, whose `models/hazard.py` the commit of the ablation models leaves
unchanged), on this project's machine and environment. Four tables on made-up cars
(`hazard_fake.known_data(60, 10)`): `models.hazard` at small settings, the same with the day
factor off, the same with a presence floor of 0.6 (so more `survival_h<t>` cells are NaN), and
`models.hazard_tuned` with a made-up settings file. A hash computed on another machine or with
other package builds may differ for reasons that are not a change of the code.
"""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import hazard_fake as hf

from fleetreserve.forecast_model import SURVIVAL_COLUMNS, write_table
from fleetreserve.models import hazard_tuned
from fleetreserve.models.hazard import HazardModel

MATRIX = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in hf.KAPPAS])
SMALL = replace(hf.FAST, epochs=4)
SETTINGS = {
    "hazard": SMALL,
    "hazard, no day factor": replace(SMALL, day_factor=False),
    "hazard, presence floor 0.6": replace(SMALL, min_presence_probability=0.6),
}
PINNED = {
    "hazard": ("f4d41d6bd30836e8bff1f88e298b2e222d2a37fe45e1d590b2ccf1a414b7cb5d", 400),
    "hazard, no day factor": (
        "3de2cd9475f2e581a01ed0a683221828eef108ba121711a7a060fe9505248ffd",
        400,
    ),
    "hazard, presence floor 0.6": (
        "60cd0020cc09746f197feca7d356e2f78399efb0876dd46f964bfc93d0797f8d",
        1800,
    ),
    "hazard_tuned": ("108eb36964b7091296906bed06934e3a04f0d49eb7cd730278a16ba010fed54a", 400),
}


def _digest(table, path) -> tuple[str, int]:
    """The table's SHA-256 as written, and its number of NaN cells in the survival columns."""
    write_table(table, path)
    nan_cells = int(table[list(SURVIVAL_COLUMNS)].isna().to_numpy().sum())
    return hashlib.sha256(path.read_bytes()).hexdigest(), nan_cells


def table_hashes(monkeypatch, folder) -> dict[str, tuple[str, int]]:
    train, fresh, _ = hf.known_data(60, 10)
    out = {}
    for name, settings in SETTINGS.items():
        model = HazardModel(rho=0.0, kappas=hf.KAPPAS, settings=settings)
        model.fit(train)
        out[name] = _digest(model.predict(fresh, hf.KAPPAS), folder / "table.csv")
    hf.use_small_tuned(monkeypatch, folder)
    model = hazard_tuned.make_model(MATRIX)
    model.fit(train)
    out["hazard_tuned"] = _digest(model.predict(fresh, hf.KAPPAS), folder / "table.csv")
    return out


def test_hazard_and_hazard_tuned_give_the_tables_of_the_commit_before_the_hook(
    monkeypatch, tmp_path
):
    found = table_hashes(monkeypatch, tmp_path)
    assert found == PINNED
    assert found["hazard, presence floor 0.6"][1] > found["hazard"][1]
