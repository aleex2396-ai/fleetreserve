"""The fleet reaches a car's forecast through the set encoder and the day factor, and only through
them (`models.hazard`, switches of `hazard_settings.HazardSettings`), on the day types of
`hazard_fake.day_type_cars`: on type E days every car leaves 3 to 4 h after τ, on type L days 7 to
8 h, and the two types differ only in how many cars plugged in early (six or two of eight).

A car that plugged in at 9:30 has the same inputs on both kinds of day. Its forecast on a type E
day and on a type L day (two frames that differ in the other cars only):

1. With the set encoder and the day factor on, the two survival curves differ: the bound made at
   τ differs by more than `MIN_GAP_H` hours, and the late day's is the later.
2. With the day factor on and the set encoder off (the day factor reads the mean of the cars'
   own encodings), they differ in the same way; with the set encoder on and the day factor off
   (the cars attend to each other) they differ in the same way.
3. With both off, the curves are identical, value for value.
4. The ablation switches do what they say: no day factor means no latent-variable networks, no
   set encoder no attention blocks, no site embedding no embedding table; the three are
   independent, and each model fits and forecasts.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import survival_fake as sf
from hazard_fake import FAST, day_type_cars, fresh_frame, train_cars

from fleetreserve.hazard_data import site_codes
from fleetreserve.models import hazard

KAPPAS = (0.1, 0.3)
MIN_GAP_H = 2.0
DAYS = 300
SETTINGS = dataclasses.replace(FAST, epochs=50)


@pytest.fixture(scope="module")
def data():
    fakes, users = day_type_cars(DAYS + 40, seed=8)
    cars = sf.cars_table(fakes, users)
    return train_cars(cars, DAYS), fresh_frame(cars, DAYS)


def _fitted(train, **switches):
    model = hazard.HazardModel(
        rho=0.0, kappas=KAPPAS, settings=dataclasses.replace(SETTINGS, **switches)
    )
    model.fit(train)
    return model


@pytest.fixture(scope="module")
def models(data):
    train, _ = data
    return {
        "both": _fitted(train),
        "day factor only": _fitted(train, set_encoder=False),
        "set encoder only": _fitted(train, day_factor=False),
        "neither": _fitted(train, set_encoder=False, day_factor=False),
    }


def _day(kind: str, template: pd.DataFrame) -> pd.DataFrame:
    """One day of eight cars, six (type E) or two (type L) of them early arrivals, the others
    arriving at 9:30. Every car is a copy of one early or one late arrival of the template's first
    day, so the 9:30 cars have the same inputs on both kinds of day."""
    first = template[template["date"] == template["date"].iloc[0]].reset_index(drop=True)
    early = first[first["arrival_hour"] == 6]
    late = first[first["arrival_hour"] == 9]
    n_early = 6 if kind == "E" else 2
    cars = pd.concat([early.iloc[[0] * n_early], late.iloc[[0] * (8 - n_early)]])
    cars = cars.reset_index(drop=True)
    cars["session_id"] = [f"{kind}-{k}" for k in range(8)]
    return cars


def _focal(frame: pd.DataFrame) -> int:
    """The place in the frame of the first car that plugged in at 9:30."""
    return int(np.flatnonzero(frame["arrival_hour"].to_numpy() == 9)[0])


def _focal_curve(model, frame: pd.DataFrame):
    x, sets = model.inputs(frame)
    assert sets.n_sets == 1
    survival, _ = model.curves(x[sets.rows], int(site_codes(sets, model.fit_sites)[0]))
    where = int(np.flatnonzero(sets.rows == _focal(frame))[0])
    return survival[where], x[_focal(frame)]


def _focal_bound(model, frame: pd.DataFrame) -> float:
    table = model.predict(frame, KAPPAS)
    return float(table[table["kappa"] == 0.3]["d_hat"].iloc[_focal(frame)])


def test_the_two_day_types_differ_in_the_other_cars_only(data):
    _, fresh = data
    early_day, late_day = _day("E", fresh), _day("L", fresh)
    assert (early_day["arrival_hour"] == 6).sum() == 6
    assert (late_day["arrival_hour"] == 6).sum() == 2
    focal = [d.iloc[_focal(d)] for d in (early_day, late_day)]
    same = [c for c in fresh.columns if c != "session_id"]
    assert focal[0][same].equals(focal[1][same])


@pytest.mark.parametrize("name", ["both", "day factor only", "set encoder only"])
def test_the_fleet_changes_a_cars_survival_curve_when_either_mechanism_is_on(data, models, name):
    _, fresh = data
    model = models[name]
    early_day, late_day = _day("E", fresh), _day("L", fresh)
    (curve_e, x_e), (curve_l, x_l) = (_focal_curve(model, d) for d in (early_day, late_day))
    assert np.array_equal(x_e, x_l)  # the car itself is the same on both days
    assert not np.array_equal(curve_e, curve_l)
    gap = _focal_bound(model, late_day) - _focal_bound(model, early_day)
    assert gap > MIN_GAP_H, (name, gap)


def test_with_the_set_encoder_and_the_day_factor_off_the_fleet_changes_nothing(data, models):
    _, fresh = data
    model = models["neither"]
    early_day, late_day = _day("E", fresh), _day("L", fresh)
    (curve_e, _), (curve_l, _) = (_focal_curve(model, d) for d in (early_day, late_day))
    assert np.array_equal(curve_e, curve_l)
    assert _focal_bound(model, early_day) == _focal_bound(model, late_day)


def _parts(model) -> set[str]:
    return {name.split(".")[0] for name, _ in model.net.named_parameters()}


def test_each_switch_removes_its_part_of_the_network_and_the_three_are_independent(data):
    train, fresh = data
    one_epoch = dataclasses.replace(FAST, epochs=1)
    full = hazard.HazardModel(rho=0.0, kappas=KAPPAS, settings=one_epoch)
    full.fit(train)
    assert {"blocks", "pool", "site", "prior", "posterior", "label_token"} <= _parts(full)
    for switch, gone in (
        ("set_encoder", {"blocks", "pool"}),
        ("site_embedding", {"site"}),
        ("day_factor", {"prior", "posterior", "label_token"}),
    ):
        model = hazard.HazardModel(
            rho=0.0, kappas=KAPPAS, settings=dataclasses.replace(one_epoch, **{switch: False})
        )
        model.fit(train)
        assert not gone & _parts(model), switch
        assert (_parts(full) - gone) <= _parts(model), switch
        assert model.predict(fresh.iloc[:16], KAPPAS)["d_hat"].notna().all()
