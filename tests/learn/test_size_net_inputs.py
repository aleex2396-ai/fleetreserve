"""The inputs of the day-size net (`size_net_inputs`) on made-up cars.

1. Cars equal up to τ give equal inputs and equal outputs, through the real path (`truth.build` on
   made-up sessions of one site-day of OSL_S, the cars table as CSV text, `apply_car_rules`,
   `car_frame`): the same four cars up to τ (plug-in, user, station, source), after τ different
   departures and session energies, one of them leaving within the decision hour in the second
   version, so that the rules leave it out of the day's fleet there. Every input column is equal,
   the day's set holds the same four cars, and a network gives the same output to the bit. The
   same day with a fifth car plugged in at τ gives another output, so the comparison can fail.
2. The columns are those of the departure model with the plan inputs
   (`models.hazard_tuned_inputs`): `hazard_data.INPUT_COLUMNS`, then its `EXTRA_INPUTS`; no
   look-ahead column, no label and no column of the cars table that is not known at τ.
3. Set k is the k-th labelled day, its cars in the order of their session ids whatever the order of
   the frame's rows; the encoder fitted on the same rows in another order gives the same bytes; a
   labelled day without a car is refused.
4. The context files: without them `load_context` could not run (a `MatrixError`); `record` names
   the columns, the 9/80 parity and the files' hashes.
"""

import io
from datetime import date

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import size_net_fake as fake
import torch

from fleetreserve import learn_features, model_inputs, size_net, size_net_inputs, truth
from fleetreserve.days_from_truth import LEFT_OUT_DEPART_ZERO, apply_car_rules
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_data import INPUT_COLUMNS
from fleetreserve.models import hazard_tuned_inputs
from fleetreserve.windows import Window

DAY = "2020-01-14"
SITE = "OSL_S"
ZONE = "Europe/Oslo"


def sessions(after: list[float], energy: list[float]) -> pd.DataFrame:
    """Cars plugged in on `DAY` at 14:00 + k half hours, leaving `after[k]` hours after τ (17:00)
    with session energy `energy[k]`."""
    rows = []
    for k, (hours, kwh) in enumerate(zip(after, energy, strict=True)):
        plug_in = pd.Timestamp(f"{DAY} 14:00") + pd.Timedelta(minutes=30 * k)
        rows.append(
            {
                "site": SITE,
                "session_id": f"s{k}",
                "user_id": f"u{k}",
                "station": str(k),
                "plug_in": plug_in.tz_localize(ZONE),
                "plug_out": (pd.Timestamp(f"{DAY} 17:00") + pd.Timedelta(hours=hours)).tz_localize(
                    ZONE
                ),
                "energy_kwh": kwh,
                "nominal_kw": 7.4,
                "nominal_rule": "source cap",
                "charge_start": None,
                "charge_end": None,
                "declared_departure": None,
            }
        )
    frame = pd.DataFrame(rows)
    for c in ("charge_start", "charge_end", "declared_departure"):
        frame[c] = pd.to_datetime(frame[c], utc=True).dt.tz_convert(ZONE)
    return frame


def ruled(table: pd.DataFrame) -> pd.DataFrame:
    cars, _ = truth.build(
        table,
        Window(17, 3, "fake"),
        ZONE,
        unit=SITE,
        days=[date.fromisoformat(DAY)],
        limit_kw=50.0,
        bound_by_charge_fields=False,
    )
    text = io.StringIO()
    cars.to_csv(text, index=False)
    text.seek(0)
    return apply_car_rules(pd.read_csv(text, dtype=str, keep_default_na=False), 0.0, "cap")


DAY_ROW = pd.DataFrame({"unit": [SITE], "site": [SITE], "date": [DAY]})


def output(frame: pd.DataFrame, encoder) -> np.ndarray:
    torch.set_num_threads(1)
    sets = size_net_inputs.fleets(frame, DAY_ROW)
    net = fake.drawn_net(encoder.n_inputs, 5)
    return size_net.predict(net, encoder.transform(frame), sets, size_net.NetSettings())


def test_cars_equal_up_to_tau_give_equal_inputs_and_outputs():
    sources = fake.context().sources
    first = ruled(sessions([2.5, 3.0, 4.2, 6.0], [10.0, 11.0, 12.0, 13.0]))
    second = ruled(sessions([0.5, 5.0, 2.2, 9.0], [3.0, 20.0, 7.5, 30.0]))
    assert (second["outcome"] == LEFT_OUT_DEPART_ZERO).sum() == 1
    assert (first["outcome"] == LEFT_OUT_DEPART_ZERO).sum() == 0
    a = size_net_inputs.car_frame(first, rho=0.0, sources=sources)
    b = size_net_inputs.car_frame(second, rho=0.0, sources=sources)
    assert len(a) == len(b) == 4
    pd.testing.assert_frame_equal(a, b)
    encoder = size_net_inputs.Encoder.fit(a)
    np.testing.assert_array_equal(output(a, encoder), output(b, encoder))
    more = ruled(sessions([2.5, 3.0, 4.2, 6.0, 3.0], [10.0, 11.0, 12.0, 13.0, 9.0]))
    c = size_net_inputs.car_frame(more, rho=0.0, sources=sources)
    assert not np.array_equal(output(c, encoder), output(a, encoder))


def test_the_columns_are_the_departure_models():
    assert (*INPUT_COLUMNS, *hazard_tuned_inputs.EXTRA_INPUTS) == size_net_inputs.COLUMNS
    assert not set(size_net_inputs.COLUMNS) & set(model_inputs.LOOK_AHEAD_COLUMNS)
    assert not set(size_net_inputs.COLUMNS) & set(learn_features.LABEL_COLUMNS)
    assert not set(size_net_inputs.COLUMNS) & set(learn_features.NOT_KNOWN_AT_TAU)
    frame = size_net_inputs.car_frame(
        ruled(sessions([2.5, 3.0, 4.2], [10.0, 11.0, 12.0])),
        rho=0.0,
        sources=fake.context().sources,
    )
    assert tuple(frame.columns) == (*KEY_COLUMNS, *size_net_inputs.COLUMNS)


def test_a_days_set_is_its_cars_in_session_order():
    cars, rows = fake.planted(np.random.default_rng(5), ("caltech", "jpl"), 6)
    sets = size_net_inputs.fleets(cars, rows)
    assert sets.n_sets == len(rows)
    for k, row in rows.iterrows():
        members = cars.iloc[sets.members(k)]
        assert (members[["unit", "site", "date"]] == row[["unit", "site", "date"]]).all().all()
        assert list(members["session_id"]) == sorted(members["session_id"])
        expected = (cars[["unit", "site", "date"]] == row[["unit", "site", "date"]]).all(axis=1)
        assert len(members) == int(expected.sum())
    shuffled = cars.sample(frac=1.0, random_state=3).reset_index(drop=True)
    again = size_net_inputs.fleets(shuffled, rows)
    for k in range(len(rows)):
        pd.testing.assert_frame_equal(
            shuffled.iloc[again.members(k)].reset_index(drop=True),
            cars.iloc[sets.members(k)].reset_index(drop=True),
        )
    one = size_net_inputs.Encoder.fit(cars.iloc[sets.rows])
    other = size_net_inputs.Encoder.fit(shuffled.iloc[again.rows])
    np.testing.assert_array_equal(one.transform(cars), other.transform(cars))
    missing = pd.concat(
        [rows, pd.DataFrame([{"unit": "jpl", "site": "jpl", "date": "2021-01-01"}])]
    )
    with pytest.raises(ValueError, match="no car"):
        size_net_inputs.fleets(cars, missing.reset_index(drop=True))


def test_without_the_context_files_it_could_not_run(tmp_path):
    with pytest.raises(MatrixError):
        size_net_inputs.load_context(root=tmp_path)
    found = size_net_inputs.record(fake.context())
    assert found["columns"] == list(size_net_inputs.COLUMNS)
    assert found["nine_eighty_parity"] == 0
    assert found["files_sha256"] == {"made/up.csv": "0" * 64}
