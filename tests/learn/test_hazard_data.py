"""Tables to tensors for the neural hazard model (`hazard_data`), on the made-up cars of
`hazard_fake`.

1. The inputs are the columns of `FEATURE_COLUMNS`, `TAIL_HISTORY_COLUMNS` and `CONTEXT_COLUMNS`
   and nothing else: no declared departure, key, label or outcome column.
2. The scaler uses the fit rows' statistics (a frame it is applied to does not change them);
   a missing history value is 0 plus a flag; a missing value elsewhere is refused; cyclic columns
   are a sine and a cosine of their phase; counts are put through `log1p`; an extreme value is
   clipped.
3. A car sits in the same set when fitting and when forecasting: the sets of a `build_table`
   table and of its `feature_frame` are the same, whatever the order of the rows, and a set is
   the rows of one unit, site and date, ordered by session.
4. A site seen in fitting has its own code, any other site the unknown code, whatever its name.
5. Batches pad to the longest set with a mask; the fit data carry the labels of the right rows.
6. Every missing-value flag is set for a car with no history and cleared by a value, one column at
   a time.
7. Departure labels are used on every row that has one, energy labels on fleet rows only: in the
   fit data, in the batches, and in the loss (an energy target poisoned on non-fleet rows does not
   reach the energy head). The rows the energy head is fitted on are the rows `models.lgbm` fits
   each quantity on, when every row with a room has a rate; a row with a room and no rate is left
   out of the energy head only.
"""

import dataclasses
import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import torch
from hazard_fake import FAST, known_data

from fleetreserve import hazard_curves, hazard_fit
from fleetreserve import hazard_data as hd
from fleetreserve.hazard_data import Scaler, batch_of, fit_data, make_sets, site_codes
from fleetreserve.learn_features import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    TAIL_HISTORY_COLUMNS,
    build_table,
    feature_frame,
    with_context,
)
from fleetreserve.models.lgbm import rows_for


@pytest.fixture(scope="module")
def tables():
    train, fresh, later = known_data()
    fit_table = with_context(build_table(train, rho=0.0))
    return fit_table, with_context(fresh), with_context(later)


# --- 1. the input columns ------------------------------------------------------------------------


def test_the_inputs_are_the_feature_tail_and_context_columns_only():
    expected = (*FEATURE_COLUMNS, *TAIL_HISTORY_COLUMNS, *CONTEXT_COLUMNS)
    assert expected == hd.INPUT_COLUMNS
    banned = {"declared_departure_h", "unit", "site", "date", "session_id", "counting"}
    assert not banned & set(hd.INPUT_COLUMNS)
    assert not set(LABEL_COLUMNS) & set(hd.INPUT_COLUMNS)
    width = len(hd.PLAIN_COLUMNS) + len(hd.MISSING_COLUMNS) + 2 * len(hd.CYCLIC_PERIODS)
    assert width == hd.N_INPUTS
    assert set(hd.CYCLIC_PERIODS) <= set(hd.INPUT_COLUMNS)
    assert set(hd.MISSING_COLUMNS) <= set(hd.PLAIN_COLUMNS)
    assert "history_sessions" not in hd.MISSING_COLUMNS


def test_the_encoding_reads_the_input_columns_only(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    base = scaler.transform(fresh)
    others = fresh.copy()
    others["declared_departure_h"] = 123.0
    others["fleet_extra"] = 5.0
    for column in LABEL_COLUMNS:
        others[column] = 9.0
    assert np.array_equal(base, scaler.transform(others))
    assert base.shape == (len(fresh), hd.N_INPUTS) and base.dtype == np.float32


# --- 2. the scaler -------------------------------------------------------------------------------


def test_the_scaler_standardises_with_the_fit_rows_statistics_not_those_of_the_frame(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    place = hd.PLAIN_COLUMNS.index("hours_since_plug_in")
    own = scaler.transform(fit_table)[:, place]
    assert abs(own.mean()) < 1e-3 and abs(own.std() - 1.0) < 1e-3
    shifted = fresh.assign(
        hours_since_plug_in=fresh["hours_since_plug_in"] + 3.0 * scaler.std[place]
    )
    moved = scaler.transform(shifted)[:, place] - scaler.transform(fresh)[:, place]
    assert np.allclose(moved, 3.0, atol=1e-4)  # three fit standard deviations, not renormalised


def test_a_missing_history_value_is_zero_with_a_flag_and_elsewhere_it_is_refused(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    assert fresh["history_min_departure_h"].isna().all()  # every car of this data is new
    x = scaler.transform(fresh)
    mean_place = hd.PLAIN_COLUMNS.index("history_min_departure_h")
    flag_place = len(hd.PLAIN_COLUMNS) + hd.MISSING_COLUMNS.index("history_min_departure_h")
    assert (x[:, mean_place] == 0).all() and (x[:, flag_place] == 1).all()
    filled = fresh.assign(history_min_departure_h=2.0)
    y = scaler.transform(filled)
    assert (y[:, flag_place] == 0).all() and (y[:, mean_place] != 0).all()
    with pytest.raises(ValueError, match="missing in the columns.*hours_since_plug_in"):
        scaler.transform(fresh.assign(hours_since_plug_in=math.nan))
    with pytest.raises(ValueError, match="infinite"):
        scaler.transform(fresh.assign(hours_since_plug_in=math.inf))


def test_cyclic_columns_are_sine_and_cosine_of_their_phase_and_counts_are_log_scaled(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    frame = fresh.iloc[:3].assign(
        weekday=[0, 3, 7], month=[1, 4, 13], arrival_hour=[0, 6, 24], day_of_year=[1, 92, 367]
    )
    x = scaler.transform(frame)
    first = len(hd.PLAIN_COLUMNS) + len(hd.MISSING_COLUMNS)
    names = list(hd.CYCLIC_PERIODS)
    for k in range(len(names)):
        sine, cosine = x[:, first + 2 * k], x[:, first + 2 * k + 1]
        assert np.allclose(sine**2 + cosine**2, 1.0, atol=1e-6)
        assert sine[0] == pytest.approx(sine[2], abs=1e-6)  # one full period apart
        assert cosine[0] == pytest.approx(cosine[2], abs=1e-6)
    weekday = names.index("weekday")
    phase = 2 * math.pi * 3 / 7
    assert x[1, first + 2 * weekday] == pytest.approx(math.sin(phase), abs=1e-6)
    assert x[1, first + 2 * weekday + 1] == pytest.approx(math.cos(phase), abs=1e-6)


def test_counts_are_put_through_log1p_before_standardising(tables):
    fit_table, fresh, _ = tables
    counts = np.arange(len(fit_table)) % 7
    scaler = Scaler.fit(fit_table.assign(history_sessions=counts), FAST)
    frame = fresh.iloc[:3].assign(history_sessions=[0, 3, 10])
    place = hd.PLAIN_COLUMNS.index("history_sessions")
    logged = np.log1p(counts.astype(float))
    expected = (np.log1p([0.0, 3.0, 10.0]) - logged.mean()) / logged.std()
    assert np.allclose(scaler.transform(frame)[:, place], expected, atol=1e-5)


def test_an_extreme_value_is_clipped_to_the_setting(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, dataclasses.replace(FAST, input_clip=4.0))
    place = hd.PLAIN_COLUMNS.index("hours_since_plug_in")
    big = scaler.transform(fresh.assign(hours_since_plug_in=1e6))[:, place]
    assert (big == 4.0).all()


# --- 3. the sets ---------------------------------------------------------------------------------


def test_a_car_sits_in_the_same_set_when_fitting_and_when_forecasting(tables):
    _, _, later = tables
    forecast_side = feature_frame(later)
    assert not set(LABEL_COLUMNS) & set(forecast_side.columns)
    fitting, forecasting = make_sets(later), make_sets(forecast_side)
    assert np.array_equal(fitting.rows, forecasting.rows)
    assert np.array_equal(fitting.starts, forecasting.starts)
    assert fitting.keys == forecasting.keys


def test_a_set_is_the_cars_of_one_unit_site_and_date_ordered_by_session(tables):
    _, fresh, _ = tables
    sets = make_sets(fresh)
    assert sets.starts[0] == 0 and sets.starts[-1] == len(fresh)
    assert sorted(sets.rows.tolist()) == list(range(len(fresh)))
    assert (
        len(set(sets.keys))
        == sets.n_sets
        == fresh[["unit", "site", "date"]].drop_duplicates().shape[0]
    )
    for k in range(sets.n_sets):
        rows = sets.members(k)
        found = {tuple(fresh.iloc[r][["unit", "site", "date"]].astype(str)) for r in rows}
        assert found == {sets.keys[k]}
        sessions = fresh["session_id"].iloc[rows].astype(str).tolist()
        assert sessions == sorted(sessions)


def test_the_sets_do_not_depend_on_the_order_of_the_rows(tables):
    _, fresh, _ = tables
    shuffled = fresh.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a, b = make_sets(fresh), make_sets(shuffled)
    assert a.keys == b.keys and np.array_equal(a.starts, b.starts)
    for k in range(a.n_sets):
        same = fresh["session_id"].iloc[a.members(k)].tolist()
        assert same == shuffled["session_id"].iloc[b.members(k)].tolist()


def test_two_units_at_one_site_and_date_are_two_sets(tables):
    _, fresh, _ = tables
    day = fresh[fresh["date"] == fresh["date"].iloc[0]].reset_index(drop=True)
    both = pd.concat([day, day.assign(unit="other", session_id=day["session_id"] + "x")])
    assert make_sets(both).n_sets == 2 * make_sets(day).n_sets
    assert make_sets(fresh.iloc[0:0]).n_sets == 0


# --- 4. the site codes ---------------------------------------------------------------------------


def test_a_seen_site_has_its_own_code_and_any_other_the_unknown_code(tables):
    _, fresh, _ = tables
    sets = make_sets(fresh)
    codes = site_codes(sets, ("OSL_S", "caltech"))
    assert {key[1]: code for key, code in zip(sets.keys, codes, strict=True)} == {
        "caltech": 2,
        "OSL_S": 1,
    }
    assert set(site_codes(sets, ("jpl",))) == {hd.UNKNOWN_SITE} == {0}
    renamed = make_sets(fresh.assign(site="somewhere"))
    assert set(site_codes(renamed, ("jpl",))) == {0}


# --- 5. batches and fit data ---------------------------------------------------------------------


def test_batches_pad_to_the_longest_set_with_a_mask_and_carry_the_labels(tables):
    fit_table, _, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    data = fit_data(fit_table, scaler, ("OSL_S", "caltech"), FAST)
    sets = data.sets
    chosen = [0, 5, 9]
    batch = batch_of(data.x, sets, data.site, chosen, data)
    longest = sets.sizes[chosen].max()
    assert batch.x.shape == (3, longest, hd.N_INPUTS)
    assert batch.mask.sum(axis=1).tolist() == sets.sizes[chosen].tolist()
    assert (batch.x[~batch.mask] == 0).all()
    for position, k in enumerate(chosen):
        rows = sets.members(k)
        assert np.array_equal(batch.x[position, : len(rows)], data.x[rows])
        assert np.array_equal(batch.departure_bin[position, : len(rows)], data.departure_bin[rows])
    assert batch.site.tolist() == data.site[chosen].tolist()
    assert data.has_departure.all() and data.has_energy.sum() > 0
    mean = (data.energy[data.has_energy]).mean(axis=0)
    assert np.allclose(mean, 0.0, atol=1e-4)


def test_fit_data_refuses_a_table_with_no_label(tables):
    fit_table, _, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    with pytest.raises(ValueError, match="departure label"):
        fit_data(fit_table.assign(label_departure_h=math.nan), scaler, (), FAST)
    with pytest.raises(ValueError, match="energy label"):
        fit_data(fit_table.assign(label_room_kwh=math.nan), scaler, (), FAST)


# --- 6. the missing-value flags ------------------------------------------------------------------


def test_every_missing_flag_is_set_for_a_car_with_no_history_and_cleared_by_a_value(tables):
    fit_table, fresh, _ = tables
    scaler = Scaler.fit(fit_table, FAST)
    first = len(hd.PLAIN_COLUMNS)
    count = len(hd.MISSING_COLUMNS)
    assert fresh[list(hd.MISSING_COLUMNS)].isna().all().all()  # no car of this data has a history
    flags = scaler.transform(fresh)[:, first : first + count]
    assert (flags == 1).all()
    filled = scaler.transform(fresh.assign(**dict.fromkeys(hd.MISSING_COLUMNS, 1.0)))
    assert (filled[:, first : first + count] == 0).all()
    for k, column in enumerate(hd.MISSING_COLUMNS):
        one = scaler.transform(fresh.assign(**{column: 1.0}))[:, first : first + count]
        assert one.sum(axis=1).tolist() == [count - 1] * len(fresh), column
        assert (one[:, k] == 0).all(), column


# --- 7. which rows each label is used on ---------------------------------------------------------


def _with_non_fleet_rows(fit_table):
    """The fit table with every third car outside its fleet (no room), departure labels kept."""
    outside = np.arange(len(fit_table)) % 3 == 0
    table = fit_table.copy()
    table.loc[outside, ["label_room_kwh", "label_need_kwh"]] = math.nan
    return table, outside


def test_departure_is_used_on_every_row_with_a_label_and_energy_on_fleet_rows_only(tables):
    fit_table, _, _ = tables
    table, outside = _with_non_fleet_rows(fit_table)
    data = fit_data(table, Scaler.fit(table, FAST), ("OSL_S", "caltech"), FAST)
    assert data.has_departure.all() and not data.has_energy[outside].any()
    kept = fit_table["label_room_kwh"].notna().to_numpy()  # the rules leave some cars out too
    assert (data.has_energy[~outside] == kept[~outside]).all() and outside.sum() > 100
    assert data.has_energy.sum() == (kept & ~outside).sum()
    assert (data.energy[outside] == 0).all()
    chosen = list(range(10))
    batch = batch_of(data.x, data.sets, data.site, chosen, data)
    assert batch.has_departure.sum() == data.sets.sizes[chosen].sum()
    assert batch.has_energy.sum() == data.has_energy[batch.rows[batch.mask]].sum()
    assert batch.has_energy.sum() < batch.has_departure.sum()


def test_the_energy_rows_are_the_rows_models_lgbm_fits_each_quantity_on_when_every_rate_is_a_number(
    tables,
):
    fit_table, _, _ = tables
    table, outside = _with_non_fleet_rows(fit_table)
    data = fit_data(table, Scaler.fit(table, FAST), ("OSL_S", "caltech"), FAST)
    assert table["label_rate_kw"].notna().all()  # a rate on every row, as in an accepted table
    for quantity in ("rate", "room", "need"):
        assert (data.has_energy == rows_for(table, quantity).to_numpy()).all(), quantity
    assert outside.sum() > 100 and data.has_energy.sum() < len(table)
    # a row with a room and no rate is a row the energy head is not fitted on, and the room
    # quantity of `models.lgbm` is
    broken = table.copy()
    row = int(np.flatnonzero(broken["label_room_kwh"].notna())[0])
    broken.loc[row, "label_rate_kw"] = math.nan
    data = fit_data(broken, Scaler.fit(broken, FAST), ("OSL_S", "caltech"), FAST)
    assert not data.has_energy[row] and rows_for(broken, "room")[row]
    assert not rows_for(broken, "rate")[row]


def test_an_energy_target_set_on_non_fleet_rows_does_not_reach_the_energy_head(tables):
    fit_table, _, _ = tables
    table, outside = _with_non_fleet_rows(fit_table)
    settings = dataclasses.replace(FAST, epochs=20, learning_rate=0.03)
    data = fit_data(table, Scaler.fit(table, settings), ("OSL_S", "caltech"), settings)
    poisoned = dataclasses.replace(
        data, energy=np.where(data.has_energy[:, None], data.energy, 1000.0).astype(np.float32)
    )
    levels, quantity = np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2])
    net = hazard_fit.fit(poisoned, settings, 2, levels, quantity).net
    batch = batch_of(poisoned.x, poisoned.sets, poisoned.site, list(range(40)), poisoned)
    with torch.no_grad():
        encoded, _ = net.encode(torch.as_tensor(batch.x), torch.as_tensor(batch.mask))
        out = net.energy_outputs(encoded).numpy()
    on_fleet = out[batch.has_energy]
    assert on_fleet.size > 0 and np.abs(on_fleet).max() < 10.0
    assert hazard_curves.n_hazard_bins(settings) == 72  # the grid of the net that was fitted


def test_the_departure_loss_counts_every_row_with_a_label_not_only_fleet_rows(tables, monkeypatch):
    fit_table, _, _ = tables
    table, outside = _with_non_fleet_rows(fit_table)
    settings = dataclasses.replace(FAST, epochs=1, day_factor=False, energy_weight=0.0)
    data = fit_data(table, Scaler.fit(table, settings), ("OSL_S", "caltech"), settings)
    marked = np.where(outside, 60, 10)
    data = dataclasses.replace(data, departure_bin=marked)

    def mean_bin(logits, departure_bin):
        # a stand-in likelihood whose loss is the mean departure bin of the rows it is weighted on
        return -departure_bin.to(logits.dtype) + 0.0 * logits.sum()

    monkeypatch.setattr(hazard_fit, "log_likelihood", mean_bin)
    levels, quantity = np.array([0.1]), np.array([0])
    loss = hazard_fit.fit(data, settings, 2, levels, quantity).history[0]
    assert loss == pytest.approx(marked.mean(), rel=1e-4)  # every row has a label
    assert abs(marked.mean() - 10.0) > 5.0  # fleet rows alone would give 10
