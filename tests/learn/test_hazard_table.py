"""`models.hazard` as a departure model: the table it writes and what the table depends on, on the
made-up cars of `hazard_fake` (two training sites, 300 days of ten cars, 40 later days).

1. Table: `forecast_model.check_table` accepts it; it has a row for every car of the frame at
   every level; `d_hat` does not decrease in the level; each `survival_h<t>` is at least t or NaN
   and does not decrease in t; `d_median` is at or above `d_hat` below level 0.5; no number is
   negative.
2. Conditional bound: the bound given presence after the early group has left lies in the late
   group, and equals `forecast_model.survival_quantile` applied to the model's own curve.
3. No outcome reaches a forecast: `predict` works on a `feature_frame`, raises on a missing input
   column, reads no label, outcome or declared-departure column; the forecasts of a site-day do
   not change when the other site-days are dropped, when the rows are shuffled or when the cars
   of a day are reordered. `fit_units` is the training units and `learn_run` accepts the model.
4. Unknown site: a frame at a site not seen in fitting is forecast, and renaming that site (to
   another of the same type and calendar) changes nothing; a site seen in fitting is not forecast
   as an unknown one.
5. Two fits in one process on the same rows give byte-identical tables.
6. The levels: a level the model was not fitted at is refused; an empty frame gives an empty
   table; a site missing from the committed mappings is refused.
7. Hours the data do not reach: `survival_h<t>` is NaN for every car of a set whose site type has
   fewer than `min_presence_support` fit rows with a departure at or above t (a model fitted on
   short stays, forecasting long ones; short stays at one site type and long ones at the other),
   counted over every fit row with a departure label, and for a car whose own probability of
   being there is below the floor (tested on survival curves made by hand, through
   `hazard_curves.departure_table` and through the model with a patched curve); where there is
   support the columns are the numbers they were; `d_hat` and `d_median` never change; the rows
   of a set must agree on the site type.
8. Prediction never uses the posterior network or a label: the table is the same when the
   posterior's forward pass raises and when its weights are overwritten after the fit.
9. Sites are hidden in fitting with the set probability: the unknown-site vector is trained
   when it is 1 and not when it is 0.
"""

import copy
import dataclasses
import hashlib
from types import SimpleNamespace

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import torch
from hazard_fake import (
    FAST,
    KAPPAS,
    fitted_known,
    known_data,
    outside_fleet_cars,
    outside_fleet_model,
    short_stay_model,
    true_bound,
    typed_support_cars,
    typed_support_model,
)

from fleetreserve import hazard_curves, hazard_fit, learn_run
from fleetreserve.forecast_model import (
    SURVIVAL_COLUMNS,
    check_table,
    survival_quantile,
    write_table,
)
from fleetreserve.hazard_data import make_sets, site_codes
from fleetreserve.hazard_net import HazardNet
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.learn_features import LABEL_COLUMNS, build_table, with_context
from fleetreserve.models import hazard
from fleetreserve.truth_build import TRAIN_UNITS

BOUND_TOLERANCE = 0.5
NUMBERS = ["d_hat", "r_hat", "e_cap", "e_need", "d_median", *SURVIVAL_COLUMNS]


def _sorted(table: pd.DataFrame) -> pd.DataFrame:
    return table.sort_values(["session_id", "kappa"]).reset_index(drop=True)


# --- 1. the table --------------------------------------------------------------------------------


def test_the_table_is_accepted_and_holds_every_car_at_every_level():
    _, fresh, _, table = fitted_known()
    checked = check_table(table)
    assert len(checked) == len(fresh) * len(KAPPAS)
    for kappa in KAPPAS:
        part = checked[checked["kappa"] == kappa]
        assert part["session_id"].tolist() == fresh["session_id"].astype(str).tolist()
    assert set(checked["rho"]) == {0.0}


def test_departure_bounds_rise_with_the_level_and_the_median_is_above_the_low_levels():
    _, _, _, table = fitted_known()
    by_level = {k: table[table["kappa"] == k].reset_index(drop=True) for k in KAPPAS}
    for low, high in zip(KAPPAS, KAPPAS[1:], strict=False):
        assert (by_level[high]["d_hat"] >= by_level[low]["d_hat"]).all()
        assert (by_level[high]["d_hat"] > by_level[low]["d_hat"]).any()
    for part in by_level.values():
        assert (part["d_median"] >= part["d_hat"]).all()


def test_the_bound_given_presence_is_at_least_the_hour_and_rises_with_the_hour():
    _, _, _, table = fitted_known()
    assert table["d_hat"].notna().all()
    for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
        found = table[column].dropna()
        # at hour 8 the cars of group A, which nobody follows that long, are NaN (section 7)
        assert len(found) > (0.9 if hour < 8 else 0.4) * len(table)
        assert (found >= hour).all(), column
    values = table[list(SURVIVAL_COLUMNS)].to_numpy()
    for row in values:
        seen = row[~np.isnan(row)]
        assert (np.diff(seen) >= -1e-12).all()


def test_no_number_is_negative_or_infinite():
    _, _, _, table = fitted_known()
    numbers = table[NUMBERS].to_numpy(float)
    assert not np.isinf(numbers).any()
    assert (numbers[~np.isnan(numbers)] >= 0).all()
    assert table[["d_hat", "r_hat", "e_cap", "e_need", "d_median"]].notna().all().all()


# --- 2. the conditional bound --------------------------------------------------------------------


def test_the_bound_given_presence_after_the_early_group_has_left_lies_in_the_late_group():
    _, fresh, _, table = fitted_known()
    group = np.where(fresh["arrival_hour"].to_numpy() == 7, "A", "B")
    late = {"A": (5.0, 8.0), "B": (6.0, 9.0)}
    for kappa in KAPPAS:
        part = table[table["kappa"] == kappa].reset_index(drop=True)
        for name in "AB":
            at_tau = part["d_hat"].to_numpy()[group == name]
            for hour in (4, 5):
                bound = part[f"survival_h{hour}"].to_numpy()[group == name]
                low, high = late[name]
                assert (bound >= low - 0.25).all() and (bound <= high).all(), (kappa, name, hour)
                assert abs(bound.mean() - true_bound(name, kappa, hour)) <= BOUND_TOLERANCE
            if name == "A":  # the bound made at τ lies in the early group
                assert at_tau.max() <= 4.0
                assert (part["survival_h4"].to_numpy()[group == name] > at_tau.max()).all()


def test_the_conditional_bound_is_survival_quantile_of_the_models_own_curve():
    model, fresh, _, table = fitted_known()
    x, sets = model.inputs(fresh)
    times = hazard_curves.bin_starts(model.settings)
    codes = site_codes(sets, model.fit_sites)  # the sites in this frame were seen in fitting
    assert codes.min() > 0
    for k in range(0, sets.n_sets, 7):
        rows = sets.members(k)
        survival, _ = model.curves(x[rows], int(codes[k]))
        for kappa in KAPPAS:
            part = table[table["kappa"] == kappa].reset_index(drop=True)
            for position, row in enumerate(rows):
                curve = survival[position]
                wanted = {
                    "d_hat": survival_quantile(times, curve, kappa),
                    "d_median": survival_quantile(times, curve, 0.5),
                }
                for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
                    wanted[column] = survival_quantile(
                        times, curve, kappa, after=hour - hazard_curves.PRESENCE_SHIFT
                    )
                    # a car with almost no chance of being there gets no bound (section 7)
                    here = hazard_curves.presence_probability(times, curve[None], hour)[0]
                    if here < model.settings.min_presence_probability:
                        wanted[column] = np.nan
                for column, value in wanted.items():
                    got = part.loc[row, column]
                    assert (np.isnan(got) and np.isnan(value)) or got == value, (kappa, column)


# --- 3. no outcome reaches a forecast ------------------------------------------------------------


def test_predict_raises_on_a_missing_input_column_and_the_model_names_the_training_units():
    model, fresh, _, _ = fitted_known()
    for column in ("history_min_departure_h", "weekday", "arrival_hour", "site", "date"):
        with pytest.raises(KeyError):
            model.predict(fresh.drop(columns=[column]), KAPPAS)
    assert model.fit_units == TRAIN_UNITS
    assert model.name == "hazard"
    assert learn_run.check_model("hazard", model) is False
    assert getattr(model, "reads_other_day_outcomes", False) is False


def test_no_label_outcome_or_declared_column_is_read():
    model, fresh, later, table = fitted_known()
    extra = fresh.copy()
    for column in LABEL_COLUMNS:
        extra[column] = -7.0 * np.arange(len(fresh)) + 3.0
    for column in ("plug_out", "energy_kwh", "departure_exact", "depart_hour", "outcome"):
        extra[column] = 99.0
    extra["declared_departure_h"] = np.linspace(-5.0, 50.0, len(fresh))
    again = model.predict(extra, KAPPAS)
    pd.testing.assert_frame_equal(table, again, check_exact=True)
    assert "declared_departure_h" in fresh.columns
    assert not np.array_equal(
        fresh["declared_departure_h"].fillna(-1), extra["declared_departure_h"]
    )


def test_a_site_day_is_forecast_alone_as_it_is_among_others():
    model, fresh, _, table = fitted_known()
    key = fresh[["unit", "site", "date"]].astype(str).agg("|".join, axis=1)
    for chosen in (key.iloc[0], key.iloc[len(key) // 2], key.iloc[-1]):
        one = fresh[key == chosen].reset_index(drop=True)
        assert 0 < len(one) < len(fresh)
        alone = model.predict(one, KAPPAS)
        within = table[table["session_id"].isin(one["session_id"].astype(str))]
        pd.testing.assert_frame_equal(alone, within.reset_index(drop=True), check_exact=True)


def test_shuffling_the_rows_or_reordering_the_cars_of_a_day_changes_no_forecast():
    model, fresh, _, table = fitted_known()
    shuffled = fresh.sample(frac=1.0, random_state=2).reset_index(drop=True)
    assert not shuffled["session_id"].equals(fresh["session_id"])
    pd.testing.assert_frame_equal(
        _sorted(table), _sorted(model.predict(shuffled, KAPPAS)), check_exact=True
    )
    key = fresh[["site", "date"]].astype(str).agg("|".join, axis=1)
    reversed_in_day = fresh.iloc[
        np.concatenate([np.flatnonzero(key == k)[::-1] for k in key.drop_duplicates()])
    ].reset_index(drop=True)
    assert not reversed_in_day["session_id"].equals(fresh["session_id"])
    pd.testing.assert_frame_equal(
        _sorted(table), _sorted(model.predict(reversed_in_day, KAPPAS)), check_exact=True
    )


# --- 4. unknown site -----------------------------------------------------------------------------


def test_a_site_not_seen_in_fitting_is_forecast_and_its_name_changes_nothing():
    model, fresh, _, _ = fitted_known()
    assert set(model.fit_sites) == {"OSL_S", "caltech"}
    first = model.predict(fresh.assign(site="jpl"), KAPPAS)
    second = model.predict(fresh.assign(site="office001"), KAPPAS)  # same state, same type
    assert len(first) == len(fresh) * len(KAPPAS) and first["d_hat"].notna().all()
    pd.testing.assert_frame_equal(
        first.drop(columns="site"), second.drop(columns="site"), check_exact=True
    )
    assert set(first["site"]) == {"jpl"}


def test_a_site_seen_in_fitting_is_not_forecast_with_the_unknown_site_vector():
    model, fresh, _, table = fitted_known()
    at_caltech = (fresh["site"] == "caltech").to_numpy()
    renamed = model.predict(fresh.assign(site=fresh["site"].replace({"caltech": "jpl"})), KAPPAS)
    seen = table[table["site"] == "caltech"][NUMBERS].to_numpy()
    unseen = renamed[renamed["session_id"].isin(fresh["session_id"][at_caltech])][NUMBERS]
    assert at_caltech.any() and not np.array_equal(seen, unseen.to_numpy(), equal_nan=True)


def test_a_site_missing_from_the_committed_mappings_is_refused():
    model, fresh, _, _ = fitted_known()
    with pytest.raises(ValueError, match="no site type"):
        model.predict(fresh.assign(site="elsewhere"), KAPPAS)


# --- 5. determinism ------------------------------------------------------------------------------


def _bytes(table: pd.DataFrame, folder, name: str) -> str:
    write_table(table, folder / name)
    return hashlib.sha256((folder / name).read_bytes()).hexdigest()


def test_two_fits_in_one_process_give_byte_identical_tables(tmp_path):
    train, fresh, _ = known_data()
    settings = dataclasses.replace(FAST, epochs=3)
    tables = []
    for _ in range(2):
        model = hazard.HazardModel(rho=0.0, kappas=KAPPAS, settings=settings)
        model.fit(train)
        tables.append(model.predict(fresh, KAPPAS))
    assert _bytes(tables[0], tmp_path, "a.csv") == _bytes(tables[1], tmp_path, "b.csv")
    other = hazard.HazardModel(
        rho=0.0, kappas=KAPPAS, settings=dataclasses.replace(settings, seed=1)
    )
    other.fit(train)
    assert _bytes(other.predict(fresh, KAPPAS), tmp_path, "c.csv") != _bytes(
        tables[0], tmp_path, "a.csv"
    )


# --- 6. levels, empty frames, the matrix ---------------------------------------------------------


def test_a_level_the_model_was_not_fitted_at_is_refused_and_an_empty_frame_gives_an_empty_table():
    model, fresh, _, _ = fitted_known()
    with pytest.raises(ValueError, match="level"):
        model.predict(fresh, (0.15,))
    empty = model.predict(fresh.iloc[0:0], KAPPAS)
    assert len(empty) == 0 and list(empty.columns) == list(check_table(empty).columns)
    fresh_model = hazard.HazardModel(rho=0.0, kappas=KAPPAS, settings=FAST)
    with pytest.raises(ValueError, match="before fit"):
        fresh_model.predict(fresh, KAPPAS)


def test_make_model_reads_the_levels_and_rho_of_the_matrix():
    matrix = SimpleNamespace(rho=0.1, configurations=[SimpleNamespace(kappa=k) for k in KAPPAS])
    model = hazard.make_model(matrix)
    assert model.rho == 0.1 and model.settings == HazardSettings()
    assert set(model.levels["rate"]) == set(KAPPAS)
    assert set(model.levels["need"]) == {0.95, 0.9, 0.8, 0.7}


def test_a_model_fitted_at_the_plans_six_levels_reads_the_right_column_at_each_level():
    levels = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)  # plan section 2's grid
    matrix = SimpleNamespace(rho=0.0, configurations=[SimpleNamespace(kappa=k) for k in levels])
    model = hazard.make_model(matrix)
    assert model.levels["rate"] == model.levels["room"] == levels
    assert model.levels["need"] == (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    assert len(model.outputs) == 3 * 6
    # Hand case: the output in column j holds the number j + 1 (mean 0, standard deviation 1), the
    # columns being rate at the six levels (0 to 5), room at the six (6 to 11) and need at
    # 0.5, 0.6, 0.7, 0.8, 0.9, 0.95 (12 to 17). Level kappa reads rate and room at kappa and need
    # at 1 - kappa: kappa 0.05 reads columns 0, 6 and 17, kappa 0.5 reads columns 5, 11 and 12.
    model.energy_mean, model.energy_std = np.zeros(3), np.ones(3)
    found = SimpleNamespace(energy=np.arange(1.0, 19.0).reshape(1, 18))
    got = model.energy_columns(found, set(levels))
    expected = {
        0.05: (1.0, 7.0, 18.0),
        0.1: (2.0, 8.0, 17.0),
        0.2: (3.0, 9.0, 16.0),
        0.3: (4.0, 10.0, 15.0),
        0.4: (5.0, 11.0, 14.0),
        0.5: (6.0, 12.0, 13.0),
    }
    for kappa, (rate, room, need) in expected.items():
        assert got[kappa]["r_hat"][0] == rate, kappa
        assert got[kappa]["e_cap"][0] == room, kappa
        assert got[kappa]["e_need"][0] == need, kappa


# --- 7. hours the data do not reach --------------------------------------------------------------


WORKPLACE, OTHER = 1, 0


def test_a_model_fitted_on_short_stays_gives_no_bound_from_the_first_unsupported_hour_on():
    model = short_stay_model()
    _, fresh, _ = known_data()  # stays of up to 9 h at two sites, one of each type
    support = model.presence_support[WORKPLACE]
    first = int(np.flatnonzero(support < model.settings.min_presence_support)[0]) + 1
    assert first == 4 and support[:3].min() >= 100 and support[3:].max() == 0
    at_workplace = fresh["site"] == "caltech"
    table = model.predict(fresh[at_workplace], KAPPAS)
    assert len(table) == at_workplace.sum() * len(KAPPAS)
    for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
        if hour < first:
            assert table[column].notna().all(), column
        else:
            assert table[column].isna().all(), column
    assert table[["d_hat", "d_median"]].notna().all().all()
    # the other site type has no fit row at all: no hour is supported for its cars
    assert model.presence_support[OTHER].max() == 0
    other = model.predict(fresh[~at_workplace], KAPPAS)
    assert len(other) > 0 and other[list(SURVIVAL_COLUMNS)].isna().all().all()
    assert other[["d_hat", "d_median"]].notna().all().all()


def _hand_support(cars: pd.DataFrame, site: str) -> list[int]:
    table = build_table(cars, rho=0.0)
    labels = table.loc[table["site"] == site, "label_departure_h"].to_numpy(float)
    return [int((labels >= hour).sum()) for hour in range(1, 9)]


def test_the_support_of_an_hour_counts_the_fit_rows_of_each_site_type_at_or_above_it():
    model, _, _, _ = fitted_known()
    train, _, _ = known_data()
    assert model.presence_support[WORKPLACE].tolist() == _hand_support(train, "caltech")
    assert model.presence_support[OTHER].tolist() == _hand_support(train, "OSL_S")
    # both types have support at every hour here
    for support in model.presence_support.values():
        assert support.min() >= model.settings.min_presence_support
    typed = typed_support_model()
    cars = typed_support_cars()
    assert typed.presence_support[WORKPLACE].tolist() == _hand_support(cars, "caltech")
    assert typed.presence_support[OTHER].tolist() == _hand_support(cars, "OSL_S")


def test_short_workplace_stays_do_not_vouch_for_late_hours_that_the_other_type_reaches():
    model = typed_support_model()
    _, fresh, _ = known_data()  # stays of up to 9 h at both sites
    support = model.presence_support
    assert support[WORKPLACE][3:].max() == 0 and support[OTHER].min() >= 100
    table = model.predict(fresh, KAPPAS)
    at_workplace = (table["site"] == "caltech").to_numpy()
    assert at_workplace.any() and (~at_workplace).any()
    for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
        if hour < 4:
            assert table[column].notna().all(), column
        else:
            assert table.loc[at_workplace, column].isna().all(), column
            assert table.loc[~at_workplace, column].notna().all(), column


def test_the_support_counts_every_fit_row_with_a_departure_label_not_only_fleet_rows():
    model = outside_fleet_model()
    table = build_table(outside_fleet_cars(), rho=0.0)
    long_stays = table["label_departure_h"] >= 4.0
    assert long_stays.sum() == 300 and table.loc[long_stays, "label_room_kwh"].isna().all()
    fleet = table["label_room_kwh"].notna().to_numpy()
    labels = table["label_departure_h"].to_numpy(float)
    on_fleet_rows = [int((labels[fleet] >= hour).sum()) for hour in range(1, 9)]
    on_every_row = [int((labels >= hour).sum()) for hour in range(1, 9)]
    assert max(on_fleet_rows[3:]) == 0 and min(on_every_row) >= model.settings.min_presence_support
    assert model.presence_support[WORKPLACE].tolist() == on_every_row
    _, fresh, _ = known_data()
    forecast = model.predict(fresh[fresh["site"] == "caltech"], KAPPAS)
    assert forecast[list(SURVIVAL_COLUMNS)].notna().all().all()


def test_the_rows_of_a_set_must_agree_on_the_site_type():
    _, fresh, _ = known_data()
    context = with_context(fresh)
    sets = make_sets(context)
    kinds = hazard.site_types_of(context, sets)
    wanted = [WORKPLACE if key[1] == "caltech" else OTHER for key in sets.keys]
    assert kinds.tolist() == wanted and {WORKPLACE, OTHER} == set(wanted)
    row = sets.members(0)[0]
    tampered = context.copy()
    tampered.loc[row, "site_type"] = 1 - tampered.loc[row, "site_type"]
    with pytest.raises(ValueError, match="site types"):
        hazard.site_types_of(tampered, sets)


def _without_the_rules(model):
    clone = copy.copy(model)
    clone.settings = dataclasses.replace(
        model.settings, min_presence_support=0, min_presence_probability=0.0
    )
    return clone


def test_where_there_is_support_the_columns_are_the_numbers_they_were():
    model, fresh, _, table = fitted_known()
    free = _without_the_rules(model).predict(fresh, KAPPAS)
    nan_places = table[list(SURVIVAL_COLUMNS)].isna().to_numpy()
    assert nan_places.any() and nan_places[:, :7].sum() == 0  # only hour 8 loses numbers
    same = table.fillna(free)
    pd.testing.assert_frame_equal(same, free, check_exact=True)
    assert table["survival_h7"].notna().all()


# The presence rule on survival curves made by hand, so that the result does not depend on a fit.
# A curve with presence probability `p` at hour 8: nobody leaves before the bin that starts at
# 7.75 h, a share `1 - p` leaves in it, and everybody else in the bin that starts at 8 h. The bound
# given presence at hour 8 is then 8.0; at hours 1 to 7 (presence 1) it is 7.75.
FLOOR = 1e-3
TIMES = hazard_curves.bin_starts(HazardSettings())


def _curve_with_presence(p: float) -> np.ndarray:
    eight = int(np.flatnonzero(TIMES == 8.0)[0])
    curve = np.ones(len(TIMES))
    curve[eight - 1] = p
    curve[eight:] = 0.0
    return curve


def test_departure_table_gives_no_bound_below_the_floor_and_a_bound_at_or_above_it():
    below, equal, above = 0.9 * FLOOR, FLOOR, 1.1 * FLOOR
    survival = np.stack([_curve_with_presence(p) for p in (below, equal, above, 0.5)])
    table = hazard_curves.departure_table(TIMES, survival, [0.1], presence_floor=FLOOR)[0.1]
    assert np.isnan(table["survival_h8"][0])
    assert table["survival_h8"][1:].tolist() == [8.0, 8.0, 8.0]
    for hour in range(1, 8):  # presence 1 at the other hours
        assert table[f"survival_h{hour}"].tolist() == [7.75] * 4, hour
    free = hazard_curves.departure_table(TIMES, survival, [0.1])[0.1]  # no floor
    assert free["survival_h8"].tolist() == [8.0] * 4
    for name in ("d_hat", "d_median"):
        assert np.array_equal(table[name], free[name], equal_nan=True)


def test_departure_table_gives_no_bound_at_an_hour_that_is_not_supported_for_any_car():
    survival = np.stack([_curve_with_presence(0.5)] * 3)
    supported = [True, True, False, True, True, True, True, True]
    table = hazard_curves.departure_table(TIMES, survival, [0.1], supported)[0.1]
    assert np.isnan(table["survival_h3"]).all()
    for hour in (1, 2, 4, 5, 6, 7):
        assert table[f"survival_h{hour}"].tolist() == [7.75] * 3, hour
    assert table["survival_h8"].tolist() == [8.0] * 3


def test_the_model_applies_the_floor_to_the_curves_it_is_given(monkeypatch):
    model, fresh, _, _ = fitted_known()
    assert model.settings.min_presence_probability == FLOOR
    assert min(s.min() for s in model.presence_support.values()) >= 100
    chances = (0.9 * FLOOR, FLOOR, 1.1 * FLOOR, 0.5)

    def made_by_hand(self, x, site):
        survival = np.stack([_curve_with_presence(chances[j % 4]) for j in range(len(x))])
        return survival, np.zeros((len(x), len(self.outputs)))

    monkeypatch.setattr(hazard.HazardModel, "curves", made_by_hand)
    table = model.predict(fresh, KAPPAS)
    sets = make_sets(with_context(fresh))
    place = np.empty(len(fresh), dtype=int)
    for k in range(sets.n_sets):
        place[sets.members(k)] = np.arange(len(sets.members(k))) % 4
    for kappa in KAPPAS:
        part = table[table["kappa"] == kappa].reset_index(drop=True)
        assert part["survival_h8"][place == 0].isna().all()
        assert (part["survival_h8"][place > 0] == 8.0).all()
        for hour in range(1, 8):
            assert (part[f"survival_h{hour}"] == 7.75).all(), (kappa, hour)


# --- 8. prediction never uses the posterior ------------------------------------------------------


def test_prediction_does_not_call_the_posterior_network(monkeypatch):
    model, fresh, _, table = fitted_known()

    def used(*arguments, **keywords):
        raise AssertionError("the posterior network was used")

    monkeypatch.setattr(HazardNet, "posterior_parameters", used)
    pd.testing.assert_frame_equal(table, model.predict(fresh, KAPPAS), check_exact=True)
    x, sets = model.inputs(fresh)
    survival, _ = model.curves(x[sets.members(0)], 1)
    assert survival.shape[0] == len(sets.members(0))


def test_overwriting_the_posterior_weights_after_the_fit_changes_no_forecast():
    model, fresh, _, table = fitted_known()
    clone = copy.deepcopy(model)
    changed = 0
    with torch.no_grad():
        for name, weight in clone.net.named_parameters():
            if name.startswith(("posterior", "label_token")):
                weight.fill_(7.0)
                changed += 1
    assert changed > 0
    pd.testing.assert_frame_equal(table, clone.predict(fresh, KAPPAS), check_exact=True)


# --- 9. sites are hidden in fitting --------------------------------------------------------------


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_the_unknown_site_vector_is_trained_exactly_when_sites_are_hidden(probability):
    train, _, _ = known_data()
    settings = dataclasses.replace(FAST, epochs=6, unknown_site_probability=probability)
    model = hazard.HazardModel(rho=0.0, kappas=KAPPAS, settings=settings)
    model.fit(train)
    start = hazard_fit.build_net(
        settings, len(model.fit_sites), len(model.outputs), hazard_curves.bin_widths(settings)
    )
    moved = (model.net.site.weight - start.site.weight).abs().max(dim=1).values
    if probability == 1.0:  # every set is seen as the unknown site: the real sites stay put
        assert moved[0] > 0.01 and moved[1:].max() < 1e-3
    else:  # no set is: the unknown vector stays put
        assert moved[0] < 1e-3 and moved[1:].min() > 0.01
