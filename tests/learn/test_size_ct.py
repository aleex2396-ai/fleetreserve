"""Conformal training of the day-size model (`size_ct`) on made-up fleets (`ct_fake`).

1. By hand: the arms' links and gate; a network of constant size 0.6 on a batch of 40 days with
   labels 0.6 (20 calibrate: the allowance 0.1 x 21 - 1 = 1.1 and the 20 ramps start together at
   the scalar 1) offers the share (0.05 x 1.1 + 20 x 0.6) / 20 = 0.60275 whatever the size, so
   its gradient in the size is 0; the exact validation offer of 19 days with labels 0.05, ...,
   0.95 and value 1 is 0.1 at size 1 and at size 0.5 alike.
2. Epoch 0 is the start: with no epoch the outputs are the bytes of the day-size net at 0.1
   (`size_net.leave_one_unit_out`), so a network kept at epoch 0 is the net.
3. T7's case (`ct_fake.unequal_rule`, 250 days per unit, seed 51 fixed before any run of it),
   read through the existing route (`size_scale.sizes`, `size_resolve.calibrate` on the gate's
   grid, `size_resolve.score`; 40 random splits of 100 calibration days, tier B): on every held-out
   unit and arm conformal training offers more than the constant and more than its own outputs
   shuffled over the days, its test slot loss within alpha; pooled over the units and arms it
   offers at least as much as the net at 0.1, whose test slot loss is within alpha too.
4. Equal marginal losses (`ct_fake.equal_rule`, seed 52): on the grid any spread of sizes gains
   on the constant (pooled, the shuffled outputs gain over 10%), because the constant's scalar
   stops at the grid point below its exact value; with the exact scalar in place of the grid's
   (`ct_fake.exact_figures`), conformal training is within 10% of the constant on every unit and
   arm. In both builds the shuffled outputs are within 10% of the constant with the exact scalar
   on every unit and arm: shuffling gains nothing.
5. No input after τ: the outputs read the cars plugged in at τ (`size_net_inputs`) and the fit's
   labels and values only; a column added to the cars that no input reads, and the scored unit's
   own labels and values, change no byte.
6. Leave one unit out: a unit's own labels and values never reach its outputs, the other units'
   do; a unit alone gets none.
7. The training-only guard: a fit on a development unit, or on a unit of the final run, is refused
   before anything is fitted (`learn_tune.NotTraining`).
8. Two runs with one seed give the same bytes; another seed gives another fit (its record of
   offers per epoch differs).
"""

from dataclasses import replace

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import torch
from ct_fake import equal_rule, exact_figures, planted, split_figures, unequal_rule

from fleetreserve import crc, learn_tune, size_ct, size_gbm, size_net, size_scale
from fleetreserve.hazard_data import SetIndex

UNITS = ("caltech", "jpl", "OSL_S", "ASK")
QUICK = size_ct.CtSettings(epochs=4, start=size_net.NetSettings(epochs=4))


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


# --- 1. by hand ----------------------------------------------------------------------------------


def test_the_links_and_the_gate_by_hand():
    z = torch.tensor([0.02, size_scale.FLOOR, 0.5, 1.7], dtype=torch.float64)
    assert size_ct.sized(z, "scale").tolist() == [size_scale.FLOOR, size_scale.FLOOR, 0.5, 1.0]
    assert size_ct.sized(z, "abstain").tolist() == [0.02, size_scale.FLOOR, 0.5, 1.0]
    settings = size_ct.CtSettings()
    assert size_ct.gate(z, "scale", settings).tolist() == [1.0] * 4
    found = size_ct.gate(z, "abstain", settings).tolist()
    assert found[1] == 0.5 and found[0] < 0.05 and found[2] > 0.999


def test_a_constant_size_offers_the_same_share_by_hand():
    n = 40
    keys = tuple(("u", "s", str(k)) for k in range(n))
    sets = SetIndex(np.arange(2 * n), np.arange(0, 2 * n + 1, 2), keys)  # two cars a day
    x = np.random.default_rng(3).normal(size=(2 * n, 4)).astype(np.float32)
    net = size_net.build_net(4, [0.6], size_net.NetSettings())  # 0.6 on every day
    settings = size_ct.CtSettings(splits_per_batch=1)
    labels, values = np.full(n, 0.6), np.full(n, 10.0)
    offer = size_ct.batch_offer(
        net, x, sets, np.arange(n), labels, values, "scale", settings, np.random.default_rng(0)
    )
    assert float(offer.detach()) == pytest.approx((0.05 * 1.1 + 20 * 0.6) / 20, abs=1e-9)
    assert float(offer.detach()) == pytest.approx(0.60275, abs=1e-9)
    offer.backward()
    assert float(net.head[2].bias.grad[0]) == pytest.approx(0.0, abs=1e-6)
    s = np.array([0.05 * k for k in range(1, 20)])
    for size in (1.0, 0.5):
        found = size_ct.exact_offer(
            np.full(19, size), s, np.ones(19), [np.arange(19)], "scale", crc.ALPHA_SLOT
        )
        assert found == pytest.approx(0.1)


# --- 2. epoch 0 is the net ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_build():
    return planted(np.random.default_rng(13), ("caltech", "jpl", "ASK"), 60, unequal_rule)


def test_with_no_epoch_the_outputs_are_the_net(small_build):
    cars, rows = small_build
    settings = size_ct.CtSettings(epochs=0, start=QUICK.start)
    found = size_ct.leave_one_unit_out(rows, cars, settings=settings)
    net = size_net.leave_one_unit_out(rows, cars, (0.1,), settings=QUICK.start)
    for arm in size_scale.ARMS:
        np.testing.assert_array_equal(found[arm], net[0.1]["net"])


# --- 3. and 4. the made-up builds ----------------------------------------------------------------


def held_out(rows, outputs, unit):
    held = (rows["unit"] == unit).to_numpy()
    labels = rows.loc[held, size_gbm.LABEL].to_numpy(float)
    values = rows.loc[held, size_ct.VALUE].to_numpy(float)
    return held, labels, values


@pytest.fixture(scope="module")
def unequal_build():
    torch.set_num_threads(1)
    cars, rows = planted(np.random.default_rng(51), UNITS, 250, unequal_rule)
    net = size_net.leave_one_unit_out(rows, cars, (0.1,))[0.1]["net"]
    return rows, net, size_ct.leave_one_unit_out(rows, cars)


@pytest.fixture(scope="module")
def equal_build():
    torch.set_num_threads(1)
    cars, rows = planted(np.random.default_rng(52), UNITS, 250, equal_rule)
    return rows, size_ct.leave_one_unit_out(rows, cars)


def test_in_t7s_case_it_gains_on_the_constant_and_matches_the_net(unequal_build):
    rows, net, ct = unequal_build
    pooled = {"ct": [], "net": []}
    for unit in UNITS:
        held, labels, values = held_out(rows, ct, unit)
        for arm in size_scale.ARMS:
            raw = {
                "constant": np.ones(held.sum()),
                "ct": ct[arm][held],
                "net": net[held],
                "shuffled": np.random.default_rng(1).permutation(ct[arm][held]),
            }
            found = {name: split_figures(labels, values, q, arm, "B") for name, q in raw.items()}
            assert found["ct"][0] > found["constant"][0], (unit, arm, found)
            assert found["ct"][0] > found["shuffled"][0], (unit, arm, found)
            assert found["ct"][1] <= crc.ALPHA_SLOT and found["net"][1] <= crc.ALPHA_SLOT
            pooled["ct"].append(found["ct"][0])
            pooled["net"].append(found["net"][0])
            shuffled = exact_figures(labels, values, raw["shuffled"], arm)
            constant = exact_figures(labels, values, raw["constant"], arm)
            assert shuffled <= 1.1 * constant, (unit, arm, shuffled, constant)
    assert np.mean(pooled["ct"]) >= np.mean(pooled["net"]), pooled


def test_without_a_difference_it_is_within_noise_of_the_constant(equal_build):
    rows, ct = equal_build
    grid = {"constant": [], "shuffled": []}
    for unit in UNITS:
        held, labels, values = held_out(rows, ct, unit)
        for arm in size_scale.ARMS:
            constant = np.ones(held.sum())
            shuffled = np.random.default_rng(1).permutation(ct[arm][held])
            grid["constant"].append(split_figures(labels, values, constant, arm, "B")[0])
            grid["shuffled"].append(split_figures(labels, values, shuffled, arm, "B")[0])
            base = exact_figures(labels, values, constant, arm)
            ours = exact_figures(labels, values, ct[arm][held], arm)
            assert abs(ours - base) <= 0.1 * base, (unit, arm, ours, base)
            assert exact_figures(labels, values, shuffled, arm) <= 1.1 * base, (unit, arm)
    assert np.mean(grid["shuffled"]) > 1.1 * np.mean(grid["constant"]), grid


# --- 5. to 8. ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_outputs(small_build):
    torch.set_num_threads(1)
    cars, rows = small_build
    return size_ct.leave_one_unit_out(rows, cars, settings=QUICK)


def test_no_input_after_the_decision_time(small_build, small_outputs):
    cars, rows = small_build
    cars = cars.copy()
    cars["leaves_at_h"] = np.random.default_rng(4).uniform(0, 24, len(cars))
    rows = rows.copy()
    own = (rows["unit"] == "jpl").to_numpy()
    again = size_ct.leave_one_unit_out(rows, cars, settings=QUICK)
    rows.loc[own, size_ct.VALUE] = 1.0
    rows.loc[own, size_gbm.LABEL] = 0.0
    later = size_ct.leave_one_unit_out(rows, cars, settings=QUICK)
    for arm in size_scale.ARMS:
        np.testing.assert_array_equal(again[arm], small_outputs[arm])
        np.testing.assert_array_equal(later[arm][own], small_outputs[arm][own])


def test_a_units_own_days_never_reach_its_outputs(small_build, small_outputs):
    cars, rows = small_build
    poisoned = rows.copy()
    own = (poisoned["unit"] == "jpl").to_numpy()
    rng = np.random.default_rng(2)
    poisoned.loc[own, size_gbm.LABEL] = rng.uniform(0.0, 0.05, own.sum())
    poisoned.loc[own, size_ct.VALUE] = rng.uniform(50.0, 100.0, own.sum())
    again = size_ct.leave_one_unit_out(poisoned, cars, settings=QUICK)
    for arm in size_scale.ARMS:
        np.testing.assert_array_equal(again[arm][own], small_outputs[arm][own])
        assert (again[arm][~own] != small_outputs[arm][~own]).any()
    alone = rows[rows["unit"] == "jpl"].reset_index(drop=True)
    found = size_ct.leave_one_unit_out(alone, cars, settings=QUICK)
    assert all(np.isnan(found[arm]).all() for arm in size_scale.ARMS)


@pytest.mark.parametrize("outside", ["nrel-from-may-2019", "OSL_T"])
def test_a_unit_that_is_not_a_training_unit_is_refused(small_build, monkeypatch, outside):
    cars, rows = small_build
    renamed = rows.copy()
    renamed.loc[renamed["unit"] == "ASK", "unit"] = outside
    cars = cars.copy()
    cars.loc[cars["unit"] == "ASK", "unit"] = outside

    def no_fit(*arguments, **keywords):
        raise AssertionError("a network was fitted")

    monkeypatch.setattr(size_ct, "fit", no_fit)
    monkeypatch.setattr(size_net, "fit", no_fit)
    with pytest.raises(learn_tune.NotTraining, match=outside):
        size_ct.leave_one_unit_out(renamed, cars, settings=QUICK)
    with pytest.raises(learn_tune.NotTraining, match=outside):
        size_ct.fit_days(renamed[renamed["unit"] == outside], cars, "scale", QUICK)


def test_two_runs_with_one_seed_give_the_same_bytes(small_build, small_outputs, monkeypatch):
    cars, rows = small_build
    again = size_ct.leave_one_unit_out(rows, cars, settings=QUICK)
    for arm in size_scale.ARMS:
        assert again[arm].tobytes() == small_outputs[arm].tobytes()
        assert not np.isnan(again[arm]).any()
    fits = []
    real = size_ct.fit
    monkeypatch.setattr(size_ct, "fit", lambda *a, **k: fits.append(real(*a, **k)) or fits[-1])
    others = rows[rows["unit"] != "jpl"]
    for seed in (0, 0, 1):
        size_ct.fit_days(others, cars, "scale", replace(QUICK, seed=seed))
    assert fits[0].history == fits[1].history and fits[0].validation == fits[1].validation
    assert fits[0].history != fits[2].history
