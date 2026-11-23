"""The day-size quantile net (`size_net`) on made-up fleets (`size_net_fake`).

1. By hand: a network with weights set by hand gives 0.5 + 0.5 x (the first input, the same for
   every car of a day): 0.5 on the days with x = 0 and 1 on the days with x = 1. Its pinball loss on
   labels 0.4 and 0.9 is 0.09 at level 0.1 and 0.05 at level 0.5 (0.07 over both), equal to
   `learn_eval.pinball_loss`. Sized by it through the existing calibration (`size_scale.sizes`,
   `size_resolve.relative`, `size_resolve.calibrate`, `size_resolve.score`) on a hand-made loss
   table (a day loses above scale 0.4 when x = 0, above 0.9 when x = 1; 20 calibration days, ten of
   each), the scalar is 0.8 and the offer on one test day of each kind of value 10 is 6 kWh per day,
   against 0.4 and 4 for the constant sizing.
2. The output of a set does not depend on the order of its cars (up to rounding), and does
   depend on the cars' values.
3. Two fits with the same seed give byte-identical predictions; another seed gives others. The
   networks of the levels are fitted apart (`size_net`, reading N5), so a run on two of the
   levels gives the bytes of the full run at those levels.
4. Leave one unit out: a unit's own labels never reach its predictions; a unit alone gets none.
5. The training-only guard: a fit on a development unit, or on a unit of the final run, is refused
   before anything is fitted (`learn_tune.NotTraining`).
6. Planted context: where the label depends on a context column (school holidays) and on the
   number of cars, the held-out unit's pinball loss is below 0.9 times the constant's at every
   level, outside the 10% band of 7. Sized by the net at levels 0.1 and 0.5 (both arms) and
   calibrated on the one-way re-solve's route (`size_resolve.calibrate`, both tiers) with a day
   losing every slot above its label, the mean offer over 40 random splits of the unit's days (100
   calibration, 50 test) is above the constant's and above that of the same predictions shuffled
   over the days, and the mean test slot loss is within the slot alpha.
7. Noise: where the label depends on nothing, the held-out pinball loss is within 10% of the
   constant's at every level.
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import size_net_fake as fake
import torch

from fleetreserve import crc, learn_tune, size_gbm, size_net, size_resolve, size_scale
from fleetreserve.hazard_data import SetIndex
from fleetreserve.learn_eval import pinball_loss

UNITS = ("caltech", "jpl", "OSL_S", "ASK")


def sets_of(sizes) -> SetIndex:
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    keys = tuple(("u", "s", str(k)) for k in range(len(sizes)))
    return SetIndex(np.arange(starts[-1], dtype=int), starts, keys)


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


# --- 1. by hand ----------------------------------------------------------------------------------


def hand_net(n_inputs: int, levels) -> size_net.SizeNet:
    settings = size_net.NetSettings(width=2, hidden=2)
    net = size_net.build_net(n_inputs, np.zeros(len(levels)), settings)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        first, last = net.head[0], net.head[2]
        first.weight[0, settings.width + 0] = 1.0  # hidden unit 0 reads the first input's mean
        last.weight[:, 0] = 0.5
        last.bias[:] = 0.5
    return net


def hand_days():
    kinds = [0, 1] + [0] * 10 + [1] * 10  # days 0 and 1 are the test days
    sets = sets_of([3] * len(kinds))
    x = np.zeros((3 * len(kinds), 2), dtype=np.float32)
    x[:, 0] = np.repeat(kinds, 3)
    x[:, 1] = np.random.default_rng(3).normal(size=len(x))  # read by no weight
    return np.array(kinds), sets, x


def test_by_hand_the_loss_and_the_calibrated_offer():
    kinds, sets, x = hand_days()
    levels = (0.1, 0.5)
    net = hand_net(2, levels)
    out = size_net.predict(net, x, sets, size_net.NetSettings())
    assert out[:, 0].tolist() == [0.5 if k == 0 else 1.0 for k in kinds]
    assert out[:, 1].tolist() == out[:, 0].tolist()
    labels = np.where(kinds == 0, 0.4, 0.9)
    for column, level, by_hand in ((0, 0.1, 0.09), (1, 0.5, 0.05)):
        assert pinball_loss(labels, out[:, column], level) == pytest.approx(by_hand, abs=1e-12)
    torch_loss = size_net.loss(
        torch.as_tensor(out, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
        torch.as_tensor(levels, dtype=torch.float32),
    )
    assert float(torch_loss) == pytest.approx(0.07, abs=1e-6)

    threshold = np.where(kinds == 0, 0.4, 0.9)

    def lose(d, s):
        return (1.0 if s > threshold[d] + 1e-12 else 0.0, 0.0)

    offering = np.ones(len(kinds), dtype=bool)
    calibration, test = list(range(2, len(kinds))), [0, 1]
    value = np.full(len(kinds), 10.0)
    found = {}
    for name, raw in (("net", out[:, 0]), ("constant", np.ones(len(kinds)))):
        r = size_resolve.relative(size_scale.sizes(raw, offering, "scale"))
        losses = {pair: lose(*pair) for pair in size_resolve.wanted(r)}
        for tier in size_scale.TIERS:
            scalar = size_resolve.calibrate(losses, r, calibration, tier)
            score = size_resolve.score(losses, r, value, scalar.used, test)
            found[(name, tier)] = (scalar.used, score.mean_offer_kwh, score.slot_losses)
    for tier in size_scale.TIERS:
        assert found[("net", tier)] == (pytest.approx(0.8), pytest.approx(6.0), (0.0, 0.0))
        assert found[("constant", tier)] == (pytest.approx(0.4), pytest.approx(4.0), (0.0, 0.0))


# --- 2. order of the cars ------------------------------------------------------------------------


def test_the_order_of_the_cars_does_not_matter_and_their_values_do():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(7, 5)).astype(np.float32)
    sets = sets_of([7])
    net = fake.drawn_net(5, 3)
    first = size_net.predict(net, x, sets, size_net.NetSettings())
    for seed in range(5):
        shuffled = x[np.random.default_rng(seed).permutation(7)]
        again = size_net.predict(net, shuffled, sets, size_net.NetSettings())
        np.testing.assert_allclose(again, first, rtol=1e-6, atol=1e-6)
    moved = x.copy()
    moved[2, 1] += 1.0
    assert not np.allclose(size_net.predict(net, moved, sets, size_net.NetSettings()), first)


# --- 3. to 7. on the made-up car frames ----------------------------------------------------------


@pytest.fixture(scope="module")
def context_build():
    return fake.planted(np.random.default_rng(11), UNITS, 150, fake.context_rule)


@pytest.fixture(scope="module")
def context_predicted(context_build):
    torch.set_num_threads(1)
    cars, rows = context_build
    return size_net.leave_one_unit_out(rows, cars)


TWO_LEVELS = (0.1, 0.5)


def test_two_fits_with_one_seed_give_the_same_bytes(context_build, context_predicted):
    cars, rows = context_build
    again = size_net.leave_one_unit_out(rows, cars, TWO_LEVELS)
    other = size_net.leave_one_unit_out(
        rows, cars, TWO_LEVELS, settings=size_net.NetSettings(seed=1)
    )
    assert sorted(again) == list(TWO_LEVELS)
    for level in TWO_LEVELS:
        np.testing.assert_array_equal(again[level]["net"], context_predicted[level]["net"])
        assert not np.array_equal(other[level]["net"], context_predicted[level]["net"])
    for level in size_net.LEVELS:
        assert not np.isnan(context_predicted[level]["net"]).any()


def test_a_units_own_labels_never_reach_its_predictions(context_build, context_predicted):
    cars, rows = context_build
    poisoned = rows.copy()
    own = (poisoned["unit"] == "jpl").to_numpy()
    poisoned.loc[own, size_gbm.LABEL] = np.random.default_rng(2).uniform(0.0, 0.05, own.sum())
    again = size_net.leave_one_unit_out(poisoned, cars, TWO_LEVELS)
    for level in TWO_LEVELS:
        np.testing.assert_array_equal(
            again[level]["net"][own], context_predicted[level]["net"][own]
        )
        assert (again[level]["net"][~own] != context_predicted[level]["net"][~own]).any()


def test_a_unit_alone_gets_no_prediction(context_build):
    cars, rows = context_build
    alone = rows[rows["unit"] == "jpl"].reset_index(drop=True)
    found = size_net.leave_one_unit_out(alone, cars)
    assert all(np.isnan(found[level]["net"]).all() for level in size_net.LEVELS)


@pytest.mark.parametrize("outside", ["nrel-from-may-2019", "OSL_T"])
def test_a_unit_that_is_not_a_training_unit_is_refused(context_build, monkeypatch, outside):
    cars, rows = context_build
    renamed = rows.copy()
    renamed.loc[renamed["unit"] == "ASK", "unit"] = outside
    cars = cars.copy()
    cars.loc[cars["unit"] == "ASK", "unit"] = outside

    def no_fit(*arguments, **keywords):
        raise AssertionError("a network was fitted")

    monkeypatch.setattr(size_net, "fit", no_fit)
    with pytest.raises(learn_tune.NotTraining, match=outside):
        size_net.leave_one_unit_out(renamed, cars)
    with pytest.raises(learn_tune.NotTraining, match=outside):
        size_net.fit_days(renamed[renamed["unit"] == outside], cars)


def held_out(rows, predicted, unit, level):
    held = (rows["unit"] == unit).to_numpy()
    labels = rows.loc[held, size_gbm.LABEL].to_numpy(float)
    constant = size_gbm.constant(rows.loc[~held, size_gbm.LABEL].to_numpy(float), level)
    return held, labels, predicted[level]["net"][held], np.full(labels.size, constant)


SPLITS = 40


def split_figures(labels, raw, arm, tier) -> tuple[float, float]:
    """(mean offer, mean test slot loss) over `SPLITS` random splits of the days (100
    calibration, the rest test), a day losing every slot above its label, value 10 per day; the
    slot loss over the splits where it is defined, as `size_resolve_summary` reads it."""
    offering = np.ones(labels.size, dtype=bool)
    value = np.full(labels.size, 10.0)
    r = size_resolve.relative(size_scale.sizes(raw, offering, arm))
    losses = {
        (d, s): (1.0 if s > labels[d] + 1e-12 else 0.0, 0.0) for d, s in size_resolve.wanted(r)
    }
    offers, slots = [], []
    for split in range(SPLITS):
        order = np.random.default_rng([7, split]).permutation(labels.size)
        calibration, test = sorted(order[:100]), sorted(order[100:])
        scalar = size_resolve.calibrate(losses, r, calibration, tier)
        scored = size_resolve.score(losses, r, value, scalar.used, test)
        offers.append(scored.mean_offer_kwh)
        slot = size_scale.mean_or_none(scored.slot_losses)
        if slot is not None:
            slots.append(slot)
    return float(np.mean(offers)), float(np.mean(slots)) if slots else math.nan


@pytest.mark.parametrize("unit", ["jpl", "ASK"])
def test_a_planted_context_is_learned(context_build, context_predicted, unit):
    _, rows = context_build
    for level in size_net.LEVELS:
        _, labels, net, constant = held_out(rows, context_predicted, unit, level)
        ours, theirs = pinball_loss(labels, net, level), pinball_loss(labels, constant, level)
        assert ours < 0.9 * theirs, (unit, level, ours, theirs)
    for level in TWO_LEVELS:
        _, labels, net, _ = held_out(rows, context_predicted, unit, level)
        raw = {
            "constant": np.ones(labels.size),
            "net": net,
            "shuffled": np.random.default_rng(1).permutation(net),
        }
        for arm in size_scale.ARMS:
            for tier in size_scale.TIERS:
                found = {name: split_figures(labels, q, arm, tier) for name, q in raw.items()}
                assert found["constant"][0] > 0
                assert found["net"][0] > found["constant"][0], (level, arm, tier, found)
                assert found["net"][0] > found["shuffled"][0], (level, arm, tier, found)
                assert found["net"][1] <= crc.ALPHA_SLOT, (level, arm, tier, found)


def test_without_a_signal_the_net_is_within_noise_of_the_constant():
    torch.set_num_threads(1)
    cars, rows = fake.planted(np.random.default_rng(12), UNITS, 150, fake.noise_rule)
    predicted = size_net.leave_one_unit_out(rows, cars)
    for unit in ("jpl", "ASK"):
        for level in size_net.LEVELS:
            _, labels, net, constant = held_out(rows, predicted, unit, level)
            ours, theirs = pinball_loss(labels, net, level), pinball_loss(labels, constant, level)
            assert abs(ours - theirs) <= 0.1 * theirs, (unit, level, ours, theirs)
            assert not math.isnan(ours)
