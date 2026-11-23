"""The reference size predictor (`size_gbm`) and what it gives through `size_scale`, on planted
made-up label tables.

Three made-up units of 150 labelled days each. A context column `early_leaver` marks 30% of the
days; on those the label is 0.2 and the fixed plan fails above scale 0.2, on the others the label is
1 and it fails above 5. A second column is noise.

1. Leave one unit out: unit u's predictions are the same when u's labels are replaced by others
   (every row of u poisoned), and change when another unit's labels change (so the check can see a
   change). A unit with no other unit gets no prediction.
2. The constant predictor of a level is a minimiser of the mean pinball loss over constants (by
   brute force over every label and points between them).
3. On the planted days the booster's predictions at 0.1 are low on the marked days and high on the
   others, its pinball loss is below the constant's, and sized by them (both arms) the calibrated
   offer of the held-out unit is well above constant's, with mean test slot loss within the
   limit. The same predictions shuffled over the days do not raise it.
4. Two fits give the same predictions to the bit.
"""

import math

import numpy as np
import pandas as pd
import pytest
from size_fake import ramp_terms

from fleetreserve import crc, size_gbm, size_scale
from fleetreserve.learn_eval import pinball_loss

COLUMNS = ("early_leaver", "noise")
UNITS = ("caltech", "jpl", "OSL_S")
DAYS = 150


def planted() -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(12)
    frames, terms = [], []
    for unit in UNITS:
        early = rng.random(DAYS) < 0.3
        frames.append(
            pd.DataFrame(
                {
                    "unit": unit,
                    "early_leaver": early.astype(float),
                    "noise": rng.random(DAYS),
                    size_gbm.LABEL: np.where(early, 0.2, 1.0),
                }
            )
        )
        for e in early:
            s = 0.2 if e else 5.0
            terms.append(ramp_terms(10.0, np.array([s, s]), (np.array([-s]), np.array([1.0]), 1.0)))
    return pd.concat(frames, ignore_index=True), terms


@pytest.fixture(scope="module")
def table():
    return planted()[0]


@pytest.fixture(scope="module")
def predicted(table):
    return size_gbm.leave_one_unit_out(table, COLUMNS)


def test_a_units_own_labels_never_reach_its_predictions(table, predicted):
    poisoned = table.copy()
    rows = poisoned["unit"] == "jpl"
    poisoned.loc[rows, size_gbm.LABEL] = np.random.default_rng(2).uniform(0.0, 0.05, rows.sum())
    again = size_gbm.leave_one_unit_out(poisoned, COLUMNS)
    for level in size_gbm.LEVELS:
        for kind in ("gbm", "constant"):
            np.testing.assert_array_equal(again[level][kind][rows], predicted[level][kind][rows])
            # the other units are fitted on jpl's rows, so theirs move
            assert (again[level][kind][~rows] != predicted[level][kind][~rows]).any()


def test_a_unit_alone_gets_no_prediction(table):
    alone = table[table["unit"] == "jpl"].reset_index(drop=True)
    found = size_gbm.leave_one_unit_out(alone, COLUMNS)
    assert all(np.isnan(found[level]["gbm"]).all() for level in size_gbm.LEVELS)


@pytest.mark.parametrize("level", size_gbm.LEVELS)
def test_the_constant_minimises_the_pinball_loss(level):
    labels = np.random.default_rng(int(level * 100)).random(37)
    best = size_gbm.constant(labels, level)
    grid = np.unique(np.concatenate([labels, (labels[:-1] + labels[1:]) / 2, [0.0, 1.0]]))
    losses = [pinball_loss(labels, np.full(labels.size, c), level) for c in grid]
    assert pinball_loss(labels, np.full(labels.size, best), level) <= min(losses) + 1e-12


def test_the_planted_marker_raises_the_calibrated_offer(table, predicted):
    _, terms = planted()
    held = (table["unit"] == "OSL_S").to_numpy()
    days = np.flatnonzero(held)
    held_terms = [terms[d] for d in days]
    early = table.loc[held, "early_leaver"].to_numpy(bool)
    gbm = predicted[0.1]["gbm"][held]
    assert gbm[early].max() < 0.3 and gbm[~early].min() > 0.9
    labels = table.loc[held, size_gbm.LABEL].to_numpy(float)
    constant = predicted[0.1]["constant"][held]
    assert pinball_loss(labels, gbm, 0.1) < pinball_loss(labels, constant, 0.1)
    offering = np.ones(len(days), dtype=bool)
    calibration, test = list(range(100)), list(range(100, len(days)))
    rng = np.random.default_rng(1)
    raw = {
        "constant": np.ones(len(days)),
        "gbm": gbm,
        "shuffled": rng.permutation(gbm),
    }
    offers = {}
    for name, values in raw.items():
        for arm in size_scale.ARMS:
            q = size_scale.sizes(values, offering, arm)
            scalar = size_scale.calibrate(held_terms, q, calibration)["A"]
            scored = size_scale.score(held_terms, q, scalar.used, test)
            offers[(name, arm)] = scored.mean_offer_kwh
            assert size_scale.mean_or_none(scored.slot_losses) <= crc.ALPHA_SLOT + 0.05
    base = offers[("constant", "scale")]
    assert base == pytest.approx(2.0)
    for arm in size_scale.ARMS:
        assert offers[("gbm", arm)] > 3 * base
        assert offers[("shuffled", arm)] <= base + 1e-9


def test_two_fits_give_the_same_predictions(table, predicted):
    again = size_gbm.leave_one_unit_out(table, COLUMNS)
    for level in size_gbm.LEVELS:
        for kind in ("gbm", "constant"):
            np.testing.assert_array_equal(again[level][kind], predicted[level][kind])
    assert not math.isnan(predicted[0.5]["gbm"][0])
