"""What `seed_average` averages (reading A1), on two made-up members with given curves, no fit.

Member 1 leaves at a constant hazard of 0.5 per bin: survival 1, 0.5, 0.25. Member 2 never leaves
in the first bin and always in the second: hazards 0 and 1, survival 1, 1, 0. The mean of the
survival curves (A1) is 1, 0.75, 0.125. The mean of the hazards, 0.25 and 0.75, would give
1, 0.75, 0.1875 instead, so the two rules differ at the last point. The averaged model's curve is
the mean of the members' own `set_curves` survival, at every point, and not the curve of the mean
hazards.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fleetreserve import seed_average
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.models.hazard import SetCurves

ONE = np.array([[1.0, 0.5, 0.25]])
TWO = np.array([[1.0, 1.0, 0.0]])


class Member:
    """A made-up member of the family: its own curve for the one set of one car."""

    def __init__(self, seed: int, survival: np.ndarray) -> None:
        self.settings = HazardSettings(seed=seed)
        self.rho = 0.0
        self.survival = survival

    def set_curves(self, frame) -> SetCurves:
        return SetCurves(
            1, [np.array([0])], np.array([1]), [self.survival.copy()], np.zeros((1, 3))
        )


def _hazards(survival: np.ndarray) -> np.ndarray:
    return 1 - survival[:, 1:] / survival[:, :-1]


def _from_hazards(hazards: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(hazards), 1)), np.cumprod(1 - hazards, axis=1)], axis=1)


@pytest.fixture
def average():
    return seed_average.average_of([Member(2, TWO), Member(1, ONE)])


def test_the_two_rules_differ_on_this_case():
    mean_hazards = _from_hazards((_hazards(ONE) + _hazards(TWO)) / 2)
    assert mean_hazards[0, 2] == pytest.approx(0.1875)
    assert not np.allclose(mean_hazards, (ONE + TWO) / 2)


def test_the_averaged_curve_is_the_mean_of_the_members_survival_curves(average):
    frame = SimpleNamespace()
    mean, found = average.mean_curves(frame)
    expected = np.mean(np.stack([m.set_curves(frame).survival[0] for m in average.members]), 0)
    np.testing.assert_array_equal(mean.survival[0], expected)
    np.testing.assert_array_equal(mean.survival[0], np.array([[1.0, 0.75, 0.125]]))
    assert [f.survival[0].tolist() for f in found] == [ONE.tolist(), TWO.tolist()]


def test_the_averaged_curve_is_not_the_curve_of_the_mean_hazards(average):
    mean, _ = average.mean_curves(SimpleNamespace())
    mean_hazards = _from_hazards((_hazards(ONE) + _hazards(TWO)) / 2)
    assert not np.allclose(mean.survival[0], mean_hazards)
