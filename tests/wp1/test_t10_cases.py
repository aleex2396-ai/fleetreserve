"""Plan section 6, T10 drift guard: the two stated cases and the stated bound.

Case: from beta_1 = 0, any beta > 0 loses the day (loss 1) and beta <= 0 loses nothing.
Clipped at 0 the average loss is 0.500 for every step size; unclipped it is 0.103 at t = 116.
Exact fractions are used so that "beta = 0" is exact.
"""

from fractions import Fraction

import pytest

from fleetreserve.drift_guard import run

ALPHA = Fraction(1, 10)
ETAS = [Fraction(1, 100), Fraction(1, 10), Fraction(1, 2), Fraction(1), Fraction(7, 2)]


def _lose_if_positive(beta):
    return 1 if beta > 0 else 0


def _average(losses):
    return Fraction(sum(losses), len(losses))


@pytest.mark.parametrize("eta", ETAS)
def test_clipped_average_is_one_half(eta):
    losses = run(_lose_if_positive, 116, ALPHA, eta, clip=True)
    assert round(float(_average(losses)), 3) == 0.500


@pytest.mark.parametrize("eta", ETAS)
def test_unclipped_average_at_day_116(eta):
    losses = run(_lose_if_positive, 116, ALPHA, eta, clip=False)
    assert round(float(_average(losses)), 3) == 0.103


@pytest.mark.parametrize("eta", ETAS)
def test_unclipped_average_obeys_the_bound(eta):
    beta_1 = Fraction(0)
    losses = run(_lose_if_positive, 400, ALPHA, eta, beta_1=beta_1, clip=False)
    for t in range(1, len(losses) + 1):
        bound = ALPHA + (beta_1 / eta + 1 - ALPHA) / t
        assert _average(losses[:t]) <= bound
