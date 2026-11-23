"""Plan section 4, T3 unit test: one car, two hours, rate 1, energy in [1, 1.5].

Box centre (0.6, 0.6), half-widths lam * (1, 1). The largest lam such that the request is
met on a set of corner paths, each path solved with the future known, is found by one linear
programme: variables lam and, for each path, the car's power in each hour. The car's set is
0 <= u_t <= rate and lower <= u_1 + u_2 <= upper (plan section 2).
"""

import itertools

import numpy as np
import pytest

from fleetreserve.lp import OTHER_SECONDS, solve_lp

CENTRE = (0.6, 0.6)
RATE = 1.0
ENERGY = (1.0, 1.5)
ALL_PATHS = list(itertools.product((-1, 1), repeat=2))  # -1: lower corner, +1: upper corner


def _largest_lam(paths):
    """Largest lam with the request met on every given path. Variables: lam, then u per path."""
    n = 1 + 2 * len(paths)
    cost = np.zeros(n)
    cost[0] = -1.0  # maximise lam
    a_eq = np.zeros((2 * len(paths), n))
    b_eq = np.zeros(2 * len(paths))
    a_ub = np.zeros((2 * len(paths), n))
    b_ub = np.zeros(2 * len(paths))
    for p, path in enumerate(paths):
        for t in range(2):
            row = 2 * p + t
            a_eq[row, 1 + row] = 1.0  # u_t - lam * zeta_t = centre_t
            a_eq[row, 0] = -path[t]
            b_eq[row] = CENTRE[t]
        a_ub[2 * p, 1 + 2 * p : 3 + 2 * p] = 1.0  # u_1 + u_2 <= upper
        b_ub[2 * p] = ENERGY[1]
        a_ub[2 * p + 1, 1 + 2 * p : 3 + 2 * p] = -1.0  # -(u_1 + u_2) <= -lower
        b_ub[2 * p + 1] = -ENERGY[0]
    bounds = [(0.0, 1.0)] + [(0.0, RATE)] * (n - 1)
    result = solve_lp(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        time_limit=OTHER_SECONDS,
        method="simplex",
    ).require_optimal()
    return -result.objective


def test_hindsight_is_0_10():
    assert _largest_lam(ALL_PATHS) == pytest.approx(0.10, abs=1e-6)


def test_upper_corner_path_alone_is_0_15():
    assert _largest_lam([(1, 1)]) == pytest.approx(0.15, abs=1e-6)
