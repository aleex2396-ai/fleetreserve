"""Plan section 5, T2(c): with every car's forecast conservative the fixed plan never fails.

Sample: `N_DAYS` seeded made-up days (`made_up_offer_days`, conservative forecasts only), each
certified on its forecast sets; at scales 0.5 and 1 (the certified box and a smaller one), on all
`2^n` corner paths: the fixed plan meets every request (no failure, slot loss 0), every car ends
with at least its protected energy (driver loss 0), no car is ever capped (it draws exactly its
plan), and the re-solve rungs never re-solve and draw exactly what the fixed plan draws. Days whose
certificate is infeasible (an empty forecast set) are counted: `N_CERTIFIED` of `N_DAYS` are
certified, giving `N_DAY_PATHS` day-paths. The same holds under both readings of when a rung
learns of a departure (`rungs.py`, information model).
"""

import random

import numpy as np
import pytest
from made_up_offer_days import departure_rule, offer_day

from fleetreserve.products import corner_paths
from fleetreserve.replay import protected
from fleetreserve.rungs import DEPARTURE_SEEN, run_rungs

SEED = 11
N_DAYS = 30
N_CERTIFIED = 29
N_DAY_PATHS = 648
BETAS = (0.5, 1.0)
TOLERANCE = 1e-7


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_conservative_forecasts_fixed_plan_never_fails_and_no_driver_loses_energy(seen):
    rng = random.Random(SEED)
    certified = day_paths = 0
    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k, ("conservative",))
        if certificate.outcome != "optimal":
            assert certificate.outcome == "infeasible"
            continue
        certified += 1
        need = protected(day)
        for beta in BETAS:
            for path in corner_paths(day.n_service):
                runs = run_rungs(
                    day,
                    certificate.policy,
                    beta,
                    path,
                    departure_bound=departure_rule(day, seen),
                    departure_seen=seen,
                )
                fixed = runs["fixed"].run
                assert fixed.first_failure is None and fixed.slot_loss == 0.0
                np.testing.assert_allclose(fixed.draws, fixed.planned, atol=TOLERANCE)
                assert np.all(fixed.final >= need - TOLERANCE)
                assert fixed.driver_loss <= TOLERANCE
                for name in ("one-way", "two-way"):
                    assert runs[name].resolves == ()
                    np.testing.assert_array_equal(runs[name].run.draws, fixed.draws)
                day_paths += 1
    print(f"{certified} of {N_DAYS} days certified, {day_paths} day-paths")
    assert (certified, day_paths) == (N_CERTIFIED, N_DAY_PATHS)
