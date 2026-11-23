"""The definitions that results files carry say what the every-scale driver-loss bound is, as
`tests/wp11/test_driver_bound_everywhere.py` pins it: one test per definition, each linking a claim
the definition makes to the behaviour it describes, so that a definition cannot keep a claim the
code no longer makes.

Each test reads the claim from the written text (whether the text says it) and the fact from the
code (whether it holds on a made-up day), and requires the two to agree:

1. T7's `driver_term` (`t7_summary.definitions`): the text says the replayed driver loss can be
   above the term exactly when it is above it somewhere. On the hand day `EARLY` (car A leaves in
   hour 2 with 0.5 kW planned in it) hour 2 fails at every scale above 0, replay loses 1/8 of the
   protected energy on every corner, and the every-scale bound is 1/8 there: never below replay,
   on `EARLY` and on seeded made-up days at the scales of the grid and above it, at scales where
   an hour fails and where none does. The earlier bound is 0 on `EARLY`, below replay's 1/8.
2. T7's `STEPS` (written into the `days_changing` and `verdict` definitions and the summary): the
   text calls the driver term a ramp, with no jump, exactly when it has no jump. On `EARLY` it is
   0 at scale 0 and 1/8 at every scale above 0: a jump.
3. The sizing what-if's `scalar` (`size_whatif_summary.definitions`): its test is in
   `tests/learn/test_size_whatif_definition.py`, since that module imports the learn lane's
   packages.
4. The corner check's reading K4 (`corner_run.READINGS`, written into corner.json): the text says
   the bound assumes the plan is delivered in every service hour exactly when a corner of a day
   with a failing hour has a driver loss above the bound. On `EARLY` at 0.5 and 1 an hour fails
   and the corner maximum 1/8 equals the bound; none is above it.

Strings that are wording alone, with no behaviour to test: the sentence of `t7_summary`'s first
summary line (the term is the every-scale bound, not the replayed loss) and that of
`corner_summary`'s last note (the bound holds up to the errors of V3 and V7).
"""

import pytest
from corner_fake import offer_days
from test_driver_bound_everywhere import SWAP, E, replayed_driver

from fleetreserve import (
    corner_run,
    corner_search,
    dial_splits,
    gate_calibrate,
    size_scale,
    t7_stat,
    t7_summary,
)

TOLERANCE = 1e-9
MADE_UP_DAYS = 40
FAR = (1.5, 2.0, 5.0)


@pytest.fixture(scope="module")
def days():
    return [E, SWAP, *offer_days(MADE_UP_DAYS)]


def test_t7s_driver_term_definition_claims_replay_above_it_exactly_when_it_is(days):
    claims_replay_above = "can be above" in t7_summary.definitions()["driver_term"]
    failing_hour_pairs = 0
    replay_above = False
    for prepared in days:
        terms = size_scale.day_terms(prepared)
        for beta in (*dial_splits.GRID, *FAR):
            losses = t7_stat.day_losses([terms], beta)
            if losses["slot"][0] > 0:  # an hour fails at this scale
                failing_hour_pairs += 1
            if replayed_driver(prepared, beta) > losses["driver"][0] + TOLERANCE:
                replay_above = True
    assert failing_hour_pairs > 0  # the test reads scales where an hour fails
    assert not replay_above
    assert claims_replay_above == replay_above


def test_t7s_earlier_bound_would_have_been_below_replay_where_an_hour_fails():
    terms = size_scale.day_terms(E)
    losses = t7_stat.day_losses([terms], 0.5)
    assert losses["slot"][0] > 0
    assert replayed_driver(E, 0.5) == pytest.approx(1 / 8)
    assert losses["driver"][0] == pytest.approx(1 / 8)
    assert gate_calibrate.ramp_loss(terms.driver, 0.5) == 0.0


def test_t7s_steps_text_calls_the_driver_term_a_ramp_exactly_when_it_has_no_jump():
    terms = size_scale.day_terms(E)
    at_zero = t7_stat.day_losses([terms], 0.0)["driver"][0]
    just_above = t7_stat.day_losses([terms], 1e-9)["driver"][0]
    has_jump = just_above - at_zero > 0.1
    assert has_jump  # 0 at scale 0, 1/8 at every scale above it
    says_ramp_without_jump = "driver term a ramp" in t7_summary.STEPS
    assert says_ramp_without_jump == (not has_jump)


def test_the_corner_reading_says_the_bound_assumes_delivery_exactly_when_a_failing_hour_exceeds_it(
    days,
):
    claims_delivery = "assumes the plan is delivered" in corner_run.READINGS["K4"]
    failing_hour_pairs = 0
    exceeded = False
    for prepared in days[:12]:
        for check in corner_search.closed_form_check(prepared, (0.5, 1.0, *FAR)):
            if check.closed_hour is not None:
                failing_hour_pairs += 1
                if check.corner_driver > check.driver_bound + corner_search.DRIVER_TOLERANCE:
                    exceeded = True
    assert failing_hour_pairs > 0
    assert not exceeded
    assert claims_delivery == exceeded
