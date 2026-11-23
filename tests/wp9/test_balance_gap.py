"""The balance gap of a policy (`balance_gap`): its definitions, its exactness and its counts.

1. Cars that share the box out exactly. Three cars, two service hours, box centre (1.5, 2.25),
   widths (0.5, 0.75): planned powers (0.5, 0.25, 0.75) in hour 1 and (1, 0.75, 0.5) in hour 2,
   gains on the request of hour 1 (0.25, 0.125, 0.125) in hour 1 and (0.5, -0.25, -0.25) in hour
   2, on the request of hour 2 (0.25, 0.25, 0.25) in hour 2. Every number is a sum of powers of
   2, so the shares add up exactly: both gaps 0 in both hours, with no rounding allowed, since the
   gaps are computed exactly. Two cars planning 0.1 and 0.2 kW under a centre that is their sum
   in floating point (0.30000000000000004) have a centre gap of exactly 2^-55 kWh (the double
   0.30000000000000004 less the exact sum of the doubles 0.1 and 0.2), where the sum in floating
   point reads 0.
2. A hand-built policy of ten service hours and two cars with known gaps. Car 1 holds the box:
   in hour t it plans the centre m_t = 1 and has the gain w_t = 0.5 on the request of hour t.
   Car 2 holds the gaps: its planned power is the centre gap c_t and its gain on the request of
   hour t the diagonal part d_t of the gain gap (exactly, since 0.5 + d less 0.5 is d in exact
   arithmetic). The hours, with the gain gap gamma_t:
     1: gamma = 9.999999999999998e-13, the double below ZERO_GAIN (1e-12): above no level;
     2: gamma = 1.0000000000000002e-12, the double above ZERO_GAIN: above ZERO_GAIN only;
     3: gamma = 9.999999999999999e-10, the largest double not above FAILURE_KWH / VALIDITY_LIMIT
        (that exact quotient is 1e-9 + 4.79e-26): above ZERO_GAIN only; at the limit B = 1e5
        gamma = 9.999999999999999e-5, not above FAILURE_KWH;
     4: gamma = 1e-9, the double nearest 1e-9, above the quotient by 1.44e-26: above ZERO_GAIN and
        the quotient; at the limit B is above FAILURE_KWH by 1.44e-21;
     5: gamma = 9.999999999999998e-08, the double below ROW_TOLERANCE (1e-7): above two levels;
     6: gamma = 1.0000000000000001e-07, the double above ROW_TOLERANCE: above all three;
     7: c = 1.0000000000000002e-04, the double above FAILURE_KWH, gamma 0: B above FAILURE_KWH at
        every scale;
     8: c = 1e-4 = FAILURE_KWH, gamma 0: B never above FAILURE_KWH;
     9: an off-diagonal part: car 2 has the gain 3e-8 on the request of hour 3 and -4e-8 on the
        request of hour 9, so gamma = |3e-8| + |-4e-8| = 7e-8 exactly (the sum of those doubles):
        above ZERO_GAIN and the quotient, not ROW_TOLERANCE;
    10: gains on the request of hour 1 of 0.25 and -0.25 that cancel: shared out exactly.
   So: 10 hours, 1 shared exactly; gain gaps above ZERO_GAIN in hours 2, 3, 4, 5, 6, 9 (6), above
   the quotient in 4, 5, 6, 9 (4), above ROW_TOLERANCE in 6 (1); B above FAILURE_KWH at scale 1 in
   hour 7 only (1), at the validity limit in 4, 5, 6, 7, 9 (5), at a named scale 0.5 in 7 only;
   the largest centre gap that of hour 7, the largest gain gap that of hour 6, the largest B at
   the limit hour 6's 1e5 x 1.0000000000000001e-07, rounded once.
3. Ties at the levels. A gain gap exactly equal to a level is not above it (the counts compare with
   "above"): gains of exactly ZERO_GAIN and exactly ROW_TOLERANCE on the request of the hour, and
   a gap of exactly FAILURE_KWH / VALIDITY_LIMIT (an exact fraction, held by `HourGap` directly,
   since no sum of doubles is that quotient), whose `B_t` at the validity limit is exactly
   FAILURE_KWH and cannot fail the hour; a gap one step above each is counted.
4. On seeded made-up policies (up to 6 cars and 4 hours, causal gains), `B_t` at 0, 1, 0.37 and
   the validity limit equals the exact `B_t` of the tests of V1 to V8 (`validity_errors`,
   `tests/wp11`), written independently.
"""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from fleetreserve import balance_gap
from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN, AffinePolicy
from fleetreserve.replay import FAILURE_KWH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp11"))
from validity_errors import balance_gaps  # noqa: E402

QUOTIENT = Fraction(FAILURE_KWH) / Fraction(VALIDITY_LIMIT)
BELOW_QUOTIENT = 9.999999999999999e-10
ABOVE_QUOTIENT = 1e-9


def test_the_levels_are_the_doubles_named_in_the_text():
    assert QUOTIENT - Fraction(1, 10**9) > 0
    assert float(QUOTIENT - Fraction(1, 10**9)) == 4.79217360238593e-26
    assert Fraction(ABOVE_QUOTIENT) > QUOTIENT >= Fraction(BELOW_QUOTIENT)
    assert np.nextafter(BELOW_QUOTIENT, 1.0) == ABOVE_QUOTIENT
    levels = dict(balance_gap.GAIN_LEVELS)
    assert levels == {
        "zero_gain": Fraction(ZERO_GAIN),
        "failure_over_limit": QUOTIENT,
        "row_tolerance": Fraction(ROW_TOLERANCE),
    }


def _exact_share() -> AffinePolicy:
    g = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75], [0.0, 0.75, 0.5]])
    G = np.zeros((3, 3, 2))
    G[:, 1, 0] = (0.25, 0.125, 0.125)
    G[:, 2, 0] = (0.5, -0.25, -0.25)
    G[:, 2, 1] = (0.25, 0.25, 0.25)
    return AffinePolicy([1.5, 2.25], [0.5, 0.75], g, G)


def test_cars_that_share_the_box_exactly_have_every_gap_0():
    gaps = balance_gap.hour_gaps(_exact_share())
    assert [(g.hour, g.centre, g.gain) for g in gaps] == [(1, 0, 0), (2, 0, 0)]
    found = balance_gap.tally(gaps, {})
    assert found["hours"] == 2 and found["hours_shared_exactly"] == 2
    assert found["largest_centre_gap_kwh"] == 0.0 and found["largest_gain_gap_kwh"] == 0.0
    assert found["gain_gap_above"] == dict.fromkeys(balance_gap.GAIN_LEVELS, 0)
    for name in (balance_gap.SCALE_1, balance_gap.AT_LIMIT):
        assert found["at_scales"][name]["largest_b_kwh"] == 0.0
        assert found["at_scales"][name]["hours_can_fail_alone"] == 0


def test_a_gap_a_floating_point_sum_reads_as_0_is_measured_exactly():
    centre = 0.1 + 0.2
    assert 0.1 + 0.2 - centre == 0.0
    policy = AffinePolicy([centre], [0.0], [[0.0, 0.1], [0.0, 0.2]], np.zeros((2, 2, 1)))
    (gap,) = balance_gap.hour_gaps(policy)
    assert gap.centre == Fraction(1, 2**55) and gap.gain == 0
    assert balance_gap.tally([gap], {})["largest_centre_gap_kwh"] == 2.0**-55


CENTRES = {7: float(np.nextafter(FAILURE_KWH, 1.0)), 8: FAILURE_KWH}
DIAGONAL = {
    1: float(np.nextafter(ZERO_GAIN, 0.0)),
    2: float(np.nextafter(ZERO_GAIN, 1.0)),
    3: BELOW_QUOTIENT,
    4: ABOVE_QUOTIENT,
    5: float(np.nextafter(ROW_TOLERANCE, 0.0)),
    6: float(np.nextafter(ROW_TOLERANCE, 1.0)),
    9: -4e-8,
}


def _known() -> AffinePolicy:
    n = 10
    g = np.zeros((2, n + 1))
    G = np.zeros((2, n + 1, n))
    g[0, 1:] = 1.0
    for t in range(1, n + 1):
        G[0, t, t - 1] = 0.5
        g[1, t] = CENTRES.get(t, 0.0)
        G[1, t, t - 1] = DIAGONAL.get(t, 0.0)
    G[1, 9, 2] = 3e-8
    G[0, 10, 0], G[1, 10, 0] = 0.25, -0.25
    return AffinePolicy(np.ones(n), np.full(n, 0.5), g, G)


def test_a_hand_built_policy_has_its_known_gaps_and_counts():
    gaps = balance_gap.hour_gaps(_known())
    assert [g.hour for g in gaps] == list(range(1, 11))
    expected_gain = {t: abs(Fraction(d)) for t, d in DIAGONAL.items()}
    expected_gain[9] = Fraction(3e-8) + Fraction(4e-8)
    for g in gaps:
        assert g.centre == Fraction(CENTRES.get(g.hour, 0.0)), g
        assert g.gain == expected_gain.get(g.hour, Fraction(0)), g
    found = balance_gap.tally(gaps, {"made_up": 0.5, "absent": None})
    assert found["hours"] == 10 and found["hours_shared_exactly"] == 1
    assert found["largest_centre_gap_kwh"] == CENTRES[7]
    assert found["largest_gain_gap_kwh"] == DIAGONAL[6]
    assert found["gain_gap_above"] == {"zero_gain": 6, "failure_over_limit": 4, "row_tolerance": 1}
    scales = found["at_scales"]
    assert set(scales) == {"scale_1", "validity_limit", "made_up"}
    assert scales["scale_1"] == {
        "scale": 1.0,
        "largest_b_kwh": CENTRES[7],
        "hours_can_fail_alone": 1,
    }
    at_limit = float(Fraction(VALIDITY_LIMIT) * Fraction(DIAGONAL[6]))
    assert scales["validity_limit"] == {
        "scale": VALIDITY_LIMIT,
        "largest_b_kwh": at_limit,
        "hours_can_fail_alone": 5,
    }
    assert scales["made_up"]["hours_can_fail_alone"] == 1
    failing = [g.hour for g in gaps if g.can_fail_alone(VALIDITY_LIMIT)]
    assert failing == [4, 5, 6, 7, 9]
    assert [g.hour for g in gaps if g.can_fail_alone(1.0)] == [7]
    assert float(gaps[3].miss(VALIDITY_LIMIT) - Fraction(FAILURE_KWH)) == 1.435985543392056e-21


def _made_up(rng) -> AffinePolicy:
    cars, n = int(rng.integers(1, 7)), int(rng.integers(1, 5))
    g = rng.normal(size=(cars, n + 1)) * 10.0 ** rng.integers(-3, 3)
    G = np.zeros((cars, n + 1, n))
    for t in range(1, n + 1):
        G[:, t, :t] = rng.normal(size=(cars, t))
    m = g[:, 1:].sum(axis=0) + rng.normal(size=n) * 10.0 ** rng.integers(-16, -6)
    w = np.abs(np.einsum("its->ts", G[:, 1:, :]).diagonal()) + rng.normal(size=n) * 1e-12
    return AffinePolicy(m, np.abs(w), g, G)


def test_b_equals_the_exact_b_of_the_validity_tests_on_made_up_policies():
    rng = np.random.default_rng(20261117)
    for _ in range(200):
        policy = _made_up(rng)
        gaps = balance_gap.hour_gaps(policy)
        for beta in (0.0, 1.0, 0.37, VALIDITY_LIMIT):
            assert [g.miss(beta) for g in gaps] == balance_gaps(policy, beta)


def _gain_gap_policy(gaps) -> AffinePolicy:
    """One hour per entry of `gaps`: car 1 holds the box (centre 1, gain 0.5 on its own request),
    car 2 has the gain `gap` on it, so the gain gap of the hour is exactly `gap`."""
    n = len(gaps)
    g = np.zeros((2, n + 1))
    G = np.zeros((2, n + 1, n))
    g[0, 1:] = 1.0
    for t, gap in enumerate(gaps, start=1):
        G[0, t, t - 1] = 0.5
        G[1, t, t - 1] = gap
    return AffinePolicy(np.ones(n), np.full(n, 0.5), g, G)


def test_a_gain_gap_exactly_at_a_level_is_not_above_it():
    at = _gain_gap_policy([ZERO_GAIN, ROW_TOLERANCE, float(np.nextafter(ROW_TOLERANCE, 1.0))])
    gaps = balance_gap.hour_gaps(at)
    assert [g.gain for g in gaps] == [
        Fraction(ZERO_GAIN),
        Fraction(ROW_TOLERANCE),
        Fraction(float(np.nextafter(ROW_TOLERANCE, 1.0))),
    ]
    assert balance_gap.tally(gaps[:1], {})["gain_gap_above"] == {
        "zero_gain": 0,
        "failure_over_limit": 0,
        "row_tolerance": 0,
    }
    assert balance_gap.tally(gaps[1:2], {})["gain_gap_above"] == {
        "zero_gain": 1,
        "failure_over_limit": 1,
        "row_tolerance": 0,
    }
    assert balance_gap.tally(gaps[2:], {})["gain_gap_above"] == {
        "zero_gain": 1,
        "failure_over_limit": 1,
        "row_tolerance": 1,
    }
    assert balance_gap.tally(gaps, {})["gain_gap_above"] == {
        "zero_gain": 2,
        "failure_over_limit": 2,
        "row_tolerance": 1,
    }


def test_a_gain_gap_exactly_the_quotient_is_not_above_it_and_cannot_fail_the_limit_alone():
    tie = balance_gap.HourGap(1, Fraction(0), QUOTIENT)
    assert tie.miss(VALIDITY_LIMIT) == Fraction(FAILURE_KWH)
    assert not tie.can_fail_alone(VALIDITY_LIMIT)
    found = balance_gap.tally([tie], {})
    assert found["gain_gap_above"]["failure_over_limit"] == 0
    assert found["gain_gap_above"]["zero_gain"] == 1
    assert found["at_scales"]["validity_limit"]["hours_can_fail_alone"] == 0
    above = balance_gap.HourGap(1, Fraction(0), QUOTIENT + Fraction(1, 10**30))
    found = balance_gap.tally([above], {})
    assert found["gain_gap_above"]["failure_over_limit"] == 1
    assert found["at_scales"]["validity_limit"]["hours_can_fail_alone"] == 1
