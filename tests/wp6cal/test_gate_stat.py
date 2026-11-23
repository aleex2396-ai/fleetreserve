"""The gate arithmetic (plan section 13, WP6): shares of the ceiling and the go rule."""

import math

import pytest

from fleetreserve import gate_stat as g


def offer(values, ceilings, beta):
    return g.unit_offer(values, ceilings, beta)


def test_unit_offer_hand_worked():
    # Three test days: certified values at scale 1 of 10, 0 (no certificate) and 30 kWh; ceilings
    # 40, 20 and 60 kWh; beta 0.5. Offers 5, 0, 15 kWh: mean 20/3; share of sums 20/120 = 1/6;
    # mean of per-day ratios (5/40 + 0 + 15/60)/3 = (0.125 + 0 + 0.25)/3 = 0.125.
    u = offer([10, 0, 30], [40, 20, 60], 0.5)
    assert list(u.offers_kwh) == [5.0, 0.0, 15.0]
    assert u.mean_offer_kwh == pytest.approx(20 / 3)
    assert u.share_of_sums == pytest.approx(1 / 6)
    assert u.mean_of_daily_ratios == pytest.approx(0.125)
    assert (u.n_days, u.n_days_positive_ceiling, u.beta) == (3, 3, 0.5)


def test_ratio_of_sums_differs_from_mean_of_ratios():
    # Days: ceiling 100 with offer 10, ceiling 10 with offer 5. Ratio of sums 15/110; mean of
    # ratios (0.1 + 0.5)/2 = 0.3. The main definition is the ratio of sums.
    u = offer([10, 5], [100, 10], 1.0)
    assert u.share_of_sums == pytest.approx(15 / 110)
    assert u.mean_of_daily_ratios == pytest.approx(0.3)


def test_days_with_zero_ceiling_are_left_out_of_the_mean_of_ratios_only():
    u = offer([4, 6, 0], [0, 12, 8], 1.0)
    assert u.n_days_positive_ceiling == 2
    assert u.mean_of_daily_ratios == pytest.approx((6 / 12 + 0 / 8) / 2)
    assert u.share_of_sums == pytest.approx(10 / 20)  # the offer on the zero-ceiling day counts


def test_zero_offer_everywhere():
    u = offer([0, 0], [10, 10], 0.7)
    assert u.mean_offer_kwh == 0.0
    assert u.share_of_sums == 0.0
    assert u.mean_of_daily_ratios == 0.0
    assert offer([5, 5], [10, 10], 0.0).share_of_sums == 0.0  # beta 0 abstains


@pytest.mark.parametrize(
    ("values", "ceilings", "beta"),
    [
        ([1, 2], [1], 0.5),
        ([], [], 0.5),
        ([1, -2], [1, 1], 0.5),
        ([1, 2], [1, -1], 0.5),
        ([1, math.nan], [1, 1], 0.5),
        ([1, 2], [1, 1], math.inf),
        ([1, 2], [1, 1], -0.1),
        ([1, 2], [0, 0], 0.5),  # share undefined
    ],
)
def test_unit_offer_refuses_bad_input(values, ceilings, beta):
    with pytest.raises(ValueError):
        g.unit_offer(values, ceilings, beta)


def gate_unit(name, cal_share, ora_share, ceiling=100.0):
    """A unit with one test day: calibrated share and oracle share of a 100 kWh ceiling."""
    return g.GateUnit(
        name,
        offer([cal_share * ceiling], [ceiling], 1.0),
        offer([ora_share * ceiling], [ceiling], 1.0),
    )


def decide(units, rule, required=2, **kwargs):
    return g.gate_decision(units, required_units=required, ratio_rule_on=rule, **kwargs)


def test_go_when_two_of_three_pass_both_rules_under_every_reading():
    units = [gate_unit("a", 0.12, 0.20), gate_unit("b", 0.15, 0.30), gate_unit("c", 0.05, 0.20)]
    # c fails the 10% rule; its ratio is 4. a: ratio 1.667, b: 2.0.
    for rule in g.RATIO_RULES:
        decision = decide(units, rule)
        assert decision.go, rule
        assert (decision.n_pass_share, decision.n_pass_both) == (2, 2)
        assert [u.name for u in decision.units] == ["a", "b", "c"]
        assert decision.units[0].ratio == pytest.approx(0.20 / 0.12)


def test_the_three_readings_of_the_ratio_rule_differ():
    # a, b pass 10%; a fails the 1.5 rule (ratio 1.2), b passes (2.0); c fails 10%, ratio 5.
    units = [gate_unit("a", 0.10, 0.12), gate_unit("b", 0.20, 0.40), gate_unit("c", 0.02, 0.10)]
    assert decide(units, "all_units").go is False  # a fails the ratio
    assert decide(units, "all_share_passing_units").go is False  # a passes 10% and fails ratio
    assert decide(units, "required_units_both").go is False  # only b passes both
    # Now only c's ratio is bad, and c fails the 10% rule too:
    units = [gate_unit("a", 0.12, 0.30), gate_unit("b", 0.20, 0.40), gate_unit("c", 0.02, 0.025)]
    assert decide(units, "all_units").go is False  # c's ratio is 1.25
    assert decide(units, "all_share_passing_units").go is True  # c is not a share-passing unit
    assert decide(units, "required_units_both").go is True
    # Three units pass 10%, only two the ratio:
    units = [gate_unit("a", 0.12, 0.30), gate_unit("b", 0.20, 0.40), gate_unit("c", 0.15, 0.16)]
    assert decide(units, "all_units").go is False
    assert decide(units, "all_share_passing_units").go is False  # c passes 10% but not the ratio
    assert decide(units, "required_units_both").go is True  # a and b pass both


def test_no_go_when_fewer_than_required_pass_the_share():
    units = [gate_unit("a", 0.12, 0.30), gate_unit("b", 0.09, 0.30), gate_unit("c", 0.05, 0.30)]
    for rule in g.RATIO_RULES:
        decision = decide(units, rule)
        assert not decision.go
        assert decision.n_pass_share == 1
    # with one required unit the same numbers are go: the ratios are 2.5, 3.3 and 6 everywhere
    assert decide(units, "all_units", required=1).go is True
    assert decide(units, "all_share_passing_units", required=1).go is True


def test_boundaries_ten_percent_and_one_and_a_half_pass():
    # share exactly 0.10 and ratio exactly 1.5. 0.3 / 3 is 0.09999999999999999 in floats.
    exact = g.GateUnit("a", offer([0.3], [3.0], 1.0), offer([0.45], [3.0], 1.0))
    assert exact.calibrated.share_of_sums < 0.10  # the float artefact is real
    row = decide([exact], "all_units", required=1).units[0]
    assert row.passes_share and row.passes_ratio
    assert decide([exact], "all_units", required=1).go
    # just under: 9.9% fails
    under = gate_unit("a", 0.099, 0.30)
    assert not decide([under], "all_units", required=1).units[0].passes_share
    # just under 1.5: ratio 1.49 fails
    under_ratio = gate_unit("a", 0.20, 0.298)
    assert not decide([under_ratio], "all_units", required=1).units[0].passes_ratio


def test_zero_calibrated_offer_has_no_ratio_and_fails_both_rules():
    zero = gate_unit("z", 0.0, 0.30)
    row = decide([zero], "all_units", required=1).units[0]
    assert row.ratio is None  # undefined, not infinity
    assert not row.passes_share
    assert not row.passes_ratio
    assert not row.passes_both
    assert row.oracle_share == pytest.approx(0.30)
    # two good units and one zero unit: go when the zero unit is outside the ratio rule ...
    units = [gate_unit("a", 0.12, 0.30), gate_unit("b", 0.20, 0.40), zero]
    assert decide(units, "all_share_passing_units").go
    assert decide(units, "required_units_both").go
    # ... but not when every gate unit must pass the ratio rule (reading G4)
    assert not decide(units, "all_units").go


def test_both_zero_offers():
    units = [gate_unit("a", 0.0, 0.0), gate_unit("b", 0.0, 0.0)]
    decision = decide(units, "all_units", required=1)
    assert not decision.go
    assert all(u.ratio is None for u in decision.units)


def test_mean_of_ratios_basis():
    # Unit: ceilings 100 and 10; calibrated offers 10 and 5: ratio of sums 15/110 = 0.136,
    # mean of ratios 0.3. The oracle offers are twice as large.
    u = g.GateUnit("a", offer([10, 5], [100, 10], 1.0), offer([20, 10], [100, 10], 1.0))
    sums = decide([u], "all_units", required=1)
    ratios = decide([u], "all_units", required=1, share_basis="mean_of_ratios")
    assert sums.units[0].calibrated_share == pytest.approx(15 / 110)
    assert ratios.units[0].calibrated_share == pytest.approx(0.3)
    assert sums.go and ratios.go
    # at a threshold of 0.14 the two bases differ
    assert not decide([u], "all_units", required=1, share_threshold=0.14).go
    assert decide(
        [u], "all_units", required=1, share_threshold=0.14, share_basis="mean_of_ratios"
    ).go


def test_returns_every_units_numbers():
    units = [gate_unit("a", 0.12, 0.24), gate_unit("b", 0.05, 0.06)]
    decision = decide(units, "all_units")
    a, b = decision.units
    assert (a.calibrated_mean_kwh, a.oracle_mean_kwh) == pytest.approx((12.0, 24.0))
    assert (b.calibrated_share, b.oracle_share) == pytest.approx((0.05, 0.06))
    assert a.ratio == pytest.approx(2.0)
    assert b.ratio == pytest.approx(1.2)
    assert decision.required_units == 2 and decision.ratio_rule_on == "all_units"


def test_gate_refuses_bad_arguments():
    units = [gate_unit("a", 0.12, 0.30), gate_unit("b", 0.12, 0.30)]
    with pytest.raises(ValueError, match="ratio_rule_on"):
        decide(units, "some")
    with pytest.raises(ValueError, match="required_units"):
        decide(units, "all_units", required=3)
    with pytest.raises(ValueError, match="required_units"):
        decide(units, "all_units", required=0)
    with pytest.raises(ValueError, match="share_basis"):
        decide(units, "all_units", share_basis="x")
    with pytest.raises(ValueError, match="no gate units"):
        decide([], "all_units", required=1)
    with pytest.raises(ValueError, match="twice"):
        decide([units[0], units[0]], "all_units")
    mismatched = g.GateUnit("m", offer([1, 1], [10, 10], 1.0), offer([1], [10], 1.0))
    with pytest.raises(ValueError, match="different numbers of days"):
        decide([mismatched], "all_units", required=1)
    with pytest.raises(TypeError):
        g.gate_decision(units)  # the number of units and the reading are not defaulted
