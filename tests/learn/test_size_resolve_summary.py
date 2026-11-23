"""The figures of the re-solve what-if's results (`size_resolve_summary`), on hand-made test scores
(no day is replayed), each small enough to compute by hand.

1. Two splits; `constant` offers 1 and 2 kWh per counting test day, the sizing 1.5 and 2.5: the
   paired difference is 0.5 kWh on both, so its mean, p5 and p95 are 0.5 kWh and 1/3 of the
   constant's mean offer (1.5 kWh). The driver figure is the replayed loss, named as such.
2. The abstained share: on one split `constant` offers on 4 test days and the sizing on 3, on the
   other on 2 and 2: the mean share is 0.125; a split where `constant` offers on none is left out.
3. Pooled over two units at one seed: the offer is the mean over all their counting test days.
4. The summary's first line says it is a what-if, its third names the re-solve route, and each
   oracle line carries its note.
5. A learned sizing beside its shuffled control, by hand: on two splits `constant` offers 1 and
   1 kWh, `net-0.1-scale` 4 and 6. With ONE draw of its control `shuffled-net-0.1-scale`, 3 and
   3: the paired differences are 1 and 3 kWh, so over splits mean 2, p5 1.1 and p95 2.9 (numpy's
   linear interpolation), over the control's mean offer 3: 2/3, 1.1/3 and 2.9/3; over draws the
   one difference 5 - 3 = 2 is both p5 and p95. With TWO draws, 3 and 3 then 2 and 6: the
   control's offer per split is their mean, 2.5 and 4.5 (mean 3.5), the differences 1.5 and 1.5,
   so over splits mean, p5 and p95 1.5 and share 3/7; over draws the differences of the mean
   offers are 5 - 3 = 2 and 5 - 4 = 1: p5 1.05 and p95 1.95, shares 0.3 and 1.95/3.5. The
   difference from `constant` is unchanged (mean 4), `constant` has no control, a sizing whose
   control is missing on one unit of a pool has none in the pool, and the control is no sizing
   of the tiers. In the summary the line after the sizing's gives both spreads by name, each unit
   with controls counts the pairs its controls added, and the definition line says why the
   control is the baseline and which labelled days are moved.
6. Controls that failed on a unit are listed in the results and on a line of the summary of
   their own, apart from the failed units.
"""

import pytest

from fleetreserve import size_resolve_summary, size_scale
from fleetreserve.size_scale import Scalar


def score(offers, offering=None, slot=(), driver=()):
    days = len([o for o in offers if o > 0]) if offering is None else offering
    return size_scale.TestScore(tuple(offers), days, tuple(slot), tuple(driver))


def test_the_paired_difference_and_its_share():
    constant = {1: [score([1.0, 1.0], slot=(0.0, 0.1), driver=(0.0, 0.2))], 2: [score([2.0])]}
    sized = {1: [score([1.5, 1.5], slot=(0.0, 0.1), driver=(0.0, 0.2))], 2: [score([2.5])]}
    found = size_resolve_summary.tier_figures(sized, constant, [Scalar(0.5, 0.5, "", 9)] * 2)
    diff = found["difference_from_constant"]
    assert found["mean_offer_kwh_per_counting_test_day"] == pytest.approx(2.0)
    assert diff["mean_kwh"] == pytest.approx(0.5) and diff["p5_kwh"] == pytest.approx(0.5)
    assert diff["p95_kwh"] == pytest.approx(0.5)
    assert diff["mean_share"] == pytest.approx(1 / 3) and diff["p95_share"] == pytest.approx(1 / 3)
    assert found["mean_test_driver_loss"] == pytest.approx(0.1)
    assert found["mean_test_slot_loss"] == pytest.approx(0.05)
    assert "mean_test_driver_loss_bound" not in found and "splits_scalar_capped" not in found
    assert found["mean_scalar_used"] == 0.5 and found["splits_that_offer"] == 2


def test_the_abstained_share():
    constant = {1: [score([1.0] * 4)], 2: [score([1.0] * 2)], 3: [score([0.0] * 3, offering=0)]}
    sized = {1: [score([1.0] * 3 + [0.0])], 2: [score([1.0] * 2)], 3: [score([0.0] * 3, 0)]}
    found = size_resolve_summary.tier_figures(sized, constant)
    assert found["mean_share_of_offering_test_days_abstained"] == pytest.approx(0.125)


def test_a_pooled_split_is_the_mean_over_every_counting_test_day():
    scalar = Scalar(1.0, 1.0, "", 9)
    measured = {
        "u1": {"constant": {"A": [(1, scalar, score([3.0]))], "B": [(1, scalar, score([3.0]))]}},
        "u2": {
            "constant": {
                "A": [(1, scalar, score([1.0, 1.0, 1.0]))],
                "B": [(1, scalar, score([1.0, 1.0, 1.0]))],
            }
        },
    }
    found = size_resolve_summary._tiers(measured, ["u1", "u2"])
    for tier in ("A", "B"):
        figures = found[tier]["constant"]
        assert figures["mean_offer_kwh_per_counting_test_day"] == pytest.approx(1.5)
        assert "mean_scalar_used" not in figures  # given per unit only
    alone = size_resolve_summary._tiers(measured, ["u1"])["A"]["constant"]
    assert alone["mean_offer_kwh_per_counting_test_day"] == 3.0 and alone["mean_scalar_used"] == 1


def test_the_summary_says_what_it_is():
    entry = {
        "site_type": "workplace",
        "status": "calibratable",
        "labels": {
            "counting_days": 2,
            "offering_days": 2,
            "days_not_offering": 0,
            "certificate_timeouts": 0,
            "share_zero": 0.0,
            "share_at_or_above_one": 1.0,
            "share_centre_outside": 0.0,
            "p10": 1.0,
            "p50": 1.0,
            "p90": 1.0,
        },
        "pinball": {},
        "evaluations": {"day_scale_pairs": 20, "re_solves_of_the_one_way_rung": {"optimal": 3}},
        "tiers": {
            "A": {
                name: size_resolve_summary.tier_figures({1: [score([1.0])]}, {1: [score([1.0])]})
                for name in ("constant", "oracle-resolve-scale")
            }
        },
    }
    document = {
        "label": size_resolve_summary.WHAT,
        "definitions": size_resolve_summary.definitions(),
        "units": {"u": entry},
        "pooled_by_site_type": {},
        "failed_units": [],
        "run": {"model": "m", "splits": 1},
    }
    lines = size_resolve_summary.summary_lines(document)
    assert lines[0] == size_resolve_summary.WHAT and "no verdict is formed" in lines[1]
    assert size_resolve_summary.ROUTE in lines[2]
    assert "evaluations: 20 (day, scale) pairs" in "\n".join(lines)
    oracle = [line for line in lines if line.strip().startswith("oracle-")]
    assert oracle and all(size_resolve_summary.ORACLE_NOTE in line for line in oracle)


# --- 5. ----------------------------------------------------------------------------------------


def two_splits(offers) -> dict:
    scalar = Scalar(1.0, 1.0, "", 9)
    return {
        tier: [(seed, scalar, score([o])) for seed, o in zip((1, 2), offers, strict=True)]
        for tier in size_scale.TIERS
    }


MEASURED = {"u": {"constant": two_splits([1.0, 1.0]), "net-0.1-scale": two_splits([4.0, 6.0])}}


def controls(*draws) -> dict:
    """Unit "u"'s control of `net-0.1-scale`, one offer pair per draw."""
    return {"u": {"shuffled-net-0.1-scale": [two_splits(offers) for offers in draws]}}


def test_a_learned_sizing_against_one_draw_of_its_shuffled_control_by_hand():
    found = size_resolve_summary._tiers(MEASURED, ["u"], controls([3.0, 3.0]))
    for tier in size_scale.TIERS:
        diff = found[tier]["net-0.1-scale"]["difference_from_shuffled"]
        assert diff["draws"] == 1
        assert diff["control_mean_offer_kwh_per_counting_test_day"] == pytest.approx(3.0)
        assert diff["mean_kwh"] == pytest.approx(2.0)
        assert diff["p5_kwh"] == pytest.approx(1.1) and diff["p95_kwh"] == pytest.approx(2.9)
        assert diff["mean_share"] == pytest.approx(2 / 3)
        assert diff["p5_share"] == pytest.approx(1.1 / 3)
        assert diff["p95_share"] == pytest.approx(2.9 / 3)
        assert diff["over_draws"] == {
            "p5_kwh": pytest.approx(2.0),
            "p95_kwh": pytest.approx(2.0),
            "p5_share": pytest.approx(2 / 3),
            "p95_share": pytest.approx(2 / 3),
        }
        assert found[tier]["net-0.1-scale"]["difference_from_constant"]["mean_kwh"] == 4.0
        assert "difference_from_shuffled" not in found[tier]["constant"]
        assert set(found[tier]) == {"constant", "net-0.1-scale"}


def test_a_learned_sizing_against_two_draws_of_its_shuffled_control_by_hand():
    found = size_resolve_summary._tiers(MEASURED, ["u"], controls([3.0, 3.0], [2.0, 6.0]))
    for tier in size_scale.TIERS:
        diff = found[tier]["net-0.1-scale"]["difference_from_shuffled"]
        assert diff["draws"] == 2
        assert diff["control_mean_offer_kwh_per_counting_test_day"] == pytest.approx(3.5)
        assert diff["mean_kwh"] == pytest.approx(1.5)
        assert diff["p5_kwh"] == pytest.approx(1.5) and diff["p95_kwh"] == pytest.approx(1.5)
        assert diff["mean_share"] == pytest.approx(3 / 7)
        assert diff["p95_share"] == pytest.approx(3 / 7)
        assert diff["over_draws"] == {
            "p5_kwh": pytest.approx(1.05),
            "p95_kwh": pytest.approx(1.95),
            "p5_share": pytest.approx(0.3),
            "p95_share": pytest.approx(1.95 / 3.5),
        }


def test_a_pool_reads_a_control_only_where_every_unit_has_it():
    measured = {**MEASURED, "v": MEASURED["u"]}
    both = {**controls([3.0, 3.0]), "v": controls([3.0, 3.0])["u"]}
    found = size_resolve_summary._tiers(measured, ["u", "v"], both)
    assert found["A"]["net-0.1-scale"]["difference_from_shuffled"]["mean_kwh"] == pytest.approx(2)
    found = size_resolve_summary._tiers(measured, ["u", "v"], controls([3.0, 3.0]))
    assert "difference_from_shuffled" not in found["A"]["net-0.1-scale"]


def test_the_summary_gives_the_control_beside_the_sizing():
    found = size_resolve_summary._tiers(MEASURED, ["u"], controls([3.0, 3.0], [2.0, 6.0]))
    lines = size_resolve_summary._tier_lines(found)
    at = next(i for i, line in enumerate(lines) if line.startswith("    net-0.1-scale:"))
    assert lines[at + 1] == (
        "      against its shuffled control shuffled-net-0.1-scale, mean of 2 draws (offer 3.500 "
        "kWh): 1.500 kWh; spread over splits p5 1.500, p95 1.500; spread over draws p5 1.050, "
        "p95 1.950; share of the control's offer 0.429 (over splits p5 0.429, p95 0.429; over "
        "draws p5 0.300, p95 0.557)"
    )
    assert sum("against its shuffled control" in line for line in lines) == 2  # one per tier
    definition = size_resolve_summary.definitions()["shuffled_control"]
    assert "any spread of sizes" in definition and "grid" in definition
    assert "labelled days that offer under constant" in definition
    assert "over_draws" in definition and "over splits" in definition
    counts = {"u": {"day_scale_pairs": 7, "re_solves_of_the_one_way_rung": {"optimal": 5}}}
    assert size_resolve_summary.control_lines(counts) == [
        f"shuffled controls: {definition}",
        "u: the shuffled controls added 7 (day, scale) pairs; one-way re-solves {'optimal': 5}",
    ]


# --- 6. ----------------------------------------------------------------------------------------


def test_failed_controls_are_listed_apart_from_failed_units():
    document = size_resolve_summary.results(
        {}, {}, {}, {}, {}, {"w": "ValueError: no"}, failed_controls={"u": "RuntimeError: x"}
    )
    assert document["failed_units"] == ["w: ValueError: no"]
    assert document["failed_controls"] == ["u: RuntimeError: x"]
    lines = size_resolve_summary.summary_lines(document)
    assert "FAILED UNITS: w: ValueError: no" in lines
    assert "FAILED CONTROLS (the unit's other figures kept): u: RuntimeError: x" in lines
    clean = size_resolve_summary.results({}, {}, {}, {}, {}, {})
    assert clean["failed_controls"] == []
    assert not any("FAILED" in line for line in size_resolve_summary.summary_lines(clean))
