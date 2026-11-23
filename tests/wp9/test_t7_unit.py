"""T7's measurement on one made-up unit (`t7_unit`, units from `t7_fake`).

1. Power: a unit whose Mondays lose just above the calibrated scale is called "UNEQUAL", with
   Monday's marginal loss per unit value the largest.
2. False alarms: over 200 seeded units with exchangeable day types (999 permutations each, to keep
   the test quick), the share called "UNEQUAL" in either tier is at most 0.08; the measured share
   at the time of writing was 6 of 200 (0.03) in each tier.
3. Each "not testable" case, in its order: no split; no offer; one day type; no day changes loss
   (no offer before one day type when both apply; "no day changes" only when every window is
   flat).
4. The constant scale of each split is `size_scale.calibrate`'s with size 1 on the offering days;
   the pooled reading is at the lower median of the scalars used above 0 (3 and 3 splits, odd and
   even numbers of splits that offer, none), and "no offer" is read over all splits first.
5. The deciding loss: the slot loss in tier B; in tier A the driver term when it binds on most
   splits, and the verdict then reads the driver term's p-value.
6. The days of a pooled "other" of fewer than 10 days are reported and left out of the statistic.
7. The per-split readings: with one split they equal the readings made directly on its test days;
   with five, the mean, p5 and p95 are those of the five splits' own readings.
8. The internal check: when `size_scale.score` gives other losses, `InternalCheckFailed`.
9. Step units worked by hand (every day's scalar used is 1, capped): the verdict reads window 0.1,
   not 0.05 or 0.2; the days whose loss changes are counted per type, loss and window.
10. The permutations of a unit are seeded by `t7_stat.permutation_seed` of its own name.
11. An offering day of value 0 or below fails the unit.
"""

import math

import numpy as np
import pytest
from size_fake import ramp_terms
from t7_fake import made_up_unit, random_splits, step_unit

from fleetreserve import size_scale, t7_stat, t7_unit

NO_DRIVER = (np.zeros(0), np.zeros(0), 0.0)


def measure(terms, types, splits=None, rounds=999):
    splits = random_splits(len(terms)) if splits is None else splits
    return t7_unit.measure_unit("u", terms, types, splits, rounds=rounds)


# --- 1. and 2. power and false alarms ------------------------------------------------------------


def test_a_planted_difference_is_called_unequal():
    terms, types = made_up_unit(1, planted="Monday")
    found = measure(terms, types)
    for tier in size_scale.TIERS:
        entry = found["tiers"][tier]
        assert entry["verdict"] == t7_unit.UNEQUAL, entry["verdict"]
        types_found = entry["pooled"]["slot"]["0.1"]["types"]
        largest = max(types_found, key=lambda t: types_found[t]["marginal_per_kwh"])
        assert largest == "Monday"
        assert entry["pooled"]["slot"]["0.1"]["p_value"] < 0.01


def test_exchangeable_types_are_rarely_called_unequal():
    called = {tier: 0 for tier in size_scale.TIERS}
    units = 200
    for seed in range(1, units + 1):
        terms, types = made_up_unit(seed)
        found = measure(terms, types)
        for tier in size_scale.TIERS:
            verdict = found["tiers"][tier]["verdict"]
            assert verdict in (t7_unit.UNEQUAL, t7_unit.NOT_SHOWN)
            called[tier] += verdict == t7_unit.UNEQUAL
    assert all(c / units <= 0.08 for c in called.values()), called
    assert all(c >= 1 for c in called.values()), called  # the test can fire at all


# --- 3. not testable -----------------------------------------------------------------------------


def test_no_split():
    terms, types = made_up_unit(2)
    found = measure(terms, types, splits=[])
    assert all(e["verdict"] == t7_unit.NOT_SPLIT for e in found["tiers"].values())
    assert found["tiers"]["A"]["pooled"] is None and found["splits"] == []


def test_no_offer_when_the_scalar_is_zero_on_more_than_half_of_the_splits():
    terms = [ramp_terms(5.0, np.array([-0.1]), NO_DRIVER) for _ in range(140)]
    _, types = made_up_unit(2)
    found = measure(terms, types)
    for entry in found["tiers"].values():
        assert entry["splits_scalar_zero"] == 4 and entry["verdict"] == t7_unit.NO_OFFER
    # three of four splits offering is not "no offer"
    terms, types = made_up_unit(3)
    found = measure(terms, types, splits=random_splits(140) + [(9, [0], list(range(1, 140)))])
    assert found["tiers"]["B"]["splits_scalar_zero"] == 1
    assert found["tiers"]["B"]["verdict"] != t7_unit.NO_OFFER


def test_one_day_type():
    terms, _ = made_up_unit(4, n=39)
    types = ["Monday"] * 30 + ["Tuesday"] * 5 + ["Sunday"] * 4
    found = measure(terms, types, splits=random_splits(39, n_calibration=30))
    assert found["day_types"] == {"Monday": 30, "other": 9}
    assert found["tested_types"] == ["Monday"]
    entry = found["tiers"]["B"]
    assert entry["verdict"] == t7_unit.ONE_TYPE and entry["splits_scalar_zero"] == 0
    assert entry["pooled"]["slot"]["0.1"]["p_value"] is None


def test_no_day_changes_loss_inside_the_window():
    terms = [ramp_terms(5.0, np.array([5.0 + k % 3]), NO_DRIVER) for k in range(140)]
    _, types = made_up_unit(5)
    found = measure(terms, types)
    for entry in found["tiers"].values():
        assert entry["pooled_scalar"] == 1.0  # infinite, capped at 1
        assert entry["splits_scalar_capped"] == 4
        assert entry["verdict"] == t7_unit.FLAT
        assert entry["pooled"]["slot"]["0.1"]["p_value"] == 1.0


def test_no_offer_comes_before_one_day_type():
    # every day loses at every scale above 0 and every day is a Monday: both rules apply
    terms = [ramp_terms(5.0, np.array([-0.1]), NO_DRIVER) for _ in range(140)]
    found = measure(terms, ["Monday"] * 140)
    assert found["tested_types"] == ["Monday"]
    for entry in found["tiers"].values():
        assert entry["splits_scalar_zero"] == 4 and entry["verdict"] == t7_unit.NO_OFFER


# --- 4. the constant scale -----------------------------------------------------------------------


def test_the_constant_scale_is_size_scale_calibrate_and_the_reading_is_at_its_lower_median():
    terms, types = made_up_unit(6)
    splits = random_splits(len(terms), count=6)
    found = measure(terms, types, splits=splits)
    q = np.ones(len(terms))
    used = {tier: [] for tier in size_scale.TIERS}
    for row, (seed, calibration, _) in zip(found["splits"], splits, strict=True):
        scalars = size_scale.calibrate(terms, q, calibration)
        assert row["seed"] == seed
        for tier in size_scale.TIERS:
            assert row[tier]["used"] == scalars[tier].used
            used[tier].append(scalars[tier].used)
    for tier in size_scale.TIERS:
        entry = found["tiers"][tier]
        offering = sorted(u for u in used[tier] if u > 0)
        assert len(offering) == 6 and len(set(offering)) == 6  # six different scalars, all used
        # six splits that offer: the third smallest, a scalar a split used, not numpy's median
        assert entry["pooled_scalar"] == offering[2] != float(np.median(offering))
        assert entry["splits_that_offer"] == 6
        low, high = t7_stat.window(entry["pooled_scalar"], 0.1)
        assert entry["pooled"]["slot"]["0.1"]["low"] == low
        assert entry["pooled"]["slot"]["0.1"]["high"] == high


def tier(used, tier_name="B", unit=9):
    """`t7_unit.tier_reading` on a made-up unit with the scalars used given, one per split."""
    terms, types = made_up_unit(unit)
    pooled_types = dict(enumerate(t7_stat.pool_types(types)))
    names = tuple(t7_stat.type_counts(list(pooled_types.values())))
    rows = [
        (s, {t: size_scale.Scalar(u, u, "", 70) for t in size_scale.TIERS}, list(range(70, 140)))
        for s, u in enumerate(used, start=1)
    ]
    return t7_unit.tier_reading("u", tier_name, terms, pooled_types, names, names, rows, 9)


@pytest.mark.parametrize(
    "used, expected, offering, verdict_is_no_offer",
    [
        ([0.0, 0.0, 0.0, 0.9, 0.95, 1.0], 0.95, 3, False),  # 3 and 3: the middle of the three
        ([1.0, 0.8, 0.95, 0.9], 0.9, 4, False),  # even: the lower of the middle two
        ([0.9, 1.0, 0.8], 0.9, 3, False),  # odd: the middle
        ([0.7, 0.0, 0.5], 0.5, 2, False),  # a zero is left out: the lower of 0.5 and 0.7
        ([0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.5], 0.5, 3, True),  # no offer first, still read at 0.5
        ([0.0, 0.0, 0.0], 0.0, 0, True),  # no split offers: read at 0
    ],
)
def test_the_pooled_scalar_is_the_lower_median_of_the_splits_that_offer(
    used, expected, offering, verdict_is_no_offer
):
    found = tier(used)
    assert found["pooled_scalar"] == expected
    assert expected in used  # a scalar some split used
    assert found["splits_that_offer"] == offering
    assert found["splits_scalar_zero"] == len(used) - offering
    assert (found["verdict"] == t7_unit.NO_OFFER) is verdict_is_no_offer
    assert found["pooled"]["slot"]["0.1"]["low"] == t7_stat.window(expected, 0.1)[0]
    assert t7_unit.pooled_scalar(used) == expected


def test_a_day_that_does_not_offer_is_in_no_reading():
    terms, types = made_up_unit(7)
    terms = list(terms)
    terms[0] = None  # a Monday
    found = measure(terms, types)
    assert found["offering_days"] == 139 and found["day_types"]["Monday"] == 19
    assert found["tiers"]["B"]["pooled"]["slot"]["0.1"]["types"]["Monday"]["days"] == 19


# --- 5. the deciding loss ------------------------------------------------------------------------


def test_the_driver_loss_decides_tier_a_when_it_binds_on_most_splits():
    # Mondays differ in their slot losses only; a small protected energy makes the driver bind
    terms, types = made_up_unit(8, planted="Monday", protected=0.3)
    found = measure(terms, types)
    a, b = found["tiers"]["A"], found["tiers"]["B"]
    assert a["binding_loss_splits"] == {"slot": 0, "driver": 4}
    assert a["deciding_loss"] == "driver" and b["deciding_loss"] == "slot"
    assert b["binding_loss_splits"] == {"slot": 4, "driver": 0}
    assert a["pooled"]["driver"]["0.1"]["p_value"] >= 0.05
    assert a["pooled"]["slot"]["0.1"]["p_value"] < 0.05  # reported, does not decide
    assert a["verdict"] == t7_unit.NOT_SHOWN
    assert b["verdict"] == t7_unit.UNEQUAL


def test_a_tie_of_binding_splits_goes_to_the_slot_loss():
    terms, types = made_up_unit(9)
    offering = list(range(len(terms)))
    pooled_types = dict(zip(offering, t7_stat.pool_types(types), strict=True))
    names = tuple(t7_stat.type_counts(list(pooled_types.values())))
    test = list(range(70, 140))

    def scalars(a, b):
        return {"A": size_scale.Scalar(a, a, "", 70), "B": size_scale.Scalar(b, b, "", 70)}

    rows = [(1, scalars(0.3, 0.5), test), (2, scalars(0.5, 0.5), test)]
    found = t7_unit.tier_reading("u", "A", terms, pooled_types, names, names, rows, 99)
    assert found["binding_loss_splits"] == {"slot": 1, "driver": 1}
    assert found["deciding_loss"] == "slot"
    rows.append((3, scalars(0.2, 0.5), test))
    found = t7_unit.tier_reading("u", "A", terms, pooled_types, names, names, rows, 99)
    assert found["deciding_loss"] == "driver"
    assert (
        t7_unit.tier_reading("u", "B", terms, pooled_types, names, names, rows, 99)["deciding_loss"]
        == "slot"
    )
    # binding reads the calibrated scalars: both capped at 1, the driver's 1.5 below the slot's
    # infinity still binds, and the cap is counted
    capped = {
        "A": size_scale.Scalar(1.5, 1.0, "", 70),
        "B": size_scale.Scalar(math.inf, 1.0, "", 70),
    }
    found = t7_unit.tier_reading(
        "u", "A", terms, pooled_types, names, names, [(1, capped, test)], 9
    )
    assert found["binding_loss_splits"] == {"slot": 0, "driver": 1}
    assert found["splits_scalar_capped"] == 1 and found["pooled_scalar"] == 1.0


# --- 6. a small "other" --------------------------------------------------------------------------


def test_the_days_of_a_small_other_are_left_out_of_the_statistic():
    terms, _ = made_up_unit(10, n=69)
    types = ["Monday"] * 30 + ["Tuesday"] * 30 + ["Sunday"] * 9
    found = measure(terms, types, splits=random_splits(69, n_calibration=40))
    entry = found["tiers"]["B"]["pooled"]["slot"]["0.1"]
    assert entry["types"]["other"]["days"] == 9
    scale = found["tiers"]["B"]["pooled_scalar"]
    differences = t7_stat.loss_differences(terms[:60], scale, 0.1)
    expected = t7_stat.spread_statistic(
        differences.differences["slot"],
        np.array([t.value for t in terms[:60]]),
        np.repeat([0, 1], 30),
        2,
        differences.width,
    )
    assert entry["statistic"] == pytest.approx(expected, rel=1e-12, abs=1e-15)


# --- 7. per-split readings -----------------------------------------------------------------------


def test_one_split_gives_the_readings_of_its_test_days():
    terms, types = made_up_unit(11, planted="Friday")
    splits = random_splits(len(terms), count=1)
    found = measure(terms, types, splits=splits)
    _, _, test = splits[0]
    scale = found["splits"][0]["B"]["used"]
    days = [d for d in test if terms[d] is not None]
    differences = t7_stat.loss_differences([terms[d] for d in days], scale, 0.05)
    direct = t7_stat.type_readings(
        differences, "slot", [terms[d].value for d in days], [types[d] for d in days]
    )
    spread = found["tiers"]["B"]["per_split"]["slot"]["0.05"]
    for name, reading in direct.items():
        assert spread[name]["splits"] == 1
        assert spread[name]["mean"] == spread[name]["p5"] == spread[name]["p95"]
        assert spread[name]["mean"] == pytest.approx(reading.marginal, rel=1e-12, abs=1e-15)


def test_the_spread_over_splits_is_that_of_the_splits_own_readings():
    terms, types = made_up_unit(15)
    splits = random_splits(len(terms), count=5)
    found = measure(terms, types, splits=splits)
    for tier_name in size_scale.TIERS:
        mine = {}
        for row, (_, _, test) in zip(found["splits"], splits, strict=True):
            days = [d for d in test if terms[d] is not None]
            differences = t7_stat.loss_differences(
                [terms[d] for d in days], row[tier_name]["used"], 0.1
            )
            for loss in t7_stat.LOSSES:
                readings = t7_stat.type_readings(
                    differences, loss, [terms[d].value for d in days], [types[d] for d in days]
                )
                for name, r in readings.items():
                    mine.setdefault((loss, name), []).append(r.marginal)
        spread = found["tiers"][tier_name]["per_split"]
        apart = 0
        for (loss, name), values in mine.items():
            s = spread[loss]["0.1"][name]
            assert s["splits"] == 5
            assert s["mean"] == pytest.approx(np.mean(values), rel=1e-12, abs=1e-15)
            assert s["p5"] == pytest.approx(np.quantile(values, 0.05), rel=1e-12, abs=1e-15)
            assert s["p95"] == pytest.approx(np.quantile(values, 0.95), rel=1e-12, abs=1e-15)
            apart += np.quantile(values, 0.1) != np.quantile(values, 0.05)
        assert apart >= 3  # p10 differs from p5 for some types, so reading p10 would be seen


def test_a_type_with_no_test_day_on_a_split_is_not_counted_there():
    terms, types = made_up_unit(12)
    mondays = [d for d, t in enumerate(types) if t == "Monday"]
    others = [d for d in range(len(terms)) if d not in mondays]
    splits = [(1, mondays + others[:60], others[60:]), *random_splits(len(terms), count=2)]
    found = measure(terms, types, splits=splits)
    assert found["tiers"]["A"]["per_split"]["slot"]["0.1"]["Monday"]["splits"] == 2
    assert found["tiers"]["A"]["per_split"]["slot"]["0.1"]["Tuesday"]["splits"] == 3


# --- 8. the internal check -----------------------------------------------------------------------


def test_the_internal_check_fails_when_the_losses_differ(monkeypatch):
    terms, types = made_up_unit(13)
    real = size_scale.score

    def shifted(terms_, q, gamma, test):
        found = real(terms_, q, gamma, test)
        slot = tuple(min(1.0, x + 0.5) for x in found.slot_losses)
        return size_scale.TestScore(
            found.offers_kwh, found.offering_days, slot, found.driver_losses
        )

    monkeypatch.setattr(size_scale, "score", shifted)
    with pytest.raises(t7_unit.InternalCheckFailed, match="slot losses"):
        measure(terms, types)


def test_values_and_reasons_are_written_as_numbers():
    terms, types = made_up_unit(14)
    found = measure(terms, types)
    for row in found["splits"]:
        for tier_name in size_scale.TIERS:
            calibrated = row[tier_name]["calibrated"]
            assert calibrated == "inf" or not math.isnan(calibrated)


# --- 9. step units worked by hand ----------------------------------------------------------------


def test_the_verdict_reads_window_01_and_the_changing_days_are_counted():
    # 0.1 (0.9 to 1.1): Mondays change on 35 days, Tuesdays on 7: UNEQUAL;
    # 0.2 (0.8 to 1.2): 35 and 35 change: the statistic is 0 and p is 1;
    # 0.05 (0.95 to 1.05): no day changes
    terms, types = step_unit((35, 0, 35), (7, 28, 35))
    found = measure(terms, types)
    for tier_name in size_scale.TIERS:
        entry = found["tiers"][tier_name]
        assert entry["pooled_scalar"] == 1.0 and entry["splits_scalar_capped"] == 4
        slot = entry["pooled"]["slot"]
        assert slot["0.1"]["p_value"] < 0.01 and slot["0.2"]["p_value"] == 1.0
        assert entry["verdict"] == t7_unit.UNEQUAL
        changing = {
            w: {n: slot[w]["types"][n]["days_changing"] for n in ("Monday", "Tuesday")}
            for w in ("0.05", "0.1", "0.2")
        }
        assert changing == {
            "0.05": {"Monday": 0, "Tuesday": 0},
            "0.1": {"Monday": 35, "Tuesday": 7},
            "0.2": {"Monday": 35, "Tuesday": 35},
        }
        driver = entry["pooled"]["driver"]
        assert all(
            driver[w]["types"][n]["days_changing"] == 0 for w in changing for n in changing[w]
        )


def test_the_verdict_is_not_flat_when_only_another_window_changes():
    # only the thresholds 1.15 change, inside 0.8 to 1.2 alone: windows 0.05 and 0.1 are flat,
    # 0.2 is not (and there Mondays change on 20 days, Tuesdays on 5); at 0.1 p is 1
    terms, types = step_unit((0, 20, 50), (0, 5, 65))
    found = measure(terms, types)
    for tier_name in size_scale.TIERS:
        entry = found["tiers"][tier_name]
        slot = entry["pooled"]["slot"]
        assert slot["0.1"]["p_value"] == 1.0 and slot["0.2"]["p_value"] < 0.01
        assert entry["verdict"] == t7_unit.NOT_SHOWN


# --- 10. and 11. seeds and values ----------------------------------------------------------------


def test_each_unit_is_permuted_with_its_own_seed():
    terms, types = made_up_unit(16)
    splits = random_splits(len(terms))
    found = {u: t7_unit.measure_unit(u, terms, types, splits, rounds=99) for u in ("a", "b")}
    entry = found["a"]["tiers"]["B"]
    scale = entry["pooled_scalar"]
    tested = found["a"]["tested_types"]
    days = [d for d in range(len(terms)) if types[d] in tested]
    differences = t7_stat.loss_differences([terms[d] for d in days], scale, 0.1)
    codes = np.array([tested.index(types[d]) for d in days])
    values = np.array([terms[d].value for d in days])
    p = {
        u: t7_stat.permutation_p(
            differences.differences["slot"],
            values,
            codes,
            len(tested),
            differences.width,
            99,
            t7_stat.permutation_seed(u),
        )[1]
        for u in ("a", "b")
    }
    for u in ("a", "b"):
        assert found[u]["tiers"]["B"]["pooled"]["slot"]["0.1"]["p_value"] == p[u]
    assert p["a"] != p["b"]


def test_an_offering_day_of_value_zero_fails_the_unit():
    terms, types = made_up_unit(17)
    terms = list(terms)
    terms[3] = ramp_terms(0.0, terms[3].thresholds, terms[3].driver)
    with pytest.raises(ValueError, match="offering day 3 has value 0.0"):
        measure(terms, types)
