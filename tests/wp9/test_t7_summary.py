"""The results document and summary text of T7's measurement (`t7_summary`), on five made-up units
measured by `t7_unit` (`t7_fake`): one with a planted Monday difference, two with exchangeable
day types, one that makes no offer, and one step unit whose windows 0.1 and 0.2 give different
statistics and whose driver term changes on no day.

1. Holm: per tier, loss and window, the adjusted p-values are `t7_stat.holm` of the p-values of the
   units whose verdict is "UNEQUAL" or "not shown unequal", and None for the unit that makes no
   offer; the verdicts are unchanged by them.
2. The document is plain JSON (no NaN); its definitions name the verdicts of `t7_unit` and say what
   the permutation null is, that day types with equal expected marginal losses but different
   values or spreads of loss changes can give "UNEQUAL", that "not shown unequal" is not a finding
   of equality and the test can have little power, and that only days whose loss changes carry
   information.
3. The summary: its first line says it is a measurement, not a pass mark; the same warnings near
   the top and under the verdicts; per unit a counts line, per tier a verdict line, a line per loss
   and a line per day type, each written out here from the document's own numbers at 4
   significant digits and compared whole (so a window's, a loss's or a quantile's number in
   another's place is seen); a type whose loss changes on no day is written "no day changes".
   The driver term's definition says it is the every-scale driver-loss bound, which counts each
   car's decision-hour draw and bounds the replayed driver loss up to the errors of V1 to V8 of
   `fixed_plan_loss`.
"""

import json

import numpy as np
import pytest
from size_fake import ramp_terms
from t7_fake import made_up_unit, random_splits, step_unit

from fleetreserve import size_scale, t7_stat, t7_summary, t7_unit


def entry(terms, types, splits):
    found = t7_unit.measure_unit("u", terms, types, splits, rounds=199)
    return {
        "site_type": "workplace",
        "status": "calibratable",
        "fit_units": ["v"],
        "counting_days": len(terms),
        "days_not_offering_by_reason": {},
        "certificate_timeouts": 0,
        **found,
    }


@pytest.fixture(scope="module")
def document():
    units = {}
    for name, seed, planted in (
        ("planted", 1, "Monday"),
        ("plain-a", 2, None),
        ("plain-b", 3, None),
    ):
        terms, types = made_up_unit(seed, planted=planted)
        units[name] = entry(terms, types, random_splits(len(terms)))
    _, types = made_up_unit(4)
    broken = [ramp_terms(5.0, np.array([-0.1]), (np.zeros(0), np.zeros(0), 0.0))] * 140
    units["no-offer"] = entry(broken, types, random_splits(140))
    terms, types = step_unit((35, 0, 35), (7, 28, 35))
    units["step"] = entry(terms, types, random_splits(140))
    return t7_summary.results(units, {}, {"model": "m", "splits": 4, "rounds": 199})


def test_holm_is_over_the_testable_units(document):
    units = document["units"]
    assert units["planted"]["tiers"]["B"]["verdict"] == t7_unit.UNEQUAL
    assert units["no-offer"]["tiers"]["B"]["verdict"] == t7_unit.NO_OFFER
    for tier in size_scale.TIERS:
        testable = [u for u in units if units[u]["tiers"][tier]["verdict"] in t7_summary.TESTABLE]
        assert "no-offer" not in testable and len(testable) == 4
        for loss in t7_stat.LOSSES:
            for h in ("0.1", "0.05", "0.2"):
                p = {u: units[u]["tiers"][tier]["pooled"][loss][h]["p_value"] for u in testable}
                expected = t7_stat.holm(p)
                for u in testable:
                    assert units[u]["tiers"][tier]["pooled"][loss][h]["holm_p"] == expected[u]
                assert units["no-offer"]["tiers"][tier]["pooled"][loss][h]["holm_p"] is None


def test_the_document_is_plain_json_and_names_the_verdicts(document):
    text = json.dumps(document, allow_nan=False)
    assert json.loads(text)["label"] == t7_summary.WHAT
    verdict = document["definitions"]["verdict"]
    for name in (t7_unit.UNEQUAL, t7_unit.NOT_SHOWN, t7_unit.NO_OFFER, t7_unit.FLAT):
        assert name in verdict
    assert document["failed_units"] == []
    definitions = document["definitions"]
    for key in ("verdict", "pooled"):
        assert t7_summary.NULL in definitions[key]
        assert t7_summary.LEVEL_OR_VALUE in definitions[key]
        assert t7_summary.NOT_EQUAL in definitions[key]
        assert t7_summary.STEPS in definitions[key]
    assert "exchangeable" in t7_summary.NULL and "is at most 0.05" in t7_summary.NULL
    assert "differ in value, or" in t7_summary.LEVEL_OR_VALUE
    assert "loss level" in t7_summary.LEVEL_OR_VALUE
    assert "'UNEQUAL' can come from such a difference" in t7_summary.LEVEL_OR_VALUE
    assert "is not a finding that" in t7_summary.NOT_EQUAL
    assert "little power" in t7_summary.NOT_EQUAL
    assert "only the days whose loss changes inside the window" in t7_summary.STEPS
    assert "lower median" in definitions["pooled"] and "SHA-256" in definitions["pooled"]
    assert "days_changing" in definitions


def g(value) -> str:
    return "none" if value is None else f"{value:.4g}"


def block(lines, name):
    """The lines of unit `name`: its counts line and the indented lines after it."""
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name} ("))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].startswith(" ")), None)
    return lines[start:end]


def expected_unit_lines(name, unit):
    out = []
    for tier in size_scale.TIERS:
        t = unit["tiers"][tier]
        deciding = t7_summary.NAMES[t["deciding_loss"]]
        offer = t["splits_that_offer"]
        where = f"lower median of the {offer} splits that offer" if offer else "no split offers"
        out.append(
            f"  tier {tier}: {t['verdict']}; deciding {deciding} (binds on "
            f"{t['binding_loss_splits'][t['deciding_loss']]} of {t['splits']} splits); pooled "
            f"at scalar {g(t['pooled_scalar'])} ({where}); scalar 0 on {t['splits_scalar_zero']} "
            f"splits, capped at 1 on {t['splits_scalar_capped']}"
        )
        for loss in ("slot", "driver"):
            by_window = t["pooled"][loss]
            ps = ", ".join(
                f"p {g(by_window[w]['p_value'])} at window {w} (Holm {g(by_window[w]['holm_p'])})"
                for w in ("0.1", "0.05", "0.2")
            )
            out.append(
                f"    {t7_summary.NAMES[loss]}: {ps}; statistic "
                f"{g(by_window['0.1']['statistic'])} per kWh at window 0.1"
            )
        for type_name, r in t["pooled"]["slot"]["0.1"]["types"].items():
            parts = []
            for loss in ("slot", "driver"):
                p = t["pooled"][loss]["0.1"]["types"][type_name]
                s = t["per_split"][loss]["0.1"][type_name]
                if p["days_changing"] == 0:
                    pooled = f"no day changes pooled (0 of {p['days']} days)"
                else:
                    pooled = (
                        f"{g(p['marginal_per_kwh'])} pooled ({p['days_changing']} of {p['days']} "
                        "days change)"
                    )
                parts.append(
                    f"{t7_summary.NAMES[loss]} {pooled}, {g(s['mean'])} over {s['splits']} "
                    f"splits (p5 {g(s['p5'])}, p95 {g(s['p95'])})"
                )
            out.append(
                f"    {type_name} ({r['days']} days, mean value {g(r['mean_value_kwh'])} kWh): "
                + "; ".join(parts)
            )
    return out


def test_the_summary_lines_are_the_documents_numbers(document):
    lines = t7_summary.summary_lines(document)
    for name, unit in document["units"].items():
        assert block(lines, name)[1:] == expected_unit_lines(name, unit), name
    # the step unit: window 0.2's statistic is 0, window 0.1's is not, and its driver term
    # changes on no day, so "no day changes" is written where a bare 0 would be
    step = document["units"]["step"]["tiers"]["B"]["pooled"]
    assert step["slot"]["0.2"]["statistic"] == 0.0 < step["slot"]["0.1"]["statistic"]
    text = "\n".join(block(lines, "step"))
    assert "driver term no day changes pooled (0 of 70 days)" in text
    assert "slot loss " + g(step["slot"]["0.1"]["types"]["Tuesday"]["marginal_per_kwh"]) in text
    assert "(7 of 70 days change)" in text and "(35 of 70 days change)" in text
    # some type's p5 and p95 differ, so a swap of the two is seen
    spreads = [
        s
        for unit in document["units"].values()
        for t in unit["tiers"].values()
        if t["per_split"] is not None
        for s in t["per_split"]["slot"]["0.1"].values()
    ]
    assert any(s["p5"] != s["p95"] for s in spreads)


def test_the_summary_warns_how_to_read_the_verdicts(document):
    lines = t7_summary.summary_lines(document)
    top = lines[3]
    assert top.startswith("HOW TO READ THE VERDICTS: ")
    under = lines[-1]
    assert under.startswith("UNDER THE VERDICTS: ")
    for text in (t7_summary.NULL, t7_summary.LEVEL_OR_VALUE, t7_summary.NOT_EQUAL):
        assert text in top
    for text in (t7_summary.LEVEL_OR_VALUE, t7_summary.NOT_EQUAL, t7_summary.STEPS):
        assert text in under
    assert t7_summary.STEPS in top


def test_the_summary_reads_the_document(document):
    lines = t7_summary.summary_lines(document)
    assert lines[0] == t7_summary.WHAT and "NOT A PASS MARK" in lines[0]
    assert lines[1] == document["definitions"]["what"]
    for name, unit in document["units"].items():
        head = [line for line in lines if line.startswith(f"{name} (")]
        assert len(head) == 1 and f"{unit['offering_days']} offering" in head[0]
        for tier in size_scale.TIERS:
            t = unit["tiers"][tier]
            deciding = t7_summary.NAMES[t["deciding_loss"]]
            start = f"  tier {tier}: {t['verdict']}; deciding {deciding} (binds on"
            assert any(line.startswith(start) for line in lines)
    assert any(line.startswith("    driver term: p ") for line in lines)
    assert "not the replayed driver loss" in lines[2]
    assert "not about the replayed driver loss" in document["definitions"]["driver_term"]
    for words in ("decision-hour draw", "every-scale driver-loss bound", "V1 to V8"):
        assert words in document["definitions"]["driver_term"]
    planted = document["units"]["planted"]["tiers"]["B"]["pooled"]["slot"]["0.1"]
    p_text = f"p {planted['p_value']:.4g} at window 0.1 (Holm {planted['holm_p']:.4g})"
    assert any(p_text in line for line in lines)


def test_a_failed_unit_is_listed():
    found = t7_summary.results({}, {"jpl": "ValueError: no days"}, {})
    assert found["failed_units"] == ["jpl: ValueError: no days"]
    lines = t7_summary.summary_lines(found)
    assert lines[-2] == "FAILED UNITS: jpl: ValueError: no days"
    assert lines[-1].startswith("UNDER THE VERDICTS: ")
