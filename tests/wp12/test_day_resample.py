"""The scoring run's evaluations for the intervals over days (`day_resample.resample_pairs`), with
the made-up step replay of `interval_fake` counting every pair it is asked for.

1. One store for every resample, rung and tier: no (day, scale) pair is evaluated twice, none of
   the run's own pairs is evaluated again, the run's own store is left as it was, and the pairs
   evaluated are exactly those the searches of the resamples ask for beyond the run's own (the
   searches rerun on the run's own pairs and the new ones together meet no unknown pair). 40
   resamples of 30 made-up days whose thresholds are spread out, both tiers.
2. A failed pair: when a batch holds a pair whose replay raises, the pairs are evaluated one by
   one, so only that pair is recorded as failed (with its error) and every other pair of the
   batch is evaluated; the failed pair is asked no second time.
3. Every resample's scale from the stored pairs equals the full recalibration of `crc` on the
   resampled days with every loss known: all 40 resamples, three rungs, both tiers.
"""

from collections import Counter

import numpy as np
from interval_fake import batch, made_up_day, replay

from fleetreserve import crc, day_bootstrap, day_losses, day_resample
from fleetreserve.dial_splits import tier_alphas
from fleetreserve.gate_config import CEILINGS
from fleetreserve.rungs import RUNGS

N_CAL, N_TEST, RESAMPLES, SEED = 30, 10, 40, 5


def made_up_job():
    rng = np.random.default_rng(3)
    thresholds = rng.uniform(0.1, 1.2, N_CAL)
    offers = rng.uniform(size=N_CAL) < 0.9
    days = [
        made_up_day(bool(o), float(t), (0.5, float(t) + 0.2))
        for o, t in zip(offers, thresholds, strict=True)
    ]
    days += [made_up_day(True) for _ in range(N_TEST)]
    roles = ["calibration"] * N_CAL + ["test"] * N_TEST
    offering = [d for d in range(N_CAL) if offers[d]]
    own_scales = []

    def loss_at(beta):
        own_scales.append(beta)
        found = [replay(days[d], beta).rungs[RUNGS[0]] for d in offering]
        return np.array([e.slot_loss for e in found]), np.array([e.driver_loss for e in found])

    for alphas in (tier_alphas("A"), tier_alphas("B"), (1.0, 0.05)):
        crc.calibrate_search(loss_at, len(offering), *alphas)
    store = {(d, b): replay(days[d], b) for d in offering for b in own_scales}
    return days, roles, store


def counting(fails=frozenset()):
    calls = Counter()
    inner = batch(fails)

    def evaluate(pairs):
        calls.update((id(day), beta) for day, beta in pairs)
        return inner(pairs)

    return evaluate, calls


def test_one_store_evaluates_each_pair_once():
    days, roles, store = made_up_job()
    before = dict(store)
    evaluate, calls = counting()
    found = day_resample.resample_pairs(
        days, store, roles, "u", SEED, RESAMPLES, ("A", "B"), evaluate
    )
    assert store == before
    position = {id(day): d for d, day in enumerate(days)}
    asked = Counter({(position[i], b): n for (i, b), n in calls.items()})
    assert max(asked.values()) == 1
    assert set(asked) == set(found.evaluated) and not set(asked) & set(store)
    assert found.failed == {} and len(found.evaluated) > 0
    assert found.searches == RESAMPLES * len(RUNGS) * 2
    assert found.own_calibration_pairs == len(store)
    # the searches on the run's pairs and the new ones together meet no unknown pair
    again, more = counting()
    known = {**store, **found.evaluated}
    day_resample.resample_pairs(days, known, roles, "u", SEED, RESAMPLES, ("A", "B"), again)
    assert sum(more.values()) == 0


def test_a_failed_pair_is_recorded_and_costs_no_other():
    days, roles, store = made_up_job()
    evaluate, _ = counting()
    clean = day_resample.resample_pairs(
        days, store, roles, "u", SEED, RESAMPLES, ("A", "B"), evaluate
    )
    bad = min(clean.evaluated, key=lambda key: (key[1], key[0]))
    evaluate, calls = counting(frozenset({(id(days[bad[0]]), bad[1])}))
    found = day_resample.resample_pairs(
        days, store, roles, "u", SEED, RESAMPLES, ("A", "B"), evaluate
    )
    assert set(found.failed) == {bad}
    assert found.failed[bad] == "RuntimeError: a made-up evaluation failed"
    assert bad not in found.evaluated
    position = {id(day): d for d, day in enumerate(days)}
    singles = Counter((position[i], b) for (i, b) in calls.elements())
    assert singles[bad] == 2  # once in its batch, once alone; never again
    section = day_losses.resampled_section(found)
    assert section["failed"] == {
        "day": [bad[0]],
        "scale": [bad[1]],
        "error": ["RuntimeError: a made-up evaluation failed"],
    }


def test_every_resample_scale_is_the_full_recalibration():
    days, roles, store = made_up_job()
    evaluate, _ = counting()
    found = day_resample.resample_pairs(
        days, store, roles, "u", SEED, RESAMPLES, ("A", "B"), evaluate
    )
    record = {
        "unit": "u",
        "configurations": ["c"],
        "days": {
            "date": [str(d) for d in range(len(days))],
            "site": ["s"] * len(days),
            "role": roles,
            "cars": [3] * len(days),
            "offers": [not day.abstains for day in days],
            "certified_value_kwh": [1.0] * len(days),
            "ceiling_kwh": {name: [2.0] * len(days) for name in CEILINGS},
        },
        "calibration": _own_tables(days, roles, store),
        "test": {
            "days": list(range(N_CAL, N_CAL + N_TEST)),
            "rungs": {r: {t: {"scale": 0.0} for t in ("A", "B")} for r in RUNGS},
        },
        "resampled": day_losses.resampled_section(found),
    }
    job = day_bootstrap.Job.from_record(record, RUNGS)
    rows, _ = day_bootstrap.draws(SEED, "u", N_CAL, N_TEST, RESAMPLES)
    checked = 0
    for row in rows:
        drawn = [d for d in row if not days[d].abstains]
        for rung in RUNGS:
            for tier in ("A", "B"):
                full = crc.calibrate_search(
                    lambda b, drawn=drawn: _losses(days, drawn, b), len(drawn), *tier_alphas(tier)
                ).beta
                mine = job.scale(rung, tier, row)
                assert mine.exact and mine.low == mine.high == full
                checked += 1
    assert checked == RESAMPLES * len(RUNGS) * 2


def _losses(days, drawn, beta):
    found = [replay(days[d], beta).rungs[RUNGS[0]] for d in drawn]
    return np.array([e.slot_loss for e in found]), np.array([e.driver_loss for e in found])


def _own_tables(days, roles, store) -> dict:
    offering = [d for d, r in enumerate(roles) if r == "calibration" and not days[d].abstains]
    scales = sorted({b for d, b in store})
    return {
        "days": offering,
        "scales": scales,
        "slot_loss": {
            r: [[store[(d, b)].rungs[r].slot_loss for b in scales] for d in offering] for r in RUNGS
        },
        "driver_loss": {
            r: [[store[(d, b)].rungs[r].driver_loss for b in scales] for d in offering]
            for r in RUNGS
        },
    }
