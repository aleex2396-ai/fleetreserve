"""Made-up drive logs, run folders and summaries in the real writers' forms, for the tests of WP16.

Not a test file. `MadeUp(top)` writes into the folder `top`:
- committed files (`commit_file`): the run list with its nine units, matrices with a
  `sealed_choice`;
- run folders (`run`): `runs.noindex/<stamp>-<label>/` with the output files and a `manifest.json`
  as the run recorder writes it (commit, command, start, end, exit_code, outputs by SHA-256);
- a drive log (`drive`): `runs.noindex/drives/<stamp>-sealed-run.jsonl` in the driver's form
  (`run_driver.encode`, `end_digest`): a `start` line, a `begin` and an `entry` line per step, a
  `failed` line after an entry that failed, a `not_run` line for a step not started, an optional
  `stop` line and an `end` line whose `log_sha256` recomputes.
The summaries (`learn_document` and the others) follow the writers' keys as read from `learn_run`,
`learn_eval`, `gate_calibrate`, `day_intervals`, `day_bootstrap`, `ledger_run`, `ceiling_report`,
`o3_run`, `corner_run`, `cold_start` and `compare_run`; made-up numbers, no data.
"""

import hashlib
import itertools
import json
from pathlib import Path

from fleetreserve.run_driver import LOG_FORMAT, encode, end_digest

COMMIT = "a" * 40
TAG = "b" * 40
HINDSIGHT = "hindsight-free-decision-hour"
BEST = "best-causal"
UNITS = ["u1", "u2", "u3"]
NAME_MAIN = "main-k0.5-d0.02-empirical"
NAME_OTHER = "main-k0.5-d0.03-empirical"
NAME_BESIDE = "beside-k0.5-d0.02-empirical"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(document) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=1) + "\n").encode()


def scored_block(offer=2.0, per_car=0.5, share=0.4, best=0.5, slot=0.02, driver=0.01) -> dict:
    """A rung's `test` as `gate_calibrate.score_test` writes it."""
    return {
        "days": 5,
        "offering_days": 4,
        "failing_days": 0,
        "mean_offer_kwh_per_day": offer,
        "mean_offer_kwh_per_offering_day": offer * 5 / 4,
        "offer_kwh_per_car_per_day": per_car,
        "share_of_ceiling": {
            HINDSIGHT: {"ratio_of_sums": share, "mean_of_daily_ratios": share},
            BEST: {"ratio_of_sums": best, "mean_of_daily_ratios": best},
        },
        "mean_slot_loss_offering_days": slot,
        "mean_driver_loss_offering_days": driver,
    }


def rungs(offer_fixed, offer_one_way, **kwargs) -> dict:
    """`learn_eval._rungs`: the fixed and one-way rungs, tier B under `tier_b`."""
    out = {}
    for rung, offer in (("fixed", offer_fixed), ("one-way", offer_one_way)):
        out[rung] = {
            "beta_hat": 0.5,
            "reason": "",
            "test": scored_block(offer, **kwargs),
            "tier_b": {
                "beta_hat": 0.6,
                "reason": "",
                "test": scored_block(offer * 1.1, **kwargs),
                "stay_to_bound": {},
            },
        }
    return out


def matrix_document() -> dict:
    def config(name, family, driver):
        return {
            "name": name,
            "family": family,
            "kappa": 0.5,
            "driver_risk": driver,
            "departure": "empirical",
        }

    return {
        "configurations": [
            config(NAME_MAIN, "main", 0.02),
            config(NAME_OTHER, "main", 0.03),
            config(NAME_BESIDE, "beside", 0.02),
        ],
        "sealed_choice": {
            "headline": {"workplaces": NAME_MAIN, "others": NAME_OTHER},
            "units": {"workplaces": ["u1"], "others": ["u2", "u3"]},
        },
    }


def matrix_same() -> dict:
    """The matrix with one headline configuration for both site types."""
    document = matrix_document()
    document["sealed_choice"]["headline"] = {"workplaces": NAME_MAIN, "others": NAME_MAIN}
    return document


def run_list_document(units=None) -> dict:
    """The parts of the run list WP16 reads: the nine units and the fixed seed and counts."""
    return {
        "units": {"sealed-run": {"nine": list(units or UNITS)}},
        "fixed": {"seeds": {"intervals": 20261122}, "counts": {"resamples": 2000}},
    }


def learn_document(
    matrix_sha: str, offers: dict, headline: dict | None = None, beside: float | None = None
) -> dict:
    """`learn.json` for the units in `offers` (unit -> (fixed offer, one-way offer) or a text that
    is the reason the unit's configuration failed); every unit holds the configurations of
    `headline` (unit -> name; by default the matrix's)."""
    headline = headline or {"u1": NAME_MAIN, "u2": NAME_OTHER, "u3": NAME_OTHER}
    units = {}
    for unit, found in offers.items():
        name = headline[unit]
        if isinstance(found, str):
            configurations = {name: {"rungs": {"reason": found}}}
        else:
            dial = 0.02 if name == NAME_MAIN else 0.03
            configurations = {
                name: {"family": "main", "driver_risk": dial, "rungs": rungs(*found)},
                f"main-k0.5-none-{name[-9:]}": {
                    "family": "main",
                    "driver_risk": None,
                    "rungs": rungs(found[0] * 2, found[1] * 2),
                },
            }
            if beside is not None:  # the beside reading: other offers
                configurations[NAME_BESIDE] = {
                    "family": "beside",
                    "driver_risk": 0.02,
                    "rungs": rungs(beside, beside),
                }
        units[unit] = {
            "counting_days": 10,
            "calibration_days_used": 5,
            "test_days": 5,
            "status": "calibratable",
            "configurations": configurations,
        }
    return {
        "model": "made-up",
        "matrix": {"file": "m.json", "sha256": matrix_sha},
        "units": units,
        "failed_configurations": [],
        "t5": {"day_paths_checked": 100, "breaks_a": 0, "breaks_b": 0},
    }


def e_prime_entry(
    test_days=5, certified=4, in_e_prime=2, steered=12, steered_in=8, **shares
) -> dict:
    """One configuration's entry of `units.<unit>.e_prime` as `learn_run.e_prime_figures` writes
    it: the five counts of `learn_run.COUNTS` and the three shares, each its count over its
    denominator and null where the denominator is 0; `shares` replaces a share by key."""

    def share(count, of):
        return None if of == 0 else count / of

    found = {
        "test_days": test_days,
        "certified_test_days": certified,
        "days_in_e_prime": in_e_prime,
        "steered_cars_on_certified_test_days": steered,
        "steered_cars_in_e_prime_i": steered_in,
        "share_of_test_days_in_e_prime": share(in_e_prime, test_days),
        "share_of_certified_test_days_in_e_prime": share(in_e_prime, certified),
        "share_of_steered_cars_in_e_prime_i": share(steered_in, steered),
    }
    return {**found, **shares}


def with_e_prime(document: dict, figures: dict) -> dict:
    """`document` (a `learn_document`) with `figures` (unit -> configuration -> an `e_prime_entry`
    or `{"reason": ...}`) under `units.<unit>.e_prime`, and `e_prime_definition` at the top, as
    `learn_run.evaluate` writes them; a unit of `figures` the document does not hold is added with
    nothing else."""
    for unit, found in figures.items():
        document["units"].setdefault(unit, {})["e_prime"] = found
    document["e_prime_definition"] = "E' (plan T2): made-up definition of the denominators"
    return document


def not_calibratable(document: dict, unit: str, days: int = 12) -> dict:
    """`document` (a `learn_document`, with `with_e_prime` applied) with `unit` as a unit too short
    to split, in the writers' form: `gate_run.plan_unit` gives its days the role "not calibratable"
    (status "not calibratable: <n> days", no calibration or test day), `gate_run.assemble` gives
    each configuration the reason alone, `learn_eval._rungs` writes it as `rungs: {"reason": ...}`,
    and `learn_run.e_prime_figures`, which finds no test day, writes counts of 0 and null shares
    with no reason, for every configuration of the unit."""
    status = f"not calibratable: {days} days"
    entry = document["units"][unit]
    entry.update(
        {"counting_days": days, "calibration_days_used": 0, "test_days": 0, "status": status}
    )
    for _name, found in entry.get("configurations", {}).items():
        found["rungs"] = {"reason": status}
    figures = entry.setdefault("e_prime", {})
    for name in {*figures, *entry.get("configurations", {})}:
        figures[name] = e_prime_entry(0, 0, 0, 0, 0)
    return document


def bracketed(point, low=None, high=None) -> dict:
    return {
        "point": point,
        "resamples_left_out": 0,
        "interval": None if low is None else [low, high],
    }


def intervals_document(name: str, units: dict, pooled: dict | None = None) -> dict:
    """`day-intervals.json`: `units` unit -> {availability, shortfall, offer} intervals as
    (point, low, high) of the one-way rung, tier A."""
    found = {}
    for unit, figures in units.items():
        cell = {
            "offer_kwh_per_counting_test_day": bracketed(*figures["offer"]),
            "offer_kwh_per_car_per_day": bracketed(0.5, 0.4, 0.6),
            "share_of_ceiling": {
                HINDSIGHT: bracketed(0.4, 0.3, 0.5),
                BEST: bracketed(0.5, 0.4, 0.6),
            },
            "availability": {
                "point": figures["availability"][0],
                "interval": figures["availability"][1:],
            },
            "shortfall": {"point": figures["shortfall"][0], "interval": figures["shortfall"][1:]},
        }
        found[unit] = {"one-way": {"A": cell, "B": cell}, "fixed": {"A": cell}}
    document = {
        "seed": 20261122,
        "resamples": 2000,
        "configurations": {name: found},
        "pooled": {"configurations": {}},
        "problems": [],
    }
    if pooled is not None:
        document["pooled"]["configurations"][name] = pooled
    return document


def pooled_block(units: list) -> dict:
    one = {
        "offer_kwh_per_counting_test_day": bracketed(7.0, 5.0, 9.0),
        "offer_kwh_per_car_per_day": bracketed(1.5, 1.0, 2.0),
        "share_of_ceiling": {HINDSIGHT: bracketed(0.41, 0.3, 0.5), BEST: bracketed(0.51, 0.4, 0.6)},
    }
    total = {**one, "offer_kwh_per_counting_test_day": bracketed(2.3, 1.7, 3.0)}
    return {
        "units": units,
        "left_out": [],
        "rungs": {"one-way": {"A": {"sum_of_unit_offers": one, "total_over_total_days": total}}},
    }


def verdict_block(of_days, covered=None, lines=(), verdict="passed", reasons=()) -> dict:
    """The part every check of `dev-day-checks.json` holds (`dev_day_verdicts.verdict`)."""
    return {
        "verdict": verdict,
        "of_days": of_days,
        "covered_days": of_days if covered is None else covered,
        "timeout_days": 0,
        "other_missing_days": 0,
        "failure_lines": list(lines),
        "could_not_run_reasons": list(reasons),
    }


def t4_group(days, both, below, gap_below, with_affine, affine_below, gap_affine) -> dict:
    return {
        "days": days,
        "days_with_hind_and_tree": both,
        "days_tree_below_hind": below,
        "mean_gap_hind_minus_tree_kwh": gap_below,
        "days_with_tree_and_affine": with_affine,
        "days_affine_below_tree": affine_below,
        "mean_gap_tree_minus_affine_kwh": gap_affine,
    }


def dev_days_document() -> dict:
    """`dev-day-checks.json` in the keys `dev_day_checks.main` and `dev_day_verdicts` write, with
    made-up numbers that differ from one key to the next (so that a figure read from the wrong key
    shows): 50 counting days, a sample of 10, 8 certified, 4 in E'."""
    groups = {
        "meets T4's hypotheses": t4_group(12, 11, 1, 0.25, 10, 2, 0.5),
        "other days, no in-window departure": t4_group(18, 17, 7, 1.25, 16, 3, 0.75),
        "other days, in-window departure": t4_group(20, 19, 9, 2.5, 18, 4, 1.5),
    }
    t4 = {
        **verdict_block(50),
        "groups": groups,
        "hypothesis_days": 12,
        "hypothesis_days_checked": 11,
        "findings_tree_above_affine": [],
    }
    t2 = {
        **verdict_block(10),
        "certified_days": 8,
        "days_without_certificate_by_rule": {"no certificate: no steered car": 2},
        "days_in_e_prime": 4,
        "share_of_sampled_days_in_e_prime": 0.4,
        "share_of_certified_days_in_e_prime": 0.5,
        "steered_cars_on_certified_days": 40,
        "share_of_steered_cars_in_e_prime_i": 0.85,
        "cars_on_sampled_days": 61,
        "cars_left_unsteered_no_forecast": 3,
        "cars_left_unsteered_empty_forecast_set": 2,
        "forecast_rows_checked": 507,
        "replay_mismatches": 6,
        "cars_checked_in_e_prime_i": 31,
        "true_row_violations_in_e_prime_i": 5,
        "cars_outside_e_prime_i_breaking_a_true_row": 7,
        "paths_with_request_missed": 4,
    }
    return {
        "configuration": {
            "rho": 0.0,
            "budget_kwh": 5.0,
            "energy_bounds": "known",
            "uncertifiable_car": "leave-unsteered",
            "kappa": 0.1,
            "departure_floor": "as-planned",
            "forecast_departure": "empirical kappa-quantile by site and arrival hour",
            "forecast_reference": "other-days",
            "forecast_min_cell": 5,
        },
        "sample": {
            "size": 10,
            "pass_mark_size": 100,
            "seed": 2026,
            "label": "t2-check",
            "counting_days": {"d1": 30, "d2": 20},
            "sampled_days": {"d1": 6, "d2": 4},
        },
        "counting_days": 50,
        "checks": {
            "order": verdict_block(50),
            "t4_log": t4,
            "t2": t2,
            "two_way": {**verdict_block(10), "days_both_products_matched": 9},
        },
    }


def tier_block(tier, offer, offering=150, points=(0.0, 5.0, 12.0), **more) -> dict:
    """One tier of one configuration as `dial_splits.summarise` writes it (200 splits)."""
    found = {
        "splits": 200,
        "mean_offer_kwh_per_test_day": offer,
        "offer_p10": points[0],
        "offer_p50": points[1],
        "offer_p90": points[2],
        "splits_that_offer": offering,
        "share_of_splits_that_offer": offering / 200,
        "mean_scale": 0.4,
        "splits_at_scale_zero_by_reason": {},
        "splits_with_test_losses": offering,
        "mean_test_slot_loss": 0.03,
        "mean_test_driver_loss": 0.02,
        "slot_loss_over_limit": {"limit": 0.1, "splits": 15, "share_of_splits_that_offer": 0.1},
        "splits_with_ceiling_share": offering,
        "mean_share_of_best_causal_ceiling": 0.35,
    }
    if tier == "A":
        found["driver_loss_over_limit"] = {
            "limit": 0.05,
            "splits": 3,
            "share_of_splits_that_offer": 0.02,
        }
    return {**found, **more}


def gate_block(offer, oracle, ceilings: dict, splits=200) -> dict:
    """The gate reading of one configuration as `dial_splits.summarise_gate` writes it;
    `ceilings`: name -> (mean ceiling, splits left out)."""
    by_ceiling = {
        name: {
            "splits_with_ceiling": splits - left,
            "splits_left_out_ceiling_missing": left,
            "mean_ceiling_kwh_per_test_day": ceiling,
            "mean_offer_kwh_per_test_day_on_those_splits": offer,
            "share_of_ceiling": offer / ceiling,
        }
        for name, (ceiling, left) in ceilings.items()
    }
    return {
        "tier": "A",
        "splits": splits,
        "mean_offer_kwh_per_test_day": offer,
        "mean_oracle_offer_kwh_per_test_day": oracle,
        "oracle_to_offer_ratio": oracle / offer,
        "oracle_to_offer_ratio_note": "oracle mean over offer mean",
        "ceilings": by_ceiling,
        "oracle_partner": "oracle",
    }


CEILINGS = ("hindsight-free-decision-hour", "best-causal")


def dial_choice_document() -> dict:
    """`dial-choice.json` in the writers' keys (`dial_choice.run`), made-up numbers: unit d1 with
    the configuration without rows and the dials 0.01 and 0.02 (tier A best dial 0.02, tier B
    best dial 0.01) and unit d2, too short to split."""
    offers = {  # configuration -> (dial, tier A offer, tier B offer, p10, p50, p90)
        "cfg-none": (None, 3.0, 4.0, (0.0, 2.5, 6.0)),
        "cfg-0.01": (0.01, 5.5, 9.5, (0.0, 4.5, 11.0)),
        "cfg-0.02": (0.02, 7.25, 8.5, (0.5, 6.5, 15.0)),
    }
    configurations = {
        name: {
            "driver_risk": dial,
            "offering_days": 18,
            "evaluations": {},
            "tiers": {
                "A": tier_block("A", a, points=points),
                "B": tier_block("B", b, points=points),
            },
            "gate_reading": gate_block(a, a * 3, {CEILINGS[0]: (64.0, 0), CEILINGS[1]: (40.0, 2)}),
        }
        for name, (dial, a, b, points) in offers.items()
    }
    return {
        "label": "dial choice: measured, not fixed",
        "model": "m1",
        "run": {
            "splits": 200,
            "seeds": [1, 200],
            "kappa": 0.3,
            "tiers": ["A", "B"],
            "rung": "one-way",
            "dials": [0.01, 0.02],
            "ceilings": list(CEILINGS),
            "gate_ceiling": CEILINGS[0],
            "thresholds": {"share_of_ceiling": 0.5, "oracle_to_offer_ratio": 2.0},
        },
        "units": {
            "d1": {
                "counting_days": 30,
                "calibration_days_used": 10,
                "test_days": 20,
                "status": "calibratable",
                "configurations": configurations,
            },
            "d2": {
                "counting_days": 3,
                "calibration_days_used": 0,
                "test_days": 0,
                "status": "not calibratable: 3 days",
                "configurations": {},
            },
        },
        "dial_choice": {
            "d1": {
                "A": {"dial": 0.02, "mean_offer_kwh_per_test_day": 7.25},
                "B": {"dial": 0.01, "mean_offer_kwh_per_test_day": 9.5},
            }
        },
        "timeouts": {"certificates": 1},
        "failed_configurations": [],
    }


class Numbers:
    """Made-up numbers that never repeat, in three ranges that cannot meet: whole numbers from 1001,
    reals 3000.25, 3000.75, ... (never whole), and shares (2k+1)/4096 below 1 (an odd number over
    4096 is never a whole number over 200, nor a real of the other range)."""

    def __init__(self):
        self.wholes = itertools.count(1001)
        self.reals = itertools.count()
        self.shares = itertools.count()

    def whole(self) -> int:
        return next(self.wholes)

    def real(self) -> float:
        return 3000.25 + 0.5 * next(self.reals)

    def share(self) -> float:
        return (2 * next(self.shares) + 1) / 4096


def distinct_dev_days_document() -> dict:
    """`dev-day-checks.json` as above, but every figure has a value of its own wherever the file's
    own rules allow (so that two labels swapped between keys show). The rules force equal values for
    these keys: `counting_days`, `checks.order.of_days` and `checks.t4_log.of_days` (9000);
    `sample.size`, `checks.t2.of_days` and `checks.two_way.of_days` (800); `hypothesis_days` and the
    first T4 group's `days` (3000). A pass needs `covered_days` equal to `of_days`, so the verdicts
    here are "failed" with a line, or "could not run" with a reason."""
    n = Numbers()

    def block(of_days, covered, verdict, lines=(), reasons=()):
        found = verdict_block(of_days, covered, lines, verdict, reasons)
        found["timeout_days"], found["other_missing_days"] = n.whole(), n.whole()
        return found

    def group(days, both, below, with_affine, affine_below):
        return t4_group(days, both, below, n.real(), with_affine, affine_below, n.real())

    groups = {
        "meets T4's hypotheses": group(3000, 2701, 211, 2603, 307),
        "other days, no in-window departure": group(2900, 2711, 223, 2509, 317),
        "other days, in-window departure": group(3100, 2811, 233, 2403, 331),
    }
    t4 = {
        **block(9000, 8993, "failed", ["t4 line"]),
        "groups": groups,
        "hypothesis_days": 3000,
        "hypothesis_days_checked": 2801,
        "findings_tree_above_affine": ["finding day"],
    }
    mismatches, rows = n.whole(), n.whole()  # a count and the count it is part of, in that order
    violations, checked = n.whole(), n.whole()
    t2 = {
        **block(800, 797, "could not run", reasons=["no certificate"]),
        "certified_days": 700,
        "days_without_certificate_by_rule": {
            "no certificate: no steered car": n.whole(),
            "no certificate: no forecast": n.whole(),
        },
        "days_in_e_prime": 280,
        "share_of_sampled_days_in_e_prime": 280 / 800,
        "share_of_certified_days_in_e_prime": 280 / 700,
        "steered_cars_on_certified_days": n.whole(),
        "share_of_steered_cars_in_e_prime_i": n.share(),
        "cars_on_sampled_days": n.whole(),
        "cars_left_unsteered_no_forecast": n.whole(),
        "cars_left_unsteered_empty_forecast_set": n.whole(),
        "forecast_rows_checked": rows,
        "replay_mismatches": mismatches,
        "cars_checked_in_e_prime_i": checked,
        "true_row_violations_in_e_prime_i": violations,
        "cars_outside_e_prime_i_breaking_a_true_row": n.whole(),
        "paths_with_request_missed": n.whole(),
    }
    return {
        "configuration": {
            "rho": n.share(),
            "budget_kwh": n.real(),
            "energy_bounds": "known",
            "uncertifiable_car": "leave-unsteered",
            "kappa": n.share(),
            "departure_floor": "as-planned",
            "forecast_departure": "empirical kappa-quantile by site and arrival hour",
            "forecast_reference": "other-days",
            "forecast_min_cell": n.whole(),
        },
        "sample": {
            "size": 800,
            "pass_mark_size": n.whole(),
            "seed": n.whole(),
            "label": "t2-check",
            "counting_days": {"d1": 5500, "d2": 3500},
            "sampled_days": {"d1": 450, "d2": 350},
        },
        "counting_days": 9000,
        "checks": {
            "order": block(9000, 8991, "failed", ["order line one", "order line two"]),
            "t4_log": t4,
            "t2": t2,
            "two_way": {
                **block(800, 791, "failed", ["two-way line"]),
                "days_both_products_matched": 740,
            },
        },
    }


def distinct_dial_choice_document() -> dict:
    """`dial-choice.json` as above, but every figure has a value of its own wherever the file's own
    rules allow. The rules force equal values for: the chosen dial and offer of a unit and tier and
    the dial and offer of the configuration they name; the two dials of `run.dials` and the chosen
    dials; `run.gate_ceiling` and the first of `run.ceilings`; the zero calibration days and zero
    test days of the unit too short to split. Unit d1 has the configurations without rows, with the
    dial 0.01 and with the dial 0.02; the offers grow from one configuration to the next, so that
    the dial 0.02 has the largest tier A offer and the dial 0.01 the largest tier B offer."""
    n = Numbers()
    offering = iter(range(61, 200, 11))  # splits that offer, below the 200 splits of a tier
    totals = iter(range(300, 400, 7))  # splits of a gate reading
    left_out = iter(range(3, 100, 5))  # splits left out for want of a ceiling

    def tier(name):
        low, middle, high = n.real(), n.real(), n.real()
        offers = next(offering)
        found = {
            "splits": 200,
            "mean_offer_kwh_per_test_day": n.real(),
            "offer_p10": low,
            "offer_p50": middle,
            "offer_p90": high,
            "splits_that_offer": offers,
            "share_of_splits_that_offer": offers / 200,
            "mean_scale": n.share(),
            "splits_at_scale_zero_by_reason": {},
            "splits_with_test_losses": offers,
            "mean_test_slot_loss": n.share(),
            "mean_test_driver_loss": n.share(),
            "slot_loss_over_limit": {
                "limit": 0.1,
                "splits": 15,
                "share_of_splits_that_offer": n.share(),
            },
            "splits_with_ceiling_share": offers,
            "mean_share_of_best_causal_ceiling": n.share(),
        }
        if name == "A":
            found["driver_loss_over_limit"] = {
                "limit": 0.05,
                "splits": 3,
                "share_of_splits_that_offer": n.share(),
            }
        return found

    def gate():
        total, offer, oracle = next(totals), n.real(), n.real()
        ceilings = {}
        for name in CEILINGS:
            left, mean, on_those = next(left_out), n.real(), n.real()
            ceilings[name] = {
                "splits_with_ceiling": total - left,
                "splits_left_out_ceiling_missing": left,
                "mean_ceiling_kwh_per_test_day": mean,
                "mean_offer_kwh_per_test_day_on_those_splits": on_those,
                "share_of_ceiling": on_those / mean,
            }
        return {
            "tier": "A",
            "splits": total,
            "mean_offer_kwh_per_test_day": offer,
            "mean_oracle_offer_kwh_per_test_day": oracle,
            "oracle_to_offer_ratio": oracle / offer,
            "oracle_to_offer_ratio_note": "oracle mean over offer mean",
            "ceilings": ceilings,
            "oracle_partner": "oracle",
        }

    configurations = {}
    for name, dial in (("cfg-none", None), ("cfg-0.01", 0.01), ("cfg-0.02", 0.02)):
        configurations[name] = {
            "driver_risk": dial,
            "offering_days": 18,
            "evaluations": {},
            "tiers": {"A": tier("A"), "B": tier("B")},
            "gate_reading": gate(),
        }
    best_b = configurations["cfg-0.01"]["tiers"]["B"]
    best_b["mean_offer_kwh_per_test_day"] = n.real()  # larger than the other two tier B offers
    chosen_a = configurations["cfg-0.02"]["tiers"]["A"]["mean_offer_kwh_per_test_day"]
    chosen = {
        "A": {"dial": 0.02, "mean_offer_kwh_per_test_day": chosen_a},
        "B": {"dial": 0.01, "mean_offer_kwh_per_test_day": best_b["mean_offer_kwh_per_test_day"]},
    }
    return {
        "label": "dial choice: measured, not fixed",
        "model": "m1",
        "run": {
            "splits": n.whole(),
            "seeds": [1, 200],
            "kappa": n.share(),
            "tiers": ["A", "B"],
            "rung": "one-way",
            "dials": [0.01, 0.02],
            "ceilings": list(CEILINGS),
            "gate_ceiling": CEILINGS[0],
            "thresholds": {"share_of_ceiling": n.share(), "oracle_to_offer_ratio": n.real()},
        },
        "units": {
            "d1": {
                "counting_days": n.whole(),
                "calibration_days_used": n.whole(),
                "test_days": n.whole(),
                "status": "calibratable",
                "configurations": configurations,
            },
            "d2": {
                "counting_days": 4,
                "calibration_days_used": 0,
                "test_days": 0,
                "status": "not calibratable: 4 days",
                "configurations": {},
            },
        },
        "dial_choice": {"d1": chosen},
        "timeouts": {"certificates": n.whole()},
        "failed_configurations": [],
    }


class MadeUp:
    """A made-up top folder with run folders and a drive log (module text)."""

    def __init__(self, top: Path):
        self.top = Path(top)
        self.count = 0
        self.list_sha256 = "0" * 64  # that of the run list committed, when there is one
        self.commands: dict[str, list] = {}  # label -> the step's command as logged

    def command(self, label: str) -> list:
        return self.commands.get(label, ["made-up", label])

    def commit_file(self, relative: str, document) -> bytes:
        data = dumps(document)
        if relative == "run-lists/run-list.json":
            self.list_sha256 = sha(data)
        path = self.top / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def run(self, label: str, files: dict, exit_code: int = 0) -> dict:
        """A run folder of `label` with `files` (name -> a JSON-able document, text or bytes);
        the driver's `entry` fields about it."""
        self.count += 1
        relative = f"runs.noindex/20261123T12{self.count:02d}00Z-{label}"
        folder = self.top / relative
        folder.mkdir(parents=True)
        outputs = {}
        for name, content in files.items():
            if isinstance(content, bytes):
                data = content
            elif isinstance(content, str):
                data = content.encode()
            else:
                data = dumps(content)
            (folder / name).write_bytes(data)
            outputs[name] = sha(data)
        manifest = {
            "commit": COMMIT,
            "command": self.command(label),
            "start": "2026-11-23T12:05:00Z",
            "end": "2026-11-23T12:06:00Z",
            "exit_code": exit_code,
            "outputs": outputs,
            "long": True,
        }
        data = dumps(manifest)
        (folder / "manifest.json").write_bytes(data)
        return {
            "run": relative,
            "exit_status": exit_code,
            "manifest_sha256": sha(data),
        }

    def drive(self, steps: list, stamp: str = "20261123T120420Z", stop: bool = False) -> Path:
        """Write a drive log. `steps`: (label, outcome) in order; outcome is a dict of files (a
        step that passed), "fail" (a step that exited 1, with a `failed` line) or "skip" (a step
        not started, with a `not_run` line)."""
        records = [
            {
                "record": "start",
                "format": LOG_FORMAT,
                "start": "2026-11-23T12:04:20Z",
                "commit": COMMIT,
                "list": "run-lists/run-list.json",
                "list_sha256": self.list_sha256,
                "purpose": "sealed-run",
                "tag_commit": TAG,
                "entries": len(steps),
                "steps": [
                    {"label": label, "command": self.command(label), "max_seconds": 10}
                    for label, _ in steps
                ],
                "pid": 1,
            }
        ]
        failed, not_run, ran = [], [], 0
        for k, (label, outcome) in enumerate(steps):
            if stop and outcome == "skip":
                records.append({"record": "stop", "reason": "made-up stop"})
                break
            if outcome == "skip":
                line = {
                    "record": "not_run",
                    "index": k,
                    "label": label,
                    "reason": "an earlier step",
                }
                records.append(line)
                not_run.append({key: line[key] for key in ("index", "label", "reason")})
                continue
            records.append(
                {
                    "record": "begin",
                    "index": k,
                    "label": label,
                    "command": self.command(label),
                    "max_seconds": 10,
                    "start": "2026-11-23T12:05:00Z",
                }
            )
            found = self.run(
                label, {} if outcome == "fail" else outcome, 1 if outcome == "fail" else 0
            )
            entry = {
                "record": "entry",
                "index": k,
                "label": label,
                "timed_out": False,
                "interrupted": None,
                "start": "2026-11-23T12:05:00Z",
                "end": "2026-11-23T12:06:00Z",
                "seconds": 60.0,
                "recorder_status": found["exit_status"],
                "recorder_output": "",
                **found,
            }
            records.append(entry)
            ran += 1
            if outcome == "fail":
                line = {"record": "failed", "index": k, "label": label, "reason": "exit status 1"}
                records.append(line)
                failed.append({key: line[key] for key in ("index", "label", "reason")})
        completed = not failed and not not_run and not stop
        above = b"".join(encode(r) for r in records)
        end = {
            "record": "end",
            "end": "2026-11-23T12:30:00Z",
            "seconds": 100.0,
            "entries_run": ran,
            "completed": completed,
            "exit_status": 0 if completed else 1,
            "unchanged_after": True,
            "change_after": None,
            "failed": failed,
            "not_run": not_run,
        }
        data = above + encode({**end, "log_sha256": end_digest(above, end)})
        path = self.top / "runs.noindex" / "drives" / f"{stamp}-sealed-run.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
