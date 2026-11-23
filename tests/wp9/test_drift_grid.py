"""The measurement grid of the drift guard's step size (`drift_run.ETAS`, reading E9; plan section
6, T10) reaches above 1.

1. The grid is 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 20, 50, distinct and in
   increasing order, the step sizes `--etas` accepts, and what the command runs when `--etas` is
   not given (off the route).
2. T10's stated cases hold at every step size of the grid, exact fractions: clipped at 0 the
   average loss is 0.500, unclipped it is 0.103 at t = 116, and it obeys alpha + (1 - alpha)/t
   from beta_1 = 0 at every t.
3. A step above 1 is valid, not merely unmeasured: the scale offered never exceeds the cap (the
   state, unclipped, can), the state never goes below -eta(1 - alpha), and the final mean loss is
   never above the bound, whatever the step.
4. Worked by hand on one made-up unit (`unit`): two calibration days, then three guarded days.
   Day 1 offers 1 kWh and day 2 offers 1 kWh, day 3 does not offer; both guards start at 0, so
   day 1 abstains and raises the slot state to 0.1 eta and the driver state to 0.05 eta; day 2 is
   offered the smaller of the two, capped at 1: 0.05 eta in tier A, 0.1 eta in tier B. The mean
   offer per counting day is that scale over the three days.
   (a) No loss anywhere. Tier B: the offer is 0.1 eta / 3 up to eta 10 and 1/3 from eta 10, so
       the rule "largest offer among the steps within alpha" takes 10 (a tie goes to the smaller
       step size), above 1. Tier A: 0.05 eta / 3 up to eta 20 and 1/3 from there, the pick is 20.
   (b) A day at the full scale 1 loses (slot and driver loss 1), a smaller scale loses nothing.
       Tier B: eta 10, 20 and 50 reach the scale 1 on day 2, the mean loss over the two offering
       days is 0.5 against alpha 0.1, so they are not within alpha, and the rule takes 7 (offer
       0.7/3), where the rule without the condition takes 10. Tier A: eta 20 and 50 reach scale 1
       and break their levels; the rule takes 10 (0.5/3), the rule without the condition 20.
   A large step that is within alpha and offers the most is picked, and one that breaks its level
   is not, however large its offer.
"""

import math
from fractions import Fraction
from types import SimpleNamespace

import pytest
from gate_fake import matrix_document

from fleetreserve import drift_guard, drift_path, drift_run, drift_settings
from fleetreserve.gate_config import parse

GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0)
OLD_GRID = GRID[:7]
ALPHA = Fraction(1, 10)


def _fraction(eta: float) -> Fraction:
    return Fraction(drift_run.eta_key(eta))


# --- 1. the grid ---------------------------------------------------------------------------------


def test_the_grid_is_the_fourteen_step_sizes():
    assert drift_run.ETAS == GRID
    assert list(GRID) == sorted(set(GRID))
    assert drift_run.check_etas([drift_run.eta_key(e) for e in drift_run.ETAS]) == GRID


def test_the_command_runs_the_grid_when_no_step_size_is_given():
    arguments = SimpleNamespace(etas=None, calibration_days=None)
    assert drift_run.counts(arguments, held_out=False) == (drift_path.N_CALIBRATION, GRID)


@pytest.mark.parametrize("eta", GRID)
def test_every_step_size_is_written_and_read_back_as_itself(eta):
    key = drift_run.eta_key(eta)
    assert float(key) == eta
    assert "e" not in key


# --- 2. T10's stated cases at every step size ----------------------------------------------------


def _lose_if_positive(beta):
    return 1 if beta > 0 else 0


def _average(losses):
    return Fraction(sum(losses), len(losses))


@pytest.mark.parametrize("eta", GRID)
def test_t10_stated_cases_hold_at_every_step_size_of_the_grid(eta):
    step = _fraction(eta)
    clipped = drift_guard.run(_lose_if_positive, 116, ALPHA, step, clip=True)
    assert round(float(_average(clipped)), 3) == 0.500
    unclipped = drift_guard.run(_lose_if_positive, 116, ALPHA, step, clip=False)
    assert round(float(_average(unclipped)), 3) == 0.103
    losses = drift_guard.run(_lose_if_positive, 400, ALPHA, step, beta_1=Fraction(0), clip=False)
    for t in range(1, len(losses) + 1):
        assert _average(losses[:t]) <= ALPHA + (1 - ALPHA) / t


# --- 3 and 4. one made-up unit -------------------------------------------------------------------

N_CAL = 2
KEYS = [f"2020-01-{d + 1:02d} U" for d in range(5)]
OFFERING = [True, True, True, True, False]
VALUES = [1.0, 1.0, 1.0, 1.0, 0.0]


def no_loss(pairs):
    return {(d, s): (0.0, 0.0) for d, s in pairs}


def full_scale_loses_on_day_2(pairs):
    return {(d, s): ((1.0, 1.0) if d == 3 and s >= 1.0 else (0.0, 0.0)) for d, s in pairs}


def unit(evaluate, etas=None):
    etas = drift_run.ETAS if etas is None else etas
    return drift_path.run_unit(
        KEYS, VALUES, OFFERING, evaluate, n_calibration=N_CAL, etas=etas, starts=("zero",)
    )


def _document(found, etas=None):
    etas = drift_run.ETAS if etas is None else etas
    names = ("main-rounded-hour-k0.3-eps0.02-empirical", "main-rounded-hour-k0.3-eps0.02-oracle")
    configuration = parse(matrix_document(names)).configuration(names[0])
    entry = drift_run.unit_entry("U", found, len(KEYS), fit_units=("x",), rotation=None)
    return drift_run.results(
        "lgbm_tail_rotation", configuration, ["U"], etas, N_CAL, {"U": entry}, failed={}
    )


def _pick(document, tier, rule):
    return drift_settings.RULES[rule](drift_settings.figures(document, tier, "zero"))


def _paths(found, tier):
    return {p.eta: p for p in found.paths if p.tier == tier}


@pytest.mark.parametrize("evaluate", [no_loss, full_scale_loses_on_day_2])
def test_a_large_step_keeps_the_scale_under_the_cap_and_the_state_above_its_floor(evaluate):
    found = unit(evaluate)
    for path in found.paths:
        assert all(row.scale <= drift_path.CAP for row in path.rows)
        floor = -path.eta * (1 - 0.1)
        assert all(s >= floor - 1e-9 for row in path.rows for s in row.state_before.values())
        assert all(s >= floor - 1e-9 for s in path.state.values())
        summary = path.summary()
        for loss in summary["losses"].values():
            if loss["guarded"]:
                assert loss["days_over_bound"] == 0


def test_the_state_above_the_cap_is_counted_not_clipped():
    """No loss: after the abstaining first day the slot state is 0.1 eta; above 1 for eta 20, 50."""
    paths = _paths(unit(no_loss), "B")
    assert {e: p.summary()["days_state_above_cap"] for e, p in paths.items() if e >= 5} == {
        5.0: 0,
        7.0: 0,
        10.0: 0,
        20.0: 1,
        50.0: 1,
    }
    assert math.isclose(paths[50.0].rows[1].state_before["slot"], 5.0)
    assert paths[50.0].rows[1].scale == drift_path.CAP


def test_hand_case_a_the_largest_offer_among_the_steps_within_alpha_is_above_1():
    document = _document(unit(no_loss))
    third = 1 / 3
    offers = {
        tier: {e: f.offer for e, f in drift_settings.figures(document, tier, "zero").items()}
        for tier in drift_path.TIERS
    }
    for eta in GRID:
        assert offers["B"][eta] == pytest.approx(min(0.1 * eta, 1.0) * third)
        assert offers["A"][eta] == pytest.approx(min(0.05 * eta, 1.0) * third)
    rule = "largest-offer-within-alpha"
    assert _pick(document, "B", rule) == 10.0
    assert _pick(document, "A", rule) == 20.0


def test_hand_case_b_a_large_step_that_breaks_its_level_is_not_picked():
    document = _document(unit(full_scale_loses_on_day_2))
    within = {
        tier: {e: f.within_alpha for e, f in drift_settings.figures(document, tier, "zero").items()}
        for tier in drift_path.TIERS
    }
    assert [e for e, ok in within["B"].items() if not ok] == [10.0, 20.0, 50.0]
    assert [e for e, ok in within["A"].items() if not ok] == [20.0, 50.0]
    rule = "largest-offer-within-alpha"
    assert _pick(document, "B", rule) == 7.0
    assert _pick(document, "A", rule) == 10.0
    # without the condition the broken steps win: the condition, not the offer, keeps them out
    assert _pick(document, "B", "largest-offer") == 10.0
    assert _pick(document, "A", "largest-offer") == 20.0
    offer = drift_settings.figures(document, "B", "zero")
    assert offer[7.0].offer == pytest.approx(0.7 / 3)
    assert offer[10.0].offer == pytest.approx(1 / 3)


def test_the_old_seven_step_sizes_give_the_same_paths_inside_the_wider_grid():
    old = unit(full_scale_loses_on_day_2, OLD_GRID)
    new = unit(full_scale_loses_on_day_2)
    wanted = {(p.tier, p.start, p.eta): p.summary() for p in new.paths if p.eta in OLD_GRID}
    assert {(p.tier, p.start, p.eta): p.summary() for p in old.paths} == wanted
    assert [(r.day, r.scale, r.offer_kwh, r.losses) for p in old.paths for r in p.rows] == [
        (r.day, r.scale, r.offer_kwh, r.losses)
        for p in new.paths
        if p.eta in OLD_GRID
        for r in p.rows
    ]
