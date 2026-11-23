"""The method registry (`compare_methods`), an abstaining day, and one method under two names.

1. `select` gives every method of the registry in its order, or the named ones; it refuses an
   unknown name, a name asked twice, a name registered twice with a different builder, a bad
   kind, a controller without a builder and a re-solve with one. The same entry listed twice is
   one method.
2. On a day that abstains, and at a scale of 0, every controller method has losses 0 and no
   controller is built.
3. A method registered under two names gets identical rows on a made-up gate unit: the same
   splits, calibration days, test days and losses.
4. Which rule each name runs: the builders of `edf`, `llf` and `smoothed-llf` return
   `priority_rules.earliest_deadline_first`, `least_laxity_first` and `smoothed_least_laxity`;
   and on 40 seeded made-up offer days at three scales each name's day-level losses are those of
   its own rule run through `compare_day.evaluate` under another name, while every two of the three
   rules differ on at least one day (so a name that ran another rule would be seen).
"""

import dataclasses

import pytest
from compare_fake import LOW, NAMES, offer_days
from gate_fake import gate_cars, matrix_document, write_tables

from fleetreserve import (
    compare_day,
    compare_methods,
    compare_run,
    dial_choice,
    gate_config,
    gate_day,
    gate_run,
)
from fleetreserve.compare_methods import Method, MethodError
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.plans import Plan
from fleetreserve.priority_rules import (
    earliest_deadline_first,
    least_laxity_first,
    smoothed_least_laxity,
)

# --- 1. the registry -----------------------------------------------------------------------------


def test_the_registry_lists_the_methods_of_this_unit_in_order():
    names = [m.name for m in compare_methods.select(None)]
    first = ["re-solve", "fixed-plan", "edf", "llf", "smoothed-llf", "asa", "asa-es"]
    assert names[: len(first)] == first
    assert names == [m.name for m in compare_methods.METHODS]
    chosen = compare_methods.select(["llf", "re-solve"])
    assert [m.name for m in chosen] == ["llf", "re-solve"]
    assert [m.kind for m in chosen] == ["controller", "re-solve"]


def _other(inputs):
    return compare_methods.llf(inputs)


@pytest.mark.parametrize(
    ("registry", "names", "message"),
    [
        (None, ["no-such"], "unknown method 'no-such'"),
        (None, ["edf", "edf"], "a method is named twice"),
        (
            (*compare_methods.METHODS, Method("edf", "controller", _other)),
            None,
            "'edf' is registered twice with different controllers",
        ),
        ((Method("x", "rung", _other),), None, "kind must be one of"),
        ((Method("x", "controller"),), None, "a controller needs a builder"),
        ((Method("x", "re-solve", _other),), None, "the re-solve rung has no builder"),
    ],
)
def test_select_refuses(registry, names, message):
    with pytest.raises(MethodError, match=message):
        compare_methods.select(names, registry)


def test_the_same_entry_twice_is_one_method():
    twice = (*compare_methods.METHODS, compare_methods.METHODS[2])
    assert compare_methods.select(None, twice) == compare_methods.METHODS


# --- 2. an abstaining day ------------------------------------------------------------------------


def _never(inputs):
    raise AssertionError("a controller was built on a day that offers nothing")


NEVER = (Method("never", "controller", _never), *compare_methods.select(["edf", "fixed-plan"]))


def test_an_abstaining_day_has_no_loss_for_any_method_and_builds_no_controller():
    (prepared,) = offer_days(1)
    abstaining = dataclasses.replace(prepared, abstain_reason="no steered car", value=0.0)
    for beta in (0.5, 1.0):
        found = compare_day.evaluate(abstaining, beta, NEVER)
        assert set(found) == {"never", "edf", "fixed-plan"}
        assert all(d == compare_day.MethodDay(0.0, 0.0, ()) for d in found.values())
    for beta in (0.0, -1.0):
        found = compare_day.evaluate(prepared, beta, NEVER)
        assert all(d == compare_day.MethodDay(0.0, 0.0, ()) for d in found.values())


def test_the_re_solve_method_is_not_a_controller():
    (prepared,) = offer_days(1)
    with pytest.raises(ValueError, match="'re-solve' is not a controller"):
        compare_day.evaluate(prepared, 1.0, compare_methods.select(["re-solve"]))


# --- 3. one method under two names ---------------------------------------------------------------


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    folder = tmp_path_factory.mktemp("unit")
    write_tables(folder, "u", gate_cars(60, early_every=3))
    days = load_unit_days(folder, "u", rho=0.0, no_fit="cap", small_fleet="keep")
    matrix = gate_config.parse(matrix_document(NAMES))
    configuration = matrix.configuration(LOW[0])
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    settings = gate_run.settings(matrix)
    prepared = [gate_day.prepare_day(t, forecaster, configuration, settings) for t in days.days]
    splits = dial_choice.split_indices("u", days, matrix, range(1, 9))
    return prepared, splits


def test_a_method_under_two_names_gets_identical_rows(unit):
    prepared, splits = unit
    twin = Method("fixed-plan-twin", "controller", compare_methods.fixed_plan)
    methods = (*compare_methods.select(["fixed-plan", "llf"]), twin)
    losses, counts = compare_run.measure_variant(
        prepared,
        [None] * len(prepared),
        methods,
        None,
        lambda jobs: [compare_day.evaluate(*job) for job in jobs],
    )
    assert counts["controller_day_scale_pairs"] > 0
    rows = compare_run.score_methods(losses, splits, ("A", "B"))
    assert rows["fixed-plan-twin"] == rows["fixed-plan"]
    assert (losses["fixed-plan-twin"].slot == losses["fixed-plan"].slot).all()
    assert (losses["fixed-plan-twin"].driver == losses["fixed-plan"].driver).all()
    assert losses["fixed-plan"].slot.any(), "the unit must hold losses"
    assert rows["fixed-plan"]["B"]["splits"] == len(splits) == 8


def test_the_re_solve_losses_and_rows_are_dial_choices(unit):
    prepared, splits = unit

    def batch(pairs):
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    losses, counts = compare_run.measure_variant(
        prepared, [None] * len(prepared), compare_methods.select(["re-solve"]), batch, None
    )
    expected, cache = dial_choice.measure(prepared, [None] * len(prepared), batch)
    found = losses["re-solve"]
    for field in ("value", "offering", "slot", "driver"):
        assert (getattr(found, field) == getattr(expected, field)).all(), field
    assert counts["re_solve_day_scale_pairs"] == len(cache.calls)
    rows = compare_run.score_methods(losses, splits, ("A", "B"))
    assert rows["re-solve"] == dial_choice.score_splits(expected, splits, ("A", "B"))
    assert expected.slot.any(), "the unit must hold re-solve losses"


# --- 4. which rule each name runs ----------------------------------------------------------------


def _own_edf(inputs):
    return earliest_deadline_first


def _own_llf(inputs):
    return least_laxity_first


def _own_smoothed(inputs):
    return smoothed_least_laxity


RULES = {
    "edf": (earliest_deadline_first, _own_edf),
    "llf": (least_laxity_first, _own_llf),
    "smoothed-llf": (smoothed_least_laxity, _own_smoothed),
}


def test_each_rule_name_builds_its_rule():
    (prepared,) = offer_days(1)
    policy = prepared.certificate.policy
    inputs = compare_methods.MethodInput(prepared.day, policy, 0.5, Plan.from_policy(policy, 0.5))
    for name, (rule, _) in RULES.items():
        (method,) = compare_methods.select([name])
        assert method.build(inputs) is rule, name


def test_each_rule_name_gives_the_losses_of_its_rule():
    named = compare_methods.select(list(RULES))
    own = tuple(Method(f"own-{n}", "controller", b) for n, (_, b) in RULES.items())
    seen: dict = {}
    for k, prepared in enumerate(offer_days(40)):
        for beta in (0.3, 0.7, 1.0):
            found = compare_day.evaluate(prepared, beta, (*named, *own))
            for name in RULES:
                assert found[name] == found[f"own-{name}"], (k, beta, name)
                seen.setdefault(name, []).append(found[name])
    names = list(RULES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert any(x != y for x, y in zip(seen[a], seen[b], strict=True)), (a, b)
