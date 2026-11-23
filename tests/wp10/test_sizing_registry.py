"""The sizing methods in the registry (`compare_methods`).

1. `also-x` and `cvar` come last, after `max-entropy-feedback`, of kind "sizing", with the rules
   `chance_sizing.also_x_rule` and `cvar_rule` and no builder; `KINDS` holds the three kinds.
2. `select` refuses a sizing method with a builder or without a rule, a controller or a re-solve
   with a rule, and one name registered with two different rules.
"""

import pytest

from fleetreserve import compare_methods
from fleetreserve.chance_sizing import also_x_rule, cvar_rule
from fleetreserve.compare_methods import Method, MethodError


def test_the_sizing_methods_come_last_with_their_rules():
    names = [m.name for m in compare_methods.METHODS]
    assert names[-3:] == ["max-entropy-feedback", "also-x", "cvar"]
    also_x, cvar = compare_methods.METHODS[-2:]
    assert also_x == Method("also-x", "sizing", None, also_x_rule)
    assert cvar == Method("cvar", "sizing", None, cvar_rule)
    assert compare_methods.KINDS == ("controller", "re-solve", "sizing")
    assert [m.kind for m in compare_methods.select(["cvar", "re-solve"])] == ["sizing", "re-solve"]


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        (
            (Method("x", "sizing", compare_methods.llf, also_x_rule),),
            "a sizing method has no builder",
        ),
        ((Method("x", "sizing"),), "a sizing method needs a rule"),
        (
            (Method("x", "controller", compare_methods.llf, also_x_rule),),
            "only a sizing method has a rule",
        ),
        ((Method("x", "re-solve", None, also_x_rule),), "only a sizing method has a rule"),
        (
            (*compare_methods.METHODS, Method("also-x", "sizing", None, cvar_rule)),
            "'also-x' is registered twice",
        ),
    ],
)
def test_select_refuses(registry, message):
    with pytest.raises(MethodError, match=message):
        compare_methods.select(None, registry)
