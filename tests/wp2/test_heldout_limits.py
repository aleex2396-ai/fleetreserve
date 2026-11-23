"""The held-out loaders' removal limits, kept in `heldout_limits` and nowhere else.

A parsing fix after the tag may change how a held-out file is read, never a removal rule or its
limit; the opener accepts a change to a loader and its tests only, so a limit in a module of its
own cannot be changed by a fix. This file pins each value as it was when the limits lay in the
loaders, and checks that each loader reads the limit from that module (a loader that held its
own copy would let a fix change it).
"""

import ast
from pathlib import Path

from fleetreserve import heldout_limits, load_asr, load_heldout_norway, load_korea


def test_the_values_are_those_the_loaders_held_before_the_limits_were_moved():
    assert heldout_limits.LIMIT_KW == 22.0
    assert heldout_limits.ENERGY_ABOVE_KWH == 0
    assert (heldout_limits.SLOW, heldout_limits.FAST) == (0.0, 1.0)
    assert heldout_limits.FIRST_TIME == "2021-09-01 00:00:00"
    assert heldout_limits.LAST_TIME == "2022-10-31 23:59:59"


def test_each_loader_holds_the_limit_it_uses_from_the_module():
    assert load_asr.LIMIT_KW is heldout_limits.LIMIT_KW
    assert load_asr.ENERGY_ABOVE_KWH is heldout_limits.ENERGY_ABOVE_KWH
    for name in ("LIMIT_KW", "ENERGY_ABOVE_KWH", "SLOW", "FAST", "FIRST_TIME", "LAST_TIME"):
        assert getattr(load_korea, name) is getattr(heldout_limits, name)


def _assigned(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names |= {t.id for t in ast.walk(target) if isinstance(t, ast.Name)}
    return names


def test_no_loader_assigns_a_limit_itself():
    limits = {name for name in vars(heldout_limits) if name.isupper()}
    for module in (load_asr, load_korea, load_heldout_norway):
        assert not _assigned(module) & limits, module.__name__


def test_the_limits_module_is_no_loader_module():
    assert heldout_limits.__name__ not in {
        load_asr.__name__,
        load_korea.__name__,
        load_heldout_norway.__name__,
    }
