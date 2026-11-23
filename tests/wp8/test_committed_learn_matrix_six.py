"""The committed WP8 run matrix (`learn/matrix.json`) lists plan section 2's six levels of kappa
(0.05, 0.1, 0.2, 0.3, 0.4, 0.5), read from the file itself, not from a made-up matrix.

Hand-worked count. One family ("main" in this file: departures seen at the start of each hour),
an empirical configuration and its oracle partner at each level: 6 levels x 2 readings = 12
configurations, against 8 with four levels. Oracle partners with known energy bounds do not depend
on kappa (`Configuration.work_key`), so the 4 new configurations add 2 pieces of work (the
empirical ones) to the 5 of the four-level matrix (4 empirical, 1 shared oracle): 7.

1. The file lists every level of the grid, an empirical configuration and an oracle one at each.
2. A configuration at 0.4 or 0.5 is the one at 0.3 with only its level, its name and its oracle
   partner's name changed.
3. The eight configurations of the four-level file are there, unchanged and in their old order,
   first in the list; the two new levels follow in the order 0.4 and 0.5.
4. Every field of the file but the configurations and the text equals the gate matrix's.
5. The text names the six levels.
"""

import json

import pytest
from gate_fake import TOP

from fleetreserve import gate_config
from fleetreserve.forecast_empirical import KAPPAS

PATH = "learn/matrix.json"
OLD_LEVELS = (0.05, 0.1, 0.2, 0.3)
NEW_LEVELS = (0.4, 0.5)
TEXT_FIELDS = ("about", "gate_units", "rotation_whatif")
SHARED = {
    "family": "main",
    "energy_bounds": "known",
    "uncertifiable_car": "leave-unsteered",
    "shape_rule": "largest-smallest-width",
    "departure_floor": "as-planned",
    "driver_loss_cars": "steered-only",
    "departure_seen": "hour-start",
    "driver_risk": None,
}


def _document(path: str = PATH) -> dict:
    return json.loads((TOP / path).read_text())


def _old_entries() -> list[dict]:
    """The eight configurations of the four-level file, written out by hand."""
    entries = []
    for level in OLD_LEVELS:
        entries.append(
            {
                "name": f"main-k{level:g}-empirical",
                "kappa": level,
                "departure": "empirical",
                "oracle_partner": f"main-k{level:g}-oracle",
            }
        )
        entries.append(
            {
                "name": f"main-k{level:g}-oracle",
                "kappa": level,
                "departure": "oracle",
                "oracle_partner": None,
            }
        )
    return [{**e, **{k: v for k, v in SHARED.items() if k not in e}} for e in entries]


def test_the_file_lists_the_six_levels_with_an_oracle_partner_at_each():
    matrix = gate_config.load(TOP / PATH)
    assert KAPPAS == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    found = [(c.family, c.kappa, c.departure) for c in matrix.configurations]
    assert sorted(found) == sorted(("main", k, d) for k in KAPPAS for d in ("empirical", "oracle"))
    assert len(matrix.configurations) == 12
    assert len({c.work_key() for c in matrix.configurations}) == 7
    for c in matrix.configurations:
        if c.departure == "empirical":
            assert matrix.configuration(c.oracle_partner).kappa == c.kappa


@pytest.mark.parametrize("level", NEW_LEVELS)
def test_a_configuration_at_a_new_level_is_the_one_at_0_3_with_the_level_changed(level):
    configurations = {c["name"]: c for c in _document()["configurations"]}
    for departure in ("empirical", "oracle"):
        old = configurations[f"main-k0.3-{departure}"]
        made = configurations[f"main-k{level:g}-{departure}"]
        assert made["kappa"] == level and old["kappa"] == 0.3
        partner = old["oracle_partner"]
        assert made["oracle_partner"] == (
            None if partner is None else partner.replace("-k0.3-", f"-k{level:g}-")
        )
        assert {**made, "name": 0, "kappa": 0, "oracle_partner": 0} == {
            **old,
            "name": 0,
            "kappa": 0,
            "oracle_partner": 0,
        }


def test_the_eight_old_configurations_are_unchanged_and_first_then_0_4_then_0_5():
    configurations = _document()["configurations"]
    assert len(configurations) == 12
    assert configurations[:8] == _old_entries()
    assert [c["name"] for c in configurations[8:]] == [
        f"main-k{level:g}-{departure}"
        for level in NEW_LEVELS
        for departure in ("empirical", "oracle")
    ]


def test_every_field_but_the_configurations_and_the_text_equals_the_gate_matrix():
    learn = _document()
    gate = _document("gate/matrix.json")
    skip = {"configurations", *TEXT_FIELDS}
    assert {k: v for k, v in learn.items() if k not in skip} == {
        k: v for k, v in gate.items() if k not in skip
    }


def test_the_text_names_the_six_levels():
    levels = ", ".join(f"{k:g}" for k in KAPPAS)
    assert levels == "0.05, 0.1, 0.2, 0.3, 0.4, 0.5"
    assert f"at every kappa of plan section 2 ({levels})" in _document()["about"]
