"""The committed run matrices list plan section 2's six levels of kappa (0.05, 0.1, 0.2, 0.3, 0.4,
0.5), read from the files themselves (`gate/matrix.json`, `gate/t8-training-a.json`,
`gate/t8-training-b.json`), not from a made-up matrix.

Hand-worked count. The main family ("main", departures seen from the rounded-down hour) and the
family beside it ("beside", departures seen at the start of the next hour) each have, at each
level, an empirical configuration and its oracle partner: 2 families x 2 readings x 6 levels = 24
configurations. The 3 alternates at 0.1 with their partners (6) and the 8 dials at 0.3 with their
partners (16) are 22 more: 46 in all, against 38 with four levels (16 + 22 = 38; the difference
is 2 families x 2 readings x 2 new levels = 8). Oracle partners with known energy bounds do not
depend on kappa (`Configuration.work_key`), so the 8 new configurations add 4 pieces of work (the
empirical ones) to the 31 of the four-level matrix: 35.

1. Both families list every level of the grid, both readings of the departure.
2. A configuration at 0.4 or 0.5 is the one at 0.3 with only its level, its name and its oracle
   partner's name changed; every configuration of the four-level matrix is still there.
3. The names of the new configurations follow the existing pattern, `-k0.4-` and `-k0.5-`.
4. The two training matrices equal the gate matrix in every field but the text, the gate units and
   `rotation_whatif`.
5. The text of the gate matrix names the six levels; the training matrices' text still says they
   equal it.
6. With the committed matrix, the commands that exited 2 for want of the levels find their
   configurations: `dial_choice.family_at` at 0.4 and 0.5 (the plain configuration, the dial
   configurations and every oracle partner), and `dial_choice` read for `--what-if-fit-levels
   grid` (the matrix lists exactly the grid's six).
"""

import dataclasses
import json

import pytest
from gate_fake import TOP

from fleetreserve import dial_choice, driver_risk, gate_config
from fleetreserve.forecast_empirical import KAPPAS

FILES = ("gate/matrix.json", "gate/t8-training-a.json", "gate/t8-training-b.json")
NEW_LEVELS = (0.4, 0.5)
OLD_NAMES_AT_0_3 = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-k0.3-empirical",
    "main-k0.3-oracle",
)
TEXT_FIELDS = ("about", "gate_units", "rotation_whatif")


def _document(path: str) -> dict:
    return json.loads((TOP / path).read_text())


def _by_name(document: dict) -> dict:
    return {c["name"]: c for c in document["configurations"]}


@pytest.mark.parametrize("path", FILES)
def test_the_main_and_beside_families_list_the_six_levels(path):
    matrix = gate_config.load(TOP / path)
    assert KAPPAS == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    main = [c for c in matrix.configurations if c.family in ("main", "beside")]
    assert sorted({c.kappa for c in main}) == list(KAPPAS)
    found = [(c.family, c.kappa, c.departure) for c in main]
    assert sorted(found) == sorted(
        (f, k, d) for f in ("main", "beside") for k in KAPPAS for d in ("empirical", "oracle")
    )
    assert len(main) == 24
    assert len(matrix.configurations) == 46
    assert len({c.work_key() for c in matrix.configurations}) == 35


@pytest.mark.parametrize("path", FILES)
@pytest.mark.parametrize("level", NEW_LEVELS)
def test_a_configuration_at_a_new_level_is_the_one_at_0_3_with_the_level_changed(path, level):
    configurations = _by_name(_document(path))
    for name in OLD_NAMES_AT_0_3:
        made = configurations[name.replace("-k0.3-", f"-k{level:g}-")]
        old = configurations[name]
        assert made["kappa"] == level and old["kappa"] == 0.3
        partner = old["oracle_partner"]
        expected_partner = None if partner is None else partner.replace("-k0.3-", f"-k{level:g}-")
        assert made["oracle_partner"] == expected_partner
        assert {**made, "name": 0, "kappa": 0, "oracle_partner": 0} == {
            **old,
            "name": 0,
            "kappa": 0,
            "oracle_partner": 0,
        }


@pytest.mark.parametrize("path", FILES)
def test_no_configuration_of_the_four_level_matrix_was_dropped_or_renamed(path):
    names = set(_by_name(_document(path)))
    four_levels = {
        f"{family}-k{k:g}-{d}"
        for family in ("main", "main-rounded-hour")
        for k in (0.05, 0.1, 0.2, 0.3)
        for d in ("empirical", "oracle")
    }
    alternates = {
        f"alt-{what}-k0.1-{d}"
        for what in ("energy-empirical", "day-abstains", "floored")
        for d in ("empirical", "oracle")
    }
    dials = {
        f"main-rounded-hour-k0.3-eps{e:g}-{d}"
        for e in (0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
        for d in ("empirical", "oracle")
    }
    assert len(four_levels | alternates | dials) == 38
    assert (four_levels | alternates | dials) <= names
    assert len(names) == 46
    new = names - (four_levels | alternates | dials)
    assert new == {
        f"{family}-k{k:g}-{d}"
        for family in ("main", "main-rounded-hour")
        for k in NEW_LEVELS
        for d in ("empirical", "oracle")
    }


def test_the_training_matrices_equal_the_gate_matrix_but_for_their_text_and_units():
    gate = _document("gate/matrix.json")
    for path in FILES[1:]:
        training = _document(path)
        assert {k: v for k, v in training.items() if k not in TEXT_FIELDS} == {
            k: v for k, v in gate.items() if k not in TEXT_FIELDS
        }, path


def test_the_text_of_the_gate_matrix_names_the_six_levels():
    about = _document("gate/matrix.json")["about"]
    levels = ", ".join(f"{k:g}" for k in KAPPAS)
    assert levels == "0.05, 0.1, 0.2, 0.3, 0.4, 0.5"
    assert f"at every kappa of plan section 2 ({levels})" in about


@pytest.mark.parametrize("path", FILES[1:])
def test_the_text_of_a_training_matrix_still_says_it_equals_the_gate_matrix(path):
    about = _document(path)["about"]
    assert (
        "Equal to gate/matrix.json in every field but this text, gate_units and rotation_whatif"
        in about
    )


# --- 6. the commands that needed the levels --------------------------------------------------


@pytest.mark.parametrize("level", NEW_LEVELS)
def test_the_dial_family_is_found_at_the_new_levels_in_the_committed_matrix(level):
    matrix = gate_config.load(TOP / "gate" / "matrix.json")
    family = dial_choice.find_family(matrix)
    measured, (plain_at, dials_at) = dial_choice.family_at(matrix, family, level)
    assert plain_at == matrix.configuration(f"main-rounded-hour-k{level:g}-empirical")
    assert plain_at.kappa == level and plain_at.oracle_partner is not None
    assert matrix.configuration(plain_at.oracle_partner).kappa == level
    assert len(dials_at) == 8 and {d.kappa for d in dials_at} == {level}
    for dial, dial_at in zip(family[1], dials_at, strict=True):
        back = dataclasses.replace(
            dial_at, kappa=0.3, name=dial.name, oracle_partner=dial.oracle_partner
        )
        assert back == dial
        assert measured.configuration(dial_at.oracle_partner).kappa == level


def test_the_committed_matrix_takes_the_grid_fit_option():
    matrix = gate_config.load(TOP / "gate" / "matrix.json")
    assert sorted({c.kappa for c in matrix.configurations}) == list(driver_risk.LEVELS)
    dial_choice.check_fit_options(matrix, None, "grid", None, False)  # no MatrixError
    dial_choice.check_fit_options(matrix, 3, "grid", 0.4, True)
