"""The gate run's matrix file (`gate_config`): the committed matrix, and what is refused.

1. The committed `gate/matrix.json` loads: 46 named configurations (the main family at the six
   kappa of plan section 2 with both departure variants and both readings of `departure_seen`:
   the family "main" sees departures from the rounded-down departure hour, the family "beside"
   at the start of the following hour; 24 configurations), three alternates at kappa 0.1 with
   their oracle partners (6), and the driver-risk family (16): the main candidate at kappa 0.3
   and rounded-hour with each dial of plan section 2, each with an oracle partner at the same
   dial. They are computed as 35 distinct pieces of work: an oracle with known energy bounds does
   not depend on kappa, so the two levels 0.4 and 0.5 add 4 pieces to the 31 of four levels.
   Every alternate without driver-risk rows changes one field of the candidate that sees
   departures at the start of each hour; every reading of the report-only decisions is listed.
2. An unknown field, a missing field, a repeated name, a value outside its set, and an oracle
   partner that is missing, not an oracle, or differs in another field are each refused.
3. Work keys: kappa changes nothing for an oracle with known energy bounds, the departure floor
   nothing for any oracle; everything else keeps configurations apart.
"""

import copy
import json

import pytest
from gate_fake import COMMITTED, TOP

from fleetreserve import gate_config
from fleetreserve.gate_stat import RATIO_RULES

MAIN = {
    "energy_bounds": "known",
    "uncertifiable_car": "leave-unsteered",
    "shape_rule": "largest-smallest-width",
    "departure_floor": "as-planned",
    "driver_loss_cars": "steered-only",
    "departure_seen": "hour-start",
    "driver_risk": None,
}


def test_the_committed_matrix_loads():
    matrix = gate_config.load(TOP / "gate" / "matrix.json")
    assert len(matrix.configurations) == 46
    assert len({c.work_key() for c in matrix.configurations}) == 35
    assert matrix.gate_units == (
        "nrel-before-may-2019",
        "nrel-from-may-2019",
        "norway-development-pooled",
    )
    assert (matrix.required_units, matrix.n_calibration, matrix.rho) == (2, 100, 0.0)
    assert matrix.two_way_budget_kwh == 5.0
    assert matrix.ceilings == gate_config.CEILINGS
    assert matrix.ratio_rules == RATIO_RULES
    main = [c for c in matrix.configurations if c.family in ("main", "beside")]
    assert {c.departure_seen for c in main if c.family == "main"} == {"rounded-hour"}
    assert {c.departure_seen for c in main if c.family == "beside"} == {"hour-start"}
    assert sorted({c.kappa for c in main}) == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    found = [(c.kappa, c.departure, c.departure_seen) for c in main]
    assert sorted(found) == sorted(
        (k, d, s)
        for k in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
        for d in ("empirical", "oracle")
        for s in gate_config.DEPARTURE_SEEN
    )
    for c in main:
        assert {f: getattr(c, f) for f in MAIN if f != "departure_seen"} == {
            f: v for f, v in MAIN.items() if f != "departure_seen"
        }
    for c in matrix.configurations:
        if c.family == "alternate" and c.driver_risk is None:
            changed = [f for f in MAIN if getattr(c, f) != MAIN[f]]
            assert len(changed) == 1 and c.kappa == 0.1


def _document():
    return copy.deepcopy(COMMITTED)


def _refused(document, text):
    with pytest.raises(gate_config.MatrixError, match=text):
        gate_config.parse(document)


def test_an_unknown_top_field_is_refused():
    document = _document()
    document["extra"] = 1
    _refused(document, "unknown fields")


def test_a_missing_top_field_is_refused():
    document = _document()
    del document["split_seed"]
    _refused(document, "missing fields")


def test_an_unknown_configuration_field_is_refused():
    document = _document()
    document["configurations"][0]["rho"] = 0.1
    _refused(document, "unknown fields")


def test_a_missing_configuration_field_is_refused():
    document = _document()
    del document["configurations"][0]["departure_floor"]
    _refused(document, "missing fields")


def test_a_repeated_name_is_refused():
    document = _document()
    document["configurations"].append(copy.deepcopy(document["configurations"][1]))
    _refused(document, "appear twice")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kappa", 0.15),
        ("departure", "declared"),
        ("energy_bounds", "user-rate"),
        ("uncertifiable_car", "drop"),
        ("shape_rule", "widest"),
        ("departure_floor", "rounded"),
        ("driver_loss_cars", "all-cars"),
        ("departure_seen", "half-hour"),
        ("family", "other"),
    ],
)
def test_a_value_outside_its_set_is_refused(field, value):
    document = _document()
    for entry in document["configurations"][:2]:  # a pair, so the partner check cannot refuse it
        entry[field] = value
    _refused(document, rf"{field} must be one of")


def test_report_readings_must_be_known_and_not_empty():
    document = _document()
    document["report"]["ceilings"] = []
    _refused(document, "report.ceilings")
    document = _document()
    document["report"]["ratio_rules"] = ["most_units"]
    _refused(document, "report.ratio_rules")


def test_oracle_partners_are_checked():
    document = _document()
    document["configurations"][0]["oracle_partner"] = "no-such"
    _refused(document, "no configuration")
    document = _document()
    document["configurations"][0]["oracle_partner"] = document["configurations"][2]["name"]
    _refused(document, "not oracle")
    document = _document()
    document["configurations"][1]["uncertifiable_car"] = "day-abstains"
    _refused(document, "differs in")
    document = _document()
    document["configurations"][1]["departure_seen"] = "rounded-hour"
    _refused(document, r"differs in \['departure_seen'\]")
    document = _document()
    document["configurations"][1]["oracle_partner"] = document["configurations"][0]["name"]
    _refused(document, "an oracle has no oracle_partner")


def test_a_bad_file_is_refused(tmp_path):
    with pytest.raises(gate_config.MatrixError, match="cannot be read"):
        gate_config.load(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(gate_config.MatrixError, match="cannot be read"):
        gate_config.load(tmp_path / "bad.json")
    (tmp_path / "ok.json").write_text(json.dumps(_document()))
    assert gate_config.load(tmp_path / "ok.json").split_seed == COMMITTED["split_seed"]


def _configuration(**change):
    fields = {"name": "c", "family": "main", "kappa": 0.1, "departure": "oracle", **MAIN}
    fields["oracle_partner"] = None
    fields.update(change)
    return gate_config.Configuration(**fields)


def test_work_keys():
    oracle = _configuration()
    assert oracle.work_key() == _configuration(kappa=0.3).work_key()
    assert oracle.work_key() == _configuration(departure_floor="floored").work_key()
    assert oracle.work_key() != _configuration(uncertifiable_car="day-abstains").work_key()
    assert oracle.work_key() != _configuration(departure_seen="rounded-hour").work_key()
    empirical_energy = _configuration(energy_bounds="empirical")
    assert (
        empirical_energy.work_key()
        != _configuration(energy_bounds="empirical", kappa=0.3).work_key()
    )
    forecast = _configuration(departure="empirical")
    assert forecast.work_key() != _configuration(departure="empirical", kappa=0.3).work_key()
    assert (
        forecast.work_key()
        != _configuration(departure="empirical", departure_floor="floored").work_key()
    )
