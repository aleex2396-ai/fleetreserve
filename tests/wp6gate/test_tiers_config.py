"""The tiers of plan section 6 in the run matrix (`gate_config`).

The list of tiers to report is a required shared field, `report.tiers`: a non-empty list of
distinct values from "A" (driver-protected) and "B" (deadline contract). A missing field, an empty
list, an unknown tier, a repeated tier and a value that is not a list are each refused. Both
committed matrices (`gate/matrix.json`, `learn/matrix.json`) list both tiers.
"""

import copy

import pytest
from gate_fake import COMMITTED, TOP

from fleetreserve import gate_config


def _document():
    return copy.deepcopy(COMMITTED)


def _refused(document, text):
    with pytest.raises(gate_config.MatrixError, match=text):
        gate_config.parse(document)


def test_the_committed_matrices_list_both_tiers():
    for name in ("gate", "learn"):
        matrix = gate_config.load(TOP / name / "matrix.json")
        assert matrix.tiers == ("A", "B")


def test_a_missing_tier_list_is_refused():
    document = _document()
    del document["report"]["tiers"]
    _refused(document, r"report: missing fields \['tiers'\]")


def test_an_empty_tier_list_is_refused():
    document = _document()
    document["report"]["tiers"] = []
    _refused(document, r"report\.tiers must be a non-empty list")


@pytest.mark.parametrize("value", ["A", None, {"A": 1}, 0])
def test_a_tier_list_that_is_not_a_list_is_refused(value):
    document = _document()
    document["report"]["tiers"] = value
    _refused(document, r"report\.tiers must be a non-empty list")


def test_an_unknown_or_repeated_tier_is_refused():
    document = _document()
    document["report"]["tiers"] = ["A", "C"]
    _refused(document, r"report\.tiers must be one of")
    document = _document()
    document["report"]["tiers"] = ["B", "B"]
    _refused(document, r"report\.tiers lists a value twice")


def test_one_tier_alone_is_allowed():
    for tiers in (["A"], ["B"], ["B", "A"]):
        document = _document()
        document["report"]["tiers"] = tiers
        assert gate_config.parse(document).tiers == tuple(tiers)


def test_the_labels_name_both_tiers():
    assert gate_config.TIER_LABELS == {
        "A": "tier A (driver-protected)",
        "B": "tier B (deadline contract)",
    }
