"""The workplace window is a parameter of the truth build: a given window reaches the ACN and
NREL tables and the window report, the Norway sites keep their own, and the default is the
constant `windows.WORKPLACE`."""

import pytest
from fake_raw import TEST_WINDOW, full_tree

from fleetreserve import load_acn, truth_build
from fleetreserve.windows import WORKPLACE, Window

OTHER = TEST_WINDOW  # τ 10:00, S 11:00 to 17:00: a window other than the workplace default
WORKPLACE_UNITS = ("caltech", "jpl", "office001", "nrel-before-may-2019", "nrel-from-may-2019")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return full_tree(tmp_path_factory.mktemp("window"))


def _bytes(result) -> dict[str, bytes]:
    return {
        f"{unit}-{kind}": truth_build.csv_bytes(frame)
        for unit, tables in result.tables.items()
        for kind, frame in zip(("cars", "days"), tables, strict=True)
    }


def test_another_window_changes_tau_and_service_hours_of_the_workplace_units(root):
    other = truth_build.build_all(root, OTHER)
    default = truth_build.build_all(root)
    for unit in WORKPLACE_UNITS:
        cars, days = other.tables[unit]
        assert set(days["service_hours"]) == {6}, unit
        assert set(days["tau"].dt.hour) == {10}, unit
        assert set(cars["tau"].dt.hour) == {10}, unit
        assert (other.windows[unit]["tau"], other.windows[unit]["service_hours"]) == ("10:00", 6)
        assert other.windows[unit]["service"] == "11:00 to 17:00"
        cars, days = default.tables[unit]
        assert set(days["service_hours"]) == {3} and set(days["tau"].dt.hour) == {8}, unit
        assert (default.windows[unit]["tau"], default.windows[unit]["service"]) == (
            "08:00",
            "09:00 to 12:00",
        )


def test_norway_sites_keep_their_own_window(root):
    other, default = truth_build.build_all(root, OTHER), truth_build.build_all(root)
    other_bytes, default_bytes = _bytes(other), _bytes(default)
    for site in ("OSL_S", "ASK", "OSL_1", "BAR_2"):
        assert other.windows[site] == default.windows[site]
        assert other.windows[site]["tau"] == "17:00"
        assert other_bytes[f"{site}-days"] == default_bytes[f"{site}-days"]
        assert other_bytes[f"{site}-cars"] == default_bytes[f"{site}-cars"]


def test_the_default_window_is_the_workplace_constant(root):
    default = truth_build.build_all(root)
    explicit = truth_build.build_all(root, WORKPLACE)
    assert _bytes(default) == _bytes(explicit)
    assert default.windows == explicit.windows
    assert (WORKPLACE.tau_hour, WORKPLACE.service_hours) == (8, 3)
    assert (
        _bytes(truth_build.build_all(root, OTHER))["caltech-days"]
        != _bytes(default)["caltech-days"]
    )


def test_acn_site_tables_take_the_window(root):
    sessions, _ = load_acn.load("caltech", root)
    cars, days = truth_build.acn_site_tables(sessions, "caltech", OTHER)
    assert set(days["service_hours"]) == {6} and set(cars["tau"].dt.hour) == {10}
    cars, days = truth_build.acn_site_tables(sessions, "caltech", WORKPLACE)
    assert set(days["service_hours"]) == {3} and set(cars["tau"].dt.hour) == {8}


def test_nrel_alternative_readings_use_the_window(root):
    # The made-up NREL cars plug in at 08:00 and leave at 16:30. At τ 10:00 (the test window)
    # each regime has one counting day under both alternative plug-out readings; at τ 07:00
    # nobody is plugged in yet, so neither regime has one.
    def counts(result):
        return {
            name: tuple(numbers.values()) for name, numbers in result.alternatives["nrel"].items()
        }

    default = counts(truth_build.build_all(root, TEST_WINDOW))
    early = counts(truth_build.build_all(root, Window(7, 3, "made-up window for the test")))
    assert set(default) == set(early) and len(default) == 2
    assert all(value == (1, 1) for value in default.values()), default
    assert all(value == (0, 0) for value in early.values()), early
