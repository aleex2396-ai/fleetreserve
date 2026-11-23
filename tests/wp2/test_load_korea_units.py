"""The Korean loader and its five named portfolios, on made-up bytes.

A portfolio with no row in the file, or with a row that is not a fast-charger row and none kept,
is a `LayoutError` naming the portfolio and the counts of rows read and removed by each reason
(never a value of a row): a fault in reading the file, so the refusal is logged and the
parsing-fix route applies. A portfolio whose every row is a fast-charger row (the slow-charger
rule of plan section 3) abstains later, as before: the loader returns it with no session and
raises nothing. A file in which each portfolio keeps a row is read as before.
"""

import pytest
from fake_korea import korea_csv, korea_row

from fleetreserve import load_korea
from fleetreserve.sessions import LayoutError

FAST = "fast charger (ChargerType 1)"
ENERGY = "energy missing or not positive"
NO_OUT = "no plug-out time recorded"
TYPE_OF = {
    "korea-home": "Apartment",
    "korea-lodging": "Hotel",
    "korea-work": "Company",
    "korea-public-parking": "Public area",
    "korea-visits": "Market",
}
START, END = "2022-06-01 08:00:00", "2022-06-01 17:00:00"


def _row(unit, session=None, energy=9.0, end=END, charger_type="0"):
    return korea_row(
        session or f"{unit}-1", "5", "C1", TYPE_OF[unit], START, end, energy,
        charger_type=charger_type,
    )  # fmt: skip


def _file(*, leave_out=(), extra=()):
    rows = [_row(unit) for unit in load_korea.UNITS if unit not in leave_out]
    return korea_csv([*rows, *extra])


def test_the_table_of_this_file_covers_every_portfolio():
    assert set(TYPE_OF) == set(load_korea.UNITS)


def test_a_file_in_which_each_portfolio_keeps_a_row_is_read():
    sessions, dropped = load_korea.from_bytes(_file())
    assert sorted(sessions["site"]) == sorted(load_korea.UNITS) and len(dropped) == 0


@pytest.mark.parametrize("unit", load_korea.UNITS)
def test_a_portfolio_with_no_row_in_the_file_is_refused_with_its_name(unit):
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(_file(leave_out=(unit,)))
    assert str(refused.value) == f"Korea file: portfolio {unit} has no row kept: 0 rows read"


@pytest.mark.parametrize("unit", load_korea.UNITS)
def test_a_portfolio_whose_slow_charger_rows_are_all_removed_is_refused_with_the_counts(unit):
    extra = [
        _row(unit, "secret-1", end=None),
        _row(unit, "secret-2", energy=0.0),
        _row(unit, "secret-3", energy=0.0),
    ]
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(_file(leave_out=(unit,), extra=extra))
    assert str(refused.value) == (
        f"Korea file: portfolio {unit} has no row kept: 3 rows read, "
        f"removed by reason {NO_OUT}: 1, {ENERGY}: 2"
    )
    assert "secret" not in str(refused.value)


def test_fast_charger_rows_count_in_the_rows_read_when_a_slow_charger_row_is_removed_too():
    unit = "korea-work"
    extra = [_row(unit, "f1", charger_type="1"), _row(unit, "s1", energy=0.0)]
    with pytest.raises(LayoutError) as refused:
        load_korea.from_bytes(_file(leave_out=(unit,), extra=extra))
    assert str(refused.value) == (
        f"Korea file: portfolio {unit} has no row kept: 2 rows read, "
        f"removed by reason {FAST}: 1, {ENERGY}: 1"
    )


@pytest.mark.parametrize("unit", load_korea.UNITS)
def test_a_portfolio_of_fast_charger_rows_only_abstains_as_before_and_raises_nothing(unit):
    extra = [_row(unit, "f1", charger_type="1"), _row(unit, "f2", end=None, charger_type="1")]
    sessions, dropped = load_korea.from_bytes(_file(leave_out=(unit,), extra=extra))
    assert unit not in set(sessions["site"]) and len(sessions) == 4
    gone = dropped[dropped["site"] == unit]
    assert list(gone["reason"]) == [FAST, FAST]
