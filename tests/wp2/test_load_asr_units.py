"""The a.s.r. loader and its two named periods, on made-up bytes.

A period (by the local date of plug-in: before and from 1 March 2023) with no row in the file, or
whose rows are all removed by the loader's removal rules, is a `LayoutError` naming the period and
the counts of rows read and removed by each reason (never a value of a row): a fault in reading
the file, so the refusal is logged and the parsing-fix route applies. A file in which each period
keeps a row is read as before.
"""

import pytest
from fake_heldout import asr_csv, asr_row

from fleetreserve import load_asr
from fleetreserve.sessions import LayoutError

BEFORE, FROM = load_asr.UNITS
ENERGY = "energy missing or not positive"
NO_OUT = "no plug-out time recorded"


def _before(ev="EV1", energy=20.0, end="2023-02-27T17:30:00", day="2023-02-27"):
    return asr_row(ev, f"{day}T07:30:00", end, energy)


def _from(ev="EV2", energy=20.0, end="2023-03-02T17:30:00", day="2023-03-02"):
    return asr_row(ev, f"{day}T07:30:00", end, energy)


def test_a_file_in_which_each_period_keeps_a_row_is_read():
    sessions, dropped = load_asr.from_bytes(asr_csv([_before(), _from(), _from("EV3", 0.0)]))
    assert sorted(load_asr.periods(sessions)["unit"]) == [BEFORE, FROM] and len(dropped) == 1


@pytest.mark.parametrize("missing, rows", [(FROM, [_before()]), (BEFORE, [_from()])])
def test_a_period_with_no_row_in_the_file_is_refused_with_its_name(missing, rows):
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    assert str(refused.value) == f"a.s.r. file: period {missing} has no row kept: 0 rows read"


def test_a_file_with_no_row_at_all_is_refused_for_its_first_period():
    with pytest.raises(LayoutError, match=f"period {BEFORE} has no row kept: 0 rows read"):
        load_asr.from_bytes(asr_csv([]))


def test_a_period_whose_rows_are_all_removed_is_refused_with_the_counts_by_reason():
    rows = [
        _before(),
        _from("secret-ev", end=None, day="2023-03-02"),
        _from("secret-ev", 0.0, day="2023-03-03", end="2023-03-03T17:30:00"),
        _from("secret-ev", -1.0, day="2023-03-04", end="2023-03-04T17:30:00"),
    ]
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    assert str(refused.value) == (
        f"a.s.r. file: period {FROM} has no row kept: 3 rows read, "
        f"removed by reason {NO_OUT}: 1, {ENERGY}: 2"
    )
    assert "secret" not in str(refused.value)


def test_a_row_is_counted_in_the_period_of_its_plug_in_date():
    rows = [
        _before(),
        _from("EV2", 0.0, day="2023-03-01", end="2023-03-01T17:30:00"),
        _before("EV3", 0.0, day="2023-02-28", end="2023-02-28T17:30:00"),
    ]
    with pytest.raises(LayoutError) as refused:
        load_asr.from_bytes(asr_csv(rows))
    assert str(refused.value).endswith(
        f"has no row kept: 1 rows read, removed by reason {ENERGY}: 1"
    )
    assert f"period {FROM}" in str(refused.value)
