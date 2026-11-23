"""The held-out Norwegian loader and its two named sites, on made-up bytes.

A named site with no row in the file, or whose rows are all removed by the loader's removal rules,
is a `LayoutError` naming the site and the counts of rows read and removed by each reason (never
a value of a row): a fault in reading the file, so the refusal is logged and the parsing-fix route
applies. A file in which each site keeps a row is read as before.
"""

import pytest
from fake_raw import norway_row, raw_root, write_norway

from fleetreserve import load_heldout_norway
from fleetreserve.sessions import LayoutError

ENERGY = "energy missing or not positive"
NO_OUT = "no plug-out time recorded"


def _bytes(tmp_path, rows):
    return write_norway(raw_root(tmp_path / "made"), "held-out", rows).read_bytes()


def _good(site, tag="g"):
    return norway_row(
        site, "u" + tag, f"{site}-{tag}", "2020-03-02 17:00:00", "2020-03-02 23:00:00", 10.0
    )


def test_a_file_in_which_each_site_keeps_a_row_is_read(tmp_path):
    rows = [
        _good("OSL_T"),
        _good("TRO_R"),
        norway_row("TRO_R", "x", "z", "2020-03-03 17:00:00", None, 3.0),
    ]
    sessions, dropped = load_heldout_norway.from_bytes(_bytes(tmp_path, rows))
    assert sorted(sessions["site"]) == ["OSL_T", "TRO_R"] and len(dropped) == 1


@pytest.mark.parametrize("missing, other", [("TRO_R", "OSL_T"), ("OSL_T", "TRO_R")])
def test_a_site_with_no_row_in_the_file_is_refused_with_its_name(tmp_path, missing, other):
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_bytes(tmp_path, [_good(other)]))
    assert str(refused.value) == (
        f"Norway held-out file: site {missing} has no row kept: 0 rows read"
    )


def test_a_file_with_no_row_at_all_is_refused_for_its_first_site(tmp_path):
    with pytest.raises(LayoutError, match="site OSL_T has no row kept: 0 rows read"):
        load_heldout_norway.from_bytes(_bytes(tmp_path, []))


def test_a_site_whose_rows_are_all_removed_is_refused_with_the_counts_by_reason(tmp_path):
    rows = [
        _good("OSL_T"),
        norway_row("TRO_R", "secret-user", "secret-id-1", "2020-03-02 17:00:00", None, 5.0),
        norway_row(
            "TRO_R", "secret-user", "secret-id-2", "2020-03-03 17:00:00", "2020-03-03 20:00:00", 0.0
        ),
        norway_row(
            "TRO_R", "secret-user", "secret-id-3", "2020-03-04 17:00:00", "2020-03-04 20:00:00", 0.0
        ),
    ]
    with pytest.raises(LayoutError) as refused:
        load_heldout_norway.from_bytes(_bytes(tmp_path, rows))
    assert str(refused.value) == (
        "Norway held-out file: site TRO_R has no row kept: 3 rows read, "
        f"removed by reason {NO_OUT}: 1, {ENERGY}: 2"
    )
    assert "secret" not in str(refused.value)


def test_a_site_with_one_row_kept_among_removed_rows_is_read(tmp_path):
    rows = [
        _good("OSL_T"),
        _good("TRO_R"),
        norway_row("TRO_R", "u", "r1", "2020-03-03 17:00:00", "2020-03-03 20:00:00", 0.0),
    ]
    sessions, dropped = load_heldout_norway.from_bytes(_bytes(tmp_path, rows))
    assert len(sessions) == 2 and list(dropped["reason"]) == [ENERGY]
