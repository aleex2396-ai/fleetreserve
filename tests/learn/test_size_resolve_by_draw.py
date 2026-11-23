"""`by_draw` of a sizing's `difference_from_shuffled` (`size_resolve_summary.control_figures`),
by hand on one made-up unit with two splits (the sizing offers 4 and 6, constant 1 and 1).

1. Per draw, in the order of the draws, the gain is the sizing's mean offer over the splits (5)
   minus the draw's: draws (3, 3) and (2, 6) gain 2 and 1, smallest 1, largest 2, no draw at or
   above the sizing; a draw equal to the sizing (4, 6) gains 0 and one above it (6, 7) -1.5, both
   at or above it.
2. `over_draws` is the spread of those gains (numpy's interpolated 5th and 95th percentiles).
3. A unit that is not split has every `by_draw` figure None.
4. The results' definitions say what `by_draw` is and the chance it bounds.
"""

import numpy as np
import pytest
from test_size_resolve_summary import MEASURED, controls

from fleetreserve import size_resolve_summary


def by_draw(*draws) -> dict:
    found = size_resolve_summary._tiers(MEASURED, ["u"], controls(*draws))
    figures = {tier: found[tier]["net-0.1-scale"]["difference_from_shuffled"] for tier in found}
    assert figures["A"] == figures["B"]
    return figures["A"]


def test_the_gain_of_each_draw_by_hand():
    found = by_draw([3.0, 3.0], [2.0, 6.0])["by_draw"]
    assert found == {
        "gain_kwh": [pytest.approx(2.0), pytest.approx(1.0)],
        "smallest_gain_kwh": pytest.approx(1.0),
        "largest_gain_kwh": pytest.approx(2.0),
        "draws_at_or_above": 0,
    }


def test_a_draw_equal_to_or_above_the_sizing_is_counted():
    found = by_draw([3.0, 3.0], [4.0, 6.0], [6.0, 7.0])["by_draw"]
    assert found["gain_kwh"] == [pytest.approx(2.0), 0.0, pytest.approx(-1.5)]
    assert found["smallest_gain_kwh"] == pytest.approx(-1.5)
    assert found["draws_at_or_above"] == 2


def test_over_draws_is_the_spread_of_the_gains():
    diff = by_draw([3.0, 3.0], [2.0, 6.0], [4.0, 6.0], [6.0, 7.0])
    low, high = np.quantile(diff["by_draw"]["gain_kwh"], size_resolve_summary.POINTS)
    assert diff["over_draws"]["p5_kwh"] == pytest.approx(low)
    assert diff["over_draws"]["p95_kwh"] == pytest.approx(high)


def test_a_unit_that_is_not_split_has_no_gains():
    figures = size_resolve_summary.control_figures({}, [{}, {}])
    assert figures["by_draw"] == dict.fromkeys(size_resolve_summary.BY_DRAW)
    assert figures["draws"] == 2


def test_the_definitions_say_what_by_draw_is():
    definition = size_resolve_summary.definitions()["shuffled_control_by_draw"]
    assert "draws_at_or_above" in definition and "1/(n+1)" in definition
