"""The largest box in LP (24)'s and LP (27)'s certified sets (`inner_box`) against the ceiling.

On `N_DAYS` seeded made-up days (`lp24_cases`, 2 to 4 service hours), each one-way and two-way
(`N_SOLVED` cases with a non-trivial day-average base set), for the hourly and the block product:

1. LP (24)'s closed form equals the programme over corners (`vertex_box`) on `p + alpha U0`, with
   hour 0 free per corner and with hour 0 shared, within `TOLERANCE`.
2. Every box lies in the fleet's set (`exact.box_inside` on its `lo`, `hi`), and its value never
   exceeds the ceiling of `exact` (hourly: b(S) - p(S); block: `exact.block_ceiling`): for LP
   (24), for LP (27) with hour 0 free and with hour 0 shared. With hour 0 shared, LP (27)'s box
   is never above its box with hour 0 free, and it is below it on at least `MIN_SHARED_BELOW`
   cases (so the hour-0 reading is not idle).
3. The general split (`certify(..., structure="full")`) is compared with the boxes of LP (24) and
   LP (27): the plan makes no claim, so the counts (larger, equal, smaller) are printed, and only
   their total is asserted.
"""

import functools

import numpy as np
from lp24_cases import cases

from fleetreserve import exact, products
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.inner_box import largest_box, vertex_box
from fleetreserve.lp24 import inner_set

SEED = 31
N_DAYS = 30
N_SOLVED = 56
PRODUCTS = (products.hourly(), products.block())
N_COMPARED = N_SOLVED * len(PRODUCTS)
TOLERANCE = 1e-6
MIN_SHARED_BELOW = 20


def _ceiling(functions, product):
    if product.kind == "hourly":
        return exact.hourly_ceiling(functions)
    result = exact.block_ceiling(functions)
    assert result.outcome == "optimal"
    return result.value


@functools.cache
def _compared():
    found = [c for c in cases(SEED, N_DAYS) if c[3] is not None]
    assert len(found) == N_SOLVED
    compared = []
    for day, budget, rows, base in found:
        functions = exact.day_functions(day, budget)
        sets = {p: inner_set(rows, base, p) for p in ("24", "27")}
        assert all(s.outcome == "optimal" for s in sets.values())
        for product in PRODUCTS:
            boxes = {
                "24": largest_box(sets["24"], product),
                "24 corners": vertex_box(sets["24"].p, sets["24"].P, base, product),
                "24 corners shared": vertex_box(
                    sets["24"].p, sets["24"].P, base, product, shared_hour0=True
                ),
                "27": largest_box(sets["27"], product),
                "27 shared": largest_box(sets["27"], product, shared_hour0=True),
            }
            assert all(b.outcome == "optimal" for b in boxes.values()), boxes
            split = certify(rows, product, "full", shape_rule=ANY_OPTIMAL)
            assert split.outcome == "optimal"
            compared.append((functions, product, boxes, split.value, _ceiling(functions, product)))
    return compared


def _close(a, b):
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


def _le(a, b):
    return a <= b + TOLERANCE * max(1.0, abs(b))


def test_closed_form_equals_the_corner_programme():
    compared = _compared()
    assert len(compared) == N_COMPARED
    differ = [
        (boxes["24"].value, boxes[k].value)
        for _, _, boxes, _, _ in compared
        for k in ("24 corners", "24 corners shared")
        if not _close(boxes["24"].value, boxes[k].value)
    ]
    assert not differ, differ


def test_boxes_lie_in_the_fleet_set_and_below_the_ceiling():
    outside, above = [], []
    for functions, product, boxes, _, ceiling in _compared():
        for name in ("24", "27", "27 shared"):
            box = boxes[name]
            if not exact.box_inside(functions, box.m - box.w, box.m + box.w):
                outside.append((product.kind, name, box))
            if not _le(box.value, ceiling):
                above.append((product.kind, name, box.value, ceiling))
        assert _le(boxes["27 shared"].value, boxes["27"].value)
    assert not outside, outside
    assert not above, above
    shared_below = sum(
        boxes["27 shared"].value < boxes["27"].value - TOLERANCE * max(1.0, boxes["27"].value)
        for *_, boxes, _, _ in _compared()
    )
    print(f"LP (27) box with hour 0 shared below hour 0 free on {shared_below}")
    assert shared_below >= MIN_SHARED_BELOW


def test_general_split_against_the_inner_boxes():
    counts = {}
    for _, product, boxes, split, _ in _compared():
        for name in ("24", "27", "27 shared"):
            value = boxes[name].value
            key = "equal" if _close(split, value) else ("larger" if split > value else "smaller")
            counts.setdefault((product.kind, name), {"larger": 0, "equal": 0, "smaller": 0})
            counts[(product.kind, name)][key] += 1
    for (kind, name), c in sorted(counts.items()):
        print(f"{kind}: general split against LP ({name}) box: {c}")
    assert all(sum(c.values()) == N_SOLVED for c in counts.values())
    assert len(counts) == len(PRODUCTS) * 3
    assert np.isfinite([split for *_, split, _ in _compared()]).all()
