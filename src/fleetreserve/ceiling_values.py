"""The ceilings and best real-time values of one day on its true sets, one-way (plan 2, 4, 13 WP2).

One `DayTask` (a day with its unit, site and date) gives one `DayResult`.

Every day (the hourly product, hour 0 free):

- `hourly_hind`: `exact.hourly_ceiling`, b(S) - p(S), where a hindsight plan may choose the hour-0
  powers knowing the whole request path.
- `hourly_causal`: the certificate's best value (`certificate.certify`, hourly product, causal
  gains) on the true rows. Only the value is reported and the value does not depend on the rule
  that picks one shape among boxes of equal value, so the rule is `ANY_OPTIMAL`.

Recorded finding, on made-up days: every causal rule fixes the hour-0 powers before the first
request, while `exact.hourly_ceiling` lets a hindsight plan choose them per request path. The
hourly product's best causal value can therefore be below `exact.hourly_ceiling`; it equals a
hindsight value in which hour 0 is one plan shared by all paths. How large the difference is on
real days is what these values measure.

Sampled days also carry (`DayTask.sampled`):

- `hourly_tree`: the corner tree's hourly value, a cross-check that tree = causal;
- the block product (largest half-width, centre free; value 2 lam |S|): `block_hind`
  (`exact.block_ceiling`), `block_caus` (the corner tree), `block_aff` (`certify`, causal gains)
  and `block_diag` (`certify`, diagonal gains). The block value does not depend on the shape rule
  either, so `ANY_OPTIMAL` is used.

A solve that does not end optimal (a timeout included) gives `None` and its outcome is kept; it is
never read as zero or as infeasible.

Readings where the plan does not decide (marked):

- A day whose fleet is empty (every car left out by the rules of `days_from_truth`) has all its
  values 0 without a solve, and does not count as a day meeting T4's hypotheses.
- T4's hypotheses (plan 4, one-way, known cars): no car departs inside the window, and
  r_i times the recovery hours before d_i is at least (1 - rho) e_i for every car
  (`carsets.true_recovery_supply`).
- Time limits are those of `lp.py`: 30 s for the box check and the certificate, 300 s for the tree.
"""

from dataclasses import dataclass, field

from fleetreserve import exact, products
from fleetreserve.carsets import true_recovery_supply, true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day
from fleetreserve.tree import corner_tree

HYPOTHESIS_TOLERANCE = 1e-9  # kWh, on r times the recovery hours against the protected energy

HOURLY_KEYS = ("hourly_hind", "hourly_causal", "hourly_tree")
BLOCK_KEYS = ("block_hind", "block_caus", "block_aff", "block_diag")
VALUE_KEYS = (*HOURLY_KEYS, *BLOCK_KEYS)
EMPTY_FLEET = "empty fleet"
NOT_SOLVED = "not solved"
EXACT = "exact"


@dataclass(frozen=True)
class DayTask:
    unit: str
    site: str
    date: str
    day: Day
    sampled: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)


@dataclass(frozen=True)
class DayResult:
    """`values[key]` is a number in kWh or None (missing); `outcomes[key]` says why: "optimal",
    "exact" (no solve), "empty fleet", "not solved" (not asked for) or the solver's outcome."""

    unit: str
    site: str
    date: str
    cars: int
    n_service: int
    n_recovery: int
    in_window: bool
    t4_hypotheses: bool
    sampled: bool
    error: str | None
    values: dict[str, float | None]
    outcomes: dict[str, str]
    seconds: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)


def meets_t4_hypotheses(day: Day) -> bool:
    """One-way (always, here), no in-window departure, and r_i times the recovery hours before
    d_i at least (1 - rho) e_i for every car; false for an empty fleet."""
    if not day.cars or day.has_in_window_departure:
        return False
    return all(
        true_recovery_supply(car, day.n_service) >= car.need(day.rho) - HYPOTHESIS_TOLERANCE
        for car in day.cars
    )


def _read(solved) -> tuple[float | None, str, float]:
    """(value or None, outcome, seconds) of a certificate, a tree result or a block ceiling."""
    value = None if solved.value is None else float(solved.value) + 0.0  # -0.0 becomes 0.0
    return value, solved.outcome, float(solved.seconds)


def block_values(functions, rows) -> dict[str, tuple[float | None, str, float]]:
    """The four block values of a day: hindsight, tree, affine (causal) and diagonal."""
    block = products.block()
    return {
        "block_hind": _read(exact.block_ceiling(functions)),
        "block_caus": _read(corner_tree(rows, block)),
        "block_aff": _read(certify(rows, block, "causal", shape_rule=ANY_OPTIMAL)),
        "block_diag": _read(certify(rows, block, "diagonal", shape_rule=ANY_OPTIMAL)),
    }


def hourly_values(functions, rows, with_tree: bool) -> dict[str, tuple[float | None, str, float]]:
    """The hourly ceiling (hour 0 free) and the best causal value, and the tree if asked."""
    hourly = products.hourly()
    found = {
        "hourly_hind": (float(exact.hourly_ceiling(functions)) + 0.0, EXACT, 0.0),
        "hourly_causal": _read(certify(rows, hourly, "causal", shape_rule=ANY_OPTIMAL)),
    }
    if with_tree:
        found["hourly_tree"] = _read(corner_tree(rows, hourly))
    return found


def compute_day(task: DayTask) -> DayResult:
    """All values of one day. A day whose sets cannot be built gives a result with `error` set."""
    day = task.day
    values: dict[str, float | None] = dict.fromkeys(VALUE_KEYS)
    outcomes = dict.fromkeys(VALUE_KEYS, NOT_SOLVED)
    seconds: dict[str, float] = {}
    error = None
    if not day.cars:
        asked = ("hourly_hind", "hourly_causal", *(VALUE_KEYS[2:] if task.sampled else ()))
        for key in asked:
            values[key], outcomes[key] = 0.0, EMPTY_FLEET
    else:
        try:
            functions = exact.day_functions(day)
            rows = true_rows(day)
        except ValueError as problem:
            error = str(problem)
        else:
            found = hourly_values(functions, rows, task.sampled)
            if task.sampled:
                found |= block_values(functions, rows)
            for key, (value, outcome, took) in found.items():
                values[key], outcomes[key], seconds[key] = value, outcome, took
    return DayResult(
        unit=task.unit,
        site=task.site,
        date=task.date,
        cars=len(day.cars),
        n_service=day.n_service,
        n_recovery=day.n_recovery,
        in_window=day.has_in_window_departure,
        t4_hypotheses=meets_t4_hypotheses(day),
        sampled=task.sampled,
        error=error,
        values=values,
        outcomes=outcomes,
        seconds=seconds,
    )
