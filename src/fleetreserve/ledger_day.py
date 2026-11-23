"""The values of one sampled day on its true sets that the ledger of plan T9 needs beyond a
scoring run's own files (plan sections 1, 2, 4, 5 (T9) and 11 (O6)).

`solve_day(task)`, for a `ceiling_values.DayTask` (one-way true sets, every car of the day):

- from `ceiling_values.compute_day` with the day sampled: `hourly_hind` (`exact.hourly_ceiling`,
  hour 0 free), `hourly_causal` (V_aff, `certificate.certify` with causal gains), `hourly_tree`
  (V_caus, the corner tree), and the block product's `block_hind`, `block_caus`, `block_aff` and
  `block_diag`;
- `hourly_diag`: V_diag of the hourly product (`certify` with diagonal gains, the box limit),
  which neither a scoring run nor `ceiling_values` gives;
- `lp24_alpha` and `lp24_box`: LP (24) (`lp24.inner_set`, its 600-second limit) on the true rows
  with the day-average base set (`base_sets.day_average`), and the largest hourly box in its set
  `p + alpha U0` with hour 0 free (`inner_box.largest_box`, its closed form for LP (24)). The
  format loss of the day is `hourly_hind - lp24_box`.

Every value is in kWh (`lp24_alpha` is a number) or None, and `outcomes[key]` says why: "optimal",
"exact" (no solve), "empty fleet" (a day with no car: every value 0, nothing solved), "trivial
base set" (the day-average base set holds only the zero profile, so LP (24) is not solved and the
largest box is 0, reading below), or the solver's outcome; a timeout is None, never 0 and never
infeasible. A day whose sets cannot be built has `error` set and no value.

Reading (the plan does not decide it; its alternative in brackets): a day-average base set that
holds only the zero profile (`BaseSet.is_trivial`: energy at most 1e-7 kWh or no hour with power)
gives `p + alpha U0 = {p}` for every alpha, whose largest box has value 0 (`lp24` refuses such a
base set, since every alpha is feasible), with alpha None. [Alternative: the day's format loss is
missing.]
"""

import time
from dataclasses import dataclass, field, replace

from fleetreserve import ceiling_values, products
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.inner_box import largest_box
from fleetreserve.lp import BOX_SECONDS
from fleetreserve.lp24 import inner_set

HOURLY_DIAG = "hourly_diag"
LP24_ALPHA = "lp24_alpha"
LP24_BOX = "lp24_box"
EXTRA_KEYS = (HOURLY_DIAG, LP24_ALPHA, LP24_BOX)
KEYS = (*ceiling_values.VALUE_KEYS, *EXTRA_KEYS)
TRIVIAL_BASE = "trivial base set"


@dataclass(frozen=True)
class SampleDay:
    """`values[key]` in kWh (or a number) or None; `outcomes[key]` says why (module text)."""

    unit: str
    site: str
    date: str
    cars: int
    n_service: int
    error: str | None
    values: dict
    outcomes: dict
    seconds: dict = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.site, self.date)


def _extra(day) -> tuple[dict, dict, dict]:
    values = dict.fromkeys(EXTRA_KEYS)
    outcomes = dict.fromkeys(EXTRA_KEYS, ceiling_values.NOT_SOLVED)
    seconds = {}
    if not day.cars:
        for key in EXTRA_KEYS:
            values[key], outcomes[key] = 0.0, ceiling_values.EMPTY_FLEET
        values[LP24_ALPHA] = None
        return values, outcomes, seconds
    rows = true_rows(day)
    diagonal = certify(
        rows, products.hourly(), "diagonal", shape_rule=ANY_OPTIMAL, time_limit=BOX_SECONDS
    )
    outcomes[HOURLY_DIAG], seconds[HOURLY_DIAG] = diagonal.outcome, float(diagonal.seconds)
    if diagonal.value is not None:
        values[HOURLY_DIAG] = float(diagonal.value) + 0.0  # -0.0 becomes 0.0
    base = day_average(rows)
    if base.is_trivial():
        values[LP24_BOX] = 0.0
        outcomes[LP24_ALPHA] = outcomes[LP24_BOX] = TRIVIAL_BASE
        return values, outcomes, seconds
    started = time.perf_counter()
    inner = inner_set(rows, base, "24")
    box = largest_box(inner, products.hourly())
    seconds[LP24_BOX] = time.perf_counter() - started
    outcomes[LP24_ALPHA] = inner.outcome
    outcomes[LP24_BOX] = box.outcome
    if inner.outcome == "optimal":
        values[LP24_ALPHA] = float(inner.value) + 0.0
    if box.outcome == "optimal":
        values[LP24_BOX] = float(box.value) + 0.0
    return values, outcomes, seconds


def solve_day(task: ceiling_values.DayTask) -> SampleDay:
    """Every value of one sampled day (module text)."""
    found = ceiling_values.compute_day(replace(task, sampled=True))
    values, outcomes, seconds = dict(found.values), dict(found.outcomes), dict(found.seconds)
    error = found.error
    if error is None:
        try:
            extra_values, extra_outcomes, extra_seconds = _extra(task.day)
        except ValueError as problem:
            error = str(problem)
            extra_values = dict.fromkeys(EXTRA_KEYS)
            extra_outcomes = dict.fromkeys(EXTRA_KEYS, ceiling_values.NOT_SOLVED)
            extra_seconds = {}
        values |= extra_values
        outcomes |= extra_outcomes
        seconds |= extra_seconds
    else:
        values |= dict.fromkeys(EXTRA_KEYS)
        outcomes |= dict.fromkeys(EXTRA_KEYS, ceiling_values.NOT_SOLVED)
    return SampleDay(
        task.unit,
        task.site,
        task.date,
        len(task.day.cars),
        task.day.n_service,
        error,
        {key: values[key] for key in KEYS},
        {key: outcomes[key] for key in KEYS},
        seconds,
    )
