"""The WP5 order values of one development day on its true one-way sets (plan sections 2, 4 and 13
WP5: "diag <= affine <= tree <= hindsight every development day").

Per product, four values, each a number in kWh or None (missing):

- hourly: `hourly_diag` (`certificate.certify`, diagonal gains), `hourly_causal` (causal affine
  gains, V_aff), `hourly_tree` (the corner tree, V_caus) and `hourly_hind`
  (`exact.hourly_ceiling`, hour 0 free);
- block: `block_diag`, `block_aff`, `block_caus` (the corner tree) and `block_hind`
  (`exact.block_ceiling`).

All but `hourly_diag` come from `ceiling_values.compute_day` (the programmes are not written
again); `hourly_diag` is one more call of `certify` with diagonal gains. Every value does not
depend on the rule that picks one shape among boxes of equal value, so the rule is `ANY_OPTIMAL`.

Recorded finding (`ceiling_values`): on a day with a decision hour the hourly product's best
causal value can be below `exact.hourly_ceiling`, because a hindsight plan may choose the hour-0
powers per request path. So the order is what is checked, never equality with the ceiling.

A day with an empty fleet has every value 0 without a solve ("empty fleet"). A solve that does not
end optimal gives None with its outcome kept ("timeout" among them): missing, never zero.
"""

from dataclasses import dataclass, field

from fleetreserve import products
from fleetreserve.carsets import true_rows
from fleetreserve.ceiling_values import EMPTY_FLEET, NOT_SOLVED, DayResult, DayTask, compute_day
from fleetreserve.certificate import ANY_OPTIMAL, certify

HOURLY_ORDER = ("hourly_diag", "hourly_causal", "hourly_tree", "hourly_hind")
BLOCK_ORDER = ("block_diag", "block_aff", "block_caus", "block_hind")
ORDER_KEYS = (*HOURLY_ORDER, *BLOCK_ORDER)


@dataclass(frozen=True)
class OrderValues:
    """The eight values of a day, their outcomes and solve seconds, and the day's flags."""

    unit: str
    site: str
    date: str
    cars: int
    n_service: int
    in_window: bool
    t4_hypotheses: bool
    error: str | None
    values: dict[str, float | None]
    outcomes: dict[str, str]
    seconds: dict[str, float] = field(default_factory=dict, compare=False)


def hourly_diagonal(task: DayTask) -> tuple[float | None, str, float]:
    """The hourly product's best value with diagonal gains on the day's true one-way rows."""
    found = certify(true_rows(task.day), products.hourly(), "diagonal", shape_rule=ANY_OPTIMAL)
    value = None if found.value is None else float(found.value) + 0.0
    return value, found.outcome, float(found.seconds)


def order_values(task: DayTask) -> OrderValues:
    """The day's eight values; `task.sampled` is ignored (every day gets every value)."""
    base: DayResult = compute_day(DayTask(task.unit, task.site, task.date, task.day, sampled=True))
    values = {key: base.values.get(key) for key in ORDER_KEYS}
    outcomes = {key: base.outcomes.get(key, NOT_SOLVED) for key in ORDER_KEYS}
    seconds = dict(base.seconds)
    if base.error is None:
        if task.day.cars:
            values["hourly_diag"], outcomes["hourly_diag"], seconds["hourly_diag"] = (
                hourly_diagonal(task)
            )
        else:
            values["hourly_diag"], outcomes["hourly_diag"] = 0.0, EMPTY_FLEET
    return OrderValues(
        unit=base.unit,
        site=base.site,
        date=base.date,
        cars=base.cars,
        n_service=base.n_service,
        in_window=base.in_window,
        t4_hypotheses=base.t4_hypotheses,
        error=base.error,
        values=values,
        outcomes=outcomes,
        seconds=seconds,
    )
