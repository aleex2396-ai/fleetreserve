"""The values of outcome O3 on one day's true sets (plan sections 1, 2, 5 (T2), 10 and 11 (O3)).

Plan O3: "Headline measurement, true sessions, 100-200 sampled days: LP (24), LP (27), general
split at X, Y, Z% of the ceiling." One `DayTask` (a day of a unit, its base-set rule and the time
limits) gives one `DayResult`: per product (`PRODUCTS`: the hourly and the block product, both
built for every value below) the ceiling and each method's value (`METHODS`), in kWh, each with
its outcome.

- `ceiling`: V_hind on the true one-way sets, hour 0 free and chosen per request path (the
  owner's decision of 16 November 2026: this is the ceiling). Hourly: `exact.hourly_ceiling`,
  b(S) - p(S), no solve (outcome "exact"); block: `exact.block_ceiling`, one programme with the
  box limit.
- `lp24`: the largest box of the product in LP (24)'s certified set `p + alpha U0`
  (`lp24.inner_set(..., "24")`, then `inner_box.largest_box`: its closed form, no solve). The
  plan's format loss (section 1) is the ceiling minus this value.
- `lp27_smallest_found` and `lp27_largest_found`: the smallest and the largest box of the product
  over the optimal solutions of LP (27) that the stated search of `o3_lp27` finds (reading D3).
  LP (27)'s trace has many optimal solutions whose sets hold very different boxes, and the
  largest (or smallest) box over all of them is not a linear programme. The search is a local
  search and does not find the largest box over all of them. Each solution searched is optimal to
  within the face's tolerance (`o3_lp27` module text), so, to the solver's precision, the
  smallest over all optimal solutions is at most `lp27_smallest_found` and the largest over all
  is at least `lp27_largest_found`; how far either lies from the value over all optimal solutions
  is not known.
- `lp27_largest_bound`: a number no optimal solution of LP (27) holds a larger box than
  (`o3_lp27_bound`: a linear relaxation of the joint programme and branch and bound, at most
  `DayTask.bound_nodes` nodes per product and `DayTask.bound_rounds` relaxations per node; valid
  to the solver's tolerances; at most the ceiling). The largest box over all optimal solutions
  lies between `lp27_largest_found` and it; its outcome says whether the two met (`closed`), the
  relaxations' bound is at or above the ceiling (`ceiling`) or the node budget ran out first
  (`open`). When LP (27) or a root programme of the bound is not solved, the bound is missing
  with that outcome.
- `general_split`: T2's programme on the true sets with every non-causal gain, the decision hour's
  included, those summing to zero over the cars (`o3_split.general_split`, reading D2); LP (24)'s
  box is at most this value.
- `general_split_no_hour0_gain`: the same programme with no decision-hour gain, each car's
  decision-hour power one number for every request (`certificate.certify(..., "full",
  shape_rule=ANY_OPTIMAL)`), shown beside it; at most `general_split`.

LP (24) is solved once per day and serves both products; `alpha_24` (LP (24)'s alpha) and
`trace_27` (LP (27)'s trace, which is the same for every optimal solution) are kept beside the
boxes.

The base set of LP (24) and LP (27) is named by the caller (`BASE_SETS`, no default): only
"day-average" (`base_sets.day_average` of the day's true rows) is built.

A value is a number or None (missing). Every solve that ends in any outcome but optimal (a
timeout included) gives None with the solver's outcome; it is never read as zero or as infeasible.
A missing inner set makes both of its boxes missing with the inner set's outcome.

Readings (each with its alternative in brackets):

  D1. Base set "day-average": the average over the day's own fleet (plan section 12, the earlier
      analysis's day-average form; this reading was confirmed before the
      tag). ["template": the same average over the cars of other days pooled (plan section 12);
      not built, since which days it pools is an open choice of the owner.]
  D2. "General split" (plan section 2: a rule is causal when "G_i(t,s)=0 if s > t or t precedes
      S") lifts both causal restrictions: gains on later requests and gains in the decision hour,
      the latter summing to zero over the cars because the decision hour carries no request (this
      reading was confirmed before the tag). [Decision hour without gain:
      shown beside it as `general_split_no_hour0_gain`. Decision-hour gains with the fleet's
      decision-hour power free per request: not built.] Hour 0 is free per corner in the boxes
      of LP (24) and LP (27) (`inner_box` reading 1), as in the ceiling.
  D3. LP (27)'s value is a range, as the owner decided on 16 November 2026: the largest box found
      by `o3_lp27`'s search over its optimal solutions (`lp27_largest_found`), with the proved
      upper bound `lp27_largest_bound` beside it, a missing bound printed as missing. The largest
      box over all optimal solutions is not a linear programme and the search does not find it;
      it lies between the two. `lp27_smallest_found`, the smallest box over the solutions
      searched, is shown beside them. [The box of the solver's own optimal vertex: it changes
      with the order of the cars, so not reported.]
  D4. Time limits (plan section 13: 30 s box, 600 s LP (24), 300 s others): LP (24) the LP (24)
      limit; LP (27) and every programme of the search over its maps the limit of the others;
      every box programme of that search, both general splits and the block ceiling the box
      limit, as `ceiling_values` gives the certificate. [The general splits under the limit of
      the others.]
  D5. A day whose fleet is empty (every car left out by `days_from_truth`) has every value 0
      without a solve (outcome "empty fleet"), as in `ceiling_values`.
  D6. A day whose base set holds only the zero profile (`base_sets.BaseSet.is_trivial`: every car
      without room), which `lp24` refuses, has LP (24) and LP (27) unsolved (outcome "trivial
      base set"): their value is 0 when the hourly ceiling is at most `ZERO` (every box lies
      between 0 and the ceiling), else missing. [Missing on every such day.]
  D7. The day's cars are put in one fixed order (`car_order`: by departure, rate, room) before
      any set is built, so every value is the same whatever order the cars were listed in.
  D8. LP (27)'s bound is computed for the hourly product first; the block bound is at most the
      hourly one, a box the branch and bound met above the largest found (on the optimal face)
      raises `lp27_largest_found`, and the hourly largest found is at least the block one (a
      block box is an hourly box). [Each product alone.]
  D9. LP (27) is run on some days only (`DayTask.lp27`, plan section 11, O3: 100 of a unit's
      sampled days), and only while the unit's time cap lasts (`DayTask.lp27_deadline`, an
      absolute time from `time.time`). A day that is not one of LP (27)'s days has LP (27)'s
      three values missing with the outcome `NOT_SAMPLED`, whatever its fleet, and the summary
      leaves it out of LP (27)'s statistics (`DayResult.lp27` is False). A day of LP (27) that
      starts after the deadline has them missing with the outcome `CAP_REACHED`: the clock is
      read once, when the day's programmes begin, and only for a day with cars; a day that has
      begun runs to its own end, each solve under its own limit, so a unit's LP (27) can
      run past the cap by up to one day ("about two hours": plan section 11). The values that
      remain (the ceiling, LP (24), both general splits) are computed in any case. [The clock
      read again before the bound, so that a day over the cap gives its search but no bound;
      or a cap on the seconds the solves add up to, not on the clock.]
"""

import dataclasses
import time
from dataclasses import dataclass, field

from fleetreserve import exact, products
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day
from fleetreserve.inner_box import largest_box
from fleetreserve.lp import BOX_SECONDS, LP24_SECONDS, OTHER_SECONDS
from fleetreserve.lp24 import inner_set
from fleetreserve.o3_lp27 import Effort, search
from fleetreserve.o3_lp27_bound import largest_bound, map_ranges
from fleetreserve.o3_split import general_split

PRODUCTS = ("hourly", "block")
METHODS = (
    "lp24",
    "lp27_smallest_found",
    "lp27_largest_found",
    "lp27_largest_bound",
    "general_split",
    "general_split_no_hour0_gain",
)
KEYS = ("ceiling", *METHODS)
LABELS = {
    "lp24": "the largest box in LP (24)'s set",
    "lp27_smallest_found": "LP (27): the smallest box over the optimal solutions searched; the "
    "smallest over all optimal solutions is at most this",
    "lp27_largest_found": "LP (27): the largest box over the optimal solutions searched; the "
    "largest over all optimal solutions is at least this",
    "lp27_largest_bound": "LP (27): no optimal solution's set holds a larger box (a bound proved "
    "by a linear relaxation, to the solver's tolerances)",
    "general_split": "general split, decision-hour gains summing to zero over the cars",
    "general_split_no_hour0_gain": "general split, decision hour without gain",
}
BOUND = "lp27_largest_bound"
INNER = ("lp24", "lp27_smallest_found", "lp27_largest_found", BOUND)
BOUND_NODES = 1  # nodes per product for the bound when the task names none (the root only)
BOUND_ROUNDS = 1  # relaxations per node of the bound when the task names none
BASE_SETS = {"day-average": day_average}  # reading D1
ZERO = 1e-7  # kWh: a value at or below this is zero (as `ceiling_summary.ZERO`)

EXACT = "exact"
OPTIMAL = "optimal"
EMPTY_FLEET = "empty fleet"
TRIVIAL_BASE = "trivial base set"
NOT_BUILT = "day not built"
NOT_SAMPLED = "not a day of LP (27)'s sample"
CAP_REACHED = "not reached within the unit's LP (27) time cap"
LP27_KEYS = ("lp27_smallest_found", "lp27_largest_found", BOUND)  # the values LP (27) gives


def now() -> float:
    """The clock a unit's cap starts on and is read on (`DayTask.lp27_deadline`)."""
    return time.time()


@dataclass(frozen=True)
class TimeLimits:
    """Seconds per solve (reading D4)."""

    lp24: float = LP24_SECONDS
    others: float = OTHER_SECONDS
    box: float = BOX_SECONDS


@dataclass(frozen=True)
class DayTask:
    """One day to compute; `effort` is LP (27)'s search settings (None: the full settings,
    `o3_lp27.full_effort`), `bound_nodes` the nodes per product of its bound and `bound_rounds`
    the relaxations per node (`o3_lp27_bound`). `lp27` says whether the day is one of LP (27)'s
    days and `lp27_deadline` is the time (`time.time`) after which a day's LP (27) is not begun
    (None: no cap); reading D9. `rank` is the day's place in its unit's draw (0 first), the order
    the days are started in."""

    unit: str
    site: str
    date: str
    day: Day
    base_set: str
    limits: TimeLimits
    effort: Effort | None = None
    bound_nodes: int = BOUND_NODES
    bound_rounds: int = BOUND_ROUNDS
    lp27: bool = True
    lp27_deadline: float | None = None
    rank: int = 0

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)


@dataclass(frozen=True)
class DayResult:
    """`values[product][key]` in kWh or None (missing), `outcomes[product][key]` why (module
    text); `alpha_24` and `trace_27` are None unless that programme ended optimal. `error` is set
    when the day's sets could not be built or a step raised; every value is then missing.
    `lp27_skipped` counts what LP (27)'s search passed over (`o3_lp27`: a solution off the
    optimal face, a candidate programme ended infeasible or with a solver error, a candidate
    with a box programme ended with a solver error, an ascent so ended). `lp27` says whether the
    day is one of LP (27)'s days (reading D9)."""

    unit: str
    site: str
    date: str
    cars: int
    n_service: int
    error: str | None
    values: dict[str, dict[str, float | None]]
    outcomes: dict[str, dict[str, str]]
    alpha_24: float | None
    trace_27: float | None
    lp27_skipped: int = 0
    lp27: bool = True
    seconds: dict[str, float] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.unit, self.date, self.site)

    def share(self, product: str, method: str) -> float | None:
        """The method's value as a share of the ceiling; None when either is missing or the
        ceiling is at most `ZERO`."""
        value, ceiling = self.values[product][method], self.values[product]["ceiling"]
        if value is None or ceiling is None or ceiling <= ZERO:
            return None
        return value / ceiling


def car_order(day: Day) -> Day:
    """The day with its cars in one fixed order, by departure, rate and room (reading D7)."""
    cars = tuple(sorted(day.cars, key=lambda car: (car.departure, car.rate, car.room)))
    return dataclasses.replace(day, cars=cars, forecasts=None, kappa=None)


def _number(value) -> float | None:
    return None if value is None else float(value) + 0.0  # -0.0 becomes 0.0


def _blank(value, outcome):
    return (
        {p: dict.fromkeys(KEYS, value) for p in PRODUCTS},
        {p: dict.fromkeys(KEYS, outcome) for p in PRODUCTS},
    )


def _inner_values(
    rows, functions, task: DayTask, hourly_ceiling: float, values, outcomes, seconds, run_lp27
):
    """LP (24)'s box and LP (27)'s range for both products (LP (27)'s only when `run_lp27`);
    returns (alpha_24, trace_27, stops)."""
    if task.base_set not in BASE_SETS:
        raise ValueError(f"unknown base set {task.base_set!r}; built: {sorted(BASE_SETS)}")
    base = BASE_SETS[task.base_set](rows)
    if base.is_trivial():  # reading D6
        value = 0.0 if hourly_ceiling <= ZERO else None
        for product in PRODUCTS:
            for method in INNER:
                values[product][method], outcomes[product][method] = value, TRIVIAL_BASE
        return None, None, 0
    inner = inner_set(rows, base, "24", time_limit=task.limits.lp24)
    seconds["lp24"] = inner.seconds
    for product in PRODUCTS:
        box = largest_box(inner, products.Product(product), time_limit=task.limits.box)
        values[product]["lp24"] = _number(box.value)
        outcomes[product]["lp24"] = box.outcome
    if not run_lp27:
        return _number(inner.value), None, 0
    found = search(rows, base, PRODUCTS, task.limits.others, task.limits.box, task.effort)
    seconds.update(found.seconds)
    for product in PRODUCTS:
        values[product]["lp27_smallest_found"] = _number(found.low[product])
        values[product]["lp27_largest_found"] = _number(found.high[product])
        for key in ("lp27_smallest_found", "lp27_largest_found"):
            outcomes[product][key] = found.outcomes[product]
    _bounds(rows, base, functions, task, found.trace, values, outcomes, seconds)
    return _number(inner.value), _number(found.trace), found.skipped


def _bounds(rows, base, functions, task: DayTask, trace, values, outcomes, seconds):
    """LP (27)'s bound per product (`o3_lp27_bound`), the hourly product first (reading D8),
    with the map ranges solved once for both; `functions` the day's set functions
    (`exact.day_functions`)."""
    ranges = None
    if trace is not None:
        ranges = map_ranges(rows, base, task.limits.others)
        seconds["lp27_bound_ranges"] = ranges[3]
    for product in PRODUCTS:
        ceiling = values[product]["ceiling"]
        found = values[product]["lp27_largest_found"]
        if trace is None:  # LP (27) not solved: no bound, its outcome
            outcomes[product][BOUND] = outcomes[product]["lp27_largest_found"]
            continue
        bound = largest_bound(
            rows,
            base,
            trace,
            products.Product(product),
            functions,
            ceiling,
            found,
            task.bound_nodes,
            task.limits.others,
            task.limits.box,
            ranges,
            task.bound_rounds,
        )
        seconds[f"lp27_bound_{product}"] = bound.seconds
        value = bound.value
        hourly = values["hourly"][BOUND]
        if product == "block" and value is not None and hourly is not None:
            value = min(value, hourly)  # a block box is an hourly box of the same value
        values[product][BOUND], outcomes[product][BOUND] = _number(value), bound.outcome
        if found is not None and bound.incumbent is not None and bound.incumbent > found:
            values[product]["lp27_largest_found"] = _number(bound.incumbent)
    hourly, block = (values[p]["lp27_largest_found"] for p in PRODUCTS)
    if hourly is not None and block is not None and block > hourly:
        values["hourly"]["lp27_largest_found"] = block


def _values(task: DayTask, values, outcomes, seconds, run_lp27):
    day = car_order(task.day)
    functions = exact.day_functions(day)
    rows = true_rows(day)
    hourly = _number(exact.hourly_ceiling(functions))
    values["hourly"]["ceiling"], outcomes["hourly"]["ceiling"] = hourly, EXACT
    block = exact.block_ceiling(functions, time_limit=task.limits.box)
    values["block"]["ceiling"], outcomes["block"]["ceiling"] = _number(block.value), block.outcome
    seconds["ceiling_block"] = block.seconds
    found = _inner_values(rows, functions, task, hourly, values, outcomes, seconds, run_lp27)
    for product in PRODUCTS:
        kind = products.Product(product)
        split = general_split(rows, kind, time_limit=task.limits.box)
        values[product]["general_split"] = _number(split.value)
        outcomes[product]["general_split"] = split.outcome
        seconds[f"general_split_{product}"] = split.seconds
        fixed = certify(rows, kind, "full", shape_rule=ANY_OPTIMAL, time_limit=task.limits.box)
        values[product]["general_split_no_hour0_gain"] = _number(fixed.value)
        outcomes[product]["general_split_no_hour0_gain"] = fixed.outcome
        seconds[f"general_split_no_hour0_gain_{product}"] = fixed.seconds
    return found


def compute_day(task: DayTask) -> DayResult:
    """Every value of one day (module text)."""
    day = task.day
    seconds: dict[str, float] = {}
    error = alpha = trace = None
    stops = 0
    capped = False
    if not day.cars:  # reading D5
        values, outcomes = _blank(0.0, EMPTY_FLEET)
    else:
        values, outcomes = _blank(None, NOT_BUILT)
        capped = task.lp27 and task.lp27_deadline is not None and now() >= task.lp27_deadline
        try:
            alpha, trace, stops = _values(task, values, outcomes, seconds, task.lp27 and not capped)
        except Exception as problem:  # recorded; the command's exit status says so
            error = f"{type(problem).__name__}: {problem}"
            values, outcomes = _blank(None, NOT_BUILT)
            alpha = trace = None
            stops = 0
            capped = False
    if error is None and (not task.lp27 or capped):  # reading D9
        for product in PRODUCTS:
            for key in LP27_KEYS:
                values[product][key] = None
                outcomes[product][key] = CAP_REACHED if task.lp27 else NOT_SAMPLED
        trace = None
    return DayResult(
        unit=task.unit,
        site=task.site,
        date=task.date,
        cars=len(day.cars),
        n_service=day.n_service,
        error=error,
        values=values,
        outcomes=outcomes,
        alpha_24=alpha,
        trace_27=trace,
        lp27_skipped=stops,
        lp27=task.lp27,
        seconds=seconds,
    )
