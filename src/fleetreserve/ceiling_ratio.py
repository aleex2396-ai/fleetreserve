"""Outcome O8's ceiling ratio V at the two-way budget, on the true sets of a run's test days, and
its offer ratio X with outcome O5's offer order, from the run's own offers (plan sections 2, 4 (T1),
5 (T5), 9 and 11 (O5, O8)); `ledger_run` reports them.

Outcome O8 reads "B=5: offer ratio X (pass: >= 1 daily), discharge D kWh per car-day, ceiling
ratio V", and section 9 sets the offer ratio "beside the exact two-way/one-way ceiling ratio".
This module gives V and X. The discharge D is not run (the owner's decision of 16 November 2026:
"the energy given back is reported as not run").

- `day_ceilings(day, budget)`: the day's two hourly hindsight ceilings on its true cars, the
  decision hour and the recovery hours free (plan section 2: V_hind = b(S) - p(S)): one-way,
  `exact.hourly_ceiling(exact.day_functions(day))`, the value a scoring run records as
  `ceiling_hindsight_kwh` (`gate_day.day_ceilings`); and two-way at `budget` kWh,
  `exact.hourly_ceiling(exact.day_functions(day, budget))`. A day without cars has both 0, as in
  `gate_day.day_ceilings`. The budget has no default: the caller passes the matrix's.
- `ratio(days)`: over one unit's test days, given as (day text, one-way, two-way): the two sums,
  V, and the days whose one-way ceiling is 0, counted and named with their two-way sum.
- `offer_share(offers, two_way)`: the two-way offer's share of the two-way ceiling over the same
  days (a ratio of sums), or None when the two-way sum is 0.
- `offer_ratio(days)`: over one unit's test days, given as (day text, fixed, one-way, two-way offer
  of one configuration and tier, the run's own offers in kWh): the three sums, X (the two-way sum
  over the one-way sum), the days whose one-way offer is 0, counted and named with their two-way
  sum, and, apart, the days on which fixed <= one-way holds and the days on which one-way <=
  two-way holds (plan T5, outcome O5; the second is outcome O8's daily pass), each step with its
  share and its other days named with their three offers.
- `budget_label(budget)`: "B=<budget>", written from the value itself, so it reads "B=5" only for
  5 kWh; `budget_note(budget)` is empty at 5 kWh and says otherwise that the figure is not
  outcome O8's.

What "exact" rests on. Each car's two-way true set (`day.TrueCar.two_way`, reading A of `day`:
power in [-r, r] before the departure hour, the energy stored since the decision time in [-B, e]
after every hour, at least (1 - rho) e stored at the end) gives its set functions p_i(A), b_i(A)
by `exact`'s forward and backward passes. Their argument is in `exact`'s module text: from a
lower stored energy the car can follow any plan from a higher one while drawing at least as much
every hour, and from a higher one any plan from a lower one while losing at most the difference,
so the best value ahead never rises with the stored energy and falls by at most one unit per unit.
The fleet's functions are the cars' sums (plan T1; Frank 2011, Theorem 14.2.15), and the hourly
ceiling is b(S) - p(S). It is a closed form: no programme, no time limit, no timeout. It is
tested against a programme over every corner path with the stored-energy rows written out on 50
made-up days at budgets 0, 0.5 and 2 kWh (`tests/wp4/test_ceilings.py`), and against such a
programme written in `tests/wp14/test_ceiling_ratio.py` on hand-worked days at 2 and 5 kWh.

What it is exact for: the true sets as the tables model them (`day`: departures rounded down to
the whole hour, limits after every hour, lossless cars, no site cap, the run's truth setting and
rho). It is not exact for the cars' behaviour within an hour, nor for a two-way set read another
way (reading B of `day`, total-energy limits only, is not used for true cars).

Readings (where the plan does not decide; each alternative in brackets):

  C1. V is the sum over the split's test days of the two-way ceiling over the sum of the one-way
      ceiling: a ratio of sums, as the ledger's shares of the ceiling (`ledger_run`, L8).
      [Alternatives: the mean of daily ratios, undefined on a day whose one-way ceiling is 0;
      best-causal two-way ceilings, a programme per day, not built.]
  C2. Beside V, per configuration and tier, the run's two-way offer as a share of the two-way
      ceiling over the same test days (outcome O1's "Z% of the ceiling" form), from the run's own
      day rows. [Alternative: also its share of the one-way ceiling; the ledger's tables give it
      on the test days with V_aff.]
  C3. A day whose one-way ceiling is 0 (at most `ZERO_KWH`) stays in both sums; such days are
      counted and named with their two-way sum. V is not defined (None) when the one-way sum is
      at most `ZERO_KWH`. [Alternative: V over the days with a one-way ceiling above 0 only,
      which the shown two-way sum of the other days gives.]
  C4. The budget is the matrix's `two_way_budget_kwh`, which is the budget of the run's two-way
      rung (the matrix is the run's: its SHA-256 is checked); no option sets another. At another
      value than 5 kWh the figures are reported at the matrix's value, labelled with it, and
      every line says that it is not outcome O8's budget. A matrix without the field is refused
      by `gate_config` before anything is read. [Alternative: refuse any budget but 5 kWh.]
  C5. The one-way ceiling recomputed here must equal the run's `ceiling_hindsight_kwh` on every
      test day of every configuration within `ledger_run.CROSS_TOLERANCE` of max(1, |value|),
      the tolerance of the ledger's check 3; a difference is a failed check naming the day.
      [Alternative: equal floats.]
  C6. V and its sums are per unit and the same for every configuration (the true sets do not
      depend on it); units are never pooled and tiers never averaged. X and its share (C7 to C9)
      are per unit, configuration and tier, and never pooled either.

  C7. X is, per unit, configuration and tier, the sum over the split's test days of the run's
      two-way offer over the sum of its one-way offer, a ratio of sums as V is (C1). The offers are
      the run's day rows' `<rung>_offer_kwh` (tier A) or `<rung>_tier_b_offer_kwh` (tier B), the
      calibrated offers the run scored; the two-way rung is the run's, at the matrix's budget
      (C4). [Alternative: the mean of the daily ratios, which O8's "pass: >= 1 daily" could also
      mean; undefined on a day whose one-way offer is 0, and the daily check of C8 gives its
      pass-mark reading.]
  C8. "Day-paths" of outcome O5 (offer ratio >= 1 on 100% of day-paths) are read as the test
      days: a run records one calibrated offer per rung and day, none per request path. A day is
      ordered when fixed <= one-way <= two-way, each step with the tolerance `ORDER_TOLERANCE` of
      max(1, the larger offer) (plan T5; the three offers are the scale of the rung times the
      same certified value, so the check follows the order of the scales on every day with a
      certified value above 0). The two steps are reported apart, each with its share of test
      days and its broken days named with their three offers: fixed <= one-way, and one-way <=
      two-way (outcome O8's daily pass). Outcome O5 calls a violation a bug and plan T5 says the
      order always holds, so a day that breaks either step is a failed check of the ledger
      (`ledger_run`, check 7: exit status 1, outputs still written). [Alternatives: a report
      that leaves the exit status alone; the check per request path, which would need the
      offers per path, not recorded.]
  C9. A day whose one-way offer is 0 (at most `ZERO_KWH`) stays in both sums and in the share; such
      days are counted and named with their two-way sum. X is not defined (None) when the one-way
      sum is at most `ZERO_KWH`. [Alternative: X over the days with a one-way offer above 0 only.]

Leak list (on the sealed route and off it alike): nothing is fitted, chosen, tuned, sampled or
calibrated here. The ceilings are hindsight quantities of the scored test days, computed on their
true sets after the run scored its offers; the budget is the matrix's; no scale, configuration,
kappa, dial or day is picked from V, from X or from the shares. On the route `ledger_run` hands this
module the days it has already rebuilt from the held-out build's tables; this module opens no
file.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from fleetreserve import exact
from fleetreserve.day import Day

PLAN_BUDGET_KWH = 5.0  # outcome O8's budget, "B=5"
ZERO_KWH = exact.TOLERANCE  # a ceiling or a sum at most this is 0 (reading C3)
ORDER_TOLERANCE = 1e-6  # relative to max(1, the larger offer): the offer order's tolerance (C8)
READINGS = tuple(f"C{k}" for k in range(1, 10))
DISCHARGE = "not run (the owner's decision of 16 November 2026)"
OFFER_RATIO = (
    "X = the sum over a unit's test days of the run's two-way offer over the sum of its one-way "
    "offer (the day rows' two-way_offer_kwh and one-way_offer_kwh, or their tier_b columns), per "
    "unit, configuration and tier, never pooled; beside it the share of test days on which the "
    "run's offers obey fixed <= one-way and the share on which they obey one-way <= two-way "
    "(plan T5, outcome O5; the second is the daily pass), the days that do not named; a day that "
    "breaks either step fails the ledger"
)
DEFINITION = (
    "V = the sum over the split's test days of the exact two-way hindsight ceiling at the budget "
    "over the sum of the exact one-way one, per unit, on the true sets, the decision hour and the "
    "recovery hours free (exact.hourly_ceiling); beside it, per configuration and tier, the run's "
    "two-way offer over the two-way ceiling, summed over the same days"
)


def day_ceilings(day: Day, budget: float) -> tuple[float, float]:
    """(one-way, two-way at `budget` kWh) hourly hindsight ceilings of the day's true cars, in kWh
    (module text). Raises `ValueError` when a set cannot be built."""
    if not day.cars:
        return 0.0, 0.0
    one_way = exact.hourly_ceiling(exact.day_functions(day))
    two_way = exact.hourly_ceiling(exact.day_functions(day, budget))
    return one_way, two_way


@dataclass(frozen=True)
class Ratio:
    """V and its sums over one unit's test days (module text)."""

    test_days: int
    one_way_sum_kwh: float
    two_way_sum_kwh: float
    v: float | None
    one_way_zero_days: tuple[str, ...]
    two_way_sum_on_one_way_zero_days_kwh: float


def ratio(days: Sequence[tuple[str, float, float]]) -> Ratio:
    """V over `days`, each (day text, one-way ceiling, two-way ceiling) in kWh (reading C1, C3)."""
    one = float(sum(d[1] for d in days))
    two = float(sum(d[2] for d in days))
    zero = [d for d in days if d[1] <= ZERO_KWH]
    return Ratio(
        test_days=len(days),
        one_way_sum_kwh=one,
        two_way_sum_kwh=two,
        v=None if one <= ZERO_KWH else two / one,
        one_way_zero_days=tuple(d[0] for d in zero),
        two_way_sum_on_one_way_zero_days_kwh=float(sum(d[2] for d in zero)),
    )


def offer_share(offers: Sequence[float], two_way: Sequence[float]) -> float | None:
    """The offers' sum over the two-way ceilings' sum of the same days (reading C2)."""
    if len(offers) != len(two_way):
        raise ValueError("one offer and one two-way ceiling are needed per day")
    total = float(sum(two_way))
    return None if total <= ZERO_KWH else float(sum(offers)) / total


@dataclass(frozen=True)
class OfferRatio:
    """X, the order of the offers and their sums over one unit's test days (module text)."""

    test_days: int
    fixed_sum_kwh: float
    one_way_sum_kwh: float
    two_way_sum_kwh: float
    x: float | None
    one_way_zero_days: tuple[str, ...]
    two_way_sum_on_one_way_zero_days_kwh: float
    fixed_one_way_days: int
    fixed_one_way_share: float | None
    fixed_one_way_broken_days: tuple[str, ...]
    one_way_two_way_days: int
    one_way_two_way_share: float | None
    one_way_two_way_broken_days: tuple[str, ...]


def _ordered(first: float, second: float) -> bool:
    """Whether first <= second, within `ORDER_TOLERANCE` of max(1, the larger)."""
    return first - second <= ORDER_TOLERANCE * max(1.0, abs(first), abs(second))


def _named_day(day: tuple[str, float, float, float]) -> str:
    return f"{day[0]} (fixed {day[1]:.6g}, one-way {day[2]:.6g}, two-way {day[3]:.6g} kWh)"


def offer_ratio(days: Sequence[tuple[str, float, float, float]]) -> OfferRatio:
    """X and the offer order over `days`, each (day text, fixed, one-way, two-way offer) in kWh
    (readings C7 to C9)."""
    fixed = float(sum(d[1] for d in days))
    one = float(sum(d[2] for d in days))
    two = float(sum(d[3] for d in days))
    zero = [d for d in days if d[2] <= ZERO_KWH]
    first = [d for d in days if not _ordered(d[1], d[2])]
    second = [d for d in days if not _ordered(d[2], d[3])]
    return OfferRatio(
        test_days=len(days),
        fixed_sum_kwh=fixed,
        one_way_sum_kwh=one,
        two_way_sum_kwh=two,
        x=None if one <= ZERO_KWH else two / one,
        one_way_zero_days=tuple(d[0] for d in zero),
        two_way_sum_on_one_way_zero_days_kwh=float(sum(d[3] for d in zero)),
        fixed_one_way_days=len(days) - len(first),
        fixed_one_way_share=(len(days) - len(first)) / len(days) if days else None,
        fixed_one_way_broken_days=tuple(_named_day(d) for d in first),
        one_way_two_way_days=len(days) - len(second),
        one_way_two_way_share=(len(days) - len(second)) / len(days) if days else None,
        one_way_two_way_broken_days=tuple(_named_day(d) for d in second),
    )


def budget_label(budget: float) -> str:
    """The label of a budget, written from the value itself: "B=5" at 5 kWh, "B=0.5" at 0.5 kWh
    (reading C4)."""
    value = float(budget)
    return f"B={int(value)}" if value.is_integer() else f"B={value!r}"


def budget_note(budget: float) -> str:
    """Empty at outcome O8's budget; otherwise the words that the figure is not O8's (C4)."""
    if float(budget) == PLAN_BUDGET_KWH:
        return ""
    return f", not outcome O8's {budget_label(PLAN_BUDGET_KWH)}"
