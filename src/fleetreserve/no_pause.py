"""The draw of the no-pause sensitivity: which cars of a day charge as logged (plan section 9: "a
seeded 0 or 1/3 of cars charge as logged, without flexibility (no-pause; Brinkel et al. 2024)").

Brinkel et al. (2024) report that around one third of the car models they tested could not
handle long charging pauses. Here a drawn car is taken out of the steered fleet: given to
`gate_day.prepare_day` as `no_pause`, it is left unsteered with the reason `gate_day.NO_PAUSE`,
by the route a car without a usable forecast takes under "leave-unsteered": it charges as logged,
is outside the offered box, loses nothing and is in no programme.

`SHARES`: the two shares the plan names, by their written form ("0" and "1/3"); `read_share(text)`
gives the share of one of them and refuses any other text with `NoPauseError`. `DEFAULT_SEED`: the
seed used when none is given.

`is_drawn(site, session_id, share, seed)`: whether one car is drawn. The first 8 bytes of the
SHA-256 of the text `no-pause|<seed>|<site>|<session id>`, read as a whole number h (big-endian,
0 <= h < 2^64), draw the car when h < share x 2^64, compared in whole numbers with the share as a
fraction (at share 0 no car is drawn). A share that is not one of `SHARES`, a seed that is not
a whole number, and a site or session identifier that contains `|` (which would make two cars'
texts equal) are refused with `NoPauseError`.

`drawn(truth, share, seed)`: the positions in `truth.day.cars` of the cars drawn, in order, read
from the fleet's site and session identifier (`days_from_truth.FleetCar`).

`without(day, cars)`: the same day with the cars at those positions removed (the other fields
kept), for the ceilings of the fleet without the drawn cars.

`counts(truths, draws)`: over a unit's days and the draw of each, the numbers of cars, of cars
drawn and of days on which every car is drawn (a day without cars is not one of them).

Readings of the plan's wording (each is the closest reading; the alternative is in brackets):

  N1. What is drawn: each car of a day's fleet (plan section 2: the cars present at τ) by itself,
      with probability equal to the share, from its site and session identifier. A session
      plugged in over several days (`days_from_truth`) keeps its identifier, so it is drawn alike
      on each of those days. No count per day is fixed: a day of n cars has every car drawn with
      probability share^n (1/27 for three cars at 1/3), and then it has no steered car and
      abstains (`gate_day`: "no steered car"); it may also have no car drawn. [Alternative: exactly
      a third of each day's cars, rounded, which sets the count of a small day by the rounding
      rule (of three cars always one), not by the share; or a third of drivers by the tables'
      `user_id`, which follows the paper's point that the fault belongs to the car model, but an
      identifier of a driver is not one of a car, and the plan's word is cars.]
  N2. The same draw everywhere: the draw reads only the share, the seed and the car's site and
      session identifier, and nothing of the configuration, the departure variant, the method,
      the split, the scale, the path, the number of worker processes or the order of the days.
      [Alternative: a draw per variant or per split, which would mix the spread of the draw into
      every comparison between methods.]
  N4. Where the draw is used: only `compare_run` takes it, and of the rungs it writes only the
      one-way re-solve rung (`dial_choice.RUNG`), beside the controllers. Plan section 9 names
      the sensitivity under two-way charging; no command writes a two-way rung's figures under
      the draw. The two-way rung is still computed: `gate_day.evaluate`, which `compare_run`
      calls for the re-solve, replays every rung of `rungs.RUNGS` on a day that offers, at a
      scale above 0, and `compare_run` reads only `dial_choice.RUNG` of it. [Alternative: also
      the two-way rungs of `gate_run`, not built.]
  N5. The forecasts: the draw decides only which cars of a day are steered. The model, the units
      it is fitted on and the units' cars that the forecaster reads are those of the run without
      the draw, so the drawn cars stay in the data the forecasts come from (the forecast table
      `compare_run` writes is the same bytes); a drawn car is only not forecast on a day it is
      drawn. [Alternative: take the drawn cars out of that data, not built.]
"""

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction

from fleetreserve.day import Day
from fleetreserve.days_from_truth import TruthDay

SHARES = {"0": Fraction(0), "1/3": Fraction(1, 3)}
DEFAULT_SEED = 1
_SPACE = 2**64


class NoPauseError(ValueError):
    """A share or seed of the no-pause sensitivity that cannot be used; the message says why."""


def read_share(text: str) -> Fraction:
    """The share written as `text`, one of `SHARES`; raises `NoPauseError` otherwise."""
    if text not in SHARES:
        raise NoPauseError(
            f"--no-pause-share must be one of {list(SHARES)} (plan section 9 names only these "
            f"two), got {text!r}"
        )
    return SHARES[text]


def share_text(share: Fraction) -> str:
    """The written form of a share of `SHARES`."""
    for text, value in SHARES.items():
        if value == share:
            return text
    raise NoPauseError(f"the share {share} is not one of {list(SHARES)}")


def _check(share: Fraction, seed: int) -> None:
    share_text(share)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NoPauseError(f"the seed must be a whole number, got {seed!r}")


def is_drawn(site: str, session_id: str, share: Fraction, seed: int) -> bool:
    """Whether the car of `site` and `session_id` is drawn (module text)."""
    _check(share, seed)
    if "|" in str(site) or "|" in str(session_id):
        raise NoPauseError(f"a site or session identifier holds '|': {site!r}, {session_id!r}")
    text = f"no-pause|{seed}|{site}|{session_id}".encode()
    h = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    return h * share.denominator < share.numerator * _SPACE


def drawn(truth: TruthDay, share: Fraction, seed: int) -> tuple[int, ...]:
    """The positions of the day's cars that are drawn (module text)."""
    if len(truth.fleet) != len(truth.day.cars):
        raise ValueError("the fleet and the day's cars must have one entry per car")
    return tuple(
        i for i, car in enumerate(truth.fleet) if is_drawn(car.site, car.session_id, share, seed)
    )


def without(day: Day, cars) -> Day:
    """`day` without the cars at the positions `cars` (module text)."""
    removed = set(cars)
    if not removed <= set(range(len(day.cars))):
        raise ValueError(f"car positions {sorted(removed)} are not all cars of the day")
    kept = tuple(c for i, c in enumerate(day.cars) if i not in removed)
    forecasts = None
    if day.forecasts is not None:
        forecasts = tuple(f for i, f in enumerate(day.forecasts) if i not in removed)
    return replace(day, cars=kept, forecasts=forecasts)


@dataclass(frozen=True)
class Counts:
    cars: int  # cars of the days, a session on several days counted once per day
    drawn: int
    days_every_car_drawn: int

    def as_dict(self) -> dict:
        return {
            "cars": self.cars,
            "cars_drawn": self.drawn,
            "days_every_car_drawn": self.days_every_car_drawn,
        }


def counts(truths, draws) -> Counts:
    """The counts of `truths` (days) under `draws` (the drawn positions of each day)."""
    truths, draws = list(truths), list(draws)
    if len(truths) != len(draws):
        raise ValueError("one draw per day is needed")
    cars = sum(len(t.day.cars) for t in truths)
    found = sum(len(d) for d in draws)
    every = sum(
        1 for t, d in zip(truths, draws, strict=True) if t.day.cars and len(d) == len(t.day.cars)
    )
    return Counts(cars, found, every)
