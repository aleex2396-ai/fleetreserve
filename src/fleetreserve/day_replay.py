"""One day's offer replayed by a rung on a set of request paths: the call the calibration makes.

For a day (true cars and forecasts), its certificate (`certificate.Certificate`, from the forecast
sets), a rung ("fixed", "one-way" or "two-way"), a scale `beta` and paths (`(paths, n)`, entries in
`[-1, 1]`), `replay_day` returns, for that rung and each rung below it, per path: the first failing
service hour, slot loss, driver loss, each car's energy at departure and the re-solves made (hour,
outcome, solver seconds).

Abstain (plan section 1: "offer nothing; cars charge as logged"): a day whose certificate is
missing (any outcome but optimal, a timeout included) and any `beta <= 0` offer nothing. Their
losses are 0 on every path, no energies are given (charging as logged is not modelled here), and
`abstain_reason` says why, so the caller can count them.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fleetreserve.certificate import Certificate
from fleetreserve.day import Day
from fleetreserve.lp import OTHER_SECONDS
from fleetreserve.rungs import RUNGS, TWO_WAY_BUDGET, Resolve, check_departure_seen, run_rungs


@dataclass(frozen=True)
class PathReplay:
    """One rung on one path. `final` is None on an abstaining day."""

    path: np.ndarray
    first_failure: int | None
    slot_loss: float
    driver_loss: float
    final: np.ndarray | None
    resolves: tuple[Resolve, ...]

    def outcomes(self) -> Counter:
        """Re-solves by outcome ("optimal", "infeasible", "timeout", ...)."""
        return Counter(r.outcome for r in self.resolves)


@dataclass(frozen=True)
class DayReplay:
    """`by_rung[name]`: one `PathReplay` per path for the rung asked for and each rung below."""

    rung: str
    beta: float
    abstain_reason: str | None
    by_rung: dict

    @property
    def abstained(self) -> bool:
        return self.abstain_reason is not None

    @property
    def paths(self) -> tuple[PathReplay, ...]:
        return self.by_rung[self.rung]


def replay_day(
    day: Day,
    certificate: Certificate,
    rung: str,
    beta: float,
    paths,
    *,
    departure_bound: Callable[[int, int], float],
    departure_seen: str,
    budget: float = TWO_WAY_BUDGET,
    time_limit: float = OTHER_SECONDS,
) -> DayReplay:
    """Replay `certificate`'s offer at scale `beta` on `paths` with `rung` (see the module text).

    `departure_bound(car, hour)`: the survival-tightened forecast departure of a car present at
    the start of `hour`. `departure_seen`: "hour-start" or "rounded-hour", when a rung learns that
    a car has gone (`rungs.py`, information model), no default. The certificate's outcome and
    `beta`, the only inputs of the abstain decision, are fixed before the day."""
    if rung not in RUNGS:
        raise ValueError(f"rung must be one of {RUNGS}, got {rung!r}")
    check_departure_seen(departure_seen)
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    if paths.shape[1] != day.n_service:
        raise ValueError(f"each path must have {day.n_service} entries")
    names = RUNGS[: RUNGS.index(rung) + 1]
    reason = None
    if certificate.outcome != "optimal":
        reason = f"no certificate ({certificate.outcome})"
    elif beta <= 0:
        reason = "beta <= 0"
    if reason is not None:
        nothing = tuple(PathReplay(z, None, 0.0, 0.0, None, ()) for z in paths)
        return DayReplay(rung, float(beta), reason, {name: nothing for name in names})
    by_rung = {name: [] for name in names}
    for z in paths:
        runs = run_rungs(
            day,
            certificate.policy,
            beta,
            z,
            departure_bound=departure_bound,
            departure_seen=departure_seen,
            last=rung,
            budget=budget,
            time_limit=time_limit,
        )
        for name in names:
            r = runs[name]
            by_rung[name].append(
                PathReplay(
                    z,
                    r.run.first_failure,
                    r.run.slot_loss,
                    r.run.driver_loss,
                    r.run.final,
                    r.resolves,
                )
            )
    return DayReplay(rung, float(beta), None, {k: tuple(v) for k, v in by_rung.items()})
