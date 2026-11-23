"""The scoring run's evaluations for the intervals over days (plan section 11: "intervals: 95%
bootstrap over days"; the intervals are `day_intervals`'s).

The intervals recompute the scale of every resample by the run's own calibration:
`crc.calibrate_search` on the resample's offering calibration days, as
`gate_calibrate.calibrate_rung` calls it. That search asks for the losses of the resampled days
at scales the run's own searches never asked for, and a day's loss at a new scale needs its
replay. `gate_run` and `learn_run` do that replay here when given `--interval-seed`: after the
run's own calibration and scoring are done, with the same prepared days, the same evaluation
(`gate_day.evaluate`) and the same worker pool.

`resample_pairs(...)`, for one job (one configuration group on one unit): the draws are
`day_bootstrap.draws(seed, unit, ...)`, the ones `day_intervals` uses. For every resample, every
rung and every tier the run scores (tier A: both losses at `crc.ALPHA_SLOT` and `crc.ALPHA_DRIVER`;
tier B: the slot loss alone, as `gate_calibrate` calibrates it), the search is run on the
resample's offering calibration days, each drawn day as often as it is drawn. Every (day, scale)
pair the search asks for that is not yet known is evaluated once, and kept in one store shared by
every resample, rung and tier of the job. The store starts from a copy of the run's own
evaluations, which are left as they are, so no other output of the run changes.

A pair whose evaluation raises an error is recorded as failed, with the error; it is evaluated no
second time. A search that meets a failed pair is run twice, with that pair's losses 1 and then 0
(`day_bootstrap` reading I3), and every pair either run asks for is evaluated, so the intervals
command can bound that resample's scale from the stored pairs alone. An evaluation has no time
limit of its own: each re-solve inside it has the run's own time limit and records its outcome
(`gate_day`), as in the run's own calibration, so a re-solve that times out gives the losses the
run itself would give at that pair.
"""

import time
from dataclasses import dataclass

import numpy as np

from fleetreserve import crc, day_bootstrap
from fleetreserve.dial_splits import tier_alphas
from fleetreserve.rungs import RUNGS


@dataclass(frozen=True)
class Resampled:
    """The pairs one job's resamples needed beyond the run's own (`resample_pairs`)."""

    seed: int
    resamples: int
    tiers: tuple
    evaluated: dict  # (day, scale) -> the day's evaluation (`gate_day.DayEvaluation`, compact)
    failed: dict  # (day, scale) -> the error, as text
    own_calibration_pairs: int  # the run's own evaluations of offering calibration days
    searches: int  # calibration searches run, a failed pair's second run included
    seconds: float


class _Store:
    """Every known evaluation of one job, and per scale the losses as arrays over its days."""

    def __init__(self, prepared, store: dict, evaluate_batch):
        self.prepared = prepared
        self.known = dict(store)
        self.own = frozenset(store)
        self.failed: dict = {}
        self.evaluate_batch = evaluate_batch
        self.columns: dict = {}
        self.by_scale: dict = {}
        for d, beta in self.known:
            self.by_scale.setdefault(beta, []).append(d)

    def _column(self, beta: float) -> dict:
        found = self.columns.get(beta)
        if found is None:
            n = len(self.prepared)
            found = {
                "known": np.zeros(n, dtype=bool),
                "failed": np.zeros(n, dtype=bool),
                **{(rung, kind): np.zeros(n) for rung in RUNGS for kind in ("slot", "driver")},
            }
            self.columns[beta] = found
            self._fill(found, beta, self.by_scale.get(beta, []))
        return found

    def _fill(self, column: dict, beta: float, days) -> None:
        for d in days:
            e = self.known[(d, beta)]
            column["known"][d] = True
            for rung in RUNGS:
                column[(rung, "slot")][d] = e.rungs[rung].slot_loss
                column[(rung, "driver")][d] = e.rungs[rung].driver_loss

    def _evaluate(self, beta: float, days: list) -> list:
        """Evaluate `days` at `beta`; return the days evaluated (the failed ones left out)."""
        pairs = [(d, beta) for d in days]
        try:
            results = self.evaluate_batch([(self.prepared[d], beta) for d in days])
        except Exception:  # isolated pair by pair below, so one bad pair costs no other
            results = None
        done = []
        if results is not None:
            for key, result in zip(pairs, results, strict=True):
                self.known[key] = result
                done.append(key[0])
            return done
        for key in pairs:
            try:
                (result,) = self.evaluate_batch([(self.prepared[key[0]], beta)])
            except Exception as problem:  # recorded with the pair
                self.failed[key] = f"{type(problem).__name__}: {problem}"
                continue
            self.known[key] = result
            done.append(key[0])
        return done

    def losses(self, days: np.ndarray, beta: float, rung: str, fill: float):
        """The losses of `days` (with repeats) at `beta` for `rung`, evaluating what is not known;
        a failed pair's losses are `fill`. Also whether a failed pair was met."""
        column = self._column(beta)
        wanted = np.unique(days[~(column["known"][days] | column["failed"][days])])
        if wanted.size:
            done = self._evaluate(beta, [int(d) for d in wanted])
            self._fill(column, beta, done)
            for d in set(int(x) for x in wanted) - set(done):
                column["failed"][d] = True
        failed = column["failed"][days]
        slot = np.where(failed, fill, column[(rung, "slot")][days])
        driver = np.where(failed, fill, column[(rung, "driver")][days])
        return slot, driver, bool(failed.any())


def _search(store: _Store, days: np.ndarray, rung: str, tier: str, fill: float) -> bool:
    """One calibration search; True when it met a failed pair."""
    met = False

    def loss_at(beta):
        nonlocal met
        slot, driver, failed = store.losses(days, float(beta), rung, fill)
        met = met or failed
        return slot, driver

    crc.calibrate_search(loss_at, int(days.size), *tier_alphas(tier))
    return met


def resample_pairs(
    prepared, store: dict, roles, unit: str, seed: int, resamples: int, tiers, evaluate_batch
) -> Resampled:
    """The pairs the resamples of one job need (module text). `prepared`: the job's prepared days;
    `store`: the run's own evaluations (`gate_calibrate.DayCache.store`), not changed; `roles`:
    per day "calibration" or "test" (others are neither); `evaluate_batch`: as
    `gate_calibrate.DayCache` takes it."""
    started = time.perf_counter()
    calibration = np.array([d for d, role in enumerate(roles) if role == "calibration"], dtype=int)
    n_test = sum(role == "test" for role in roles)
    offers = np.array([not prepared[d].abstains for d in calibration], dtype=bool)
    rows, _ = day_bootstrap.draws(seed, unit, int(calibration.size), n_test, resamples)
    found = _Store(prepared, store, evaluate_batch)
    searches = 0
    for row in rows:
        days = calibration[row][offers[row]]
        for rung in RUNGS:
            for tier in tiers:
                searches += 1
                if _search(found, days, rung, tier, 1.0):
                    searches += 1
                    _search(found, days, rung, tier, 0.0)
    offering = set(calibration[offers].tolist())
    return Resampled(
        seed,
        resamples,
        tuple(tiers),
        {k: v for k, v in found.known.items() if k not in found.own},
        dict(found.failed),
        sum(1 for d, _ in found.own if d in offering),
        searches,
        time.perf_counter() - started,
    )
