"""One scoring run of a model through the full method, as WP16 reads it (plan sections 7, 11).

A `ModelRun` is a learn step of the drive (`learn-<model>`, `fleetreserve.learn_run`): its
`learn.json`, the committed matrix it ran on (its SHA-256 must be the one `learn.json` records),
each unit's headline configuration (`sealed_choice` of that matrix), and, when the run list has
one for the run, its `day-intervals.json` (`intervals-<model>`, `fleetreserve.day_intervals`).

Figures are read from `learn.json` as `learn_run` and `learn_eval` write it:
`units.<unit>.configurations.<name>.rungs.<rung>.test` (tier A) and
`...rungs.<rung>.tier_b.test` (tier B); a unit that has no entry (the held-out truth build gave it
no window) or whose configuration holds only a reason (`failed: <error>`, "not calibratable") has no
figure, and `unit_test` says why. Nothing is recomputed: a figure is the run's own.
"""

from dataclasses import dataclass

from fleetreserve.wp16_drive import Missed, Source
from fleetreserve.wp16_units import beside_name, headline_by_unit, interval_settings, matrix_of

RUNGS = ("fixed", "one-way")
RESULTS = "learn.json"
INTERVALS = "day-intervals.json"
HINDSIGHT = "hindsight-free-decision-hour"
BEST_CAUSAL = "best-causal"


@dataclass
class ModelRun:
    label: str
    learn: dict
    matrix: dict
    headline: dict[str, str]
    intervals: dict | None
    intervals_why: str


def load(source: Source, learn_label: str, intervals_label: str | None = None) -> ModelRun:
    """The run of the learn step `learn_label`; raises `Missed` when its `learn.json` or its
    matrix cannot be read. A missing `day-intervals.json` is not an error: `intervals` is None
    and `intervals_why` says why; so it is when the file's seed and number of resamples are not the
    run list's (plan section 11: 2,000 resamples, the seed fixed in the run list)."""
    learn = source.json(learn_label, RESULTS)
    matrix_hash = (
        learn.get("matrix", {}).get("sha256") if isinstance(learn.get("matrix"), dict) else None
    )
    matrix = matrix_of(source, learn_label, matrix_hash)
    intervals, why = None, "no day-intervals entry in the run list for this run (plan section 11)"
    if intervals_label is not None:
        try:
            found = source.json(intervals_label, INTERVALS)
            seed, resamples = interval_settings(source)
            if (found.get("seed"), found.get("resamples")) != (seed, resamples):
                raise Missed(
                    f"unreadable: {intervals_label}/{INTERVALS}: seed {found.get('seed')} and "
                    f"{found.get('resamples')} resamples, not the run list's {seed} and {resamples}"
                )
            intervals, why = found, ""
        except Missed as problem:
            why = problem.text
    return ModelRun(learn_label, learn, matrix, headline_by_unit(matrix), intervals, why)


def names_for(run: ModelRun, units: list[str], reading: str) -> dict[str, str | None]:
    """Each unit's configuration: its headline (`reading` "main") or the `beside` configuration
    of the same kappa and dial (`reading` "beside"); None where there is none."""
    out: dict[str, str | None] = {}
    for unit in units:
        name = run.headline.get(unit)
        if name is not None and reading == "beside":
            name = beside_name(run.matrix, name)
        out[unit] = name
    return out


def _get(document, *keys):
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def unit_test(
    run: ModelRun, unit: str, name: str | None, rung: str, tier: str, absent: dict[str, str]
) -> tuple[dict | None, str]:
    """(the rung's test-day figures of a unit under a configuration, "") or (None, why there is
    none); `absent`: the units the held-out truth build gave no window, with its reason."""
    entry = _get(run.learn, "units", unit)
    if not isinstance(entry, dict):
        if unit in absent:
            return None, f"no figure: no window ({absent[unit]}); counts as not passing"
        return None, "no figure: the unit is not in the run"
    if name is None:
        return None, "no figure: the matrix names no such configuration for the unit"
    rungs = _get(entry, "configurations", name, "rungs")
    if not isinstance(rungs, dict):
        return None, f"no figure: the run holds no configuration {name} for the unit"
    if rung not in rungs:
        reason = rungs.get("reason", "no result")
        text = reason if str(reason).startswith("failed:") else f"no figure: {reason}"
        return None, f"{text}; counts as not passing"
    found = rungs[rung] if tier == "A" else rungs[rung].get("tier_b")
    test = found.get("test") if isinstance(found, dict) else None
    if not isinstance(test, dict):
        return None, f"no figure: no tier {tier} test-day result"
    return test, ""


def scale_of(run: ModelRun, unit: str, name: str | None, rung: str, tier: str = "A"):
    rungs = _get(run.learn, "units", unit, "configurations", name, "rungs")
    found = rungs.get(rung) if isinstance(rungs, dict) else None
    if tier == "B" and isinstance(found, dict):
        found = found.get("tier_b")
    return found.get("beta_hat") if isinstance(found, dict) else None


def counting_days(run: ModelRun, unit: str) -> int | None:
    days = _get(run.learn, "units", unit, "test_days")
    return days if isinstance(days, int) and not isinstance(days, bool) else None


def unit_intervals(run: ModelRun, name: str | None, unit: str, rung: str, tier: str) -> dict | None:
    """The intervals `day-intervals.json` holds for a cell, or None (no such cell, or the cell's
    per-day file differed from the run, so none was written)."""
    found = _get(run.intervals, "configurations", name, unit, rung, tier)
    return found if isinstance(found, dict) else None


def pooled_block(run: ModelRun, name: str, rung: str, tier: str) -> dict | None:
    """The pooled figures of one configuration (`day_intervals` readings I13 to I17), or None."""
    found = _get(run.intervals, "pooled", "configurations", name, "rungs", rung, tier)
    return found if isinstance(found, dict) else None


class Runs:
    """The model runs of one drive, each loaded once; `get` raises the `Missed` of a run that
    cannot be read, the same one again on a second ask."""

    def __init__(self, source: Source):
        self.source = source
        self._found: dict[tuple, ModelRun | Missed] = {}

    def get(self, learn_label: str, intervals_label: str | None = None) -> ModelRun:
        key = (learn_label, intervals_label)
        if key not in self._found:
            try:
                self._found[key] = load(self.source, learn_label, intervals_label)
            except Missed as problem:
                self._found[key] = problem
        found = self._found[key]
        if isinstance(found, Missed):
            raise found
        return found
