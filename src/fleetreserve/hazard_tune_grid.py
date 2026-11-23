"""The candidate settings of the neural hazard model and the rule that picks one (plan section 7,
hazard model; section 13, WP8).

Settings (`Setting`): `width`, `learning_rate`, `kl_weight` (the weight of the divergence term of
the day factor) and `epochs`. Every other field of `hazard_settings.HazardSettings` stays at its
default, and the seed is not a setting: it is run over the grid file's list of seeds
(`Grid.seeds`), a candidate's score being the mean over the seeds. `PRESENT` holds the defaults of
`HazardSettings` (a test compares the two).

Grid file (`GRID`, `learn/hazard-grid.json`): the lists `seeds`, `epochs`, `width`,
`learning_rate` and `kl_weight`. A fitted candidate (`Grid.fits`) is a combination of one width,
one learning rate and one divergence weight, in the order of the file (width, then learning rate,
then divergence weight); a candidate (`Grid.candidates`) is a fitted candidate with one of the
epoch values, so the epoch values are read off one fit as checkpoints (`hazard_fit.fit`,
argument `checkpoints`; the epoch values are sorted, the candidates run through them last).
`load_grid` refuses (`GridError`) a file that cannot be read or has other keys, an empty list or
a value repeated in a list (1 and 1.0 are the same value), a seed that is not a whole number of
at least 0, an epoch value or width that is not a whole number of at least 1, a learning rate
outside (0, 1], a divergence weight below 0, a combination that `HazardSettings` refuses (a
width that is not a multiple of its number of heads), or a grid that does not hold `PRESENT`: the
present settings are always a candidate, so a result always says how the present model ranks.
The grid's hash is the SHA-256 of the file's bytes.

Choice rule (`choose`), fixed before any run on training data: the candidate with the lowest
score. Candidates whose score is within `learn_tune_grid.TIE` of the lowest are tied, and the tie
goes to the smaller width, then the fewer epochs, then the larger divergence weight, then the
larger learning rate. Nothing else enters: no one-standard-error rule and no judgement. `rank` of
a candidate is `learn_tune_grid.rank`: one more than the number of candidates with a strictly
lower score.

This module needs no package of the `learn` dependency group.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.learn_tune_grid import TIE, GridError, rank
from fleetreserve.sessions import TOP

__all__ = ["GridError", "TIE", "rank"]

GRID = TOP / "learn" / "hazard-grid.json"
GRID_KEYS = ("about", "seeds", "epochs", "width", "learning_rate", "kl_weight")
CHOICE_RULE = (
    "lowest mean score; a tie within 1e-9 goes to the smaller width, then the fewer epochs, then "
    "the larger divergence weight, then the larger learning rate"
)


@dataclass(frozen=True)
class Setting:
    width: int
    learning_rate: float
    kl_weight: float
    epochs: int

    def as_dict(self) -> dict:
        return asdict(self)

    def fit_key(self) -> tuple:
        """What a fit depends on apart from the epoch checkpoint and the seed."""
        return (self.width, self.learning_rate, self.kl_weight)

    def hazard(self, seed: int = 0, **changes) -> HazardSettings:
        """The `HazardSettings` of these settings at `seed`, with `changes` to other fields."""
        return replace(
            HazardSettings(
                width=self.width,
                learning_rate=self.learning_rate,
                kl_weight=self.kl_weight,
                epochs=self.epochs,
                seed=seed,
            ),
            **changes,
        )

    def text(self) -> str:
        return (
            f"width {self.width}, learning rate {self.learning_rate:g}, divergence weight "
            f"{self.kl_weight:g}, {self.epochs} epochs"
        )


_DEFAULT = HazardSettings()
PRESENT = Setting(
    width=_DEFAULT.width,
    learning_rate=_DEFAULT.learning_rate,
    kl_weight=_DEFAULT.kl_weight,
    epochs=_DEFAULT.epochs,
)


@dataclass(frozen=True)
class Grid:
    candidates: tuple[Setting, ...]
    fits: tuple[tuple, ...]  # (width, learning rate, divergence weight), in the order of the file
    epochs: tuple[int, ...]  # the checkpoints, ascending
    seeds: tuple[int, ...]
    sha256: str
    file: str


def _whole(value, what: str, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise GridError(f"{what} must be a whole number of at least {least}, got {value!r}")
    return value


def _number(value, what: str, low: float, high: float | None, *, open_low: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise GridError(f"{what} must be a number, got {value!r}")
    outside = value <= low if open_low else value < low
    if outside or (high is not None and value > high):
        raise GridError(f"{what} is outside its range, got {value!r}")
    return float(value)


def settings_from(document) -> Setting:
    """The settings a JSON object with exactly the four keys holds; raises `GridError`."""
    keys = ("width", "learning_rate", "kl_weight", "epochs")
    if not isinstance(document, dict) or sorted(document) != sorted(keys):
        raise GridError(f"settings have exactly the keys {list(keys)}, got {document!r}")
    setting = Setting(
        _whole(document["width"], "width"),
        _number(document["learning_rate"], "learning_rate", 0.0, 1.0, open_low=True),
        _number(document["kl_weight"], "kl_weight", 0.0, None, open_low=False),
        _whole(document["epochs"], "epochs"),
    )
    try:
        setting.hazard()
    except ValueError as problem:
        raise GridError(f"settings cannot be used by the hazard model: {problem}") from problem
    return setting


def _same_value_key(value) -> tuple:
    """Equal for two values that are equal as numbers (1 and 1.0), otherwise for two values with
    the same JSON text. A boolean is not a number here, and a NaN is compared by its text."""
    if isinstance(value, int | float) and not isinstance(value, bool) and value == value:
        return ("number", value)
    return ("text", json.dumps(value))


def _list(document: dict, key: str) -> list:
    values = document[key]
    if not isinstance(values, list) or not values:
        raise GridError(f"{key} must be a non-empty list")
    if len({_same_value_key(v) for v in values}) != len(values):
        raise GridError(f"{key} names a value twice")
    return values


def load_grid(path: Path | None = None) -> Grid:
    """The grid of the file `path` (default `GRID`); raises `GridError` (module text)."""
    path = GRID if path is None else Path(path)
    try:
        content = path.read_bytes()
        document = json.loads(content)
    except (OSError, ValueError) as problem:
        raise GridError(f"the grid file {path.name} cannot be read: {problem}") from problem
    if not isinstance(document, dict) or sorted(document) != sorted(GRID_KEYS):
        raise GridError(f"the grid file {path.name} has exactly the keys {list(GRID_KEYS)}")
    seeds = tuple(_whole(v, "a seed", 0) for v in _list(document, "seeds"))
    epochs = tuple(sorted(_whole(v, "an epoch value") for v in _list(document, "epochs")))
    candidates = tuple(
        settings_from({"width": width, "learning_rate": rate, "kl_weight": weight, "epochs": epoch})
        for width in _list(document, "width")
        for rate in _list(document, "learning_rate")
        for weight in _list(document, "kl_weight")
        for epoch in epochs
    )
    fits = tuple(dict.fromkeys(c.fit_key() for c in candidates))
    if PRESENT not in candidates:
        raise GridError(
            f"the grid file {path.name} does not hold the present settings ({PRESENT.text()})"
        )
    return Grid(candidates, fits, epochs, seeds, hashlib.sha256(content).hexdigest(), path.name)


def _order(setting: Setting) -> tuple:
    return (setting.width, setting.epochs, -setting.kl_weight, -setting.learning_rate)


def choose(scored: Sequence[tuple[Setting, float]]) -> Setting:
    """The candidate the choice rule picks from (settings, score) pairs (module text)."""
    if not scored:
        raise ValueError("no candidate was scored")
    if any(not math.isfinite(score) for _, score in scored):
        raise ValueError("a candidate's score is not a finite number")
    lowest = min(score for _, score in scored)
    tied = [setting for setting, score in scored if score - lowest <= TIE]
    return min(tied, key=_order)
