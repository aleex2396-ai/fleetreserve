"""The candidate settings of the pooled LightGBM departure model and the rule that picks one
(plan section 7, pooled LightGBM; section 13, WP8).

Settings (`Settings`): `num_leaves`, `min_data_in_leaf`, `learning_rate` and `rounds` (the number
of boosting rounds). Every other LightGBM parameter stays as `models.lgbm.PARAMETERS` has it.
`PRESENT` holds the settings `models.lgbm` is fitted with (a test compares the two).

Grid file (`GRID`, `learn/tune-grid.json`): the lists `num_leaves` and `min_data_in_leaf` and the
list `learning_rate_and_rounds` of (learning rate, rounds) pairs. The candidates are every
combination, in the order of the file (leaves, then rows in a leaf, then the pair). `load_grid`
refuses (`GridError`) a file with other keys, an empty or repeated list entry, a value that is not
a positive number, or a grid that does not hold `PRESENT`: the present settings are always a
candidate, so a result always says how the present model ranks. The grid's hash is the SHA-256 of
the file's bytes.

Choice rule (`choose`), fixed before any run on training data: the candidate with the lowest
score. Candidates whose score is within `TIE` (1e-9) of the lowest are tied, and the tie goes to
the fewer leaves, then the larger `min_data_in_leaf`, then the fewer rounds (then, for a grid
that still leaves two, the smaller learning rate). Nothing else enters: no one-standard-error
rule and no judgement. `rank` of a candidate: one more than the number of candidates with a
strictly lower score.

This module needs no package of the `learn` dependency group.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from fleetreserve.sessions import TOP

GRID = TOP / "learn" / "tune-grid.json"
GRID_KEYS = ("about", "num_leaves", "min_data_in_leaf", "learning_rate_and_rounds")
TIE = 1e-9


class GridError(ValueError):
    """The grid file cannot be used; the message says why."""


@dataclass(frozen=True)
class Settings:
    num_leaves: int
    min_data_in_leaf: int
    learning_rate: float
    rounds: int

    def overrides(self) -> dict:
        """The LightGBM parameters these settings replace (the rounds are not a parameter)."""
        return {
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
        }

    def as_dict(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        return (
            f"{self.num_leaves} leaves, {self.min_data_in_leaf} rows in a leaf, learning rate "
            f"{self.learning_rate:g}, {self.rounds} rounds"
        )


PRESENT = Settings(num_leaves=15, min_data_in_leaf=20, learning_rate=0.05, rounds=200)


@dataclass(frozen=True)
class Grid:
    candidates: tuple[Settings, ...]
    sha256: str
    file: str


def _whole(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GridError(f"{what} must be a whole number of at least 1, got {value!r}")
    return value


def settings_from(document) -> Settings:
    """The settings a JSON object with exactly the four keys holds; raises `GridError`."""
    keys = ("num_leaves", "min_data_in_leaf", "learning_rate", "rounds")
    if not isinstance(document, dict) or sorted(document) != sorted(keys):
        raise GridError(f"settings have exactly the keys {list(keys)}, got {document!r}")
    rate = document["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, int | float) or not 0 < rate <= 1:
        raise GridError(f"learning_rate must be a number in (0, 1], got {rate!r}")
    return Settings(
        _whole(document["num_leaves"], "num_leaves"),
        _whole(document["min_data_in_leaf"], "min_data_in_leaf"),
        float(rate),
        _whole(document["rounds"], "rounds"),
    )


def _list(document: dict, key: str) -> list:
    values = document[key]
    if not isinstance(values, list) or not values:
        raise GridError(f"{key} must be a non-empty list")
    if len({json.dumps(v) for v in values}) != len(values):
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
    pairs = _list(document, "learning_rate_and_rounds")
    if any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs):
        raise GridError("learning_rate_and_rounds holds pairs of a learning rate and rounds")
    candidates = tuple(
        settings_from(
            {
                "num_leaves": leaves,
                "min_data_in_leaf": rows,
                "learning_rate": rate,
                "rounds": rounds,
            }
        )
        for leaves in _list(document, "num_leaves")
        for rows in _list(document, "min_data_in_leaf")
        for rate, rounds in pairs
    )
    if PRESENT not in candidates:
        raise GridError(
            f"the grid file {path.name} does not hold the present settings ({PRESENT.text()})"
        )
    return Grid(candidates, hashlib.sha256(content).hexdigest(), path.name)


def _order(settings: Settings) -> tuple:
    return (
        settings.num_leaves,
        -settings.min_data_in_leaf,
        settings.rounds,
        settings.learning_rate,
    )


def choose(scored: Sequence[tuple[Settings, float]]) -> Settings:
    """The candidate the choice rule picks from (settings, score) pairs (module text)."""
    if not scored:
        raise ValueError("no candidate was scored")
    if any(not math.isfinite(score) for _, score in scored):
        raise ValueError("a candidate's score is not a finite number")
    lowest = min(score for _, score in scored)
    tied = [settings for settings, score in scored if score - lowest <= TIE]
    return min(tied, key=_order)


def rank(scored: Sequence[tuple[Settings, float]], settings: Settings) -> int:
    """One more than the number of candidates with a strictly lower score than `settings`."""
    own = dict(scored)[settings]
    return 1 + sum(1 for _, score in scored if score < own)
