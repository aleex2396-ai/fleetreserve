"""Extra input columns of a departure model: the plan inputs of `hazard_inputs` (and, for a
labelled look-ahead what-if only, `weather_lookahead`) given to a model that declares them
(plan section 7; WP8, phase two).

A model declares extra inputs with the attribute `extra_inputs`, a tuple of column names, each of
`COLUMNS`, and holds what they are read from as `context_sources` (`hazard_inputs.ContextSources`,
usually from `load_context` in its `make_model`). A model without the attribute has none, and
nothing here touches its frames: every existing model reads the same frames as before.

Where the columns are added (`ExtraInputs.add`, which appends them to a `build_table` table built
from a cars table, row by row, via `hazard_inputs.with_inputs` and `weather_lookahead`):

- the fit table: in the model's own `fit`, which builds its table from the cars tables it is
  given, as every learned model does (`models.hazard_tuned_inputs`);
- the predict frame: in `learn_run.forecast_frame`, before `learn_features.feature_frame`, for the
  `ExtraInputs` that `extra_of(model)` gives. `learn_run.evaluate` and `dial_choice.make_factory`
  pass it, and every other command that fits and predicts a model goes through `make_factory`
  (`compare_run`, `size_whatif`, `t7_test`, `asa_diagnosis_run`, `corner_run`,
  `resolve_diagnosis`, `fleet_size_whatif`).

The look-ahead fence. A column of `LOOK_AHEAD_COLUMNS` holds the weather of the service hours,
after τ: not known at τ. `extra_of(model, look_ahead_allowed)` refuses (`LookAheadRefused`, a
`ValueError`) a model that declares one unless the caller allows it; `ExtraInputs.add` refuses
the columns unless its `look_ahead` is true; `learn_run.check_model` refuses any model that reads
them (`reads_look_ahead`) or whose module is in `forecast_model.LOOK_AHEAD_MODELS`, unless its
caller allows look-ahead, and a model that reads them must be in that list. Only
`learn_run.main`, the development evaluation, allows it; every command that chooses or uses a
model for an offer (`dial_choice` and every command through `make_factory`, the held-out run)
does not, so there the model is refused. The look-ahead model also refuses in its own `predict`
any unit that is not a development unit (`models.hazard_tuned_inputs_lookahead`).

Labels: `record(model)` (the columns, the 9/80 parity and the SHA-256 of every context file
read) goes into `learn_run`'s results under `context_inputs`; for a look-ahead model the key
`look_ahead` holds `LOOK_AHEAD_WHAT` and the columns, and `summary_head` gives the first lines of
the summary. A model without extra inputs gets neither key.

Encoding (`InputsScaler`): the inputs of `hazard_data.Scaler` unchanged, then per extra column
its value standardised with the mean and standard deviation of the fit rows' finite values (a
standard deviation at most 1e-9 keeps scale 1), clipped to plus or minus `input_clip`, 0 where
the value is missing, then per extra column a 0/1 input saying it is missing. Every extra column
may be missing (school terms before a source's publication, weather outside the files, Korea's
school terms); an infinite value is an error.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fleetreserve import hazard_inputs, school_inputs, weather_lookahead
from fleetreserve.context_places import SITE_PLACES, WEATHER_SET, weather_file
from fleetreserve.gate_config import MatrixError
from fleetreserve.hazard_data import N_INPUTS, Scaler
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.nine_eighty_settings import PHASE, load_parity
from fleetreserve.sessions import TOP

LOOK_AHEAD_COLUMNS = weather_lookahead.COLUMNS
COLUMNS = (*hazard_inputs.NEW_COLUMNS, *LOOK_AHEAD_COLUMNS)
CONTEXT_KEY = "context_inputs"
LOOK_AHEAD_KEY = "look_ahead"
LOOK_AHEAD_WHAT = (
    "LOOK-AHEAD WHAT-IF, NOT A MODEL OF THE PLAN: the model reads the weather observed in the "
    "service hours of the day itself, which is not known at the decision time; its figures are "
    "an upper bound on what a weather forecast could add, on development units only"
)


class ContextError(MatrixError):
    """A context file a model needs is missing or cannot be read: the run could not run."""


class LookAheadRefused(ValueError):
    """A look-ahead column asked for where it is not allowed (module text)."""


def declared(model) -> tuple[str, ...]:
    """The model's extra input columns (module text); `ValueError` for a name not in `COLUMNS`
    or named twice."""
    names = tuple(getattr(model, "extra_inputs", ()))
    unknown = [n for n in names if n not in COLUMNS]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"extra inputs not known or named twice: {list(names)}")
    return names


def reads_look_ahead(model) -> bool:
    """Whether the model declares a look-ahead column or the attribute `look_ahead`."""
    return getattr(model, "look_ahead", False) is True or bool(
        set(declared(model)) & set(LOOK_AHEAD_COLUMNS)
    )


@dataclass(frozen=True)
class ExtraInputs:
    """The extra columns `names`, read from `sources`; `look_ahead` allows the look-ahead ones."""

    names: tuple[str, ...]
    sources: hazard_inputs.ContextSources = field(repr=False)
    look_ahead: bool = False

    def add(self, table: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
        """`table` (a `build_table` table of `cars`, row by row) with the columns `names`
        appended in that order (module text)."""
        ahead = [n for n in self.names if n in LOOK_AHEAD_COLUMNS]
        if ahead and not self.look_ahead:
            raise LookAheadRefused(f"look-ahead columns are not allowed here: {ahead}")
        plain = [n for n in self.names if n not in LOOK_AHEAD_COLUMNS]
        out = hazard_inputs.with_inputs(table, cars, plain, self.sources)
        if ahead:
            if self.sources.weather is None:
                raise ValueError("the look-ahead columns need the weather files")
            aligned = hazard_inputs.aligned(table, cars)
            found = weather_lookahead.lookahead_columns(aligned, self.sources.weather)
            out = pd.concat([out, found[ahead]], axis=1)
        return out[[*table.columns, *self.names]]


def extra_of(model, look_ahead_allowed: bool = False) -> ExtraInputs | None:
    """The `ExtraInputs` of a model, or None when it declares none (module text)."""
    names = declared(model)
    if not names:
        return None
    if reads_look_ahead(model) and not look_ahead_allowed:
        raise LookAheadRefused(
            f"model {getattr(model, 'name', '?')} reads the look-ahead columns, which only the "
            "development evaluation of learn_run allows"
        )
    sources = getattr(model, "context_sources", None)
    if not isinstance(sources, hazard_inputs.ContextSources):
        raise ValueError("a model with extra inputs holds its context_sources")
    return ExtraInputs(names, sources, look_ahead_allowed)


def _values(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    return frame[list(names)].astype(float).to_numpy(copy=True)


@dataclass(frozen=True)
class InputsScaler:
    """`hazard_data.Scaler` plus the extra columns `names` (module text)."""

    base: Scaler
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    clip: float

    @classmethod
    def fit(cls, table: pd.DataFrame, settings: HazardSettings, names) -> "InputsScaler":
        values = _values(table, names)
        mean = np.zeros(values.shape[1])
        std = np.ones(values.shape[1])
        for place in range(values.shape[1]):
            seen = values[:, place][np.isfinite(values[:, place])]
            if seen.size:
                mean[place] = seen.mean()
                spread = seen.std()
                std[place] = spread if spread > 1e-9 else 1.0
        return cls(Scaler.fit(table, settings), tuple(names), mean, std, settings.input_clip)

    @property
    def n_inputs(self) -> int:
        return N_INPUTS + 2 * len(self.names)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """The inputs of every row of `frame`, float32, shape (rows, `n_inputs`)."""
        first = self.base.transform(frame)
        values = _values(frame, self.names)
        if np.isinf(values).any():
            raise ValueError("an extra input value is infinite")
        absent = np.isnan(values)
        scaled = np.clip((values - self.mean) / self.std, -self.clip, self.clip)
        scaled = np.where(absent, 0.0, scaled)
        extra = np.column_stack([scaled, absent.astype(float)]).astype(np.float32)
        return np.concatenate([first, extra], axis=1)


@dataclass(frozen=True)
class Context:
    """What `load_context` read: the sources and the SHA-256 of each file, by path from `root`."""

    sources: hazard_inputs.ContextSources = field(repr=False)
    files: dict


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context(root: Path = TOP, phase: Path | None = None) -> Context:
    """The context files of every site (`hazard_inputs.load_sources` under `root`), the committed
    school terms and the 9/80 parity (`nine_eighty_settings.load_parity`); raises
    `ContextError` or `nine_eighty_settings.PhaseError`, both `MatrixError`s."""
    phase = PHASE if phase is None else Path(phase)
    parity = load_parity(phase)
    sites = sorted(SITE_PLACES)
    places = sorted({place for site in sites for place in SITE_PLACES[site]})
    root = Path(root)
    paths = [root / "data" / "raw" / WEATHER_SET / weather_file(p) for p in places]
    paths += [root / hazard_inputs.OXCGRT_FILE]
    try:
        sources = hazard_inputs.load_sources(root, sites, parity)
        files = {p.relative_to(root).as_posix(): _digest(p) for p in paths}
    except (OSError, ValueError) as problem:
        raise ContextError(f"the context files under data/raw cannot be read: {problem}") from None
    files[f"learn/{school_inputs.TERMS_FILE.name}"] = _digest(school_inputs.TERMS_FILE)
    files[f"learn/{phase.name}"] = _digest(phase)
    return Context(sources, files)


def record(model) -> dict | None:
    """What `learn_run` records of a model's extra inputs, or None when it has none."""
    names = declared(model)
    if not names:
        return None
    sources = getattr(model, "context_sources", None)
    return {
        "columns": list(names),
        "look_ahead_columns": [n for n in names if n in LOOK_AHEAD_COLUMNS],
        "nine_eighty_parity": getattr(sources, "nine_eighty_parity", None),
        "files_sha256": dict(sorted(getattr(model, "context_files", {}).items())),
    }


def mark(document: dict, model) -> None:
    """Add `context_inputs` and, for a look-ahead model, `look_ahead` to a results document."""
    found = record(model)
    if found is None:
        return
    document[CONTEXT_KEY] = found
    if reads_look_ahead(model):
        document[LOOK_AHEAD_KEY] = {"what": LOOK_AHEAD_WHAT, "columns": found["look_ahead_columns"]}


def summary_head(document: dict) -> list[str]:
    """The first lines of a summary of a results document: the look-ahead label and its columns,
    or nothing for a document without the key `look_ahead`."""
    found = document.get(LOOK_AHEAD_KEY)
    if found is None:
        return []
    return [found["what"], "Look-ahead columns: " + ", ".join(found["columns"])]
