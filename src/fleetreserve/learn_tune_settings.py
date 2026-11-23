"""The tuned settings file of the pooled LightGBM departure model: written from the results of a
recorded tuning run, read by the tuned models (plan section 7; section 13, WP8).

Command line, from the project's top folder:
    python -m fleetreserve.learn_tune_settings --results <run folder>/tune.json
        [--out learn/tuned.json] [--grid learn/tune-grid.json]

It writes the settings file `TUNED` (`learn/tuned.json`, a tracked file) from the results file of
`fleetreserve.learn_tune`:

- `settings`: the chosen candidate (`learn_tune_grid.Settings`) and `score`, its score;
- `grid`: the committed grid file's name and SHA-256;
- `results`: the results file's name and SHA-256, its run folder's name and the commit read from
  that run's manifest.

It writes nothing, and says why, unless all of these hold:

- the results file is verified against its run's manifest (`run_verify.require_verified`: the
  run ended with exit code 0, did not hit its time limit, was not interrupted, lists the file
  with the hash it has now, and its commit exists) and lies directly in its run folder;
- the grid hash the results record equals the hash of the committed grid file;
- the results were fitted on every training unit (`truth_build.TRAIN_UNITS`), not on a part;
- the results score exactly the candidates of the committed grid, and the choice rule
  (`learn_tune_grid.choose`), applied again here to those scores, gives the candidate the results
  name as chosen.

`load_tuned()` is how a model reads the file: it returns the settings, or raises `SettingsError`
when the file is absent or unreadable, when the grid hash it records differs from the committed
grid's, or when its settings are not a candidate of that grid. `SettingsError` is a
`gate_config.MatrixError`, so `learn_run` and `dial_choice` end with "could not run" (exit 2)
for a model that cannot read its settings. The file's path is read from `TUNED` at each call.

Exit status of the command: 0 the file was written; 1 it was refused (a condition above failed);
2 could not run (the results file, its manifest or the grid file is missing or unreadable).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from fleetreserve.gate_config import MatrixError
from fleetreserve.learn_tune_grid import GridError, Settings, choose, load_grid, settings_from
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import TRAIN_UNITS

TUNED = TOP / "learn" / "tuned.json"
TUNED_KEYS = ("about", "grid", "results", "score", "settings")
ABOUT = (
    "Settings of the tuned pooled LightGBM departure models (models.lgbm_tuned, "
    "models.lgbm_tuned_surv), written by fleetreserve.learn_tune_settings from the results of a "
    "recorded fleetreserve.learn_tune run on the training sites. Not edited by hand."
)


class SettingsError(MatrixError):
    """The tuned settings file cannot be used, or must not be written; the message says why."""


def load_tuned(path: Path | None = None, grid_path: Path | None = None) -> Settings:
    """The settings of the tuned settings file (default `TUNED`), checked against the committed
    grid (default `learn_tune_grid.GRID`); raises `SettingsError` (module text)."""
    path = TUNED if path is None else Path(path)
    if not path.is_file():
        raise SettingsError(
            f"the tuned settings file learn/{path.name} is absent: a tuned model is fitted only "
            "with settings written by fleetreserve.learn_tune_settings from a recorded "
            "fleetreserve.learn_tune run"
        )
    try:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or sorted(document) != sorted(TUNED_KEYS):
            raise ValueError(f"it has exactly the keys {list(TUNED_KEYS)}")
        recorded = document["grid"]["sha256"]
        settings = settings_from(document["settings"])
        grid = load_grid(grid_path)
    except (OSError, ValueError, KeyError, TypeError) as problem:
        raise SettingsError(
            f"the tuned settings file {path.name} cannot be used: {problem}"
        ) from problem
    if recorded != grid.sha256:
        raise SettingsError(
            f"the tuned settings file {path.name} was made with another grid: it records the "
            f"grid hash {str(recorded)[:12]}, the committed {grid.file} has {grid.sha256[:12]}"
        )
    if settings not in grid.candidates:
        raise SettingsError(
            f"the tuned settings file {path.name} holds settings that are not a candidate of "
            f"{grid.file}: {settings.text()}"
        )
    return settings


def settings_document(results_path: Path, grid_path: Path | None = None) -> dict:
    """The content of the settings file for a results file (module text). Raises `NotVerified`
    (the run is not verified), `GridError` (the grid file cannot be read) or `SettingsError` (the
    results do not fit the committed grid, or are not consistent)."""
    file = Path(os.path.abspath(results_path))
    manifest = require_verified(file)
    if not (file.parent / MANIFEST_NAME).is_file():
        raise SettingsError(f"{file.name} does not lie directly in its run folder")
    grid = load_grid(grid_path)
    content = file.read_bytes()
    try:
        results = json.loads(content)
        recorded = results["grid"]["sha256"]
        scored = [(settings_from(c["settings"]), float(c["score"])) for c in results["candidates"]]
        named = settings_from(results["chosen"]["settings"])
        fit_units = list(results["fit_units"])
    except (ValueError, KeyError, TypeError) as problem:
        raise SettingsError(f"{file.name} is not a tuning results file: {problem}") from problem
    if recorded != grid.sha256:
        raise SettingsError(
            f"{file.name} was made with another grid: it records the grid hash "
            f"{str(recorded)[:12]}, the committed {grid.file} has {grid.sha256[:12]}"
        )
    if sorted(fit_units) != sorted(TRAIN_UNITS):
        raise SettingsError(
            f"{file.name} was fitted on {fit_units}, not on every training unit "
            f"({', '.join(TRAIN_UNITS)})"
        )
    if len(scored) != len(grid.candidates) or {s for s, _ in scored} != set(grid.candidates):
        raise SettingsError(f"{file.name} does not score exactly the candidates of {grid.file}")
    chosen = choose(scored)
    if chosen != named:
        raise SettingsError(
            f"{file.name} names {named.text()} as chosen, the choice rule gives {chosen.text()}"
        )
    return {
        "about": ABOUT,
        "settings": chosen.as_dict(),
        "score": dict(scored)[chosen],
        "grid": {"file": grid.file, "sha256": grid.sha256},
        "results": {
            "file": file.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "run_folder": file.parent.name,
            "commit": manifest["commit"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None)
    args = parser.parse_args(argv)
    out = TUNED if args.out is None else args.out
    try:
        document = settings_document(args.results, args.grid)
    except (CannotCheck, GridError) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except (NotVerified, SettingsError) as problem:
        print(f"REFUSED, nothing written: {problem}")
        return 1
    out.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    chosen = settings_from(document["settings"])
    print(
        f"Wrote {out.name}: {chosen.text()} (score {document['score']:.6f}), from "
        f"{document['results']['run_folder']} at commit {document['results']['commit'][:7]}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
