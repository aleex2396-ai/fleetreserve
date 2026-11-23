"""The tuned settings file of the neural hazard model: written from the results of a recorded
tuning run, read by the tuned model, its seed ensemble and its four ablations
(`models.hazard_tuned`, `models.hazard_ens`, `models.hazard_tuned_no_set`, `_no_site`, `_no_day`,
`_calendar`; plan section 7; section 13, WP8).

Command line, from the project's top folder:
    python -m fleetreserve.hazard_tune_settings <run folder>/hazard-tune.json
        [--out learn/hazard-tuned.json] [--grid learn/hazard-grid.json]

It writes the settings file `TUNED` (`learn/hazard-tuned.json`) from the results
file of `fleetreserve.hazard_tune`, in the way `learn_tune_settings` writes `learn/tuned.json`:

- `settings`: the chosen candidate (`hazard_tune_grid.Setting`: width, learning rate, divergence
  weight, epochs) and `score`, its score;
- `grid`: the name and SHA-256 of the grid file the checks below use: the committed
  `learn/hazard-grid.json`, or the file given with `--grid` instead;
- `results`: the results file's name and SHA-256, its run folder's name and the commit read from
  that run's manifest.

It writes nothing, and says why, unless all of these hold:

- the results file is verified against its run's manifest (`run_verify.require_verified`: the
  run ended with exit code 0, did not hit its time limit, was not interrupted, lists the file
  with the hash it has now, and its commit exists) and lies directly in its run folder;
- the grid hash the results record equals the hash of that grid file (not of the committed one,
  when `--grid` names another);
- the results were fitted on every training unit (`truth_build.TRAIN_UNITS`), not on a part;
- the results score exactly the candidates of that grid file, and the choice rule
  (`hazard_tune_grid.choose`), applied again here to those scores, gives the candidate the results
  name as chosen.

`load_tuned()` is how a model reads the file: it returns the settings, or raises `SettingsError`
when the file is absent, unreadable or has other keys, when the grid hash it records differs
from the hash of the grid file it is checked against (the committed one unless the caller gives
`grid_path`), or when its settings are not a candidate of that grid.
`SettingsError` is a `gate_config.MatrixError`, so `learn_run` and `dial_choice` end with "could
not run" (exit 2) for a model that cannot read its settings. The file's path is read from `TUNED`
at each call.

The file is not edited by hand. None exists before a recorded run on the training sites has made
one.

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
from fleetreserve.hazard_tune_grid import GridError, Setting, choose, load_grid, settings_from
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import TRAIN_UNITS

TUNED = TOP / "learn" / "hazard-tuned.json"
TUNED_KEYS = ("about", "grid", "results", "score", "settings")
ABOUT = (
    "Settings of the tuned neural hazard departure model (models.hazard_tuned), written by "
    "fleetreserve.hazard_tune_settings from the results of a recorded fleetreserve.hazard_tune "
    "run on the training sites. Not edited by hand."
)


class SettingsError(MatrixError):
    """The tuned settings file cannot be used, or must not be written; the message says why."""


def load_tuned(path: Path | None = None, grid_path: Path | None = None) -> Setting:
    """The settings of the tuned settings file (default `TUNED`), checked against the grid file at
    `grid_path` (default `hazard_tune_grid.GRID`, the committed one); raises `SettingsError`
    (module text)."""
    path = TUNED if path is None else Path(path)
    if not path.is_file():
        raise SettingsError(
            f"the tuned settings file learn/{path.name} is absent: the tuned model is fitted only "
            "with settings written by fleetreserve.hazard_tune_settings from a recorded "
            "fleetreserve.hazard_tune run"
        )
    try:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or sorted(document) != sorted(TUNED_KEYS):
            raise ValueError(f"it has exactly the keys {list(TUNED_KEYS)}")
        recorded = document["grid"]["sha256"]
        setting = settings_from(document["settings"])
        grid = load_grid(grid_path)
    except (OSError, ValueError, KeyError, TypeError) as problem:
        raise SettingsError(
            f"the tuned settings file {path.name} cannot be used: {problem}"
        ) from problem
    if recorded != grid.sha256:
        raise SettingsError(
            f"the tuned settings file {path.name} was made with another grid: it records the "
            f"grid hash {str(recorded)[:12]}, the grid {grid.file} has {grid.sha256[:12]}"
        )
    if setting not in grid.candidates:
        raise SettingsError(
            f"the tuned settings file {path.name} holds settings that are not a candidate of "
            f"{grid.file}: {setting.text()}"
        )
    return setting


def settings_document(results_path: Path, grid_path: Path | None = None) -> dict:
    """The content of the settings file for a results file (module text). Raises `NotVerified`
    (the run is not verified), `GridError` (the grid file cannot be read) or `SettingsError` (the
    results do not fit the grid file at `grid_path` (default the committed one), or are not
    consistent)."""
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
            f"{str(recorded)[:12]}, the grid {grid.file} has {grid.sha256[:12]}"
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
    parser.add_argument("results", type=Path)
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
