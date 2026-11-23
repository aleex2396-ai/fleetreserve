"""The settings file of JPL's 9/80 Fridays off: written from the results of a recorded
`fleetreserve.nine_eighty_run`, read by the models that take the column `nine_eighty_friday`
(`model_inputs`; plan section 7).

Command line, from the project's top folder:
    python -m fleetreserve.nine_eighty_settings <run folder>/nine-eighty-phase.json
        [--out learn/nine-eighty-phase.json]

It writes `PHASE` (`learn/nine-eighty-phase.json`) from the results file of the run, as
`hazard_tune_settings` writes the tuned settings: `parity` (0 or 1, the fortnight parity of the
Fridays off, `calendar_inputs.fortnight_parity`), the counts and means it was chosen from, and
`results`: the results file's name and SHA-256, its run folder's name and the commit read from
that run's manifest. It writes nothing, and says why, unless all of these hold:

- the results file is verified against its run's manifest (`run_verify.require_verified`: exit
  code 0, no time limit hit, not interrupted, the file listed with the hash it has now, its
  commit exists) and lies directly in its run folder;
- it was estimated on the unit `jpl` (a training unit, `truth_build.TRAIN_UNITS`);
- each parity has at least one counting Friday, the two means differ, and the parity the results
  name is the one with the fewer mean cars (`calendar_inputs.nine_eighty_phase`'s rule, applied
  again to the recorded means).

`load_parity()` is how a model reads the file: the parity, or `PhaseError` (a
`gate_config.MatrixError`, so a command loading the model ends with "could not run", exit 2) when
the file is absent, unreadable, has other keys or a parity other than 0 or 1. The file is not
edited by hand. None exists before a recorded run on JPL's training days has made one.

Exit status of the command: 0 the file was written; 1 it was refused; 2 could not run (the results
file or its manifest is missing or unreadable).
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from fleetreserve.gate_config import MatrixError
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified
from fleetreserve.sessions import TOP

PHASE = TOP / "learn" / "nine-eighty-phase.json"
RESULTS_NAME = "nine-eighty-phase.json"
UNIT = "jpl"
ESTIMATE_KEYS = ("parity", "fridays_0", "fridays_1", "mean_cars_0", "mean_cars_1")
PHASE_KEYS = ("about", "estimate", "parity", "results")
ABOUT = (
    "The fortnight parity of JPL's 9/80 Fridays off (calendar_inputs.fortnight_parity), written "
    "by fleetreserve.nine_eighty_settings from the results of a recorded "
    "fleetreserve.nine_eighty_run on JPL's training days. Not edited by hand."
)


class PhaseError(MatrixError):
    """The settings file cannot be used, or must not be written; the message says why."""


def load_parity(path: Path | None = None) -> int:
    """The parity of the settings file (default `PHASE`); raises `PhaseError` (module text)."""
    path = PHASE if path is None else Path(path)
    if not path.is_file():
        raise PhaseError(
            f"the settings file learn/{path.name} is absent: the parity of JPL's Fridays off is "
            "written only by fleetreserve.nine_eighty_settings from a recorded "
            "fleetreserve.nine_eighty_run"
        )
    try:
        document = json.loads(path.read_text())
        if not isinstance(document, dict) or sorted(document) != sorted(PHASE_KEYS):
            raise ValueError(f"it has exactly the keys {list(PHASE_KEYS)}")
        parity = document["parity"]
    except (OSError, ValueError, KeyError, TypeError) as problem:
        raise PhaseError(f"the settings file {path.name} cannot be used: {problem}") from problem
    if type(parity) is not int or parity not in (0, 1):
        raise PhaseError(f"the settings file {path.name} has the parity {parity!r}, not 0 or 1")
    return parity


def check_estimate(estimate: dict) -> int:
    """The parity of a recorded estimate, checked by the rule of the module text; raises
    `PhaseError`."""
    try:
        counts = [int(estimate[f"fridays_{p}"]) for p in (0, 1)]
        means = [float(estimate[f"mean_cars_{p}"]) for p in (0, 1)]
        named = estimate["parity"]
    except (KeyError, TypeError, ValueError) as problem:
        raise PhaseError(f"the estimate is not complete: {problem}") from problem
    if min(counts) < 1 or not all(math.isfinite(m) for m in means):
        raise PhaseError(f"a parity has no counting Friday (counts {counts})")
    if means[0] == means[1]:
        raise PhaseError(f"the two parities have the same mean ({means[0]})")
    chosen = 0 if means[0] < means[1] else 1
    if named != chosen:
        raise PhaseError(f"the results name the parity {named!r}, the rule gives {chosen}")
    return chosen


def phase_document(results_path: Path) -> dict:
    """The content of the settings file for a results file (module text). Raises `NotVerified`
    or `PhaseError`."""
    file = Path(os.path.abspath(results_path))
    manifest = require_verified(file)
    if not (file.parent / MANIFEST_NAME).is_file():
        raise PhaseError(f"{file.name} does not lie directly in its run folder")
    content = file.read_bytes()
    try:
        results = json.loads(content)
        unit = results["unit"]
        estimate = {key: results["estimate"][key] for key in ESTIMATE_KEYS}
    except (ValueError, KeyError, TypeError) as problem:
        raise PhaseError(f"{file.name} is not a 9/80 results file: {problem}") from problem
    if unit != UNIT:
        raise PhaseError(f"{file.name} was estimated on {unit!r}, not on {UNIT}")
    parity = check_estimate(estimate)
    return {
        "about": ABOUT,
        "parity": parity,
        "estimate": estimate,
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
    args = parser.parse_args(argv)
    out = PHASE if args.out is None else args.out
    try:
        document = phase_document(args.results)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except (NotVerified, PhaseError) as problem:
        print(f"REFUSED, nothing written: {problem}")
        return 1
    out.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(
        f"Wrote {out.name}: parity {document['parity']}, from "
        f"{document['results']['run_folder']} at commit {document['results']['commit'][:7]}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
