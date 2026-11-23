"""A made-up held-setting truth build made from a made-up main one (`gate_fake.write_build`).

The made-up tables are the same bytes in both settings, so a test that mixes the two settings
fails only on the setting checks, never on the table hashes. Real held tables differ from main
ones in their energy-before-τ and room columns.
"""

import hashlib
import json
import shutil
from pathlib import Path

from fleetreserve import truth, truth_build

HELD_RUN_NAME = "20260102T000000Z-wp2-truth-held"


def held_copy(main_run: Path, main_tables: Path, folder: Path) -> tuple[Path, Path]:
    """(run, tables) of a held build that has the tables of the given main build: a copy of the
    tables folder with the held note, and a recorded run named like a held truth build whose
    summary carries the key `setting`."""
    tables = folder / "tables-held"
    shutil.copytree(main_tables, tables)
    (tables / truth_build.SETTING_FILE).write_text(truth.HELD + "\n")
    run = folder / HELD_RUN_NAME
    run.mkdir()
    summary = json.loads((main_run / "truth-summary.json").read_text())
    summary["setting"] = truth.HELD
    path = run / "truth-summary.json"
    path.write_text(json.dumps(summary))
    manifest = json.loads((main_run / "manifest.json").read_text())
    manifest["outputs"] = {"truth-summary.json": hashlib.sha256(path.read_bytes()).hexdigest()}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run, tables
