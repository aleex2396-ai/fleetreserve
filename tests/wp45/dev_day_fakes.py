"""The development-day checks command on a small made-up build (`ceiling_fakes`, scaled down).

`run_command` runs `dev_day_checks.main` on a fresh build in a given folder and returns the exit
status and the run folder. `reference_document` runs it once per test session (workers 1) and
keeps the results file's document, so the tool tests read what the command really writes.
"""

import functools
import json
import tempfile
from pathlib import Path

import pytest
from ceiling_fakes import setup_build

from fleetreserve import dev_day_checks

SCALE = 0.2
N_DAYS = 15  # counting days at that scale: 4, 5 and 3 + 3
SAMPLE = 5
CONFIG = [
    "--energy-bounds",
    "known",
    "--uncertifiable-car",
    "leave-unsteered",
    "--kappa",
    "0.1",
    "--departure-floor",
    "as-planned",
]


def run_command(
    root: Path,
    patch,
    *,
    workers: int = 1,
    sample: int = SAMPLE,
    config=None,
    layout=None,
    units=None,
):
    """(exit status, run folder) of the command on a fresh made-up build in the new folder
    `root`: the three default units, or the units of `layout` (`ceiling_fakes.setup_build`);
    `units`, when given, is passed as `--units`."""
    root.mkdir(parents=True)
    run, tables = setup_build(root, patch, scale=SCALE, layout=layout)
    out = root / "out"
    out.mkdir()
    patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    code = dev_day_checks.main(
        [
            "--run",
            str(run),
            "--tables",
            str(tables),
            "--t2-sample-size",
            str(sample),
            *(CONFIG if config is None else config),
            "--workers",
            str(workers),
            *([] if units is None else ["--units", *units]),
        ]
    )
    return code, out


_KEEP = []


@functools.cache
def reference_document() -> str:
    """The results file (as text) of one run with workers 1, sample `SAMPLE`."""
    folder = tempfile.TemporaryDirectory()
    _KEEP.append(folder)
    with pytest.MonkeyPatch.context() as patch:
        code, out = run_command(Path(folder.name) / "build", patch)
    assert code == 0
    text = (out / dev_day_checks.RESULTS_NAME).read_text()
    json.loads(text)
    return text
