"""Re-run the earlier analysis's published code, unchanged, and record the three numbers (WP3).

Run as `python tools/earlier_rerun.py [--workers N]` with `FLEETRESERVE_RUN_DIR` set, in the way
the run recorder does (`python -m fleetreserve.run_record --name wp3-rerun --long
--max-seconds 21600 -- ...`). `--check-only` does the checks of step 1 and nothing else, needs no
run folder and writes nothing.

Steps, all inside the run folder:
1. Check that the earlier package (`reference/part1`) and its own interpreter exist, that the
   package is the stated export (the text of `PROVENANCE.txt` and the SHA-256 of the named
   source files below), that its modules import, and that the Caltech month files from
   `FIRST_MONTH` to `LAST_MONTH` are present in `data/raw/acn/caltech`.
2. Make a data root of links to exactly those month files, so the package cannot see a later
   month, and a scratch git repository (see DISCLOSURE).
3. Run the package's own stage `evaluate evaluation` with its own interpreter, nothing in it
   changed, its output going straight to this program's output.
4. Read the three numbers by calling the package's own `levels.same_days` on the stage's tables
   with the package's interpreter (`DRIVER`), as the package's report step does. No logic of the
   package is copied here. Check that its files are still the pinned ones.
5. Write `earlier-rerun.json` in the run folder: the three numbers at full precision, the row
   filters, the scored days, the times, the workers, the package's provenance text, the
   disclosure and the published values (labelled as published). Whether the numbers match the
   published ones is decided by `tools/earlier_rerun_check.py`, not here.

The package writes no bytecode (PYTHONDONTWRITEBYTECODE) and is never changed. The stage reads no
access token and makes no network call; the token variables are removed from its environment.

Exit status: 0 the stage ran and the three numbers were read (matching or not); 1 the stage or
the reading failed, or a pinned file changed during the run; 2 it could not run (no run folder,
package or data missing or not the stated export).

The standard library only: the stage and the reading driver run under the package's interpreter.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

RUN_DIR_VARIABLE = "FLEETRESERVE_RUN_DIR"
RESULTS_NAME = "earlier-rerun.json"
PACKAGE_PARTS = ("reference", "part1")
INTERPRETER_PARTS = (".venv", "bin", "python")
SOURCE_PARTS = ("src",)
MODULE_FOLDER = ("src", "fleetoffer")
MONTH_PARTS = ("data", "raw", "acn", "caltech")
# The stage opens every stored Caltech month before March 2020; the first stored one is April 2018.
FIRST_MONTH = "2018-04"
LAST_MONTH = "2020-02"
PLAN_TAG = "analysis-plan"
STAGE_ARGUMENTS = ("evaluate", "evaluation")
DEFAULT_WORKERS = 6

# What the earlier package's export must say about itself, and the source files whose bytes decide
# the numbers (SHA-256 of the files exported at the stated commit).
PROVENANCE_PHRASES = ("exported unchanged", "commit 05a6cf9")
PINNED_FILES = {
    "pipeline.py": "d9f232dc74dff0182997f2be8e1b6123305b7c04b920242e82a30d2cae07b7ef",
    "data.py": "d6a431ab57f749772d57528726dccb9d2403845eb4f75d2a36e1fb916954a470",
    "evaluation.py": "dafe7d6638dc1d85e3e663f8e1c0a8eae8f9ac013a30508260815455c3e05dcb",
    "levels.py": "b1ff73d96557d2b1e64ac692df10c32b489f9b1ecec91e7069704e356c083dce",
    "offer.py": "89086786ad4f08ad57b7f2e79568b1c71bf1dd046ae36ff74ac6995e97feff45",
    "percar.py": "5eecca8ac48ae6ba5264828b0f0b57d7444f0b20762b9660998c074fa8b6a21a",
    "options.py": "c4307ee209181002fa23db0fb877d925e5c029b5592080a31700d47b5ddfb1ea",
    "h0.json": "a9099f0441e1a4c2e608e062e52172ac5cc798502159ceae944dd2c2bc96b47d",
}
IMPORT_CHECK = (
    "import fleetoffer.pipeline, fleetoffer.levels, fleetoffer.evaluation, fleetoffer.data"
)

# The values the earlier analysis published (plan section 12) and the pass mark's limits (plan
# section 13, WP3). Shares are fractions: 0.004 is 0.4%.
PUBLISHED = {"perfect_knowledge_kwh": 45.6, "learned_share": 0.004, "population_share": 0.039}
TOLERANCE_KWH = 0.1
TOLERANCE_POINTS = 0.1

FORM = "average"
ROW_FILTERS = {
    "perfect_knowledge_kwh": {
        "form": FORM,
        "step": "perfect knowledge",
        "column": "size_mean_kwh",
    },
    "learned_share": {
        "form": FORM,
        "step": "day-level rule as run",
        "who": "4d",
        "column": "share_of_perfect",
    },
    "population_share": {
        "form": FORM,
        "step": "day-level rule as run",
        "who": "5d",
        "column": "share_of_perfect",
    },
}

DISCLOSURE = (
    f"The package refuses to run its evaluation without a git tag named {PLAN_TAG}, and the "
    "export of the earlier analysis has no git history, so that check was satisfied by a scratch "
    "git repository made inside the run folder that holds one empty commit with that tag. No "
    "file of the earlier analysis was changed."
)

# Run by the package's interpreter, from the package folder, with the stage's output folder as
# its one argument. It reads the stage's tables with the package's own reader and applies the
# package's own `levels.same_days`, exactly as the package's report step does, then prints every
# row of the day-average form as one JSON line. It uses no table library calls of its own.
DRIVER = """
import json, sys
from pathlib import Path
from fleetoffer import evaluation, levels, pipeline
folder = Path(sys.argv[1])
start = evaluation.OFFERS_FROM["evaluation"]
frame = levels.same_days(
    pipeline._read(folder, "accounting.csv"),
    pipeline._read(folder, "sweep.csv.gz"),
    pipeline._read(folder, "days.csv.gz"),
    start,
)
rows = [row for row in frame.to_dict("records") if row.get("form") == "average"]
def plain(value):
    return value.item() if hasattr(value, "item") else str(value)
print("RESULT " + json.dumps({"start": str(start), "rows": rows}, default=plain))
"""


class CannotRun(Exception):
    """The re-run cannot start or its inputs are not the stated ones (exit status 2)."""


class Failed(Exception):
    """The stage or the reading of its tables failed (exit status 1)."""


def say(text: str) -> None:
    print(text, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_problems(package: Path) -> list[str]:
    """The pinned files that are missing or whose bytes differ, one line each."""
    problems = []
    for relative, expected in PINNED_FILES.items():
        path = package.joinpath(*MODULE_FOLDER, relative)
        if not path.is_file():
            problems.append(f"{relative} is missing")
        elif sha256_file(path) != expected:
            problems.append(f"{relative} differs from the exported file")
    return problems


def check_package(top: Path) -> tuple[Path, Path, str]:
    """Step 1, package part: return the package folder, its interpreter and its provenance text."""
    package = top.joinpath(*PACKAGE_PARTS)
    if not package.is_dir():
        raise CannotRun(f"the earlier package is not in {'/'.join(PACKAGE_PARTS)}")
    interpreter = package.joinpath(*INTERPRETER_PARTS)
    if not (interpreter.is_file() and os.access(interpreter, os.X_OK)):
        raise CannotRun(f"the package's own interpreter is missing: {'/'.join(INTERPRETER_PARTS)}")
    try:
        provenance = (package / "PROVENANCE.txt").read_text()
    except OSError as problem:
        raise CannotRun(f"PROVENANCE.txt cannot be read: {problem}") from problem
    lacking = [phrase for phrase in PROVENANCE_PHRASES if phrase not in provenance]
    if lacking:
        raise CannotRun(f"PROVENANCE.txt does not say: {', '.join(lacking)}")
    problems = pin_problems(package)
    if problems:
        raise CannotRun("the earlier package is not the stated export: " + "; ".join(problems))
    return package, interpreter, provenance


def months_needed() -> list[str]:
    first_year, first_month = (int(part) for part in FIRST_MONTH.split("-"))
    last_year, last_month = (int(part) for part in LAST_MONTH.split("-"))
    out = []
    year, month = first_year, first_month
    while (year, month) <= (last_year, last_month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def check_months(top: Path) -> list[Path]:
    """Step 1, data part: the month files the stage opens. Return their paths in order."""
    folder = top.joinpath(*MONTH_PARTS)
    paths = [folder / f"{month}.json.gz" for month in months_needed()]
    missing = [path.name for path in paths if not (path.is_file() and path.stat().st_size > 0)]
    if missing:
        raise CannotRun(
            f"{len(missing)} Caltech month file(s) missing or empty in {'/'.join(MONTH_PARTS)}: "
            + ", ".join(missing)
        )
    return paths


def package_environment(package: Path, git_dir: Path | None = None) -> dict[str, str]:
    """The environment of everything run with the package's interpreter."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
        and not key.startswith("ACN_API_TOKEN")
        and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONPYCACHEPREFIX"}
    }
    environment["PYTHONPATH"] = str(package.joinpath(*SOURCE_PARTS))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if git_dir is not None:
        environment["GIT_DIR"] = str(git_dir)
    return environment


def check_imports(package: Path, interpreter: Path) -> None:
    done = subprocess.run(
        [str(interpreter), "-c", IMPORT_CHECK],
        cwd=package,
        env=package_environment(package),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if done.returncode != 0:
        tail = " | ".join(done.stderr.strip().splitlines()[-3:])
        raise CannotRun(f"the package's modules do not import with its own interpreter: {tail}")


def make_data_root(run_dir: Path, month_files: list[Path]) -> Path:
    """A data root whose `raw/caltech` holds links to exactly the needed month files."""
    folder = run_dir / "dataroot" / "raw" / "caltech"
    folder.mkdir(parents=True)
    for path in month_files:
        (folder / path.name).symlink_to(path.resolve())
    return run_dir / "dataroot"


def make_scratch_repository(run_dir: Path) -> Path:
    """A bare git repository with one empty commit tagged PLAN_TAG; return its folder."""
    git_dir = run_dir / "scratch-tag.git"
    identity = {
        "GIT_AUTHOR_NAME": "scratch",
        "GIT_AUTHOR_EMAIL": "scratch@example.invalid",
        "GIT_COMMITTER_NAME": "scratch",
        "GIT_COMMITTER_EMAIL": "scratch@example.invalid",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    environment = {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        **identity,
    }

    def git(*arguments: str, text_input: str | None = None) -> str:
        try:
            done = subprocess.run(
                ["git", f"--git-dir={git_dir}", *arguments],
                input=text_input,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as problem:
            raise CannotRun(f"git could not be started: {problem}") from problem
        if done.returncode != 0:
            raise CannotRun(f"git {arguments[0]} failed: {done.stderr.strip()}")
        return done.stdout.strip()

    try:
        subprocess.run(
            ["git", "init", "--bare", "-q", "--template=", str(git_dir)],
            env=environment,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as problem:
        raise CannotRun(f"the scratch git repository could not be made: {problem}") from problem
    tree = git("mktree", text_input="")
    commit = git("commit-tree", tree, "-m", "scratch commit for the tag check")
    git("update-ref", f"refs/tags/{PLAN_TAG}", commit)
    if git("tag", "--list", PLAN_TAG) != PLAN_TAG:
        raise CannotRun("the scratch git repository does not list the tag")
    return git_dir


def run_stage(
    package: Path, interpreter: Path, data_root: Path, out: Path, git_dir: Path, workers: int
) -> None:
    """Step 3: the package's own stage, its output going straight through."""
    command = [
        str(interpreter),
        "-m",
        "fleetoffer.pipeline",
        *STAGE_ARGUMENTS,
        "--data",
        str(data_root),
        "--out",
        str(out),
        "--workers",
        str(workers),
    ]
    say(f"starting the stage: {' '.join(['python -m fleetoffer.pipeline', *STAGE_ARGUMENTS])}")
    try:
        done = subprocess.run(
            command, cwd=package, env=package_environment(package, git_dir), check=False
        )
    except OSError as problem:
        raise Failed(f"the stage could not be started: {problem}") from problem
    if done.returncode != 0:
        raise Failed(f"the stage ended with exit code {done.returncode}")


def read_rows(package: Path, interpreter: Path, out: Path, git_dir: Path) -> dict:
    """Step 4: the package's own `same_days` on the stage's tables; return the driver's payload."""
    try:
        done = subprocess.run(
            [str(interpreter), "-c", DRIVER, str(out)],
            cwd=package,
            env=package_environment(package, git_dir),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as problem:
        raise Failed(f"the numbers could not be read: {problem}") from problem
    if done.returncode != 0:
        tail = " | ".join(done.stderr.strip().splitlines()[-3:])
        raise Failed(f"the package's own accounting step failed (exit {done.returncode}): {tail}")
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        raise Failed("the package's accounting step printed no result")
    try:
        payload = json.loads(lines[-1].removeprefix("RESULT "))
        rows = payload["rows"]
    except (ValueError, KeyError, TypeError) as problem:
        raise Failed(f"the accounting result cannot be read: {problem}") from problem
    if not isinstance(rows, list):
        raise Failed("the accounting result holds no list of rows")
    return payload


def describe_rows(rows: list[dict]) -> str:
    return "; ".join(
        f"({row.get('form')!r}, {row.get('step')!r}, {row.get('who')!r})" for row in rows
    )


def pick(rows: list[dict], name: str) -> tuple[float, dict]:
    """The one row that filter `name` selects and its value; anything else is a failure."""
    wanted = ROW_FILTERS[name]
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == value for key, value in wanted.items() if key != "column")
    ]
    if len(matches) != 1:
        raise Failed(
            f"{name}: {len(matches)} rows match {wanted}, not exactly one; "
            f"rows of the {FORM} form: {describe_rows(rows) or 'none'}"
        )
    value = matches[0].get(wanted["column"])
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise Failed(f"{name}: the value {value!r} in column {wanted['column']} is not a number")
    return float(value), matches[0]


def scored_days(row: dict) -> int | None:
    value = row.get("days")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return int(value)


def format_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def rerun(top: Path, run_dir: Path, workers: int) -> dict:
    """Steps 1 to 5. Return the results written; raise CannotRun or Failed."""
    began = time.monotonic()
    package, interpreter, provenance = check_package(top)
    month_files = check_months(top)
    check_imports(package, interpreter)
    say(f"checked: the earlier package is the stated export; {len(month_files)} month files found")
    data_root = make_data_root(run_dir, month_files)
    git_dir = make_scratch_repository(run_dir)
    say(DISCLOSURE)
    out = run_dir / "evaluation"
    stage_began = datetime.now(UTC)
    stage_clock = time.monotonic()
    run_stage(package, interpreter, data_root, out, git_dir, workers)
    stage_seconds = time.monotonic() - stage_clock
    stage_ended = datetime.now(UTC)
    payload = read_rows(package, interpreter, out, git_dir)
    rows = payload["rows"]
    measured, picked = {}, {}
    for name in ROW_FILTERS:
        measured[name], picked[name] = pick(rows, name)
    changed = pin_problems(package)
    if changed:
        raise Failed("the earlier package changed during the run: " + "; ".join(changed))
    results = {
        "published": {
            **PUBLISHED,
            "label": "published by the earlier analysis (plan section 12); not measured here",
        },
        "measured": measured,
        "row_filters": ROW_FILTERS,
        "scored_days": {
            "perfect_knowledge": scored_days(picked["perfect_knowledge_kwh"]),
            "learned": scored_days(picked["learned_share"]),
            "population": scored_days(picked["population_share"]),
        },
        "offers_counted_from": payload.get("start"),
        "elapsed_seconds": time.monotonic() - began,
        "stage_seconds": stage_seconds,
        "stage_start": format_time(stage_began),
        "stage_end": format_time(stage_ended),
        "workers": workers,
        "package_commit_text": ", ".join(re.findall(r"commit [0-9a-f]{7,40}", provenance)),
        "package_provenance": provenance.strip(),
        "package_files_checked": dict(PINNED_FILES),
        "month_files": [path.name for path in month_files],
        "stage": "python -m fleetoffer.pipeline " + " ".join(STAGE_ARGUMENTS),
        "disclosure": DISCLOSURE,
    }
    (run_dir / RESULTS_NAME).write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/earlier_rerun.py",
        description="Re-run the earlier analysis's code unchanged and record three numbers.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="stage worker count")
    parser.add_argument(
        "--check-only", action="store_true", help="check the package and the data, run nothing"
    )
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="folder holding reference/ and data/ (default: the repository top folder)",
    )
    options = parser.parse_args(argv)
    if options.workers < 1:
        parser.error("--workers must be at least 1")
    top = options.top
    try:
        if options.check_only:
            package, interpreter, _ = check_package(top)
            months = check_months(top)
            check_imports(package, interpreter)
            say(f"ready: the earlier package is the stated export; {len(months)} month files found")
            return 0
        run_dir_text = os.environ.get(RUN_DIR_VARIABLE, "")
        if not run_dir_text or not Path(run_dir_text).is_dir():
            raise CannotRun(f"{RUN_DIR_VARIABLE} must name an existing run folder")
        results = rerun(top, Path(run_dir_text), options.workers)
    except CannotRun as problem:
        say(f"COULD NOT RUN: {problem}")
        return 2
    except Failed as problem:
        say(f"FAILED: {problem}")
        return 1
    measured = results["measured"]
    say(
        f"measured: perfect knowledge {measured['perfect_knowledge_kwh']!r} kWh, "
        f"learned share {measured['learned_share']!r}, "
        f"population share {measured['population_share']!r}; wrote {RESULTS_NAME}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
