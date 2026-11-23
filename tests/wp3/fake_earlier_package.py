"""A made-up stand-in for the earlier analysis's package, for the tests of `tools/earlier_rerun.py`.

`build_top(folder, ...)` makes a repository top folder holding `reference/part1` (a tiny
`fleetoffer` package, a provenance text, an interpreter) and `data/raw/acn/caltech` (month files).
The tiny package has the same few entry points the tool uses: `pipeline.main` with the stage's
command line, `pipeline._read`, `levels.same_days`, `evaluation.OFFERS_FROM` and
`data.plan_is_tagged`, which really runs `git tag --list analysis-plan` in the package folder, so
the tool's workaround for that check is exercised for real. The stage writes its tables and a
`rows.json` that records what it saw (its data folder's month files, its worker count, selected
environment variables). Its accounting step returns the rows given to `build_top`.
No pandas and no real data are involved.
"""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

PROVENANCE = (
    "The earlier analysis's package, exported unchanged with an archive command from a public\n"
    "repository at commit 05a6cf9 (MIT licence). Reference and WP3 re-run only.\n"
)

# Rows of the accounting step, with decoys: another form, another arm, a second table of rows.
GOOD_ROWS = [
    {"form": "segment", "step": "perfect knowledge", "who": "1", "size_mean_kwh": 99.0,
     "share_of_perfect": 1.0, "days": 100.0},
    {"form": "average", "step": "perfect knowledge", "who": "1", "size_mean_kwh": 45.63456789012345,
     "share_of_perfect": 1.0, "days": 120.0},
    {"form": "average", "step": "day-level rule as run", "who": "4", "size_mean_kwh": 9.0,
     "share_of_perfect": 0.5, "days": 119.0},
    {"form": "average", "step": "day-level rule as run", "who": "4d", "size_mean_kwh": 0.19,
     "share_of_perfect": 0.00412345678901234, "days": 118.0},
    {"form": "average", "step": "day-level rule as run", "who": "5d", "size_mean_kwh": 1.78,
     "share_of_perfect": 0.0391, "days": 117.0},
]  # fmt: skip

PIPELINE = """
import argparse
import json
import os
from pathlib import Path

from fleetoffer import data, evaluation

ROWS = json.loads({rows!r})
STAGE_EXIT = {stage_exit}
TOUCH_PACKAGE = {touch_package}


def _read(folder, name):
    return str(Path(folder) / name)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("period")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    assert (args.command, args.period) == ("evaluate", "evaluation"), args
    if not data.plan_is_tagged():
        raise SystemExit("the runs after the tag need the analysis-plan tag")
    month_folder = args.data / "raw" / "caltech"
    months = sorted(path.name for path in month_folder.glob("*.json.gz"))
    for name in months:  # the links must lead to readable files
        (month_folder / name).read_bytes()
    if TOUCH_PACKAGE:
        with (Path(data.__file__)).open("a") as handle:
            handle.write("# changed during the stage\\n")
    args.out.mkdir(parents=True)
    for name in ("accounting.csv", "sweep.csv.gz", "days.csv.gz"):
        (args.out / name).write_text("placeholder")
    seen = {{
        "rows": ROWS,
        "months": months,
        "workers": args.workers,
        "git_dir": os.environ.get("GIT_DIR"),
        "git_index_file": os.environ.get("GIT_INDEX_FILE"),
        "token_file_variable": os.environ.get("ACN_API_TOKEN_FILE"),
        "cwd": os.getcwd(),
        "no_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }}
    (args.out / "rows.json").write_text(json.dumps(seen))
    print("wrote evaluate evaluation to the output folder in 1 s")
    return STAGE_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
"""

DATA = """
import subprocess
from pathlib import Path

PLAN_TAG = "analysis-plan"
REPO = Path(__file__).resolve().parents[2]


def plan_is_tagged():
    try:
        found = subprocess.run(
            ["git", "-C", str(REPO), "tag", "--list", PLAN_TAG],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return found.stdout.strip() == PLAN_TAG
"""

LEVELS = """
import json
from pathlib import Path


class Frame:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient):
        assert orient == "records"
        return list(self.rows)


def same_days(accounts, sweep, days, start):
    assert start == "2019-09-01", start
    folder = Path(accounts).parent
    assert Path(sweep).parent == folder and Path(days).parent == folder
    return Frame(json.loads((folder / "rows.json").read_text())["rows"])
"""

EVALUATION = 'OFFERS_FROM = {"evaluation": "2019-09-01"}\n'


def months_between(first: str, last: str) -> list[str]:
    year, month = (int(part) for part in first.split("-"))
    last_year, last_month = (int(part) for part in last.split("-"))
    out = []
    while (year, month) <= (last_year, last_month):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def build_top(
    folder: Path,
    *,
    rows: list[dict] | None = None,
    stage_exit: int = 0,
    touch_package: bool = False,
    provenance: str = PROVENANCE,
    months: list[str] | None = None,
    broken_import: bool = False,
) -> Path:
    """Make the top folder; return it. `months` defaults to 2018-04 to 2020-04 (two sealed ones)."""
    top = folder / "top"
    package = top / "reference" / "part1"
    source = package / "src" / "fleetoffer"
    source.mkdir(parents=True)
    (package / "PROVENANCE.txt").write_text(provenance)
    (source / "__init__.py").write_text("")
    text = PIPELINE.format(
        rows=json.dumps(GOOD_ROWS if rows is None else rows),
        stage_exit=stage_exit,
        touch_package=touch_package,
    )
    (source / "pipeline.py").write_text(text)
    (source / "data.py").write_text(DATA)
    (source / "levels.py").write_text(
        ("raise ImportError('broken on purpose')\n" if broken_import else "") + LEVELS
    )
    (source / "evaluation.py").write_text(EVALUATION)
    interpreter = package / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n")
    interpreter.chmod(interpreter.stat().st_mode | stat.S_IXUSR)
    month_folder = top / "data" / "raw" / "acn" / "caltech"
    month_folder.mkdir(parents=True)
    for month in months if months is not None else months_between("2018-04", "2020-04"):
        (month_folder / f"{month}.json.gz").write_bytes(f"made-up {month}".encode())
    return top


def pins_for(top: Path) -> dict[str, str]:
    """The SHA-256 of the made-up package's files that the tool pins (made-up file names)."""
    source = top / "reference" / "part1" / "src" / "fleetoffer"
    names = ("pipeline.py", "data.py", "levels.py", "evaluation.py")
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}


def files_under(folder: Path) -> list[str]:
    """Every file and folder name below `folder`, to show that nothing was added or changed."""
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


def environment_without_git() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
