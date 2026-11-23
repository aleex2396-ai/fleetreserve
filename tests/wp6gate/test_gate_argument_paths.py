"""The path arguments of `gate_run` and `learn_run` (`gate_run.argument_path`), and the bytes they
read once (`gate_run.read_inputs`, `gate_config.read`).

Made-up working copies (`git init` in a temporary folder, the current folder of the test) hold a
made-up sealed folder `data/heldout/norway` with copies of a made-up verified build's tables of one
development unit; nothing of a real data folder is named. Opens, listings and scans of a folder
are counted by identity on disk (`sealed_watch`), whatever path names them.

1. A path written as a URL, a scheme (a letter, then letters, digits, `+`, `-` or `.`, then `:`)
   before its first `/`, in any letter case, is refused with `CouldNotRun`; any other path is
   returned as written, with a leading `~` expanded.
2. Hand case, `file:`. The text `file:data/heldout/norway` names two folders: to pandas, which
   reads a string whose scheme is `file` as a URL, the working copy's sealed folder; to the file
   system, a folder of that literal name in the current folder, here holding copies of the
   build's tables (which pass the build's hashes). Given as `--tables`, and also as `--matrix` or
   `--truth-run`, both commands end with exit 2 and the one line of the URL refusal, and no file
   or folder of the sealed folder or of the literal folder is opened, listed or scanned.
3. Hand case, `~`. HOME is the working copy's `data` folder, so `~/heldout/norway` is, once
   expanded, the sealed folder; a folder of the literal name `~/heldout/norway` in the current
   folder holds copies of the build's tables. Given as `--tables`, both commands end with exit 2
   and the line saying that the first file checked there (the tables folder's setting file) lies
   in the data folder of a working copy, and nothing of either folder is opened.
4. Read once: on the made-up build outside any data folder, each command opens each table and the
   matrix file once, and the SHA-256 it records for each is that of the file's bytes.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from sealed_watch import watch

from fleetreserve import gate_run, learn_run, run_verify
from fleetreserve.truth_build import SETTING_FILE
from fleetreserve.truth_verify import CouldNotRun

UNIT = "nrel-from-may-2019"
NAMES = ("main-rounded-hour-k0.1-empirical", "main-rounded-hour-k0.1-oracle")
COMMANDS = ("gate_run", "learn_run")
SEALED = Path("data") / "heldout" / "norway"
URL_LINE = (
    "COULD NOT RUN: the path {text} is written as a URL ({scheme}); a run reads only files on "
    "this machine named by a path, never a URL; nothing read\n"
)


# --- 1. the rule ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "scheme"),
    [
        ("file:data/heldout/norway", "file:"),
        ("FILE:data/heldout/norway", "FILE:"),
        ("File:/abs/heldout", "File:"),
        ("file:///abs/heldout", "file:"),
        ("http://example.org/a.csv", "http:"),
        ("s3:bucket/key", "s3:"),
        ("git+ssh:x", "git+ssh:"),
    ],
)
def test_a_path_written_as_a_url_is_refused(text, scheme):
    with pytest.raises(CouldNotRun) as refused:
        gate_run.argument_path(text)
    assert str(refused.value) == URL_LINE.format(text=text, scheme=scheme)[15:-1]


def test_any_other_path_is_kept_with_a_leading_tilde_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for text in ("data/derived/a:b", "/abs/a:b", "relative/x", "x", "1x:y"):
        assert gate_run.argument_path(text) == Path(text)
        assert gate_run.argument_path(Path(text)) == Path(text)
    assert gate_run.argument_path("~/heldout/norway") == tmp_path / "home" / "heldout" / "norway"
    assert gate_run.argument_path("~") == tmp_path / "home"


# --- 2 and 3. the two hand cases through both commands -------------------------------------------


@pytest.fixture
def made(tmp_path, monkeypatch):
    """A made-up build outside any data folder, a working copy as the current folder holding its
    tables in its sealed folder, and a matrix scoring the unit (module text)."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    base = tmp_path / "base"
    base.mkdir()
    run, tables = write_build(base, {UNIT: gate_cars(5)})
    copy = tmp_path / "copy"
    copy.mkdir()
    subprocess.run(["git", "init", "--quiet", str(copy)], check=True)
    shutil.copytree(tables, copy / SEALED)
    document = matrix_document(NAMES)
    document["gate_units"], document["required_units"] = [UNIT], 1
    matrix = tmp_path / "matrix" / "matrix.json"
    matrix.parent.mkdir()
    matrix.write_text(json.dumps(document))
    monkeypatch.chdir(copy)
    return {"run": run, "tables": tables, "copy": copy, "matrix": matrix, "tmp": tmp_path}


def _main(command: str, made: dict, **given) -> int:
    values = {"matrix": made["matrix"], "tables": made["tables"], "truth_run": made["run"]}
    values.update(given)
    arguments = [] if command == "gate_run" else ["--model", "empirical"]
    for name, value in values.items():
        arguments += [f"--{name.replace('_', '-')}", str(value)]
    main = {"gate_run": gate_run.main, "learn_run": learn_run.main}[command]
    return main(arguments, root=made["copy"])


def _literal(made: dict, text: str) -> Path:
    """A folder named `text` literally, under the current folder, holding copies of the tables."""
    folder = made["copy"] / text
    shutil.copytree(made["tables"], folder)
    return folder


@pytest.mark.parametrize("command", COMMANDS)
def test_tables_written_as_a_file_url_are_refused_before_any_open(made, command, capsys):
    text = "file:" + str(SEALED)
    literal = _literal(made, text)
    with watch(made["copy"] / SEALED, literal) as seen:
        status = _main(command, made, tables=text)
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == URL_LINE.format(text=text, scheme="file:")


@pytest.mark.parametrize("option", ["matrix", "truth_run"])
@pytest.mark.parametrize("command", COMMANDS)
def test_a_matrix_or_truth_run_written_as_a_url_is_refused(made, command, option, capsys):
    text = f"FILE:{made[{'matrix': 'matrix', 'truth_run': 'run'}[option]]}"
    with watch(made["copy"] / SEALED) as seen:
        status = _main(command, made, **{option: text})
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == URL_LINE.format(text=text, scheme="FILE:")


@pytest.mark.parametrize("command", COMMANDS)
def test_tables_under_the_home_folder_are_checked_where_they_are_read(
    made, command, monkeypatch, capsys
):
    data = made["copy"] / "data"
    monkeypatch.setenv("HOME", str(data))
    text = "~/heldout/norway"
    literal = _literal(made, text)
    with watch(data / "heldout", literal) as seen:
        status = _main(command, made, tables=text)
    assert (status, seen) == (2, [])
    first = data / "heldout" / "norway" / SETTING_FILE  # the first file the check is given
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: a file is refused before it is opened: {first} lies in the data folder "
        f"{data} of a working copy, outside its derived tables {data / 'derived'}\n"
    )


# --- 4. read once --------------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
def test_each_table_and_the_matrix_are_read_once(made, command, monkeypatch, capsys):
    out = made["tmp"] / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    names = [f"{UNIT}-cars.csv", f"{UNIT}-days.csv"]
    with watch(made["tables"], made["matrix"].parent) as seen:
        status = _main(command, made)
    assert status == 0, capsys.readouterr().out
    opened = [os.path.basename(path) for event, path in seen if event == "open"]
    assert sorted(opened) == sorted([*names, "matrix.json"])
    results = {"gate_run": "gate.json", "learn_run": "learn.json"}[command]
    document = json.loads((out / results).read_text())
    for name in names:
        expected = hashlib.sha256((made["tables"] / name).read_bytes()).hexdigest()
        assert document["inputs"]["table_sha256"][name] == expected
    assert document["matrix"]["sha256"] == hashlib.sha256(made["matrix"].read_bytes()).hexdigest()
