"""The label of a trial made after the sealed run (`after_sealed_run`) and where it is taken
(`learn_run`, `dial_choice`), on made-up units. Written, not run.

Each test says which line of code turns it red when removed.

1. `mark`, `summary_head`, `write_note` and `refuse_held_out` of `after_sealed_run`: nothing
   happens without the flag; with it the key, the label, the first lines and the file appear.
   Red without the `if not enabled: return` line in `mark` (a run without the flag would carry the
   key), and without `document[KEY] = record()` (no key).
2. `learn_run --made-after-sealed-run`: `learn.json`, `learn-gate.json`, the printed summary and
   the label file carry the label. Red without `after_sealed_run.mark(results, ...)` (no key in
   `learn.json`), without `after_sealed_run.mark(document, ...)` (none in `learn-gate.json`),
   without `*after_sealed_run.summary_head(results)` (first line differs) and without
   `after_sealed_run.write_note(folder, ...)` (no file). A run without the flag has no trace.
3. The flag on the route of held-out units could not run (exit 2): red without the
   `refuse_held_out` call in `learn_run.main`.
4. `dial_choice --made-after-sealed-run`: `dial-choice.json`, the per-split file, the summary and
   the label file carry the label. Red without the `mark` call (no key), without `KEY` in the
   per-split file's identity tuple (that file lacks it), without the line in
   `dial_summary.summary_lines` and without `write_note`.
"""

import json
from types import SimpleNamespace

import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, gate_units, matrix_document, write_build
from learn_fake import small_gate_units, small_matrix

from fleetreserve import (
    after_sealed_run,
    dial_choice,
    gate_run,
    learn_run,
    run_verify,
    sealed_route,
)

FLAG = after_sealed_run.FLAG


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


# --- 1. the module -----------------------------------------------------------------------------


def test_mark_adds_the_key_and_prefixes_the_label_only_when_enabled():
    plain = {"label": "dial choice: measured"}
    after_sealed_run.mark(plain, False)
    assert plain == {"label": "dial choice: measured"}
    marked = {"label": "dial choice: measured"}
    after_sealed_run.mark(marked, True)
    assert marked[after_sealed_run.KEY] == after_sealed_run.record()
    assert marked["label"] == f"{after_sealed_run.WHAT}. dial choice: measured"
    unlabelled = {}
    after_sealed_run.mark(unlabelled, True)
    assert list(unlabelled) == [after_sealed_run.KEY]


def test_the_sentence_names_the_date_and_that_no_slot_is_filled():
    text = after_sealed_run.SENTENCE
    assert "24 November 2026" in text and "held-out results were seen" in text
    assert "no pre-registered slot" in text and "development units only" in text


def test_summary_head_is_the_label_or_nothing():
    assert after_sealed_run.summary_head({}) == []
    document = {}
    after_sealed_run.mark(document, True)
    assert after_sealed_run.summary_head(document) == [
        after_sealed_run.WHAT,
        after_sealed_run.SENTENCE,
    ]


def test_write_note_writes_one_file_or_none(tmp_path):
    after_sealed_run.write_note(tmp_path, False)
    assert list(tmp_path.iterdir()) == []
    after_sealed_run.write_note(tmp_path, True)
    note = tmp_path / after_sealed_run.FILE_NAME
    assert note.read_text() == f"{after_sealed_run.WHAT}\n{after_sealed_run.SENTENCE}\n"


def test_the_flag_is_refused_only_on_the_held_out_route():
    assert after_sealed_run.refuse_held_out(False, True) is None
    assert after_sealed_run.refuse_held_out(False, False) is None
    assert after_sealed_run.refuse_held_out(True, False) is None
    assert "development units only" in after_sealed_run.refuse_held_out(True, True)


# --- 2. and 3. learn_run -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def learn_build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("learn-build")
    run, tables = write_build(folder, small_gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    return ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]


def _learn(arguments, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return learn_run.main([*arguments, "--model", "empirical", *extra])


def _files(folder) -> dict:
    return {p.name: p.read_text() for p in sorted(folder.iterdir()) if p.is_file()}


def test_a_learn_run_with_the_flag_carries_the_label_in_every_output(
    learn_build, tmp_path, monkeypatch, capsys
):
    code = _learn(learn_build, tmp_path / "out", monkeypatch, FLAG)
    text = capsys.readouterr().out
    assert code == 0, text
    files = _files(tmp_path / "out")
    for name in (learn_run.RESULTS_NAME, learn_run.GATE_NAME):
        assert json.loads(files[name])[after_sealed_run.KEY] == after_sealed_run.record()
    lines = text.splitlines()
    assert lines[:2] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    assert files[after_sealed_run.FILE_NAME].splitlines() == lines[:2]


def test_a_learn_run_without_the_flag_has_no_trace(learn_build, tmp_path, monkeypatch, capsys):
    code = _learn(learn_build, tmp_path / "out", monkeypatch)
    text = capsys.readouterr().out
    assert code == 0, text
    files = _files(tmp_path / "out")
    assert after_sealed_run.FILE_NAME not in files
    for content in (text, *files.values()):
        assert after_sealed_run.KEY not in content and after_sealed_run.WHAT not in content


def test_the_flag_on_the_held_out_route_could_not_run(learn_build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sealed_route, "matrix_route", lambda *args, **kwargs: SimpleNamespace(held_out=True)
    )
    code = _learn(learn_build, tmp_path / "out", monkeypatch, FLAG)
    text = capsys.readouterr().out
    assert code == 2 and text.startswith("COULD NOT RUN") and "development units only" in text
    assert _files(tmp_path / "out") == {}


# --- 4. dial_choice ----------------------------------------------------------------------------

NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.03-empirical",
    "main-rounded-hour-k0.3-eps0.03-oracle",
    "main-rounded-hour-k0.3-eps0.1-empirical",
    "main-rounded-hour-k0.3-eps0.1-oracle",
)


@pytest.fixture(scope="module")
def dial_build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dial-build")
    found = gate_units()
    found[GATE_UNITS[1]] = gate_cars(60, early_every=3, early_out=12.0)
    run, tables = write_build(folder, found)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return [
        "--model",
        "empirical",
        "--matrix",
        str(matrix),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--splits",
        "4",
    ]


def _dial(arguments, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return dial_choice.main([*arguments, *extra])


def test_a_dial_choice_run_with_the_flag_carries_the_label_in_every_output(
    dial_build, tmp_path, monkeypatch, capsys
):
    code = _dial(dial_build, tmp_path / "out", monkeypatch, FLAG)
    text = capsys.readouterr().out
    assert code == 0, text
    files = _files(tmp_path / "out")
    for name in (dial_choice.RESULTS_NAME, dial_choice.SPLIT_OFFERS_NAME):
        document = json.loads(files[name])
        assert document[after_sealed_run.KEY] == after_sealed_run.record()
        assert document["label"].startswith(after_sealed_run.WHAT)
    lines = text.splitlines()
    assert lines[:2] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    assert files[dial_choice.SUMMARY_NAME].splitlines()[:2] == lines[:2]
    assert files[after_sealed_run.FILE_NAME].splitlines() == lines[:2]


def test_a_dial_choice_run_without_the_flag_has_no_trace(dial_build, tmp_path, monkeypatch, capsys):
    code = _dial(dial_build, tmp_path / "out", monkeypatch)
    text = capsys.readouterr().out
    assert code == 0, text
    files = _files(tmp_path / "out")
    assert after_sealed_run.FILE_NAME not in files
    for content in (text, *files.values()):
        assert after_sealed_run.KEY not in content and after_sealed_run.WHAT not in content
