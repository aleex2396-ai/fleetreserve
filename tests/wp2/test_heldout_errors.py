"""Which errors leave the gated opener and the log reader, and as what.

`open_sealed`: a failed check is a `SealRefused` with its own message and no cause; a hash list
nested too deep (a `RecursionError` inside) or that is not JSON is such a failed check, refused
as "ill-formed" with no cause and the original exception suppressed; any other `Exception` met
on the way leaves as `SealRefused` with that exception as its cause: a `TypeError` (a name or a
root of the wrong type), a `ValueError` (a root holding a NUL character), a `UnicodeDecodeError`
(a hash list that is not UTF-8), a `RuntimeError` and a `TypeError` while the log line is made.
A `KeyboardInterrupt` or `SystemExit` passes unchanged. In every case nothing is returned, no
parser is called and the log gains no line. A git call that fails is refused with git's error
output without the line ending git ends it with.

`asr` and `norway_sites`: a parser's own error after an opening passes unchanged, with the
opening logged.

The log is never read or written through a symbolic link: an opening with a link at the log's
path, to a file outside or dangling, is refused, nothing returned, the link's target unchanged
or not made; `logged_openings` on such a link raises `OSError` (`ELOOP`). A log whose last line
has no line ending gets one before the next line, so both lines read back. A new log has the
mode a new file gets from `open`.

`logged_openings`: an absent or empty log gives no opening; a folder at the log's path raises
`IsADirectoryError`; bytes that are not UTF-8 raise `UnicodeDecodeError`, and UTF-8 bytes are
read as UTF-8 under a locale whose encoding is Latin-1; a line that is not JSON, is nested too
deep, is not an object (a list of the field names among them), lacks a field, has an extra
field or has its fields in another order raises `ValueError` naming that line.
"""

import errno
import json
import os
import stat
from pathlib import Path

import pytest
from fake_heldout import ASR_FILE, NORWAY_FILE, commit, git, run_python, sealed_repo, unreadable

import fleetreserve
from fleetreserve import heldout_loader, load_asr
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.sessions import LayoutError

CONTENT = b"made-up sealed bytes\n"


def _sealed_file(root):
    return root / "data" / "heldout" / ASR_FILE[0] / ASR_FILE[1]


def _wrapped(root, cause, *names, monkeypatch=None):
    """Open `names` (default: the a.s.r. file) under `root`; it must be refused with a cause of
    type `cause`, nothing returned, no parser called and no line logged."""
    calls = []
    if monkeypatch is not None:
        monkeypatch.setattr(load_asr, "from_bytes", lambda content: calls.append(content))
    result = None
    with pytest.raises(SealRefused) as refused:
        result = heldout_loader.open_sealed(root, *(names or ASR_FILE))
    assert type(refused.value.__cause__) is cause
    assert result is None and calls == []
    return refused.value


@pytest.mark.parametrize("names", [(5, ASR_FILE[1]), (ASR_FILE[0], None), (ASR_FILE[0], b"x")])
def test_a_name_of_the_wrong_type_is_a_refusal(tmp_path, names):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))
    refused = _wrapped(root, TypeError if names[1] is not None else ValueError, *names)
    assert str(refused).startswith("refused: ")
    assert heldout_loader.logged_openings(root) == []


def test_a_root_of_the_wrong_type_is_a_refusal():
    with pytest.raises(SealRefused) as refused:
        heldout_loader.open_sealed(None, *ASR_FILE)
    assert type(refused.value.__cause__) is TypeError


def test_a_root_holding_a_nul_character_is_a_refusal(tmp_path):
    _wrapped(tmp_path / "re\0po", ValueError)


def test_a_hash_list_nested_too_deep_is_a_refusal(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    depth = 200_000
    (root / "data-hashes.json").write_text(
        '{"files": ' + "[" * depth + "]" * depth + "}", encoding="utf-8"
    )
    commit(root, "a hash list nested too deep")
    git(root, "tag", "-f", "frozen-plan")
    unreadable(_sealed_file(root))
    with pytest.raises(SealRefused, match=r"ill-formed \(RecursionError\)$") as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert refused.value.__cause__ is None and refused.value.__suppress_context__ is True
    assert heldout_loader.logged_openings(root) == []


def test_a_hash_list_that_is_not_json_is_refused_with_no_cause(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "data-hashes.json").write_text("not json\n", encoding="utf-8")
    commit(root, "a hash list that is not JSON")
    git(root, "tag", "-f", "frozen-plan")
    unreadable(_sealed_file(root))
    with pytest.raises(SealRefused) as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert str(refused.value) == "data-hashes.json at the tag is ill-formed (JSONDecodeError)"
    assert refused.value.__cause__ is None and refused.value.__suppress_context__ is True
    assert heldout_loader.logged_openings(root) == []


def test_a_failed_git_call_is_refused_with_its_error_without_a_line_ending(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "data-hashes.json").unlink()
    commit(root, "the hash list removed")
    git(root, "tag", "-f", "frozen-plan")
    head = git(root, "rev-parse", "HEAD")
    with pytest.raises(SealRefused) as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    message = str(refused.value)
    assert message.startswith(f"git show {head}:data-hashes.json failed: fatal: ")
    assert message == message.strip() and "\n" not in message


def test_a_hash_list_that_is_not_utf8_is_a_refusal(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "data-hashes.json").write_bytes(b'{"files": ["\xff"]}')
    commit(root, "a hash list that is not UTF-8")
    git(root, "tag", "-f", "frozen-plan")
    unreadable(_sealed_file(root))
    refused = _wrapped(root, UnicodeDecodeError)
    assert "could not be run" in str(refused)
    assert heldout_loader.logged_openings(root) == []


def test_a_runtime_error_while_hashing_is_a_refusal(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})

    def fails(content):
        raise RuntimeError("made up")

    monkeypatch.setattr(heldout_loader.hashlib, "sha256", fails)
    refused = _wrapped(root, RuntimeError, monkeypatch=monkeypatch)
    assert str(refused) == "refused: RuntimeError: made up"
    assert heldout_loader.logged_openings(root) == []


def test_an_error_while_the_log_line_is_made_is_a_refusal(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})

    def fails(line):
        raise TypeError("made up")

    monkeypatch.setattr(heldout_loader.json, "dumps", fails)
    _wrapped(root, TypeError, monkeypatch=monkeypatch)
    assert not heldout_loader.log_path(root).exists()


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_an_interruption_passes_unchanged(tmp_path, monkeypatch, stop):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))

    def stops(*args, **kwargs):
        raise stop()

    monkeypatch.setattr(heldout_loader.subprocess, "run", stops)
    with pytest.raises(stop):
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert heldout_loader.logged_openings(root) == []


def test_a_failed_check_keeps_its_own_message_and_no_cause(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    unreadable(_sealed_file(root))
    with pytest.raises(SealRefused) as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert type(refused.value) is SealRefused
    assert str(refused.value) == (
        "no tag frozen-plan: sealed files open only at the tag frozen-plan, or at a logged "
        "parsing fix of it"
    )
    assert refused.value.__cause__ is None


def test_a_parser_error_after_an_opening_passes_unchanged(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: b"no,such,columns\n", NORWAY_FILE: b"no;columns\n"})
    with pytest.raises(LayoutError):
        heldout_loader.asr(root)
    with pytest.raises(LayoutError):
        heldout_loader.norway_sites(root)
    openings = heldout_loader.logged_openings(root)
    assert [(o["set"], o["file"]) for o in openings] == [ASR_FILE, NORWAY_FILE]


def test_a_log_reader_on_an_absent_or_empty_log_gives_no_opening(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert heldout_loader.logged_openings(root) == []
    heldout_loader.log_path(root).write_bytes(b"")
    assert heldout_loader.logged_openings(root) == []


def test_a_log_reader_on_a_folder_raises(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).mkdir()
    with pytest.raises(IsADirectoryError):
        heldout_loader.logged_openings(root)


@pytest.mark.parametrize("target", ["outside.log", "gone.log"])
def test_a_log_reader_on_a_link_raises(tmp_path, target):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (tmp_path / "outside.log").write_text(_line() + "\n", encoding="utf-8")
    heldout_loader.log_path(root).symlink_to(tmp_path / target)
    with pytest.raises(OSError) as raised:
        heldout_loader.logged_openings(root)
    assert raised.value.errno == errno.ELOOP


@pytest.mark.parametrize("target", ["outside.log", "gone.log"])
def test_an_opening_with_a_link_at_the_log_path_is_refused(tmp_path, monkeypatch, target):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (tmp_path / "outside.log").write_text("kept\n", encoding="utf-8")
    heldout_loader.log_path(root).symlink_to(tmp_path / target)
    calls = []
    monkeypatch.setattr(load_asr, "from_bytes", lambda content: calls.append(content))
    result = None
    with pytest.raises(SealRefused) as refused:
        result = heldout_loader.asr(root)
    assert str(refused.value) == (
        "the openings log could not be written; data/heldout/asr/"
        "202410DatasetEVOfficeParking_v0.csv not returned"
    )
    assert refused.value.__cause__.errno == errno.ELOOP
    assert result is None and calls == []
    assert (tmp_path / "outside.log").read_text(encoding="utf-8") == "kept\n"
    assert not (tmp_path / "gone.log").exists()


def test_a_last_log_line_without_a_line_ending_is_ended_before_the_next(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).write_text(_line(set="earlier"), encoding="utf-8")
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    openings = heldout_loader.logged_openings(root)
    assert [o["set"] for o in openings] == ["earlier", "asr", "asr"]
    text = heldout_loader.log_path(root).read_text(encoding="utf-8")
    assert text.count("\n") == 3 and text.endswith("\n") and "\n\n" not in text


def test_a_new_log_has_the_mode_open_gives_a_new_file(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    mask = os.umask(0o022)
    try:
        heldout_loader.open_sealed(root, *ASR_FILE)
        with open(tmp_path / "plain.txt", "w", encoding="utf-8"):
            pass
    finally:
        os.umask(mask)
    mode = stat.S_IMODE(heldout_loader.log_path(root).stat().st_mode)
    assert mode == stat.S_IMODE((tmp_path / "plain.txt").stat().st_mode) == 0o644


READ_UNDER_LATIN1 = """
import json, locale, sys
from pathlib import Path
from fleetreserve import heldout_loader
assert locale.getencoding().upper().replace("-", "") in ("ISO88591", "LATIN1"), locale.getencoding()
assert sys.flags.utf8_mode == 0
root = Path(sys.argv[1])
print(json.dumps([o["command"] for o in heldout_loader.logged_openings(root)]))
"""


def test_a_log_reader_reads_utf8_whatever_the_locale(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    line = json.dumps(dict.fromkeys(heldout_loader.LOG_FIELDS, "é"), ensure_ascii=False)
    heldout_loader.log_path(root).write_bytes(line.encode("utf-8") + b"\n")
    (tmp_path / "cwd").mkdir()
    done = run_python(
        ["-X", "utf8=0", "-c", READ_UNDER_LATIN1, str(root)],
        path=Path(fleetreserve.__file__).resolve().parents[1],
        cwd=tmp_path / "cwd",
        extra={"LC_ALL": "en_US.ISO8859-1", "LANG": "en_US.ISO8859-1"},
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["é"]


def test_a_log_reader_on_bytes_that_are_not_utf8_raises(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).write_bytes(b"\xff\n")
    with pytest.raises(UnicodeDecodeError):
        heldout_loader.logged_openings(root)


def _line(**changes):
    fields = dict.fromkeys(heldout_loader.LOG_FIELDS, "x")
    fields.update(changes)
    return json.dumps(fields)


@pytest.mark.parametrize(
    "bad, message",
    [
        ("not json", "is not a log line"),
        ("[" * 200_000 + "]" * 200_000, "is not a log line"),
        ('["a list"]', "does not have the log's fields"),
        (json.dumps(list(heldout_loader.LOG_FIELDS)), "does not have the log's fields"),
        (json.dumps(dict.fromkeys(heldout_loader.LOG_FIELDS[:-1], "x")), "does not have"),
        (_line(extra="x"), "does not have the log's fields"),
        (
            json.dumps(dict.fromkeys(reversed(heldout_loader.LOG_FIELDS), "x")),
            "does not have the log's fields",
        ),
    ],
    ids=["text", "deep", "list", "list-of-the-field-names", "missing", "extra", "order"],
)
def test_a_damaged_second_line_is_named(tmp_path, bad, message):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).write_text(_line() + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"line 2 {message}"):
        heldout_loader.logged_openings(root)


def test_a_log_of_good_lines_reads_back_in_order(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    lines = [_line(set="a"), _line(set="b"), _line(set="c")]
    heldout_loader.log_path(root).write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [o["set"] for o in heldout_loader.logged_openings(root)] == ["a", "b", "c"]
