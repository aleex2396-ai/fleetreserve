"""The owner's approval of a parsing-fix commit before the opener opens a sealed file at it (the
opener's reading H7), in made-up repositories with made-up sealed files and a made-up plan.

A fix is two commits after the tag, as in `test_heldout_fix_commit.py`: a correction of one
set's loader test file, then its entry in plan section 14; the layout refusal it needs is logged
by the set's real route at the tag. The approvals record `data/heldout-fix-approvals.log` is
written here by hand (`fake_heldout.approve`, or raw bytes), as a person writes it.

1. At a fix with no record, nothing opens and nothing is logged; the record naming the fix
   commit's full id, it opens. At the tag no record is read: a malformed one does not stop it.
2. An approval does not carry: not to another set's fix on another line from the tag, not to a
   later correction of the same set after an approved fix, not to a commit whose tree is the
   approved one's and whose message differs by one byte.
3. A record that cannot be read (a folder, a symbolic link to an approval of the fix, bytes that
   are not UTF-8) or is malformed (each line below, beside a well-formed approval of the fix) is
   a refusal; the held-out truth build at an unapproved fix exits 2, could not run.
4. No module of the package or tool writes the record: its file name is written once, in the
   opener, and the opener uses it only to read the record.
"""

import ast
import json
import re

import pytest
from fake_heldout import (
    APPROVALS,
    ASR_FILE,
    NORWAY_FILE,
    approve,
    git,
    sealed_repo,
)
from fake_heldout import correction as _correction
from fake_heldout import entry as _entry
from fake_heldout import freeze as _freeze

from fleetreserve import heldout_loader, heldout_truth_build
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.sessions import TOP, LayoutError

MATRIX = TOP / "gate" / "matrix.json"
CONTENT = b"made-up sealed bytes\n"
KOREA_FILE = ("korea", "ChargingRecords.csv")
FILES = {ASR_FILE: CONTENT, NORWAY_FILE: CONTENT, KOREA_FILE: CONTENT}
ROUTES = {"asr": heldout_loader.asr, "norway": heldout_loader.norway_sites}
NAMES = {"asr": ASR_FILE, "norway": NORWAY_FILE}
WORDS = "yes, open the sealed files at this made-up fix"


def _refusal(root, set_name="asr"):
    """The set's route at the checked-out commit: it opens the made-up bytes and its parser
    refuses them, so the opening and its layout refusal are logged."""
    with pytest.raises(LayoutError):
        ROUTES[set_name](root)


def _fix(root, set_name="asr", name="fix", *earlier):
    """A correction of the set's loader test file and its entry naming it (and `earlier`); the
    entry commit."""
    corrected = _correction(root, set_name, name=name)
    return _entry(root, *earlier, corrected), corrected


def _opens(root, set_name="asr"):
    before = len(heldout_loader.logged_openings(root))
    assert heldout_loader.open_sealed(root, *NAMES[set_name]) == CONTENT
    openings = heldout_loader.logged_openings(root)
    assert len(openings) == before + 1
    return openings[-1]


def _refused(root, pattern, set_name="asr"):
    before = heldout_loader.logged_openings(root)
    result = None
    with pytest.raises(SealRefused, match=pattern):
        result = heldout_loader.open_sealed(root, *NAMES[set_name])
    assert result is None
    assert heldout_loader.logged_openings(root) == before


def _not_approved(root, head, absent=False):
    note = " (the record is absent)" if absent else ""
    _refused(
        root,
        "^"
        + re.escape(
            f"no approval of the fix commit {head} is recorded in data/heldout-fix-approvals.log"
            f"{note}: sealed files open at a parsing fix only after the owner's approval of that "
            "exact commit is written there by hand (reading H7)"
        )
        + "$",
    )


def _malformed(head, text):
    return (
        "^"
        + re.escape(
            f"the approvals record data/heldout-fix-approvals.log is malformed: {text}; nothing "
            f"opened at the fix commit {head} (reading H7)"
        )
        + "$"
    )


def _fixed(root):
    """The made-up repository frozen, the a.s.r. layout refused at the tag, then a fix of the
    set; (tag's commit, entry commit)."""
    tagged = _freeze(root)
    _refusal(root)
    head, _ = _fix(root)
    return tagged, head


# --- 1. an approval of the fix commit, none at the tag ----------------------------------------


def test_a_fix_opens_only_once_its_approval_names_its_full_id(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged, head = _fixed(root)
    _not_approved(root, head, absent=True)
    approve(root, head)
    line = _opens(root)
    assert (line["commit"], line["tag_commit"], line["fix_of_tag"]) == (head, tagged, tagged)
    assert heldout_loader.logged_openings(root, heldout_loader.APPROVALS_NAME) == [
        {"time_utc": "2026-01-01T00:00Z", "fix_commit": head, "words": WORDS}
    ]
    assert git(root, "status", "--porcelain") == ""


def test_the_tag_opens_with_no_record_and_never_reads_one(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    line = _opens(root)
    assert (line["commit"], line["tag_commit"]) == (tagged, tagged)
    (root / APPROVALS).write_bytes(b"not a record\n")
    line = _opens(root)
    assert "fix_of_tag" not in line


def test_the_approvals_record_lies_beside_the_openings_log(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    assert heldout_loader.approvals_path(root) == root / APPROVALS
    assert heldout_loader.approvals_path(root).parent == heldout_loader.log_path(root).parent


# --- 2. an approval does not carry -------------------------------------------------------------


def test_an_approval_of_another_sets_fix_does_not_carry(tmp_path):
    # Hand-worked: from the tag T, line 1 corrects norway (N1, its entry EN) and EN is approved;
    # line 2, also from T, corrects a.s.r. (A1, its entry EA). EA is a fix (its refusal at T),
    # and no approval names EA, so nothing opens there.
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "norway")
    _refusal(root, "asr")
    norway_fix, _ = _fix(root, "norway")
    approve(root, norway_fix)
    _opens(root, "norway")
    git(root, "switch", "-q", "-c", "other", tagged)
    asr_fix, _ = _fix(root, "asr")
    _not_approved(root, asr_fix)
    approve(root, asr_fix)
    _opens(root, "asr")


def test_an_approval_does_not_carry_to_a_later_correction_of_the_same_set(tmp_path):
    # Hand-worked: T, refusal at T; C1, E1 approved; the route at E1 opens and its layout is
    # refused there; C2, E2 naming C1 and C2. E2 is a fix (C2's refusal at E1) and differs from
    # E1, so E1's approval does not open it.
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    _refusal(root)
    first_fix, first = _fix(root)
    approve(root, first_fix)
    _refusal(root)  # opened at the approved first fix, its layout refused and logged there
    later_fix, _ = _fix(root, "asr", "second", first)
    _not_approved(root, later_fix)
    approve(root, later_fix)
    _opens(root)


def test_an_approval_does_not_carry_to_a_commit_that_differs_by_one_byte(tmp_path):
    # The approved entry commit made again with one byte added to its message: the same tree,
    # the same correction named in section 14, a different commit.
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    approve(root, head)
    message = git(root, "log", "-1", "--format=%B", head)
    git(root, "commit", "-q", "--amend", "--no-verify", "-m", message + "!")
    again = git(root, "rev-parse", "HEAD")
    assert again != head
    assert git(root, "rev-parse", f"{again}^{{tree}}") == git(root, "rev-parse", f"{head}^{{tree}}")
    _not_approved(root, again)


# --- 3. a record that cannot be read, or is malformed -----------------------------------------


def test_a_folder_at_the_records_path_is_a_refusal(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    (root / APPROVALS).mkdir()
    pattern = "^" + re.escape(
        "the approvals record data/heldout-fix-approvals.log cannot be read (IsADirectoryError): "
    )
    _refused(root, pattern + r".*" + re.escape(f"; nothing opened at the fix commit {head}"))


def test_a_symbolic_link_at_the_records_path_is_a_refusal(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    elsewhere = tmp_path / "elsewhere.log"  # a well-formed approval of the fix, elsewhere
    line = {"time_utc": "2026-01-01T00:00Z", "fix_commit": head, "words": WORDS}
    elsewhere.write_text(json.dumps(line) + "\n", encoding="utf-8")
    (root / APPROVALS).symlink_to(elsewhere)
    pattern = "^" + re.escape(
        "the approvals record data/heldout-fix-approvals.log cannot be read (OSError): "
    )
    _refused(root, pattern + r".*" + re.escape(f"; nothing opened at the fix commit {head}"))


def test_a_record_that_is_not_utf8_is_a_refusal(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    approve(root, head)
    with (root / APPROVALS).open("ab") as record:
        record.write(b'{"time_utc": "2026-01-01T00:00Z", "fix_commit": "\xff", "words": "y"}\n')
    pattern = "^" + re.escape(
        "the approvals record data/heldout-fix-approvals.log cannot be read (UnicodeDecodeError): "
    )
    _refused(root, pattern)


GOOD = '"time_utc": "2026-01-01T00:00Z", "fix_commit": "{head}"'
# (kind, a second line beside a well-formed approval of the fix): "json" not JSON, "shape" not
# the record's fields in their order, and the values: "words", "time", "commit".
MALFORMED = [
    ("json", "not json"),
    ("json", ""),
    ("shape", '["2026-01-01T00:00Z", "{head}", "yes"]'),
    ("shape", "{" + GOOD + "}"),
    ("shape", '{"fix_commit": "{head}", "time_utc": "2026-01-01T00:00Z", "words": "yes"}'),
    ("shape", "{" + GOOD + ', "words": "yes", "more": 1}'),
    ("words", "{" + GOOD + ', "words": ""}'),
    ("words", "{" + GOOD + ', "words": "  "}'),
    ("words", "{" + GOOD + ', "words": 1}'),
    ("time", '{"time_utc": "2026-01-01T00:00", "fix_commit": "{head}", "words": "yes"}'),
    ("time", '{"time_utc": "2026-01-01 00:00Z", "fix_commit": "{head}", "words": "yes"}'),
    ("time", '{"time_utc": "2026-13-01T00:00Z", "fix_commit": "{head}", "words": "yes"}'),
    ("time", '{"time_utc": "yesterday", "fix_commit": "{head}", "words": "yes"}'),
    ("time", '{"time_utc": null, "fix_commit": "{head}", "words": "yes"}'),
    ("commit", '{"time_utc": "2026-01-01T00:00Z", "fix_commit": "{short}", "words": "yes"}'),
    ("commit", '{"time_utc": "2026-01-01T00:00Z", "fix_commit": "{upper}", "words": "yes"}'),
    ("commit", '{"time_utc": "2026-01-01T00:00Z", "fix_commit": " {head}", "words": "yes"}'),
]
PROBLEMS = {
    "json": "line 2 is not a log line",
    "shape": "line 2 does not have the log's fields",
    "words": "line 2 has no words: the owner's words of the approval, as he gave them",
    "time": "line 2 has a time_utc that is not a UTC time written YYYY-MM-DDThh:mm[:ss[.ffffff]]Z",
    "commit": "line 2 has a fix_commit that is not a full commit id in lower-case hexadecimal",
}
IDS = [f"{kind}{number}" for number, (kind, _) in enumerate(MALFORMED)]


@pytest.mark.parametrize(("kind", "text"), MALFORMED, ids=IDS)
def test_a_malformed_line_beside_an_approval_of_the_fix_is_a_refusal(tmp_path, kind, text):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    approve(root, head)  # line 1: a well-formed approval of this fix commit
    line = text.replace("{head}", head).replace("{short}", head[:12])
    line = line.replace("{upper}", head.upper())
    with (root / APPROVALS).open("a", encoding="utf-8") as record:
        record.write(line + "\n")
    _refused(root, _malformed(head, f"heldout-fix-approvals.log {PROBLEMS[kind]}"))


def test_the_truth_build_at_an_unapproved_fix_could_not_run(tmp_path, monkeypatch, capsys):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    args = ["--matrix", str(MATRIX), "--out", str(tmp_path / "out")]
    assert heldout_truth_build.main(args, root=root) == 1  # norway's layout refused at the tag
    capsys.readouterr()
    head, _ = _fix(root, "norway")
    before = heldout_loader.logged_openings(root)
    assert heldout_truth_build.main(args, root=root) == 2
    printed = capsys.readouterr().out
    assert printed == (
        "COULD NOT RUN: the norway file was refused by the opener: no approval of the fix commit "
        f"{head} is recorded in data/heldout-fix-approvals.log (the record is absent): sealed "
        "files open at a parsing fix only after the owner's approval of that exact commit is "
        "written there by hand (reading H7); openings logged by this run: 0; nothing written\n"
    )
    assert heldout_loader.logged_openings(root) == before
    assert not (tmp_path / "out").exists()


# --- 4. no tool writes the record ----------------------------------------------------------------


def test_no_module_or_tool_writes_the_approvals_record():
    name = "heldout-fix-approvals"
    files = sorted((TOP / "src").rglob("*.py")) + sorted((TOP / "tools").rglob("*.py"))
    naming = [path.relative_to(TOP).as_posix() for path in files if name in path.read_text()]
    assert naming == ["src/fleetreserve/heldout_loader.py"]
    opener = TOP / "src" / "fleetreserve" / "heldout_loader.py"
    tree = ast.parse(opener.read_text(encoding="utf-8"))
    texts = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    named = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and name in node.value
        and id(node) not in texts
    ]
    assert named == ["heldout-fix-approvals.log"]  # the value of APPROVALS_NAME, and no other
    users = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.Name) and node.id in ("APPROVALS_NAME", "approvals_path"):
                    users.setdefault(node.id, set()).add(function.name)
    assert users == {
        "APPROVALS_NAME": {"approvals_path", "logged_openings", "_check_approval"},
        "approvals_path": {"logged_openings", "_check_approval"},
    }
    elsewhere = [
        path.relative_to(TOP).as_posix()
        for path in files
        if path != opener and re.search(r"\bapprovals_path\b|\bAPPROVALS_NAME\b", path.read_text())
    ]
    assert elsewhere == []


def test_the_approvals_record_is_read_whole_before_any_line_counts(tmp_path):
    # Two lines naming the fix, the second malformed: the first does not open it.
    root = sealed_repo(tmp_path, FILES, tag=False)
    _, head = _fixed(root)
    approve(root, head)
    approve(root, head, words="")
    _refused(root, _malformed(head, f"heldout-fix-approvals.log {PROBLEMS['words']}"))
    record = json.loads((root / APPROVALS).read_text().splitlines()[0])
    assert record["fix_commit"] == head
