"""The gated opener at a logged parsing fix after the tag (check 5, reading H6), in made-up
repositories with made-up sealed files and a made-up plan.

The made-up sealed bytes are no set's published layout, so each set's parser raises
`LayoutError` on them: a route called at the tag's commit opens the file, logs the opening and
then logs the refusal of its layout. A fix is two commits after the tag: a loader correction of
one set (its loader, or a test file of it), then an entry in plan section 14 naming it. Each
entry commit made here is approved at once in the approvals record (`_entry`), so the rules of
the fix are tested apart from the owner's approval, whose own tests are in
`test_heldout_fix_approval.py`.

Opens: at the tag, with a line of the seven fields as before; at a two-commit fix of the loader,
run in a new Python process from the made-up repository's own code (check 8 compares the
running code with the fix commit, whose loader differs from the tag's), with `fix_of_tag` on
its line; at a fix of a test file in this process; at a chain that corrects two sets, the second
refused at the first fix's entry commit.

A later correction of a set (reading H6(e)): refused with a refusal of the set logged at the tag
only, opens once one is logged at the first fix; a refusal of another set logged there does not
count; two corrections of one set with no fix commit between are refused; after another set's
fix, a refusal at that later fix commit counts.

Refused, each with its own message and nothing logged: no logged refusal of the set (then, the
refusal logged at the tag, it opens); a refusal logged only after the correction (then, one at
the tag, it opens); a path that is not a set's loader, a test file of one or the plan; a path
changed and changed back inside the chain; one commit correcting two sets; a correction and its
entry in one commit; a commit that changes nothing; the plan changed outside section 14;
section 14 naming the correction by a short id only; a merge between the tag and HEAD; HEAD not
descending from the tag; HEAD at the correction without its entry (then, the entry added, it
opens); an entry with no correction before it; an unclean tree at the fix (then, cleaned, it
opens); the running code at a fix that is the tag's and not the fix commit's.

The sealed routes' gate (`sealed_route.require_gate`) returns the tag's commit, with no fix
commit at the tag and with the entry commit as the fix commit at a fix of a test file; it refuses
an unclean tree at a fix (then, cleaned, it passes there); at a fix of the loader it refuses this
process, which runs the tag's loader, and passes in a new Python process run from the made-up
repository's own code.

The refusal log: a route's `LayoutError` at the tag passes unchanged and writes one line with
the opening's set, file, hash, commit and tag commit and the parser's message; when that line
cannot be written the error still passes, with a note. The held-out truth build, stopped at the
tag by the first set's layout (exit 1), leaves that refusal in the log; run again at a fix of
that set, it opens it with `fix_of_tag` and logs its refusal at the fix commit. The real plan
holds the heading of section 14 exactly once.
"""

import hashlib
import re

import pytest
from fake_heldout import (
    ASR_FILE,
    NORWAY_FILE,
    approve,
    commit,
    git,
    log_refusal,
    run_python,
    sealed_repo,
    write_command,
)
from fake_heldout import append_line as _change
from fake_heldout import correction as _correction
from fake_heldout import entry as _made_entry
from fake_heldout import freeze as _freeze
from fake_heldout import loader_test_file as _test_file
from fake_heldout import write_plan as _write_plan

from fleetreserve import heldout_loader, heldout_truth_build, sealed_route
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.sessions import TOP, LayoutError

MATRIX = TOP / "gate" / "matrix.json"
CONTENT = b"made-up sealed bytes\n"
KOREA_FILE = ("korea", "ChargingRecords.csv")
FILES = {ASR_FILE: CONTENT, NORWAY_FILE: CONTENT, KOREA_FILE: CONTENT}
OPENING_FIELDS = ["time_utc", "set", "file", "sha256", "commit", "tag_commit", "command"]
REFUSAL_FIELDS = ["time_utc", "set", "file", "sha256", "commit", "tag_commit", "layout"]


def _entry(root, *named):
    """The entry in section 14 naming `named` (`fake_heldout.entry`), then the owner's approval
    of the entry commit written by hand, so that the opener's check 8a (reading H7) passes at
    every fix these tests make; the approval's own rules: `test_heldout_fix_approval.py`."""
    made = _made_entry(root, *named)
    approve(root, made)
    return made


def _route(root, set_name):
    """The set's route at the checked-out commit."""
    if set_name == "asr":
        return heldout_loader.asr(root)
    if set_name == "norway":
        return heldout_loader.norway_sites(root)
    return heldout_loader.korea(root)


def _refusal(root, set_name="asr"):
    """The set's route at the checked-out commit: the parser refuses the made-up bytes."""
    with pytest.raises(LayoutError):
        _route(root, set_name)


def _head(root):
    return git(root, "rev-parse", "HEAD")


def _opens(root, names=ASR_FILE):
    before = len(heldout_loader.logged_openings(root))
    assert heldout_loader.open_sealed(root, *names) == CONTENT
    openings = heldout_loader.logged_openings(root)
    assert len(openings) == before + 1
    return openings[-1]


def _refused(root, match, names=ASR_FILE):
    before = heldout_loader.logged_openings(root)
    result = None
    with pytest.raises(SealRefused, match=match):
        result = heldout_loader.open_sealed(root, *names)
    assert result is None
    assert heldout_loader.logged_openings(root) == before


def _not_a_fix(root, tagged, reason):
    prefix = (
        f"HEAD {_head(root)} is a commit after the tag frozen-plan ({tagged}) and not a logged "
        "parsing fix of it: "
    )
    _refused(root, "^" + re.escape(f"{prefix}{reason} (reading H6)") + "$")


def _fix_line(line, head, tagged):
    assert list(line) == [*OPENING_FIELDS, "fix_of_tag"]
    assert (line["commit"], line["tag_commit"], line["fix_of_tag"]) == (head, tagged, tagged)


# --- opens -----------------------------------------------------------------------------------


def test_the_tag_opens_as_before_with_a_line_of_seven_fields(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    line = _opens(root)
    assert list(line) == OPENING_FIELDS
    assert (line["commit"], line["tag_commit"]) == (tagged, tagged)


def test_a_two_commit_fix_of_the_loader_opens_from_its_own_code_after_a_logged_refusal(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    corrected = _correction(root, loader=True)
    entry = _entry(root, corrected)
    command = write_command(tmp_path / "elsewhere" / "open_one.py")
    done = run_python([str(command), str(root), *ASR_FILE], path=root / "src", cwd=tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, f"opened {len(CONTENT)}\n", "")
    first, line = heldout_loader.logged_openings(root)
    assert list(first) == OPENING_FIELDS and first["commit"] == tagged
    _fix_line(line, entry, tagged)
    assert line["command"] == [str(command), str(root), *ASR_FILE]
    assert git(root, "status", "--porcelain") == ""


def test_a_fix_of_a_test_file_opens_in_this_process_with_fix_of_tag_on_its_line(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    corrected = _correction(root)
    entry = _entry(root, corrected)
    _fix_line(_opens(root), entry, tagged)


def test_a_chain_correcting_two_sets_opens_the_second_refused_at_the_first_fix(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "asr")
    first = _correction(root, "asr")
    entry = _entry(root, first)
    _refusal(root, "korea")  # opened at the first fix's entry commit, then refused
    refusals = heldout_loader.logged_refusals(root)
    assert [(r["set"], r["commit"]) for r in refusals] == [("asr", tagged), ("korea", entry)]
    second = _correction(root, "korea")
    _not_a_fix(
        root,
        tagged,
        "HEAD is a loader correction (korea) without its entry in plan section 14 after it",
    )
    last = _entry(root, first, second)
    _fix_line(_opens(root, KOREA_FILE), last, tagged)


def _refused_at(root, commit, set_name="asr"):
    """The set's layout refused by its route with HEAD truly at `commit` (logged there), then
    HEAD back on the branch."""
    git(root, "checkout", "-q", "--detach", commit)
    _refusal(root, set_name)
    git(root, "checkout", "-q", "main")


UNOPENED = (
    " (a refusal line of it there follows no matching opening in data/heldout-openings.log, so "
    "it does not count)"
)


def _second_needs(set_name, previous, corrected, fixes, note="", tests_only=True):
    """The refusal of a later correction; `tests_only`: it changes test files only and follows
    an entry (the `correction` helper's default), so the line adds that it may be left out."""
    leave_out = f", or leave out the later correction {corrected}, which changes tests only"
    return (
        f"no layout refusal of the {set_name} file is logged in data/heldout-refusals.log after "
        f"its previous correction {previous} and before the correction {corrected}{note}; a later "
        f"correction of a set needs a refusal of its own (reading H6(e)): open the {set_name} "
        f"file at the fix commit {' or '.join(fixes)} (the held-out truth build there), so that "
        f"its layout refusal is logged there{leave_out if tests_only else ''}"
    )


def test_a_second_fix_of_a_set_needs_a_refusal_of_its_own_logged_at_the_first_fix(tmp_path):
    # Hand-worked: refusal at T; C1 (asr), E1; C2 (asr), E2. The refusal at T lies before C1 and
    # so counts for C1 only; C2 needs one at a commit after C1 and before C2: E1.
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    second = _correction(root, name="second")
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix]))
    _refused_at(root, fix)  # opened at the first fix, its layout refused and logged there
    refusals = heldout_loader.logged_refusals(root)
    assert [(r["set"], r["commit"]) for r in refusals] == [("asr", tagged), ("asr", fix)]
    _opens(root)


def test_a_refusal_of_another_set_at_the_first_fix_does_not_count_for_the_second(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "asr")
    first = _correction(root, "asr")
    fix = _entry(root, first)
    _refused_at(root, fix, "korea")  # korea refused at the first fix: not asr's refusal
    second = _correction(root, "asr", name="second")
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix]))


def test_a_set_corrected_twice_with_no_fix_commit_between_is_refused(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    second = _correction(root, name="second")
    _entry(root, first, second)
    _not_a_fix(
        root,
        tagged,
        f"no layout refusal of the asr file is logged in data/heldout-refusals.log after its "
        f"previous correction {first} and before the correction {second}; a later correction of "
        "a set needs a refusal of its own (reading H6(e)): no fix commit lies between the two, "
        f"so none can be logged: start a new line from {tagged} with the corrections as one",
    )


def _parent(root, commit):
    return git(root, "rev-parse", f"{commit}^")


def test_a_set_corrected_twice_after_a_fix_names_the_commit_to_start_again_from(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "korea")
    korea = _correction(root, "korea")
    _refused_at(root, _entry(root, korea))
    first = _correction(root)
    second = _correction(root, name="second")
    _entry(root, korea, first, second)
    _not_a_fix(
        root,
        tagged,
        f"no layout refusal of the asr file is logged in data/heldout-refusals.log after its "
        f"previous correction {first} and before the correction {second}; a later correction of "
        "a set needs a refusal of its own (reading H6(e)): no fix commit lies between the two, "
        f"so none can be logged: start a new line from {_parent(root, first)} with the "
        "corrections as one",
    )


def test_a_later_correction_of_the_loader_itself_is_not_offered_to_be_left_out(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    second = _correction(root, loader=True)  # the loader module itself
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix], tests_only=False))


def test_a_second_fix_after_another_sets_fix_counts_a_refusal_at_either_fix_commit(tmp_path):
    # T: asr refused; C1 (asr), E1; korea refused at E1; C2 (korea), E2; C3 (asr), E3. C2's
    # refusal at E1 counts (korea's first correction). C3 needs an asr refusal after C1: none at
    # E1 or E2 is refused; one logged at E2, the fix after korea's, counts.
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "asr")
    first = _correction(root, "asr")
    fix = _entry(root, first)
    _refused_at(root, fix, "korea")
    korea = _correction(root, "korea")
    later_fix = _entry(root, first, korea)
    third = _correction(root, "asr", name="third")
    _entry(root, first, korea, third)
    _not_a_fix(root, tagged, _second_needs("asr", first, third, [fix, later_fix]))
    _refused_at(root, later_fix, "asr")
    _opens(root)


# --- a refusal counts only after its logged opening (reading H6(e)) ------------------------

LATE = "2999-01-01T00:00:00.000000Z"  # after every opening these tests log


def test_a_hand_written_refusal_where_nothing_was_opened_does_not_count(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    # nothing opened at E1; the line's time after every opening, so only its commit can fail it
    log_refusal(root, ASR_FILE, CONTENT, fix, tagged, opened=False, time=LATE)
    second = _correction(root, name="second")
    _entry(root, first, second)
    assert [o["commit"] for o in heldout_loader.logged_openings(root)] == [tagged]
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix], UNOPENED))
    _refused_at(root, fix)  # the honest opening and refusal at E1
    _opens(root)


def test_a_refusal_at_another_sets_correction_commit_does_not_count(tmp_path):
    # X, Y, X: the a.s.r. refusal written at the Korean correction, a commit no route can open at
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "asr")
    first = _correction(root, "asr")
    fix = _entry(root, first)
    _refused_at(root, fix, "korea")
    korea = _correction(root, "korea")
    log_refusal(root, ASR_FILE, CONTENT, korea, tagged, opened=False, time=LATE)
    later_fix = _entry(root, first, korea)
    third = _correction(root, "asr", name="third")
    _entry(root, first, korea, third)
    message = _second_needs("asr", first, third, [fix, later_fix], UNOPENED)
    _not_a_fix(root, tagged, message)


def test_a_refusal_timed_before_its_opening_does_not_count(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    git(root, "checkout", "-q", "--detach", fix)
    _opens(root)  # a real opening at E1, the bytes not parsed, so no refusal logged
    git(root, "checkout", "-q", "main")
    early = "1990-01-01T00:00:00.000000Z"  # before the tag was made
    log_refusal(root, ASR_FILE, CONTENT, fix, tagged, opened=False, time=early)
    second = _correction(root, name="second")
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix], UNOPENED))


def test_a_refusal_timed_after_another_opening_but_before_its_matching_one_does_not_count(
    tmp_path,
):
    # The time is compared with the opening that matches in all five fields (E1's), not with any
    # opening: the refusal line at E1 is timed at the tag's real opening, before E1's own.
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    git(root, "checkout", "-q", "--detach", fix)
    _opens(root)  # a real opening at E1, the bytes not parsed, so no refusal logged
    git(root, "checkout", "-q", "main")
    times = {o["commit"]: o["time_utc"] for o in heldout_loader.logged_openings(root)}
    assert list(times) == [tagged, fix]
    assert times[tagged] < times[fix]
    log_refusal(root, ASR_FILE, CONTENT, fix, tagged, opened=False, time=times[tagged])
    second = _correction(root, name="second")
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix], UNOPENED))
    # the same line timed at E1's own opening follows it: now the refusal counts
    log_refusal(root, ASR_FILE, CONTENT, fix, tagged, opened=False, time=times[fix])
    _opens(root)


def _hand_opening(root, at, tagged, **changed):
    """A line of the openings log written here: the opening of the a.s.r. file at the commit `at`
    of the tag's commit `tagged`, with the fields in `changed` set to other values."""
    line = {
        "time_utc": "2026-01-01T00:00:00.000000Z",
        "set": ASR_FILE[0],
        "file": ASR_FILE[1],
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "commit": at,
        "tag_commit": tagged,
        "command": ["made up"],
    } | changed
    if line["tag_commit"] != line["commit"]:
        line["fix_of_tag"] = line["tag_commit"]
    heldout_loader._append(heldout_loader.log_path(root), line)


def _other_value(field, tagged):
    return {
        "set": "norway",
        "file": "other.csv",
        "sha256": hashlib.sha256(b"other bytes").hexdigest(),
        "commit": tagged,  # an opening at the tag, for a refusal at the fix commit
        "tag_commit": "f" * 40,
    }[field]


@pytest.mark.parametrize("field", ["set", "file", "sha256", "commit", "tag_commit"])
def test_a_refusal_whose_only_matching_opening_differs_in_one_field_does_not_count(tmp_path, field):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    first = _correction(root)
    fix = _entry(root, first)
    # at E1 a refusal line, timed after every opening, and an opening that differs from it in
    # this one field only; nothing else was opened there
    _hand_opening(root, fix, tagged, **{field: _other_value(field, tagged)})
    log_refusal(root, ASR_FILE, CONTENT, fix, tagged, opened=False, time=LATE)
    second = _correction(root, name="second")
    _entry(root, first, second)
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix], UNOPENED))
    _hand_opening(root, fix, tagged)  # the opening that matches: now the refusal counts
    _opens(root)


def test_a_refusal_before_the_previous_correction_does_not_count_for_the_next(tmp_path):
    # T: asr and korea refused; K (korea), EK; asr refused at EK; C1 (asr), E1; C2 (asr), E2. The
    # refusal at EK lies before C1, so it counts for C1 only; C2 needs one logged at E1.
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _refusal(root, "korea")
    korea = _correction(root, "korea")
    before = _entry(root, korea)
    _refused_at(root, before)
    first = _correction(root)
    fix = _entry(root, korea, first)
    second = _correction(root, name="second")
    _entry(root, korea, first, second)
    refusals = heldout_loader.logged_refusals(root)
    asr_refusals = [r["commit"] for r in refusals if r["set"] == "asr"]
    assert asr_refusals == [tagged, before]
    _not_a_fix(root, tagged, _second_needs("asr", first, second, [fix]))
    _refused_at(root, fix)
    _opens(root)


# --- refused ---------------------------------------------------------------------------------


def test_refused_without_a_logged_refusal_of_the_set(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "korea")  # another set's refusal does not count
    corrected = _correction(root, "asr")
    _entry(root, corrected)
    _not_a_fix(
        root,
        tagged,
        "no layout refusal of the asr file is logged in data/heldout-refusals.log at the tag's "
        f"commit or at a commit of the fix before the correction {corrected}",
    )
    git(root, "checkout", "-q", "--detach", "frozen-plan")
    _refusal(root, "asr")
    git(root, "checkout", "-q", "main")
    _opens(root)


def test_a_refusal_logged_after_its_correction_does_not_count(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    corrected = _correction(root)
    entry = _entry(root, corrected)
    line = {
        "time_utc": "2026-11-17T00:00:00.000000Z",
        "set": ASR_FILE[0],
        "file": ASR_FILE[1],
        "sha256": hashlib.sha256(CONTENT).hexdigest(),
        "commit": entry,
        "tag_commit": tagged,
        "layout": "made up",
    }
    heldout_loader._append(heldout_loader.refusals_path(root), line)
    _not_a_fix(
        root,
        tagged,
        "no layout refusal of the asr file is logged in data/heldout-refusals.log at the tag's "
        f"commit or at a commit of the fix before the correction {corrected}",
    )
    # an opening at the tag and its refusal, as a route at the tag writes them (reading H6(e))
    opening = {name: line[name] for name in OPENING_FIELDS[:-1]} | {"command": ["made up"]}
    heldout_loader._append(heldout_loader.log_path(root), {**opening, "commit": tagged})
    heldout_loader._append(heldout_loader.refusals_path(root), {**line, "commit": tagged})
    _opens(root)


def test_refused_when_a_path_outside_the_loader_its_tests_and_the_plan_differs(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _change(_test_file(root, "asr"))
    _change(root / "src" / "fleetreserve" / "load_norway.py")  # the training loader
    corrected = commit(root, "a correction that also changes another module")
    _entry(root, corrected)
    _not_a_fix(
        root,
        tagged,
        "src/fleetreserve/load_norway.py differs from the tag's commit and is not a held-out "
        "set's loader, a test file of one or docs/plan.md",
    )


def test_refused_when_a_removal_limit_of_the_held_out_loaders_differs(tmp_path):
    """The loaders' removal limits are in `heldout_limits`, which is no set's loader: a fix that
    changes a limit changes a path outside the loader, its tests and the plan."""
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _change(_test_file(root, "asr"))
    _change(root / "src" / "fleetreserve" / "heldout_limits.py")
    corrected = commit(root, "a correction that also changes a removal limit")
    _entry(root, corrected)
    _not_a_fix(
        root,
        tagged,
        "src/fleetreserve/heldout_limits.py differs from the tag's commit and is not a held-out "
        "set's loader, a test file of one or docs/plan.md",
    )


def test_refused_when_a_path_is_changed_and_changed_back_inside_the_chain(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _change(_test_file(root, "asr"))
    (root / "later.txt").write_text("made up\n", encoding="utf-8")
    first = commit(root, "a correction with another file")
    (root / "later.txt").unlink()
    _change(_test_file(root, "asr"))
    second = commit(root, "the other file taken out again")
    _entry(root, first, second)
    _not_a_fix(
        root,
        tagged,
        f"commit {first} changes later.txt, which is not a held-out set's loader, a test file "
        "of one or docs/plan.md",
    )


def test_refused_when_one_commit_corrects_two_sets(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root, "asr")
    _refusal(root, "korea")
    _change(_test_file(root, "asr"))
    _change(root / "src" / "fleetreserve" / "load_korea.py")
    both = commit(root, "one commit correcting two sets")
    _entry(root, both)
    _not_a_fix(
        root, tagged, f"commit {both} changes the files of more than one held-out set (asr, korea)"
    )


def test_refused_when_a_correction_and_its_entry_are_one_commit(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _change(_test_file(root, "asr"))
    _write_plan(root, "- a made-up parsing fix in this commit.\n")
    together = commit(root, "a correction with its entry")
    _not_a_fix(
        root,
        tagged,
        f"commit {together} changes docs/plan.md together with other files; a correction and "
        "its section-14 entry are two commits",
    )


def test_refused_when_a_commit_of_the_chain_changes_nothing(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    empty = commit(root, "a commit that changes nothing")
    corrected = _correction(root)
    _entry(root, corrected)
    _not_a_fix(root, tagged, f"commit {empty} changes nothing")


def test_refused_when_the_plan_changes_outside_section_14(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    corrected = _correction(root)
    _write_plan(root, f"- {corrected}: a made-up parsing fix.\n", work="Changed work packages.\n")
    entry = commit(root, "an entry that also changes section 13")
    _not_a_fix(root, tagged, f"commit {entry} changes docs/plan.md outside section 14")


def test_refused_when_section_14_names_the_correction_by_a_short_id_only(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    corrected = _correction(root)
    _entry(root, corrected[:12])
    _not_a_fix(
        root,
        tagged,
        f"plan section 14 at HEAD does not name the loader correction {corrected} by its full id",
    )


def test_refused_with_a_merge_between_the_tag_and_head(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    git(root, "checkout", "-q", "-b", "side")
    side = _correction(root, name="side")
    git(root, "checkout", "-q", "main")
    main = _correction(root, name="main")
    git(root, "merge", "-q", "--no-ff", "-m", "a made-up merge", "side")
    merge = _head(root)
    _entry(root, side, main, merge)
    _not_a_fix(root, tagged, f"commit {merge} between the tag and HEAD has 2 parents, not one")


def test_refused_when_head_does_not_descend_from_the_tag(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _write_plan(root)
    base = commit(root, "a made-up plan")
    tagged = commit(root, "the tagged commit, the same tree")
    git(root, "tag", "frozen-plan")
    _refusal(root)
    git(root, "checkout", "-q", "--detach", base)
    _entry(root, _correction(root))
    message = (
        f"HEAD {_head(root)} does not descend from the tag frozen-plan ({tagged}); sealed files "
        "open only at the tag's commit or at a logged parsing fix after it (readings H1 and H6)"
    )
    _refused(root, "^" + re.escape(message) + "$")


def test_refused_at_the_correction_without_its_entry(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    corrected = _correction(root)
    _not_a_fix(
        root,
        tagged,
        "HEAD is a loader correction (asr) without its entry in plan section 14 after it",
    )
    _entry(root, corrected)
    _opens(root)


def test_refused_with_an_entry_and_no_correction_before_it(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    _entry(root)
    _not_a_fix(root, tagged, "no loader correction lies between the tag and HEAD")


def test_refused_with_an_unclean_tree_at_the_fix(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    _refusal(root)
    _entry(root, _correction(root))
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    _refused(root, "^the git tree has uncommitted changes; sealed files open only clean$")
    (root / "stray.txt").unlink()
    _opens(root)


def test_the_running_code_at_a_fix_is_compared_with_the_fix_commit(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    _refusal(root)
    _entry(root, _correction(root, loader=True))  # this process runs the tag's loader
    _refused(
        root,
        "^the running code is not the fix commit's: these module files differ from the fix "
        r"commit or are not at it \(reading H5\): src/fleetreserve/load_asr.py$",
    )


# --- the sealed routes' gate (`sealed_route.require_gate`) -----------------------------------

GATE_TEXT = """import sys
from fleetreserve import sealed_route
found = sealed_route.require_gate(sys.argv[1])
print(found, found.fix_commit)
"""


def test_the_routes_gate_returns_the_tags_commit_and_at_a_fix_the_fix_commit(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    found = sealed_route.require_gate(root)
    assert (found, found.fix_commit, sealed_route.fix_record(found)) == (tagged, None, {})
    _refusal(root)
    entry = _entry(root, _correction(root))
    found = sealed_route.require_gate(root)
    assert (found, found.fix_commit) == (tagged, entry)
    assert sealed_route.fix_record(found) == {"fix_commit": entry}


def test_the_routes_gate_refuses_an_unclean_tree_at_a_fix(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    entry = _entry(root, _correction(root))
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    with pytest.raises(
        SealRefused, match="^the git tree has uncommitted changes; sealed files open only clean$"
    ):
        sealed_route.require_gate(root)
    (root / "stray.txt").unlink()
    found = sealed_route.require_gate(root)
    assert (found, found.fix_commit) == (tagged, entry)


def test_the_routes_gate_at_a_fix_compares_the_running_code_with_the_fix_commit(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    _refusal(root)
    entry = _entry(root, _correction(root, loader=True))
    with pytest.raises(SealRefused, match="^the running code is not the fix commit's: "):
        sealed_route.require_gate(root)  # this process runs the tag's loader
    done = run_python(["-c", GATE_TEXT, str(root)], path=root / "src", cwd=tmp_path)
    assert (done.returncode, done.stdout, done.stderr) == (0, f"{tagged} {entry}\n", "")


# --- the refusal log -------------------------------------------------------------------------


@pytest.mark.parametrize("set_name", ["asr", "norway", "korea"])
def test_a_layout_refusal_passes_unchanged_and_is_logged_with_its_opening(tmp_path, set_name):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    names = {"asr": ASR_FILE, "norway": NORWAY_FILE, "korea": KOREA_FILE}[set_name]
    with pytest.raises(LayoutError) as refused:
        _route(root, set_name)
    [opening] = heldout_loader.logged_openings(root)
    [line] = heldout_loader.logged_refusals(root)
    assert list(line) == REFUSAL_FIELDS
    assert line["time_utc"] >= opening["time_utc"]
    assert (line["set"], line["file"]) == names
    assert (line["sha256"], line["commit"], line["tag_commit"]) == (
        hashlib.sha256(CONTENT).hexdigest(),
        tagged,
        tagged,
    )
    assert line["layout"] == str(refused.value)
    assert getattr(refused.value, "__notes__", []) == []


def test_a_refusal_that_cannot_be_logged_leaves_the_layout_error_with_a_note(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    heldout_loader.refusals_path(root).parent.mkdir(exist_ok=True)
    heldout_loader.refusals_path(root).mkdir()
    with pytest.raises(LayoutError) as refused:
        heldout_loader.asr(root)
    [note] = refused.value.__notes__
    assert note.startswith("the layout refusal could not be logged in data/heldout-refusals.log: ")
    assert len(heldout_loader.logged_openings(root)) == 1


def test_the_refusal_log_reader(tmp_path):
    root = sealed_repo(tmp_path, FILES, tag=False)
    _freeze(root)
    assert heldout_loader.refusals_path(root) == root / "data" / "heldout-refusals.log"
    assert heldout_loader.logged_refusals(root) == []
    heldout_loader.refusals_path(root).write_bytes(b"")
    assert heldout_loader.logged_refusals(root) == []
    heldout_loader.refusals_path(root).write_bytes(b'{"time_utc": "made up"}\n')
    with pytest.raises(ValueError, match="^heldout-refusals.log line 1 does not have the log's"):
        heldout_loader.logged_refusals(root)


def test_the_truth_build_stopped_by_a_layout_logs_the_refusal_a_fix_needs(
    tmp_path, monkeypatch, capsys
):
    root = sealed_repo(tmp_path, FILES, tag=False)
    tagged = _freeze(root)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    args = ["--matrix", str(MATRIX), "--out", str(tmp_path / "out")]
    assert heldout_truth_build.main(args, root=root) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("FAILED: the norway file does not have its published layout: ")
    [line] = heldout_loader.logged_refusals(root)
    assert (line["set"], line["commit"], line["tag_commit"]) == ("norway", tagged, tagged)
    entry = _entry(root, _correction(root, "norway"))
    assert heldout_truth_build.main(args, root=root) == 1
    refusals = heldout_loader.logged_refusals(root)
    assert [(r["set"], r["commit"]) for r in refusals] == [("norway", tagged), ("norway", entry)]
    openings = heldout_loader.logged_openings(root)
    assert [o.get("fix_of_tag") for o in openings] == [None, tagged]
    assert git(root, "status", "--porcelain") == ""


def test_the_plan_holds_the_heading_of_section_14_exactly_once():
    lines = (TOP / "docs" / "plan.md").read_bytes().splitlines()
    assert lines.count(heldout_loader.SECTION_14) == 1
