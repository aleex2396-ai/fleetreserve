"""The sealed routes' gate (`sealed_route.require_gate`) at a logged parsing fix of a made-up
repository, with a symbolic link at a place the held-out loader's check 5 reads through.

At a fix the loader's check 5 reads the openings log and the refusal log. Each of the data folder
`data`, the sealed folder in it, the openings log and the refusal log moved away and replaced by
a symbolic link to where it went (or the openings log replaced by a link that leads nowhere), or
the top folder given by a path through `data/..`: the gate raises `SealRefused` naming the path,
and neither log nor the sealed folder is opened, listed or scanned (`sealed_watch`, by identity
on disk, so an open through the link is seen). The same fix with every place plain: the gate
passes and reads both logs, so the watch sees the opens there are. At the tag no log is read and
no place is looked at: with the sealed folder a link (inside the ignored `data`, so the tree stays
clean) the gate passes there as before.
"""

import os
from pathlib import Path

import pytest
from fake_heldout import ASR_FILE, SEALED, correction, entry, freeze, log_refusal, sealed_repo
from sealed_watch import watch

from fleetreserve import sealed_route
from fleetreserve.heldout_loader import SealRefused

CONTENT = b"made-up sealed bytes\n"
OPENINGS = "data/heldout-openings.log"
REFUSALS = "data/heldout-refusals.log"
PLACES = {"data": "data", "sealed": SEALED, "openings": OPENINGS, "refusals": REFUSALS}
LINK_TEXT = (
    " is a symbolic link: the gate never follows a link at the data folder, the sealed folder or "
    "a log of it; nothing opened"
)


def at_a_fix(tmp_path) -> tuple[Path, str, str]:
    """A made-up repository at a logged parsing fix of the a.s.r. set: the refusal logged at the
    tag by hand, a correction of its loader's test file, its entry in plan section 14; (top
    folder, tag's commit, fix commit)."""
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    tagged = freeze(root)
    log_refusal(root, ASR_FILE, CONTENT, tagged)
    fix = entry(root, correction(root, "asr"))
    return root, tagged, fix


def moved_and_linked(path: Path, target: Path) -> None:
    """The file or folder at `path` moved to `target` and replaced by a symbolic link to it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    path.symlink_to(target)


def watched(root: Path) -> list[Path]:
    """Where the sealed folder and both logs of `root` are, links followed (no file opened)."""
    return [(root / place).resolve() for place in (SEALED, OPENINGS, REFUSALS)]


def refused_unopened(root: Path, given: Path, text: str) -> None:
    """The gate given the top folder `given` raises `SealRefused` with exactly `text`, and no
    file or folder of `watched(root)` is opened, listed or scanned meanwhile."""
    with watch(*watched(root)) as seen, pytest.raises(SealRefused) as refusal:
        sealed_route.require_gate(given)
    assert str(refusal.value) == text
    assert seen == []


def test_every_place_plain_at_a_fix_the_gate_passes_and_reads_both_logs(tmp_path):
    root, tagged, fix = at_a_fix(tmp_path)
    with watch(*watched(root)) as seen:
        found = sealed_route.require_gate(root)
    assert (found, found.fix_commit) == (tagged, fix)
    assert {Path(name).name for _, name in seen} == {Path(OPENINGS).name, Path(REFUSALS).name}


@pytest.mark.parametrize("which", list(PLACES))
def test_a_linked_place_at_a_fix_is_refused_naming_it_with_neither_log_opened(tmp_path, which):
    root, _, _ = at_a_fix(tmp_path)
    path = root / PLACES[which]
    moved_and_linked(path, tmp_path / "moved" / which / path.name)
    refused_unopened(root, root, f"{path}{LINK_TEXT}")


def test_a_dangling_link_at_the_openings_log_at_a_fix_is_refused_naming_it(tmp_path):
    root, _, _ = at_a_fix(tmp_path)
    path = root / OPENINGS
    path.unlink()
    path.symlink_to(tmp_path / "nowhere.log")
    refused_unopened(root, root, f"{path}{LINK_TEXT}")


def test_a_top_folder_given_through_dot_dot_at_a_fix_is_refused_with_neither_log_opened(tmp_path):
    root, _, _ = at_a_fix(tmp_path)
    through = root / "data" / ".."
    named = through / "data"
    text = (
        f"{named} holds a '..' part: the gate never reads the data folder, the sealed folder or "
        "a log of it by such a path; nothing opened"
    )
    refused_unopened(root, through, text)


def test_at_the_tag_a_linked_sealed_folder_is_not_looked_at_and_the_gate_passes(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    tagged = freeze(root)
    moved_and_linked(root / SEALED, tmp_path / "moved" / "sealed")
    with watch(*watched(root)) as seen:
        found = sealed_route.require_gate(root)
    assert (found, found.fix_commit) == (tagged, None)
    assert seen == []
    assert os.path.islink(root / SEALED)
