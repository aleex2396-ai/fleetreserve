"""The held-out loader's own opener (`heldout_loader.open_sealed`), called without the sealed
routes' gate (as `heldout_truth_build` calls it), at a logged parsing fix of a made-up repository,
with a symbolic link at a place its check 5 reads through.

Check 5 reads the openings log and the refusal log. Each of the data folder `data`, the sealed
folder in it, the openings log and the refusal log moved away and replaced by a symbolic link to
where it went (or the openings log replaced by a link that leads nowhere), or the top folder
given by a path through `data/..`: `open_sealed` raises `SealRefused` naming the path, and neither
log nor the sealed folder is opened, listed or scanned (`sealed_watch`, by identity on disk, so an
open through the link is seen). The same fix with every place plain: the opener returns the
bytes and reads both logs, so the watch sees the opens there are. At the tag no log is read and no
place is looked at.
"""

from pathlib import Path

import pytest
from fake_heldout import (
    ASR_FILE,
    SEALED,
    approve,
    correction,
    entry,
    freeze,
    log_refusal,
    sealed_repo,
)
from sealed_watch import watch

from fleetreserve import heldout_loader
from fleetreserve.heldout_loader import SealRefused

CONTENT = b"made-up sealed bytes\n"
OPENINGS = "data/heldout-openings.log"
REFUSALS = "data/heldout-refusals.log"
PLACES = {"data": "data", "sealed": SEALED, "openings": OPENINGS, "refusals": REFUSALS}
LINK_TEXT = (
    " is a symbolic link: the gate never follows a link at the data folder, the sealed folder or "
    "a log of it; nothing opened"
)


def at_a_fix(root: Path) -> None:
    """The made-up repository `root` (from `sealed_repo`, so the guard of
    `test_heldout_routes` sees a made-up root) brought to a logged parsing fix of the a.s.r. set,
    as the gate's test does: the tag, the refusal logged at the tag by hand, a correction of its
    loader's test file with its entry in plan section 14, and the owner's approval of the fix
    commit."""
    tagged = freeze(root)
    log_refusal(root, ASR_FILE, CONTENT, tagged)
    fix = entry(root, correction(root, "asr"))
    approve(root, fix)


def moved_and_linked(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    path.symlink_to(target)


def watched(root: Path) -> list[Path]:
    return [(root / place).resolve() for place in (SEALED, OPENINGS, REFUSALS)]


def refused_unopened(root: Path, given: Path, text: str) -> None:
    with watch(*watched(root)) as seen, pytest.raises(SealRefused) as refusal:
        heldout_loader.open_sealed(given, *ASR_FILE)
    assert str(refusal.value) == text
    assert seen == []


def test_every_place_plain_at_a_fix_the_opener_returns_the_bytes_and_reads_both_logs(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    at_a_fix(root)
    with watch(*watched(root)) as seen:
        content = heldout_loader.open_sealed(root, *ASR_FILE)
    assert content == CONTENT
    assert {Path(name).name for _, name in seen} >= {Path(OPENINGS).name, Path(REFUSALS).name}


@pytest.mark.parametrize("which", list(PLACES))
def test_a_linked_place_at_a_fix_is_refused_naming_it_with_neither_log_opened(tmp_path, which):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    at_a_fix(root)
    path = root / PLACES[which]
    moved_and_linked(path, tmp_path / "moved" / which / path.name)
    refused_unopened(root, root, f"{path}{LINK_TEXT}")


def test_a_dangling_link_at_the_openings_log_at_a_fix_is_refused_naming_it(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    at_a_fix(root)
    path = root / OPENINGS
    path.unlink()
    path.symlink_to(tmp_path / "nowhere.log")
    refused_unopened(root, root, f"{path}{LINK_TEXT}")


def test_a_top_folder_given_through_dot_dot_at_a_fix_is_refused_with_neither_log_opened(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    at_a_fix(root)
    through = root / "data" / ".."
    named = through / "data"
    text = (
        f"{named} holds a '..' part: the gate never reads the data folder, the sealed folder or "
        "a log of it by such a path; nothing opened"
    )
    refused_unopened(root, through, text)


def test_at_the_tag_a_linked_sealed_folder_is_not_looked_at_and_the_opener_returns_the_bytes(
    tmp_path,
):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    freeze(root)
    moved_and_linked(root / SEALED, tmp_path / "moved" / "sealed")
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
