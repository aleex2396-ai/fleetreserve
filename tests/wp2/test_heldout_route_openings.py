"""`sealed_route.openings_log`, the openings log for WP13's check of its pass mark, in made-up
repositories with made-up sealed folders in the temporary folder.

1. An absent log gives the loader's log path and no opening, in a repository with no tag: the
   function needs no tag, and it opens, lists or scans nothing in the sealed folder
   (`tests/wp6gate/sealed_watch.py`).
2. A line written by the real opener at a made-up tag is returned as the loader's own reader
   returns it.
3. A line that is not a log line raises `ValueError`, and a link at the log's path `OSError`, as
   the loader's reader raises them.
"""

import pytest
from fake_heldout import ASR_FILE, sealed_repo
from sealed_watch import watch

from fleetreserve import heldout_loader, sealed_route

CONTENT = b"made-up sealed bytes\n"


def test_an_absent_log_gives_no_opening_without_a_tag_and_touches_nothing_sealed(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False, package=False)
    with watch(root / "data" / "heldout") as seen:
        found = sealed_route.openings_log(root)
    assert found == (heldout_loader.log_path(root), [])
    assert seen == []


def test_a_line_written_by_the_opener_is_returned_as_the_loader_reads_it(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    log, openings = sealed_route.openings_log(root)
    assert log == heldout_loader.log_path(root)
    assert openings == heldout_loader.logged_openings(root)
    assert [(line["set"], line["file"]) for line in openings] == [ASR_FILE]


def test_an_ill_formed_line_and_a_link_are_raised_as_the_loader_raises_them(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False, package=False)
    log = heldout_loader.log_path(root)
    log.write_text('{"time_utc": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 does not have the log's fields"):
        sealed_route.openings_log(root)
    log.unlink()
    log.symlink_to(tmp_path / "nowhere.log")
    with pytest.raises(OSError):
        sealed_route.openings_log(root)
