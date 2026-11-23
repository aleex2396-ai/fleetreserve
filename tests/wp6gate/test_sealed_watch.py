"""The sealed-folder watcher (`sealed_watch.watch`) sees an open of a made-up sealed file made
from a folder's descriptor (`dir_fd`), as `fleetreserve.plain_read` opens every name, and still
matches only files and folders at or under the watched folders, by identity on disk."""

import os
from pathlib import Path

import pytest
from sealed_watch import watch

from fleetreserve import plain_read


def made_up_sealed(top: Path) -> Path:
    sealed = top / "data" / "heldout" / "made-up" / "x.csv"
    sealed.parent.mkdir(parents=True)
    sealed.write_text("sealed\n")
    return sealed


def read_from_folder(folder: Path, name: str) -> bytes:
    """The first bytes of `name`, opened from the descriptor of `folder` (`dir_fd`)."""
    above = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        inner = os.open(name, os.O_RDONLY, dir_fd=above)
        try:
            return os.read(inner, 100)
        finally:
            os.close(inner)
    finally:
        os.close(above)


def test_an_open_from_a_folder_descriptor_of_a_sealed_file_is_seen(tmp_path):
    sealed = made_up_sealed(tmp_path)
    with watch(sealed.parent) as seen:
        assert read_from_folder(sealed.parent, "x.csv") == b"sealed\n"
    # the folder's own open is by path (the hook); the file's is by descriptor
    assert seen == [("open", str(sealed.parent)), ("open", "x.csv (from a folder descriptor)")]


def test_plain_read_of_a_sealed_file_is_seen(tmp_path):
    """`plain_read` opens each name from the folder above it: before the descriptor's identity
    was matched, this read was recorded as nothing."""
    sealed = made_up_sealed(tmp_path)
    with watch(tmp_path / "data" / "heldout") as seen:
        assert plain_read.read(tmp_path, sealed) == b"sealed\n"
    assert seen == [
        ("open", "heldout (from a folder descriptor)"),
        ("open", "made-up (from a folder descriptor)"),
        ("open", "x.csv (from a folder descriptor)"),
    ]


def test_an_open_from_a_folder_descriptor_of_a_same_named_file_elsewhere_is_not_seen(tmp_path):
    made_up_sealed(tmp_path)
    other = tmp_path / "elsewhere" / "x.csv"
    other.parent.mkdir()
    other.write_text("not sealed\n")
    with watch(tmp_path / "data" / "heldout") as seen:
        assert read_from_folder(other.parent, "x.csv") == b"not sealed\n"
        assert plain_read.read(tmp_path, other) == b"not sealed\n"
    assert seen == []


def test_an_open_by_path_is_seen_once(tmp_path):
    sealed = made_up_sealed(tmp_path)
    with watch(sealed.parent) as seen:
        descriptor = os.open(sealed, os.O_RDONLY)
        os.close(descriptor)
    assert seen == [("open", str(sealed))]


def test_an_open_from_a_folder_descriptor_after_the_watch_ended_is_not_seen(tmp_path):
    sealed = made_up_sealed(tmp_path)
    with watch(sealed.parent) as seen:
        pass
    assert read_from_folder(sealed.parent, "x.csv") == b"sealed\n"
    assert seen == []


def test_the_wrapper_of_os_open_is_taken_out_when_the_watch_ends(tmp_path):
    """Nothing is recorded outside a watch, so only `os.open` itself shows the wrapper is gone."""
    sealed = made_up_sealed(tmp_path)
    before = os.open
    with watch(sealed.parent):
        assert os.open is not before
    assert os.open is before


def test_a_wrapper_left_in_place_under_a_later_one_never_calls_itself(tmp_path):
    """As `opens_of` (tests/wp14) does, `os.open` is patched inside a watch and put back after the
    watch ended, so the wrapper stays in place; then, as `scoring._opens()` (tests/wp2) does, a
    function that calls the `os.open` it took (the wrapper) is put in place before a watch, which
    wraps it in turn. Each open goes wrapper, function, wrapper: the second entry calls the
    `os.open` taken at import, where calling the function again would never end
    (`RecursionError`)."""
    sealed = made_up_sealed(tmp_path)
    original = os.open
    try:
        with pytest.MonkeyPatch.context() as patch, watch(sealed.parent):
            inner = os.open
            patch.setattr(os, "open", lambda *args, **kwargs: inner(*args, **kwargs))
        names: list = []
        earlier = os.open

        def recording(path, *args, **kwargs):
            names.append(os.fsdecode(path))
            return earlier(path, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(os, "open", recording)
            with watch(sealed.parent) as seen:
                assert read_from_folder(sealed.parent, "x.csv") == b"sealed\n"
    finally:
        os.open = original
    assert seen == [("open", str(sealed.parent)), ("open", "x.csv (from a folder descriptor)")]
    assert names == [str(sealed.parent), "x.csv"]


def test_a_name_from_a_folder_descriptor_that_also_names_it_from_the_current_folder_is_seen_once(
    tmp_path, monkeypatch
):
    """The audit hook matches the name from the current folder; the wrapper then records nothing
    more for the same file."""
    sealed = made_up_sealed(tmp_path)
    monkeypatch.chdir(sealed.parent)
    with watch(sealed.parent) as seen:
        assert read_from_folder(sealed.parent, "x.csv") == b"sealed\n"
    assert seen == [("open", str(sealed.parent)), ("open", "x.csv")]
