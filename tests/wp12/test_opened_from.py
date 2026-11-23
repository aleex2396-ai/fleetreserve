"""The opens counter of chosen files (`opened_from.opened`) matches a name opened from a folder's
descriptor (`dir_fd`), as `fleetreserve.plain_read` opens every name, by the identity on disk of
the file it opened, never by its last name: a same-named file in another folder is not counted."""

import os

from opened_from import opened

from fleetreserve import plain_read


def made_up(folder, text: str):
    path = folder / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def read_from_folder(folder, name: str) -> bytes:
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


def test_a_same_named_file_in_another_folder_opened_from_a_folder_descriptor_is_not_counted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    chosen = made_up(tmp_path / "run", "chosen\n")
    other = made_up(tmp_path / "elsewhere", "other\n")
    with opened([chosen], plain_read) as seen:
        assert plain_read.read(tmp_path, other) == b"other\n"
        assert read_from_folder(other.parent, "report.json") == b"other\n"
    assert seen == []


def test_the_chosen_file_opened_from_a_folder_descriptor_is_counted_with_where_it_was_opened(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    chosen = made_up(tmp_path / "run", "chosen\n")
    with opened([chosen], plain_read) as seen:
        assert plain_read.read(tmp_path, chosen) == b"chosen\n"
        assert read_from_folder(chosen.parent, "report.json") == b"chosen\n"
    assert seen == [("report.json", True), ("report.json", False)]


def test_the_chosen_file_opened_by_its_path_is_counted_once(tmp_path):
    chosen = made_up(tmp_path / "run", "chosen\n")
    with opened([chosen], plain_read) as seen:
        assert chosen.read_text() == "chosen\n"
    assert seen == [(str(chosen), False)]


def test_nothing_is_counted_and_os_open_is_put_back_when_the_count_ends(tmp_path):
    chosen = made_up(tmp_path / "run", "chosen\n")
    before = os.open
    with opened([chosen], plain_read) as seen:
        pass
    assert os.open is before
    assert read_from_folder(chosen.parent, "report.json") == b"chosen\n"
    assert seen == []
