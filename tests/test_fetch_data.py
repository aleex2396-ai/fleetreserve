"""The download tool `tools/fetch_data.py`, on made-up bytes; no network is used.

The real source table is also checked against the figures the plan fixes (plan section 3).
"""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "fetch_data.py"
_spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
fetch_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_data)

MARKER = b"CELL-VALUE-7731,OSL_T,12.5\n"
CONTENT = b"col_a,col_b\n" + MARKER * 50


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


SIZE = len(CONTENT)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to reach the network fails the test instead."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", refuse)


def source(role="held-out", size=SIZE, checksum=None, name="file.csv", set_name="demo"):
    return fetch_data.Source(
        set_name, name, role, size, md5(CONTENT) if checksum is None else checksum, "https://x/y"
    )


class Served:
    """A fake opener: serves fixed bytes for every URL and counts the requests."""

    def __init__(self, data: bytes = CONTENT, error: str | None = None):
        self.data, self.error, self.calls = data, error, 0

    def __call__(self, url):
        self.calls += 1
        if self.error:
            raise fetch_data.FetchError(self.error)
        return io.BytesIO(self.data)


def run(root, sources, opener):
    lines: list[str] = []
    status = fetch_data.fetch_all(root, sources, opener=opener, say=lines.append, pause=0.0)
    return status, lines


def files_under(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_matching_held_out_file_goes_to_the_sealed_folder(tmp_path):
    status, lines = run(tmp_path, [source()], Served())
    assert status == 0
    assert files_under(tmp_path) == ["data/heldout/demo/file.csv"]
    assert (tmp_path / "data/heldout/demo/file.csv").read_bytes() == CONTENT
    assert lines[-1] == "1 in place, 0 failed, 0 could not be fetched"


def test_other_roles_go_to_the_raw_folder(tmp_path):
    sources = [source("development", name="a.csv"), source("document", name="b.txt")]
    status, _ = run(tmp_path, sources, Served())
    assert status == 0
    assert files_under(tmp_path) == ["data/raw/demo/a.csv", "data/raw/demo/b.txt"]


def test_no_byte_of_content_is_printed(tmp_path):
    _, lines = run(tmp_path, [source(), source("development", name="b.csv")], Served())
    printed = "\n".join(lines)
    assert "CELL-VALUE-7731" not in printed
    assert "OSL_T" not in printed
    assert "col_a" not in printed


def test_wrong_size_is_failed_and_nothing_is_kept(tmp_path):
    status, lines = run(tmp_path, [source(size=len(CONTENT) + 1)], Served())
    assert status == 1
    assert files_under(tmp_path) == []
    assert lines[0].startswith("FAILED: data/heldout/demo/file.csv:")


def test_wrong_checksum_is_failed_and_nothing_is_kept(tmp_path):
    status, lines = run(tmp_path, [source(checksum="0" * 32)], Served())
    assert status == 1
    assert files_under(tmp_path) == []
    assert "MD5 differs" in lines[0]


def test_existing_different_file_is_never_overwritten(tmp_path):
    target = tmp_path / "data/heldout/demo/file.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"other bytes")
    served = Served()
    status, lines = run(tmp_path, [source()], served)
    assert status == 1
    assert target.read_bytes() == b"other bytes"
    assert served.calls == 0
    assert "already there and differs" in lines[0]


def test_existing_matching_file_is_left_and_not_fetched_again(tmp_path):
    run(tmp_path, [source()], Served())
    served = Served()
    status, lines = run(tmp_path, [source()], served)
    assert status == 0
    assert served.calls == 0
    assert lines[0].startswith("present: data/heldout/demo/file.csv")


def test_network_error_could_not_run_and_leaves_nothing(tmp_path):
    served = Served(error="HTTP 503")
    status, lines = run(tmp_path, [source()], served)
    assert status == 2
    assert served.calls == fetch_data.ATTEMPTS
    assert files_under(tmp_path) == []
    assert lines[0] == "COULD NOT RUN: data/heldout/demo/file.csv: HTTP 503"


def test_failed_outranks_could_not_run(tmp_path):
    sources = [source(name="a.csv"), source(name="b.csv", checksum="0" * 32)]

    class Mixed(Served):
        def __call__(self, url):
            self.calls += 1
            if self.calls == 1:
                raise fetch_data.FetchError("HTTP 500")
            return io.BytesIO(self.data)

    status, _ = run(tmp_path, sources, Mixed())
    assert status == 1


def test_file_without_published_size_or_checksum_is_kept(tmp_path):
    sources = [fetch_data.Source("demo", "n.csv", "development", None, None, "https://x/y")]
    status, lines = run(tmp_path, sources, Served())
    assert status == 0
    assert "no published size, no published checksum" in lines[0]


# --- the real table ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table():
    return {(s.set_name, s.name): s for s in fetch_data.read_table(fetch_data.TABLE)}


def test_table_held_out_files_have_the_plan_sizes(table):
    assert table[("asr", "202410DatasetEVOfficeParking_v0.csv")].size == 4_270_702
    assert table[("korea", "ChargingRecords.csv")].size == 7_605_859


def test_table_roles_follow_the_plan(table):
    held_out = {key for key, s in table.items() if s.role == "held-out"}
    assert held_out == {
        ("asr", "202410DatasetEVOfficeParking_v0.csv"),
        ("korea", "ChargingRecords.csv"),
        ("norway-original", "Dataset1_charging_reports.csv"),
        ("norway-original", "Dataset2_user_predictions.csv"),
        ("norway-original", "Dataset3_session_predictions.csv"),
        ("norway-original", "Dataset4_hourly_predictions.csv"),
    }
    assert table[("nrel", "dataSet_Oct2021.csv")].role == "development"


def test_table_lists_only_approved_files(table):
    names = {name for _, name in table}
    assert "Survey_supplementary.zip" not in names
    assert len(table) == 10


def test_table_korean_code_is_a_document_of_the_korean_record(table):
    """The Korean authors' plotting scripts, approved by the owner on 15 November 2026 to learn
    the file's labels before the plan is frozen: code, no data, so a document, never held-out."""
    code = table[("korea", "Code.zip")]
    assert (code.role, code.size, code.md5) == (
        "document",
        5914,
        "3421656f0b8a583f196d75baa1f65552",
    )
    assert code.url == "https://ndownloader.figshare.com/files/39952261"
    assert fetch_data.destination(Path("/r"), code) == Path("/r/data/raw/korea/Code.zip")


def test_table_norway_files_add_up_to_the_plan_size(table):
    total = sum(s.size for (set_name, _), s in table.items() if "norway" in set_name)
    assert round(total / 1e6, 1) == 48.4


def test_table_every_held_out_file_has_a_published_checksum(table):
    for s in table.values():
        if s.role == "held-out":
            assert s.size is not None and s.md5 is not None and len(s.md5) == 32


def test_table_empty_size_and_checksum_mean_none_published(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("set,name,role,bytes,md5,url\na,b.csv,development,,,https://x\n")
    (only,) = fetch_data.read_table(path)
    assert only.size is None and only.md5 is None


def test_table_with_unknown_role_cannot_be_read(tmp_path):
    bad = tmp_path / "t.csv"
    bad.write_text("set,name,role,bytes,md5,url\na,b,secret,,,https://x\n")
    with pytest.raises(fetch_data.TableError):
        fetch_data.read_table(bad)
