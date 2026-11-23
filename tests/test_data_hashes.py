"""The hash list tool `tools/data_hashes.py`, on a made-up data folder in a temporary place.

The tool is run as a separate process, as the pass mark runs it, for each of its three exit
statuses, and its refusal to change a listed held-out hash is checked.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "data_hashes.py"
_spec = importlib.util.spec_from_file_location("data_hashes", SCRIPT)
data_hashes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_hashes)

FILES = {
    "data/raw/acn/caltech/2019-01.json.gz": b"\x1f\x8bfake",
    "data/raw/acn/caltech-fingerprints.csv": b"month,sessions,sha256\n",
    "data/raw/nrel/dataSet_Oct2021.csv": b"a,b\n1,2\n",
    "data/raw/norway/Dataset1_charging_reports-train.csv": b"x\n1\n",
    "data/raw/norway/Dataset1_charging_reports-development.csv": b"x\n2\n",
    "data/raw/norway/paper.pdf": b"%PDF fake",
    "data/raw/asr/README.txt": b"readme",
    "data/heldout/korea/ChargingRecords.csv": b"SEALED-CELL-4471\n",
    "data/heldout/norway/Dataset1_charging_reports-held-out.csv": b"x\n3\n",
}


def make(root: Path, files: dict[str, bytes] = FILES) -> Path:
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return root


def run(root: Path, mode: str) -> tuple[int, list[str]]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--root", str(root)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done.returncode, done.stdout.splitlines()


def listed(root: Path) -> dict[str, dict]:
    text = (root / "data-hashes.json").read_text()
    return {entry["path"]: entry for entry in json.loads(text)["files"]}


@pytest.fixture
def written(tmp_path):
    root = make(tmp_path)
    status, lines = run(root, "--write")
    assert status == 0, lines
    return root


def test_write_lists_every_file_with_size_hash_and_role(written):
    entries = listed(written)
    assert sorted(entries) == sorted(FILES)
    korea = entries["data/heldout/korea/ChargingRecords.csv"]
    assert korea == {
        "path": "data/heldout/korea/ChargingRecords.csv",
        "bytes": len(FILES["data/heldout/korea/ChargingRecords.csv"]),
        "sha256": hashlib.sha256(FILES["data/heldout/korea/ChargingRecords.csv"]).hexdigest(),
        "role": "held-out",
    }
    roles = {path: entry["role"] for path, entry in entries.items()}
    assert roles["data/raw/acn/caltech/2019-01.json.gz"] == "train"
    assert roles["data/raw/nrel/dataSet_Oct2021.csv"] == "development"
    assert roles["data/raw/norway/Dataset1_charging_reports-train.csv"] == "train"
    assert roles["data/raw/norway/Dataset1_charging_reports-development.csv"] == "development"
    assert roles["data/raw/norway/paper.pdf"] == "document"
    assert roles["data/heldout/norway/Dataset1_charging_reports-held-out.csv"] == "held-out"


def test_list_holds_only_path_size_hash_and_role(written):
    for entry in listed(written).values():
        assert set(entry) == {"path", "bytes", "sha256", "role"}
    assert "SEALED-CELL" not in (written / "data-hashes.json").read_text()


def test_nothing_of_the_content_is_printed(tmp_path):
    root = make(tmp_path)
    _, write_lines = run(root, "--write")
    _, check_lines = run(root, "--check")
    assert "SEALED-CELL" not in "\n".join(write_lines + check_lines)


def test_check_passes_on_matching_data(written):
    status, lines = run(written, "--check")
    assert status == 0
    assert lines == [
        "9 listed files match (2 held-out); no unlisted file under data/raw or data/heldout"
    ]


def test_check_fails_when_a_hash_differs(written):
    (written / "data/heldout/korea/ChargingRecords.csv").write_bytes(b"SEALED-CELL-4472\n")
    status, lines = run(written, "--check")
    assert status == 1
    assert lines == [
        "FAILED: data/heldout/korea/ChargingRecords.csv: SHA-256 differs from the list"
    ]


def test_check_fails_when_a_size_differs(written):
    (written / "data/raw/nrel/dataSet_Oct2021.csv").write_bytes(b"a,b\n1,2\n3,4\n")
    status, lines = run(written, "--check")
    assert status == 1
    assert lines[0].startswith("FAILED: data/raw/nrel/dataSet_Oct2021.csv: 12 bytes")


def test_check_fails_on_an_unlisted_file(written):
    make(written, {"data/heldout/asr/extra.csv": b"x"})
    status, lines = run(written, "--check")
    assert status == 1
    assert lines == ["FAILED: data/heldout/asr/extra.csv: not in the hash list"]


def test_check_fails_on_a_left_over_partial_download(written):
    make(written, {"data/raw/nrel/.dataSet_Oct2021.csv.part": b"x"})
    assert run(written, "--check")[0] == 1


def test_check_skips_folder_metadata_files(written):
    make(written, {"data/heldout/.DS_Store": b"x", "data/raw/acn/.DS_Store": b"y"})
    assert run(written, "--check")[0] == 0


def test_check_fails_when_a_role_is_edited(written):
    text = (written / "data-hashes.json").read_text()
    path = "data/heldout/korea/ChargingRecords.csv"
    entry = json.dumps(listed(written)[path])
    edited = entry.replace('"role": "held-out"', '"role": "train"')
    (written / "data-hashes.json").write_text(text.replace(entry, edited))
    status, lines = run(written, "--check")
    assert status == 1
    assert lines == [f"FAILED: {path}: listed as train, its folder makes it held-out"]


def test_check_cannot_run_when_a_listed_file_is_missing(written):
    (written / "data/heldout/korea/ChargingRecords.csv").unlink()
    status, lines = run(written, "--check")
    assert status == 2
    assert lines == ["COULD NOT RUN: data/heldout/korea/ChargingRecords.csv: listed but missing"]


def test_difference_outranks_a_missing_file(written):
    (written / "data/heldout/korea/ChargingRecords.csv").unlink()
    (written / "data/raw/asr/README.txt").write_bytes(b"README")
    assert run(written, "--check")[0] == 1


def test_check_cannot_run_without_a_list(tmp_path):
    status, lines = run(make(tmp_path), "--check")
    assert status == 2
    assert lines == ["COULD NOT RUN: no hash list data-hashes.json"]


def test_check_cannot_run_on_an_empty_list(tmp_path):
    root = make(tmp_path)
    (root / "data-hashes.json").write_text('{"files": []}\n')
    status, lines = run(root, "--check")
    assert status == 2
    assert lines == ["COULD NOT RUN: data-hashes.json lists no files"]


def test_check_cannot_run_on_an_ill_formed_list(tmp_path):
    root = make(tmp_path)
    (root / "data-hashes.json").write_text("not json")
    assert run(root, "--check")[0] == 2


def test_check_cannot_run_without_a_data_folder(written, tmp_path_factory):
    bare = tmp_path_factory.mktemp("bare")
    (bare / "data-hashes.json").write_bytes((written / "data-hashes.json").read_bytes())
    status, lines = run(bare, "--check")
    assert status == 2
    assert len(lines) == len(FILES)


def test_write_refuses_to_change_a_held_out_hash(written):
    before = (written / "data-hashes.json").read_bytes()
    (written / "data/heldout/korea/ChargingRecords.csv").write_bytes(b"SEALED-CELL-9999\n")
    status, lines = run(written, "--write")
    assert status == 1
    assert lines == [
        "FAILED: data/heldout/korea/ChargingRecords.csv: "
        "a listed held-out hash may not change; not written"
    ]
    assert (written / "data-hashes.json").read_bytes() == before


def test_write_refuses_to_drop_a_listed_file(written):
    before = (written / "data-hashes.json").read_bytes()
    (written / "data/heldout/korea/ChargingRecords.csv").unlink()
    assert run(written, "--write")[0] == 2
    assert (written / "data-hashes.json").read_bytes() == before


def test_write_refuses_a_file_without_a_role(written):
    before = (written / "data-hashes.json").read_bytes()
    make(written, {"data/raw/elsewhere/x.csv": b"x"})
    status, lines = run(written, "--write")
    assert status == 1
    assert lines == [
        "FAILED: data/raw/elsewhere/x.csv: no role for this path; the list is not written"
    ]
    assert (written / "data-hashes.json").read_bytes() == before


def test_write_adds_new_files_and_keeps_held_out_entries(written):
    make(written, {"data/heldout/asr/new.csv": b"n"})
    status, lines = run(written, "--write")
    assert status == 0
    assert lines[-1] == "wrote data-hashes.json: 10 files, 1 added, 0 changed"
    assert run(written, "--check")[0] == 0


def test_write_updates_a_changed_file_that_is_not_held_out(written):
    (written / "data/raw/nrel/dataSet_Oct2021.csv").write_bytes(b"new")
    status, lines = run(written, "--write")
    assert status == 0
    assert lines[-1] == "wrote data-hashes.json: 9 files, 0 added, 1 changed"


def test_write_is_deterministic(tmp_path):
    a, b = make(tmp_path / "a"), make(tmp_path / "b")
    run(a, "--write")
    run(b, "--write")
    assert (a / "data-hashes.json").read_bytes() == (b / "data-hashes.json").read_bytes()


def test_wrong_call_cannot_run(tmp_path):
    assert run(tmp_path, "--neither")[0] == 2


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("data/raw/acn/office001/2019-01.json.gz", "train"),
        ("data/raw/acn/office001-fingerprints.csv", "train"),
        ("data/heldout/norway-original/Dataset4_hourly_predictions.csv", "held-out"),
        ("data/raw/norway/Dataset1_charging_reports-held-out.csv", None),
        ("data/raw/korea/ChargingRecords.csv", None),
        ("data/raw/korea/Code.zip", "document"),
        ("data/raw/korea/Code.csv", None),
        ("data/heldoutx/a.csv", None),
    ],
)
def test_roles_follow_the_path(path, role):
    assert data_hashes.role_of(path) == role
