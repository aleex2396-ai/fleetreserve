"""The Norway split `tools/norway_split.py`, on fake session files made from the published schema.

The columns are those of the data paper's Table 1 (Dataset 1). Every cell value of the fake rows
is distinctive, so a test can assert that none of them is ever printed.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "norway_split.py"
_spec = importlib.util.spec_from_file_location("norway_split", SCRIPT)
norway_split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(norway_split)

HEADER = "location,session_id,user_id,plugin_time,plugout_time,connection_time,energy_session"
ALL_SITES = [site for sites in norway_split.SITES.values() for site in sites]


def fake_rows(sites=ALL_SITES, per_site=3) -> list[str]:
    rows = []
    for i, site in enumerate(sites):
        for j in range(per_site):
            n = 100 * i + j
            rows.append(
                f"{site},SESS-{n:05d},USER-{n:05d},2019-11-{1 + j:02d} 19:{n % 60:02d}:00,"
                f"2019-11-{2 + j:02d} 07:{n % 60:02d}:00,{12.0 + n / 1000:.3f},{3.5 + n / 1000:.3f}"
            )
    return rows


def cell_values(rows: list[str]) -> set[str]:
    return {value for row in rows for value in row.split(",")}


def write_input(root: Path, text: str | bytes) -> Path:
    path = root / norway_split.INPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode() if isinstance(text, str) else text)
    return path


def run(root: Path) -> tuple[int, list[str]]:
    lines: list[str] = []
    return norway_split.split_file(root, say=lines.append), lines


def output(root: Path, role: str) -> bytes:
    return (root / norway_split.OUTPUTS[role]).read_bytes()


def outputs_exist(root: Path) -> list[bool]:
    return [(root / path).exists() for path in norway_split.OUTPUTS.values()]


@pytest.fixture
def split(tmp_path):
    rows = fake_rows()
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    status, lines = run(tmp_path)
    assert status == 0, lines
    return tmp_path, rows, lines


def test_held_out_rows_go_only_to_the_sealed_folder(split):
    root, rows, _ = split
    held_out = output(root, "held-out").decode()
    assert norway_split.OUTPUTS["held-out"].startswith("data/heldout/")
    for role in ("train", "development"):
        text = output(root, role).decode()
        assert norway_split.OUTPUTS[role].startswith("data/raw/")
        assert "OSL_T," not in text and "TRO_R," not in text
    for row in rows:
        assert (row in held_out) == row.startswith(("OSL_T,", "TRO_R,"))
    raw_files = list((root / "data/raw").rglob("*"))
    assert all("OSL_T" not in p.read_text() for p in raw_files if p.is_file())


def test_each_output_has_the_header_and_its_sites_rows_in_order(split):
    root, rows, _ = split
    for role, sites in norway_split.SITES.items():
        expected = [HEADER] + [row for row in rows if row.split(",")[0] in sites]
        assert output(root, role).decode().splitlines() == expected


def test_counts_add_up_and_are_printed(split):
    _, rows, lines = split
    assert lines[1].startswith("train: data/raw/norway/Dataset1_charging_reports-train.csv ")
    assert ", 15 rows, sha256 " in lines[1]
    assert ", 15 rows, sha256 " in lines[2]
    assert ", 6 rows, sha256 " in lines[3]
    assert lines[-1] == f"rows add up: 15 + 15 + 6 = 36 of {len(rows)}"


def test_no_cell_value_is_printed(split):
    _, rows, lines = split
    printed = "\n".join(lines)
    for value in cell_values(rows):
        assert value not in printed, value


def test_command_line_prints_no_cell_value(tmp_path):
    rows = fake_rows()
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0
    for value in cell_values(rows):
        assert value not in done.stdout + done.stderr, value
    assert len(done.stdout.splitlines()) == 5


def test_same_input_gives_the_same_bytes(tmp_path):
    text = HEADER + "\n" + "\n".join(fake_rows()) + "\n"
    for name in ("a", "b"):
        write_input(tmp_path / name, text)
        assert run(tmp_path / name)[0] == 0
    for role in norway_split.OUTPUTS:
        assert output(tmp_path / "a", role) == output(tmp_path / "b", role)


def test_second_run_leaves_identical_outputs(split):
    root, _, _ = split
    before = [output(root, role) for role in norway_split.OUTPUTS]
    assert run(root)[0] == 0
    assert [output(root, role) for role in norway_split.OUTPUTS] == before


def test_existing_different_output_fails_closed(split):
    root, _, _ = split
    (root / norway_split.OUTPUTS["train"]).write_bytes(b"other")
    (root / norway_split.OUTPUTS["held-out"]).unlink()
    status, lines = run(root)
    assert status == 1
    assert lines == [
        "FAILED: data/raw/norway/Dataset1_charging_reports-train.csv already exists with "
        "different bytes; nothing written"
    ]
    assert not (root / norway_split.OUTPUTS["held-out"]).exists()


def test_unknown_site_fails_closed_naming_only_the_code(tmp_path):
    rows = fake_rows() + fake_rows(["XYZ_9"], per_site=1)
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    status, lines = run(tmp_path)
    assert status == 1
    assert lines == ["FAILED: 1 site codes not among the twelve: ['XYZ_9']; nothing written"]
    assert outputs_exist(tmp_path) == [False, False, False]


def test_unknown_values_that_do_not_look_like_codes_are_only_counted(tmp_path):
    rows = fake_rows() + ["2019-11-28 19:08:00,SESS-X,USER-X,a,b,1.0,2.0"]
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    status, lines = run(tmp_path)
    assert status == 1
    assert lines == [
        "FAILED: 1 site codes not among the twelve: [] and 1 other values not shown; "
        "nothing written"
    ]


def test_missing_site_column_fails_closed_naming_the_columns(tmp_path):
    header = HEADER.replace("location", "garage")
    write_input(tmp_path, header + "\n" + "\n".join(fake_rows()) + "\n")
    status, lines = run(tmp_path)
    assert status == 1
    assert lines[0].startswith("FAILED: no column 'location'; the header's columns are ['garage',")
    assert "OSL_S" not in lines[0]
    assert outputs_exist(tmp_path) == [False, False, False]


def test_record_over_two_lines_fails_closed(tmp_path):
    rows = fake_rows()
    rows[4] = rows[4] + ',"a quoted field\nover two lines"'
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows) + "\n")
    status, lines = run(tmp_path)
    assert status == 1
    assert "spans more than one line" in lines[0]
    assert outputs_exist(tmp_path) == [False, False, False]


def test_blank_line_fails_closed(tmp_path):
    rows = fake_rows()
    write_input(tmp_path, HEADER + "\n" + "\n".join(rows[:3]) + "\n\n" + "\n".join(rows[3:]))
    assert run(tmp_path)[0] == 1
    assert outputs_exist(tmp_path) == [False, False, False]


def test_quoted_header_byte_order_mark_and_windows_line_ends(tmp_path):
    header = ",".join(f'"{name}"' for name in HEADER.split(","))
    rows = fake_rows()
    data = "﻿".encode() + (header + "\r\n" + "\r\n".join(rows)).encode()
    write_input(tmp_path, data)
    status, lines = run(tmp_path)
    assert status == 0, lines
    held_out = output(tmp_path, "held-out")
    assert held_out.startswith("﻿".encode() + header.encode() + b"\r\n")
    assert held_out.endswith(b"\r\n")  # the last row had no line ending; it gets the header's
    assert lines[-1] == "rows add up: 15 + 15 + 6 = 36 of 36"


def test_other_column_order_is_followed(tmp_path):
    names = HEADER.split(",")
    header = ",".join(names[1:] + names[:1])
    rows = [",".join(r.split(",")[1:] + r.split(",")[:1]) for r in fake_rows()]
    write_input(tmp_path, header + "\n" + "\n".join(rows) + "\n")
    assert run(tmp_path)[0] == 0
    assert output(tmp_path, "held-out").decode().count("\n") == 7


def test_missing_input_could_not_run(tmp_path):
    status, lines = run(tmp_path)
    assert status == 2
    assert lines == [f"COULD NOT RUN: {norway_split.INPUT} is missing"]


def test_site_roles_are_the_plans():
    assert norway_split.SITES == {
        "train": ("OSL_S", "ASK", "TRO", "BAR", "BOD"),
        "development": ("OSL_1", "BER", "KRO", "OSL_2", "BAR_2"),
        "held-out": ("OSL_T", "TRO_R"),
    }


def test_semicolon_separated_file_with_quoted_names_is_split(tmp_path):
    names = ["location", "user_id", "session_id", *HEADER.split(",")[3:]]
    header = ";".join(f'"{name}"' for name in names)
    rows = [";".join(f'"{v}"' for v in row.split(",")) for row in fake_rows()]
    write_input(tmp_path, header + "\n" + "\n".join(rows) + "\n")
    status, lines = run(tmp_path)
    assert status == 0, lines
    assert lines[-1] == "rows add up: 15 + 15 + 6 = 36 of 36"
    assert output(tmp_path, "held-out").decode().splitlines()[1].startswith('"OSL_T";')
